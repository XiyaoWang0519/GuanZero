"""Learner forward op diet (Oct 18 2026): fewer device operations, the same bits.

The PPO learner's forward and loss were rewritten to dispatch fewer kernels and
to never build a device tensor from Python data (each is a host-to-device copy
that waits for the GPU queue): the auxiliary coefficient is a fill, the
sinusoidal position table is cached, candidate rows and the next-token head's
position mask are planned on the host, next-token targets and 0/1 diagnostics
are fused with integer-exact reductions, and the auxiliary mode's zero response
bridge keeps only its (zero) gradient. All of that is bitwise: a full two-epoch
``learn()`` reproduces the reference implementation (frozen below as it was at
commit 090f337) in every parameter, Adam moment and logged statistic.

``learner_compact_next`` is the one tier-2 switch here (fused next-token loss on
valid positions only): checked against the default path with tolerances.
"""
from dataclasses import replace
import json
import math
import pickle

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from train import history_model
from train.history_aux import (BOMB, BOMB_CLASSES, MULTI_HOT_FIELDS, NEXT_FIELDS, NEXT_LOGITS,
                               NEXT_SLICES, ONE_HOT_FIELDS, belief_loss, next_token_loss,
                               next_token_loss_compact, next_token_targets, outcome_loss,
                               round_outcomes)
from train.history_model import (HistoryActor, HistoryPolicyConfig, RESPONSE_CLASSES,
                                  position_table, sinusoidal)
from train.history_ppo import (HistoryPPOConfig, HistoryTrainer, RESUME_OVERRIDES, build_parser,
                               parse_resume_overrides, segment_entropy)
from train.history_response import opponent_response_labels
from train.logs import TOKEN_DIM

LEARNER = dict(response_mode="auxiliary", aux_heads="next,belief,outcome", aux_coef=0.1,
               entropy=0.03, epochs=2, lr=3e-4, causal_sdpa=True, learner_length_groups=2,
               learner_batched_attention=True, minibatch_matches=3, seed=7, num_envs=2,
               steps_per_update=16, snapshot_updates=0)
TINY = dict(width=16, layers=1, heads=2)
WIDE = dict(width=128, layers=4, heads=8)


# ---- the reference implementation (commit 090f337), frozen ---------------------------

def legacy_next_token_loss(logits, tokens, valid):
    logits = logits.reshape(-1, NEXT_LOGITS).float()
    tokens = tokens.reshape(-1, TOKEN_DIM)
    weight = valid.reshape(-1).to(logits.dtype)
    total = weight.sum().clamp_min(1.0)

    def masked_mean(values):
        return (values * weight).sum() / total

    stats = {}
    loss = logits.new_zeros(())
    for name, field in ONE_HOT_FIELDS:
        target = tokens[:, field].argmax(-1)
        part = logits[:, NEXT_SLICES[name]]
        stats[f"next_{name}_loss"] = masked_mean(F.cross_entropy(part, target, reduction="none"))
        loss = loss + stats[f"next_{name}_loss"]
        if name in ("type", "seat"):
            stats[f"next_{name}_accuracy"] = masked_mean((part.argmax(-1) == target).to(logits.dtype))
    cards_right = torch.ones_like(weight, dtype=torch.bool)
    for name, field in MULTI_HOT_FIELDS:
        target = tokens[:, field].to(logits.dtype)
        part = logits[:, NEXT_SLICES[name]]
        stats[f"next_{name}_loss"] = masked_mean(
            F.binary_cross_entropy_with_logits(part, target, reduction="none").mean(-1))
        loss = loss + stats[f"next_{name}_loss"]
        cards_right &= ((part > 0) == (target > 0.5)).all(-1)
    stats["next_cards_exact"] = masked_mean(cards_right.to(logits.dtype))
    bomb = tokens[:, BOMB]
    bomb_target = torch.where(bomb.sum(-1) > 0, bomb.argmax(-1),
                              torch.full_like(bomb[:, 0], BOMB_CLASSES - 1, dtype=torch.long))
    stats["next_bomb_loss"] = masked_mean(
        F.cross_entropy(logits[:, NEXT_SLICES["bomb"]], bomb_target, reduction="none"))
    loss = loss + stats["next_bomb_loss"]
    stats["next_loss"] = loss / NEXT_FIELDS
    return stats


def legacy_candidate_outputs(self, state, cand, offsets, *, predict=False, rows=None,
                             response_rows=None):
    if rows is None:
        counts = offsets[1:] - offsets[:-1]
        rows = torch.repeat_interleave(torch.arange(len(state), device=state.device), counts,
                                       output_size=len(cand))
    fused = torch.cat((state[rows], self.action_tower(cand.float())), dim=-1)
    response = None
    mode = self.config.response_mode
    if response_rows is not None and mode != "auxiliary":
        raise ValueError("response_rows needs the auxiliary response mode")
    if mode != "none" and (predict or mode == "explicit"):
        response = self.response_head(fused if response_rows is None else fused[response_rows])
    if mode == "explicit":
        feature = response.softmax(-1).detach() - 1.0 / RESPONSE_CLASSES
        fused = fused + self.response_bridge(feature)
    elif mode == "auxiliary" and torch.is_grad_enabled():
        fused = fused + self.response_bridge(fused.new_zeros(len(cand), RESPONSE_CLASSES))
    return self.fusion(fused).squeeze(-1), response


def legacy_auxiliary_losses(self, state, inputs, row, encoded_streams):
    heads = self.actor.aux_head_names
    stats = {}
    total = state.new_zeros(())
    if "next" in heads:
        s = inputs.streams
        parts = []
        for encoded, group in encoded_streams:
            if group is None:
                tokens, lengths = s.tokens, s.lengths
                cited = torch.zeros_like(lengths).scatter_reduce(
                    0, inputs.match_index, inputs.prefix, reduce="amax")
            else:
                n = group.tokens
                tokens = s.tokens[group.matches, :n]
                lengths = s.lengths[group.matches].clamp(max=n)
                cited = torch.zeros_like(lengths).scatter_reduce(
                    0, group.local_match, inputs.prefix[group.rows], reduce="amax")
            count = tokens.shape[1]
            limit = torch.minimum(lengths, cited)
            valid = torch.arange(count, device=tokens.device)[None] < limit[:, None]
            logits = self.actor.next_logits(encoded[:, :count])
            parts.append((logits.reshape(-1, logits.shape[-1]),
                          tokens.reshape(-1, tokens.shape[-1]), valid.reshape(-1)))
        stats.update(legacy_next_token_loss(torch.cat([p[0] for p in parts]),
                                            torch.cat([p[1] for p in parts]),
                                            torch.cat([p[2] for p in parts])))
        total = total + stats["next_loss"]
    if "belief" in heads:
        stats.update(belief_loss(self.actor.belief_logits(state), row["hidden"]))
        total = total + stats["belief_loss"]
    if "outcome" in heads:
        stats.update(outcome_loss(self.actor.outcome_value(state), row["outcome_target"]))
        total = total + stats["outcome_loss"]
    stats["aux_loss"] = total
    stats["aux_coefficient"] = state.new_tensor(self.aux_coefficient())
    return stats


def legacy_minibatch_loss(self, rows, *unused, **unused_keywords):
    cfg = self.config
    predict = cfg.response_mode != "none"
    extra = ({"response_target": opponent_response_labels(
        self.buffer, self.store, rows, round_streams=cfg.habit_round)} if predict else {})
    if "outcome" in self.actor.aux_head_names:
        extra["outcome_target"] = round_outcomes(self.buffer, rows)
    extra.update(self.habit_extra(rows))
    batch = self.buffer.training_batch(rows, self.store, self.device, extra=extra or None,
                                       length_groups=cfg.learner_length_groups,
                                       width=cfg.width, round_streams=cfg.habit_round)
    inputs, chosen, row = batch.inputs, batch.chosen, batch.fields
    # The reference had no host-planned candidate rows or next-token mask.
    inputs.candidate_rows, inputs.next_weight, inputs.next_positions = None, None, 0
    inputs.style = row.get("style")
    response_stats = {}
    aux_stats = {}
    heads = self.actor.aux_head_names
    if not predict and not heads:
        all_log_probs = self.actor.candidate_log_probs(inputs)
    else:
        from train.history_model import segment_log_softmax
        encoded_streams = []
        state = self.actor.decision_states(None, inputs, encoded_streams if heads else None)
        chosen_only = cfg.learner_chosen_response
        logits, response = self.actor.candidate_outputs(
            state, inputs.cand, inputs.offsets, predict=predict,
            response_rows=chosen if (predict and chosen_only) else None, rows=inputs.rows)
        all_log_probs = segment_log_softmax(logits, inputs.rows, inputs.decisions)
        if predict:
            targets = row["response_target"]
            prediction = response if chosen_only else response[chosen]
            response_stats = {"response_loss": F.cross_entropy(prediction, targets),
                              "response_accuracy": (prediction.argmax(-1) == targets).float().mean(),
                              "response_event_fraction": (targets != 0).float().mean()}
        if heads:
            aux_stats = legacy_auxiliary_losses(self, state, inputs, row, encoded_streams)
    log_prob = all_log_probs[chosen]
    entropy = segment_entropy(all_log_probs, inputs.rows, inputs.decisions)
    old, behaviour = row["logp"], row["behaviour_logp"]
    advantage, returns = row["advantage"], row["returns"]
    log_ratio = log_prob - old
    ratio = log_ratio.exp()
    if self.advantage_moments is not None:
        advantage = (advantage - self.advantage_moments[0]) / (self.advantage_moments[1] + 1e-8)
    elif cfg.normalize_advantages and len(advantage) > 1:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    raw_weight = (old - behaviour).exp()
    weight = raw_weight.clamp(max=cfg.behaviour_weight_cap)
    advantage = advantage * weight
    clipped = torch.minimum(ratio * advantage,
                            ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advantage)
    surrogate = -clipped.mean()
    policy_total = surrogate - cfg.entropy * entropy.mean()
    if response_stats:
        policy_total = policy_total + cfg.response_coef * response_stats["response_loss"]
    if aux_stats:
        policy_total = policy_total + aux_stats["aux_coefficient"] * aux_stats["aux_loss"]
    value_loss = F.mse_loss(self.critic(inputs.obs, row["hidden"]), returns)
    with torch.no_grad():
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clip_fraction = ((ratio - 1).abs() > cfg.clip).float().mean()
        ratio_deviation = (ratio - 1).abs().max()
        weight_capped = (raw_weight > cfg.behaviour_weight_cap).float().mean()
    return {"policy_total": policy_total, "value_total": cfg.value_coef * value_loss,
            "policy_loss": surrogate, "value_loss": value_loss, "entropy": entropy.mean(),
            "approx_kl": approx_kl, "clip_fraction": clip_fraction,
            "ratio_deviation": ratio_deviation, "behaviour_weight_mean": weight.mean(),
            "behaviour_weight_cap_fraction": weight_capped, **response_stats, **aux_stats}


def use_legacy(monkeypatch) -> None:
    """Swap the reference implementation in for the learner forward."""
    monkeypatch.setattr(HistoryTrainer, "minibatch_loss", legacy_minibatch_loss)
    monkeypatch.setattr(HistoryActor, "candidate_outputs", legacy_candidate_outputs)
    monkeypatch.setattr(history_model, "position_table",
                        lambda length, width, device, dtype: sinusoidal(length, width, device, dtype))


# ---- fixtures ------------------------------------------------------------------------

@pytest.fixture
def exact_cpu():
    """Run-to-run bitwise CPU kernels: deterministic algorithms, one thread."""
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    threads = torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)
        torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def collected(tmp_path_factory) -> bytes:
    """A fixed rollout buffer and its public streams (a few self-play updates)."""
    threads = torch.get_num_threads()
    try:
        trainer = HistoryTrainer(HistoryPPOConfig(width=16, layers=1, heads=2, num_envs=8,
                                                  steps_per_update=64, seed=11, snapshot_updates=0,
                                                  minibatch_matches=4, torch_threads=1),
                                 tmp_path_factory.mktemp("collect"))
        for _ in range(2):
            trainer.update()
        trainer.collect()
        assert len(trainer.buffer.completed_rows()) > 100
        return pickle.dumps((trainer.buffer, trainer.store))
    finally:
        torch.set_num_threads(threads)


def learner(tmp_path, collected: bytes, **overrides) -> HistoryTrainer:
    options = {**LEARNER, **TINY, **overrides}
    trainer = HistoryTrainer(HistoryPPOConfig(**options), tmp_path)
    trainer.buffer, trainer.store = pickle.loads(collected)
    trainer.rng = np.random.default_rng(123)
    return trainer


def finalized(tmp_path, collected: bytes, **overrides) -> HistoryTrainer:
    trainer = learner(tmp_path, collected, **overrides)
    trainer.buffer.finalize(trainer.refresh_values())
    return trainer


def learn_record(trainer: HistoryTrainer) -> dict:
    stats = trainer.learn()
    stats.pop("learn_phase_seconds")
    return dict(stats=stats,
                params=[p.detach().clone() for p in (*trainer.actor.parameters(),
                                                     *trainer.critic.parameters())],
                adam=[opt.state_dict()["state"] for opt in (trainer.actor_optimizer,
                                                            trainer.critic_optimizer)])


# ---- full learn() against the reference ----------------------------------------------

@pytest.mark.parametrize("overrides", [
    dict(),
    dict(learner_batched_attention=False),
    dict(learner_length_groups=0),
    dict(learner_length_groups=0, learner_batched_attention=False, causal_sdpa=False),
    dict(learner_chosen_response=True, aux_warmup_updates=3),
    dict(response_mode="none", aux_heads=""),
    dict(response_mode="explicit", aux_heads="next", learner_length_groups=0),
    dict(**WIDE),
    dict(**WIDE, learner_length_groups=0, learner_batched_attention=False),
], ids=["production", "unbatched", "ungrouped", "ungrouped-unbatched-masked", "chosen-warmup",
        "plain", "explicit", "wide", "wide-ungrouped-unbatched"])
def test_learn_is_bitwise_the_reference(tmp_path, monkeypatch, collected, exact_cpu, overrides):
    with monkeypatch.context() as patch:
        use_legacy(patch)
        reference = learn_record(learner(tmp_path / "reference", collected, **overrides))
    current = learn_record(learner(tmp_path / "current", collected, **overrides))
    assert reference["stats"]["minibatches"] >= 4
    assert reference["stats"].keys() == current["stats"].keys()
    assert len(reference["params"]) == len(current["params"])
    for key, value in reference["stats"].items():
        # Python floats (FP32 numbers summed in double), ints or None: exact.
        assert current["stats"][key] == value or (
            isinstance(value, float) and math.isnan(value) and math.isnan(current["stats"][key])), key
    for a, b in zip(reference["params"], current["params"]):
        assert torch.equal(a, b)
    for old, new in zip(reference["adam"], current["adam"]):
        assert old.keys() == new.keys()
        for index in old:
            for name in old[index]:
                assert torch.equal(torch.as_tensor(old[index][name]),
                                   torch.as_tensor(new[index][name])), (index, name)


# ---- no host synchronization, fewer operations ---------------------------------------

class Dispatches(TorchDispatchMode):
    """aten operations dispatched after the batch upload."""

    def __init__(self):
        super().__init__()
        self.armed = False
        self.names: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if self.armed:
            self.names.append(func.overloadpacket.__name__)
        return func(*args, **(kwargs or {}))


def dispatched(trainer: HistoryTrainer, rows) -> list[str]:
    trainer.actor.zero_grad(set_to_none=True)
    trainer.critic.zero_grad(set_to_none=True)
    mode = Dispatches()
    upload = trainer.buffer.training_batch

    def armed(*args, **kwargs):
        batch = upload(*args, **kwargs)
        mode.armed = True
        return batch

    trainer.buffer.training_batch = armed
    try:
        with mode:
            terms = trainer.minibatch_loss(rows)
            terms["policy_total"].backward()
            terms["value_total"].backward()
    finally:
        del trainer.buffer.training_batch
    return mode.names


@pytest.mark.parametrize("compact", [False, True])
def test_forward_and_backward_never_wait_for_the_device(tmp_path, collected, compact):
    trainer = finalized(tmp_path, collected, learner_compact_next=compact)
    rows = next(iter(trainer.minibatches()))
    trainer.minibatch_loss(rows)            # caches built (position table, field layouts)
    names = dispatched(trainer, rows)
    assert len(names) > 100
    # Tensors from Python data (host-to-device copies) and host reads of device values.
    for name in ("lift_fresh", "lift_fresh_copy", "_local_scalar_dense", "item", "nonzero"):
        assert name not in names, name
    # Device work the host planned or the cache holds.
    for name in ("repeat_interleave", "sin", "cos"):
        assert name not in names, name


def test_fewer_operations_than_the_reference(tmp_path, monkeypatch, collected):
    trainer = finalized(tmp_path, collected)
    rows = next(iter(trainer.minibatches()))
    trainer.minibatch_loss(rows)
    current = dispatched(trainer, rows)
    with monkeypatch.context() as patch:
        use_legacy(patch)
        reference = dispatched(trainer, rows)
    assert reference.count("repeat_interleave") == 3 and reference.count("sin") >= 1
    # Per length group ~11 position-table and ~6 next-token mask kernels; 3
    # candidate-row rebuilds, ~17 next-token target/statistic kernels, 3
    # zero-bridge kernels and ~8 diagnostic kernels (about 80 in production).
    assert len(current) <= len(reference) - 60, (len(current), len(reference))


# ---- the parts ---------------------------------------------------------------------------

@pytest.mark.parametrize("width", [16, 128, 30])
def test_position_table_slices_are_the_fresh_table(width):
    device = torch.device("cpu")
    history_model._POSITION_TABLES.pop((width, device, torch.float32), None)
    lengths = [1, 2, 3, 7, 64, 255, 256, 257, 511, 600, 1023, 1500, 2049, 3001]
    for length in lengths + lengths[::-1]:
        table = position_table(length, width, device, torch.float32)
        assert table.shape == (length, width) and not table.requires_grad
        assert torch.equal(table, sinusoidal(length, width, device, torch.float32)), length
    cached = history_model._POSITION_TABLES[(width, device, torch.float32)]
    assert len(cached) >= 3001
    # Every shorter length is a prefix of the grown table, bit for bit.
    for length in range(1, len(cached) + 1, 37):
        assert torch.equal(cached[:length], sinusoidal(length, width, device, torch.float32))
    wide = position_table(5, width, device, torch.float64)
    assert wide.dtype == torch.float64 and torch.equal(
        wide, sinusoidal(5, width, device, torch.float64))
    with torch.inference_mode():
        history_model._POSITION_TABLES.pop((width, device, torch.float32))
        position_table(9, width, device, torch.float32)
    assert not history_model._POSITION_TABLES[(width, device, torch.float32)].is_inference()
    # A table first built under inference mode still serves autograd.
    x = torch.zeros(9, width, requires_grad=True)
    (x + position_table(9, width, device, torch.float32)).sum().backward()
    assert torch.equal(x.grad, torch.ones(9, width))


def random_tokens(count: int, generator: torch.Generator) -> torch.Tensor:
    """Public-token-shaped rows: one-hot fields, card bits, some bombs, some all-zero rows."""
    tokens = torch.zeros(count, TOKEN_DIM, dtype=torch.uint8)
    pick = lambda high: torch.randint(0, high, (count,), generator=generator)
    for name, field in ONE_HOT_FIELDS:
        tokens[torch.arange(count), field.start + pick(field.stop - field.start)] = 1
    cards = torch.rand(count, 108, generator=generator) < 0.08
    tokens[:, 4:112] = cards.to(torch.uint8)
    bombs = torch.rand(count, generator=generator) < 0.2
    tokens[torch.arange(count)[bombs], BOMB.start + pick(BOMB.stop - BOMB.start)[bombs]] = 1
    tokens[torch.rand(count, generator=generator) < 0.1] = 0          # padding rows
    return tokens


@pytest.mark.parametrize("float_mask", [False, True])
def test_next_token_loss_is_the_reference(float_mask):
    generator = torch.Generator().manual_seed(3)
    tokens = random_tokens(3000, generator)
    logits = (torch.randn(3000, NEXT_LOGITS, generator=generator) * 2)
    logits[:5, NEXT_SLICES["seat"]] = 0.0           # argmax ties: first index
    valid = torch.rand(3000, generator=generator) < 0.8
    results = []
    for fn in (legacy_next_token_loss, next_token_loss):
        x = logits.clone().requires_grad_()
        stats = fn(x, tokens, valid.float() if float_mask and fn is next_token_loss else valid)
        stats["next_loss"].backward()
        results.append((stats, x.grad))
    (ref, ref_grad), (new, new_grad) = results
    assert ref.keys() == new.keys()
    for key in ref:
        assert torch.equal(ref[key], new[key]), key
    assert torch.equal(ref_grad, new_grad)
    none = next_token_loss(logits, tokens, torch.zeros(3000, dtype=torch.bool))
    assert all(value.item() == 0.0 for value in none.values())


def test_targets_are_each_fields_argmax():
    tokens = random_tokens(500, torch.Generator().manual_seed(5))
    targets = next_token_targets(tokens)
    assert len(targets) == len(ONE_HOT_FIELDS) + 1
    for target, (_, field) in zip(targets, ONE_HOT_FIELDS):
        assert torch.equal(target, tokens[:, field].argmax(-1))
    bomb = tokens[:, BOMB]
    expected = torch.where(bomb.sum(-1) > 0, bomb.argmax(-1), BOMB_CLASSES - 1)
    assert torch.equal(targets[-1], expected) and (expected == BOMB_CLASSES - 1).any()
    assert (expected < BOMB_CLASSES - 1).any()


def test_zero_bridge_keeps_an_exact_zero_gradient(exact_cpu):
    torch.manual_seed(2)
    actor = HistoryActor(HistoryPolicyConfig(width=32, layers=1, heads=4,
                                             response_mode="auxiliary"))
    counts = torch.tensor([3, 1, 5, 2])
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    state = torch.randn(len(counts), 32, requires_grad=True)
    cand = torch.randint(0, 2, (int(offsets[-1]), actor.config.act_dim), dtype=torch.uint8)
    results = []
    for outputs in (legacy_candidate_outputs, HistoryActor.candidate_outputs):
        actor.zero_grad(set_to_none=True)
        state.grad = None
        logits, response = outputs(actor, state, cand, offsets, predict=True)
        (logits.square().sum() + response.logsumexp(-1).sum()).backward()
        results.append((logits.detach(), state.grad.clone(),
                        [None if p.grad is None else p.grad.clone() for p in actor.parameters()]))
    (l0, s0, g0), (l1, s1, g1) = results
    assert torch.equal(l0, l1) and torch.equal(s0, s1)
    for a, b in zip(g0, g1):
        assert (a is None and b is None) or torch.equal(a, b)
    bridge = actor.response_bridge.weight.grad
    assert bridge is not None and bridge.shape == actor.response_bridge.weight.shape
    assert not bridge.any()


def test_training_batch_plans_rows_and_next_token_mask(tmp_path, collected):
    trainer = finalized(tmp_path, collected)
    rows = trainer.buffer.samples
    for groups in (0, 1, 3):
        inputs = trainer.buffer.training_batch(rows, trainer.store, "cpu", length_groups=groups,
                                               width=16).inputs
        counts = inputs.offsets[1:] - inputs.offsets[:-1]
        assert torch.equal(inputs.candidate_rows, torch.repeat_interleave(
            torch.arange(inputs.decisions), counts))
        if not groups:
            assert inputs.next_weight is None and inputs.next_positions == 0
            continue
        s, masks = inputs.streams, []
        for group in inputs.match_groups:
            lengths = s.lengths[group.matches].clamp(max=group.tokens)
            cited = torch.zeros_like(lengths).scatter_reduce(
                0, group.local_match, inputs.prefix[group.rows], reduce="amax")
            limit = torch.minimum(lengths, cited)
            masks.append((torch.arange(group.tokens)[None] < limit[:, None]).reshape(-1))
        mask = torch.cat(masks)
        assert inputs.next_weight.dtype == torch.float32
        assert torch.equal(inputs.next_weight, mask.float())
        assert inputs.next_positions == int(mask.sum()) > 0


# ---- learner_compact_next (tier 2) ---------------------------------------------------

def test_compact_next_loss_matches_the_default_path(tmp_path, collected, exact_cpu):
    trainer = finalized(tmp_path, collected, **WIDE)
    rows = next(iter(trainer.minibatches()))
    params = list(trainer.actor.parameters())
    results = []
    for compact in (False, True):
        trainer.config = replace(trainer.config, learner_compact_next=compact)
        terms = trainer.minibatch_loss(rows)
        grads = torch.autograd.grad(terms["policy_total"], params, allow_unused=True)
        results.append((terms, grads))
    (ref, ref_grads), (compact, grads) = results
    assert ref.keys() == compact.keys()
    worst = 0.0
    for key in ref:
        if key.startswith("next_") or key in ("aux_loss", "policy_total"):
            scale = max(ref[key].abs().item(), 1e-6)
            worst = max(worst, (compact[key] - ref[key]).abs().item() / scale)
            torch.testing.assert_close(compact[key], ref[key], rtol=1e-5, atol=1e-6, msg=key)
        else:
            assert torch.equal(compact[key], ref[key]), key   # untouched by the head
    names = [name for name, _ in trainer.actor.named_parameters()]
    for name, a, b in zip(names, grads, ref_grads):
        assert (a is None) == (b is None), name
        if b is not None:
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-7, msg=name)
    assert worst < 1e-5


def test_compact_loss_alone_matches_next_token_loss():
    generator = torch.Generator().manual_seed(9)
    tokens = random_tokens(4000, generator)
    valid = torch.rand(4000, generator=generator) < 0.7
    logits = torch.randn(4000, NEXT_LOGITS, generator=generator) * 3
    a = logits.clone().requires_grad_()
    ref = next_token_loss(a, tokens, valid)
    ref["next_loss"].backward()
    b = logits[valid].clone().requires_grad_()
    compact = next_token_loss_compact(b, tokens[valid])
    compact["next_loss"].backward()
    assert ref.keys() == compact.keys()
    for key in ref:
        torch.testing.assert_close(compact[key], ref[key], rtol=1e-5, atol=1e-6, msg=key)
    torch.testing.assert_close(b.grad, a.grad[valid], rtol=1e-4, atol=1e-9)
    assert not a.grad[~valid].any()
    empty = next_token_loss_compact(logits[:0].clone().requires_grad_(), tokens[:0])
    assert all(value.item() == 0.0 for value in empty.values())


def test_compact_switch_defaults_off_and_is_resumable(tmp_path):
    assert HistoryPPOConfig().learner_compact_next is False
    assert build_parser().parse_args(["--output", "x"]).learner_compact_next is False
    assert build_parser().parse_args(["--output", "x", "--learner-compact-next"]).learner_compact_next
    with pytest.raises(ValueError):
        HistoryPPOConfig(learner_compact_next=True)            # needs length groups
    assert "learner_compact_next" in RESUME_OVERRIDES
    assert parse_resume_overrides(["learner_compact_next=true"]) == {"learner_compact_next": True}
    from bench.history_arms import SWITCHABLE
    assert "learner_compact_next" in SWITCHABLE
    trainer = HistoryTrainer(HistoryPPOConfig(**{**LEARNER, **TINY, "learner_compact_next": True}),
                             tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["inference"]["learner_compact_next"] is True
    assert trainer.config.learner_compact_next
