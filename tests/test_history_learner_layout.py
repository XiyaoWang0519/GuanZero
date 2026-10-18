"""Learner layout and response-head savings.

Host-planned match attention layout: ``training_batch`` computes each row's
rank within its match and the per-match slot count on the host, the same
integers ``_attend_by_match`` would otherwise derive on the device (with a host
sync). Identical integers, so identical floats: checked bitwise.

Chosen-only auxiliary response: the response head is row-wise, so predicting
only the executed candidates gives the same loss and gradients as indexing the
full prediction afterwards, up to FP32 reduction order.

The auxiliary bridge adds an exact zero; inference skips it bitwise.
"""
import os

import numpy as np
import pytest
import torch

from train.history_model import HistoryActor, HistoryPolicyConfig
from train.history_ppo import HistoryPPOConfig, HistoryTrainer

DEVICES = ["cpu", "cuda"]
# Deterministic cuBLAS for the bitwise comparison; must precede cuBLAS init.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def trainer(tmp_path, device="cpu", **overrides):
    config = dict(width=32, layers=2, heads=4, num_envs=6, steps_per_update=80, seed=5,
                  rollout_kv_cache=True, minibatch_matches=6, snapshot_updates=1,
                  learner_batched_attention=True, response_mode="auxiliary")
    config.update(overrides)
    t = HistoryTrainer(HistoryPPOConfig(**config), tmp_path, device=device)
    for _ in range(2):
        t.update()
    t.collect()
    t.buffer.finalize(t.refresh_values())
    return t


def device_layout(match_index: torch.Tensor, matches: int) -> tuple[torch.Tensor, int]:
    """The device derivation ``_attend_by_match`` falls back to."""
    per_match = torch.bincount(match_index, minlength=matches)
    order = torch.argsort(match_index, stable=True)
    starts = torch.cumsum(per_match, 0) - per_match
    rank = torch.empty_like(match_index)
    rank[order] = torch.arange(len(match_index), device=match_index.device) - starts[match_index[order]]
    return rank, int(per_match.max())


def skip_without(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA acceptance")


def test_layout_keeps_existing_packed_field_alignment(tmp_path, monkeypatch):
    from train import history_rollout
    from train.history_transfers import upload_arrays

    captured = {}

    def packed_upload(arrays, device):
        tensors = upload_arrays(arrays, device, packed=True)
        captured.update(arrays=arrays, tensors=tensors)
        return tensors

    t = trainer(tmp_path)
    # An odd row count moves later fields by eight bytes if ranks precede them.
    rows = t.buffer.samples[:37]
    assert len(rows) == 37
    monkeypatch.setattr(history_rollout, "upload_arrays", packed_upload)
    batch = t.buffer.training_batch(rows, t.store, "cpu")
    rank_index = next(i for i, value in enumerate(captured["tensors"])
                      if value is batch.inputs.match_rank)
    legacy = upload_arrays([array for i, array in enumerate(captured["arrays"])
                            if i != rank_index], "cpu", packed=True)
    for i, name in enumerate(batch.fields):
        # CUDA reduction kernels can change float order with vector alignment.
        assert batch.fields[name].data_ptr() % 16 == legacy[11 + i].data_ptr() % 16, name


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("groups", [0, 1, 3])
def test_host_layout_equals_device_layout(tmp_path, device, groups):
    skip_without(device)
    t = trainer(tmp_path, device)
    rng = np.random.default_rng(groups)
    rows = rng.permutation(t.buffer.samples)[: len(t.buffer.samples) * 2 // 3]
    batch = t.buffer.training_batch(rows, t.store, device, length_groups=groups, width=32)
    inputs = batch.inputs
    rank, slots = device_layout(inputs.match_index, len(inputs.streams.lengths))
    assert torch.equal(inputs.match_rank, rank) and inputs.match_slots == slots
    for g in inputs.match_groups or []:
        rank, slots = device_layout(g.local_match, len(g.matches))
        assert torch.equal(g.match_rank, rank) and g.match_slots == slots


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("groups", [0, 3])
def test_planned_layout_is_bitwise(tmp_path, device, groups):
    skip_without(device)
    t = trainer(tmp_path, device, learner_length_groups=groups)
    rows = t.buffer.samples
    results = []
    # Same-path backward is not run-to-run bitwise on the CPU encoder without it.
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    # warn_only=True permits the nondeterministic CUDA attention backward.
    torch.use_deterministic_algorithms(True)
    try:
        for planned in (True, False):
            batch = t.buffer.training_batch(rows, t.store, device, length_groups=groups, width=32)
            if not planned:
                batch.inputs.match_rank = None
                for g in batch.inputs.match_groups or []:
                    g.match_rank = None
            t.actor.zero_grad()
            state = t.actor.decision_states(None, batch.inputs)
            log_probs = t.actor.candidate_log_probs(batch.inputs)
            (state.square().sum() + log_probs[batch.chosen].sum()).backward()
            results.append((state.detach(), log_probs.detach(),
                            [None if p.grad is None else p.grad.clone()
                             for p in t.actor.parameters()]))
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)
    (s0, l0, g0), (s1, l1, g1) = results
    assert torch.equal(s0, s1) and torch.equal(l0, l1)
    names = [name for name, _ in t.actor.named_parameters()]
    for name, a, b in zip(names, g0, g1):
        assert (a is None) == (b is None), name
        if a is not None:
            assert torch.equal(a, b), (name, (a - b).abs().max().item())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("groups", [0, 3])
def test_chosen_response_matches_full_response(tmp_path, device, groups):
    skip_without(device)
    t = trainer(tmp_path, device, learner_length_groups=groups)
    rows = t.buffer.samples
    results = []
    for chosen_only in (False, True):
        t.config.learner_chosen_response = chosen_only
        terms = t.minibatch_loss(rows)
        # The actor's share of the update: policy terms plus the response loss.
        total = terms["policy_total"] + terms["response_loss"]
        grads = torch.autograd.grad(total, list(t.actor.parameters()), allow_unused=True)
        results.append((terms, grads))
    (ref, ref_grads), (chosen, grads) = results
    for key in ("response_loss", "response_accuracy", "response_event_fraction",
                "policy_loss", "entropy", "approx_kl"):
        torch.testing.assert_close(chosen[key], ref[key], rtol=1e-5, atol=1e-6)
    named = [name for name, _ in t.actor.named_parameters()]
    assert any(name.startswith("response_head") for name in named)
    for name, a, b in zip(named, grads, ref_grads):
        assert (a is None) == (b is None), name
        if b is not None:
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6, msg=name)


def test_response_rows_are_the_indexed_prediction():
    torch.manual_seed(0)
    actor = HistoryActor(HistoryPolicyConfig(width=32, layers=1, heads=4,
                                             response_mode="auxiliary"))
    counts = torch.tensor([3, 1, 5, 2])
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    state = torch.randn(len(counts), 32)
    cand = torch.randint(0, 2, (int(offsets[-1]), actor.config.act_dim), dtype=torch.uint8)
    picked = offsets[:-1] + torch.tensor([2, 0, 4, 1])
    logits, full = actor.candidate_outputs(state, cand, offsets, predict=True)
    same_logits, part = actor.candidate_outputs(state, cand, offsets, predict=True,
                                                response_rows=picked)
    assert torch.equal(logits, same_logits)
    torch.testing.assert_close(part, full[picked], rtol=0, atol=1e-6)
    explicit = HistoryActor(HistoryPolicyConfig(width=32, layers=1, heads=4,
                                                response_mode="explicit"))
    with pytest.raises(ValueError):
        explicit.candidate_outputs(state, cand, offsets, predict=True, response_rows=picked)


@pytest.mark.parametrize("device", DEVICES)
def test_inference_skips_the_zero_bridge_bitwise(device):
    skip_without(device)
    torch.manual_seed(1)
    actor = HistoryActor(HistoryPolicyConfig(width=32, layers=1, heads=4,
                                             response_mode="auxiliary")).to(device)
    # A trained bridge weight is still exactly zero: its only input is zeros.
    assert not actor.response_bridge.weight.any()
    counts = torch.tensor([4, 1, 7], device=device)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)))
    state = torch.randn(len(counts), 32, device=device)
    cand = torch.randint(0, 2, (int(offsets[-1]), actor.config.act_dim), dtype=torch.uint8,
                         device=device)
    trained = actor.candidate_logits(state, cand, offsets)
    with torch.no_grad():
        skipped = actor.candidate_logits(state, cand, offsets)
    with torch.inference_mode():
        inferred = actor.candidate_logits(state, cand, offsets)
    assert torch.equal(trained.detach(), skipped) and torch.equal(skipped, inferred)


def test_config_resume_and_manifest(tmp_path):
    from train.history_ppo import RESUME_OVERRIDES, parse_resume_overrides
    with pytest.raises(ValueError):
        HistoryPPOConfig(learner_chosen_response=True)
    with pytest.raises(ValueError):
        HistoryPPOConfig(learner_chosen_response=True, response_mode="explicit")
    HistoryPPOConfig(learner_chosen_response=True, response_mode="auxiliary")
    assert "learner_chosen_response" in RESUME_OVERRIDES
    assert parse_resume_overrides(["learner_chosen_response=true"]) == {
        "learner_chosen_response": True}
    t = HistoryTrainer(HistoryPPOConfig(width=32, layers=1, heads=4, num_envs=4,
                                        steps_per_update=60, seed=2, minibatch_matches=2,
                                        response_mode="auxiliary", learner_chosen_response=True,
                                        learner_batched_attention=True),
                       tmp_path)
    import json
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["inference"]["learner_chosen_response"] is True
    stats = t.update()
    assert np.isfinite(stats["response_loss"])
    assert all(torch.isfinite(p).all() for p in t.actor.parameters())
