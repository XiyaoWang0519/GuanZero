"""Bitwise rollout sampler trims (unconditional; acceptance tier 1).

The learner's ``HistoryActor.explore``/``act`` and the merged snapshot
softmax and sampler (``merged_log_probs``, ``merged_sample``) shed device
kernels without changing a bit: host-planned candidate counts and flat
sampling-table cells ride the step's upload, the no-grad segment log-softmax
forms its shift once, the Gumbel noise skips a negation (IEEE ``a + (-b) ==
a - b``), ``behaviour_logp`` aliases ``logp`` without exploration (and is not
downloaded twice), the merged mask is one comparison and the rollout
attention's position ``arange`` comes from a cached buffer.

The references are the pre-change functions, copied verbatim below. Every
observable (choices, log-probabilities, behaviour log-probabilities, uniform
picks, entropies, generator state, the stored buffer and the collect
counters) must be bit for bit equal, on random ragged cases and on real
VecEnv rollouts with the production rollout options and snapshot seats.
"""
from __future__ import annotations

import collections
from dataclasses import asdict

import gd
import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import train.history_model as hm
import train.history_snapshot_batch as hsb
from train import history_rollout as rollout
from train.history_model import (DecisionInputs, ExplorationSample, HistoryActor,
                                 HistoryPolicyConfig, StreamBatch, fresh_player)
from test_history_snapshot_batch import SPECS, Step, merged, players, run_collector

CONFIG = HistoryPolicyConfig(width=32, layers=1, heads=4, response_mode="auxiliary")


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA bitwise sampler acceptance")
    return request.param


@pytest.fixture(autouse=True)
def strict():
    # CUDA index_add_ is nondeterministic without deterministic algorithms.
    previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.set_num_threads(previous[0])
        torch.use_deterministic_algorithms(previous[1], warn_only=previous[2])


# ---- pre-change code, copied verbatim (modulo self -> actor) -----------------------

def legacy_segment_log_softmax(scores, rows, count):
    top = torch.full((count,), float("-inf"), device=scores.device, dtype=scores.dtype)
    top = top.scatter_reduce(0, rows, scores, reduce="amax")
    shifted = (scores - top[rows]).exp()
    total = torch.zeros(count, device=scores.device, dtype=scores.dtype).index_add_(0, rows, shifted)
    return scores - top[rows] - total[rows].log()


def legacy_gumbel_argmax(table, generator):
    uniform = torch.rand(table.shape, generator=generator, device=table.device,
                         dtype=table.dtype).clamp_min(1e-12)
    gumbel = -(-uniform.log()).log()
    return (table + gumbel).argmax(1)


def legacy_log_prob_table(actor, inputs, encoded, max_candidates, inference_log_probs=None):
    if inference_log_probs is None:
        state = actor.decision_states(encoded, inputs)
        rows = inputs.rows
        logits = actor.candidate_logits(state, inputs.cand, inputs.offsets, rows=rows)
        log_probs = legacy_segment_log_softmax(logits, rows, inputs.decisions)
    else:
        log_probs = inference_log_probs
    counts = inputs.counts
    offsets = inputs.offsets
    longest = max_candidates if max_candidates is not None else (int(counts.max()) if len(counts) else 0)
    table = torch.full((inputs.decisions, max(longest, 1)), float("-inf"),
                       device=log_probs.device, dtype=log_probs.dtype)
    rows = inputs.rows
    local = torch.arange(len(log_probs), device=log_probs.device) - offsets[:-1][rows]
    table[rows, local] = log_probs
    return table


@torch.no_grad()
def legacy_explore(actor, inputs, generator=None, *, temperature=1.0, epsilon=0.0,
                   encoded=None, max_candidates=None, inference_log_probs=None):
    table = legacy_log_prob_table(actor, inputs, encoded, max_candidates, inference_log_probs)
    counts = inputs.counts
    valid = torch.arange(table.shape[1], device=table.device)[None] < counts[:, None]
    behaviour = table if temperature == 1 else torch.log_softmax(table / temperature, 1)
    choice = legacy_gumbel_argmax(behaviour, generator)
    uniform_pick = torch.zeros(len(choice), dtype=torch.bool, device=choice.device)
    if epsilon > 0:
        coin = torch.rand(len(choice), generator=generator, device=table.device,
                          dtype=torch.float64)
        draw = torch.rand(len(choice), generator=generator, device=table.device,
                          dtype=torch.float64)
        index = torch.minimum((draw * counts).long(), counts - 1)
        uniform_pick = coin < epsilon
        choice = torch.where(uniform_pick, index, choice)
        import math
        mixed = torch.logaddexp(behaviour + math.log1p(-epsilon) if epsilon < 1
                                else torch.full_like(behaviour, float("-inf")),
                                (math.log(epsilon) - counts.to(table.dtype).log())[:, None])
        behaviour = torch.where(valid, mixed, table)
    logp = table.gather(1, choice[:, None])[:, 0]
    behaviour_logp = behaviour.gather(1, choice[:, None])[:, 0]
    terms = torch.where(valid, behaviour.exp() * behaviour, torch.zeros_like(behaviour))
    return ExplorationSample(choice, logp, behaviour_logp, uniform_pick, -terms.sum(1))


@torch.no_grad()
def legacy_act(actor, inputs, generator=None, greedy=False, *, encoded=None,
               max_candidates=None, inference_log_probs=None):
    table = legacy_log_prob_table(actor, inputs, encoded, max_candidates, inference_log_probs)
    choice = table.argmax(1) if greedy else legacy_gumbel_argmax(table, generator)
    chosen = table.gather(1, choice[:, None])[:, 0]
    return choice, chosen


@torch.no_grad()
def legacy_merged_log_probs(heads, layout, fields, memory):
    _linear, _norm, _pad = hsb._linear, hsb._norm, hsb._pad
    tensors, cfg = heads.tensors, heads.config
    S, M, Cm = layout.slots, layout.rows_per_slot, layout.cands_per_slot
    width, heads_n = cfg.width, cfg.heads
    depth = width // heads_n
    rows = len(layout.rows)
    x = _pad(fields["obs"].float(), fields["row_pad"], S * M).view(S, M, -1)
    x = torch.relu(_linear(x, tensors, "p0", S))
    x = torch.relu(_linear(x, tensors, "p1", S))
    query = x + tensors["seat"][:S].reshape(S * 4, width)[fields["seat_pad"]].view(S, M, width)
    q = _linear(query, tensors, "q", S).view(S, M, heads_n, depth).transpose(1, 2)
    folded = torch.matmul(q, tensors["k_w"][:S]) * depth ** -0.5
    folded = folded.transpose(1, 2).reshape(S * M, heads_n, width)[fields["row_pad"]]
    own = memory if layout.identity_streams else memory[fields["stream"]]
    scores = torch.bmm(folded, own.transpose(1, 2))
    allowed = torch.arange(own.shape[1], device=own.device)[None] <= fields["prefix"][:, None]
    scores = scores.masked_fill(~allowed[:, None], float("-inf"))
    context = torch.bmm(torch.softmax(scores, dim=-1), own)
    context = _pad(context, fields["row_pad"], S * M).view(S, M, heads_n, width).transpose(1, 2)
    attended = torch.matmul(context, tensors["v_w"][:S])
    attended = attended.transpose(1, 2).reshape(S, M, width) + tensors["v_b"][:S]
    attended = _linear(attended, tensors, "o", S)
    state = _norm(query + attended, tensors, "n1", S, heads.eps)
    hidden = _linear(torch.relu(_linear(state, tensors, "f0", S)), tensors, "f1", S)
    state = _norm(state + hidden, tensors, "n2", S, heads.eps)
    actions = _pad(fields["cand"].float(), fields["cand_pad"], S * Cm).view(S, Cm, -1)
    actions = torch.relu(_linear(actions, tensors, "a0", S))
    actions = torch.relu(_linear(actions, tensors, "a1", S))
    states = state.reshape(S * M, width)[fields["cand_state_pad"]].view(S, Cm, width)
    fused = torch.cat((states, actions), dim=-1)
    fused = torch.relu(_linear(fused, tensors, "u0", S))
    fused = torch.relu(_linear(fused, tensors, "u1", S))
    logits = _linear(fused, tensors, "u2", S).reshape(S * Cm)[fields["cand_pad"]]
    return legacy_segment_log_softmax(logits, fields["cand_row"], rows)


@torch.no_grad()
def legacy_merged_sample(log_probs, layout, fields, generator):
    rows, K = len(layout.rows), layout.table_width
    table = torch.full((rows * K,), float("-inf"), device=log_probs.device,
                       dtype=log_probs.dtype)
    table = table.index_put_((fields["cand_table"],), log_probs).view(rows, K)
    uniform = torch.empty(layout.draws[-1][1], device=log_probs.device, dtype=log_probs.dtype)
    for begin, end in layout.draws:
        uniform[begin:end].uniform_(generator=generator)
    columns = torch.arange(K, device=log_probs.device)[None]
    index = fields["uniform_base"][:, None] + torch.minimum(columns, fields["uniform_last"][:, None])
    uniform = uniform[index].clamp_min(1e-12)
    choice = (table + -(-uniform.log()).log()).argmax(1)
    return choice, table.gather(1, choice[:, None])[:, 0]


def use_legacy_sampler(monkeypatch):
    """The whole pre-change sampling path inside a current collector."""
    monkeypatch.setattr(hm, "segment_log_softmax", legacy_segment_log_softmax)
    monkeypatch.setattr(hm, "_gumbel_argmax", legacy_gumbel_argmax)
    monkeypatch.setattr(HistoryActor, "explore", legacy_explore)
    monkeypatch.setattr(HistoryActor, "act", legacy_act)
    monkeypatch.setattr(hsb, "merged_log_probs", legacy_merged_log_probs)
    monkeypatch.setattr(hsb, "merged_sample", legacy_merged_sample)


# ---- helpers ------------------------------------------------------------------------

VIEWS = {"view", "_unsafe_view", "transpose", "t", "permute", "expand", "select", "slice",
         "unsqueeze", "squeeze", "as_strided", "alias", "split", "detach", "lift_fresh",
         "_reshape_alias"}


class OpCount(TorchDispatchMode):
    """Non-view aten operations (about one kernel launch each on a GPU)."""

    def __init__(self):
        super().__init__()
        self.ops = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops[func.overloadpacket.__name__] += 1
        return func(*args, **(kwargs or {}))

    @property
    def launches(self):
        return sum(n for name, n in self.ops.items() if name not in VIEWS)


def assert_bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def random_inputs(rng, n, device, planned):
    """Ragged decisions with an encoded memory, optionally with the collector's
    host-planned counts and table cells (built here independently)."""
    counts = rng.integers(1, 60, n)
    counts[rng.integers(n)] = int(rng.integers(60, 300))
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    prefix = rng.integers(0, 50, n)
    rows = np.repeat(np.arange(n), counts)
    K = int(counts.max())
    planned_fields = {}
    if planned:
        local = np.arange(offsets[-1]) - offsets[:-1][rows]
        planned_fields = dict(candidate_counts=torch.as_tensor(counts, device=device),
                              table_index=torch.as_tensor(rows * K + local, device=device),
                              table_width=K)
    as_t = lambda a: torch.as_tensor(a, device=device)
    streams = StreamBatch(torch.empty(n, 0, 186, dtype=torch.uint8, device=device),
                          torch.empty(n, 0, dtype=torch.long, device=device),
                          torch.empty(n, 0, dtype=torch.long, device=device), as_t(prefix))
    inputs = DecisionInputs(
        streams=streams, match_index=as_t(np.arange(n)), prefix=as_t(prefix),
        obs=as_t(rng.integers(0, 3, (n, int(gd.OBS_DIM)), dtype=np.uint8)),
        seat=as_t(rng.integers(0, 4, n)),
        cand=as_t(rng.integers(0, 3, (offsets[-1], int(gd.ACT_DIM)), dtype=np.uint8)),
        offsets=as_t(offsets), candidate_rows=as_t(rows), one_decision_per_stream=True,
        **planned_fields)
    span = (int(prefix.max()) + 64) // 64 * 64
    encoded = torch.randn(n, span, CONFIG.width, generator=torch.Generator().manual_seed(n)).to(device)
    return inputs, encoded, K


def batched_actor(device):
    actor = fresh_player(CONFIG, 3)[0].to(device)
    actor.batched_private_attention = True
    actor.wide_private_projection = True
    return actor


# ---- unit level ------------------------------------------------------------------------

def test_no_grad_segment_log_softmax_is_bitwise(device):
    rng = np.random.default_rng(0)
    for trial in range(200):
        count = int(rng.integers(1, 120))
        lengths = rng.integers(1, 40, count)
        host_rows = np.repeat(np.arange(count), lengths)
        rows = torch.as_tensor(host_rows, device=device)
        scores = torch.as_tensor(rng.standard_normal(int(lengths.sum())) * 10 ** rng.uniform(-3, 2),
                                 dtype=torch.float32, device=device)
        if trial % 7 == 0:          # -inf candidates, never a row's first
            later = np.r_[False, host_rows[1:] == host_rows[:-1]]
            scores[torch.as_tensor(later & (rng.random(len(scores)) < 0.2), device=device)] = \
                float("-inf")
        expected = legacy_segment_log_softmax(scores, rows, count)
        with torch.no_grad():
            assert_bits(hm.segment_log_softmax(scores, rows, count), expected)
        assert_bits(hm.segment_log_softmax(scores, rows, count), expected)  # no requires_grad
    # The autograd path is the original expression: same values and gradient.
    weights = torch.randn(len(scores), device=device)
    leaves = [scores.clone().requires_grad_(), scores.clone().requires_grad_()]
    outputs = [hm.segment_log_softmax(leaves[0], rows, count),
               legacy_segment_log_softmax(leaves[1], rows, count)]
    for out, leaf in zip(outputs, leaves):
        (out.nan_to_num(0.0, 0.0, 0.0) * weights).sum().backward()
    assert_bits(outputs[0].detach(), outputs[1].detach())
    assert_bits(leaves[0].grad, leaves[1].grad)


@pytest.mark.parametrize("temperature,epsilon", [(1.0, 0.0), (0.7, 0.0), (1.0, 0.1), (0.7, 0.1)])
def test_explore_and_act_match_the_pre_change_sampler(device, temperature, epsilon):
    actor = batched_actor(device)
    rng = np.random.default_rng(int(temperature * 10 + epsilon * 100))
    for trial in range(25):
        n = int(rng.integers(1, 90))
        for planned in (True, False):
            inputs, encoded, K = random_inputs(rng, n, device, planned)
            generators = [torch.Generator(device=device).manual_seed(trial) for _ in range(4)]
            before = legacy_explore(actor, inputs, generators[0], temperature=temperature,
                                    epsilon=epsilon, encoded=encoded, max_candidates=K)
            after = actor.explore(inputs, generators[1], temperature=temperature,
                                  epsilon=epsilon, encoded=encoded, max_candidates=K)
            for name in ("choice", "logp", "behaviour_logp", "uniform_pick", "behaviour_entropy"):
                assert_bits(getattr(after, name), getattr(before, name))
            assert (after.behaviour_logp is after.logp) == (temperature == 1 and epsilon == 0)
            assert_bits(generators[1].get_state(), generators[0].get_state())
            for greedy in (False, True):
                old = legacy_act(actor, inputs, generators[2], greedy, encoded=encoded,
                                 max_candidates=K)
                new = actor.act(inputs, generators[3], greedy, encoded=encoded, max_candidates=K)
                assert all(torch.equal(a, b) for a, b in zip(new, old))
                assert_bits(generators[3].get_state(), generators[2].get_state())


def test_planned_table_needs_the_matching_width(device):
    # A table width other than the planned one falls back to the device scatter.
    actor = batched_actor(device)
    inputs, encoded, K = random_inputs(np.random.default_rng(5), 12, device, True)
    with torch.no_grad():
        wide = actor._log_prob_table(inputs, encoded, K + 3)
        assert wide.shape == (12, K + 3)
        assert_bits(wide, legacy_log_prob_table(actor, inputs, encoded, K + 3))
        assert_bits(actor._log_prob_table(inputs, encoded, None),
                    legacy_log_prob_table(actor, inputs, encoded, None))


def test_explore_launches_fewer_kernels():
    actor = batched_actor("cpu")
    inputs, encoded, K = random_inputs(np.random.default_rng(9), 40, "cpu", True)
    counts = []
    for explore in (legacy_explore, HistoryActor.explore):
        with OpCount() as ops:
            explore(actor, inputs, torch.Generator().manual_seed(1), encoded=encoded,
                    max_candidates=K)
        counts.append(ops.launches)
    print(f"learner explore non-view aten ops: {counts[0]} -> {counts[1]}")
    assert counts[1] <= counts[0] - 9


@pytest.mark.parametrize("name", list(SPECS))
def test_merged_softmax_and_sampler_match_the_pre_change_code(device, name):
    step = Step(SPECS[name], seed=list(SPECS).index(name) + 20,
                shorter_prefix=name == "multi-decision")
    policies = players({i for i, _, _ in SPECS[name]}, device)
    groups, layout, fields, log_probs = merged(step, policies, device)
    memory = torch.zeros(layout.streams, layout.length, policies[groups[0].identity].config.width,
                         device=device)
    with torch.no_grad():
        for group, (first, last) in zip(groups, layout.group_streams):
            encoded = policies[group.identity].encode_batch(
                StreamBatch.from_streams(group.streams, device))
            memory[first:last, :encoded.shape[1]] = encoded
        heads = hsb.SnapshotHeads()
        for group in groups:
            heads.slot(group.identity, policies[group.identity])
        assert_bits(hsb.merged_log_probs(heads, layout, fields, memory),
                    legacy_merged_log_probs(heads, layout, fields, memory))
    for seed in range(4):
        generators = [torch.Generator(device=device).manual_seed(seed) for _ in range(2)]
        before = legacy_merged_sample(log_probs, layout, fields, generators[0])
        after = hsb.merged_sample(log_probs, layout, fields, generators[1])
        assert all(torch.equal(a, b) for a, b in zip(after, before))
        assert_bits(generators[1].get_state(), generators[0].get_state())


# ---- collector replay ------------------------------------------------------------------

PRODUCTION = dict(kv_cache=True, paged_cache=True, page_span=True, batch_snapshot_encoder=True,
                  prefill_learner_cache=True)


def production_players(device):
    policies = players((0, 1, 2, 3), device)
    for actor in policies.values():
        actor.causal_sdpa = True
    policies[0].batched_private_attention = True
    policies[0].wide_private_projection = True
    return policies


def collect_run(device, merge, **options):
    return run_collector(device, merge, steps=60, policies=production_players(device), **options)


@pytest.mark.parametrize("merge,epsilon", [(True, 0.0), (True, 0.05), (False, 0.0)])
def test_collector_replay_is_bitwise(device, merge, epsilon, monkeypatch):
    """Production rollout options with snapshot seats (merged or per identity):
    the trimmed sampler collects the identical rollout."""
    options = dict(PRODUCTION, batch_snapshot_encoder=merge, epsilon=epsilon)
    fast, fast_stats = collect_run(device, merge, **options)
    with monkeypatch.context() as patch:
        use_legacy_sampler(patch)
        plain, plain_stats = collect_run(device, merge, **options)
    assert len(plain.choice_log) == len(fast.choice_log) == 60
    for before, after in zip(plain.choice_log, fast.choice_log):
        assert before.tobytes() == after.tobytes()
    data, fast_data = plain.buffer.compact(), fast.buffer.compact()
    assert set(data) == set(fast_data)
    for name, before in data.items():
        assert before.dtype == fast_data[name].dtype and before.tobytes() == fast_data[name].tobytes()
    assert_bits(fast.generator.get_state(), plain.generator.get_state())
    assert fast.policy_decisions == plain.policy_decisions
    for before, after in zip(plain_stats, fast_stats):
        assert asdict(before) == asdict(after)      # entropy sum and epsilon picks included
    assert sum(s.learner_rows for s in fast_stats) > 0
    if epsilon:
        assert sum(s.epsilon_picks for s in fast_stats) > 0
    else:
        assert np.array_equal(data["behaviour_logp"], data["logp"])
