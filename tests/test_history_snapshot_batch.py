"""Opt-in merged snapshot inference (``batch_snapshot_policies``).

Acceptance tier 2 for frozen snapshot seats: the merged call over stacked head
weights computes the same per-candidate log-probabilities as each identity's
own ``candidate_log_probs`` up to FP32 reduction order, and draws the same
uniforms from the same generator in the same order. With the flag off the
collector is untouched (bitwise). Learner rows are never evaluated by the
merged path.
"""
from dataclasses import asdict
import math

import numpy as np
import pytest
import torch

from train import history_rollout as rollout
from train.history_model import (DecisionInputs, HistoryPolicyConfig, PublicStream, StreamBatch,
                                 fresh_player)
from train.history_ppo import (HistoryPPOConfig, HistoryTrainer, build_parser, config_from_args,
                               parse_arm_schedule, parse_resume_overrides)
from train.history_snapshot_batch import (LAYOUT_FIELDS, SnapshotHeads, _head_values,
                                          merged_layout, merged_log_probs, merged_sample)
from train.logs import TOKEN_DIM
from test_history_rollout import make_env

# Merged vs per-identity log-probabilities, FP32 (observed ~1e-6; see report).
ATOL = 2e-5
OBS_DIM, ACT_DIM = 1849, 154
TIMING_KEYS = {"collection_phase_seconds", "collection_profile_synchronized",
               "collection_group_phase_seconds", "collection_policy_call_rows",
               "collection_policy_batches", "decisions_per_sec", "collect_seconds",
               "learn_seconds", "learn_decisions_per_sec", "learn_exposures_per_sec",
               "learner_collect_decisions_per_sec", "elapsed_seconds",
               "rollout_batch_snapshot_policies", "snapshot_heads"}


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA merged snapshot inference")
    return request.param


@pytest.fixture(autouse=True)
def strict_fp32():
    previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_num_threads(previous[0])
        torch.use_deterministic_algorithms(previous[1], warn_only=previous[2])
        torch.backends.cuda.matmul.allow_tf32 = previous[3]
        torch.backends.cudnn.allow_tf32 = previous[4]


def config(response_mode="auxiliary", width=32, heads=4):
    return HistoryPolicyConfig(width=width, layers=1, heads=heads, action_width=16,
                               fusion_width=24, critic_width=16, response_mode=response_mode)


def players(identities, device, response_mode="auxiliary", seed=40):
    return {i: fresh_player(config(response_mode), seed + i)[0].to(device) for i in identities}


# ---- synthetic vector steps ------------------------------------------------------

def token(rng):
    value = np.zeros(TOKEN_DIM, np.uint8)
    value[rng.integers(4)] = 1
    value[4 + rng.choice(146, size=rng.integers(0, 6), replace=False)] = 1
    value[158 + rng.integers(TOKEN_DIM - 158)] = 1
    return value


class Step:
    """A synthetic pending batch: interleaved identities, ragged candidates, and
    either one decision per stream or several decisions sharing streams."""

    def __init__(self, spec, seed, max_candidates=9, longest=40, shorter_prefix=False):
        rng = np.random.default_rng(seed)
        identities = np.concatenate([[i] * rows for i, rows, _ in spec])
        order = rng.permutation(len(identities))
        self.identities = identities[order]
        n = len(identities)
        self.env_id = np.zeros(n, np.int64)
        self.match_id = np.zeros(n, np.int64)
        self.streams = {}
        env = 0
        for identity, rows, streams in spec:
            where = np.flatnonzero(self.identities == identity)
            owners = np.concatenate([np.arange(streams), rng.integers(streams, size=rows - streams)])
            for row, owner in zip(where, owners):
                self.env_id[row] = env + owner
                self.match_id[row] = 3
            for owner in range(streams):
                stream = PublicStream(3)
                for _ in range(int(rng.integers(0, longest))):
                    stream.append_token(token(rng), int(rng.integers(0, 3)), int(rng.integers(0, 4)))
                self.streams[(env + owner, 3)] = stream
            env += streams
        self.counts = rng.integers(1, max_candidates + 1, size=n)
        self.counts[0] = 1
        self.offsets = np.concatenate(([0], np.cumsum(self.counts))).astype(np.int64)
        self.obs = rng.integers(0, 2, size=(n, OBS_DIM), dtype=np.uint8)
        self.cand = rng.integers(0, 2, size=(int(self.offsets[-1]), ACT_DIM), dtype=np.uint8)
        self.seat = rng.integers(0, 4, size=n).astype(np.int64)
        self.prefix = np.asarray([self.streams[(e, m)].prefix
                                  for e, m in zip(self.env_id, self.match_id)], np.int64)
        if shorter_prefix:
            self.prefix = (self.prefix * rng.random(n)).astype(np.int64)

    def groups(self):
        result = []
        for identity in np.unique(self.identities):
            rows = np.flatnonzero(self.identities == identity)
            keys = list(zip(self.env_id[rows].tolist(), self.match_id[rows].tolist()))
            unique = list(dict.fromkeys(keys))
            result.append(rollout._PolicyBatch(int(identity), rows, unique,
                                               [self.streams[k] for k in unique],
                                               len(keys) == len(unique),
                                               int(self.counts[rows].max())))
        return result

    def inputs(self, group, device):
        rows = group.rows
        index = {key: i for i, key in enumerate(group.keys)}
        keys = zip(self.env_id[rows].tolist(), self.match_id[rows].tolist())
        src = rollout.ragged_index(self.offsets[rows], self.counts[rows])
        return DecisionInputs(
            streams=StreamBatch.from_streams(group.streams, device),
            match_index=torch.as_tensor([index[k] for k in keys], device=device),
            prefix=torch.as_tensor(self.prefix[rows], device=device),
            obs=torch.as_tensor(self.obs[rows], device=device),
            seat=torch.as_tensor(self.seat[rows], device=device),
            cand=torch.as_tensor(self.cand[src], device=device),
            offsets=torch.as_tensor(np.concatenate(([0], np.cumsum(self.counts[rows]))),
                                    device=device),
            one_decision_per_stream=group.one_decision_per_stream)


def merged(step, policies, device, heads=None):
    """The collector's merged path on a synthetic step (dense public encoding)."""
    groups = step.groups()
    heads = heads or SnapshotHeads()
    slots = [heads.slot(g.identity, policies[g.identity]) for g in groups]
    layout = merged_layout(groups, slots, step.prefix, step.obs, step.seat, step.cand,
                           step.offsets, step.counts, step.env_id, step.match_id)
    fields = {name: torch.as_tensor(array, device=device)
              for name, array in zip(LAYOUT_FIELDS, layout.arrays)}
    width = next(iter(policies.values())).config.width
    memory = torch.zeros(layout.streams, layout.length, width, device=device)
    with torch.no_grad():
        for group, (first, last) in zip(groups, layout.group_streams):
            encoded = policies[group.identity].encode_batch(
                StreamBatch.from_streams(group.streams, device))
            memory[first:last, :encoded.shape[1]] = encoded
        log_probs = merged_log_probs(heads, layout, fields, memory)
    return groups, layout, fields, log_probs


def reference(step, policies, groups, device):
    """Each identity's own ``candidate_log_probs``, in merged candidate order."""
    with torch.no_grad():
        return torch.cat([policies[g.identity].candidate_log_probs(step.inputs(g, device))
                          for g in groups])


SPECS = {
    "one-per-stream": [(1, 5, 5), (2, 1, 1), (3, 9, 9), (4, 2, 2)],
    "single-identity": [(3, 7, 7)],
    "single-row-identity": [(1, 1, 1)],
    "multi-decision": [(1, 6, 3), (2, 1, 1), (3, 8, 2)],
    "mixed": [(1, 4, 4), (2, 5, 2), (5, 3, 3)],
}
MAX_DIFFERENCE = {}


@pytest.mark.parametrize("response_mode", ["none", "auxiliary"])
@pytest.mark.parametrize("name", list(SPECS))
def test_merged_log_probs_match_each_identity(device, name, response_mode):
    step = Step(SPECS[name], seed=list(SPECS).index(name), shorter_prefix=name == "multi-decision")
    policies = players({i for i, _, _ in SPECS[name]}, device, response_mode)
    groups, layout, fields, log_probs = merged(step, policies, device)
    expected = reference(step, policies, groups, device)
    assert log_probs.shape == expected.shape and log_probs.dtype == torch.float32
    difference = float((log_probs - expected).abs().max())
    MAX_DIFFERENCE[(device, name, response_mode)] = difference
    print(f"max |merged - per-identity| log-prob {device} {name} {response_mode}: {difference:.3e}")
    assert difference < ATOL
    assert layout.identity_streams == all(g.one_decision_per_stream for g in groups)


def test_production_size_step_matches_each_identity(device):
    # The production architecture (width 128, 4 layers, 8 heads, auxiliary) with
    # 12 identities, 1..31 rows each and public prefixes up to ~700 tokens.
    rng = np.random.default_rng(8)
    spec = [(i, int(rows), int(rows)) for i, rows in
            zip(range(1, 13), [31, 1, 16, 8, 2, 12, 5, 1, 20, 9, 3, 7])]
    step = Step(spec, seed=8, max_candidates=int(rng.integers(150, 300)), longest=700)
    production = HistoryPolicyConfig(width=128, layers=4, heads=8, response_mode="auxiliary")
    policies = {i: fresh_player(production, 60 + i)[0].to(device) for i, _, _ in spec}
    groups, layout, fields, log_probs = merged(step, policies, device)
    difference = float((log_probs - reference(step, policies, groups, device)).abs().max())
    print(f"max |merged - per-identity| log-prob {device} production-size: {difference:.3e}")
    assert difference < ATOL


def test_merged_path_reads_each_identitys_own_weights(device):
    # Same inputs, the weights of one identity perturbed: only its rows move.
    step = Step(SPECS["one-per-stream"], seed=3)
    policies = players((1, 2, 3, 4), device)
    groups, layout, _, before = merged(step, policies, device)
    with torch.no_grad():
        policies[3].fusion[1].bias.add_(0.5)
        policies[3].feed_forward[0].weight.mul_(1.5)
    _, _, _, after = merged(step, policies, device)
    moved = np.zeros(len(before), bool)
    start = 0
    for group in groups:
        size = int(step.counts[group.rows].sum())
        moved[start:start + size] = group.identity == 3
        start += size
    delta = (after - before).abs().cpu().numpy()
    assert delta[~moved].max() == 0
    assert delta[moved].max() > 1e-4
    assert float((after - reference(step, policies, groups, device)).abs().max()) < ATOL


@pytest.mark.parametrize("name", ["one-per-stream", "multi-decision", "single-row-identity"])
def test_sampling_matches_per_identity_act_and_generator(device, name):
    step = Step(SPECS[name], seed=11, shorter_prefix=name == "multi-decision")
    policies = players({i for i, _, _ in SPECS[name]}, device)
    groups, layout, fields, log_probs = merged(step, policies, device)
    for seed in range(5):
        generators = [torch.Generator(device=device).manual_seed(seed) for _ in range(2)]
        choice, logp = merged_sample(log_probs, layout, fields, generators[0])
        expected = []
        with torch.no_grad():
            for group in groups:
                expected.append(policies[group.identity].act(
                    step.inputs(group, device), generators[1],
                    max_candidates=group.max_candidates)[0])
        # Same uniforms, same order: the generator advanced exactly as before.
        assert torch.equal(generators[0].get_state(), generators[1].get_state())
        # Choices agree unless float noise flips a near tie (none on these seeds).
        assert torch.equal(choice, torch.cat(expected))
        counts = torch.as_tensor(step.counts[layout.rows], device=device)
        assert bool(((choice >= 0) & (choice < counts)).all())
        starts = torch.as_tensor(np.concatenate(([0], np.cumsum(step.counts[layout.rows])))[:-1],
                                 device=device)
        assert torch.equal(logp, log_probs[starts + choice])


def test_sampled_choices_follow_each_identitys_distribution(device):
    step = Step([(1, 3, 3), (2, 1, 1), (4, 2, 2)], seed=5, max_candidates=6)
    policies = players((1, 2, 4), device)
    for actor in policies.values():     # flatter policies exercise every candidate
        with torch.no_grad():
            actor.fusion[1].weight.mul_(0.2)
    groups, layout, fields, log_probs = merged(step, policies, device)
    probabilities = reference(step, policies, groups, device).exp().cpu().numpy()
    generator = torch.Generator(device=device).manual_seed(123)
    draws = 4000
    counts = step.counts[layout.rows]
    starts = np.concatenate(([0], np.cumsum(counts)))
    hits = [np.zeros(c) for c in counts]
    for _ in range(draws):
        choice = merged_sample(log_probs, layout, fields, generator)[0].cpu().numpy()
        for row, c in enumerate(choice):
            hits[row][c] += 1
    for row, count in enumerate(counts):
        expected = probabilities[starts[row]:starts[row + 1]]
        observed = hits[row] / draws
        assert 0.5 * np.abs(observed - expected).sum() < 0.04
        if count > 1:
            chi2 = float((((hits[row] - draws * expected) ** 2) / (draws * expected)).sum())
            df = count - 1
            assert chi2 < df + 8 * math.sqrt(2 * df) + 10    # far tail, deterministic seed


def test_head_slots_refresh_on_every_resident_set_change(device):
    policies = players((1, 2, 3), device)
    heads = SnapshotHeads(capacity=1)

    def assert_slot(identity, actor):
        index = heads.slot(identity, actor)
        for name, value in _head_values(actor).items():
            assert torch.equal(heads.tensors[name][index], value), name
        return index

    first = assert_slot(1, policies[1])
    second = assert_slot(2, policies[2])        # added: capacity grows, slot 1 intact
    assert (first, second) == (0, 1) and heads.capacity >= 2
    assert_slot(1, policies[1])
    writes = heads.writes
    assert heads.slot(1, policies[1]) == first and heads.writes == writes   # unchanged: no copy
    with torch.no_grad():                       # in-place update (version bump)
        policies[1].q_proj.weight.add_(1.0)
    assert assert_slot(1, policies[1]) == first and heads.writes == writes + 1
    policies[1].load_state_dict(policies[3].state_dict())
    assert_slot(1, policies[1])
    replacement = fresh_player(config(), 99)[0].to(device)  # same identity, new actor object
    assert assert_slot(2, replacement) == second
    heads.retain({2})                           # identity 1 pruned: its slot is reused
    assert set(heads.slots) == {2}
    assert assert_slot(3, policies[3]) == first
    writes = heads.writes
    with torch.no_grad():                       # rebound storage, same version counter
        policies[3].seat.weight.data = policies[3].seat.weight.data.clone() + 1
    assert assert_slot(3, policies[3]) == first and heads.writes == writes + 1
    assert heads.bytes == sum(t.numel() * 4 for t in heads.tensors.values())


def test_refuses_unsupported_actors(device):
    explicit = fresh_player(config("explicit"), 1)[0].to(device)
    with pytest.raises(ValueError, match="explicit"):
        SnapshotHeads().slot(1, explicit)
    windowed = fresh_player(HistoryPolicyConfig(width=32, layers=1, window=8, action_width=16,
                                                fusion_width=24), 1)[0]
    with pytest.raises(ValueError, match="full-history"):
        rollout.HistoryCollector(make_env(2, 1), windowed, rollout.MatchEventStore(),
                                 rollout.SequenceRolloutBuffer(), batch_snapshot_policies=True)
    hooked = fresh_player(config(), 2)[0].to(device)
    hooked.q_proj.register_forward_hook(lambda *args: None)
    with pytest.raises(ValueError, match="hooks"):
        SnapshotHeads().slot(1, hooked)
    heads = SnapshotHeads()
    heads.slot(1, fresh_player(config(), 3)[0].to(device))
    with pytest.raises(ValueError, match="one architecture"):
        heads.slot(2, fresh_player(config(width=64, heads=4), 3)[0].to(device))
    with pytest.raises(ValueError, match="explicit"):
        HistoryPPOConfig(batch_snapshot_policies=True, response_mode="explicit",
                         response_coef=0.1)


# ---- collector ---------------------------------------------------------------------

def run_collector(device, merge, *, kv_cache, steps=36, policies=None, seat_policy=None,
                  between=None, **kwargs):
    policies = policies or players((0, 1, 2, 3), device)
    if merge is not None:
        kwargs["batch_snapshot_policies"] = merge
    collector = rollout.HistoryCollector(
        make_env(6, 23), policies[0], rollout.MatchEventStore(), rollout.SequenceRolloutBuffer(),
        torch.Generator(device=device).manual_seed(31), device=device,
        seat_policy=seat_policy or (lambda env, match: [0, 1 + env % 3, 2, 1 + (env + match) % 3]),
        resolve_policy=policies.__getitem__, record_choices=True, kv_cache=kv_cache, **kwargs)
    stats = []
    for chunk in range(3):
        stats.append(collector.collect(steps // 3))
        if between is not None:
            between(chunk, policies, collector)
    return collector, stats


def assert_same_collection(plain, merged_run):
    assert len(plain.choice_log) == len(merged_run.choice_log)
    for before, after in zip(plain.choice_log, merged_run.choice_log):
        assert before.tobytes() == after.tobytes()
    for name, before in plain.buffer.compact().items():   # learner rows, bit for bit
        after = merged_run.buffer.compact()[name]
        assert before.dtype == after.dtype and before.tobytes() == after.tobytes()
    assert torch.equal(plain.generator.get_state(), merged_run.generator.get_state())
    assert plain.policy_decisions == merged_run.policy_decisions


@pytest.mark.parametrize("kv_cache", [False, True])
def test_collector_merged_matches_per_identity_rollout(device, kv_cache):
    (plain, plain_stats), (fast, fast_stats) = (
        run_collector(device, merge, kv_cache=kv_cache, profile=True) for merge in (False, True))
    # Tier 2 allows different snapshot choices; on these seeds the float noise
    # flips none, so the whole rollout (learner rows, sampler state) is identical.
    assert_same_collection(plain, fast)
    for before, after in zip(plain_stats, fast_stats):
        same = {k: v for k, v in asdict(before).items()
                if k not in rollout.PROFILE_STATS and k != "policy_batches"}
        assert same == {k: v for k, v in asdict(after).items()
                        if k not in rollout.PROFILE_STATS and k != "policy_batches"}
        # One merged snapshot call per step that has snapshot rows.
        assert after.policy_call_rows["learner"] == before.policy_call_rows["learner"]
        merged_calls = after.policy_call_rows["snapshot"]
        assert sum(merged_calls.values()) <= after.steps < sum(
            before.policy_call_rows["snapshot"].values())
        assert (sum(size * n for size, n in merged_calls.items())
                == sum(size * n for size, n in before.policy_call_rows["snapshot"].items()))
        assert set(after.group_phase_seconds["snapshot"]) == {"public_cache_or_collation",
                                                             "actor_and_sampling"}
    assert fast.snapshot_heads is not None and plain.snapshot_heads is None


def test_collector_refreshes_heads_when_snapshots_change(device):
    # Matches pin identities; new matches bring new identities (added), old ones
    # leave (pruned), identity 2 is replaced by a new actor object and identity 1
    # is updated in place between collects. A stale stacked copy would change
    # the snapshot choices (and so the whole rollout) against the plain run.
    def seats(env, match):
        base = 1 + (env + 2 * match) % 5
        return [0, base, 1 + (base % 5), 1 + ((base + 2) % 5)]

    def between(chunk, policies, collector):
        with torch.no_grad():
            if chunk == 0:
                policies[2] = fresh_player(config(), 77)[0].to(device)
            if chunk == 1:
                policies[1].fusion[1].bias.add_(0.3)

    runs = [run_collector(device, merge, kv_cache=True, steps=60, seat_policy=seats,
                          between=between, policies=players(range(6), device))[0]
            for merge in (False, True)]
    assert_same_collection(*runs)
    heads = runs[1].snapshot_heads
    active = {int(i) for seats_ in runs[1].assignments.values() for i in seats_} - {0}
    assert set(heads.slots) <= active and heads.writes > len(heads.slots)


def test_flag_off_is_the_unchanged_collector(device):
    # Default and explicit False take the identical path (no stacked heads);
    # the pre-existing bitwise suites pin that path to its legacy references.
    runs = [run_collector(device, merge, kv_cache=True)[0] for merge in (None, False)]
    assert runs[0].batch_snapshot_policies is False
    assert_same_collection(*runs)
    assert all(r.snapshot_heads is None for r in runs)


def test_switching_between_collects_keeps_one_consistent_stream(device):
    def between(chunk, policies, collector):
        collector.batch_snapshot_policies = chunk == 0      # on, off, on
    reference_run = run_collector(device, False, kv_cache=True)[0]
    switched = run_collector(device, True, kv_cache=True, between=between)[0]
    assert_same_collection(reference_run, switched)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA private graphs and Triton cache")
def test_production_cuda_options_with_merged_snapshots():
    # Private graphs stay on the learner; snapshot identities bypass them.
    pytest.importorskip("triton")
    options = dict(kv_cache=True, private_graphs=True, triton_cache=True, triton_min_batch=4)
    policies = players((0, 1, 2, 3), "cuda")
    for actor in policies.values():
        actor.causal_sdpa = True
        actor.batched_private_attention = True
        actor.wide_private_projection = True
    plain = run_collector("cuda", False, policies=policies, **options)[0]
    fast = run_collector("cuda", True, policies=policies, **options)[0]
    assert set(fast.decision_graphs) <= {0}
    assert_same_collection(plain, fast)


# ---- trainer -------------------------------------------------------------------------

def test_cli_resume_and_schedule(tmp_path):
    args = build_parser().parse_args(["--output", str(tmp_path), "--batch-snapshot-policies"])
    assert config_from_args(args).batch_snapshot_policies
    assert not config_from_args(build_parser().parse_args(["--output", "x"])).batch_snapshot_policies
    assert parse_arm_schedule("2:off, 1:on") == [(2, False), (1, True)]
    for bad in ("2", "0:on", "1:maybe", "x:on"):
        with pytest.raises(ValueError):
            parse_arm_schedule(bad)
    base = dict(width=16, layers=1, heads=4, num_envs=2, steps_per_update=40, epochs=1,
                minibatch_matches=1, snapshot_updates=1, seed=5, causal_sdpa=True,
                rollout_kv_cache=True)
    trainer = HistoryTrainer(HistoryPPOConfig(**base), tmp_path / "start")
    trainer.update()
    checkpoint = trainer.save()
    overrides = parse_resume_overrides(["batch_snapshot_policies=true",
                                        "batch_snapshot_policies_schedule=1:off,2:on"])
    resumed = HistoryTrainer(HistoryPPOConfig(updates=4), tmp_path / "resumed", resume=checkpoint,
                             resume_overrides=overrides)
    assert resumed.config_changes[-1] == dict(
        at_update=1, changes={"batch_snapshot_policies": [False, True],
                              "batch_snapshot_policies_schedule": ["", "1:off,2:on"]})
    arms = [resumed.update()["rollout_batch_snapshot_policies"] for _ in range(3)]
    assert arms == [False, True, True]
    manifest = (tmp_path / "resumed" / "manifest.json").read_text()
    assert '"batch_snapshot_policies": true' in manifest


def test_trainer_with_merged_snapshots_trains_identically(tmp_path):
    # A resume restarts the environments, so every new match draws snapshot
    # seats at once. Tier 2 in general; on this seed no snapshot choice flips,
    # so metrics, weights and the sampler state match the per-identity trainer.
    base = dict(width=16, layers=1, heads=4, num_envs=4, steps_per_update=30, epochs=1,
                minibatch_matches=1, snapshot_updates=1, seed=5, causal_sdpa=True,
                rollout_kv_cache=True, snapshot_probability=1.0)
    start = HistoryTrainer(HistoryPPOConfig(**base), tmp_path / "start")
    for _ in range(3):
        start.update()
    checkpoint = start.save()
    trainers, lines = {}, {}
    for name, merge in (("plain", "false"), ("merged", "true")):
        trainer = HistoryTrainer(HistoryPPOConfig(updates=6), tmp_path / name, resume=checkpoint,
                                 resume_overrides=parse_resume_overrides(
                                     [f"batch_snapshot_policies={merge}"]))
        lines[name] = [trainer.update() for _ in range(3)]
        trainers[name] = trainer
    assert all(line["snapshot_heads"].get("slots") for line in lines["merged"])
    assert not any(line["snapshot_heads"] for line in lines["plain"])
    for before, after in zip(lines["plain"], lines["merged"]):
        assert after["collection_policy_batches"] <= before["collection_policy_batches"]
        assert ({k: v for k, v in before.items() if k not in TIMING_KEYS}
                == {k: v for k, v in after.items() if k not in TIMING_KEYS})
    assert (sum(line["collection_policy_batches"] for line in lines["merged"])
            < sum(line["collection_policy_batches"] for line in lines["plain"]))
    for a, b in zip(trainers["plain"].actor.parameters(), trainers["merged"].actor.parameters()):
        assert torch.equal(a, b)
    assert torch.equal(trainers["plain"].generator.get_state(),
                       trainers["merged"].generator.get_state())
