"""STAGE_C T2 acceptance: event store, prefixes, privacy and reward attribution."""
from collections import Counter

import gd
import numpy as np
import pytest
import torch

from train.history_model import HistoryPolicyConfig, PublicStream, fresh_player
from train.history_rollout import (HistoryCollector, MatchEventStore, SequenceRolloutBuffer,
                                   compute_gae)
from train.logs import public_token

PLAY = int(gd.Phase.Play)
NUM_ENVS = 4
SEED = 11
STEPS = 160          # a random actor finishes a round in roughly 60-80 vector steps
CONFIG = HistoryPolicyConfig(width=32, layers=1, heads=4)


def make_env(num_envs: int = NUM_ENVS, seed: int = SEED) -> gd.VecEnv:
    return gd.VecEnv(num_envs=num_envs, num_threads=1, seed=seed, log_public_actions=True,
                     log_env_limit=num_envs)


@pytest.fixture(scope="module")
def collected():
    actor, _ = fresh_player(CONFIG, 5)
    store, buffer = MatchEventStore(), SequenceRolloutBuffer()
    generator = torch.Generator().manual_seed(7)
    collector = HistoryCollector(make_env(), actor, store, buffer, generator,
                                 record_choices=True)
    stats = collector.collect(STEPS)
    assert stats.rounds >= NUM_ENVS, "every environment should finish at least one round"
    return actor, store, buffer, collector, stats


def test_prefix_matches_independent_recount(collected):
    _, store, buffer, collector, _ = collected
    data = buffer.compact()
    env = make_env()
    env.reset()
    counts: Counter = Counter()
    tokens: dict[tuple[int, int], list[np.ndarray]] = {}
    row = 0
    for choices in collector.choice_log:
        batch = env.pending()
        for event in env.drain_public_actions():
            key = (int(event.env_id), int(event.match_id))
            counts[key] += 1
            tokens.setdefault(key, []).append(public_token(event))
        env.drain_finished_rounds()
        env_id = np.asarray(batch.env_id)
        match_id = np.asarray(batch.match_id)
        phase = np.asarray(batch.phase)
        for i in np.flatnonzero(phase == PLAY):
            key = (int(env_id[i]), int(match_id[i]))
            assert int(data["env"][row]) == key[0] and int(data["match"][row]) == key[1]
            assert int(data["prefix"][row]) == counts[key]
            row += 1
        env.step(choices)
    assert row == len(buffer)
    for key, stream in store.streams.items():
        expected = np.stack(tokens[key]) if key in tokens else np.zeros((0, 186), np.uint8)
        assert stream.prefix == counts[key]
        assert np.array_equal(stream.arrays()[0], expected)
    phases = {p for s in store.streams.values() for p in s.phases}
    assert {int(gd.Phase.Tribute), int(gd.Phase.BackTribute)} <= phases, \
        "tribute-phase exchange events must be in the stream"
    seats = {int(np.flatnonzero(t[:4])[0]) for s in store.streams.values() for t in s.tokens}
    assert seats == {0, 1, 2, 3}, "opponent and partner events must be in the stream"


class _NoForced:
    """Event view whose ``forced`` flag blows up when read."""

    def __init__(self, event, flip: bool) -> None:
        self._event = event
        self._flip = flip

    def __getattr__(self, name: str):
        if name == "forced":
            if self._flip:
                return not self._event.forced
            raise AssertionError("the public store must not read the forced flag")
        return getattr(self._event, name)


def test_store_never_reads_forced_flag():
    env = make_env(2, 3)
    env.reset()
    events = []
    for _ in range(40):
        batch = env.pending()
        events.extend(env.drain_public_actions())
        env.step(np.array(batch.greedy_choice, dtype=np.int32))
    events.extend(env.drain_public_actions())
    assert any(e.forced for e in events), "the sample should contain an engine-forced pass"
    plain, flipped, blind = MatchEventStore(), MatchEventStore(), MatchEventStore()
    plain.ingest(events)
    flipped.ingest([_NoForced(e, True) for e in events])
    blind.ingest([_NoForced(e, False) for e in events])
    for key, stream in plain.streams.items():
        for other in (flipped, blind):
            assert np.array_equal(stream.arrays()[0], other.streams[key].arrays()[0])
            assert np.array_equal(stream.rounds, other.streams[key].rounds)
            assert np.array_equal(stream.phases, other.streams[key].phases)
    for stream in plain.streams.values():
        assert not np.stack(stream.tokens)[:, 4 + 146:4 + 154].any()


def _log_probs(actor, buffer, store, rows, streams=None, device="cpu"):
    inputs, chosen = buffer.decision_inputs(rows, store, device, streams)
    with torch.no_grad():
        return actor.candidate_log_probs(inputs)[chosen].numpy()


def test_behaviour_log_probs_match_recompute(collected):
    actor, store, buffer, _, _ = collected
    rows = np.arange(len(buffer))
    stored = buffer.compact()["logp"]
    recomputed = _log_probs(actor, buffer, store, rows)
    assert np.abs(recomputed - stored).max() < 1e-5
    counts = buffer.compact()["cand_count"]
    assert (stored[counts == 1] == 0).all()
    assert (stored[counts > 1] < 0).all()


def test_prefix_truncation_and_other_rows_do_not_change_log_probs(collected):
    actor, store, buffer, _, _ = collected
    data = buffer.compact()
    stored = data["logp"]
    rng = np.random.default_rng(0)
    for row in rng.choice(len(buffer), 12, replace=False).tolist():
        key = (int(data["env"][row]), int(data["match"][row]))
        full = store.stream(*key)
        prefix = int(data["prefix"][row])
        assert prefix < full.prefix, "the match gained tokens after this decision"
        truncated = PublicStream(key[1])
        for token, rnd, phase in zip(full.tokens[:prefix], full.rounds[:prefix],
                                     full.phases[:prefix]):
            truncated.append_token(token, rnd, phase)
        alone = _log_probs(actor, buffer, store, np.array([row]), {key: truncated})
        assert abs(float(alone[0]) - stored[row]) < 1e-5
    # Other rows' private observations in the same minibatch change nothing.
    group = np.flatnonzero((data["env"] == data["env"][0]) & (data["match"] == data["match"][0]))
    target = group[len(group) // 2]
    baseline = _log_probs(actor, buffer, store, group)
    scrambled = dict(data)
    obs = data["obs"].copy()
    others = group[group != target]
    obs[others] = rng.integers(0, 2, size=obs[others].shape, dtype=np.uint8)
    scrambled["obs"] = obs
    buffer.data = scrambled
    try:
        perturbed = _log_probs(actor, buffer, store, group)
    finally:
        buffer.data = data
    position = int(np.flatnonzero(group == target)[0])
    assert abs(float(perturbed[position]) - float(baseline[position])) < 1e-5
    assert np.abs(perturbed - baseline).max() > 1e-3, "scrambled rows should move"


def test_rewards_land_on_terminal_rows_only(collected):
    _, _, buffer, collector, stats = collected
    data = buffer.compact()
    samples = buffer.finalize(np.zeros(len(buffer), np.float32))
    assert samples > 0
    terminal = np.flatnonzero(buffer.done)
    assert (buffer.reward[~buffer.done] == 0).all()
    expected = {}
    for result in collector.results:
        for team in (0, 1):
            expected[(int(result.env_id), int(result.match_id), int(result.round_index), team)] \
                = float(result.seat_return[team])
    seen = set()
    for row in terminal.tolist():
        key = (int(data["env"][row]), int(data["match"][row]), int(data["round"][row]),
               int(data["seat"][row]) % 2)
        assert buffer.reward[row] == expected[key]
        assert key not in seen
        seen.add(key)
        # The terminal row is the team's last stored row of that round.
        same = ((data["env"] == key[0]) & (data["match"] == key[1]) & (data["round"] == key[2])
                & (data["seat"] % 2 == key[3]))
        assert row == int(np.flatnonzero(same).max())
    assert seen == set(expected), "every finished round closes both teams' trajectories"
    assert len(seen) == 2 * stats.rounds
    open_rows = np.setdiff1d(np.arange(len(buffer)), buffer.samples)
    assert not buffer.done[open_rows].any()
    assert (buffer.returns[buffer.samples] == buffer.advantage[buffer.samples]).all()


def test_gae_matches_manual_recursion():
    values = np.array([[0.5, -0.2, 0.1, 0.3]], np.float32)
    rewards = np.array([[0.0, 0.0, 0.0, 2.0]], np.float32)
    dones = np.array([[False, False, False, True]])
    adv, ret = compute_gae(values, rewards, dones, 1.0, 0.95)
    expected = np.zeros(4)
    last = 0.0
    for t in (3, 2, 1, 0):
        next_value = 0.0 if t == 3 else values[0, t + 1]
        delta = rewards[0, t] + next_value * (t != 3) - values[0, t]
        last = delta + 0.95 * (t != 3) * last
        expected[t] = last
    assert np.allclose(adv[0], expected, atol=1e-6)
    assert np.allclose(ret[0], expected + values[0], atol=1e-6)
    assert ret[0, 3] == pytest.approx(2.0)


def test_carry_over_and_store_pruning():
    actor, _ = fresh_player(CONFIG, 1)
    store, buffer = MatchEventStore(), SequenceRolloutBuffer()
    collector = HistoryCollector(make_env(2, 5), actor, store, buffer,
                                 torch.Generator().manual_seed(1))
    collector.collect(100)
    buffer.finalize(np.zeros(len(buffer), np.float32))
    open_rows = np.setdiff1d(np.arange(len(buffer)), buffer.samples)
    before = buffer.compact()
    kept = {tuple(before[k][r] for k in ("env", "match", "round", "seat", "prefix", "chosen"))
            for r in open_rows.tolist()}
    cited = buffer.next_iteration()
    after = buffer.compact()
    assert len(buffer) == open_rows.size
    assert {tuple(after[k][r] for k in ("env", "match", "round", "seat", "prefix", "chosen"))
            for r in range(len(buffer))} == kept
    assert cited == set(zip(after["env"].tolist(), after["match"].tolist()))
    assert store.prune(cited) == 0, "current matches are never pruned"
    # A finished match whose rows are gone is dropped; one still cited stays.
    stale = store.stream(0, 0)
    store.current[0] = 1
    store.stream(0, 1)
    assert store.prune(cited | {(0, 0)}) == 0
    assert store.prune(cited - {(0, 0)}) == 1 and (0, 0) not in store.streams
    assert stale.prefix > 0
    with pytest.raises(ValueError):
        store.stream(0, 0)


def test_non_learner_seats_are_refused():
    actor, _ = fresh_player(CONFIG, 1)
    collector = HistoryCollector(make_env(1, 2), actor, MatchEventStore(),
                                 SequenceRolloutBuffer(), seat_policy=lambda e, m: (0, 1, 0, 1))
    with pytest.raises(NotImplementedError):
        collector.step()
