"""T3: history-aware evaluators deliver every public action and reset correctly.

The heart is the parity test: for one deal and one deterministic action
sequence, the stream a scalar round loop builds from ``Engine.apply`` equals,
token for token, the stream ``VecEnv(log_public_actions=True)`` records.
"""
import os
import random
import sys

import gd
import numpy as np
import pytest
import torch

from eval.arena import play_matches
from eval.batched import (BatchActor, EvalConfig, play_duplicate_batch, play_match_slots_batch,
                          play_matches_batch)
from eval.duplicate import generate_deals, play_duplicate, play_duplicate_teams, play_round_seats
from eval.history_policy import (HistoryPolicy, HistoryStreamStore, apply_and_observe,
                                 history_listeners, resolve_forced_passes)
from eval.policies import GreedyPolicy, load_policy
from eval.tribute import generate_tribute_deals
from train.ckpt import save_checkpoint
from train.history_model import (HistoryPolicyConfig, PublicStream, checkpoint_payload,
                                 fresh_player, save_history_checkpoint)
from train.logs import public_token

TINY = HistoryPolicyConfig(width=16, layers=1, heads=2, action_width=16, fusion_width=16,
                           critic_width=16, critic_layers=1)
PLAY, TRIBUTE, BACK = int(gd.Phase.Play), int(gd.Phase.Tribute), int(gd.Phase.BackTribute)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.set_num_threads(1)
    actor, critic = fresh_player(TINY, seed=3)
    path = tmp_path_factory.mktemp("history") / "tiny.pt"
    save_history_checkpoint(path, checkpoint_payload(actor, critic, lineage="test", seed=3))
    return path


def history_policy(checkpoint, sample=False) -> HistoryPolicy:
    policy = load_policy(f"sample:{checkpoint}" if sample else str(checkpoint))
    assert isinstance(policy, HistoryPolicy) and policy.needs_history
    return policy


class RecordingGreedy:
    """A greedy seat that records the public stream like a history policy would."""
    needs_history = True
    name = "recording-greedy"

    def __init__(self):
        self.stream = PublicStream()
        self.events = []
        self.starts = 0

    def start_match(self):
        self.stream.reset()
        self.events.clear()
        self.starts += 1

    def observe(self, event):
        self.stream.append(event)
        self.events.append(event)

    def select(self, engine, state, actions, rng):
        return engine.greedy(state)


def vecenv_stream(deal=None, match_seed=None):
    """Greedy play of one round (deal) or one whole match (seed) with logging on."""
    env = gd.VecEnv(1, 1, 0, log_public_actions=True)
    if deal is not None:
        env.reset([deal])
    else:
        env.reset(match_seeds=[match_seed])
    events, finished, played = [], False, None
    while not finished:
        batch = env.pending()
        events.extend(env.drain_public_actions())
        for result in env.drain_finished_rounds():
            if result.match_id == 0 and played is None:
                played = result.round_index   # a deal with a previous order is round 1
            finished = result.match_id == 0 and (deal is not None or result.match_winner >= 0)
        if finished:
            break
        env.step(batch.greedy_choice)
    return [e for e in events if e.match_id == 0 and (deal is None or e.round_index == played)]


def scalar_stream(deal=None, match_seed=None, max_rounds=1000):
    recorder = RecordingGreedy()
    seats = (recorder, GreedyPolicy(), recorder, GreedyPolicy())
    engine, state = gd.Engine(), gd.MatchState()
    recorder.start_match()
    if deal is not None:
        engine.set_deal(state, deal)
        score = play_round_seats(engine, state, seats, seed=0)
        assert engine.auto_pass is True
        return recorder, [score]
    engine.new_match(state, match_seed)
    scores = []
    while state.winner < 0:
        assert len(scores) < max_rounds
        scores.append(play_round_seats(engine, state, seats, seed=0))
        if state.winner < 0:
            engine.begin_round(state)
    assert engine.auto_pass is True
    return recorder, scores


def assert_same_stream(recorder, events):
    tokens, rounds, phases = recorder.stream.arrays()
    assert len(events) == len(tokens) > 0
    np.testing.assert_array_equal(tokens, np.stack([public_token(e) for e in events]))
    np.testing.assert_array_equal(rounds, [int(e.round_index) for e in events])
    np.testing.assert_array_equal(phases, [int(e.phase) for e in events])
    assert [e.seat for e in recorder.events] == [int(e.seat) for e in events]
    assert [e.cards_left for e in recorder.events] == [int(e.cards_left) for e in events]
    assert [e.forced for e in recorder.events] == [bool(e.forced) for e in events]


# ---- scalar-vs-VecEnv parity ---------------------------------------------------

def test_scalar_round_stream_matches_vecenv_token_for_token():
    deals = generate_deals(4, 33) + generate_tribute_deals(6, 33)
    forced = tribute = 0
    for deal in deals:
        recorder, (score,) = scalar_stream(deal=deal)
        events = vecenv_stream(deal=deal)
        assert_same_stream(recorder, events)
        forced += sum(e.forced for e in recorder.events)
        tribute += sum(e.phase in (TRIBUTE, BACK) for e in recorder.events)
        # The history round replays the plain auto-pass round exactly.
        engine, state = gd.Engine(), gd.MatchState()
        engine.set_deal(state, deal)
        assert play_round_seats(engine, state, [GreedyPolicy()] * 4, seed=0) == score
    assert forced > 0 and tribute > 0


def test_scalar_match_stream_matches_vecenv_across_rounds():
    recorder, scores = scalar_stream(match_seed=81)
    events = vecenv_stream(match_seed=81)
    assert_same_stream(recorder, events)
    rounds = sorted({e.round_index for e in recorder.events})
    assert rounds == list(range(len(scores))) and len(scores) > 1
    assert any(e.phase in (TRIBUTE, BACK) for e in recorder.events)
    assert recorder.starts == 1


def test_forced_pass_resolution_requires_explicit_passes():
    engine, state = gd.Engine(), gd.MatchState()
    engine.set_deal(state, generate_deals(1, 5)[0])
    # With auto-pass on, apply already skipped them: nothing left to resolve.
    engine.apply(state, engine.legal_actions(state)[engine.greedy(state)])
    assert resolve_forced_passes(engine, state, ()) == 0
    engine.auto_pass = False
    stream = PublicStream()
    class Sink:
        def observe(self, event): stream.append(event)
    total = 0
    while state.phase == gd.Phase.Play:
        total += resolve_forced_passes(engine, state, [Sink()])
        if state.phase != gd.Phase.Play:
            break
        apply_and_observe(engine, state, engine.legal_actions(state)[engine.greedy(state)], [Sink()])
        total += 1
    assert stream.prefix == total and total > 0


# ---- scalar evaluators ---------------------------------------------------------

class CountingEngine:
    """Delegating gd.Engine proxy that counts every apply, forced passes included."""
    created = []

    def __init__(self, *args, **kwargs):
        self._engine = REAL_ENGINE(*args, **kwargs)
        self.applied = 0
        CountingEngine.created.append(self)

    def apply(self, state, action):
        self.applied += 1
        return self._engine.apply(state, action)

    @property
    def auto_pass(self):
        return self._engine.auto_pass

    @auto_pass.setter
    def auto_pass(self, value):
        self._engine.auto_pass = value

    def __getattr__(self, name):
        return getattr(self._engine, name)


REAL_ENGINE = gd.Engine


def spy_scalar(monkeypatch, policy):
    """Record (round_index, prefix, applied since start_match) at every select."""
    CountingEngine.created.clear()
    monkeypatch.setattr(gd, "Engine", CountingEngine)
    marks, seen = {}, []
    real_start, real_select = HistoryPolicy.start_match, HistoryPolicy.select

    def start_match(self, match_id=-1):
        real_start(self, match_id)
        marks[id(self)] = CountingEngine.created[-1].applied

    def select(self, engine, state, actions, rng):
        seen.append((engine.applied - marks[id(self)], state.round_index, self.events_seen))
        return real_select(self, engine, state, actions, rng)

    monkeypatch.setattr(HistoryPolicy, "start_match", start_match)
    monkeypatch.setattr(HistoryPolicy, "select", select)
    return seen


def test_duplicate_prefix_counts_public_actions_and_resets_every_leg(checkpoint, monkeypatch):
    policy = history_policy(checkpoint)
    seen = spy_scalar(monkeypatch, policy)
    deals = generate_deals(2, 9) + generate_tribute_deals(2, 9)
    for i, deal in enumerate(deals):
        play_duplicate(deal, policy, GreedyPolicy(), seed=i)
        play_duplicate_teams(deal, (policy, GreedyPolicy()), (GreedyPolicy(), policy), seed=i)
    assert policy.matches_started == 4 * len(deals)
    assert seen and all(applied == prefix for applied, _, prefix in seen)
    assert all(round_index in (0, 1) for _, round_index, _ in seen)   # a tribute deal is round 1
    # Every leg starts over: the first decision of a leg saw at most the
    # tribute exchanges and forced passes before it, never the previous leg.
    firsts = [prefix for i, (_, _, prefix) in enumerate(seen)
              if i == 0 or seen[i - 1][2] > prefix]
    assert len(firsts) == 4 * len(deals) and max(firsts) < 8
    assert all(engine.auto_pass is True for engine in CountingEngine.created)


def test_matches_keep_prefix_across_rounds_and_reset_per_match(checkpoint, monkeypatch):
    policy = history_policy(checkpoint)
    seen = spy_scalar(monkeypatch, policy)
    totals = play_matches(policy, GreedyPolicy(), [0, 1], seed=7, max_rounds=200)
    assert totals["matches"] == 2 and policy.matches_started == 2
    assert all(applied == prefix for applied, _, prefix in seen)
    # Split the trace at the two match starts (prefix returns to a small value
    # while the round index returns to zero).
    matches, current = [], []
    for applied, round_index, prefix in seen:
        if current and round_index == 0 and current[-1][0] > 0:
            matches.append(current)
            current = []
        current.append((round_index, prefix))
    matches.append(current)
    assert len(matches) == 2
    for trace in matches:
        rounds = [r for r, _ in trace]
        assert rounds == sorted(rounds) and rounds[-1] >= 1
        for r in range(1, rounds[-1] + 1):
            before = max(p for rr, p in trace if rr == r - 1)
            first = min(p for rr, p in trace if rr == r)
            assert first >= before > 0
    assert sum(len(t) for t in matches) == policy.decisions


def test_sampled_history_policy_replays_with_fixed_seed(checkpoint):
    policy = history_policy(checkpoint, sample=True)
    assert policy.sample and policy.name.endswith("/sample")
    deal = generate_deals(1, 3)[0]
    first = play_duplicate(deal, policy, GreedyPolicy(), seed=5)
    assert play_duplicate(deal, policy, GreedyPolicy(), seed=5) == first


# ---- batched evaluator ---------------------------------------------------------

def test_batched_history_play_equals_scalar_history_play(checkpoint):
    policy = history_policy(checkpoint)
    team = (policy, load_policy("styled:high-lead"))
    opponents = (GreedyPolicy(), load_policy("styled:bomb-happy"))
    deals = generate_deals(3, 33) + generate_tribute_deals(3, 33)
    expected = [play_duplicate_teams(d, team, opponents, 101 + i) for i, d in enumerate(deals)]
    config = EvalConfig(batch_size=4, engine_threads=2)
    assert play_duplicate_batch(deals, team, opponents, 101, config) == expected


def test_batched_match_slots_equal_scalar_seed_pairs(checkpoint):
    policy = history_policy(checkpoint)
    opponent = load_policy("styled:low-lead")
    slots = [(seed, team) for seed in (11, 12) for team in (0, 1)]
    expected = [play_matches(policy, opponent, [team], seed=seed - team, max_rounds=200)
                for seed, team in slots]
    got = play_match_slots_batch(policy, opponent, slots, max_rounds=200,
                                 config=EvalConfig(batch_size=3))
    assert got == expected


def test_batched_kv_cache_plays_like_full_prefix(checkpoint):
    # The KV cache equals full-prefix recomputation up to floating point; on
    # this small actor the greedy choices, hence every result, coincide.
    policy = history_policy(checkpoint)
    opponent = load_policy("styled:low-lead")
    deals = generate_deals(4, 21)
    slots = [(seed, team) for seed in (31, 32) for team in (0, 1)]
    for cached in (False, True):
        config = EvalConfig(batch_size=3, kv_cache=cached)
        result = (play_duplicate_batch(deals, (policy, policy), (opponent, opponent), 7, config),
                  play_match_slots_batch(policy, opponent, slots, max_rounds=200, config=config))
        if cached:
            assert result == reference
        reference = result


def test_batched_slots_track_drained_events_and_restart_fresh(checkpoint, monkeypatch):
    import eval.batched as batched

    policy = history_policy(checkpoint)
    counts, real_vec_env, stores = {}, gd.VecEnv, []

    class CountingVecEnv:
        def __init__(self, *args, **kwargs):
            assert kwargs["log_public_actions"] is True and kwargs["encode"] is True
            self._env = real_vec_env(*args, **kwargs)

        def drain_public_actions(self):
            events = self._env.drain_public_actions()
            for event in events:
                key = (int(event.env_id), int(event.match_id))
                counts[key] = counts.get(key, 0) + 1
            return events

        def __getattr__(self, name):
            return getattr(self._env, name)

    class SpyStore(HistoryStreamStore):
        def __init__(self, num_envs):
            super().__init__(num_envs)
            stores.append(self)

        def sync(self, env_ids, match_ids):
            super().sync(env_ids, match_ids)
            for env_id, match_id in zip(env_ids, match_ids):
                stream = self.streams[int(env_id)]
                assert stream.match_id == int(match_id)
                assert stream.prefix == counts.get((int(env_id), int(match_id)), 0)

    monkeypatch.setattr(gd, "VecEnv", CountingVecEnv)
    monkeypatch.setattr(batched, "HistoryStreamStore", SpyStore)
    totals = play_matches_batch(policy, GreedyPolicy(), [0, 1, 2], seed=5, max_rounds=200,
                                config=EvalConfig(batch_size=3))
    assert totals["matches"] == 3 and len(stores) == 1
    (store,) = stores
    restarted = [e for e, stream in enumerate(store.streams) if stream.match_id > 0]
    assert restarted, "no slot restarted a match inside the wave"
    for env_id, stream in enumerate(store.streams):
        assert stream.prefix == counts[(env_id, stream.match_id)]
        if stream.match_id:
            assert (env_id, stream.match_id - 1) in counts
    assert store.ingested == sum(counts.values())


def test_stream_store_resets_on_match_change():
    class Event:
        def __init__(self, env_id, match_id, seat=0, cards_left=5, round_index=0, phase=PLAY):
            self.env_id, self.match_id, self.seat = env_id, match_id, seat
            self.encoded_action = np.zeros(int(gd.ACT_DIM), np.float32)
            self.cards_left, self.round_index, self.phase = cards_left, round_index, phase

    store = HistoryStreamStore(2)
    store.ingest([Event(0, 0), Event(0, 0, seat=1), Event(1, 0)])
    assert [s.prefix for s in store.streams] == [2, 1]
    store.ingest([Event(0, 1)])
    assert store.streams[0].match_id == 1 and store.streams[0].prefix == 1
    store.sync(np.array([1]), np.array([3]))
    assert store.streams[1].match_id == 3 and store.streams[1].prefix == 0
    streams, index = store.select_streams(np.array([1, 0, 1]))
    assert streams == [store.streams[0], store.streams[1]] and index.tolist() == [1, 0, 1]
    with pytest.raises(ValueError):
        HistoryStreamStore(0)
    with pytest.raises(ValueError, match="stream store"):
        BatchActor(HistoryPolicy(fresh_player(TINY, 1)[0]), 0)


# ---- DanLM lockstep -------------------------------------------------------------

def test_lockstep_hook_delivers_every_mirror_action_with_a_fake_referee(checkpoint):
    """Replay a recorded action list through ``apply_ours`` without DanLM."""
    from eval.danlm.arena import LockstepRound

    policy = history_policy(checkpoint)
    deal = generate_tribute_deals(1, 11)[0]
    recorder, _ = scalar_stream(deal=deal)
    actions = [event.action for event in recorder.events]
    lockstep = LockstepRound.__new__(LockstepRound)
    lockstep.engine_full = gd.Engine(gd.RuleConfig.house(), gd.ActionConfig.full())
    lockstep.engine_full.auto_pass = False
    lockstep.state = gd.MatchState()
    lockstep.engine_full.set_deal(lockstep.state, deal)
    lockstep.history, lockstep.applied, lockstep.round_id = policy, 0, "fake"
    policy.start_match()
    for action in actions:
        assert lockstep.state.phase != gd.Phase.RoundEnd
        lockstep.apply_ours(action)
    assert lockstep.state.phase == gd.Phase.RoundEnd
    assert lockstep.applied == len(actions) == policy.events_seen
    lockstep.check_history()
    # The hook rebuilt the recorded stream exactly, forced passes included.
    for ours, theirs in zip(policy.stream.arrays(), recorder.stream.arrays()):
        np.testing.assert_array_equal(ours, theirs)
    assert sum(a.is_pass for a in actions) > 0
    lockstep.applied += 1
    with pytest.raises(RuntimeError, match="applied mirror actions"):
        lockstep.check_history()


def test_lockstep_rounds_against_danlm_feed_the_history_policy(checkpoint):
    root = os.environ.get("DANLM_ROOT", ".work/external/DanLM")
    if os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)
    pytest.importorskip("danzero.engine.game", reason="DanLM's compiled package is not importable")
    from eval.danlm.arena import DanLMSeats, LockstepRound, generate_deals as danlm_deals

    policy = history_policy(checkpoint)
    danlm = DanLMSeats("random")
    deals = danlm_deals(2, 17, tribute_fraction=1.0) + danlm_deals(1, 18, tribute_fraction=0.0)
    for i, deal in enumerate(deals):
        seats = ("ours", "danlm", "ours", "danlm") if i % 2 == 0 else ("danlm", "ours", "danlm", "ours")
        lockstep = LockstepRound(deal, seats, policy, danlm, random.Random(i), round_id=f"t{i}")
        # Decision count before each mirror apply: an apply that the next apply
        # sees without a decision increment was an engine-resolved pass on the
        # mirror (the ``ours_only_pass`` branch), not a decision of any seat.
        before, real_apply = [], lockstep.apply_ours

        def apply_ours(action, lockstep=lockstep, before=before, real_apply=real_apply):
            before.append(lockstep.decisions)
            real_apply(action)

        lockstep.apply_ours = apply_ours
        record = lockstep.play()
        assert record.status in ("ok", "mirror_failed")
        trail = before + [lockstep.decisions]
        forced = sum(trail[k + 1] == trail[k] for k in range(len(before)))
        assert lockstep.applied == len(before) > 0
        assert policy.events_seen == lockstep.decisions + forced == lockstep.applied
        assert policy.matches_started == i + 1


# ---- explicit failures -----------------------------------------------------------

def test_unsupported_and_unstarted_policies_fail_explicitly(checkpoint, tmp_path):
    actor, critic = fresh_player(TINY, 1)
    payload = checkpoint_payload(actor, critic, lineage="x", seed=1)
    unknown = tmp_path / "unknown.pt"
    save_checkpoint(unknown, {**payload, "stage": "history_gan"})
    with pytest.raises(ValueError, match="unsupported checkpoint stage"):
        load_policy(str(unknown))
    with pytest.raises(ValueError, match="margin"):
        load_policy(str(checkpoint), margin=0.5)
    with pytest.raises(ValueError, match="temperature"):
        load_policy(f"sample=0.5:{checkpoint}")
    policy = history_policy(checkpoint)
    engine, state = gd.Engine(), gd.MatchState()
    engine.set_deal(state, generate_deals(1, 1)[0])
    with pytest.raises(RuntimeError, match="start_match"):
        policy.select(engine, state, engine.legal_actions(state), random.Random(0))
    with pytest.raises(RuntimeError, match="start_match"):
        play_round_seats(engine, state, (policy, GreedyPolicy(), policy, GreedyPolicy()))
    assert engine.auto_pass is True
    # A dropped play by any seat is caught at the next decision.
    policy.start_match()
    engine.auto_pass = False
    dropped = 0
    while state.phase == gd.Phase.Play and dropped == 0:
        resolve_forced_passes(engine, state, [policy])
        action = engine.legal_actions(state)[engine.greedy(state)]
        if not action.is_pass and policy.events_seen > 3:
            engine.apply(state, action)
            dropped += 1
        else:
            apply_and_observe(engine, state, action, [policy])
    resolve_forced_passes(engine, state, [policy])
    with pytest.raises(RuntimeError, match="was not delivered"):
        policy.select(engine, state, engine.legal_actions(state), random.Random(0))
    assert history_listeners((policy, GreedyPolicy(), policy)) == [policy]


def test_two_history_policies_hear_the_same_events_and_agree_batched(checkpoint, tmp_path):
    actor, critic = fresh_player(TINY, seed=4)
    other = tmp_path / "other.pt"
    save_history_checkpoint(other, checkpoint_payload(actor, critic, lineage="test", seed=4))
    a, b = history_policy(checkpoint), history_policy(other)
    assert a.checkpoint_id != b.checkpoint_id
    deals = generate_deals(2, 5) + generate_tribute_deals(2, 5)
    scalar = []
    for i, deal in enumerate(deals):
        scalar.append(play_duplicate(deal, a, b, seed=i))
        assert a.events_seen == b.events_seen > 0
        assert a.matches_started == b.matches_started == 2 * (i + 1)
        for ours, theirs in zip(a.stream.arrays(), b.stream.arrays()):
            np.testing.assert_array_equal(ours, theirs)
    config = EvalConfig(batch_size=4)
    assert play_duplicate_batch(deals, (a, a), (b, b), 0, config) == scalar
