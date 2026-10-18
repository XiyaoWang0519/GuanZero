"""Bitwise host fast paths of the rollout collector.

``VecEnv.drain_public_arrays`` with ``ingest_arrays``, the one uint8 cast of a
batch's observations and candidates, the collector's per-row stream reuse,
the frozen ``SnapshotHeads`` slot memo and NumPy upload staging must leave
every stored byte, stream, choice, generator state and learned weight exactly
as the per-event and per-call paths they replace. The fallbacks stay
reachable (envs without the array drain, stores overriding ``stream``, no
frozen scope) and serve as the reference here.
"""
import json

import gd
import numpy as np
import pytest
import torch

from train.history_habit import RoundEventStore
from train.history_model import HistoryPolicyConfig, PublicStream, fresh_player
from train.history_ppo import HistoryPPOConfig, HistoryTrainer
from train.history_rollout import (CollectStats, HistoryCollector, MatchEventStore,
                                   SequenceRolloutBuffer)
from train.history_snapshot_batch import SnapshotHeads
import train.history_snapshot_batch as snapshot_batch
from train.history_transfers import _byte_view, pack_host, upload_arrays
from train.logs import TOKEN_DIM, public_token, public_tokens

TRIBUTE, BACK_TRIBUTE = int(gd.Phase.Tribute), int(gd.Phase.BackTribute)


# ---- reference paths ------------------------------------------------------------

class Hidden:
    """Forwarding proxy that lacks some attributes, so callers take fallbacks."""

    def __init__(self, inner, hidden):
        self._inner, self._hidden = inner, frozenset(hidden)

    def __getattr__(self, name):
        if name in self._hidden:
            raise AttributeError(name)
        return getattr(self._inner, name)


class LegacyEnv(Hidden):
    """A VecEnv without the array drain: the collector ingests event objects."""

    def __init__(self, env):
        super().__init__(env, {"drain_public_arrays"})


class LegacyStore(MatchEventStore):
    """Overrides ``stream``: the collector keeps its generic per-row loop."""

    def stream(self, env_id, match_id):
        return super().stream(env_id, match_id)


def legacy(collector):
    """Force every fallback: event objects, per-row store lookups and per-step
    snapshot-head signatures."""
    collector.env = LegacyEnv(collector.env)
    if type(collector.store) is MatchEventStore:
        collector.store.__class__ = LegacyStore
    collector._freeze_snapshot_heads = lambda: None
    return collector


def stream_state(stream):
    """Everything a stream holds, spare capacity included."""
    return (stream.match_id, stream.generation, stream.prefix, len(stream._rounds),
            stream._tokens.dtype.str, stream._tokens.tobytes(), stream._rounds.dtype.str,
            stream._rounds.tobytes(), stream._phases.dtype.str, stream._phases.tobytes())


def store_state(store):
    state = dict(keys=list(store.streams), current=list(store.current.items()),
                 streams=[stream_state(s) for s in store.streams.values()])
    if isinstance(store, RoundEventStore):
        state.update(round_keys=list(store.rounds),
                     rounds=[stream_state(s) for s in store.rounds.values()])
    return state


def buffer_state(buffer):
    data = buffer.compact()
    return dict(arrays={k: (v.dtype.str, v.shape, v.tobytes()) for k, v in data.items()},
                trajectories=[(t.env, t.team, t.match, t.round, list(t.rows), t.complete, t.reward)
                              for t in buffer.trajectories],
                open=sorted(buffer.open.items()))


# ---- binding and token builder ---------------------------------------------------

def twin_rollout(num_envs, steps, seed, rules, log_env_limit):
    """Two identical VecEnvs stepped with the same mixed greedy/random legal
    choices: per step the first env's array drain and the second's events."""
    envs = [gd.VecEnv(num_envs=num_envs, num_threads=1, seed=seed, rules=rules,
                      log_public_actions=True, log_env_limit=log_env_limit) for _ in range(2)]
    for env in envs:
        env.reset()
    rng = np.random.default_rng(seed)
    matches = 0
    for _ in range(steps):
        batch, other = envs[0].pending(), envs[1].pending()
        yield envs[0].drain_public_arrays(), envs[1].drain_public_actions(), batch
        finished = envs[0].drain_finished_rounds()
        assert len(finished) == len(envs[1].drain_finished_rounds())
        matches += sum(r.match_winner >= 0 for r in finished)
        offsets = np.asarray(batch.offsets)
        counts = offsets[1:] - offsets[:-1]
        choices = np.array(batch.greedy_choice, dtype=np.int32)
        random = (np.asarray(batch.env_id) % 2 == 1) & (rng.random(len(counts)) < 0.5)
        choices[random] = (rng.random(int(random.sum())) * counts[random]).astype(np.int32)
        assert np.array_equal(np.asarray(other.greedy_choice), np.asarray(batch.greedy_choice))
        for env in envs:
            env.step(choices)
    assert matches >= 1, "the sample should cross a match boundary"


@pytest.mark.parametrize("rules,limit", [("house", -1), ("ogd", 3)])
def test_array_drain_equals_event_objects_and_tokens(rules, limit):
    rules = getattr(gd.RuleConfig, rules)()
    phases, events_seen = set(), 0
    for arrays, events, _ in twin_rollout(6, 900, 17, rules, limit):
        n = len(events)
        assert {k: v.shape[0] for k, v in arrays.items()} == dict.fromkeys(arrays, n)
        assert arrays["encoded_action"].dtype == np.float32
        assert arrays["encoded_action"].shape == (n, 154)
        assert [arrays[k].dtype for k in ("env_id", "match_id", "round_index", "seat", "phase",
                                          "cards_left")] == [np.int32, np.int64, np.int32,
                                                             np.int8, np.uint8, np.int32]
        expected = dict(env_id=[e.env_id for e in events], match_id=[e.match_id for e in events],
                        round_index=[e.round_index for e in events], seat=[e.seat for e in events],
                        phase=[int(e.phase) for e in events],
                        cards_left=[e.cards_left for e in events])
        for name, values in expected.items():
            assert arrays[name].tolist() == values, name
        if n:
            encoded = np.stack([e.encoded_action for e in events])
            assert arrays["encoded_action"].tobytes() == encoded.tobytes()
            tokens = public_tokens(arrays["seat"], arrays["encoded_action"], arrays["cards_left"])
            assert tokens.dtype == np.uint8
            assert tokens.tobytes() == np.stack([public_token(e) for e in events]).tobytes()
        if limit > 0:
            assert all(e.env_id < limit for e in events)
        phases.update(arrays["phase"].tolist())
        events_seen += n
    assert {TRIBUTE, BACK_TRIBUTE} <= phases and events_seen > 1000


def test_public_tokens_reject_out_of_range_one_hots():
    encoded = np.zeros((2, 154), np.uint8)
    for seat, cards in (([0, 4], [1, 2]), ([-1, 0], [1, 2]), ([0, 1], [28, 2]),
                        ([0, 1], [0, -1])):
        with pytest.raises(ValueError):
            public_tokens(np.asarray(seat, np.int8), encoded, np.asarray(cards, np.int32))
    with pytest.raises(ValueError):
        public_tokens(np.zeros(2, np.int8), np.zeros((2, 150)), np.zeros(2, np.int32))
    empty = public_tokens(np.zeros(0, np.int8), np.zeros((0, 154), np.uint8), np.zeros(0))
    assert empty.shape == (0, TOKEN_DIM) and empty.dtype == np.uint8


@pytest.mark.parametrize("store_type", [MatchEventStore, RoundEventStore])
def test_array_ingest_builds_identical_stores(store_type):
    by_arrays, by_events = store_type(), store_type()
    for step, (arrays, events, _) in enumerate(twin_rollout(8, 1500, 29, gd.RuleConfig.house(), -1)):
        assert by_arrays.ingest_arrays(arrays) == by_events.ingest(events) == len(events)
        if step % 250 == 0:
            assert store_state(by_arrays) == store_state(by_events)
            # Both paths keep views of earlier prefixes intact.
            views = [(s.tokens, s.tokens.tobytes()) for s in by_arrays.streams.values()]
        else:
            assert all(v.tobytes() == b for v, b in views)
    assert store_state(by_arrays) == store_state(by_events)
    assert max(len(s._rounds) for s in by_arrays.streams.values()) >= 512   # several doublings
    assert len({key[0] for key in by_arrays.streams}) == 8
    assert len(by_arrays.streams) > 8                                         # new matches
    phases = {p for s in by_arrays.streams.values() for p in s.phases.tolist()}
    assert {TRIBUTE, BACK_TRIBUTE} <= phases


def test_array_ingest_validates_the_batch_and_keeps_match_order():
    rng = np.random.default_rng(3)
    n = 6
    events = dict(env_id=np.asarray([0, 0, 1, 1, 1, 0], np.int32),
                  match_id=np.asarray([0, 0, 0, 0, 0, 1], np.int64),
                  round_index=np.asarray([0, 0, 0, 1, 1, 0], np.int32),
                  seat=rng.integers(0, 4, n).astype(np.int8),
                  phase=np.full(n, 3, np.uint8), cards_left=rng.integers(0, 28, n).astype(np.int32),
                  encoded_action=(rng.random((n, 154)) < 0.1).astype(np.uint8))
    store = RoundEventStore()
    assert store.ingest_arrays(events) == n
    assert [s.prefix for s in store.streams.values()] == [2, 3, 1]
    assert {k: s.prefix for k, s in store.rounds.items()} == {(0, 0, 0): 2, (1, 0, 0): 1,
                                                              (1, 0, 1): 2, (0, 1, 0): 1}
    assert not any(s.tokens[:, 150:158].any() for s in store.streams.values())
    with pytest.raises(ValueError, match="went backwards"):
        MatchEventStore().ingest_arrays(dict(events, env_id=np.zeros(n, np.int32),
                                             match_id=np.asarray([0, 0, 2, 2, 1, 1])))
    for store_type in (MatchEventStore, RoundEventStore):
        store = store_type()
        with pytest.raises(ValueError, match="must not be negative"):
            store.ingest_arrays(dict(events, round_index=np.asarray([0, 0, 0, -1, 0, 0])))
        assert not store.streams         # nothing of a malformed batch is appended
    empty = {k: v[:0] for k, v in events.items()}
    assert MatchEventStore().ingest_arrays(empty) == 0


def valid_tokens(rng, k):
    tokens = np.zeros((k, TOKEN_DIM), np.uint8)
    tokens[np.arange(k), rng.integers(0, 4, k)] = 1
    tokens[:, 4:150] = rng.random((k, 146)) < 0.05
    tokens[np.arange(k), 158 + rng.integers(0, 28, k)] = 1
    return tokens


def test_extend_tokens_matches_repeated_append_token():
    rng = np.random.default_rng(5)
    bulk, single = PublicStream(4), PublicStream(4)
    views = []
    for size in (0, 1, 63, 1, 64, 200, 5, 300):
        tokens = valid_tokens(rng, size)
        rounds = rng.integers(0, 9, size)
        phases = rng.integers(0, 4, size).astype(np.int32)
        bulk.extend_tokens(tokens, rounds, phases)
        for token, rnd, phase in zip(tokens, rounds, phases):
            single.append_token(token, rnd, phase)
        assert stream_state(bulk) == stream_state(single)
        views.append((bulk.tokens, bulk.tokens.tobytes()))
    assert all(view.tobytes() == data for view, data in views)
    bulk.reset(5)
    single.reset(5)
    tokens = valid_tokens(rng, 3)
    bulk.extend_tokens(tokens, [0, 0, 0], [3, 3, 3])
    for token in tokens:
        single.append_token(token, 0, 3)
    assert stream_state(bulk) == stream_state(single)
    assert bulk.prefix == 3 and len(bulk._rounds) == 64 and bulk.generation == 1


@pytest.mark.parametrize("defect", ["seat", "cards", "private", "round", "phase", "shape"])
def test_extend_tokens_raises_like_append_token(defect):
    rng = np.random.default_rng(11)
    tokens = valid_tokens(rng, 70)
    rounds, phases = np.zeros(70, np.int64), np.full(70, 3, np.int64)
    bad = 66
    if defect == "seat":
        tokens[bad, :4] = [1, 1, 0, 0]
    elif defect == "cards":
        tokens[bad, 158:] = 0
    elif defect == "private":
        tokens[bad, 4 + 147] = 1
    elif defect == "round":
        rounds[bad] = -1
    elif defect == "phase":
        phases[bad] = -2
    else:
        tokens = tokens[:, :-1]
    bulk, single = PublicStream(), PublicStream()
    with pytest.raises(ValueError) as bulk_error:
        bulk.extend_tokens(tokens, rounds, phases)
    with pytest.raises(ValueError) as single_error:
        for token, rnd, phase in zip(tokens, rounds, phases):
            single.append_token(token, int(rnd), int(phase))
    assert str(bulk_error.value) == str(single_error.value)
    assert stream_state(bulk) == stream_state(single)
    assert bulk.prefix == (0 if defect == "shape" else bad)


# ---- collector -----------------------------------------------------------------------

POLICY = HistoryPolicyConfig(width=16, layers=1, heads=2, action_width=16, fusion_width=16,
                             critic_width=16, response_mode="auxiliary")


def production_collector(fast, store_type=MatchEventStore, **options):
    """Collector at the main lineage's rollout options with three frozen
    snapshot identities seated from the start by a seeded seat policy."""
    policies = {i: fresh_player(POLICY, 60 + i)[0].train() for i in range(4)}
    for identity in (1, 2, 3):
        policies[identity].requires_grad_(False)
    rng = np.random.default_rng(8)

    def seats(env, match):
        drawn = rng.integers(0, 4, 4)
        drawn[rng.integers(4)] = 0
        return drawn.tolist()

    options = dict(dict(kv_cache=True, paged_cache=True, page_span=True,
                        batch_snapshot_policies=True, batch_snapshot_encoder=True,
                        prefill_learner_cache=True), **options)
    env = gd.VecEnv(num_envs=8, num_threads=1, seed=41, log_public_actions=True, log_env_limit=8)
    collector = HistoryCollector(env, policies[0], store_type(), SequenceRolloutBuffer(),
                                 torch.Generator().manual_seed(9), seat_policy=seats,
                                 resolve_policy=policies.__getitem__, record_choices=True,
                                 **options)
    collector.seat_rng = rng
    return collector if fast else legacy(collector)


def run_chunks(collector, chunks=3, steps=40):
    states = []
    for chunk in range(chunks):
        with collector.frozen_weights():
            stats = collector.collect(steps, version=chunk)
        states.append(dict(stats=stats, buffer=buffer_state(collector.buffer),
                           store=store_state(collector.store),
                           choices=[c.tobytes() for c in collector.choice_log],
                           generator=collector.generator.get_state().numpy().tobytes(),
                           seats=collector.seat_rng.bit_generator.state,
                           assignments=[(k, v.tolist()) for k, v in collector.assignments.items()],
                           decisions=dict(collector.policy_decisions)))
        # As a learner update would: weights change between collects, old rows go.
        collector.invalidate_learner_cache()
        with torch.no_grad():
            for parameter in collector.actor.parameters():
                parameter.add_(1e-3 * torch.sin(torch.arange(parameter.numel()).reshape(
                    parameter.shape).float() + chunk))
        cited = collector.buffer.next_iteration()
        collector.store.prune(cited)
    return states


@pytest.mark.parametrize("store_type,options", [
    (MatchEventStore, {}),
    (MatchEventStore, dict(paged_cache=False, page_span=False, batch_snapshot_encoder=False)),
    (RoundEventStore, dict(paged_cache=False, page_span=False, batch_snapshot_encoder=False,
                           prefill_learner_cache=False, round_streams=True)),
])
def test_collector_fast_paths_replay_the_fallbacks_bit_for_bit(store_type, options):
    torch.manual_seed(0)
    fast = production_collector(True, store_type, **options)
    reference = production_collector(False, store_type, **options)
    calls = {"arrays": 0}
    ingest_arrays = fast.store.ingest_arrays

    def counted(events):
        calls["arrays"] += 1
        return ingest_arrays(events)

    fast.store.ingest_arrays = counted
    reference.store.ingest_arrays = None      # must never be reached
    expected, actual = run_chunks(reference), run_chunks(fast)
    assert calls["arrays"] == 120
    for before, after in zip(expected, actual):
        assert before == after
    assert sum(actual[-1]["decisions"].get(i, 0) for i in (1, 2, 3)) > 0, "snapshot seats played"


def test_collector_row_streams_are_the_store_streams():
    collector = production_collector(True)
    collector.env.reset()
    collector.started = True
    batch = collector.env.pending()
    collector._drain(CollectStats())
    rows = collector._read_rows(batch)
    assert rows.obs.dtype == rows.cand.dtype == np.uint8
    assert rows.obs.flags.owndata and rows.cand.flags.owndata    # valid past pending()
    play = np.flatnonzero(rows.phase == 3)
    assert np.array_equal(rows.obs[play], np.asarray(batch.obs)[play].astype(np.uint8))
    assert np.array_equal(rows.cand, np.asarray(batch.cand).astype(np.uint8))
    assert all(stream is collector.store.streams[(e, m)] for stream, e, m in
               zip(rows.streams, rows.env_id.tolist(), rows.match_id.tolist()))
    assert rows.prefix.tolist() == [s.prefix for s in rows.streams]


# ---- snapshot head slots -----------------------------------------------------------

def head_actors():
    return {i: fresh_player(POLICY, 70 + i)[0].requires_grad_(False) for i in (1, 2)}


def test_frozen_heads_sign_each_identity_once(monkeypatch):
    actors = head_actors()
    heads, plain = SnapshotHeads(encoder=True), SnapshotHeads(encoder=True)
    signs = {"count": 0}
    original = snapshot_batch.head_signature

    def counting(actor):
        signs["count"] += 1
        return original(actor)

    monkeypatch.setattr(snapshot_batch, "head_signature", counting)
    with heads.frozen_weights():
        for _ in range(5):
            assert [heads.slot(i, actors[i]) for i in (1, 2)] == [0, 1]
        assert signs["count"] == 2
        assert heads.writes == 2
    assert signs["count"] == 4                    # re-signed once each at exit
    for _ in range(3):
        assert [plain.slot(i, actors[i]) for i in (1, 2)] == [0, 1]
    assert signs["count"] == 10                   # outside a scope: every call
    for name, tensor in heads.tensors.items():
        assert torch.equal(tensor, plain.tensors[name]), name
    with pytest.raises(RuntimeError, match="nested"):
        with heads.frozen_weights():
            with heads.frozen_weights():
                pass


@pytest.mark.parametrize("change", ["in_place", "rebind"])
def test_frozen_heads_raise_when_snapshot_weights_change(change):
    actors = head_actors()
    heads = SnapshotHeads(encoder=True)
    with pytest.raises(RuntimeError, match="snapshot weights changed"):
        with heads.frozen_weights():
            heads.slot(1, actors[1])
            heads.slot(2, actors[2])
            weight = actors[2].q_proj.weight if change == "in_place" else actors[2].bos
            with torch.no_grad():
                if change == "in_place":
                    weight.add_(1.0)
                else:
                    weight.data = weight.data.clone() + 1.0
            assert heads.slot(2, actors[2]) == 1      # memo hit until the exit check
    assert set(heads.slots) == {1}                    # the stale slot is dropped
    writes = heads.writes
    assert heads.slot(2, actors[2]) == 1 and heads.writes == writes + 1
    fresh = SnapshotHeads(encoder=True)
    fresh.slot(1, actors[1])
    fresh.slot(2, actors[2])
    for name, tensor in heads.tensors.items():
        assert torch.equal(tensor, fresh.tensors[name]), name


def test_frozen_heads_follow_retain_and_new_actor_objects():
    actors = head_actors()
    heads = SnapshotHeads()
    with heads.frozen_weights():
        assert heads.slot(1, actors[1]) == 0
        heads.retain({2})
        assert heads.slot(2, actors[2]) == 0          # freed slot reused
        assert heads.slot(1, actors[1]) == 1          # dropped identity rewritten
        replacement = fresh_player(POLICY, 99)[0].requires_grad_(False)
        writes = heads.writes
        assert heads.slot(1, replacement) == 1 and heads.writes == writes + 1
        assert heads.slots[1].actor() is replacement


# ---- trainer ---------------------------------------------------------------------------

MAIN_OPTIONS = dict(
    width=16, layers=1, heads=2, response_mode="auxiliary", aux_heads="next,belief,outcome",
    aux_coef=0.1, entropy=0.03, epochs=2, lr=3e-4, rollout_epsilon=0.0,
    snapshot_updates=1, population_recent=4, population_archive_every=0,
    causal_sdpa=True, rollout_kv_cache=True, rollout_batched_attention=True,
    rollout_wide_projection=True, rollout_paged_cache=True, rollout_page_span=True,
    batch_snapshot_policies=True, batch_snapshot_encoder=True,
    rollout_prefill_learner_cache=True, learner_length_groups=2, learner_batched_attention=True)
TIMING = ("seconds", "per_sec")


def scrub(line):
    """A metrics line without wall-clock fields, at any depth."""
    if not isinstance(line, dict):
        return line
    return {k: scrub(v) for k, v in line.items() if not any(t in str(k) for t in TIMING)}


def trainer_state(trainer):
    return dict(buffer=buffer_state(trainer.buffer), store=store_state(trainer.store),
                sampler=trainer.generator.get_state().numpy().tobytes(),
                minibatch_rng=json.dumps(trainer.rng.bit_generator.state, default=str),
                population_rng=json.dumps(trainer.population.rng.bit_generator.state, default=str),
                assignments=[(k, v.tolist()) for k, v in trainer.collector.assignments.items()],
                actor={k: v.detach().clone() for k, v in trainer.actor.state_dict().items()},
                critic={k: v.detach().clone() for k, v in trainer.critic.state_dict().items()})


def assert_same(before, after):
    for name in before:
        if name in ("actor", "critic"):
            assert before[name].keys() == after[name].keys()
            for key, tensor in before[name].items():
                assert tensor.dtype == after[name][key].dtype
                assert torch.equal(tensor, after[name][key]), (name, key)
        else:
            assert before[name] == after[name], name


def test_trainer_updates_replay_the_fallbacks_bit_for_bit(tmp_path):
    runs = {}
    for name in ("reference", "fast"):
        config = HistoryPPOConfig(**MAIN_OPTIONS, num_envs=8, steps_per_update=40,
                                  minibatch_matches=4, updates=100, seed=2026101801,
                                  torch_threads=1, num_threads=1, checkpoint_updates=10**9)
        torch.manual_seed(0)
        trainer = HistoryTrainer(config, tmp_path / name, device="cpu")
        if name == "reference":
            legacy(trainer.collector)
        runs[name] = dict(trainer=trainer, rng=torch.get_rng_state())
    snapshot_rows = 0
    for update in range(6):
        lines, states = {}, {}
        order = ("reference", "fast") if update % 2 == 0 else ("fast", "reference")
        for name in order:
            run = runs[name]
            trainer = run["trainer"]
            torch.set_rng_state(run["rng"])       # each trainer owns the global stream
            snapshots = len(trainer.population.models)
            lines[name] = scrub(trainer.update())
            run["rng"] = torch.get_rng_state()
            states[name] = trainer_state(trainer)
            if len(trainer.population.models) > snapshots:
                # Seat the new snapshot now rather than when matches end (~450 steps).
                trainer.collector.assignments.clear()
        assert json.dumps(lines["reference"], sort_keys=True, default=str) == \
            json.dumps(lines["fast"], sort_keys=True, default=str)
        assert_same(states["reference"], states["fast"])
        assert torch.equal(runs["reference"]["rng"], runs["fast"]["rng"])
        decisions = runs["fast"]["trainer"].collector.policy_decisions
        snapshot_rows += sum(count for identity, count in decisions.items() if identity)
    assert snapshot_rows > 0 and len(runs["fast"]["trainer"].population.models) >= 2
    assert runs["fast"]["trainer"].collector.snapshot_heads.writes > 0
    assert isinstance(runs["reference"]["trainer"].collector.env, LegacyEnv)


# ---- upload staging -------------------------------------------------------------------

def torch_staging(arrays):
    """The packed staging before NumPy: torch.zeros plus one _byte_view copy per field."""
    tensors = tuple(torch.from_numpy(np.asarray(array)) for array in arrays)
    layout, total = [], 0
    for tensor in tensors:
        start = (total + 7) // 8 * 8
        total = start + tensor.numel() * tensor.element_size()
        layout.append((start, total))
    host = torch.zeros(total, dtype=torch.uint8)
    for tensor, (start, end) in zip(tensors, layout):
        host[start:end].copy_(_byte_view(tensor))
    views = tuple(host[start:end].view(t.dtype).reshape(t.shape) for t, (start, end)
                  in zip(tensors, layout))
    return host, layout, views


def field_matrix():
    rng = np.random.default_rng(4)
    matrix = rng.standard_normal((6, 7)).astype(np.float32)
    nan = np.asarray([np.nan, -0.0, np.inf], np.float32)
    nan.view(np.uint32)[0] = 0x7fc00123            # a NaN payload must survive
    return [
        rng.integers(-9, 9, 5), rng.integers(0, 2, (3, 4)).astype(np.uint8),
        rng.random(3) < 0.5, np.asarray(7, np.int64), np.asarray(True),
        np.zeros(0, np.int64), np.zeros((0, 154), np.uint8), matrix, matrix[:, ::2],
        matrix.T, rng.standard_normal(3), nan, rng.integers(0, 5, 3).astype(np.int32),
        np.asarray([1.5], np.float64), rng.integers(0, 255, 13).astype(np.uint8),
        np.arange(10, dtype=np.int16)[::3], np.asarray(3.0, np.float32),
    ]


@pytest.mark.parametrize("subset", ["all", "odd_sizes", "single", "empty_first"])
def test_numpy_staging_matches_torch_staging_byte_for_byte(subset, monkeypatch):
    arrays = field_matrix()
    if subset == "odd_sizes":
        arrays = [a for a in arrays if a.nbytes % 8]
    elif subset == "single":
        arrays = arrays[:1]
    elif subset == "empty_first":
        arrays = [arrays[5], arrays[1], arrays[6], arrays[2]]
    expected, expected_layout, expected_views = torch_staging(arrays)
    empty = np.empty

    def dirty(*args, **kwargs):
        # Uninitialized staging made visibly dirty: only the gaps may be zeroed.
        out = empty(*args, **kwargs)
        out.reshape(-1).view(np.uint8)[:] = 0xAB
        return out

    monkeypatch.setattr(np, "empty", dirty)
    host, layout, fields = pack_host(arrays)
    assert layout == expected_layout
    assert host.dtype == torch.uint8 and host.shape == expected.shape
    assert host.numpy().tobytes() == expected.numpy().tobytes()      # gap bytes included
    uploaded = upload_arrays(arrays, "cpu", packed=True)
    for array, view, reference in zip(arrays, uploaded, expected_views):
        assert view.dtype == reference.dtype and view.shape == reference.shape
        assert view.numpy().tobytes() == reference.numpy().tobytes()
        assert view.numpy().tobytes() == np.ascontiguousarray(array).tobytes()
    for plain, view in zip(upload_arrays(arrays, "cpu", packed=False), uploaded):
        assert plain.dtype == view.dtype and plain.shape == view.shape
        assert plain.numpy().tobytes() == view.numpy().tobytes()
    assert upload_arrays([], "cpu", packed=True) == ()
    assert [(dtype, shape) for dtype, shape in fields] == [(v.dtype, v.shape)
                                                          for v in expected_views]
