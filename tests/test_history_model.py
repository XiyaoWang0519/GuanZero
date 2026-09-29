"""Adversarial tests for the T1 history player (train/history_model.py).

Each T1 acceptance criterion (docs/STAGE_C_TODO.md row T1, DESIGN.md 7.2,
7.3, 8.4) gets at least one test that tries to break it: fresh start without
disk access, every canonical candidate selectable, private queries kept out
of the public stream, forced/tribute data kept out of tokens, checkpoint
lineage markers, old evaluator checkpoints still loading, a disjoint critic
and gradients reaching the stream encoder.
"""
from dataclasses import asdict, replace
import inspect
from types import SimpleNamespace

import gd
import numpy as np
import pytest
import torch

from eval.policies import load_policy
from train import history_model as hm
from train.ckpt import load_checkpoint, save_checkpoint
from train.history_model import (DecisionInputs, HistoryActor, HistoryCritic, HistoryPolicyConfig,
                                 PublicStream, StreamBatch, checkpoint_payload, fresh_player,
                                 load_history_checkpoint, save_history_checkpoint)
from train.logs import TOKEN_DIM
from train.model import GuandanModel, ModelConfig

PLAY = int(gd.Phase.Play)
TRIBUTE_BITS = slice(4 + 146, 4 + 154)      # token bits 150..157
SMALL = dict(width=32, layers=2, heads=2, action_width=16, fusion_width=16,
             critic_width=16, critic_layers=1, max_rounds=4)


def small_config(**overrides) -> HistoryPolicyConfig:
    return HistoryPolicyConfig(**{**SMALL, **overrides})


# ---- rollout helper -------------------------------------------------------------

class Snapshot:
    """One pending batch of a VecEnv plus the public streams that precede it."""

    def __init__(self, streams, batch, prefix, match_ids=None):
        self.streams = streams                          # list of arrays() tuples per env
        self.match_ids = list(match_ids or [])
        self.obs = np.array(batch.obs, dtype=np.float32, copy=True)
        self.cand = np.array(batch.cand, dtype=np.float32, copy=True)
        self.offsets = np.asarray(batch.offsets, dtype=np.int64).copy()
        self.seat = np.asarray(batch.seat, dtype=np.int64).copy()
        self.env_id = np.asarray(batch.env_id, dtype=np.int64).copy()
        self.phase = np.asarray(batch.phase, dtype=np.int64).copy()
        self.hidden = np.array(batch.hidden_counts, dtype=np.uint8, copy=True)
        self.prefix = np.asarray(prefix, dtype=np.int64)

    @property
    def counts(self) -> np.ndarray:
        return np.diff(self.offsets)

    def inputs(self, device="cpu", streams=None, obs=None, seat=None) -> DecisionInputs:
        arrays = self.streams if streams is None else streams
        return DecisionInputs(
            streams=StreamBatch.from_arrays(arrays, device),
            match_index=torch.tensor(self.env_id, device=device),
            prefix=torch.tensor(self.prefix, device=device),
            obs=torch.tensor(self.obs if obs is None else obs, device=device),
            seat=torch.tensor(self.seat if seat is None else seat, device=device),
            cand=torch.tensor(self.cand, device=device),
            offsets=torch.tensor(self.offsets, device=device))


class Rollout:
    """Greedy self-play of a few envs with public-event logging and per-env streams.

    ``snapshots`` holds the pending batch at a few steps; ``events`` every
    drained ``PublicActionEvent``; ``step0`` is the first pending batch (all
    prefixes 0, opening leads with hundreds of candidates).
    """

    def __init__(self, num_envs=4, steps=220, seed=3, capture=(0, 9, 60, 219)):
        env = gd.VecEnv(num_envs, num_threads=1, seed=seed, log_public_actions=True)
        env.reset()
        streams = [PublicStream() for _ in range(num_envs)]
        self.events = []
        self.snapshots = {}
        self.consistency_checks = 0
        self.widest_count = 0
        self.widest = None
        pending_check = None
        for step in range(steps):
            batch = env.pending()
            drained = env.drain_public_actions()
            self.events.extend(drained)
            for event in drained:
                stream = streams[int(event.env_id)]
                if stream.match_id != int(event.match_id):
                    stream.reset(int(event.match_id))
                stream.append(event)
            if pending_check is not None:
                # The previous step's chosen candidate must be the token at its prefix.
                for env_id, prefix, seat, action in pending_check:
                    stream = streams[env_id]
                    assert stream.prefix > prefix, "no event followed a decision"
                    token = stream.tokens[prefix]
                    assert int(token[:4].argmax()) == seat and token[:4].sum() == 1
                    np.testing.assert_array_equal(token[4:150], action[:146])
                    self.consistency_checks += 1
            prefix = []
            for row in range(len(batch.seat)):
                stream = streams[int(batch.env_id[row])]
                if stream.match_id != int(batch.match_id[row]):
                    stream.reset(int(batch.match_id[row]))
                prefix.append(stream.prefix)
            match_ids = [s.match_id for s in streams]
            if step in capture:
                self.snapshots[step] = Snapshot([s.arrays() for s in streams], batch, prefix, match_ids)
            widest = int(np.diff(np.asarray(batch.offsets)).max())
            if widest > self.widest_count:
                self.widest_count = widest
                self.widest = Snapshot([s.arrays() for s in streams], batch, prefix, match_ids)
            choice = np.asarray(batch.greedy_choice, dtype=np.int64)
            offsets = np.asarray(batch.offsets, dtype=np.int64)
            pending_check = [(int(batch.env_id[r]), prefix[r], int(batch.seat[r]),
                              np.array(batch.cand[offsets[r] + choice[r]], copy=True))
                             for r in range(len(batch.seat))]
            env.step(np.asarray(choice, dtype=np.int32))
        self.step0 = self.snapshots[min(capture)]
        self.last = self.snapshots[max(capture)]
        self.middle = self.snapshots[sorted(capture)[-2]]
        self.num_envs = num_envs

    def middle_with_later_streams(self) -> Snapshot:
        """The middle decisions read against the later streams of the same matches,
        so real tokens exist after every prefix."""
        assert self.middle.match_ids == self.last.match_ids, "a match restarted between snapshots"
        snap = Snapshot.__new__(Snapshot)
        snap.__dict__.update(self.middle.__dict__)
        snap.streams = self.last.streams
        for t_old, t_new in zip(self.middle.streams, self.last.streams):
            np.testing.assert_array_equal(t_old[0], t_new[0][:len(t_old[0])])
        return snap


@pytest.fixture(scope="module")
def rollout() -> Rollout:
    roll = Rollout()
    assert roll.consistency_checks > 100
    assert any(int(e.phase) != PLAY for e in roll.events), "no tribute-phase event was played"
    assert int(roll.step0.counts.max()) > 32 and (roll.step0.prefix == 0).all()
    assert roll.widest_count > 100, "some lead should offer hundreds of candidates"
    assert int(roll.last.prefix.max()) > 100
    return roll


def actor_in(mode: str, config=None, seed=0, sharp: bool = False) -> HistoryActor:
    """A fresh actor; ``sharp`` scales the fusion head so that state changes
    move the log-probabilities (at random init the head is nearly flat)."""
    actor, _ = fresh_player(config or small_config(), seed)
    if sharp:
        with torch.no_grad():
            actor.fusion[-1].weight.mul_(50)
    return actor.train() if mode == "train" else actor.eval()


def log_probs_and_states(actor: HistoryActor, inputs: DecisionInputs):
    with torch.no_grad():
        state = actor.decision_states(None, inputs)
        logits = actor.candidate_logits(state, inputs.cand, inputs.offsets)
        return state, hm.segment_log_softmax(logits, inputs.rows, inputs.decisions)


def report_diff(a: torch.Tensor, b: torch.Tensor) -> str:
    return f"max abs diff {float((a - b).abs().max()):.3e}"


# ---- 1. fresh initialisation --------------------------------------------------------

def test_fresh_player_never_touches_disk_and_is_seed_deterministic(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("fresh_player must not load anything from disk")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr("train.ckpt.load_checkpoint", forbidden)
    monkeypatch.setattr("train.history_model.load_checkpoint", forbidden)
    config = small_config()
    a1, c1 = fresh_player(config, 11)
    a2, c2 = fresh_player(config, 11)
    a3, c3 = fresh_player(config, 12)
    for m1, m2, m3 in ((a1, a2, a3), (c1, c2, c3)):
        s1, s2, s3 = m1.state_dict(), m2.state_dict(), m3.state_dict()
        assert s1.keys() == s2.keys() == s3.keys()
        for key in s1:
            assert torch.equal(s1[key], s2[key]), key
        assert any(s1[k].dtype.is_floating_point and not torch.equal(s1[k], s3[k]) for k in s1)
    assert isinstance(a1, HistoryActor) and isinstance(c1, HistoryCritic)


# ---- 2. every canonical candidate selectable ------------------------------------------

@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("step", ["first", "widest", "last"])
def test_candidate_log_probs_are_finite_and_normalised_on_real_batches(rollout, mode, step):
    snap = {"first": rollout.step0, "widest": rollout.widest, "last": rollout.last}[step]
    actor = actor_in(mode)
    inputs = snap.inputs()
    with torch.no_grad():
        log_probs = actor.candidate_log_probs(inputs)
    assert log_probs.shape == (int(snap.offsets[-1]),)
    assert torch.isfinite(log_probs).all()
    total = torch.zeros(snap.offsets.size - 1).index_add_(0, inputs.rows, log_probs.exp())
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)
    if step != "last":
        assert int(inputs.counts.max()) > 32


@pytest.mark.parametrize("step", ["first", "widest"])
def test_sampling_reaches_beyond_thirty_two_and_never_past_the_count(rollout, step):
    snap = rollout.step0 if step == "first" else rollout.widest
    actor = actor_in("eval")
    inputs = snap.inputs()
    counts = torch.as_tensor(snap.counts)
    generator = torch.Generator().manual_seed(0)
    choices = torch.stack([actor.act(inputs, generator=generator)[0] for _ in range(96)])
    assert (choices < counts[None]).all()
    assert (choices >= 0).all()
    wide = counts > 32
    assert wide.any()
    assert int(choices[:, wide].max()) > 32, "candidates past index 32 must be reachable"
    # Every decision with more than one candidate should see more than one distinct choice.
    for i in torch.nonzero(counts > 1)[:, 0].tolist():
        assert len(set(choices[:, i].tolist())) > 1


@pytest.mark.parametrize("greedy", [False, True])
def test_act_returns_the_chosen_candidates_own_log_prob(rollout, greedy):
    snap = rollout.last
    actor = actor_in("eval")
    inputs = snap.inputs()
    choice, chosen = actor.act(inputs, generator=torch.Generator().manual_seed(1), greedy=greedy)
    with torch.no_grad():
        log_probs = actor.candidate_log_probs(inputs)
    offsets = torch.as_tensor(snap.offsets)
    assert torch.equal(chosen, log_probs[offsets[:-1] + choice])
    if greedy:
        table = torch.full((snap.offsets.size - 1, int(snap.counts.max())), float("-inf"))
        table[inputs.rows, torch.arange(len(log_probs)) - offsets[:-1][inputs.rows]] = log_probs
        assert torch.equal(choice, table.argmax(1))


def test_actor_has_no_top_k_reference_or_pruning_machinery():
    actor = actor_in("train")
    names = set(dir(actor)) | set(actor.state_dict().keys())
    names |= {n for n, _ in actor.named_modules()} | {n for n, _ in actor.named_buffers()}
    names |= set(vars(actor)) | set(asdict(actor.config))
    bad = [n for n in names if any(s in n.lower() for s in ("top_k", "topk", "reference", "prune"))]
    assert not bad, bad
    assert not hasattr(actor, "reference") and not hasattr(actor, "top_k")


# ---- 3. private queries never enter the public stream ----------------------------------

def test_encode_stream_takes_public_tokens_only():
    params = list(inspect.signature(HistoryActor.encode_stream).parameters)
    assert params == ["self", "tokens", "rounds", "phases", "lengths"]
    assert not any(p in {"obs", "observation", "seat", "cand", "hidden"} for p in params)


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_public_encoding_is_bit_identical_under_private_changes(rollout, mode):
    snap = rollout.last
    actor = actor_in(mode)
    inputs = snap.inputs()
    with torch.no_grad():
        base = actor.encode_batch(inputs.streams)
        rng = np.random.default_rng(0)
        other = snap.inputs(obs=rng.integers(0, 2, snap.obs.shape).astype(np.float32),
                            seat=(snap.seat + 1) % 4)
        again = actor.encode_batch(other.streams)
    assert torch.equal(base, again)
    # And the private query of one decision cannot reach another decision.
    states, log_probs = log_probs_and_states(actor, inputs)
    states2, log_probs2 = log_probs_and_states(actor, other)
    assert not torch.equal(states, states2)
    victim = 0
    partial_obs = snap.obs.copy()
    partial_obs[1:] = other.obs.numpy()[1:]
    partial_seat = snap.seat.copy()
    partial_seat[1:] = other.seat.numpy()[1:]
    mixed = snap.inputs(obs=partial_obs, seat=partial_seat)
    states3, log_probs3 = log_probs_and_states(actor, mixed)
    assert torch.equal(states[victim], states3[victim]), report_diff(states[victim], states3[victim])
    lo, hi = int(snap.offsets[victim]), int(snap.offsets[victim + 1])
    assert torch.equal(log_probs[lo:hi], log_probs3[lo:hi])
    # A precomputed encoding from a batch with different private data is the same encoding.
    with torch.no_grad():
        via_other = actor.decision_states(actor.encode_batch(other.streams), inputs)
    assert torch.equal(states, via_other)


def flip_token(arrays, env, index, rng):
    """A copy of ``arrays`` whose token ``index`` of env ``env`` is a different valid token."""
    arrays = [(t.copy(), r.copy(), p.copy()) for t, r, p in arrays]
    tokens = arrays[env][0]
    old = tokens[index].copy()
    new = np.zeros(TOKEN_DIM, np.uint8)
    new[(int(old[:4].argmax()) + 1) % 4] = 1
    new[4 + rng.integers(0, 108)] = 1
    new[158 + (int(old[158:].argmax()) + 3) % 28] = 1
    assert not np.array_equal(new, old)
    tokens[index] = new
    return arrays


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_decisions_cannot_read_tokens_at_or_after_their_prefix(rollout, mode):
    snap = rollout.middle_with_later_streams()
    actor = actor_in(mode, sharp=True)
    states, log_probs = log_probs_and_states(actor, snap.inputs())
    rng = np.random.default_rng(1)
    lengths = np.array([len(t) for t, _, _ in snap.streams])
    assert (snap.prefix < lengths[snap.env_id]).all(), "every decision has real tokens after it"
    checked = log_prob_moves = 0
    for i in range(len(snap.prefix)):
        env, prefix = int(snap.env_id[i]), int(snap.prefix[i])
        if prefix < 2 or prefix >= lengths[env]:
            continue
        lo, hi = int(snap.offsets[i]), int(snap.offsets[i + 1])
        for future in (prefix, lengths[env] - 1):
            arrays = flip_token(snap.streams, env, future, rng)
            s2, lp2 = log_probs_and_states(actor, snap.inputs(streams=arrays))
            assert torch.equal(states[i], s2[i]), \
                f"token {future} leaked into decision {i} (prefix {prefix}): " + \
                report_diff(states[i], s2[i])
            assert torch.equal(log_probs[lo:hi], lp2[lo:hi])
        for past in (0, prefix - 1):
            arrays = flip_token(snap.streams, env, past, rng)
            s2, lp2 = log_probs_and_states(actor, snap.inputs(streams=arrays))
            assert not torch.equal(states[i], s2[i]), f"token {past} did not reach decision {i}"
            log_prob_moves += not torch.equal(log_probs[lo:hi], lp2[lo:hi])
        checked += 1
    assert checked > 0
    # A tiny ReLU fusion head can leave one decision's few logits untouched by a
    # state change, so the log-probability response is asserted in aggregate.
    assert log_prob_moves > checked


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_encoded_positions_are_causal_in_the_stream_itself(rollout, mode):
    snap = rollout.last
    actor = actor_in(mode)
    env = int(np.argmax([len(t) for t, _, _ in snap.streams]))
    n = len(snap.streams[env][0])
    assert n > 10
    with torch.no_grad():
        base = actor.encode_batch(StreamBatch.from_arrays(snap.streams, "cpu"))
        flipped = flip_token(snap.streams, env, 5, np.random.default_rng(2))
        again = actor.encode_batch(StreamBatch.from_arrays(flipped, "cpu"))
    # Position s has seen tokens 0..s-1; token 5 sits at position 6.
    assert torch.equal(base[env, :6], again[env, :6]), report_diff(base[env, :6], again[env, :6])
    assert not torch.equal(base[env, 6:n + 1], again[env, 6:n + 1])
    for other in range(len(snap.streams)):
        if other != env:
            m = len(snap.streams[other][0]) + 1
            assert torch.equal(base[other, :m], again[other, :m])


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_padding_does_not_change_any_match(rollout, mode):
    snap = rollout.last
    actor = actor_in(mode)
    lengths = np.array([len(t) for t, _, _ in snap.streams])
    longest, shortest = int(lengths.argmax()), int(lengths.argmin())
    assert lengths[longest] > lengths[shortest]
    full_states, full_log_probs = log_probs_and_states(actor, snap.inputs())
    for env in (longest, shortest):
        rows = np.nonzero(snap.env_id == env)[0]
        if not len(rows):
            continue
        solo = Snapshot.__new__(Snapshot)
        solo.streams = [snap.streams[env]]
        solo.obs, solo.seat, solo.hidden = snap.obs[rows], snap.seat[rows], snap.hidden[rows]
        solo.prefix, solo.phase = snap.prefix[rows], snap.phase[rows]
        solo.env_id = np.zeros(len(rows), np.int64)
        keep = np.concatenate([np.arange(snap.offsets[r], snap.offsets[r + 1]) for r in rows])
        solo.cand = snap.cand[keep]
        solo.offsets = np.concatenate(([0], np.cumsum(snap.counts[rows]))).astype(np.int64)
        inputs = solo.inputs()
        assert inputs.streams.tokens.shape[1] == lengths[env]
        states, log_probs = log_probs_and_states(actor, inputs)
        assert torch.allclose(states, full_states[rows], atol=1e-5), \
            f"env {env}: " + report_diff(states, full_states[rows])
        assert torch.allclose(log_probs, full_log_probs[keep], atol=1e-5)


def window_snapshot(rollout, k):
    """Middle decisions read against the later streams, keeping prefixes above k + 1
    so that tokens outside the window exist on both sides."""
    snap = rollout.middle_with_later_streams()
    keep = np.nonzero(snap.prefix > k + 1)[0]
    assert len(keep) > 0
    return snap, keep


@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("k", [1, 3])
def test_window_sees_exactly_the_last_k_tokens(rollout, mode, k):
    actor = actor_in(mode, small_config(window=k), sharp=True)
    snap, keep = window_snapshot(rollout, k)
    states, log_probs = log_probs_and_states(actor, snap.inputs())
    rng = np.random.default_rng(3)
    future_checks = log_prob_moves = 0
    for i in keep.tolist():
        env, prefix = int(snap.env_id[i]), int(snap.prefix[i])
        lo, hi = int(snap.offsets[i]), int(snap.offsets[i + 1])
        for outside in (0, prefix - k - 1, prefix, len(snap.streams[env][0]) - 1):
            future_checks += outside >= prefix
            s2, lp2 = log_probs_and_states(actor, snap.inputs(streams=flip_token(snap.streams, env, outside, rng)))
            assert torch.equal(states[i], s2[i]), \
                f"window {k}: token {outside} reached decision {i} (prefix {prefix}): " + \
                report_diff(states[i], s2[i])
            assert torch.equal(log_probs[lo:hi], lp2[lo:hi])
        for inside in (prefix - k, prefix - 1):
            s2, lp2 = log_probs_and_states(actor, snap.inputs(streams=flip_token(snap.streams, env, inside, rng)))
            assert not torch.equal(states[i], s2[i]), f"window {k}: token {inside} ignored"
            log_prob_moves += not torch.equal(log_probs[lo:hi], lp2[lo:hi])
    assert future_checks >= 2 * len(keep)
    assert log_prob_moves > len(keep)


@pytest.mark.parametrize("window", [0, 3])
def test_prefix_zero_inside_a_long_stream_sees_only_bos(rollout, window):
    """A new match's first decision reads nothing, however long the buffer is."""
    snap = rollout.last
    actor = actor_in("train", small_config(window=window))
    zero = Snapshot.__new__(Snapshot)
    zero.__dict__.update(snap.__dict__)
    zero.prefix = np.zeros_like(snap.prefix)
    with_history, lp_history = log_probs_and_states(actor, zero.inputs())
    empty = [(np.zeros((0, TOKEN_DIM), np.uint8), np.zeros(0, np.int64), np.zeros(0, np.int64))
             for _ in snap.streams]
    without, lp_empty = log_probs_and_states(actor, zero.inputs(streams=empty))
    assert torch.allclose(with_history, without, atol=1e-6), report_diff(with_history, without)
    assert torch.allclose(lp_history, lp_empty, atol=1e-6)
    full_states, _ = log_probs_and_states(actor, snap.inputs())
    assert not torch.equal(full_states, with_history)


def test_eval_and_train_modes_agree_on_behaviour_log_probs(rollout):
    """PPO recomputes the log-probs act() stored; the two code paths must agree."""
    snap = rollout.last
    actor = actor_in("eval")
    choice, stored = actor.act(snap.inputs(), greedy=True)
    actor.train()
    recomputed = actor.candidate_log_probs(snap.inputs()).detach()
    offsets = torch.as_tensor(snap.offsets)
    assert torch.allclose(stored, recomputed[offsets[:-1] + choice], atol=1e-5), \
        report_diff(stored, recomputed[offsets[:-1] + choice])


def test_windowed_actor_matches_full_actor_on_short_prefixes(rollout):
    """A window at least as long as every prefix is the full-history actor."""
    snap = rollout.snapshots[9]
    longest = int(snap.prefix.max())
    assert 0 < longest <= 12
    config = small_config()
    full = actor_in("train", config)
    windowed = HistoryActor(small_config(window=longest)).train()
    windowed.load_state_dict(full.state_dict())
    s1, lp1 = log_probs_and_states(full, snap.inputs())
    s2, lp2 = log_probs_and_states(windowed, snap.inputs())
    assert torch.allclose(lp1, lp2, atol=1e-5), report_diff(lp1, lp2)


# ---- 4. forced flag and private data cannot enter -----------------------------------------

def valid_token(seat=0, action_bit=7, cards_left=20) -> np.ndarray:
    token = np.zeros(TOKEN_DIM, np.uint8)
    token[seat] = 1
    token[4 + action_bit] = 1
    token[158 + cards_left] = 1
    return token


def test_append_token_rejects_tribute_flags_and_malformed_tokens():
    stream = PublicStream()
    stream.append_token(valid_token(), 0, PLAY)
    assert stream.prefix == 1
    for bit in range(150, 158):
        token = valid_token()
        token[bit] = 1
        with pytest.raises(ValueError):
            stream.append_token(token, 0, PLAY)
    two_seats = valid_token()
    two_seats[1] = 1
    no_seat = valid_token()
    no_seat[0] = 0
    two_counts = valid_token()
    two_counts[158 + 3] = 1
    for bad in (two_seats, no_seat, two_counts, valid_token()[:-1], np.zeros(TOKEN_DIM + 1, np.uint8)):
        with pytest.raises(ValueError):
            stream.append_token(bad, 0, PLAY)
    with pytest.raises(ValueError):
        stream.append_token(valid_token(), -1, PLAY)
    with pytest.raises(ValueError):
        stream.append_token(valid_token(), 0, -1)
    assert stream.prefix == 1


def test_stream_stores_no_forced_flag_and_forced_passes_are_indistinguishable():
    assert not any("forced" in name for name in PublicStream.__slots__)
    assert not hasattr(PublicStream(), "__dict__")
    encoded = np.zeros(gd.ACT_DIM, np.float32)
    encoded[0] = 1
    make = lambda forced: SimpleNamespace(seat=2, encoded_action=encoded, cards_left=9,
                                          round_index=1, phase=PLAY, forced=forced)
    voluntary, forced = PublicStream(), PublicStream()
    voluntary.append(make(False))
    forced.append(make(True))
    np.testing.assert_array_equal(voluntary.tokens[0], forced.tokens[0])
    assert voluntary.arrays()[1].tolist() == [1] and voluntary.arrays()[2].tolist() == [PLAY]


def test_append_strips_private_tribute_flags_of_older_engines():
    encoded = np.zeros(gd.ACT_DIM, np.float32)
    encoded[3] = 1
    encoded[146:154] = 1
    stream = PublicStream()
    stream.append(SimpleNamespace(seat=1, encoded_action=encoded, cards_left=27, round_index=0,
                                  phase=int(gd.Phase.Tribute)))
    assert not stream.tokens[0][TRIBUTE_BITS].any()
    assert stream.tokens[0][4 + 3] == 1


def test_real_events_including_tribute_phases_enter_without_private_bits(rollout):
    streams = {}
    seen_phases = set()
    for event in rollout.events:
        stream = streams.setdefault((int(event.env_id), int(event.match_id)), PublicStream(event.match_id))
        stream.append(event)
        seen_phases.add(int(event.phase))
    assert {int(gd.Phase.Tribute), int(gd.Phase.BackTribute), PLAY} <= seen_phases
    for stream in streams.values():
        tokens, rounds, phases = stream.arrays()
        assert tokens.shape == (stream.prefix, TOKEN_DIM) and tokens.dtype == np.uint8
        assert not tokens[:, TRIBUTE_BITS].any()
        assert (tokens[:, :4].sum(1) == 1).all() and (tokens[:, 158:].sum(1) == 1).all()
        assert (np.diff(rounds) >= 0).all()
        tribute_rows = np.nonzero(phases != PLAY)[0]
        if len(tribute_rows):
            assert rounds[tribute_rows].min() >= 1      # no tribute in the first round
    assert any((s.arrays()[2] != PLAY).any() for s in streams.values())


def test_stream_batch_pads_and_records_lengths():
    a, b = PublicStream(), PublicStream()
    for i in range(3):
        a.append_token(valid_token(seat=i, action_bit=i), 0, PLAY)
    b.append_token(valid_token(), 1, int(gd.Phase.Tribute))
    batch = StreamBatch.from_streams([a, b, PublicStream()], "cpu")
    assert batch.tokens.shape == (3, 3, TOKEN_DIM) and batch.tokens.dtype == torch.uint8
    assert batch.lengths.tolist() == [3, 1, 0]
    assert batch.phases[1, 0] == int(gd.Phase.Tribute) and batch.rounds[1, 0] == 1
    empty = StreamBatch.from_streams([PublicStream()], "cpu")
    assert empty.tokens.shape == (1, 0, TOKEN_DIM)



def test_stream_storage_grows_contiguously_and_views_stay_valid():
    import copy
    stream = PublicStream(3)
    empty = stream.arrays()
    assert empty[0].shape == (0, TOKEN_DIM) and empty[0].dtype == np.uint8
    assert empty[1].shape == empty[2].shape == (0,) and empty[1].dtype == empty[2].dtype == np.int64
    expected = []
    early = None
    for i in range(200):                      # crosses several doublings
        token = valid_token(seat=i % 4, action_bit=i % 90, cards_left=27 - i % 28)
        stream.append_token(token, i // 30, i % 4)
        expected.append((token, i // 30, i % 4))
        if i == 9:
            early = stream.arrays()
    tokens, rounds, phases = stream.arrays()
    assert stream.prefix == len(stream.tokens) == 200 and tokens.shape == (200, TOKEN_DIM)
    np.testing.assert_array_equal(tokens, np.stack([t for t, _, _ in expected]))
    assert rounds.tolist() == [r for _, r, _ in expected]
    assert phases.tolist() == [p for _, _, p in expected]
    # Views, not copies; row and slice access as the list-backed stream allowed.
    assert np.shares_memory(tokens, stream.tokens) and np.shares_memory(rounds, stream.rounds)
    np.testing.assert_array_equal(stream.tokens[5], expected[5][0])
    np.testing.assert_array_equal(stream.tokens[3:7], tokens[3:7])
    assert stream.rounds[199] == 6 and stream.phases[-1] == 3
    # An earlier view survives growth; a reset starts new storage.
    np.testing.assert_array_equal(early[0], tokens[:10])
    duplicate = copy.deepcopy(stream)
    generation = stream.generation
    stream.reset(4)
    assert stream.prefix == 0 and stream.generation == generation + 1 and stream.match_id == 4
    stream.append_token(valid_token(seat=2), 0, PLAY)
    np.testing.assert_array_equal(tokens[0], expected[0][0])
    np.testing.assert_array_equal(duplicate.tokens, tokens)
    assert duplicate.prefix == 200 and stream.prefix == 1 and stream.tokens[0][2] == 1
    # The appended token is copied in; changing the caller's array changes nothing.
    token = valid_token(seat=1)
    stream.append_token(token, 0, PLAY)
    token[:] = 0
    assert stream.tokens[1][1] == 1

# ---- 5. checkpoint marker -------------------------------------------------------------

def test_history_checkpoint_round_trips_exactly(tmp_path):
    config = small_config(window=2)
    actor, critic = fresh_player(config, 5)
    payload = checkpoint_payload(actor, critic, lineage="test", seed=5, progress={"step": 3})
    assert payload["stage"] == "history_ppo" and payload["init"] == "random"
    assert payload["teacher"] is None and payload["token_schema"] == hm.TOKEN_SCHEMA_VERSION
    assert payload["config"]["stage"] == "history_ppo"
    path = tmp_path / "history.pt"
    save_history_checkpoint(path, payload)
    actor2, critic2, loaded = load_history_checkpoint(path)
    assert actor2.config == config and critic2.config == config
    for fresh, back in ((actor, actor2), (critic, critic2)):
        s1, s2 = fresh.state_dict(), back.state_dict()
        assert s1.keys() == s2.keys()
        for key in s1:
            assert torch.equal(s1[key], s2[key]), key
    assert loaded["progress"] == {"step": 3} and loaded["seed"] == 5
    assert set(loaded) >= {"model_config", "model", "optimizer", "config", "progress", "rng"}


def old_style_payload(stage: str | None) -> dict:
    torch.manual_seed(5)
    config = ModelConfig(state_width=8, state_layers=1, action_width=8,
                         action_layers=1, fusion_width=8, fusion_layers=1)
    payload = {"model_config": asdict(config), "model": GuandanModel(config).state_dict(),
               "optimizer": {}, "config": {"seed": 7, "action_mode": "canonical"},
               "progress": {}, "rng": {}}
    if stage is not None:
        payload["stage"] = stage
    return payload


@pytest.mark.parametrize("stage", [None, "dmc", "ppo", "a2"])
def test_loading_refuses_old_stages(tmp_path, stage):
    path = tmp_path / "old.pt"
    save_checkpoint(path, old_style_payload(stage))
    with pytest.raises(ValueError, match="history_ppo"):
        load_history_checkpoint(path)
    with pytest.raises(ValueError):
        save_history_checkpoint(tmp_path / "again.pt", old_style_payload(stage))
    assert not (tmp_path / "again.pt").exists()


@pytest.mark.parametrize("mutation", [
    {"init": "ppo"}, {"init": "dmc_warm_start"}, {"teacher": "runs/m1/final.pt"},
    {"teacher": {"path": "x"}}, {"token_schema": hm.TOKEN_SCHEMA_VERSION + 1},
    {"token_schema": None}, {"stage": "ppo"}, {"stage": None},
])
def test_loading_refuses_forged_lineage_markers(tmp_path, mutation):
    actor, critic = fresh_player(small_config(), 1)
    payload = {**checkpoint_payload(actor, critic, lineage="x", seed=1), **mutation}
    path = tmp_path / "forged.pt"
    save_checkpoint(path, payload)
    with pytest.raises(ValueError):
        load_history_checkpoint(path)
    if mutation.get("stage", "history_ppo") != "history_ppo":
        with pytest.raises(ValueError):
            save_history_checkpoint(tmp_path / "x.pt", payload)


def test_payload_refuses_mismatched_actor_and_critic():
    actor, _ = fresh_player(small_config(), 1)
    _, critic = fresh_player(small_config(critic_width=8), 1)
    with pytest.raises(ValueError):
        checkpoint_payload(actor, critic, lineage="x", seed=1)


# ---- 6. old evaluator checkpoints still load --------------------------------------------

def test_old_mlp_checkpoint_still_loads_in_the_evaluator(tmp_path):
    path = tmp_path / "frozen.pt"
    save_checkpoint(path, old_style_payload(None))
    policy = load_policy(str(path))
    assert policy is not None
    assert load_checkpoint(path).get("stage", "dmc") == "dmc"


@pytest.mark.parametrize("prefix", ["", "sample:"])
def test_evaluator_never_loads_a_history_checkpoint_as_an_mlp_policy(tmp_path, prefix):
    """Before T3 the evaluator refuses the stage marker explicitly; with T3 it
    returns a history-aware policy. Silent acceptance as an MLP policy is the
    failure mode this guards against."""
    actor, critic = fresh_player(small_config(), 2)
    path = tmp_path / "history.pt"
    save_history_checkpoint(path, checkpoint_payload(actor, critic, lineage="x", seed=2))
    try:
        policy = load_policy(f"{prefix}{path}")
    except ValueError as error:
        assert "stage" in str(error)
    else:
        assert getattr(policy, "needs_history", False) is True, type(policy)
        assert "history" in type(policy).__name__.lower()
        assert not hasattr(policy, "reference") and not hasattr(policy, "top_k")


@pytest.mark.parametrize("bad", [
    dict(width=0), dict(layers=0), dict(heads=0), dict(width=30, heads=4), dict(window=-1),
    dict(max_rounds=0), dict(action_width=0), dict(fusion_width=0), dict(critic_width=0),
    dict(critic_layers=0), dict(obs_dim=int(gd.OBS_DIM) + 1), dict(act_dim=int(gd.ACT_DIM) - 1),
])
def test_config_rejects_bad_sizes(bad):
    with pytest.raises(ValueError):
        small_config(**bad)


def test_config_accepts_windows_and_survives_asdict_round_trip():
    config = small_config(window=5)
    assert HistoryPolicyConfig(**asdict(config)) == config
    assert replace(config, window=0).window == 0


# ---- 7. critic ---------------------------------------------------------------------------

def test_critic_reads_privileged_state_and_shares_nothing_with_the_actor(rollout):
    snap = rollout.last
    actor, critic = fresh_player(small_config(), 3)
    first = next(m for m in critic.modules() if isinstance(m, torch.nn.Linear))
    assert first.in_features == int(gd.OBS_DIM) + 3 * 54
    value = critic(torch.as_tensor(snap.obs), torch.as_tensor(snap.hidden))
    assert value.shape == (len(snap.obs),) and torch.isfinite(value).all()
    actor_ids = {id(p) for p in actor.parameters()}
    critic_ids = {id(p) for p in critic.parameters()}
    assert actor_ids.isdisjoint(critic_ids) and critic_ids
    actor_data = {p.data_ptr() for p in actor.parameters()}
    assert actor_data.isdisjoint({p.data_ptr() for p in critic.parameters()})
    _, other = fresh_player(small_config(), 4)
    assert any(not torch.equal(a, b) for a, b in zip(critic.state_dict().values(),
                                                     other.state_dict().values()))
    # The hidden counts must matter: they are the critic's privileged input.
    flipped = torch.as_tensor(snap.hidden).clone()
    flipped[:, 0, :] = 3
    assert not torch.equal(value, critic(torch.as_tensor(snap.obs), flipped))
    with pytest.raises((RuntimeError, ValueError)):
        critic(torch.as_tensor(snap.obs), torch.zeros(len(snap.obs), 3, 53))


# ---- 8. gradients ------------------------------------------------------------------------

@pytest.mark.parametrize("window", [0, 3])
def test_loss_reaches_stream_encoder_private_tower_and_fusion(rollout, window):
    snap = rollout.last
    actor, _ = fresh_player(small_config(window=window), 6)
    actor.train()
    inputs = snap.inputs()
    log_probs = actor.candidate_log_probs(inputs)
    rows = inputs.rows
    target = torch.zeros(inputs.decisions, dtype=torch.long)
    picked = log_probs[torch.as_tensor(snap.offsets[:-1]) + target]
    loss = -(picked * torch.linspace(0.5, 1.5, len(picked))).mean()
    loss.backward()

    def has_grad(module: torch.nn.Module) -> bool:
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None for g in grads), f"{module.__class__.__name__} has None grads"
        return any(float(g.abs().max()) > 0 for g in grads)

    assert has_grad(actor.stream)
    assert has_grad(actor.public)
    assert has_grad(actor.round_embedding)
    assert has_grad(actor.private)
    assert has_grad(actor.seat)
    assert has_grad(actor.fusion)
    assert has_grad(actor.action_tower)
    assert actor.bos.grad is not None and float(actor.bos.grad.abs().max()) > 0
    assert rows.max() == inputs.decisions - 1


def test_segment_log_softmax_matches_dense_log_softmax():
    torch.manual_seed(0)
    counts = torch.tensor([1, 5, 400, 2])
    scores = torch.randn(int(counts.sum())) * 30
    rows = torch.repeat_interleave(torch.arange(4), counts)
    out = hm.segment_log_softmax(scores, rows, 4)
    start = 0
    for c in counts.tolist():
        expected = torch.log_softmax(scores[start:start + c], 0)
        assert torch.allclose(out[start:start + c], expected, atol=1e-5)
        start += c
