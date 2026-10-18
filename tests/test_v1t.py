"""eval.v1t: the pure DanZero-V1T player against recorded compiled-agent references.

``tests/fixtures/v1t_reference.json.gz`` holds four rounds (no tribute, single,
double and anti-tribute) recorded by ``python -m eval.v1t.verify --fixture``:
the applied actions and, at every decision, the compiled agent's exact model
input rows and Q values. Replaying the actions through ``gd`` must rebuild
those rows bit for bit anywhere (no DanLM needed). The Q check needs
onnxruntime and the model file (``V1T_ONNX`` or the DanLM checkout); the
direct comparison with DanLM's compiled encoder needs DanLM (macOS only).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys

import gd
import numpy as np
import pytest

from eval.danlm.arena import deal_from_json
from eval.history_policy import apply_and_observe
from eval.v1t import encoder as enc
from eval.v1t.player import V1TPlayer, action_identity, danlm_order_key, default_model_path

FIXTURE = Path(__file__).parent / "fixtures" / "v1t_reference.json.gz"
ROUNDS = json.loads(gzip.open(FIXTURE, "rt").read())["rounds"]


def dense(sparse: list, size: int) -> np.ndarray:
    out = np.zeros(size, np.float32)
    out[sparse[0]] = sparse[1]
    return out


class ZeroModel:
    """Stands in for the network when only the inputs are checked."""

    def __call__(self, rows: np.ndarray) -> np.ndarray:
        return np.zeros(len(rows), np.float32)


def real_model():
    pytest.importorskip("onnxruntime")
    if not default_model_path().exists():
        pytest.skip("V1T ONNX model not found (set V1T_ONNX)")
    from eval.v1t.player import V1TModel

    return V1TModel()


def replay(round_: dict, player: V1TPlayer):
    """Yield (player decision, reference decision) along the recorded round."""
    deal = deal_from_json(round_["deal"])
    engine = gd.Engine(gd.RuleConfig.house(), gd.ActionConfig.full())
    engine.auto_pass = False
    state = gd.MatchState()
    engine.set_deal(state, deal)
    player.begin_round(deal, state)
    player.start_match()
    for step in round_["steps"]:
        if "decision" in step:
            yield player.decide(state), step["decision"]
            continue
        kind, cards, key = step["action"]
        target = (kind, tuple(cards), key)
        action = next(a for a in engine.legal_actions(state) if action_identity(a) == target)
        apply_and_observe(engine, state, action, [player])
    assert state.phase == gd.Phase.RoundEnd


def test_fixture_covers_every_tribute_kind():
    assert sorted(r["tribute"] for r in ROUNDS) == ["anti", "double", "none", "single"]
    phases = {d["decision"]["phase"] for r in ROUNDS for d in r["steps"] if "decision" in d}
    assert phases == {"play", "give", "back"}


@pytest.mark.parametrize("index", range(len(ROUNDS)), ids=[r["tribute"] for r in ROUNDS])
def test_inputs_match_compiled_agent(index):
    player = V1TPlayer(ZeroModel())
    decisions = 0
    for mine, ref in replay(ROUNDS[index], player):
        decisions += 1
        state = dense(ref["state"], enc.DIM_STATE)
        assert np.array_equal(mine.rows[0, :enc.DIM_STATE], state), ref["phase"]
        theirs = {dense(a, enc.DIM_PLAY).tobytes() for a in ref["actions"]}
        ours = {row.tobytes() for row in mine.rows[:, enc.DIM_STATE:]}
        assert ours == theirs, ref["phase"]
    assert decisions > 20


@pytest.mark.parametrize("index", range(len(ROUNDS)), ids=[r["tribute"] for r in ROUNDS])
def test_q_values_and_choices_match_compiled_agent(index):
    player = V1TPlayer(real_model())
    worst = 0.0
    for mine, ref in replay(ROUNDS[index], player):
        q_ref = {dense(a, enc.DIM_PLAY).tobytes(): q for a, q in zip(ref["actions"], ref["q"])}
        q_mine = np.array([q_ref[row.tobytes()] for row in mine.rows[:, enc.DIM_STATE:]])
        worst = max(worst, float(np.max(np.abs(q_mine - mine.q))))
        # bitwise on the platform the fixture was recorded on (macOS arm64);
        # elsewhere the int8 kernels may round differently, hence the tolerance
        np.testing.assert_allclose(mine.q, q_mine, rtol=1e-4, atol=1e-4)
        assert np.array_equal(mine.vector, dense(ref["chosen"], enc.DIM_PLAY)), ref["phase"]
    print(f"max |dQ| = {worst:.3g} on {sys.platform}")


def test_order_key_lists_fewer_wilds_then_lexicographic():
    level = 5                                     # DanLM wild = card 5 (heart of rank 5)
    natural = enc.single_play(1)                  # H3, rank 1
    spade = enc.single_play(14)                   # S3
    declared = _single(5, 1)                      # the wild declared as a 3
    assert danlm_order_key(spade, level) < danlm_order_key(natural, level)
    assert danlm_order_key(natural, level) < danlm_order_key(declared, level)


def _single(card: int, rank: int) -> np.ndarray:
    row = enc.single_play(card)
    row[enc.DIM_CARDS + 11:] = 0
    row[enc.DIM_CARDS + 11 + rank] = 1
    return row


def test_compiled_encoder_agrees_on_random_states():
    """Direct comparison with DanLM's binary encoder (skipped without DanLM)."""
    from eval.danlm.arena import danlm_root

    root = str(danlm_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    pytest.importorskip("danzero.encoding.encoder_v1")
    from danzero.encoding import encoder_v1 as compiled
    from danzero.engine import cards, game

    rng = np.random.default_rng(3)
    checked = 0
    for r in range(4):
        rnd = game.GuanDanRound(level=int(rng.integers(2, 15)), hands=cards.deal_hands(seed=r),
                                first_player=int(rng.integers(4)))
        obs = rnd.get_observation()
        while obs is not None and not rnd.done:
            tribute = None if r % 2 else rng.integers(0, 2, (4, 54)).astype(np.float32)
            reference = compiled.encode_batch_v1t(obs, tribute)
            state = enc.play_state(obs.player, obs.hand, [obs.played_cards[s] for s in range(4)],
                                   obs.trick_actions, obs.finish_order, obs.round_level, tribute)
            assert np.array_equal(reference, enc.batch(state, obs.legal_plays))
            checked += 1
            obs = rnd.step(int(rng.integers(len(obs.legal_plays))), obs)
    hand = cards.deal_hands(seed=9)[0]
    other = 2 - hand.astype(np.int64)
    other[52:] -= 1
    for phase, receiver in (("give", 3), ("give", 0), ("back", 2)):
        tribute = rng.integers(0, 2, (4, 54)).astype(np.float32)
        reference = compiled.encode_tribute_state(hand, other, 9, phase, receiver, tribute)
        assert np.array_equal(reference, enc.tribute_state(hand, 9, phase, receiver, tribute))
    assert checked > 100
