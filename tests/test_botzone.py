"""Botzone bot: protocol conversions, NumPy actor parity, round reconstruction.

The end-to-end test plays games through the local judge and checks that the
bot's play at every decision equals the torch ``HistoryPolicy`` run on the
referee's true state and stream.
"""
import json
import random

import gd
import numpy as np
import pytest
import torch

from eval.botzone import judge, mirror
from eval.botzone.bot import Bot
from eval.botzone.export import export
from eval.botzone.numpy_actor import NumpyHistoryActor
from eval.botzone.protocol import (RoundLog, card_to_gd, claim_faces, claim_matches,
                                   face_to_bz, level_to_gd, physical_claim, reading_ranks)
from train.history_model import (HistoryPolicyConfig, PublicStream, StreamBatch, DecisionInputs,
                                 checkpoint_payload, fresh_player, save_history_checkpoint)

SMALL = HistoryPolicyConfig(width=32, layers=2, heads=4, action_width=24, fusion_width=24,
                            critic_width=16, critic_layers=1, response_mode="auxiliary",
                            aux_heads="next,belief,outcome")


@pytest.fixture(scope="module")
def player(tmp_path_factory):
    torch.set_num_threads(1)
    actor, critic = fresh_player(SMALL, seed=11)
    root = tmp_path_factory.mktemp("botzone")
    path = root / "small.pt"
    save_history_checkpoint(path, checkpoint_payload(actor, critic, lineage="test", seed=11))
    weights = root / "small.npz"
    export(path, weights)
    return actor, path, weights


# ---- protocol ------------------------------------------------------------------

def test_cards_and_levels():
    assert card_to_gd(0) == gd.card_id("HA") and card_to_gd(54) == gd.card_id("HA")
    assert card_to_gd(4) == gd.card_id("H2") and card_to_gd(1) == gd.card_id("DA")
    assert card_to_gd(2) == gd.card_id("SA") and card_to_gd(3) == gd.card_id("CA")
    assert card_to_gd(51) == gd.card_id("CK") and card_to_gd(52) == 52 and card_to_gd(107) == 53
    faces = sorted(card_to_gd(c) for c in range(108))
    assert faces == sorted(list(range(54)) * 2)
    for face in range(54):
        assert {face_to_bz(face, 0), face_to_bz(face, 1)} == {
            c for c in range(108) if card_to_gd(c) == face}
    assert [level_to_gd(x) for x in ("2", "10", "J", "A")] == [0, 8, 9, 12]


def test_claims_name_exactly_one_reading():
    rng = random.Random(5)
    checked = 0
    for _ in range(60):
        deck = [c for c in range(54) for _ in range(2)]
        rng.shuffle(deck)
        hand, level = sorted(deck[:27]), rng.randrange(13)
        readings = gd.legal_actions(hand, level, None, False)
        by_cards = {}
        for kind, key, cards in readings:
            k = key[1] if isinstance(key, tuple) else key
            by_cards.setdefault(tuple(sorted(cards)), []).append((kind, k))
        for cards, options in by_cards.items():
            for kind, key in options:
                declared = claim_faces(kind, key, list(cards), level)
                if declared is None:
                    continue
                checked += 1
                used = [face_to_bz(f, list(cards[:i]).count(f)) for i, f in enumerate(cards)]
                ids = physical_claim(list(cards), declared, used)
                assert len(set(ids)) == len(ids)
                assert sorted(ranks for ranks in reading_ranks(kind, key, list(cards), level)) == \
                    sorted(c // 4 if c < 52 else c - 39 for c in declared)
                matching = [o for o in options if claim_matches(o[0], o[1], list(cards), level, ids)]
                assert matching == [(kind, key)], (cards, level, kind, key, matching)
    assert checked > 10_000


def test_fullhouse_with_level_triple_and_wild_pair_is_inexpressible():
    level = 5                                               # sevens
    cards = gd.cards("S7 H7 H7 C7 D7")
    assert claim_faces("FullHouse", 12, cards, level) is None
    assert claim_faces("Bomb", 12, cards, level) == cards


def test_round_log_reads_both_history_formats():
    deal = {"stage": "deal", "deliver": list(range(27)), "your_id": 1,
            "global": {"level": "5", "tribute": 0, "first": -1, "last": -1}}
    positional = {"stage": "play", "done": [], "pass_on": -1,
                  "history": [[], [], [[40], [40]], [[], []]], "global": {"level": "5"}}
    wiki = {"stage": "play", "done": [], "pass_on": -1, "global": {"level": "5"},
            "history": [{"player": 3, "response": [[40], [40]]},
                        {"player": 0, "response": [[], []]}]}
    platform = {"stage": "play", "done": [], "pass_on": -1, "global": {"level": "5"},
                "history": [{}, {}, {"player": 3, "response": [[40], [40]]},
                            {"player": 0, "response": [[], []]}]}
    for request in (positional, wiki, platform):
        log = RoundLog.from_turns([deal, request], [[]])
        assert log.me == 1 and log.level == 3
        assert [(m.seat, m.action, m.is_pass) for m in log.moves] == [(3, [40], False),
                                                                      (0, [], True)]
        assert log.leader_hint == 3


# ---- the NumPy actor -------------------------------------------------------------

def test_numpy_actor_matches_torch(player):
    actor, _, weights = player
    numpy_actor = NumpyHistoryActor.load(weights)
    engine = gd.Engine(gd.RuleConfig.house(), gd.ActionConfig())
    engine.auto_pass = False
    state = gd.MatchState()
    engine.new_match(state, 7)
    stream = PublicStream()
    rng = random.Random(3)
    from eval.history_policy import apply_and_observe

    class Listener:
        def observe(self, event):
            stream.append(event)

    compared = 0
    while int(state.phase) in (1, 2, 3) and compared < 40:
        actions = engine.legal_actions(state)
        seat = int(state.to_move)
        if int(state.phase) == 3 and len(actions) > 1:
            obs = np.asarray(state.observation(seat), dtype=np.float32)
            cand = np.stack([engine.encode_action(a, state, seat) for a in actions]).astype(np.float32)
            inputs = DecisionInputs(
                streams=StreamBatch.from_streams([stream], "cpu"),
                match_index=torch.zeros(1, dtype=torch.long),
                prefix=torch.tensor([stream.prefix]), obs=torch.as_tensor(obs[None]),
                seat=torch.tensor([seat]), cand=torch.as_tensor(cand),
                offsets=torch.tensor([0, len(actions)]))
            with torch.no_grad():
                state_t = actor.decision_states(None, inputs)
                expected = actor.candidate_logits(state_t, inputs.cand, inputs.offsets).numpy()
            tokens, rounds, phases = stream.arrays()
            got = numpy_actor.logits(tokens, rounds, phases, obs, seat, cand)
            np.testing.assert_allclose(got, expected, atol=2e-5, rtol=1e-4)
            compared += 1
        apply_and_observe(engine, state, actions[rng.randrange(len(actions))], (Listener(),))
    assert compared >= 20


# ---- reconstruction ----------------------------------------------------------------

def test_token_stream_equals_public_stream():
    engine = gd.Engine(gd.RuleConfig.house(), gd.ActionConfig())
    engine.auto_pass = False
    state = gd.MatchState()
    engine.new_match(state, 2)
    ours, theirs = mirror.TokenStream(), PublicStream()
    from eval.history_policy import apply_and_observe

    class Both:
        def observe(self, event):
            ours.append(event)
            theirs.append(event)

    while int(state.phase) in (1, 2, 3):
        actions = engine.legal_actions(state)
        apply_and_observe(engine, state, actions[engine.greedy(state)], (Both(),))
    tokens, rounds, phases = ours.arrays()
    np.testing.assert_array_equal(tokens, theirs.tokens)
    np.testing.assert_array_equal(rounds, theirs.rounds)
    np.testing.assert_array_equal(phases, theirs.phases)


def test_actor_inputs_do_not_depend_on_the_hidden_filling(player):
    """Different fillings of the unseen hands give the same observation,
    candidates and public stream at every bot decision."""
    _, _, weights = player
    bot = Bot(weights)
    checked = []
    original = bot._respond

    def spy(log):
        if log.stage == "play":
            views = []
            for seed in (0, 1, 2):
                built = mirror.rebuild(log, seed=seed)
                actions = built.state.legal_actions()
                obs, cand = mirror.decision_arrays(built, actions)
                views.append((obs, cand, *built.stream.arrays()))
            for other in views[1:]:
                for a, b in zip(views[0], other):
                    np.testing.assert_array_equal(a, b)
            checked.append(1)
        return original(log)

    bot._respond = spy
    rng = random.Random(4)
    for kind in ("none", "single", "double", "double"):
        seats = [judge.InProcessSeat(bot), judge.GreedySeat(), judge.InProcessSeat(bot),
                 judge.GreedySeat()]
        judge.Referee(seats, random.Random(rng.getrandbits(32))).play_game(kind)
    assert len(checked) > 50


@pytest.mark.parametrize("history_format", ["botzone", "positional"])
def test_judge_games_match_the_torch_policy(player, history_format):
    _, checkpoint, weights = player
    report = judge.run(["bot", "greedy", "bot", "greedy"], 12, 9, str(weights),
                       reference=str(checkpoint), history_format=history_format)
    counts = report["counts"]
    assert report["illegal"] == []
    assert not any(key.startswith("note_") for key in counts), report["note_samples"]
    assert counts.get("agree_different", 0) == 0
    assert counts["agree_same"] > 200


def test_bot_process_both_modes(player, tmp_path):
    _, _, weights = player
    import sys
    command = f"{sys.executable} -m eval.botzone.bot --weights {weights}"
    for seats in (["proc:" + command, "greedy", "bot", "greedy"],
                  ["proc:" + command + " --traditional", "greedy", "greedy", "greedy"]):
        report = judge.run(seats, 1, 3, str(weights), swap=False)
        assert report["illegal"] == [], report
        assert not any(key.startswith("note_") for key in report["counts"]), report


def test_packed_zip_runs_without_gd_or_torch(player, tmp_path):
    """The upload archive runs from a clean path: no repo, no gd, no torch."""
    import subprocess
    import sys
    from eval.botzone.pack import pack

    _, _, weights = player
    archive = tmp_path / "gz_bot.zip"
    pack(weights, archive, embed=True)
    guard = tmp_path / "guard"
    guard.mkdir()
    # Any import of gd or torch fails inside the bot process.
    (guard / "gd.py").write_text("raise ImportError('gd is not available on Botzone')\n")
    (guard / "torch.py").write_text("raise ImportError('torch is not available on Botzone')\n")
    command = f"{sys.executable} {archive}"
    env_path = str(guard)
    import os
    old = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = env_path
    try:
        report = judge.run(["proc:" + command, "greedy", "proc:" + command + " --traditional",
                            "greedy"], 1, 4, None, swap=False)
    finally:
        if old is None:
            del os.environ["PYTHONPATH"]
        else:
            os.environ["PYTHONPATH"] = old
    assert report["illegal"] == [], report
    assert not any(key.startswith("note_") for key in report["counts"]), report
