"""Fault-injected fallback games under the user's unmodified Botzone judge.

The fixture is the judge supplied on October 22, 2026, not our gd referee.
Its private allocation is used only by this test driver, never given to a bot.
"""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import random
import runpy

import pytest

from eval.botzone.bot import Bot, answer
from eval.botzone.protocol import Move, RoundLog, level_to_gd


JUDGE = Path(__file__).parent / "fixtures" / "botzone_guandan_judge.py"


@pytest.fixture(scope="module")
def judge_source():
    return runpy.run_path(str(JUDGE), run_name="fallback_test_judge")


def run_judge(judge_source, full_input):
    # runpy returns a copy of the module dict; functions retain their globals.
    namespace = judge_source["main"].__globals__
    namespace.update(printed=False, errored=[None] * 4, currplayer=-1,
                     pointorder=list("234567890JQKA"))
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        try:
            namespace["main"](copy.deepcopy(full_input))
        except SystemExit as error:
            assert error.code == 0
    result = json.loads(output.getvalue())
    assert not result.get("display", {}).get("error"), result
    return result


def fail_respond(self, log):
    raise RuntimeError("injected mirror/inference failure")


def test_original_judge_fixture():
    assert hashlib.sha256(JUDGE.read_bytes()).hexdigest() == (
        "49a7346ddc0a06ebd8922e582d535fbc6fb5418ddea16b023a5f3384fc9b6fd9")


def test_judge_ten_level_alias():
    assert level_to_gd("0") == level_to_gd("10") == 8


@pytest.mark.parametrize("level", range(13))
def test_random_exchange_choices_under_original_judge(level, judge_source, monkeypatch):
    monkeypatch.setattr(Bot, "_respond", fail_respond)
    rng = random.Random(81000 + level)
    ns = judge_source["main"].__globals__
    level_name = "234567890JQKA"[level]
    ns["pointorder"] = list("234567890JQKA")
    ns["set_level"](level_name)
    bot = Bot(None)
    for _ in range(100):
        log = RoundLog()
        log.me, log.level = 0, level
        log.deliver = rng.sample(range(108), 27)
        for stage, validator in (("tribute", ns["isValidTribute"]),
                                 ("return", ns["isValidReturn"])):
            log.stage = stage
            response = bot.respond(log)
            assert len(response) == 1 and response[0] in log.deliver
            assert validator(log.deliver, response[0], level_name, 0)


def test_tribute_fallback_uses_largest_nonwild(monkeypatch):
    monkeypatch.setattr(Bot, "_respond", fail_respond)
    log = RoundLog()
    log.stage, log.me, log.level = "tribute", 0, 0
    log.deliver = [4, 53]  # heart 2 is wild; the big joker must be paid
    assert Bot(None).respond(log) == [53]


@pytest.mark.parametrize("level", range(13))
def test_return_uses_original_hand_and_judge_cap(level, judge_source, monkeypatch):
    monkeypatch.setattr(Bot, "_respond", fail_respond)
    log = RoundLog()
    log.stage, log.me, log.level = "return", 0, level
    # First card is illegal (big joker). Neither a promised receipt nor our
    # earlier return may change the hand used at the return stage.
    log.deliver = [53, 0, 36, 32, 28, 24, 20, 16, 12, 8, 4]
    log.my_return = 4
    response = Bot(None).respond(log)
    ns = judge_source["main"].__globals__
    ns["pointorder"] = list("234567890JQKA")
    level_name = "234567890JQKA"[level]
    ns["set_level"](level_name)
    assert ns["isValidReturn"](log.deliver, response[0], level_name, 0)


@pytest.mark.parametrize("stage", ["tribute", "return"])
def test_anti_tribute_returns_empty(stage, monkeypatch):
    monkeypatch.setattr(Bot, "_respond", fail_respond)
    log = RoundLog()
    log.stage, log.resist = stage, True
    log.deliver = [4, 53, 107]
    assert Bot(None).respond(log) == []


def test_pending_window_allows_jiefeng_lead(monkeypatch):
    monkeypatch.setattr(Bot, "_respond", fail_respond)
    log = RoundLog()
    log.me, log.stage, log.deliver = 2, "play", [4, 8]
    # A finished partner's old play remains in the accumulated record, but
    # the pending judge window has only passes: we must lead, not pass.
    log.moves = [Move(0, [12], [12]), Move(1, [], [])]
    log.feed({"stage": "play", "history": [
        {}, {"player": 1, "response": [[], []]},
        {"player": 2, "response": [[], []]},
        {"player": 3, "response": [[], []]}]})
    assert log.leading
    assert Bot(None).respond(log) == [[8], [8]]  # 3 is below the level 2


@pytest.mark.parametrize("positional", [False, True])
def test_pending_window_requires_following(positional, monkeypatch):
    monkeypatch.setattr(Bot, "_respond", fail_respond)
    log = RoundLog()
    log.me, log.stage, log.deliver = 0, "play", [4, 8]
    history = ([[[], []], [[12], [12]], [[], []], [[], []]] if positional else
               [{"player": 1, "response": [[12], [12]]},
                {"player": 2, "response": [[], []]},
                {"player": 3, "response": [[], []]}])
    log.feed({"stage": "play", "history": history})
    assert not log.leading
    assert Bot(None).respond(log) == [[], []]


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("me", range(4))
def test_play_uses_received_physical_cards(tied, me, monkeypatch):
    monkeypatch.setattr(Bot, "_respond", fail_respond)
    log = RoundLog()
    log.me, log.stage, log.tribute = me, "play", 2
    log.first, log.last = 0, 3
    log.tribute_cards = {1: 49, 3: 50 if tied else 45}
    log.return_cards = {0: 4, 2: 8}
    # first=0: payer 1 -> 0 on a tie or with a higher tribute.
    outgoing = {0: 4, 1: 49, 2: 8, 3: log.tribute_cards[3]}
    incoming = {0: 49, 1: 4, 2: log.tribute_cards[3], 3: 8}
    log.deliver = [outgoing[me], 12]
    log.moves = [Move(me, [12], [12])]
    # Public exchange dictionaries suffice, even without my_* caches.
    assert Bot(None).respond(log) == [[incoming[me]], [incoming[me]]]


@pytest.mark.parametrize("level", range(13))
@pytest.mark.parametrize("mode", ["none", "single", "double", "double-tie", "anti-single",
                                  "anti-double-split", "anti-double-both"])
def test_forced_fallback_full_game(level, mode, judge_source, monkeypatch):
    monkeypatch.setattr(Bot, "_respond", fail_respond)
    rng = random.Random(20261022 + level * 17 + len(mode))
    deck = list(range(108))
    rng.shuffle(deck)
    allocation = [deck[s * 27:(s + 1) * 27] for s in range(4)]
    first, last = level % 4, (level + 3) % 4
    tribute = 0 if mode == "none" else (1 if "single" in mode else 2)
    payers = [last] if tribute == 1 else [last, (last + 2) % 4]
    if mode == "double-tie":
        # Equal rank, different suit: verify the physical exchange routing
        # against the real judge, rather than just our own engine's rules.
        base = ((level + 1) % 13) * 4
        targets = {53: first, 107: first, 52: first, 106: first,
                   base + 1: last, base + 2: (last + 2) % 4}
        for card, target in targets.items():
            holder = next(s for s in range(4) if card in allocation[s])
            if holder != target:
                other = next(c for c in allocation[target] if c not in targets)
                allocation[holder].remove(card)
                allocation[target].remove(other)
                allocation[holder].append(other)
                allocation[target].append(card)
    if mode.startswith("anti"):
        targets = [last, payers[-1]] if mode.endswith("split") else [last, last]
        for joker, target in zip((53, 107), targets):
            holder = next(s for s in range(4) if joker in allocation[s])
            if holder != target:
                other = next(c for c in allocation[target] if c not in (53, 107))
                allocation[holder].remove(joker)
                allocation[target].remove(other)
                allocation[holder].append(other)
                allocation[target].append(joker)
    initdata = {"allocation": allocation, "level": ["234567890JQKA"[level]] * 2,
                "tribute": tribute, "first": first, "last": last}
    bots = [Bot(None) for _ in range(4)]
    requests, responses = [[] for _ in range(4)], [[] for _ in range(4)]
    log = []
    stages = set()
    for turn in range(500):
        out = run_judge(judge_source, {"initdata": initdata, "log": log})
        if "initdata" in out:
            initdata = out["initdata"]
        if out["command"] == "finish":
            assert all(value >= 0 for value in out["content"].values())
            assert "play" in stages
            assert ("return" in stages) == bool(tribute)
            return
        seat_text, request = next(iter(out["content"].items()))
        seat = int(seat_text)
        stages.add(request["stage"])
        if request["stage"] in ("tribute", "return") and mode.startswith("anti"):
            assert request["global"]["resist"]
        if request["stage"] == "return" and mode == "double-tie":
            cards = list(request["global"]["tribute_cards"].values())
            assert cards[0] % 54 // 4 == cards[1] % 54 // 4
            assert cards[0] % 4 != cards[1] % 4
        requests[seat].append(request)
        reply = answer(bots[seat], requests[seat], responses[seat])
        response = reply["response"]
        if request["stage"] != "deal":
            assert "injected mirror/inference failure" in reply["debug"]
        responses[seat].append(response)
        log.extend([{"output": out}, {seat_text: {"response": response}}])
    pytest.fail("fallback game failed to terminate")
