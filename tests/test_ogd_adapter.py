"""Fast tests for eval/ogd_adapter: normalize.py round trips and trace
parsing against a small checked-in fixture. No java/jar dependency; anything
that needs the jar is marked `skipif` and skipped by default. Runtime target:
under 5 seconds.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from eval.ogd_adapter import normalize as N

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _jar_available() -> bool:
    try:
        from eval.ogd_adapter import bridge

        jar = bridge.ogd_root() / "guandan-java" / "guandan-java-action.jar"
        return jar.is_file() and shutil.which("java") is not None
    except Exception:
        return False


requires_java = pytest.mark.skipif(not _jar_available(), reason="OpenGuanDan jar/java not available")


# ---- normalize.py: cards ----------------------------------------------------


@pytest.mark.parametrize("cid", range(54))
def test_card_id_round_trip(cid: int) -> None:
    assert N.card_id(N.id_to_card(cid)) == cid


@pytest.mark.parametrize(
    "text,expected",
    [("S2", 0), ("H2", 1), ("C2", 2), ("D2", 3), ("SA", 48), ("HT", 33), ("DA", 51), ("SB", 52), ("HR", 53)],
)
def test_card_id_known_values(text: str, expected: int) -> None:
    assert N.card_id(text) == expected


# ---- normalize.py: power order (RULES.md section 4, 12) -------------------


def test_power_level_card_is_12() -> None:
    level = N.RANK_INDEX["7"]
    assert N.rank_char_power("7", level) == 12


def test_power_ordering_matches_rules_examples() -> None:
    # T-ORD-04: level 7, power of 8 and of 6 -> 5 and 4.
    level = N.RANK_INDEX["7"]
    assert N.rank_char_power("8", level) == 5
    assert N.rank_char_power("6", level) == 4
    # T-ORD-05: level A, power of A and of K -> 12 and 11.
    level_a = N.RANK_INDEX["A"]
    assert N.rank_char_power("A", level_a) == 12
    assert N.rank_char_power("K", level_a) == 11


def test_power_jokers_above_everything() -> None:
    level = N.RANK_INDEX["7"]
    assert N.rank_char_power("B", level) == 13
    assert N.rank_char_power("R", level) == 14


# ---- normalize.py: sequence windows (RULES.md section 4) ------------------


def test_straight_window_labels() -> None:
    assert N.straight_window("A") == 0     # A2345, the ace-low window
    assert N.straight_window("T") == 9     # TJQKA, the highest window


def test_tube_and_plate_window_labels() -> None:
    assert N.tube_window("A") == 0
    assert N.tube_window("Q") == 11        # QQKKAA, the highest tube window
    assert N.plate_window("A") == 0
    assert N.plate_window("K") == 12       # KKKAAA, the highest plate window


# ---- normalize.py: normalize_action, against RULES.md test vectors --------


def test_normalize_full_house_joker_pair() -> None:
    # T-FH-01: level 7, S5 D5 C5 SB SB -> full house, key 3.
    t, key, cards = N.normalize_action(["ThreeWithTwo", "5", ["S5", "C5", "D5", "SB", "SB"]], "7")
    assert t == "FullHouse"
    assert key == 3
    assert cards == tuple(sorted(N.card_id(c) for c in ["S5", "C5", "D5", "SB", "SB"]))


def test_normalize_bomb_key_is_size_power_tuple() -> None:
    # T-BOMB-01: level 7, all eight 7s -> bomb (8, 12).
    cards = ["S7", "H7", "C7", "D7", "S7", "H7", "C7", "D7"]
    t, key, _ = N.normalize_action(["Bomb", "7", cards], "7")
    assert t == "Bomb"
    assert key == (8, 12)


def test_normalize_pass() -> None:
    assert N.normalize_action(["PASS", "PASS", "PASS"], "7") == ("Pass", None, tuple())


def test_normalize_joker_bomb() -> None:
    t, key, cards = N.normalize_action(["FourKings", "B", ["SB", "SB", "HR", "HR"]], "7")
    assert t == "JokerBomb"
    assert cards == (52, 52, 53, 53)


def test_normalize_tribute_and_back() -> None:
    t, key, cards = N.normalize_action(["tribute", "tribute", ["D2"]], "5")
    assert t == "Tribute"
    assert cards == (N.card_id("D2"),)
    t2, _, cards2 = N.normalize_action(["back", "back", ["S3"]], "5")
    assert t2 == "BackTribute"
    assert cards2 == (N.card_id("S3"),)


def test_normalize_action_list_is_a_set() -> None:
    raw = [
        ["PASS", "PASS", "PASS"],
        ["Single", "5", ["D5"]],
        ["Single", "5", ["S5"]],
    ]
    out = N.normalize_action_list(raw, "7")
    assert ("Pass", None, tuple()) in out
    assert len(out) == 3  # PASS plus two distinct Single(D5)/Single(S5) concrete actions


def test_all_ogd_type_strings_mapped() -> None:
    # O9: the 13 type strings the jar is known to emit (README section 5.2
    # plus FourKings), each maps to one of our engine's type strings.
    expected = {
        "Single", "Pair", "Trips", "ThreePair", "ThreeWithTwo", "TwoTrips",
        "Straight", "StraightFlush", "Bomb", "FourKings", "tribute", "back", "PASS",
    }
    assert set(N.OGD_TYPE_TO_OURS) == expected


# ---- trace parsing on the checked-in fixture -------------------------------


def _load_fixture() -> list[dict]:
    path = FIXTURES / "ogd_trace_sample.jsonl"
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def test_fixture_exists_and_parses() -> None:
    events = _load_fixture()
    assert len(events) > 0
    assert all(isinstance(e, dict) for e in events)


def test_fixture_has_expected_event_kinds() -> None:
    events = _load_fixture()
    kinds = {e["event"] for e in events}
    assert {"deal", "act", "tribute", "back", "episodeOver", "gameResult"} <= kinds


def test_fixture_deal_events_have_four_hands() -> None:
    for e in _load_fixture():
        if e["event"] == "deal":
            assert len(e["hands"]) == 4
            for hand in e["hands"]:
                assert len(hand) == 27
            assert set(e["team_levels"]) == {"0", "1"}


def test_fixture_act_events_are_normalizable() -> None:
    events = _load_fixture()
    level = None
    for e in events:
        if e["event"] == "deal":
            level = e["level"]
        elif e["event"] == "act":
            assert level is not None
            normalized = N.normalize_action_list(e["action_list"], level)
            assert len(normalized) == len(e["action_list"])
            # The chosen action must itself normalize and be legal-shaped.
            chosen = N.normalize_action(e["chosen_action"], level)
            assert chosen in normalized


def test_fixture_episode_over_events_well_formed() -> None:
    for e in _load_fixture():
        if e["event"] == "episodeOver":
            assert isinstance(e["order"], list)
            assert len(e["order"]) in (2, 3, 4)
            assert len(set(e["order"])) == len(e["order"])  # no seat repeated


def test_fixture_game_result_well_formed() -> None:
    results = [e for e in _load_fixture() if e["event"] == "gameResult"]
    assert len(results) == 1
    assert results[0]["winning_team"] in (0, 1)
    assert set(results[0]["final_ranks"]) == {"0", "1"}


# ---- java-requiring smoke test ---------------------------------------------


@requires_java
def test_bridge_legal_moves_smoke() -> None:
    from eval.ogd_adapter import bridge

    moves = bridge.legal_moves(["S3", "H3", "C4", "D5", "S6", "H7"], 0, "3")
    assert len(moves) > 0
    assert moves[0][0] in ("PASS", "Single", "Pair", "Trips", "Bomb", "Straight", "StraightFlush")
    bridge.shutdown()
