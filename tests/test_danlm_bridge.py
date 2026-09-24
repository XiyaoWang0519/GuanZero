"""Stage C C0: conversions between DanLM's encoding and gd (pure Python)."""
import gd
import numpy as np
import pytest

from eval.danlm import bridge as b


def test_card_conversion_round_trips_and_matches_named_cards():
    for card in range(54):
        assert b.card_theirs_to_ours(b.card_ours_to_theirs(card)) == card
        assert b.card_ours_to_theirs(b.card_theirs_to_ours(card)) == card
    # DanLM: H2 = 0, S2 = 13, C2 = 26, D2 = 39, DA = 51; jokers shared.
    assert gd.card_str(b.card_theirs_to_ours(0)) == "H2"
    assert gd.card_str(b.card_theirs_to_ours(13)) == "S2"
    assert gd.card_str(b.card_theirs_to_ours(26)) == "C2"
    assert gd.card_str(b.card_theirs_to_ours(39)) == "D2"
    assert gd.card_str(b.card_theirs_to_ours(51)) == "DA"
    assert b.card_theirs_to_ours(52) == 52 and b.card_theirs_to_ours(53) == 53


def test_power_matches_engine_and_inverts():
    for level in range(13):
        for rank in range(15):
            assert b.power(rank, level) == gd.power(rank, level)
            assert b.rank_of_power(b.power(rank, level), level) == rank


def test_sequence_rank_convention():
    # Straight A2345: highest natural rank 5 (index 3) <-> gd window 0.
    assert b.key_from_their_rank("Straight", 3, 5) == 0
    assert b.key_from_their_rank("Straight", 12, 5) == 9      # TJQKA
    assert b.key_from_their_rank("Tube", 1, 0) == 0            # AA2233
    assert b.key_from_their_rank("Tube", 12, 0) == 11          # QQKKAA
    assert b.key_from_their_rank("Plate", 0, 0) == 0           # AAA222
    assert b.key_from_their_rank("Plate", 12, 0) == 12         # KKKAAA
    for kind in ("Straight", "StraightFlush", "Tube", "Plate"):
        for key in range(10):
            assert b.key_from_their_rank(kind, b.their_rank_from_key(kind, key, 4), 4) == key


def test_play_encode_decode_round_trip_and_matches_engine_action():
    level = 5
    hand = gd.cards("S3 D3 H7 S9 D9")
    actions = gd.legal_actions(hand, level, None)
    for kind, key, cards in actions:
        play = b.NormalizedPlay(kind, tuple(sorted(cards)),
                                None if kind in ("Pass", "JokerBomb") else
                                (key[1] if isinstance(key, tuple) else key))
        vector = b.encode_play(play, level)
        assert vector.shape == (80,) and vector[:54].sum() == len(cards)
        assert b.decode_play(vector, level) == play


def test_play_index_finds_exact_then_reading_then_missing():
    level = 2
    pair = b.NormalizedPlay("Pair", tuple(sorted(gd.cards("S9 D9"))), 7)
    other_key = b.NormalizedPlay("Pair", pair.cards, 6)
    plays = np.stack([b.encode_play(other_key, level), b.encode_play(pair, level)])
    index = b.PlayIndex(plays, level)
    assert index.find(pair) == (1, "exact")
    assert index.find(b.NormalizedPlay("Pair", pair.cards, 5)) == (0, "reading")
    assert index.find(b.NormalizedPlay("Single", (0,), 0)) == (None, "missing")


def test_arena_module_imports_without_danlm():
    from eval.danlm import arena

    deals = arena.generate_deals(5, 1, tribute_fraction=1.0)
    assert all(len(d.prev_order) == 4 for d in deals)
    back = arena.deal_from_json(arena.deal_to_json(deals[0]))
    assert list(back.prev_order) == list(deals[0].prev_order) and back.level == deals[0].level
    with pytest.raises(ValueError):
        arena.generate_deals(0, 1)
