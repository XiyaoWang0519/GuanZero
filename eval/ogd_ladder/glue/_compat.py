"""OpenGuanDan -> 2020-server differences that bot_host's ``to_2020`` does not
(yet) translate; shared by the glue modules that need them."""
from collections import Counter

# OpenGuanDan engine/types.py ``Card.RANKS``: index -> rank character.
_RANKS = ("", "", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")


def fix_beginning(msg: dict) -> dict:
    """OpenGuanDan's ``beginning`` notify carries ``curRank``/``selfRank``/
    ``oppoRank`` as engine indices (ints 2..14) while every other message uses
    rank characters ('2'..'A'); the 2020 bots expect characters (and
    ``State.parse`` stores whatever it gets, so an int breaks later
    ``'H' + curRank``)."""
    if msg.get("stage") != "beginning":
        return msg
    msg = dict(msg)
    for key in ("curRank", "selfRank", "oppoRank"):
        if isinstance(msg.get(key), int):
            msg[key] = _RANKS[msg[key]]
    return msg


def unordered_getindex(self, actionlist, color):
    """Order-insensitive replacement for the 3rd-ez/369 ``getindex``.

    The original compares a card list the bot built from its hand with
    ``actionList[i][2]`` by ``==``, relying on the 2020 server listing an
    action's cards in the same order as ``handCards``. OpenGuanDan uses a
    different within-action order (e.g. ``['HK', 'SK']`` against a hand
    ``[..., 'SK', 'HK']``), so nearly every lookup misses and returns 0. Same
    contract otherwise: the index of the last action whose cards match, else 0.
    """
    index = 0
    if not isinstance(color, list):
        return index
    want = Counter(color)
    for i, a in enumerate(actionlist):
        if isinstance(a[2], list) and Counter(a[2]) == want:
            index = actionlist.index(a)
    return index
