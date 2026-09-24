"""A drop-in replacement for DanLM's ``GuanDanRound.get_observation``.

DanLM's compiled ``get_observation`` takes about 40 ms of wall clock per call
while using about 1 ms of CPU (measured Sept. 23, 2026; the cause is inside
the binary). Since ``GuanDanRound`` is an ordinary Python class, the method
can be replaced. This module builds the same ``Observation`` from
``round.state`` in pure Python and numpy. ``verify`` proves field-by-field
equality against the original on random play before ``install`` is trusted;
``arena`` calls ``install`` only after a verification run has been recorded.
"""
from __future__ import annotations

import numpy as np

_ORIGINAL = None


def _modules():
    from danzero.engine import actions, game  # noqa: WPS433

    return actions, game


def _hand_plays(rnd, actions, hand, player: int, level_index: int) -> np.ndarray:
    """Reuse a seat's decomposition until its hand or level changes.

    A pass changes the trick but leaves the hand unchanged. Keep only the
    latest hand per seat, attached to the round rather than a global cache:
    memory stays bounded and a new round cannot inherit stale state. The
    lead-dependent filtering still runs for every observation.
    """
    cache = getattr(rnd, "_gd_hand_plays", None)
    if cache is None:
        cache = rnd._gd_hand_plays = {}
    array = np.asarray(hand)
    key = (level_index, array.dtype.str, array.shape, array.tobytes())
    previous = cache.get(player)
    if previous is not None and previous[0] == key:
        return previous[1]
    # Own the array rather than retaining a possible compiled scratch view.
    plays = np.array(actions.hand_calculator_v2(hand, level_index), copy=True)
    plays.setflags(write=False)
    cache[player] = key, plays
    return plays


def fast_get_observation(self):
    """Same fields as DanLM's ``GuanDanRound.get_observation``."""
    actions, game = _modules()
    state = self.state
    player = int(state.current_player)
    hand = state.hands[player]
    lead_play = state.lead_play
    is_leading = bool(actions.play_type_of(lead_play) == "pass")
    # The original builds the list with hand_calculator_v2 and filter_valid_plays
    # (row order verified 1,500/1,500); get_legal_actions orders rows differently.
    level_index = self.level - 2
    plays = _hand_plays(self, actions, hand, player, level_index)
    if is_leading:
        plays = plays[plays[:, 54] < 0.5]          # drop the PASS row (type one-hot 0)
    else:
        plays = np.asarray(actions.filter_valid_plays(plays, level_index, lead_play))
    legal_plays = np.ascontiguousarray(plays.astype(np.int8))
    # DanLM subtracts only the OTHER seats' played cards (measured, not a
    # choice of ours): the player's own played cards count as unseen.
    played_others = np.zeros(54, np.int64)
    for seat, vector in enumerate(state.played_cards):
        if seat != player:
            played_others += np.asarray(vector, np.int64)
    other_unseen = (2 - np.asarray(hand, np.int64) - played_others).astype(np.int8)
    previous = state.last_actions[(player + 3) % 4]
    last_action = (np.zeros(54, np.int8) if previous is None
                   else np.asarray(previous, np.int8).copy())
    # The sentinel also stands in once the teammate has finished (measured).
    partner = (player + 2) % 4
    teammate = None if partner in state.finish_order else state.last_actions[partner]
    last_teammate_action = (np.full(54, -1, np.int8) if teammate is None
                            else np.asarray(teammate, np.int8).copy())
    played_cards = {seat: np.asarray(state.played_cards[seat], np.int8).copy() for seat in range(4)}
    remaining_counts = {seat: int(np.asarray(state.hands[seat]).sum()) for seat in range(4)}
    team = player % 2
    return game.Observation(
        player=player, hand=np.asarray(hand, np.int8).copy(), legal_plays=legal_plays,
        lead_play=np.asarray(lead_play).copy(), is_leading=is_leading,
        other_unseen=other_unseen, last_action=last_action,
        last_teammate_action=last_teammate_action, played_cards=played_cards,
        remaining_counts=remaining_counts,
        own_team_level=int(self.team_levels[team]), opp_team_level=int(self.team_levels[1 - team]),
        round_level=int(self.level),
        trick_actions=[None if x is None else np.asarray(x).copy() for x in state.trick_actions],
        finish_order=list(state.finish_order),
    )


def install() -> None:
    global _ORIGINAL
    _, game = _modules()
    if _ORIGINAL is None:
        _ORIGINAL = game.GuanDanRound.get_observation
    game.GuanDanRound.get_observation = fast_get_observation


def uninstall() -> None:
    _, game = _modules()
    if _ORIGINAL is not None:
        game.GuanDanRound.get_observation = _ORIGINAL


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def verify(rounds: int = 100, seed: int = 0) -> dict:
    """Random play; every step compares the original and the replacement."""
    import dataclasses
    import time

    from danzero.engine import cards  # noqa: WPS433

    actions, game = _modules()
    uninstall()
    original = game.GuanDanRound.get_observation
    rng = np.random.default_rng(seed)
    steps = mismatches = 0
    fields = [f.name for f in dataclasses.fields(game.Observation)]
    bad: dict[str, int] = {}
    original_seconds = fast_seconds = 0.0
    for r in range(rounds):
        level = int(rng.integers(2, 15))
        hands = cards.deal_hands(seed=seed * 100_000 + r)
        rnd = game.GuanDanRound(level=level, hands=hands, first_player=int(rng.integers(4)),
                                team_levels=(int(rng.integers(2, 15)), int(rng.integers(2, 15))))
        obs = original(rnd)
        while obs is not None and not rnd.done:
            t = time.perf_counter()
            mine = fast_get_observation(rnd)
            fast_seconds += time.perf_counter() - t
            for name in fields:
                if not _same(getattr(obs, name), getattr(mine, name)):
                    bad[name] = bad.get(name, 0) + 1
            mismatches += any(not _same(getattr(obs, name), getattr(mine, name)) for name in fields)
            steps += 1
            choice = int(rng.integers(len(obs.legal_plays)))
            t = time.perf_counter()
            obs = rnd.step(choice, obs)   # the original's get_observation runs inside
            original_seconds += time.perf_counter() - t
    return {"rounds": rounds, "steps": steps, "mismatching_steps": mismatches,
            "mismatches_by_field": bad,
            "original_ms_per_step": original_seconds / max(steps, 1) * 1000,
            "fast_ms_per_step": fast_seconds / max(steps, 1) * 1000}


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(verify(int(sys.argv[1]) if len(sys.argv) > 1 else 100), indent=2))
