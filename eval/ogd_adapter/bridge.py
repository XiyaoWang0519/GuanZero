"""Locates the OpenGuanDan clone and exposes its move generator and table.

HARD CONSTRAINT (see the M0 task 10 brief): no code or data from the clone is
copied into this repository. This module only *locates* the clone through the
``OGD_ROOT`` environment variable (or the fallback path below) and imports its
`guandan-java/engine` package off `sys.path` at call time. Traces produced by
`trace_logger.py` are our own generated data; they are never written back into
the clone and the clone's tree is never read as a data source, only as code.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

# Fallback location of the clone, as provided in the M0 task 10 brief. Override
# with the OGD_ROOT environment variable; nothing here is copied from the
# clone itself, only the path to it.
_DEFAULT_OGD_ROOT = Path(
    "/private/tmp/claude-501/-Users-xiyaowang-Documents-Projects-GuanZero/"
    "3b5488a2-cb79-430b-877a-e2c8ad4db17a/scratchpad/OpenGuanDan"
)

_sys_path_ready = False


def ogd_root() -> Path:
    """Return the OpenGuanDan clone root, from ``OGD_ROOT`` or the fallback."""
    override = os.environ.get("OGD_ROOT", "").strip()
    root = Path(override) if override else _DEFAULT_OGD_ROOT
    engine_dir = root / "guandan-java" / "engine"
    if not engine_dir.is_dir():
        raise FileNotFoundError(
            f"OpenGuanDan clone not found at {root} (looked for {engine_dir}). "
            "Set OGD_ROOT to the clone's root directory."
        )
    return root


def ogd_traces_dir() -> Path:
    """Default output directory for generated traces: ``OGD_TRACES`` or a
    scratch directory outside the repository. Never a path inside this repo;
    `.gitignore` also ignores `ogd_traces/` as a defense in depth."""
    override = os.environ.get("OGD_TRACES", "").strip()
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / "gd_ogd_traces"


def _ensure_sys_path() -> Path:
    """Add the clone's `guandan-java/` directory to `sys.path` (idempotent)."""
    global _sys_path_ready
    root = ogd_root()
    bridge_dir = str(root / "guandan-java")
    if not _sys_path_ready or bridge_dir not in sys.path:
        sys.path.insert(0, bridge_dir)
        _sys_path_ready = True
    return root


def make_env(num_players: int = 4) -> Any:
    """Return a fresh `Table` (aliased `Environment`) with `num_players` seats
    added. The table is *not* started: call `.start()` on it (or drive it
    through `trace_logger`) to deal the first round and get its messages."""
    _ensure_sys_path()
    from engine.environment import Environment  # noqa: E402  (path set above)

    env = Environment()
    for i in range(num_players):
        env.add_player(f"p{i}", i)
    return env


def legal_moves(
    hand: Sequence[str],
    hearts_num: int,
    current_rank: str,
    greater: tuple[str, str, list[str]] | None = None,
) -> list[tuple[str, str, tuple[str, ...]]]:
    """Query the jar's move generator directly, bypassing `Table`.

    This is the differential-testing primitive: `hand` is a list of card
    strings (`"S5"`, `"SB"`, `"HR"`, ...), `hearts_num` is the number of wild
    heart-of-`current_rank` cards physically held in `hand` (see
    docs/ogd_parity_probes.md for what this argument actually controls),
    `current_rank` is the round level as a rank character (`"7"`, `"T"`,
    `"A"`, ...), and `greater` is the `(type, rank, cards)` tuple of the top
    play to beat, or `None` to enumerate a leading player's moves.

    Returns the jar's raw action tuples, each converted to
    `(type_str, rank_str, cards_tuple)` with `cards_tuple` possibly empty
    (for `PASS`). Types and rank strings are exactly what the jar returns
    (`"Single"`, `"ThreeWithTwo"`, `"5"`, `"B"`, ...); `normalize.py` maps
    them onto our engine's `(type, key, cards)` representation.
    """
    _ensure_sys_path()
    from engine.moves import Moves  # noqa: E402
    from engine.types import Move, Trick  # noqa: E402

    m = Moves()
    if greater is None:
        m.parse_first_action(list(hand), hearts_num, current_rank)
    else:
        g_type, g_rank, g_cards = greater
        g = Move(g_type, g_rank, list(g_cards) if g_cards else [])
        # rank_order/number_order are accepted by parse_second_action but are
        # not forwarded to the jar (see moves.py); a fresh Trick supplies the
        # library's own defaults rather than us transcribing them.
        t = Trick()
        m.parse_second_action(list(hand), hearts_num, current_rank, g, t.rank_order, t.number_order)
    out: list[tuple[str, str, tuple[str, ...]]] = []
    for entry in m.action_list:
        typ, rank, cards = entry[0], entry[1], entry[2]
        out.append((typ, rank, tuple(cards) if cards else tuple()))
    return out


def shutdown() -> None:
    """Stop the jar worker subprocess cleanly, if one was started."""
    if not _sys_path_ready:
        return
    from engine import moves as _moves_mod  # noqa: E402

    _moves_mod._J.close()
