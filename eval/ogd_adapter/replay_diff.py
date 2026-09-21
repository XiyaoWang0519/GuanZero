"""Replays an OpenGuanDan trace through OUR engine and reports divergences.

Our C++ engine (`gd`, the pybind11 module in `python/gd/`) does not exist yet
as of M0 task 10. The engine side therefore sits behind a single
`EngineAdapter` object; when `import gd` fails, this module still imports
cleanly, still runs its own self-tests (pure `normalize.py` checks, no jar
and no engine needed) and reports "engine not built" instead of crashing, so
that CI and casual runs stay green while `cpp/` is under construction. Once
`gd` exists, `EngineAdapter` starts doing real comparisons with no other
change needed here.

At every decision, compares the legal action set (full mode, normalized to
`(type, key, sorted card multiset)`, see `normalize.py`). At every round end,
compares the finishing order, the level changes and the tribute flow.

Usage:
    python -m eval.ogd_adapter.replay_diff --trace <dir-or-file> [--limit N]
    python -m eval.ogd_adapter.replay_diff            # self-tests only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from eval.ogd_adapter import normalize as N


# ---- the engine side, behind one seam --------------------------------------


class EngineAdapter:
    """Wraps our C++ engine (`gd`). `available` is False until `cpp/` builds
    a `gd` module; every method raises `EngineNotBuilt` until then, and
    callers are expected to check `available` first (`diff_trace` does)."""

    def __init__(self) -> None:
        self._gd: Any = None
        self.available = False
        self.import_error: str | None = None
        try:
            import gd  # type: ignore  # noqa: F401
        except Exception as e:  # pragma: no cover - exercised once gd exists
            self.import_error = f"{type(e).__name__}: {e}"
            return
        self._gd = gd
        self.available = True

    def legal_actions(self, hand_ids: tuple[int, ...], level: int, top) -> set[tuple]:
        if not self.available:
            raise EngineNotBuilt(self.import_error or "gd not built")
        return set(self._gd.legal_actions(list(hand_ids), level, top, canonical=False))


class EngineNotBuilt(RuntimeError):
    pass


# ---- trace replay -----------------------------------------------------------


@dataclass
class DivergenceReport:
    decisions_checked: int = 0
    round_ends_checked: int = 0
    engine_available: bool = False
    classes: Counter = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=lambda: {})

    def note(self, cls: str, detail: str, max_examples: int = 3) -> None:
        self.classes[cls] += 1
        bucket = self.examples.setdefault(cls, [])
        if len(bucket) < max_examples:
            bucket.append(detail)

    def summary_table(self) -> str:
        if not self.classes:
            return "(no divergences recorded)"
        width = max(len(k) for k in self.classes)
        lines = [f"{'class'.ljust(width)}  count"]
        for cls, n in self.classes.most_common():
            lines.append(f"{cls.ljust(width)}  {n}")
        return "\n".join(lines)


def _iter_events(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _trace_files(trace_arg: Path) -> list[Path]:
    if trace_arg.is_dir():
        return sorted(trace_arg.glob("*.jsonl"))
    return [trace_arg]


def diff_trace(path: Path, engine: EngineAdapter, limit: int | None, report: DivergenceReport) -> None:
    """Replay one trace file, recording divergences into `report`."""
    current_level: str | None = None
    checked = 0
    for ev in _iter_events(path):
        if limit is not None and checked >= limit:
            return
        kind = ev["event"]
        if kind == "deal":
            current_level = ev["level"]
            continue
        if kind == "act":
            checked += 1
            report.decisions_checked += 1
            assert current_level is not None, "act event before any deal event"
            try:
                ogd_actions = N.normalize_action_list(ev["action_list"], current_level)
            except Exception as e:
                report.note("normalize_error", f"{path.name}#{ev.get('round')}: {e!r}")
                continue
            if not engine.available:
                continue  # normalization-only pass; no engine to compare against
            hand_ids = tuple(sorted(N.card_id(c) for c in ev["hand"]))
            level_idx = N.RANK_INDEX[current_level]
            top = None  # TODO(M0 task 4/5): derive the top play once gd.Action exists
            try:
                engine_actions = engine.legal_actions(hand_ids, level_idx, top)
            except EngineNotBuilt:
                continue
            if ogd_actions != engine_actions:
                only_ogd = ogd_actions - engine_actions
                only_engine = engine_actions - ogd_actions
                if only_ogd:
                    report.note("legal_set_missing_from_engine", f"{path.name}#{ev['round']} seat{ev['seat']}: {sorted(only_ogd)[:3]}")
                if only_engine:
                    report.note("legal_set_extra_in_engine", f"{path.name}#{ev['round']} seat{ev['seat']}: {sorted(only_engine)[:3]}")
        elif kind == "episodeOver":
            report.round_ends_checked += 1
            # Finishing order / level-change / tribute-flow comparison needs
            # gd.MatchState (task 6); until then this only checks that the
            # trace itself is well-formed.
            if not isinstance(ev.get("order"), list) or len(ev["order"]) not in (2, 3, 4):
                report.note("malformed_episode_over", f"{path.name}#{ev.get('round')}: {ev.get('order')}")


def self_test() -> None:
    """Pure `normalize.py` checks -- no jar, no engine, always runnable."""
    fh = N.normalize_action(["ThreeWithTwo", "5", ["S5", "C5", "D5", "SB", "SB"]], "7")
    assert fh[0] == "FullHouse" and fh[1] == 3, fh
    straight = N.normalize_action(["Straight", "A", ["S2", "D3", "C4", "H5", "SA"]], "7")
    assert straight[0] == "Straight" and straight[1] == 0, straight
    bomb = N.normalize_action(["Bomb", "7", ["S7", "H7", "C7", "D7", "S7", "H7", "C7", "D7"]], "7")
    assert bomb[0] == "Bomb" and bomb[1] == (8, 12), bomb
    passa = N.normalize_action(["PASS", "PASS", "PASS"], "7")
    assert passa == ("Pass", None, tuple())
    fk = N.normalize_action(["FourKings", "B", ["SB", "SB", "HR", "HR"]], "7")
    assert fk[0] == "JokerBomb" and fk[2] == (52, 52, 53, 53), fk
    print("[ok] normalize.py self-tests")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, default=None, help="a trace .jsonl file or a directory of them")
    ap.add_argument("--limit", type=int, default=None, help="max decisions to check per file")
    args = ap.parse_args(argv)

    self_test()

    engine = EngineAdapter()
    if engine.available:
        print("[ok] engine (gd) is built: comparing legal action sets")
    else:
        print(f"[skip] engine not built ({engine.import_error}); "
              "validating trace normalization only")

    if args.trace is None:
        print("no --trace given; ran self-tests only")
        return 0

    files = _trace_files(args.trace)
    if not files:
        print(f"no trace files found at {args.trace}")
        return 1

    report = DivergenceReport(engine_available=engine.available)
    for f in files:
        diff_trace(f, engine, args.limit, report)

    print(f"\nfiles={len(files)} decisions_checked={report.decisions_checked} "
          f"round_ends_checked={report.round_ends_checked} engine_available={report.engine_available}")
    print("\ndivergence classes:")
    print(report.summary_table())
    for cls, examples in report.examples.items():
        print(f"\n  {cls}:")
        for ex in examples:
            print(f"    {ex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
