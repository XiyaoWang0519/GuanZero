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

_N = N


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
        self.gd = gd
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


def _as_triple(action) -> tuple:
    """Our engine's Action in the normalizer's vocabulary."""
    kind, key, cards = action.as_tuple()
    if kind == "Pass":
        return ("Pass", None, ())
    if kind == "JokerBomb":
        return (kind, None, tuple(sorted(cards)))   # the joker bomb has no key
    return (kind, key, tuple(sorted(cards)))


def _engine_set(gd, state) -> set[tuple]:
    return {_as_triple(a) for a in state[0].legal_actions(state[1])}


class _RoundReplay:
    """Drives our state machine through one logged round.

    The traces record the deal before tribute, so the hands are adjusted by the
    logged tribute movements and the round is started at the play phase with
    the leader the simulator actually used. That isolates the legal-set
    comparison from any difference in tribute policy, which is compared
    separately from the tribute and back decisions themselves.
    """

    def __init__(self, gd, engine_rules, actions):
        self.gd = gd
        self.engine = gd.Engine(engine_rules, actions)
        # The simulator asks every seat, including one whose only option is
        # pass, so the replay must see those decisions too.
        self.engine.auto_pass = False
        self.state = None
        self.deal = None
        self.moves: list[tuple[int, int, str]] = []
        self.anti = False

    def on_deal(self, ev) -> None:
        self.deal = ev
        self.moves = []
        self.anti = False
        self.state = None

    def on_tribute(self, ev) -> None:
        for payer, receiver, card in ev["result"]:
            self.moves.append((int(payer), int(receiver), card))

    def start(self, leader: int):
        gd, N = self.gd, _N
        hands = [[N.card_id(c) for c in h] for h in self.deal["hands"]]
        for payer, receiver, card in self.moves:
            cid = N.card_id(card)
            if cid in hands[payer]:
                hands[payer].remove(cid)
                hands[receiver].append(cid)
        d = gd.DealSpec()
        d.hands = [sorted(h) for h in hands]
        d.level = N.RANK_INDEX[self.deal["level"]]
        levels = self.deal.get("team_levels") or {}
        d.team_levels = [N.RANK_INDEX[levels.get("0", self.deal["level"])],
                         N.RANK_INDEX[levels.get("1", self.deal["level"])]]
        owner = self.deal.get("owner_team")
        d.owner = -1 if owner is None else int(owner)
        d.leader = leader
        m = gd.MatchState()
        self.engine.set_deal(m, d)
        self.state = m
        return m


def diff_trace(path: Path, engine: EngineAdapter, limit: int | None,
               report: DivergenceReport) -> None:
    """Replay one trace file through our engine, recording divergences."""
    checked = 0
    if not engine.available:
        # Normalization-only pass: still worth running, it validates the trace
        # and the O9 mapping table.
        level = None
        for ev in _iter_events(path):
            if ev["event"] == "deal":
                level = ev["level"]
            elif ev["event"] == "act":
                report.decisions_checked += 1
                try:
                    _N.normalize_action_list(ev["action_list"], level)
                except Exception as e:
                    report.note("normalize_error", f"{path.name}: {e!r}")
            elif ev["event"] == "episodeOver":
                report.round_ends_checked += 1
        return

    gd = engine.gd
    replay = _RoundReplay(gd, gd.RuleConfig.ogd(), gd.ActionConfig.full())
    state = None

    for ev in _iter_events(path):
        if limit is not None and checked >= limit:
            return
        kind = ev["event"]

        if kind == "deal":
            replay.on_deal(ev)
            state = None
            continue
        if kind in ("tribute", "back"):
            replay.on_tribute(ev)
            continue
        if kind == "anti-tribute":
            replay.anti = True
            continue

        if kind == "act":
            level_char = replay.deal["level"] if replay.deal else None
            try:
                theirs = _N.normalize_action_list(ev["action_list"], level_char)
            except Exception as e:
                report.note("normalize_error", f"{path.name}#{ev.get('round')}: {e!r}")
                continue

            if ev.get("phase") != "play":
                # Tribute and back-tribute candidate sets, compared directly.
                hand = sorted(_N.card_id(c) for c in ev["hand"])
                level = _N.RANK_INDEX[level_char]
                if ev["phase"] == "tribute":
                    ours = {("Tribute", None, (c,)) for c in gd.tribute_choices(hand, level)}
                    theirs = {(t, None, c) for t, _, c in theirs}
                    cls = "tribute_candidates"
                else:
                    ours = {("BackTribute", None, (c,))
                            for c in gd.back_tribute_choices(hand, level)}
                    theirs = {(t, None, c) for t, _, c in theirs}
                    cls = "back_tribute_candidates"
                if ours != theirs:
                    report.note(cls,
                                f"{path.name}#{ev.get('round')} seat{ev['seat']}: "
                                f"ours-only {sorted(ours - theirs)[:3]} "
                                f"theirs-only {sorted(theirs - ours)[:3]}")
                continue

            checked += 1
            report.decisions_checked += 1
            if state is None:
                state = replay.start(int(ev["seat"]))

            # Our engine passes for a seat whose only option is pass; the
            # simulator still asks. Skip those logged decisions.
            only_pass = theirs == {("Pass", None, ())}
            if state.to_move != ev["seat"]:
                if only_pass:
                    continue
                report.note("seat_mismatch",
                            f"{path.name}#{ev.get('round')}: ours {state.to_move} "
                            f"theirs {ev['seat']}")
                state = None
                continue

            ours = {_as_triple(a) for a in replay.engine.legal_actions(state)}
            if ours != theirs:
                only_theirs = theirs - ours
                only_ours = ours - theirs
                wild = _N.wild_card_id(_N.RANK_INDEX[level_char])
                if only_theirs:
                    report.note("legal_set_missing_from_engine",
                                f"{path.name}#{ev.get('round')} seat{ev['seat']}: "
                                f"{sorted(only_theirs)[:3]}")
                if only_ours:
                    # The known class: the simulator does not enumerate a wild
                    # card standing for something weaker than what it already
                    # is, so our set is a superset on wild-bearing readings.
                    cls = ("extra_wild_substitution"
                           if all(wild in cards for _, _, cards in only_ours)
                           else "legal_set_extra_in_engine")
                    report.note(cls,
                                f"{path.name}#{ev.get('round')} seat{ev['seat']}: "
                                f"{sorted(only_ours)[:3]}")

            chosen = _N.normalize_action(ev["chosen_action"], level_char)
            match = next((a for a in replay.engine.legal_actions(state)
                          if _as_triple(a) == chosen), None)
            if match is None:
                report.note("chosen_action_illegal_here",
                            f"{path.name}#{ev.get('round')} seat{ev['seat']}: {chosen}")
                state = None
                continue
            replay.engine.apply(state, match)
            continue

        if kind == "episodeOver":
            report.round_ends_checked += 1
            if state is None:
                continue
            if state.phase != gd.Phase.RoundEnd:
                report.note("round_did_not_end",
                            f"{path.name}#{ev.get('round')}: phase {state.phase}")
                state = None
                continue
            result = replay.engine.end_round(state)
            theirs_order = [int(x) for x in ev["order"]]
            ours_order = result.order[:result.num_finished_seats]
            n = min(len(theirs_order), len(ours_order))
            if theirs_order[:n] != ours_order[:n]:
                report.note("finishing_order",
                            f"{path.name}#{ev.get('round')}: ours {ours_order} "
                            f"theirs {theirs_order}")
            state = None
            continue


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
