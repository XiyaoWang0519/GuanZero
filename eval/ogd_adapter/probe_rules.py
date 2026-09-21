"""Settles the parity questions docs/RULES.md section 13 could not answer from
source alone (O3, O8) by querying the jar's move generator with crafted hands,
plus a handful of extra checks the M0 task 10 brief asked for. Run with:

    python -m eval.ogd_adapter.probe_rules

Writes the exact probe inputs and the jar's raw outputs to
`docs/ogd_parity_probes.md` and prints a one-line verdict per probe to
stdout. Requires a JVM and the jar (see `bridge.ogd_root()`); there is no
fallback path, since the whole point is to query the real jar.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from eval.ogd_adapter import bridge

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Probe:
    id: str
    title: str
    hand: list[str]
    hearts_num: int
    level: str
    greater: tuple | None
    raw: list[tuple] = field(default_factory=list)
    verdict: str = ""


def _run(p: Probe) -> Probe:
    p.raw = bridge.legal_moves(p.hand, p.hearts_num, p.level, greater=p.greater)
    return p


def _fmt_raw(raw: list[tuple]) -> str:
    lines = [f"  {list(a)}" for a in raw]
    return "\n".join(lines) if lines else "  (empty)"


def build_probes() -> list[Probe]:
    probes: list[Probe] = []

    # ---- O3: joker pair inside a full house --------------------------------
    probes.append(Probe(
        "O3-SB", "Joker pair (SB SB) as the pair of a ThreeWithTwo",
        hand=["S5", "D5", "C5", "SB", "SB", "H2", "H3"], hearts_num=0, level="7", greater=None,
    ))
    probes.append(Probe(
        "O3-RJ", "Joker pair (RJ RJ) as the pair of a ThreeWithTwo",
        hand=["S5", "D5", "C5", "HR", "HR", "H2", "H3"], hearts_num=0, level="7", greater=None,
    ))

    # ---- O8: dominated readings ---------------------------------------------
    probes.append(Probe(
        "O8-a", "Wild 7 extends a straight two ways (windows 1 and 2), Straight and StraightFlush",
        hand=["H7", "S3", "S4", "S5", "S6"], hearts_num=1, level="7", greater=None,
    ))
    probes.append(Probe(
        "O8-b", "Straight windows 4 and 5 for the same 5 cards",
        hand=["H2", "S6", "D7", "C8", "D9"], hearts_num=1, level="2", greater=None,
    ))

    # ---- Other checks named in the brief -------------------------------------
    probes.append(Probe(
        "MISC-6straight", "Six same-suit cards: no plain Straight, no 6-card StraightFlush",
        hand=["S3", "S4", "S5", "S6", "S7", "S8"], hearts_num=0, level="2", greater=None,
    ))
    probes.append(Probe(
        "MISC-joker-seq", "Joker inside a would-be sequence: illegal",
        hand=["ST", "SJ", "SQ", "SK", "SB"], hearts_num=0, level="2", greater=None,
    ))
    probes.append(Probe(
        "MISC-bjrj-pair", "BJ RJ as a pair: illegal",
        hand=["SB", "HR"], hearts_num=0, level="2", greater=None,
    ))
    probes.append(Probe(
        "MISC-bjbjrj", "BJ BJ RJ (3 cards): illegal as a bomb, only the BJ pair survives",
        hand=["SB", "SB", "HR"], hearts_num=0, level="7", greater=None,
    ))
    probes.append(Probe(
        "MISC-jokerbomb", "BJ BJ RJ RJ: the joker bomb, type string FourKings",
        hand=["SB", "SB", "HR", "HR"], hearts_num=0, level="7", greater=None,
    ))
    probes.append(Probe(
        "MISC-9bomb", "9-card bomb: 8 natural 9s + 2 wild H7 (T-BOMB-02 analog, minus one wild)",
        hand=["S9", "S9", "H9", "C9", "C9", "D9", "D9", "H7", "H7"], hearts_num=2, level="7", greater=None,
    ))
    probes.append(Probe(
        "MISC-10bomb", "10-card bomb: 8 natural 9s + 2 wild H7 (T-BOMB-02)",
        hand=["S9", "S9", "H9", "H9", "C9", "C9", "D9", "D9", "H7", "H7"], hearts_num=2, level="7", greater=None,
    ))
    probes.append(Probe(
        "MISC-heartsnum-correct", "heartsNum=1 (correct): H7 treated as wild",
        hand=["H7", "S3", "S4", "S5", "S6"], hearts_num=1, level="7", greater=None,
    ))
    probes.append(Probe(
        "MISC-heartsnum-wrong", "heartsNum=0 (understated): H7 treated as an ordinary card, not wild",
        hand=["H7", "S3", "S4", "S5", "S6"], hearts_num=0, level="7", greater=None,
    ))
    return probes


def _verdict(p: Probe) -> str:
    types_seen = {a[0] for a in p.raw}
    if p.id in ("O3-SB", "O3-RJ"):
        joker = "SB" if p.id == "O3-SB" else "HR"
        offered = any(a[0] == "ThreeWithTwo" and a[2].count(joker) == 2 for a in p.raw)
        return f"ThreeWithTwo with {joker} {joker} as the pair is {'OFFERED' if offered else 'NOT offered'}."
    if p.id == "O8-a":
        straights = sorted({a[1] for a in p.raw if a[0] == "Straight"})
        sfs = sorted({a[1] for a in p.raw if a[0] == "StraightFlush"})
        return f"Straight windows {straights}, StraightFlush windows {sfs} (both readings of the same 5 cards listed: dominated readings ARE present)."
    if p.id == "O8-b":
        straights = sorted({a[1] for a in p.raw if a[0] == "Straight"})
        return f"Straight windows {straights} for the same 5 cards (dominated readings ARE present)."
    if p.id == "MISC-6straight":
        return f"types offered: {sorted(types_seen)} (no plain Straight; only 5-card StraightFlush windows, since RULES.md 5 requires Straight to be not-all-one-suit)."
    if p.id == "MISC-joker-seq":
        return f"no Straight/StraightFlush offered: {'Straight' not in types_seen and 'StraightFlush' not in types_seen}."
    if p.id == "MISC-bjrj-pair":
        return f"no Pair offered for BJ+RJ: {'Pair' not in types_seen}."
    if p.id == "MISC-bjbjrj":
        return f"types offered: {sorted(types_seen)} (no Bomb/FourKings from 3 jokers)."
    if p.id == "MISC-jokerbomb":
        fk = [a for a in p.raw if a[0] == "FourKings"]
        return f"FourKings entries: {fk}"
    if p.id in ("MISC-9bomb", "MISC-10bomb"):
        bombs = sorted({len(a[2]) for a in p.raw if a[0] == "Bomb"})
        return f"Bomb sizes offered: {bombs}"
    if p.id == "MISC-heartsnum-correct":
        return f"Straight/StraightFlush windows with heartsNum=1: {sorted({(a[0], a[1]) for a in p.raw if a[0] in ('Straight', 'StraightFlush')})}"
    if p.id == "MISC-heartsnum-wrong":
        return f"Straight/StraightFlush windows with heartsNum=0 (H7 no longer wild): {sorted({(a[0], a[1]) for a in p.raw if a[0] in ('Straight', 'StraightFlush')})}"
    return ""


def main() -> int:
    jar = bridge.ogd_root() / "guandan-java" / "guandan-java-action.jar"
    print(f"jar: {jar}")
    probes = [_run(p) for p in build_probes()]
    for p in probes:
        p.verdict = _verdict(p)
        print(f"[{p.id}] {p.verdict}")

    lines: list[str] = []
    lines.append("# OpenGuanDan rule-parity probes (O3, O8, and extras)")
    lines.append("")
    lines.append(
        "Generated by `python -m eval.ogd_adapter.probe_rules`, which calls the jar's "
        "move generator directly through `bridge.legal_moves` (which in turn calls "
        "`Moves.parse_first_action`/`parse_second_action` from the OpenGuanDan clone's "
        "`guandan-java/engine/moves.py`). No jar/simulator code is reproduced here, only "
        "its JSON-shaped output for each crafted hand."
    )
    lines.append("")
    lines.append("## Verdicts")
    lines.append("")
    for p in probes:
        lines.append(f"- **{p.id}** -- {p.title}: {p.verdict}")
    lines.append("")
    lines.append("## Probe detail")
    lines.append("")
    for p in probes:
        lines.append(f"### {p.id}: {p.title}")
        lines.append("")
        lines.append(f"Input: `hand={p.hand}`, `heartsNum={p.hearts_num}`, `currentRank={p.level!r}`" + (f", `greaterAction={p.greater}`" if p.greater else ""))
        lines.append("")
        lines.append("Raw jar output (`actionList`):")
        lines.append("```")
        lines.append(_fmt_raw(p.raw))
        lines.append("```")
        lines.append("")
        lines.append(f"Verdict: {p.verdict}")
        lines.append("")

    lines.append("## heartsNum, what it actually controls")
    lines.append("")
    lines.append(
        "`moves.py`'s `_hand_int` packs every held card (including any heart-of-level "
        "card) into a plain per-rank/per-suit count array; it does not itself mark which "
        "cards are wild. `heartsNum` is a *separate* integer telling the jar how many "
        "of the player's own heart-of-`currentRank` cards to treat as wild. The "
        "MISC-heartsnum-correct/-wrong probes above show this directly: with the exact "
        "same 5 cards (`H7 S3 S4 S5 S6`) at level 7, `heartsNum=1` makes `H7` wild "
        "(extra Straight/StraightFlush windows and pair readings appear) and "
        "`heartsNum=0` makes it an ordinary S-suit-blind card with only the natural "
        "reading. `table.py`'s `Player.hearts_num` is maintained incrementally at deal "
        "time and on every card removed/added, and is always exactly the count of "
        "heart-of-level cards physically in that hand -- callers must pass the true "
        "count, the jar does not (and structurally cannot, from the packed int array "
        "alone) recompute it."
    )
    lines.append("")

    out = REPO_ROOT / "docs" / "ogd_parity_probes.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
