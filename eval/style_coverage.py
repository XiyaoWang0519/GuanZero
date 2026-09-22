"""Coverage report for a styled belief collection.

Reads a collection directory written by `eval/collect_belief.py` and produces
`coverage.json` plus a short `coverage.md`:

* a histogram of every style slot over the sampled matches,
* behaviour histograms (bomb timing, play-type frequency, lead rank) per
  style-slot bin, so a slot that does nothing is visible,
* a disjointness check between the training and held-out style regions.

This is the coverage-report acceptance of task 2 in `docs/M2_TODO.md`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train import styles as style_lib
from train.logs import DRIVER_BOT

BINS = 5


def _style_space(provenance: dict) -> style_lib.StyleSpace:
    """Rebuild the recorded style space, falling back to the current default."""
    description = provenance.get("style_space") or {}
    bounds = description.get("bounds")
    rule = description.get("heldout_rule")
    if not bounds or not rule:
        return style_lib.StyleSpace.default()
    names = tuple(description["slots"])
    lower = tuple(float(bounds[n][0]) for n in names)
    upper = tuple(float(bounds[n][1]) for n in names)
    heldout = tuple((n, float(v[0]), float(v[1])) for n, v in sorted(rule.items()))
    return style_lib.StyleSpace(dim=int(description["dim"]), names=names,
                                lower=lower, upper=upper, heldout=heldout)


def _bot_styles(match: dict) -> list[tuple[int, list]]:
    """Style vectors of the bot-driven seats of one match."""
    return [(seat, match["styles"][seat]) for seat in range(4)
            if match["seat_driver"][seat] == DRIVER_BOT]


def _finite(vector: list) -> np.ndarray | None:
    array = np.asarray([np.nan if x is None else float(x) for x in vector], dtype=np.float64)
    return None if not np.isfinite(array).all() else array


def slot_histograms(space: style_lib.StyleSpace, vectors: list[np.ndarray]) -> dict:
    """Per-slot histogram and summary statistics over the sampled styles."""
    out: dict[str, dict] = {}
    if not vectors:
        return out
    stack = np.stack(vectors)
    for index, name in enumerate(space.names):
        column = stack[:, index]
        edges = np.linspace(space.lower[index], space.upper[index], BINS + 1)
        counts, _ = np.histogram(column, bins=edges)
        out[name] = {
            "samples": int(column.size), "min": float(column.min()),
            "max": float(column.max()), "mean": float(column.mean()),
            "bin_edges": [float(e) for e in edges],
            "counts": [int(c) for c in counts],
            "empty_bins": int((counts == 0).sum()),
        }
    return out


def behaviour_by_slot_bin(space: style_lib.StyleSpace, matches: list[dict]) -> dict:
    """Behaviour histograms grouped by the bin of each style slot."""
    samples: list[tuple[np.ndarray, dict]] = []
    for match in matches:
        for seat, vector in _bot_styles(match):
            value = _finite(vector)
            counts = (match.get("behaviour") or {}).get(str(seat))
            if value is not None and counts:
                samples.append((value, {str(seat): counts}))
    out: dict[str, dict] = {}
    for index, name in enumerate(space.names):
        edges = np.linspace(space.lower[index], space.upper[index], BINS + 1)
        grouped: dict[str, list[dict]] = {}
        for value, counts in samples:
            slot = int(np.clip(np.digitize([value[index]], edges[1:-1])[0], 0, BINS - 1))
            label = f"[{edges[slot]:.3g}, {edges[slot + 1]:.3g})"
            grouped.setdefault(label, []).append(counts)
        out[name] = {label: style_lib.merge_behaviour(parts)
                     for label, parts in sorted(grouped.items())}
    return out


def region_check(space: style_lib.StyleSpace, matches: list[dict]) -> dict:
    """Confirm the training and held-out style regions never overlap."""
    labelled: dict[str, int] = {}
    violations: list[dict] = []
    unknown = 0
    for match in matches:
        for seat, vector in _bot_styles(match):
            value = _finite(vector)
            if value is None:
                unknown += 1
                labelled["unknown"] = labelled.get("unknown", 0) + 1
                continue
            region = style_lib.region_of(space, value)
            labelled[region] = labelled.get(region, 0) + 1
            if match.get("region") not in (region, "mixed"):
                violations.append({"env_id": match["env_id"], "match_id": match["match_id"],
                                   "seat": seat, "labelled": match.get("region"),
                                   "actual": region})
            if not space.in_bounds(value):
                violations.append({"env_id": match["env_id"], "match_id": match["match_id"],
                                   "seat": seat, "error": "outside the style box"})
    return {
        "seat_styles_by_region": labelled, "unknown_styles": unknown,
        "heldout_rule": {n: [lo, hi] for n, lo, hi in space.heldout},
        "disjoint": not violations,
        "violations": violations[:20],
        "note": ("training styles are rejection-sampled outside the held-out box, "
                 "so a training style inside it would be a sampling bug"),
    }


def build_report(directory: Path) -> dict:
    """Assemble the JSON coverage report for one collection directory."""
    provenance = json.loads((directory / "provenance.json").read_text())
    matches_path = directory / "matches.json"
    matches = json.loads(matches_path.read_text()) if matches_path.exists() else []
    space = _style_space(provenance)
    vectors = [v for match in matches for _, raw in _bot_styles(match)
               if (v := _finite(raw)) is not None]
    return {
        "schema_version": 1,
        "collection": str(directory),
        "status": provenance.get("status"),
        "styled": provenance.get("styled", False),
        "styled_bot_unavailable": provenance.get("styled_bot_unavailable", False),
        "style_region": provenance.get("style_region"),
        "style_seed": provenance.get("style_seed"),
        "collected_rounds": provenance.get("collected_rounds"),
        "collected_decisions": provenance.get("collected_decisions"),
        "matches_sampled": len(matches),
        "bot_seat_styles": len(vectors),
        "style_space": space.describe(),
        "slot_histograms": slot_histograms(space, vectors),
        "behaviour_by_slot_bin": behaviour_by_slot_bin(space, matches),
        "region_check": region_check(space, matches),
        "behaviour_total": provenance.get("behaviour_total", {}),
    }


def render_markdown(report: dict) -> str:
    """Short human-readable summary of the JSON report."""
    lines = ["# Style coverage report", "",
             f"Collection: `{report['collection']}`  ",
             f"Status: {report['status']}; styled: {report['styled']}; "
             f"region: {report['style_region']}; style seed: {report['style_seed']}  ",
             f"Rounds: {report['collected_rounds']}; decisions: {report['collected_decisions']}; "
             f"matches: {report['matches_sampled']}; bot seat styles: {report['bot_seat_styles']}",
             ""]
    if report["styled_bot_unavailable"]:
        lines += ["> **The styled-bot binding was unavailable.** Opponent seats played",
                  "> greedily and their style vectors are recorded as unknown. The slot",
                  "> histograms below are therefore empty by construction.", ""]
    lines += ["## Style-slot histograms", "",
              "| Slot | Samples | Min | Mean | Max | Counts | Empty bins |",
              "|---|---:|---:|---:|---:|---|---:|"]
    for name, entry in report["slot_histograms"].items():
        lines.append(f"| {name} | {entry['samples']} | {entry['min']:.3g} | "
                     f"{entry['mean']:.3g} | {entry['max']:.3g} | "
                     f"{' '.join(str(c) for c in entry['counts'])} | {entry['empty_bins']} |")
    if not report["slot_histograms"]:
        lines.append("| (no sampled styles) | 0 | | | | | |")
    total = report.get("behaviour_total") or {}
    lines += ["", "## Behaviour totals", "",
              f"Plays {total.get('plays', 0)}, passes {total.get('passes', 0)}, "
              f"bombs {total.get('bombs', 0)}, leads {total.get('leads', 0)}.  ",
              f"Bomb fraction {total.get('bomb_fraction')}, mean bomb step "
              f"{total.get('mean_bomb_step')}, mean lead rank {total.get('mean_lead_rank')}.",
              "", "Play-type frequency: "
              + (", ".join(f"{k} {v}" for k, v in (total.get("type_frequency") or {}).items())
                 or "none"),
              "", "## Behaviour by style-slot bin", ""]
    for name, bins in report["behaviour_by_slot_bin"].items():
        if not bins:
            continue
        lines += [f"### {name}", "", "| Bin | Plays | Bomb fraction | Mean bomb step | Mean lead rank |",
                  "|---|---:|---:|---:|---:|"]
        for label, counts in bins.items():
            def fmt(value: float | None) -> str:
                return "n/a" if value is None else f"{value:.4g}"
            lines.append(f"| {label} | {counts['plays']} | {fmt(counts['bomb_fraction'])} | "
                         f"{fmt(counts['mean_bomb_step'])} | {fmt(counts['mean_lead_rank'])} |")
        lines.append("")
    check = report["region_check"]
    lines += ["## Train/held-out disjointness", "",
              f"Disjoint: **{check['disjoint']}**  ",
              f"Held-out rule: {json.dumps(check['heldout_rule'])}  ",
              f"Seat styles by region: {json.dumps(check['seat_styles_by_region'])}  ",
              f"Violations: {len(check['violations'])}", ""]
    return "\n".join(lines) + "\n"


def write_report(directory: Path, output: Path | None = None) -> dict:
    """Write `coverage.json` and `coverage.md` and return the JSON report."""
    report = build_report(directory)
    target = directory if output is None else output
    target.mkdir(parents=True, exist_ok=True)
    (target / "coverage.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (target / "coverage.md").write_text(render_markdown(report))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = write_report(args.collection, args.output)
    print(json.dumps({"matches_sampled": report["matches_sampled"],
                      "bot_seat_styles": report["bot_seat_styles"],
                      "disjoint": report["region_check"]["disjoint"],
                      "styled_bot_unavailable": report["styled_bot_unavailable"]}))


if __name__ == "__main__":
    main()
