"""Capacity x opponent-pool 2x2 on the entropy-0.03 recipe.

Cells (seeds and update budget of the history-budget screen, so the
small/recent cell is that screen's entropy arm, reused as is):

    small / recent   width 64, 2 layers; last 4 snapshots        (reused)
    small / wide     + archive of snapshots every 32 updates, at most 16,
                       thinned to stay spread; half the snapshot seats draw from it
    large / recent   width 128, 4 layers, 8 heads
    large / wide     both

Primary: the capacity and pool main effects, each averaged over the other
factor, paired by seed and deal. Secondary: the interaction. Pod-side
execution, resume and the health gate are ``infra.history_budget_experiment``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import shutil
from pathlib import Path

from infra import history_budget_experiment as base
from infra.history_artifacts import ROOT, pack_source, sha256
from infra.history_pilot import write_json

REFERENCE = ROOT / ".work/history-budget-2026-09-27"
SEEDS = base.SEEDS
UPDATES = base.UPDATES
WIDE = dict(population_archive_every=32, population_archive_size=16, population_archive_share=0.5)
LARGE = dict(width=128, layers=4, heads=8)
CELLS = {"small_recent": {}, "small_wide": WIDE, "large_recent": LARGE,
         "large_wide": dict(LARGE, **WIDE)}
NEW_CELLS = ("small_wide", "large_recent", "large_wide")
# Preflight: 7.3 s/update for the large cells -> about 4.35 h per pod at $0.72/h.
CAPS = {"screen": (3.80, 5.1), "preflight": (0.45, 0.7)}
HOURLY_CAP = 0.75


def cell_config(seed: int, cell: str) -> dict:
    """The screen's entropy-arm configuration plus the cell's settings."""
    config = dict(base.screen_config(seed), **base.ARMS["entropy"])
    return dict(config, **CELLS[cell])


def run_script(kind: str, command: str) -> str:
    return base.run_script(*CAPS[kind], command, hourly_usd=HOURLY_CAP)


def lifecycle(kind: str) -> dict:
    budget, hours = CAPS[kind]
    return dict(proposed_budget_usd=budget, max_hours_from_create=hours, max_hourly_usd=HOURLY_CAP,
                status="sized from the preflight (7.3 s/update)")


def prepare(root: Path) -> dict:
    from train.history_ppo import HistoryPPOConfig

    if root.exists():
        raise ValueError("refusing to overwrite an existing experiment")
    reference = json.loads((REFERENCE / "campaign.json").read_text())
    root.mkdir(parents=True)
    shutil.copytree(REFERENCE / "evaluation", root / "evaluation")
    for name, digest in reference["evaluation_freeze"].items():
        if sha256(root / "evaluation" / name) != digest:
            raise ValueError("reference evaluation freeze changed")
    source = pack_source(root / "source.tar.gz")
    kits = []

    def kit_dir(name: str) -> tuple[Path, Path]:
        kit = root / name
        (kit / "payload").mkdir(parents=True)
        shutil.copy2(root / "source.tar.gz", kit / "source.tar.gz")
        shutil.copy2(ROOT / "infra/history_setup.sh", kit / "payload/setup.sh")
        return kit, kit / "payload"

    common = dict(source=source, tests=base.TESTS + ["tests/test_history_factorial_experiment.py"])
    for seed in SEEDS:
        kit, payload = kit_dir(f"screen-{seed}")
        jobs = {}
        for cell in NEW_CELLS:
            values = cell_config(seed, cell)
            HistoryPPOConfig(**values)
            jobs[cell] = dict(kind="fresh", config=values, target_updates=UPDATES)
        (payload / "run.sh").write_text(run_script("screen", (
            "python -m infra.history_budget_experiment workload "
            "--manifest /workspace/payload/run-manifest.json --output /workspace/results")))
        kits.append(base.write_manifest(kit, dict(id=f"history-factorial-{seed}", kind="screen",
                                                  seed=seed, jobs=jobs, lifecycle=lifecycle("screen"),
                                                  **common)))
    kit, payload = kit_dir("preflight")
    shutil.copy2(root / f"screen-{SEEDS[0]}/run-manifest.json", payload / "screen-manifest.json")
    (payload / "run.sh").write_text(run_script("preflight", (
        "python -m infra.history_budget_experiment preflight --updates 60 "
        "--manifest /workspace/payload/screen-manifest.json --output /workspace/results")))
    kits.append(base.write_manifest(kit, dict(id="history-factorial-preflight", kind="preflight",
                                              lifecycle=lifecycle("preflight"), **common)))
    campaign = dict(
        status="prepared; user approved a $12 total (option A), preflight included",
        prepared_at=datetime.now(timezone.utc).isoformat(),
        source=dict(revision=source["revision"], dirty=source["dirty"],
                    source_sha256=source["source_sha256"], archive_sha256=source["archive_sha256"]),
        base_recipe="history-budget entropy arm (T7 arm B + entropy 0.03)",
        seeds=list(SEEDS), updates=UPDATES, cells=CELLS,
        reused_cell=dict(small_recent=f"{REFERENCE.name} entropy arm, same seeds and budget"),
        primary_comparisons=["capacity main effect: mean(large_*) - mean(small_*)",
                             "pool main effect: mean(*_wide) - mean(*_recent)"],
        secondary=["interaction: (large_wide - large_recent) - (small_wide - small_recent)",
                   "each new cell minus small_recent", "full-match win rates", "segment-2 baseline"],
        primary_metric="net levels per round, 256 duplicate deals (512 rounds) vs b11-main",
        endpoint=f"final.pt at exactly {UPDATES} updates; never select by scores",
        decision_rule=reference["decision_rule"],
        caveat=("the reused cell ran earlier on other pods and under the earlier health gates; "
                "it never produced a ratio spike, so the gate did not act on it"),
        final_test="existing sealed final test remains unopened",
        kits=[str(k) for k in kits], evaluation_freeze=reference["evaluation_freeze"])
    write_json(root / "campaign.json", campaign)
    return campaign


def evaluate(root: Path) -> None:
    import numpy as np
    import torch
    from eval.history_entropy import measure
    from eval.history_frozen import evaluate as evaluate_frozen

    torch.set_num_threads(2)
    campaign = json.loads((root / "campaign.json").read_text())
    freeze = root / "evaluation/freeze.json"
    if sha256(freeze) != campaign["evaluation_freeze"]["freeze.json"]:
        raise ValueError("evaluation freeze changed after preparation")
    out = root / "results"
    out.mkdir(exist_ok=True)

    def endpoint(kit_root: Path, kit: str, job: str, name: str) -> dict:
        checkpoint = base.job_file(kit_root, kit, job, "final.pt")
        receipt = json.loads((checkpoint.parent / "receipt.json").read_text())
        if not receipt["complete"] or sha256(checkpoint) != receipt["final_sha256"]:
            raise ValueError(f"fixed endpoint incomplete or modified: {checkpoint}")
        output = out / name
        if output.exists():
            report = json.loads(output.read_text())
            if report["candidate_sha256"] == sha256(checkpoint) and report["freeze_sha256"] == sha256(freeze):
                return report["reports"]
        return evaluate_frozen(freeze, checkpoint, output)["reports"]

    reports, entropy = {}, {}
    for seed in SEEDS:
        reports[seed] = {"small_recent": endpoint(REFERENCE, f"screen-{seed}", "entropy",
                                                  f"{seed}-small_recent-final.json")}
        for cell in NEW_CELLS:
            reports[seed][cell] = endpoint(root, f"screen-{seed}", cell, f"{seed}-{cell}-final.json")
            entropy[f"{seed}/{cell}"] = measure(base.job_file(root, f"screen-{seed}", cell, "final.pt"))
    write_json(out / "entropy-strata.json", entropy)
    rng = np.random.default_rng(2026092741)
    scores = lambda cell, b: np.array([reports[s][cell][b]["duplicates"]["pair_scores"] for s in SEEDS])
    summary = dict(primary=campaign["primary_comparisons"], decision_rule=campaign["decision_rule"],
                   comparisons={})
    for b in reports[SEEDS[0]]["small_recent"]:
        cell = {c: scores(c, b) for c in CELLS}
        summary["comparisons"][b] = dict(
            capacity=base.paired((cell["large_recent"] + cell["large_wide"]) / 2,
                                 (cell["small_recent"] + cell["small_wide"]) / 2, rng),
            pool=base.paired((cell["small_wide"] + cell["large_wide"]) / 2,
                             (cell["small_recent"] + cell["large_recent"]) / 2, rng),
            interaction=base.paired(cell["large_wide"] - cell["large_recent"],
                                    cell["small_wide"] - cell["small_recent"], rng),
            **{f"{c}-small_recent": base.paired(cell[c], cell["small_recent"], rng) for c in NEW_CELLS})
    write_json(root / "comparison.json", summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "evaluate"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    (prepare if args.command == "prepare" else evaluate)(args.output.resolve())


if __name__ == "__main__":
    main()
