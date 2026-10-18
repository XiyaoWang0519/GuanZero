"""Paired, opponent-stratified match-seed bootstrap for frozen FULL/ROUND tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONDITIONS = (("control", "full"), ("control", "round"),
              ("norm16", "full"), ("norm16", "round"))
ENDPOINTS = dict(control="a3d5e42ea6f6dc9a4b514f1bce948a061ee4be0b08c376dc8ce659ccf46948c7",
                 norm16="64d3fbd56bb7ea6968a4a54b3b7fd06f9d505c844920b2aa7c8636e4bcb22bdb")


def bootstrap(values, opponents, draws=20000):
    """Resample seeds within opponent strata, retaining both team swaps/all conditions."""
    rng = np.random.default_rng(16102672)
    boot = np.zeros((draws, values.shape[1]))
    for opponent in sorted(set(opponents)):
        indices = np.flatnonzero(opponents == opponent)
        sample = rng.choice(indices, size=(draws, len(indices)), replace=True)
        boot += values[sample].sum(1) / len(values)
    return boot


def interval(point, draws):
    return dict(estimate=float(point), ci95=np.quantile(draws, [.025, .975]).tolist(),
                ci975=np.quantile(draws, [.0125, .9875]).tolist())


def analyze(root):
    by_condition = {c: {} for c in CONDITIONS}
    metas = []
    seconds = {}
    schedule = np.random.default_rng(16102671).integers(1, 2**63-1, size=(16, 16))
    for wave in range(16):
        folder = root / f"wave-{wave:02d}"
        assert json.loads((folder/"complete.json").read_text())["complete"]
        meta = json.loads((folder/"metadata.json").read_text())
        assert meta["endpoints"] == ENDPOINTS
        assert meta["waves"] == 16 and meta["envs"] == 32
        assert meta["seed"] == 16102671
        assert meta["parent"] == "a4bed2bfb4c3b8c1382a8ed5604e8e797307a37ea2088523dffb1243de6a58bd"
        assert meta["model_source_sha"] == "8019503df4a65e88583081633ee89b1caaf4b5070b8ba0562c15ccf6d15a0bd4"
        metas.append(meta)
        rows = [json.loads(s) for s in (folder/"waves.jsonl").read_text().splitlines()]
        assert len(rows) == 4 and {(r["arm"], r["view"]) for r in rows} == set(CONDITIONS)
        for result in rows:
            assert result["wave"] == wave and len(result["records"]) == 32
            condition = (result["arm"], result["view"])
            seconds.setdefault("/".join(condition), []).append(result["seconds"])
            assert {r["env"] for r in result["records"]} == set(range(32))
            for r in result["records"]:
                assert r["seed"] == int(schedule[wave, r["env"]//2])
                assert r["team"] == r["env"] % 2
                assert r["opponent"] == sorted(int(i) for i in meta["opponents"])[(r["env"]//2)%4]
                key = (r["seed"], r["team"])
                assert key not in by_condition[condition] and r["win"] in (0, 1)
                assert r["rounds"] > 1 and r["decisions"] > 0
                by_condition[condition][key] = r
    for meta in metas[1:]:
        assert meta == metas[0], "source/config/opponent mismatch between workers"
    keys = sorted(by_condition[CONDITIONS[0]])
    assert len(keys) == 512
    assert all(sorted(records) == keys for records in by_condition.values())
    seeds = sorted({k[0] for k in keys})
    assert len(seeds) == 256
    opponents = np.asarray([by_condition[CONDITIONS[0]][(s, 0)]["opponent"] for s in seeds])
    for seed in seeds:
        for team in (0, 1):
            assert len({by_condition[c][(seed, team)]["opponent"] for c in CONDITIONS}) == 1
            for arm in ("control", "norm16"):
                assert (by_condition[(arm, "full")][(seed, team)]["first_round_sha"] ==
                        by_condition[(arm, "round")][(seed, team)]["first_round_sha"])
    values = np.asarray([[np.mean([by_condition[c][(s, t)]["win"] for t in (0, 1)])
                          for c in CONDITIONS] for s in seeds])
    boot = bootstrap(values, opponents)
    conditions = {}
    for i, c in enumerate(CONDITIONS):
        rows = list(by_condition[c].values())
        conditions["/".join(c)] = dict(matches=len(rows), wins=sum(r["win"] for r in rows),
            win_rate=interval(values[:, i].mean(), boot[:, i]),
            rounds=sum(r["rounds"] for r in rows),
            mean_rounds=float(np.mean([r["rounds"] for r in rows])),
            max_rounds=max(r["rounds"] for r in rows),
            mean_return_per_round_per_match=float(np.mean([r["total_return"]/r["rounds"] for r in rows])),
            mean_total_return=float(np.mean([r["total_return"] for r in rows])),
            decisions=sum(r["decisions"] for r in rows),
            summed_worker_seconds=float(sum(seconds["/".join(c)])))
    contrasts = {}
    for arm, full, round_ in (("control", 0, 1), ("norm16", 2, 3)):
        records_f, records_r = by_condition[(arm, "full")], by_condition[(arm, "round")]
        contrasts[arm] = dict(**interval((values[:, full]-values[:, round_]).mean(),
                                        boot[:, full]-boot[:, round_]),
            full_only_wins=sum(records_f[k]["win"] > records_r[k]["win"] for k in keys),
            round_only_wins=sum(records_f[k]["win"] < records_r[k]["win"] for k in keys),
            by_opponent={int(o): float((values[opponents == o, full]-
                                       values[opponents == o, round_]).mean())
                         for o in sorted(set(opponents))})
    return dict(complete=True, matches=2048, independent_seed_clusters=256,
        first_round_gate="all 1024 within-arm FULL/ROUND first-round hashes equal",
        metadata=metas[0], conditions=conditions, full_minus_round=contrasts,
        interaction=interval((values[:,2]-values[:,3]-values[:,0]+values[:,1]).mean(),
                             boot[:,2]-boot[:,3]-boot[:,0]+boot[:,1]),
        norm16_minus_control_full=interval((values[:,2]-values[:,0]).mean(),boot[:,2]-boot[:,0]),
        uncertainty="20000 paired seed-cluster bootstrap draws, stratified by frozen opponent; "
                    "ci975 Bonferroni for two primary FULL-minus-ROUND contrasts")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.root)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"full_minus_round":result["full_minus_round"],
                      "interaction":result["interaction"]}, indent=2))


if __name__ == "__main__":
    main()
