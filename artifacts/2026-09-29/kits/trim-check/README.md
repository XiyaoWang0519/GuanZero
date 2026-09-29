# Allocator cache trim: short CUDA check (2026-09-28)

Does `rollout_trim_cuda_cache` (torch.cuda.empty_cache() after collect and after
learn, commit 0b73d2c on branch `longrun-batched-2026-09-28`) stop the reserved
memory staircase seen with `--batch-snapshot-policies`, and what does it cost?
One diagnostic segment on one Vast RTX 4090. Prepared, not launched.

## What runs

- Resume `init.pt` = `.work/actor-ranks-2026-09-28/download/results/segments/main-w4/latest.pt`
  (update 844, 55.3M decisions; sha256 `b1c55037...12bd7`, checked by `lifecycle.py`).
- Layout of the merged-snapshot A/B (= main-w4): `train.history_ddp --world-size 4`,
  256 tables and 16-match minibatches per rank, `ddp_global_minibatch`, snapshot
  cadence 2, graph budgets 512/256 MB per rank, engine threads from the host
  (as in the A/B). Recipe and every CUDA rollout flag come from the checkpoint.
- `--resume-set` (all recorded in `config_changes`): `batch_snapshot_policies_schedule=10:off,30:on`,
  `rollout_trim_cuda_cache=auto` (follows the arm: trim off for the first 10
  updates, on for the last 30), `checkpoint_updates=1000`. Profiling off.
  `--allow-source-change` (recorded in `source_changes`).
- 40 updates, 844 -> 884, environments never restarted (~11-14 min of training).
- **Output `segments/trim-check-diag/` is a diagnostic branch** (marker file
  `DIAGNOSTIC_BRANCH.txt`). Never the main lineage endpoint, never a parent of a
  main segment. The main lineage still ends at main-w4 `latest.pt`.

## Gates (before training; any failure aborts, then sync and destroy)

The five gates of the A/B (triton, graphs, wide, learner, snapshot) plus `trim`:
`tests/test_history_trim.py` on CUDA with `-s`. Required to pass (not skip):
`test_trim_leaves_training_bitwise_unchanged_on_cuda` (trainer resumed twice,
merged arm on, trim on vs off: rows, choices, metrics, weights, optimizer and
sampler bitwise equal; reserved never grows across a trim, allocated unchanged),
`test_switching_the_arm_on_releases_snapshot_graph_pools`, and the CPU case.

## Summary (`trim-summary.md` / `.json`, written on the pod and again locally)

Per update: arm, wall s, decisions/s from wall time (global decisions over rank-0
elapsed), collect/learn s (slowest rank), nvidia-smi max and mean over the
update's 5 s samples, sums over ranks of peak allocated/reserved, reserved and
allocated before/after each trim point (`graphs_released`, `after_collect`,
`after_learn`), trim seconds (slowest rank), resident snapshots, entropy, KL,
allocation retries. Per arm: the same, first 2 updates of each arm dropped.

Verdict line: **FLAT** if, over the settled on-updates, the rise of the per-update
nvidia-smi maximum (mean of the last 10 minus the first 10) exceeds the rise of the
summed peak allocated memory by at most 256 MiB, and the smi slope exceeds the
allocated slope by at most 10 MiB/update; otherwise **STAIRCASE**. The on arm
follows a resume (histories and resident snapshots still grow for ~25 updates),
so live growth is expected and subtracted. It also reports the on-arm maximum
against this run's off-arm maximum (caveat: early after the resume) and against
the A/B's settled maxima on 67872 (off 21,250 MiB; untrimmed on 23,578 MiB),
and the trim cost in s/update and % of wall time. The rule was checked on
synthetic data (flat -> FLAT, +60 MiB/update excess -> STAIRCASE).

## Machine and safety

- Ordered list: machine **20082** (every earlier baseline) if it has an offer
  passing the filters, else **67872** (UK, Xeon 8173M, 26 usable CPUs; the A/B).
  Filters: 1x RTX 4090 24 GB, Vast Secure Cloud (datacenter), verified,
  reliability >= 0.99, <= $0.80/h, traffic <= $0.05/GB. The rented machine is in
  `instance.json`, `teardown-summary.json` and `results/machine.json`.
- Label `guanzero-trim-check-20260928`; refuses a duplicate rent; refuses to launch
  if credit < worst case at the ceiling + $0.25 ($0.60).
- Timeline from create: segment stopped at 21 min if not finished, final sync by
  22 min (or on DONE), local guard destroys at 24 min, pod stops itself at 24.5 min,
  hard cap 25 min. **Worst case at the $0.80/h ceiling: 0.80 x 1470 s = $0.327
  compute + <= $0.025 traffic = $0.35.** At 67872's quoted $0.557/h: $0.23.
  At 20082's last quote $0.783/h: $0.32 + traffic.
- Guard verify on the pod is read-only (GET); training under supervisord; results
  rsynced every 90 s; final download checked against `results-manifest.json`
  before destroy; destroy confirmed by the instance's absence in the API.

## Commands

```sh
cd /Users/xiyaowang/Developer/Projects/GuanZero/.work/trim-check-2026-09-28
PY=/Users/xiyaowang/Developer/Projects/GuanZero/.work/external/danlm-venv/bin/python
$PY lifecycle.py freeze --source-root <clean checkout of longrun-batched-2026-09-28>  # done
$PY lifecycle.py dry-run            # read-only offers + checks, rents nothing (done)
$PY lifecycle.py launch --yes-spend # only after the user approves the budget
$PY summarize.py download/results --out .   # re-run the summary by hand
```

## Local evidence (CPU only; the trim is a no-op without CUDA)

- `smoke/`: `trim_check.py --smoke --device cpu --ranks 2 --off 3 --on 6` on the
  tiny checkpoint `smoke-init/`: the arm and the trim switched on at the 4th
  update, trim points `after_collect`/`after_learn` recorded on both ranks,
  `config_changes` recorded, summary rendered.

## Unverified

The trim has never run on CUDA. Whether empty_cache under
`expandable_segments:True` actually lowers nvidia-smi memory, how long it takes
per call, and whether the CUDA bitwise gate passes are exactly what this check
measures.
