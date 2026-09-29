# Overnight run: main lineage from main-w4, 8 h (2026-09-28)

Continues the **main lineage** from main-w4 `latest.pt` (update 844, 55.3M
decisions, sha256 `b1c55037...12bd7`; copied here as `init.pt`, checked by
`lifecycle.py`) for 8 hours of training on one Vast RTX 4090. It never resumes
from a diagnostic checkpoint (profile-diag, snapshot-batching-diag,
trim-check-diag, branch-s8). Prepared, not launched.

## What runs, and what differs from main-w4

- `train.history_ddp --world-size 4 --resume init.pt --allow-source-change`
  (source change recorded in `source_changes`). Source: branch
  `longrun-batched-2026-09-28` (frozen in `source.tar.gz`).
- Model, recipe and layout all come from the checkpoint, unchanged: width 128,
  4 layers, 8 heads; lr 3e-4, entropy 0.03, 2 epochs, auxiliary response 0.1,
  snapshot cadence 2, recent-4 pool; 256 tables and 16-match minibatches per rank,
  `ddp_global_minibatch`, 6 engine threads per rank, graph budgets 512/256 MB,
  every CUDA rollout flag; checkpoint every 31 updates (~2.0M decisions). No
  recipe or layout field is passed.
- The **only** difference from main-w4 is the launch-time `--mode`:

  | mode | `--resume-set` | recorded in `config_changes` |
  |---|---|---|
  | `batched` | `batch_snapshot_policies=true`, `rollout_trim_cuda_cache=auto` | `batch_snapshot_policies: [false, true]` |
  | `plain` | `batch_snapshot_policies=false`, `rollout_trim_cuda_cache=auto` | nothing (the main-w4 setting) |

  `auto` trims the allocator cache after collect and after learn exactly when the
  merged arm is on. Both fields are acceptance tier 1-2 (speed, no recipe change).
  Choose the mode after the short trim check (`.work/trim-check-2026-09-28/`).
- Output: `results/segments/main-overnight/` (rank 0; ranks 1-3 in `rank-r/`), with
  `update-NNNNNN.pt` every 31 updates and `latest.pt`.
- Profiling off. No evaluation on the pod and no local evaluation loop.

## Protections

- **Gates before training** (`remote_setup.sh`): triton, graphs, wide, learner, and
  in batched mode also snapshot and trim (all six always run; plain mode requires
  the first four). A required gate failing aborts: no training, the results are
  synced and the instance is destroyed.
- **Crash -> resume** (`longrun.py`): a trainer that exits non-zero is restarted
  from `segments/main-overnight/latest.pt` (init.pt if no checkpoint yet), at most
  6 restarts, backoff 30 s, 60 s, 120 s, ... capped at 300 s, no restart within
  10 min of the training deadline. Every start/exit/restart is in
  `orchestrator.jsonl`; each attempt has its own log `segments/main-overnight-attempt-N.log`.
  A crash repeats at most the updates since the last checkpoint (<= 31, ~8 min).
- **OOM fallback**: if a batched-mode attempt's log shows a CUDA out-of-memory
  error, every later resume sets `batch_snapshot_policies=false` (event `fallback`
  in `orchestrator.jsonl`; the change is in the checkpoint's `config_changes`), and
  the run continues at the slower main-w4 setting instead of crash-looping.
- **Stop on non-finite loss** (policy, value, entropy or KL): training stops for
  good, the checkpoints stay, the run ends as normal (sync, teardown). Also stops
  cleanly if the pod disk has < 3 GB free (~7 GB of checkpoints expected on 50 GB).
  No entropy/KL band stop.
- GPU memory sampler every 5 s (`gpu-samples.jsonl`); expandable segments on.

## Timeline and cost (from create)

| time | event |
|---|---|
| ~1 min, up to 25 min | instance reachable (a cold host first pulls the ~30 GB image; `waiting_for_ssh` status logged every minute); kit uploaded; supervisord starts the run. Not reachable by 25 min, or any failure before the run starts -> **destroyed**, absence confirmed |
| +3-7 min | build + gates done, training starts (8 h, cut at 8 h 14 m from create: >= ~7 h 42 m after a 25 min wait) |
| start + 8 h 00 m, never after 8 h 14 m | training stops at an update boundary; rank 0 saves `latest.pt` |
| ~+2 min | summary, results manifest, `DONE`; the Mac downloads, verifies, destroys |
| 8 h 20 m | the Mac stops waiting for `DONE` and asks the pod to stop the run |
| **8 h 25 m** | **local guard destroys the instance** |
| 8 h 26 m (or 20 min after `DONE`) | **pod stops itself** |
| 8 h 30 m | hard cap |

- Machine: ordered list **20082**, then **67872**, with the same filters (1x RTX
  4090 24 GB, Vast Secure Cloud / datacenter, verified, reliability >= 0.99,
  <= $0.80/h, traffic <= $0.035/GB). The rented machine is recorded in
  `instance.json`, `teardown-summary.json` and `download/results/machine.json`.
- **Worst case at the $0.80/h ceiling: 0.80 x 8.433 h = $6.75 + traffic <= 7 GB x
  $0.035 = $0.25 -> $6.99 (< $7.00).** At 67872's quoted $0.557/h: $4.70 + $0.03 =
  $4.73. At 20082's last quote $0.783/h: $6.60 + $0.23 = $6.83. Plus storage if
  the instance is left stopped (below).
- Launch refuses if credit < worst case + $0.50 ($7.49).
- Expected size: ~15-21 s per update -> ~1,400-1,900 updates, ~90-125M decisions
  added; ~45-60 checkpoints x ~105 MB = 5-7 GB, on the pod (50 GB disk) and
  locally in `download/results/` (255 GB free when prepared).

## If the Mac sleeps or goes offline

`launch` runs under `caffeinate -dimsu` (display, idle and system sleep blocked
while it runs). caffeinate cannot keep a MacBook awake on battery with the lid
closed: **keep it on power, lid open**, and do not log out.

- **What keeps running:** the pod trains on its own (supervisord); the Mac only
  downloads results and destroys the instance.
- **What stops the spending:** the pod-side guard stops the instance at 8 h 26 m, or
  20 min after `DONE` (so at ~8 h 30 m at the latest, normally ~8 h 10 m + 20 min).
  **It can only stop, not destroy**: it uses the container's own API key; destroying
  needs the account key, which stays on the Mac. A stopped instance ends GPU
  billing but keeps its 50 GB disk and bills storage: at the quoted storage
  prices about $0.037/h (67872, $0.53/GB-month) or $0.023/h (20082), i.e. about
  $0.55-0.90 per day, until destroyed. The results stay on that disk.
- **What is lost:** nothing is lost while the instance exists. What is not yet on
  the Mac is everything written since the Mac fell asleep (checkpoints, logs).
  When the Mac wakes, the local side sees the pod stopped with an unverified
  download of a run that was started on the pod (`remote-started.json`) and does **not** destroy it (event `left_stopped_for_recovery`,
  `RECOVER.md`). Then run `lifecycle.py recover --yes-spend`: it restarts the
  instance, downloads and verifies, and destroys it (bounded: 15 min to become
  reachable; if it cannot be reached it is stopped again). A restart can fail if
  someone else has rented that GPU meanwhile; the disk then stays until the
  machine is free, or `lifecycle.py destroy` gives the results up.
- If the Mac is only briefly away and the pod is still running when it returns,
  the normal loop just continues (syncs catch up).

## Commands

```sh
cd /Users/xiyaowang/Developer/Projects/GuanZero/.work/longrun-batched-2026-09-28
PY=/Users/xiyaowang/Developer/Projects/GuanZero/.work/external/danlm-venv/bin/python
$PY lifecycle.py freeze --source-root <clean checkout of longrun-batched-2026-09-28>  # done
$PY lifecycle.py dry-run --mode batched     # done
$PY lifecycle.py dry-run --mode plain       # done
# launch: exactly one of (user action, after the trim check decides)
$PY lifecycle.py launch --yes-spend --mode batched
$PY lifecycle.py launch --yes-spend --mode plain
```

Run `launch` in a terminal that stays open (it prints JSON events; `lifecycle.jsonl`
keeps them).

## Checking status in the morning

```sh
tail -5 lifecycle.jsonl                 # last events: teardown / absent_confirmed?
cat teardown-summary.json               # absence_confirmed, elapsed, cost estimate, balances
cat artifact-verification.json          # verified: true
cat longrun-summary.md                  # per-31-update windows and totals
ls RECOVER.md 2>/dev/null && cat RECOVER.md   # only if the pod stopped before the download
$PY lifecycle.py status                 # {"owned": null, ...} when nothing is left
```

`download/results/orchestrator.jsonl` has restarts and a `fallback` event if one
happened; `download/results/segments/main-overnight/latest.pt` is the new main
lineage endpoint. Strength evaluation is a separate later step.

## Local evidence (CPU only)

`smoke-batched/` and `smoke-plain/`: `longrun.py --smoke` on a tiny checkpoint
(`smoke-init/`, 2-update checkpoints) with 2 ranks.
- batched with `--simulate crash@8 --simulate oom@16`: attempt 1 killed ->
  restart 1 resumed from `latest.pt`; attempt 2 killed with a SIMULATED OOM line in
  its log -> `fallback` event, attempt 3 resumed with
  `batch_snapshot_policies=false` (checkpoint `config_changes`: `[false, true]`
  at update 3, `[true, false]` at update 18; the trim followed the arm off); it then
  ran to the training deadline and stopped cleanly with `latest.pt`; summary rendered.
- plain: ran to the deadline; no `config_changes` recorded.

## Unverified

- The allocator trim has never run on CUDA (the short check measures it).
- Batched mode has run on CUDA only in the 75-update A/B (no trim), never for hours.
- The OOM fallback was exercised with a simulated OOM line, not a real CUDA OOM.
- `recover`, and the "leave a pod-stopped instance for recovery" path, have not run
  against the provider API. The pod-side stop-after-DONE is new in this kit.
- Speed on the rented machine and the number of updates in 8 h are estimates.
