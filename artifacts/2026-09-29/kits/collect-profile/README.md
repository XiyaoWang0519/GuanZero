# Collection profile kit (2026-09-28)

Where does collection time go with 4 actor ranks on one RTX 4090? One short
diagnostic segment on Vast machine 20082, before building cross-identity
batched inference. Prepared, not launched.

## What runs

- Resume `init.pt` = `.work/actor-ranks-2026-09-28/download/results/segments/main-w4/latest.pt`
  (update 844, 55.3M decisions; sha256 `b1c55037...12bd7`, checked by `lifecycle.py`).
- Same layout as main-w4: `train.history_ddp --world-size 4`, 256 tables and 16-match
  minibatches per rank, `ddp_global_minibatch`, snapshot cadence 2. The recipe and every
  CUDA rollout flag (KV cache, batched attention, wide projection, private graphs,
  Triton cache, learner batched attention) come from the checkpoint.
- Source: branch `collect-profile-2026-09-28` (packed into `source.tar.gz` by `freeze`).
  The resume uses `--allow-source-change`; the change is recorded in the checkpoint's
  `source_changes` (same engine digest and token schema are required).
- `--resume-set` changes only diagnostic fields, all recorded in `config_changes`:
  `profile_collection=true`, `profile_collection_warmup=25`, `checkpoint_updates=1000`
  (no mid-run 105 MB saves; `latest.pt` is still saved at the end).
- 25 unprofiled warmup updates (resident snapshots and history lengths climb back
  after a resume), then 15 profiled updates. Profiling switches on between updates
  inside the same process, so the environments are not restarted. The trainer stops
  itself at update 884 (`--updates` is absolute).
- The profiled window's decisions/s is compared with the last 5 unprofiled updates.

**Profile mode calls `torch.cuda.synchronize` at every phase boundary. Its seconds
attribute collection time to phases; they are not absolute speed, and the profiled
window runs slower than an unprofiled run.** Use the unprofiled reference for speed.

**The output `segments/profile-diag/` is a diagnostic branch.** Its checkpoint is never
the main lineage endpoint and never a parent of a main segment (it also carries
`profile_collection=true` in its config and a `DIAGNOSTIC_BRANCH.txt` marker). The main
lineage still ends at main-w4 `latest.pt`.

## Measurements (per rank, last 15 profiled updates)

`summarize.py` writes `profile-summary.json` and `profile-summary.md` (on the pod into
the results, and locally into this directory after a verified download):

1. `collection_phase_seconds` per phase, per rank and mean; plus collect seconds minus
   the phase sum (weight transfer, loop overhead).
2. `public_cache_or_collation` and `actor_and_sampling` split into learner vs snapshot
   calls (`collection_group_phase_seconds`), and ms per call.
3. Rows per policy call for learner and snapshot calls: calls, calls per step, mean,
   median, p10, p90, min, max (from `collection_policy_call_rows`, an exact histogram).
4. collect/learn seconds (slowest rank), decisions/s from the `global_*` fields and
   from wall time, resident snapshots, mean prefix, CUDA peaks per rank, GPU
   utilization and memory (nvidia-smi every 5 s, matched to the window by
   `update-epochs.jsonl`).

Rank r > 0 metrics are in `segments/profile-diag/rank-r/metrics.jsonl`.

## Safety (same as the actor-ranks kit)

- Hourly ceiling $0.80; machine 20082 only; one GPU; label
  `guanzero-collect-profile-20260928`; refuses a duplicate rent.
- Timeline from create: segment stopped at 25 min if not finished (expected ~18 min),
  final sync by 27 min, local guard destroys at 34 min, pod stops itself at 34.5 min,
  hard cap 35 min. Worst case 0.80 x 34/60 = **$0.45**.
- Guard verify on the pod is read-only (GET). Never PUT state=running: it restarts the
  container.
- Training starts under supervisord (background jobs started from SSH are killed).
- Results are rsynced every 90 s; the final download is checked against the pod's
  `results-manifest.json` before destroy; destroy is confirmed by the instance's
  absence in the API.

## Commands

```sh
cd /Users/xiyaowang/Developer/Projects/GuanZero/.work/collect-profile-2026-09-28
PY=/Users/xiyaowang/Developer/Projects/GuanZero/.work/external/danlm-venv/bin/python
$PY lifecycle.py freeze --source-root <checkout of collect-profile-2026-09-28>  # done
$PY lifecycle.py dry-run            # read-only offers + checks, rents nothing
$PY lifecycle.py launch --yes-spend # only after the user approves the budget
$PY summarize.py download/results --out .   # re-run the summary by hand
```

Local CPU smoke (done while preparing): a tiny 8-table checkpoint, `profile.py --smoke
--device cpu --ranks 2 --warmup 2 --window 2`, then `summarize.py`.
