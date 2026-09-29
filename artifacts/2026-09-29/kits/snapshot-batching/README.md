# Merged snapshot inference A/B kit (2026-09-28)

Does `--batch-snapshot-policies` (all snapshot identities' rows of a vector step in
one merged actor call) speed up collection with 4 actor ranks on one RTX 4090?
One diagnostic segment on Vast machine 67872 that switches the arm between updates
inside one process. Prepared, not launched.

## Machine (retargeted 2026-09-29 01:30 UTC)

Machine 20082 (Maryland, EPYC 7B13, 32 effective cores, $0.783/h), used for the
earlier collect-profile and actor-ranks runs, had no rentable offer on 2026-09-28
(the machine-pinned offers query returned an empty list). The kit now pins
**machine 67872**: Vast Secure Cloud (datacenter, `hosting_type` 1), verified,
reliability 0.9922, 1x RTX 4090 24 GB (300 W power limit, PCIe 3.0 x16), Intel
Xeon Platinum 8173M (Skylake-SP, 3.5 GHz max), 37.3 effective cores, 84 GB RAM,
722 GB disk, ~5.8/7.2 Gbit/s up/down, driver 580.178.04 (CUDA 13.0), United
Kingdom; $0.557/h total at 50 GB disk when searched. It was the only verified
datacenter 1x 24 GB RTX 4090 offer at <= $0.80/h with reliability >= 0.99.

**Absolute numbers from this run are not comparable with the earlier 20082
measurements** (collect 13.8 s, ~20 s/update): different CPU generation and
clock, core count and GPU power limit. Only the in-run on/off comparison (same
process, same machine, alternating blocks) is valid. `ab-summary.*` records the
machine ID and host facts (CPU model, usable cores, GPU) for that reason. A
slower host may also not finish all 75 updates before the 38-minute stop; the
speed blocks end at update 65 of the process, the profiled blocks come last.

## What runs

- Resume `init.pt` = `.work/actor-ranks-2026-09-28/download/results/segments/main-w4/latest.pt`
  (update 844, 55.3M decisions; sha256 `b1c55037...12bd7`, checked by `lifecycle.py`).
- Same layout as main-w4: `train.history_ddp --world-size 4`, 256 tables and 16-match
  minibatches per rank, `ddp_global_minibatch`, snapshot cadence 2, graph budgets
  512/256 MB per rank. The recipe and every CUDA rollout flag (KV cache, batched
  attention, wide projection, private graphs, Triton cache, learner batched
  attention) come from the checkpoint.
- Source: branch `snapshot-batching-2026-09-28` (packed into `source.tar.gz` by
  `freeze`). The resume uses `--allow-source-change` (recorded in `source_changes`).
- `--resume-set` changes only diagnostic fields, all recorded in `config_changes`:
  `batch_snapshot_policies_schedule=35:off,10:on,10:off,10:on,5:off,5:on`,
  `profile_collection=true`, `profile_collection_warmup=65`, `checkpoint_updates=1000`.
- Update plan (75 updates, 844 -> 919), environments never restarted:

  | updates of the process | arm | profiled | use |
  |---|---|---|---|
  | 1-25 | off | no | warmup (resident snapshots, history lengths) |
  | 26-35 | off | no | block A1 |
  | 36-45 | on | no | block B1 |
  | 46-55 | off | no | block A2 |
  | 56-65 | on | no | block B2 |
  | 66-70 | off | yes | call shapes, phase split |
  | 71-75 | on | yes | call shapes, phase split |

- Speed comes from the unprofiled blocks only; the first 2 updates of every block are
  dropped (`summarize.py --settle 2`), because switching clears or re-admits the
  snapshot private graphs (off -> on frees them; on -> off must re-capture) and
  allocates or frees the stacked heads.

**The output `segments/snapshot-batching-diag/` is a diagnostic branch.** It is never
the main lineage endpoint and never a parent of a main segment (it carries the
schedule in its config and a `DIAGNOSTIC_BRANCH.txt` marker). The main lineage still
ends at main-w4 `latest.pt`.

## Gates (before any timing)

`remote_setup.sh` runs the four existing CUDA gates plus a fifth, `snapshot`:
`tests/test_history_snapshot_batch.py` on CUDA with `-s` (the log keeps the measured
max |merged - per-identity| log-prob differences). Required to pass (not skip): the
production-size precision case, the multi-decision case, the per-identity sampling
and exact generator-state check, the chi-square/TV check, slot refresh, whole-rollout
equality (KV cache) and the production CUDA options (private graphs + Triton + batched
attention + wide projection) case. Any failed gate aborts before training.

## Measurements

`summarize.py` writes `ab-summary.json` and `ab-summary.md` (on the pod into the
results, and locally into this directory after a verified download):

1. Per arm (mean of the settled unprofiled blocks): wall s/update, decisions/s from
   wall time (global decisions over rank-0 elapsed time), collect and learn seconds
   (slowest rank), policy calls per step (rank mean), GPU utilization and memory
   (nvidia-smi every 5 s, matched by `update-epochs.jsonl`), entropy, approx KL max,
   allocation retries; and the on/off ratio of decisions per wall second.
2. Per block: the same plus CUDA peak reserved and stacked-head MB.
3. Profiled blocks: calls per step and rows per call (mean, p10, median, p90, max) for
   learner and snapshot calls (the "snapshot" histogram shows merged call sizes in the
   on arm), and the learner/snapshot cache vs actor seconds. Synchronized: not speed.

Rank r > 0 metrics are in `segments/snapshot-batching-diag/rank-r/metrics.jsonl`.

## Safety (same as the collect-profile kit)

- Hourly ceiling $0.80; machine 67872 only (offer must also be a verified datacenter
  RTX 4090 24 GB with reliability >= 0.99); one GPU; label
  `guanzero-snapshot-batching-20260928`; refuses a duplicate rent.
- Timeline from create: segment stopped at 38 min if not finished (expected ~27-30
  min: ~1 min start, ~3 min build and gates, 75 updates at ~17-21 s on 20082; unknown
  on 67872), final sync by 40 min, local guard destroys at 44 min, pod stops itself at
  44.5 min, hard cap 45 min. **Worst case at the ceiling 0.80 x 44/60 = $0.59**; at the
  $0.557/h quoted for 67872, 0.557 x 44/60 = $0.41 (traffic extra, ~$0.004/GB up).
- Guard verify on the pod is read-only (GET). Never PUT state=running: it restarts the
  container.
- Training starts under supervisord (background jobs started from SSH are killed).
- Results are rsynced every 90 s; the final download is checked against the pod's
  `results-manifest.json` before destroy; destroy is confirmed by the instance's
  absence in the API.

## Commands

```sh
cd /Users/xiyaowang/Developer/Projects/GuanZero/.work/snapshot-batching-2026-09-28
PY=/Users/xiyaowang/Developer/Projects/GuanZero/.work/external/danlm-venv/bin/python
$PY lifecycle.py freeze --source-root <clean checkout of snapshot-batching-2026-09-28>  # done
$PY lifecycle.py refreeze           # re-hash kit files after the retarget, source kept (done)
$PY lifecycle.py dry-run            # read-only offers + checks, rents nothing (done)
$PY lifecycle.py launch --yes-spend # only after the user approves the budget
$PY summarize.py download/results --out .   # re-run the summary by hand
```

## Local evidence gathered while preparing (CPU only; does not predict CUDA)

- `flagoff-check/`: `digest.py` run on the parent commit 8303769 and on the branch
  with the flag off; `parent-8303769.json` and `branch.json` are identical (collector
  choices, stored rows, sampler state for KV on/off and two exploration settings,
  plus a resumed trainer's weights, sampler and counters).
- `cpu_measure.py` -> `cpu-measure.json`: production architecture on CPU, 64 tables,
  12 snapshot identities. Policy calls per step 11.7 -> 2.0; backend ATen calls inside
  snapshot actor calls 1,648 -> 162 per step; identical choices in both arms; wall
  time is CPU noise, not a CUDA prediction.
- `smoke/`: `snapshot_ab.py --smoke --device cpu --ranks 2 --warmup 2 --block 2
  --profile 1` on a tiny checkpoint (`smoke-init/`), then `summarize.py`: the arm
  switched on schedule and the summary rendered.
