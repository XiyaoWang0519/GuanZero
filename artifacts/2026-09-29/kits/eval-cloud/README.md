# Cloud CPU evaluation of the overnight main lineage (2026-09-29)

Prepared, not launched. Runs the 256-deal vs-B11 evaluation of seven checkpoints
on one rented Vast machine (CPU only; the GPU is never used) so the laptop's CPU stays free.

## Evaluator source: git b8c0c54 (identity 14e72581...)

The eight finished local points (`.work/longrun-batched-eval-2026-09-29/out/`)
all record `evaluation_source_sha256` 14e72581... In that directory's history, u2480
and u2623 were started from the working tree (`run_one.sh`, `PYTHONPATH=python:oracle:.`)
after later commits had landed (learner/rollout refactor, `train/history_model.py`,
`eval/history_policy.py`). They recorded 0e935769..., so they were moved to
`out-tainted-mixed-source/`. The previous agent then exported b8c0c54 read-only
(`src-b8c0c54/`) and ran from it (`run_one_export.sh`, `PYTHONPATH=$E/python:$E/oracle:$E`,
with `run_point.py` as the script, so the export's `infra`/`eval` are imported). It started
a verification re-run of u2201 (`out-verify/`) and u2623 from the export. Neither
finished. Checked here: `git archive b8c0c54` and the export both hash to 14e72581...
(213 files), and the engine digest equals freeze.json's 606ff1e7...
`payload/source.tar.gz` is packed from `git archive b8c0c54` with a
`source-identity.json` receipt, so `source_identity()` on the pod needs no git and
refuses any changed file.

## What runs

- `remote_start.sh` (supervisord) checks every uploaded file with `sha256sum -c`
  (`kit-files.sha256`, `generated.sha256`), starts the pod self-stop guard (read-only
  GET verify), then runs `remote_setup.sh`:
  torch 2.14.0 (CPU wheel) and numpy 2.5.3 in a fresh venv, the local versions (if that
  install fails it falls back to the image's torch and records `matched_local_versions:
  false` in `env.json`); checks the source identity and **engine digest against the local
  value (abort on mismatch)**; CMake build; `gd_tests` (abort on failure); places
  freeze.json, development.json and both baselines at the absolute paths freeze.json
  names (symlinks), so the freeze sha stays f77ea34d; imports, rechecks the identity after the build,
  and loads u2623 from the packaged layout.
- `eval_pod.py` runs the unchanged `run_point.py` (4 torch threads) per checkpoint,
  in priority order u2623, u0844, u2201, u2542, u2480, u2418, u2263. Parallelism =
  min(7, usable CPUs // 4, available memory // 3 GB). Usable CPUs = min(affinity, cgroup
  quota) measured on the pod (`host-facts.json`). 7 in parallel needs 28 usable CPUs;
  with fewer, the tail waits for a free slot and may be cut by the deadline.
- Every 60 s the Mac rsyncs `/workspace/results/` (per point: `out/uNNNN.json/.log/.time`)
  and re-renders `results.md` whenever a new point lands, so partial results survive.

## Comparability (pod Linux x86-64 vs local macOS arm64)

`summarize.py` compares the pod's u0844 and u2201 with the local results: identical
or not (whole `reports`: both baselines, per-deal scores, match pairs), number of differing
B11 deals, and pod minus local mean with a paired interval.
(a) The pod table is always produced: new points are paired against the **pod-run** u0844
and the previous pod point. (b) The merged table with the 8 local points is produced
**only if both references are bit-identical**. Otherwise the pod table stands alone with
the measured platform difference, and nothing is merged.
Both tables report update, decisions, 31-update mean training entropy (local
metrics.jsonl), vs B11 with interval, match win rate with interval, paired vs u844 and
vs previous, the pooled late-plateau estimate (from u2108 on; smoothed, one lineage,
shared deals), a trend statement ("cannot distinguish" unless intervals exclude zero)
and the next run's starting checkpoint (the endpoint unless an earlier point beats it
with a paired interval entirely below zero).

## Machine

Pinned, in order (Secure Cloud/datacenter, verified, reliability >= 0.99, >= 28 effective
CPUs, >= 32 GB RAM, <= $0.80/h, CUDA >= 12.8 for the image, traffic <= $0.035/GB):
**62158** (RTX 5070 Ti, EPYC 7V13, 64 effective CPUs, 129 GB, rel 0.9991, $0.552/h, Texas),
38639 (A100, EPYC 7713, 32, $0.676/h; rented at dry-run time), 33405 (RTX 4090, EPYC 7B13,
32, $0.743/h), 20082 (RTX 4090, EPYC 7B13, 32, $0.783/h). The dry run would rent 62158.

## Timeline and cost (from create)

| time | event |
|---|---|
| up to 25 min | SSH reachable (status logged every minute); not reachable -> **destroyed** |
| next | upload 912 MB (rsync, <= 30 min); run must start by 1 h 20 m, else **destroyed** |
| +5-10 min | venv, build, gd_tests, checks |
| 2 h 05 m | unfinished evaluations killed; manifest; `DONE` |
| 2 h 08 m | the Mac stops its sync loop, final sync, verify, summarize, destroy |
| **2 h 20 m** | **local guard destroys** |
| 2 h 22 m (or 20 min after `DONE`) | **pod stops itself** |
| 2 h 30 m | hard cap |

Worst case at the $0.80/h ceiling: 0.80 x 2.367 h = $1.89 + traffic 1.3 GB x $0.035 =
$0.05 -> **$1.94**. At 62158's $0.552/h: **$1.34**. Launch refuses below $2.44 credit
(credit at dry run: $5.39). Expected: about 1.2-1.9 h -> about $0.66-1.05 on 62158.

## If the Mac sleeps or goes offline

`launch` runs under `caffeinate -dimsu`. That does not keep a MacBook awake with the lid
closed on battery: keep it on power with the lid open.
- The pod keeps evaluating (supervisord); the Mac only uploads, syncs, summarizes, destroys.
- If the Mac sleeps **before the upload finished**, launch fails on wake. The run was never
  started, so the instance is destroyed. The detached local guard also destroys it at 2 h 20 m.
  The guard sleeps with the Mac too.
- If it sleeps **after the start**, the pod stops itself at 2 h 22 m or 20 min after `DONE`.
  **Stop, not destroy**: compute billing ends, the 50 GB disk bills storage
  (~$0.02/h, ~$0.45/day) until destroyed. Results stay on that disk; whatever was synced
  before sleep is already in `download/results/`. On wake the Mac sees a pod-stopped
  instance with an unverified download and leaves it (`RECOVER.md`, event
  `left_stopped_for_recovery`). Run `lifecycle.py recover --yes-spend`. It restarts the
  instance, downloads, verifies, summarizes and destroys it. It waits at most 15 min for
  the instance to come up, else stops it again. The restart can fail if the machine's GPU was rented meanwhile;
  `lifecycle.py destroy` gives the rest up.
- Briefly offline while the pod still runs: the loop just continues.

## Commands

```sh
cd /Users/xiyaowang/Developer/Projects/GuanZero/.work/eval-cloud-2026-09-29
PY=/Users/xiyaowang/Developer/Projects/GuanZero/.work/external/danlm-venv/bin/python
$PY lifecycle.py freeze          # done
$PY lifecycle.py dry-run         # done (online, read-only)
$PY lifecycle.py launch --yes-spend    # user action
# afterwards
tail -5 lifecycle.jsonl; cat teardown-summary.json artifact-verification.json; cat results.md
$PY lifecycle.py status          # {"owned": null, ...} when nothing is left
```

## Local smoke (`smoke/`, see `smoke/NOTE.txt`)

- The packed tree (`local-src/`, plus the export's macOS extension) imports. Its
  `source_identity()` returns 14e72581 from the receipt. It loads u2623.
- `eval_pod.py` on a 2-deal, 1-match-seed freeze with u2623 and u0844 from `payload/`:
  both finished. u0844's two per-deal scores (both baselines) and first match pair equal the local
  256-deal run's first entries. Deadline kill: running points killed, the rest `not_started`.
- `summarize.py` on fake pod folders: identical references -> merged table; one deal perturbed
  -> "NOT identical, 1/256", no merge.
- Peak RSS of one evaluation (2 deals): 2.7 GB (hence 3 GB per evaluation in planning).

## Unverified

- Nothing ran on Linux: `remote_setup.sh` (venv with torch 2.14.0 CPU wheel, build,
  gd_tests), `remote_start.sh` and the upload have not run on a pod.
- Per-evaluation speed on EPYC with 7 concurrent is an estimate. The actual CPU quota of 62158 is unknown
  until it runs; 67872 gave 26 of an advertised 37.
- Whether x86-64 reproduces the macOS per-deal scores bit for bit (that is what the
  reference comparison measures).
- `recover` and the stop-for-recovery path have not run against the provider API.
