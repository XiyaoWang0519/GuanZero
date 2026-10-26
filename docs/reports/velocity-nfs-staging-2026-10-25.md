# Velocity NFS relief: node-local env and GPU allocation (October 25–26, 2026)

Scope: operational change on Velocity `max-compute-11`, triggered by another
user's report that small-file reads from `/home` were very slow (`import numpy`
over a minute) while our league/next-coef arms were running. No training
config was changed except the GPU layout described below. Implemented and
measured; no playing-strength claim.

## Measurements

| Check | Value |
|---|---|
| Our jobs: none running, cold `import numpy` from `~/guanzero/env` (NFS) | 17.96 s (warm repeat 2.03 s) |
| Same, small-file `stat`/`cat` on `/home` | 0.00 s / about 0.10 s each |
| Same, `pip freeze` in the NFS env | 15.5 s |
| Cold sequential read, 25 MB and 106 MB checkpoints on `/home` | 0.10 s, 0.09 s |
| Earlier (other agent) `tar` of the 6 GB env tree from `/home` | about 0.05 MB/s sustained |
| Node-local env build from the internet (uv, 3.12.14, cu128 wheels) | 27 s |
| `import numpy` / `import torch` from `/dev/shm` env | 0.55 s / 3.15 s |
| With three arms on node-local staging: NFS `import numpy`, `cat` | 5.3 s, 0.00–0.02 s |

`/home` bandwidth is fine; per-file opens are slow even with zero load
from us. So the harm came from many small reads (Python imports, pycache,
frequent appends), not from checkpoint size.

## Changes

- `scripts/velocity/` (deployed to `~/guanzero/bench-kit/`): `build_shm_env.sh`
  (env and source staging, flock, markers), `gz_job_env.sh` (single entry point:
  caches, run-dir staging, 300 s sync, guards), `shm_stage.sh` (compat name used
  by the training kits), `requirements-velocity.txt`. Rules:
  [TRAINING.md](../TRAINING.md#velocity-node-local-staging-october-25-2026).
- `infra/node_local.py`: with `GZ_REQUIRE_NODE_LOCAL=1`, `train.history_ddp`
  ranks and PPO actors raise if `sys.prefix`, `torch.__file__` or any mapped
  file resolves under `GZ_FORBID_PREFIX` (default `/home`).
- The campaign snapshot `src-campaign-2026-10-20` is staged by directory walk
  (about 45 s once per node; 360/360 files match `source-identity.json`). Its prebuilt
  `_gd_core` is kept byte-identical except RPATH: it pointed into the NFS conda
  env, so each trainer mapped `libstdc++`/`libgcc_s` from `/home`. Those two
  libraries are copied next to it (identical sha256) and RPATH is
  `$ORIGIN/.libs`. The snapshot's own `bindings.cpp` does not compile against
  its headers, so rebuilding is not an option for that tree.
- On Velocity, every legacy `*.sbatch` that ran `conda activate "$GZ/env"`
  (20 files, `.bak-20261026-pre-gzenv` copies kept) now sources
  `gz_job_env.sh`. `kit/setup_env.sh` and the unused `kit-lib/shm_runtime.sh`
  refuse to run.

## GPU layout change (mid-run)

`max-compute-11` has a fourth L40S (`GPU-39c6c4f9`, device minor 3) that is
listed in `gres.conf` but not in SLURM's `gres/gpu=3`. Every job could see it,
and the kits' health probe added it, so each arm ran `--spread-gpus` over its
own card plus the shared one. The earlier league-near "giving up" crashes were
CUDA OOMs on that card, with 12 processes from three jobs on it. By user
decision (October 25), jobs use only their SLURM allocation. `gz_job_env.sh`
maps allocation `IDX` to device minor to UUID, sets `CUDA_VISIBLE_DEVICES`, and
exits if torch sees more GPUs than allocated. The kits probe only those UUIDs.
Each arm now runs its 4 ranks on one card (other settings unchanged). Segments
before and after the switch differ in throughput and must be annotated when
compared; switch points are in the Project status note.

## Limitations

- Legacy kits get the guarded env, caches and source staging, but their
  outputs are not shm-staged unless they opt in (`GZ_STAGE_RUN=1` or
  `shm_stage.sh`). Only the league and next-coef kits plus one eval-style
  smoke job were exercised; the other legacy kits were not run.
- A `python -c` started with cwd on `/home` outside the entry point is not
  caught by `sitecustomize` (Python adds `''` to `sys.path` after site runs);
  the entry point always `cd`s into shm.
- Engine tests that call `git rev-parse` fail on the snapshot (no `.git`); 48
  engine, encoder, league and env tests pass there.
