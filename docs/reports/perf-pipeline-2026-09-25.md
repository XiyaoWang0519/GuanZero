# RTX 4090: in-process rollout pipeline, actors, two arms per GPU

Date: September 25, 2026 (00:39–01:01 UTC). One RTX 4090 secure pod, EUR-IS-1,
about 22 minutes and $0.27. Source: commit cbc3a4a (`rollout_pipeline`).
Kit and raw results: `.work/runpod-pipeline-2026-09-25/` (git-ignored;
`results/`, `payload/run.sh`, `payload/equiv_cuda.py`).

Host: AMD EPYC 7702, 256 logical CPUs, cgroup quota 54.4, no throttling.
CPU steal was 0.00% in all 106 ten-second samples, and host busy share
stayed at 1–6%. This host showed no CPU contention from other tenants.

## Result

The in-process pipeline is correct on CUDA but **slower**, so it is rejected.
Four actor processes remain the best measured way to fill the GPU, and two
arms on one GPU also add throughput.

## Correctness on CUDA

- `tests/test_ppo_actors.py`: 11 passed, including the CUDA-only
  pipeline-against-actors test.
- Real checkpoints, 2,048 environments, `rollout_pipeline=2` against
  `actor_processes=2`:
  - League arm (B11 main), 2 updates: observations, candidates and
    actions are identical; `logp` differs by at most 9.3e-5.
  - Frozen arm, update 0: identical.
  - Frozen arm, update 1: the rollouts diverge, because the weights
    after the first learn differ by up to 4e-4.
- Control: `rollout_pipeline=2` run twice gives identical actions over 2
  updates, but its weights also differ by 1.5e-4. CUDA learning is not
  bitwise reproducible on its own, so the divergence is float-level
  nondeterminism, not a stream race; a race would already show in update 0.
  The actors-twice control failed on a missing `__main__` guard in the
  control script, not in the code under test.

## Speed

Frozen arm (B9 RTX 4090 config, `kl_coef 0`), 3 rotated rounds. League arm
(B11 main without imports), 2 rounds. Each run is 2 warm-up plus 6 measured
updates. Collect seconds are the clean signal. Learn seconds swung from 4.8 s
to 35 s for identical work, worst in the first run of each arm.

| Variant | Frozen collect (s) | League collect (s) | GPU SM busy, bench window |
|---|---:|---:|---:|
| One shard, 16 threads (`p0`) | 6.36 / 6.78 / 7.80 | 8.46 / 8.44 | 6–18% |
| Pipeline 2, 16 threads each | 9.60 / 10.00 / 10.82 | 21.15 / 18.48 | 8–23% |
| Pipeline 4, 16 threads each | 14.92 / 14.30 / 15.23 | — | 13–19% |
| 4 actors × 4 threads | 4.58 / 4.85 / 4.59 | 5.06 / 4.93 | 24–33% |

- **Pipeline:** collect is 1.4–2.2× *slower* than one shard.
- **Actors:** 4 actor processes are 1.4–1.7× faster, as in
  [perf-gpu-2026-09-24b](perf-gpu-2026-09-24b.md).

**Why the pipeline fails.** The limit is the one Python thread, not the GPU
waiting on the engine. A step costs about 300 kernel launches plus the host
work around them. Splitting the batch in two doubles that per-decision host
cost, and the GPU is already asynchronous enough that little device time was
left to hide. Only parallel Python threads of control help: actor processes,
or separate arms. This also makes cutting launches per step (a CUDA graph over
padded buckets) the most promising single-process lever left. That change is
float-level, so it needs its own A/B.

**Two frozen arms at once**, 8 engine threads each, sharing the GPU: combined
62,974 and 94,613 decisions/s over 6 updates each, against 20,029–43,373 for a
single `p0` arm on the same pod (both noisy through learn). This is positive,
though too noisy to put a number on.

## Decisions

- `rollout_pipeline` was removed again (Sept. 25): it had no speed benefit and
  added a second shard path to maintain. The code part of cbc3a4a is
  reverted; `git show cbc3a4a` holds it if it is ever needed.
- For the long run: 4 actor processes on every arm, and two arms per GPU
  where the quota allows (54 here; at a 13.6 quota, one arm with 4 actors).
- Even with 4 actors the GPU is under a third busy, so there is headroom
  for more actors or arms per GPU; measure 8 actors and 2 arms × 4 actors
  on the first long-run pod.
