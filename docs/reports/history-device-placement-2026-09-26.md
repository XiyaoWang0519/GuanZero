# T7 execution placement: CPU engine with CUDA model execution

The measured choice is **CPU game simulation, CUDA rollout inference and CUDA
actor/critic learning**. With A/B/C sharing one RunPod host, this configuration
reached about **1,575 decisions/s**, versus **876 decisions/s** on CPU and
**1,359 decisions/s** with CPU rollout and CUDA learning. These are engineering
throughput results; playing-strength evaluation is still pending.

## Why this check was needed

The initial T7 kit hard-coded CPU training based on an older comparison of a
Mac CPU with a different RTX 4090 host. That did not establish the best device
placement on the rented machine. The user clarified that total execution speed,
including CPU/GPU cooperation, is the objective.

The initial CPU attempt was stopped after roughly 47 trainer minutes. All 75
files, including the three final interrupted checkpoints, were downloaded and
hash-verified; an independent provider query confirmed zero pods and hourly
spend before the replacement allocation. Its estimated **$0.6157** remains
inside the original **$6** authorization. Its partial models are preserved and
are excluded from the primary three-seed comparison.

## Same-host measurement

One Secure RTX PRO 4500 allocation, with 27 usable CPU cores, ran all cases.
The engine stayed on CPU in every case. FP32, disabled TF32, full public history,
all legal candidates, PPO settings and the snapshot population were fixed.
Model width 64, two layers, four heads; 32 environments, 64 steps/update, two
PPO epochs, two matches/minibatch, two engine and two Torch threads per arm.
Timing includes collection, learning and, for split placement, weight transfer.

The diagnostic seed was 2026092991, separate from the three research seeds.
Each case ran 16 complete updates: eight warmup and eight measured. Single-C
cases used order CPU, CUDA, split, split, CUDA, CPU. Then A/B/C ran concurrently
for each placement using the production trainer; order CUDA, CPU, split.

| Placement of rollout / learner | Single C, seconds/update, two-run mean | Concurrent A/B/C, aggregate decisions/s | Concurrent learner rows/s |
|---|---:|---:|---:|
| CPU / CPU | 6.69 | 876 | 707 |
| CPU / CUDA | 4.37 | 1,359 | 1,056 |
| CUDA / CUDA | **2.78** | **1,575** | **1,237** |

CUDA model execution provided about **1.80x decision throughput** and **1.75x
learner-row throughput** versus CPU in the concurrent check. It also exceeded
the split placement by about 16% in decision throughput. The selection uses
concurrent throughput, because the research runs share one GPU across A/B/C.

The split route copies current learner weights before each collection,
invalidates learner caches through parameter versions, and mirrors fixed
snapshots with their existing identities. It preserves full raw histories and
records the sampler device for checkpoint restore. CPU/GPU probability checks
in the single-C measurements differed by at most 7.2e-7. The dedicated CUDA
suite passed 45 checks, including split-device probability, weight-sync,
population, cache and resume checks. Local related suite: 126 passed, 11 CUDA
skips. The full pre-research cloud gate also passed 133 checks (one optional
extension skipped) and 81 C++ cases before the three CUDA arms started.

## Limits and artifacts

This is a short same-host measurement, not a global hardware optimum or a
playing-strength result. Device arithmetic and sampling change trajectories,
so the CPU and CUDA workloads are matched in configuration, not identical in
all observations. Prefixes, learner rows, memory and phase times are retained.
Concurrent placement was measured once each; single-C measurements were repeated
in reverse order. No training hyperparameters were selected using game scores.

- Selected run kits and current execution: `.work/history-response-speed-2026-09-26/`.
- Timing summaries and scripts: its `device-selection/` directory. Downloaded
  summary hashes are recorded in `verified-summaries.json`.
- Full per-update timing records remain under the first pod's
  `local/results/device-preflight/` and `local/results/concurrent-device-preflight/`
  as periodic synchronization completes; final download verification precedes
  deletion.
- Production split-device source SHA-256:
  `d730650da1edd3ad360e1d9bb1babc798772fcf4296e6726b81c01875b8700dd`.
- The first single-C pass used the frozen predecessor trainer plus the retained
  benchmark wrapper; concurrent measurements used the production trainer.

All three primary seeds restart from their declared random initialization with
CUDA model execution and 80 trainer minutes per arm. The replacement pod's
original deadline is retained, including the measurement time. Subsequent
allocations are capped at $1.75 each, below the approved $2 maximum. The ledger
includes the stopped CPU allocation, measurement, training and teardown under
the same $6 total cap. No sealed final-test material has been opened.

Restart proof: all three actors, critics and samplers reported CUDA; each arm
completed at least 26 healthy updates. Six one-second GPU samples showed
92–94% utilization and up to 2,668 MiB used. This is an initial activity check,
not a claim of sustained utilization. Raw samples and per-arm device/health
records are in the campaign's `gpu-runtime-proof.json`.
