# Performance backlog: Transformer self-play

Updated September 25, 2026. The active workload is the randomly initialized
history Transformer in [DESIGN.md](DESIGN.md), with an optional looped decision
module. Old frozen-reference MLP optimizations are not the next training plan.
No Transformer performance estimate below is reported as a measurement.

## Current work

| Priority | Work | Evidence required | Dependency |
|---|---|---|---|
| 1 | Profile standard history actor end to end | Actual remote GPU collect/learn split, CPU quota/load, history lengths, candidates, memory | T1--T4 |
| 2 | Batch public-prefix encoding and private queries | Full-prefix action/log-probability parity and no private/future leakage | T2 |
| 3 | Versioned batched KV cache for all active snapshots | Rebuild after weight changes, per-model memory, match isolation and warm/cold timings | T2 |
| 4 | Reuse shared match-prefix work inside PPO epochs | Correct gradients and prefix masks; no stale detached encoder substitution | T2 |
| 5 | Loop-depth cost curve | Latency, decisions/s, activation memory, training sample rate and strength at supported depths | T6 |
| 6 | Tune environment count, rollout chunks and candidate batching | Declared learning/staleness changes separated from exact implementation optimizations | T4 |
| 7 | History-aware batched evaluation | Scalar reference parity, explicit stochastic semantics and match resets | T3 |
| 8 | Actor parallelism after a measured bottleneck | Alternating same-host trials, CPU-steal/load and total GPU-hours; population/cache correctness | stable T4 |

Do not retain a frozen MLP forward just because an old optimization shares it
between pruning and opponents. The new trainer has neither use. Do not silently
truncate history or reduce precision to fit more environments. Shared loop
parameters do not make repeated computation free.

## Measurement protocol

1. Save actual host facts, source/config hashes, workload and resource shares.
2. Warm up, then compare changes on the same host using alternating trials.
3. Report collection, learning and end-to-end timings; preserve regressions.
4. Separate fixed-seed numerical/action checks, throughput and playing strength.
5. Benchmark candidate and history distributions resembling the actual run,
   including long matches and multiple snapshot weights.
6. Change one learning-affecting setting per arm. Evaluate at equal total
   budget; component speedups do not imply a better trained player.

## Historical work and reusable evidence

Engine move generation, feature encoding, owned uint8 staging, the old batched
evaluator, CPU-budget discovery and DanLM adapter optimizations remain in the
codebase. Their original reports define what was tested:

- [Initial scan](reports/perf-scan-2026-09-23.md).
- [Implementation follow-up](reports/perf-followup-2026-09-24.md).
- [GPU validation](reports/perf-gpu-2026-09-24.md) and
  [later GPU profile](reports/perf-gpu-2026-09-24b.md).
- [Old learner measurements](reports/perf-learner-2026-09-24.md).
- [Rejected rollout pipeline](reports/perf-pipeline-2026-09-25.md).

The rejected pipeline's negative measurements remain evidence. Old gain
estimates, fixed-reference pruning tweaks and v3 supervised-collation tasks
are removed from the active queue. New hardware and actor choices follow
[COMPUTE_GUIDE.md](COMPUTE_GUIDE.md) and [TRAINING.md](TRAINING.md).
