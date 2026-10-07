# Performance backlog: Transformer self-play

Status reconciliation: October 6, 2026. Baseline workload: the history
Transformer in [DESIGN.md](DESIGN.md). See [STATUS.md](STATUS.md) for direction
and [TRAINING.md](TRAINING.md) for operations. Old frozen-reference MLP
optimizations are historical, not the current training route.

## Implemented paths and measured scope

Reviewed October 6, 2026 against the reports below. Implementation is not a
claim that every option is enabled by default or beneficial on every device.

| Work | Recorded evidence | Remaining limit / follow-up |
|---|---|---|
| Batched public-prefix inference and versioned KV caches | [History stack](reports/history-stack-2026-09-26.md), [CUDA comparison](reports/history-stack-cuda-2026-09-26.md) | Preserve raw-history learner recomputation, weight invalidation and snapshot isolation; verify production workload |
| CUDA Graphs and Triton rollout cache | [CUDA throughput](reports/history-cuda-throughput-2026-09-28.md) | Frozen-replay parity and collection throughput do not establish full-PPO or strength gains |
| Multi-process actors and snapshot batching | [Actor ranks](reports/history-actor-ranks-2026-09-28.md), [snapshot batching](reports/history-snapshot-batching-2026-09-28.md) | Check effective batch, population and allocator behavior before changing rank count |
| Learner groups, page cache and attention layout | [Refactor](reports/training-stack-refactor-2026-09-29.md), [learner CUDA](reports/history-learner-cuda-2026-09-30.md) | Learner/component gains do not establish meaningful complete-PPO improvement |
| Host-side training work | [October 1 training stack](reports/training-stack-2026-10-01.md), [deep dive](reports/speed-deep-dive-2026-10-01.md) | Host changes alone do not prove remote end-to-end speedup |
| History-aware batched frozen evaluation | [Operator/evidence record](TRAINING_EVIDENCE.md#verified-bounded-t4-workload) | Recorded MPS parity is fixture-specific; validate a new backend/policy combination |

## Follow-up queue

These are verification targets, not an authorization or fixed experiment order.
Inspect code and recent receipts first to avoid repeating completed work.

| Work | Evidence required |
|---|---|
| Production end-to-end profile after new changes | Same-host collect/learn split, CPU quota/load, prefix/candidate distribution, snapshot count and memory |
| Long-match cache and learner sharing | Correct gradients, full-prefix parity and masks, no stale detached encodings; cache lifetime under updated weights |
| Environment/chunk/rank tuning | Effective batch and staleness recorded; paired throughput at equal hardware shares and learning-quality comparison where semantics change |
| Looped decision-module cost curve (T6) | Separate implementation/readiness receipt, latency/FLOPs, memory and strength at supported depths; not established by the standard Transformer |
| Search cost reduction | [Fused-search proposal](reports/fused-search-design-2026-10-05.md), then matched search quality/latency evidence before claiming a gain |

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
