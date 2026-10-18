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
| Page-span complete PPO comparison (October 13) | [Isolated stack A/B](reports/training-stack-ab-2026-10-13.md): 2 L40S / 8 ranks, throughput +10.28%; real-rollout parity passed | Reproduce on production's 4 GPU layout before adopting; main configuration unchanged. Learner rollout projection failed its numerical gate |
| Host-side training work | [October 1 training stack](reports/training-stack-2026-10-01.md), [deep dive](reports/speed-deep-dive-2026-10-01.md) | Host changes alone do not prove remote end-to-end speedup |
| History-aware batched frozen evaluation | [Operator/evidence record](TRAINING_EVIDENCE.md#verified-bounded-t4-workload) | Recorded MPS parity is fixture-specific; validate a new backend/policy combination |

October 18, 2026 implementation update:
[throughput refactor](reports/training-throughput-refactor-2026-10-18.md) adds
batched public-event drain/ingestion, one KV-pool reservation per append batch,
shared actor/encoder gradient reductions and single-part auxiliary input reuse.
It also centralizes the existing CUDA private math-SDPA guard across all private
training routes. FP32, full public history, PPO objectives and checkpoint schemas
are preserved. Local CPU suites and original-revision dense/paged/snapshot PPO
plus resume fingerprints passed; measured complete CPU update gains are small
and variable. Target CUDA acceptance and throughput remain pending. No production
deployment is recorded. Keep cold, warm, released and restored allocator cases
separate, including the measured
released-page regression; see the report for results and
[the operator guide](TRAINING.md#runtime-refactor-2026-10-18) for rebuild,
compatibility and benchmark commands. Earlier negative results below remain
applicable to their original candidates.

## Follow-up queue

These are verification targets, not an authorization or fixed experiment order.
Inspect code and recent receipts first to avoid repeating completed work.

| Work | Evidence required |
|---|---|
| October 18 refactor on the target CUDA layout | Matching rebuilt extension; public/cache/gradient and checkpoint-resume parity; alternating complete-PPO collect/learn/wall measurements, including warm and reused caches |
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

October 13 follow-up: [ECC diagnosis and three-GPU repeat](reports/training-startup-ecc-2026-10-13.md)
confirmed one allocated GPU fails CUDA initialization with uncorrectable ECC.
On three healthy L40S / 8 ranks, page-span complete-PPO throughput improved
8.50% (paired block interval +6.65% to +10.16%) with real-rollout parity passing.
Four-GPU validation and main-lineage adoption remain outstanding.

[Page-span baseline projection follow-up](reports/training-stack-next-2026-10-13.md):
packed projection passed CUDA parity but the full-PPO run failed a gradient-norm
check at index 97; the first four complete pairs showed no improvement (-1.08%,
diagnostic interval -2.44% to +0.31%). Preserve this negative result. Cache
collation/prefill and gradient-reduce/wait remain profiling targets; value-only
folding was not evaluated on CUDA in this run.

October 14 gradient diagnosis: [7220 results](reports/training-gradient-diagnosis-2026-10-13.md)
confirmed finite actor gradients overflowing the FP32 norm on the page-span base arm
(3 healthy GPUs, 8 ranks, 94 completed updates). Rank 1 already had a huge local
backward gradient; outcome-loss replay showed the largest component gradient.
The operator-level cause remains unresolved; this is not a successful repair or
throughput result. Main source/manifest unchanged; no new experiment submitted.

October 14 follow-up: [rank 1 fixture replay](reports/training-fixture-backward-2026-10-14.md)
reproduced the outcome loss on CPU with normal gradients, but device/version differ.
Isolated job 7323 compares CUDA default/math SDPA and repeatability before any repair
or cache/reduction benchmark. Main configuration remains unchanged.

October 14 update: [7323/7324 fixture evidence](reports/training-fixture-backward-2026-10-14.md)
localized erroneous gradient amplification to masked efficient SDPA backward on
PyTorch 2.11.0+cu128 / L40S. A grad-enabled private-attention math fallback passed
fixture finite-gradient and 1e-4 candidate/chosen-logp gates. Job 7325 now tests
120 full PPO updates on the isolated repaired baseline before cache/reduction
optimization; production remains unchanged and no new speedup is claimed.

October 14 stability result (supersedes the pending 7325 status above):
[7325](reports/training-fixture-backward-2026-10-14.md) completed 120 full PPO
updates on every one of 8 ranks / 3 healthy L40S; gradient metrics were finite
and per-update parameter checksums passed. This is bounded stability evidence.
[7366 cache/reduction comparison](reports/training-cache-reduce-2026-10-14.md)
is submitted and pending resources. It gates real-rollout/cache parity and
reduction equivalence before two separate full-PPO A/B tests against the repaired
page-span baseline. No new speedup measured; production remains unchanged.

October 14 final 7366 result (supersedes pending status above):
[cache/reduction evidence](reports/training-cache-reduce-2026-10-14.md) passed
rollout and reduction parity; both stages completed 8 ranks × 120 updates on
3 healthy L40S. Against repaired page-span baselines, cache throughput +1.23%
(paired block 95% interval +0.18% to +2.19%); pinned host reduction −1.02%
(interval −2.68% to +0.68%), no demonstrated gain. 114 files verified and local
analysis matches remote. No combined test or main-lineage adoption; production
source and manifest unchanged. This round is closed, with no new job submitted.

October 14 user-directed follow-up: retain cache optimization only in the
isolated baseline; do not adopt pinned host reduction. [7392 encoding/wait
diagnosis](reports/training-encode-wait-2026-10-14.md) is submitted to the existing
cluster, with passive arrival traces and a separate synchronized diagnostic.
No new speedup claim and no main-lineage changes.

October 14 evening: [7392 diagnosis](reports/training-encode-wait-2026-10-14.md)
completed on 3 healthy L40S; all 8 ranks finished passive 40 / sync 32 updates,
119 files SHA-verified. Synchronized rollout cache/collation averages 0.997s,
learner public forward 0.250s with 22.41% token-slot padding. Passive reduction
arrival spread averages 16.64ms (p95 48.57ms); this does not establish a causal
rank/GPU imbalance or a speedup. Next target is incremental cache internals;
keep cache optimization, reject pinned reduction. Main identity/manifest unchanged;
no new job, diagnostic monitor paused on closeout.

October 14 follow-up: [7452 cache internals/reuse](reports/training-cache-reuse-2026-10-14.md)
is submitted under user authorization. It profiles append/gather/metadata and
then gates exact append-row-index reuse within one encode call before full PPO
ABBA. Local CPU coverage passed; CUDA parity and speed remain unmeasured.
Main unchanged; rejected pinned reduction stays disabled.

October 14 final [7452 result](reports/training-cache-reuse-2026-10-14.md):
CUDA parity passed; all 8 ranks completed 120 PPO updates on 3 healthy L40S.
Index reuse throughput −0.435% (paired 95% interval −1.301% to +0.369%), no
proven gain despite 50.79% candidate gather hits. Do not adopt this candidate;
retain previous index_select. Profile points to append (ordinary 0.626s,
merged 0.408s inclusive) over final memory gather (0.086s), with nested timing
and synchronization caveats. 113 files verified; local analysis matches remote;
main identity/manifest unchanged. No new job; monitor paused on closeout.

October 14 user-directed [append detail diagnosis](reports/training-append-detail-2026-10-14.md):
7457 submitted; retain index_select, reject row-index reuse and pinned reduction.
Event and synchronized probes split event preparation/KV/attention/FFN. No new
candidate yet. Stop if no clear actionable bottleneck; do not keep profiling.
Main unchanged, existing cluster only.

October 15 closeout: [7457 append detail](reports/training-append-detail-2026-10-14.md)
completed both 8×32 stages on 3 healthy L40S; 103 files verified. Attention is
the largest measured leaf (~0.367s/rank/update in event and sync diagnostics),
but no clear removable/reusable work supports a new optimization. Per user stop
condition, stop exploration: no new candidate/job/profile. Keep prior isolated
index_select only; main identity/manifest unchanged. Monitor paused on closeout.
