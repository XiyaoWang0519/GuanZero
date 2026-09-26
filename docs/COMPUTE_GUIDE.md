# Compute guide for history and looped Transformers

Updated September 25, 2026 for [the current design](DESIGN.md).
Hardware choices follow measured sequence workloads and a fresh provider quote.
Previous MLP experiments do not establish the new model's bottleneck or cost.
Run lifecycle and implementation readiness are in [TRAINING.md](TRAINING.md).

## Measure the actual workload

The new actor retains public match history and may evaluate several internal
loops per decision. Measure full-prefix attention, cache reconstruction after
updates, sequence learning, activation memory, candidates and active snapshot
count. Shared-weight loops reduce unique parameter growth but still increase
executed compute and often latency. They are not free because an old MLP left
the GPU partly idle.

Record at least:

- GPU model/VRAM, driver/runtime, precision and effective CPU/RAM quota.
- Actual history-length distribution, environment count and cache/activation
  memory including every active Transformer encoder version.
- Collection and learner timings, decisions per second, candidate counts,
  cache hit/rebuild cost and population participation.
- For each trained loop depth: decisions per second, per-move latency,
  memory, total GPU-hours and paired playing strength.

Choose a small feasible encoder and environment count before scaling. Preserve
full-match history semantics; any context truncation is a named experiment.
Do not reduce numerical quality to obtain a throughput number without the
required numerical and playing-strength evidence.

## Fair comparisons

Use equal total training wall clock on comparable hardware/resource shares;
report samples, GPU-hours, dollars and evaluation overhead as well. Independent
arms start from separate random weights with matched seed schedules and the
same population rules, not the same old checkpoint or replay.

Standard and looped Transformers also need comparable inference budgets.
An equal-parameter model doing four times as much computation is not a
compute-matched control. Extra loops may improve a move while slowing data
generation enough to hurt training; retain both measurements.

For code performance comparisons, prefer alternating same-host trials with
host-load/CPU-quota capture. Different pods or simultaneous arms can introduce
resource contention; do not attribute that to architecture. Parallel rentals
require an explicit aggregate budget and per-resource deadline. There is no
default recommendation to multiply pods.

## Host selection and lifecycle

Use `infra.cpu_budget` host facts rather than `nproc` or `os.cpu_count()` alone;
environment variables, cpusets and cgroup quotas can change the effective
allocation. Choose GPU memory and CPU capacity from the pilot, not a fixed
MLP-era thread-count recommendation.

Inspect current provider inventory, quota and full price before provisioning.
RunPod, Kaggle or another provider may be suitable depending on the measured
workload; old price snapshots and previous account credits are not current
availability. The existing `infra.runpod` guard does not manage other providers.
Verify a provider-specific guard, checkpoint export and teardown path before
using a new runtime.

Sustained training is remote. Local CPU evaluation, bounded tests and statistical
analysis are useful but do not substitute for GPU training measurements.
Do not provision resources merely to complete a documentation update.

## Historical measurements retained

| Evidence | What it supports | Boundary |
|---|---|---|
| [M1 GPU pilot](reports/M1-runpod.md) | Old MLP update/resume and resource measurements | No history or loop throughput claim |
| [GPU profile](reports/perf-gpu-2026-09-24b.md) | Host/kernel overhead and old PPO component measurements | Re-profile new actor/sequence replay |
| [Pipeline rejection](reports/perf-pipeline-2026-09-25.md) | That in-process pipeline was slower on its tested workload | Not a general rejection of batching or all actor designs |
| [CPU isolation survey](reports/compute-cpu-isolation-2026-09-24.md) | Host quota and placement considerations at that date | Prices/inventory require live verification |
| [Kaggle continuation](reports/kaggle-campaign-results-2026-09-26.md) | Completed MLP campaign under its measured quota/protocol | Neither a ban on nor an endorsement of Kaggle for Transformer training |

Live performance tasks are tracked in [PERF_TODO.md](PERF_TODO.md). Keep
engine/algorithm semantics, implementation speed and measured playing strength
as separate claims.
