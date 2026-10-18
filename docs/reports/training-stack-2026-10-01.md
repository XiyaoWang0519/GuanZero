# Training stack optimization — October 1, 2026

The host work around PPO is faster without changing the model, precision,
public-history horizon, canonical candidates, sampling order, PPO/GAE, rewards,
or population schedule. CPU comparisons against the starting revision
`4cea0889d444ab9f25efa5b37fa689b3f378d797` match exactly. GPU and complete-PPO
throughput gains have not been measured.

## Changes

- **Learner grouping:** `SequenceRolloutBuffer` factors trajectory match IDs
  once per compact buffer and uses integer arrays for later minibatches.
  It preserves first-occurrence order, repeated rows, round-specific views,
  and packed float-field alignment. Appends and carry-over invalidate the cache.
- **Response labels:** validate executed actions and walk response horizons for
  a stream's rows together. Tiny groups keep the scalar implementation. Targets
  are recomputed from current public events; no target cache is retained.
- **Snapshot layout:** construct row/candidate scatter indices across identities
  together, reuse the collector's stream order, and avoid redundant uint8 copies.
  A dedicated single-identity branch keeps small calls fast. Slot gaps, shared
  streams, canonical candidate order and random-draw boundaries are preserved.
- **Frozen collection scope:** `HistoryTrainer.collect` checks each KV cache's
  actor parameters on entry and exit, rather than on every encode. Standalone
  collectors retain per-call checks. New caches join lazily; removed or replaced
  caches still get an exit check. A weight mutation clears stale cache state
  and raises before learning. Callers catching this error must discard the failed
  rollout. The scope also covers the paged and Triton subclasses.
- **Packed transfer repair:** normalize the byte-view stride of empty/singleton
  tensors without copying. PyTorch can call these tensors contiguous even with
  stride zero or two, while dtype reinterpretation requires unit stride. This
  addresses the transfer error recorded in the September 30 CUDA report.

Existing training configurations pick up these host changes. Snapshot-layout
improvements apply when merged snapshot policies are already enabled. No
additional training flag or checkpoint schema is required. Existing optional
numeric/attention modes retain their defaults.

## Measurements

Local macOS arm64 host, 14 logical CPUs, PyTorch 2.14.0, one Torch thread.
The final benchmarks ran serially within this task. Other desktop activity
remained present. Times are medians of alternating before/after blocks on the
same fixture, checked for exact output equality before timing.

| Component and workload | Before | After | Speed ratio |
|---|---:|---:|---:|
| Response labels, 1,731 completed real rollout rows | 16.547 ms | 1.179 ms | 14.03x |
| Response labels, 128 rows | 1.145 ms | 0.222 ms | 5.16x |
| Match grouping, 8,192 completed rows | 2.186 ms | 0.270 ms | 8.09x |
| Complete host batch preparation, 2,048 rows / 16 matches | 2.360 ms | 2.002 ms | 1.18x |
| Snapshot layout, 16 identities / 256 rows | 632.2 us | 376.8 us | 1.68x |
| Snapshot layout, 12 identities / 96 rows | 292.9 us | 168.2 us | 1.74x |
| Snapshot layout, one identity / one row | 30.8 us | 22.8 us | 1.35x |

The response fixture uses eight frozen environments and 256 vector steps; no
optimizer runs. One-row response calls stay approximately unchanged (0.010 ms).
Batch preparation uses 600-token synthetic streams, 64 total matches and a
2,048-row minibatch. It checks every tensor for both ordinary and grouped
learner layouts. Snapshot cases include shared streams and small populations;
all six measured cases improved. Block counts: learner 8 x 40 calls, response
5 x 20 calls, snapshot layout 8 x 200 calls.

These component ratios cannot be added or multiplied to predict training speed.

### Collection timing and exact replay

Eight independent processes in base/fast/fast/base/base/fast/fast/base order:
32 environments, 16 distinct frozen snapshot identities, width 128, four layers,
eight heads, full history, KV cache, causal SDPA, batched/wide private attention.
Each process runs six 48-step chunks; the first two are excluded from timing.
The learner cache is invalidated at every chunk boundary. Maximum prefix: 561.

Median collection throughput: **1,188 -> 1,254 decisions/s (1.056x)**.
However, baseline repetitions range from 619 to 1,296 and changed repetitions
from 1,232 to 1,272 decisions/s. Host load was high, and the first baseline trial
was much slower. This does **not establish a reliable collection speedup**.
All eight runs have the same replay digest, covering choices, stored arrays,
public streams, trajectories, policy assignments, RNG and non-timing statistics:
`ba2f117423861f22d0a18801fc457c3dd332a377b8265916d9511f5050e2db2e`.

## Correctness

- Three before/after configurations, four tiny update calls per configuration:
  plain reference; causal/KV/batched/merged-snapshot with auxiliary labels and
  exploration; paged/merged encoder with grouped learner attention. Every
  update matches exactly for actor/critic weights, Adam state, population,
  retained buffer/history, sampler/NumPy/Torch RNG, losses and gradient norms.
  The first update has no completed samples; subsequent updates perform real
  optimizer steps. This is a regression check, not a learning experiment.
- New coverage includes grouping/carry-over/ROUND views, 96 snapshot-layout
  scalar-reference cases, randomized label order/duplicates and malformed
  history, frozen-scope mutation/exception cleanup, and singleton transfer bits.
- Packed uploads forced on CPU passed the paged-cache tests, including merged
  snapshot encoding. Native CUDA acceptance remains pending.

- Full Python suite: **1,310 passed, 312 skipped**, one existing tensor-to-scalar
  warning, in 52.04 seconds. Command: `python -m pytest -q tests -n 4
  --dist=worksteal`, with OMP/MKL/OpenBLAS threads capped at one. This run had
  local sockets and PyTorch shared memory available. The initial sandboxed run
  was interrupted after 1,293 passes because IPC/shared-memory restrictions
  caused failures and left workers waiting; its log is retained. No test
  failure remained in the complete rerun.

## Reproduction and evidence

Use the project's Python environment with `PYTHONPATH=python:oracle:.`.
Save the corresponding file from revision `4cea088` for each `--baseline` or
`--reference-source` argument, then run:

```sh
python -m bench.history_batch_host --baseline /tmp/history_rollout_before.py --calls 40 --repeats 8
python -m bench.history_response_host --baseline /tmp/history_response_before.py
python -m bench.history_snapshot_layout --reference-source /tmp/history_snapshot_batch_before.py
python -m bench.history_host --output /tmp/frozen-collection --case wide \
  --envs 32 --width 128 --layers 4 --heads 8 --steps 48 --chunks 6 --warmup 2 \
  --batched-private-attention --wide-private-projection --frozen-weights
```

The last flag is diagnostic-only; ordinary trainer collection already enters
that scope. Compare against the original source, without the flag, in balanced
process order. Actual local interpreter:
`.work/external/danlm-venv/bin/python`.

Raw receipts and the original source archive are under
`.work/training-stack-2026-10-01/`: `learner-batch-host-final.json`,
`response-label-final.json`, `snapshot-layout-final.json`,
`collection-summary.json`, the eight `collect-*` directories,
`ppo-parity-summary.json` and `ppo-v2-*` outputs. The first parity harness
included randomly generated lineage IDs and therefore differed in its combined
state hash despite identical losses; the corrected harness fixes that metadata
across arms. Initial attempts remain available.

During this work another process advanced the shared checkout to `9f1be07` and
`90ba057`, committing pre-existing evaluation changes and some in-progress
training edits. Those commits were preserved. Comparisons use the captured
starting source, not the moving HEAD. This task made no commits or GPU rentals.

## Remaining performance evidence

Measure the combined changes on the actual GPU using the existing fixed-source,
fixed-checkpoint CUDA gates and balanced complete-PPO benchmark. The September
30 result already showed that a faster learner component can leave overall PPO
unchanged. Full production history lengths, four-rank scheduling, peak memory
and numerical parity still need that target-hardware check. No playing-strength
claim follows from these host measurements.
