# Performance backlog

Found by the 2026-09-23 scan ([report](reports/perf-scan-2026-09-23.md)) and
not yet implemented. Gains are the scan's skeptic-corrected estimates; "pod"
means a B11-like host (three arms on one RTX 4090). Prototypes and evidence
live in `.work/perf-scan-2026-09-23/` (git-ignored; `REPORT.md` there has the
full ranking and the rejected ideas).

Done on 2026-09-23: single move generation in VecEnv, the eval prune skip,
shared-reference opponent forwards, CPU thread budget (`infra/cpu_budget.py`).
Also done: the batched VecEnv duplicate evaluator (`eval/batched.py`).

Implemented in the [September 24 follow-up](reports/perf-followup-2026-09-24.md):
movegen feasibility and allocation removal, encoder/card bit operations,
PPO uint8 staging reuse and indexed observations, exact DanLM hand/play memos
with lazy decode, arm-specific belief collation, and bounded parallel tests.
The follow-up separates measured component gains from unmeasured CUDA gains.

The [evening RTX 4090 profile](reports/perf-gpu-2026-09-24b.md) measured
the learner pass on CUDA, profiled steady collect and learn steps, and
added lazy opponent features (league collect 15% faster, bitwise).

The [RTX 4090 validation](reports/perf-gpu-2026-09-24.md) measured the local
pass on CUDA, added a density-gated first-fusion split, skipped encoding for
scripted batched evaluation, and added an RTX 4090 preset for the single frozen
B9 arm's measured engine/Torch settings. The full PPO effect of the split
remains unresolved because host throughput varied sharply. The report
distinguishes fixed-minibatch and end-to-end measurements. Repeat CPU tuning
for co-running arms.

## Measure first (next pod, about 5 minutes before the arms start)

1. `python -m infra.cpu_budget --facts runs/host-facts.json`, plus
   `nvidia-smi -q` (PCIe, clocks, power cap).
2. Engine thread scaling on a frozen arm:
   `bench/ppo_throughput.py --config train/configs/b9-exploit-league.json`
   at `--num-threads 4 8 16 24`.
3. Per-spec league cost against a steady cap-16 pool, alone and with one
   co-running arm, old and new code. Headline numbers with `--profile` off.
4. `--num-envs 2048/4096 --rollout-steps 64/32`, and `--actors 0 2 4`, alone
   and co-running; probe NVIDIA MPS in a private pipe directory.
5. Learner: measured on September 24 (`learn_on_device` best step 19 vs
   28 ms frozen, 29.6 vs 33.7 ms league; the reference skip has no GPU
   effect). Left: `candidate_chunk`
   32768 vs 262144; fused Adam.
6. Repeat item 4 as alternating pairs on one pod: the evening pod saw
   actors W=4 at 1.15–1.3× end to end on the frozen arm from single runs.

Profile with `.work/runpod-gpuopt-2026-09-24/payload/profile_collect.py`
(four warm-up updates first; an earlier, shorter warm-up gave misleading
first-round numbers). Compare code only on one pod with alternating runs.

`.work/perf-scan-2026-09-23/gap-pod-host-variance/pod_calibrate.sh` covers 1-2.
The bench needs a `--league-steady` option and a per-spec timer for 3.

## Training throughput

| Item | Gain (estimate) | Effort | Semantics | Where / prototype |
|---|---|---|---|---|
| Build opponent features only for network rows | done ([evening 4090 profile](reports/perf-gpu-2026-09-24b.md)): league collect −15%, +9% end to end | done | bitwise | `train/ppo.py` (`RowGather`) |
| Rollout data kept on the device for learn (`learn_on_device`) and the reference forward only on rows it can prune (`skip_unpruned_reference`), [September 24 learner pass](reports/perf-learner-2026-09-24.md) | measured on CUDA ([evening profile](reports/perf-gpu-2026-09-24b.md)): best learner step 19 vs 28 ms frozen, 29.6 vs 33.7 ms league (of three runs); skip has no GPU effect | done | as stated | `train/rollout_buffer.py`, `train/ppo.py` |
| Shrink `buffer_candidates` (2.74 GiB per arm, ~14x oversized) | memory only | S | bitwise | `train/ppo.py` |
| League arms at `num_envs 4096 × rollout_steps 32` (same decisions per update), cap about 20 | 1.1-1.2x after the fusion work | S (config) | **changes results**: 2x staleness, slower league ramp | league configs |
| Actor processes (W = 2-4, `num_threads // W`) on main/control/frozen arms, never the pinned exploiter | 1.0-1.35x, may be negative without MPS | S (config) | stat. equivalent | bench first (item 4 above) |
| In-process rollout pipeline (`rollout_pipeline`, Sept. 24–25): **rejected**. Correct on CUDA, but collect was 1.4–2.2× slower than one shard, because the single Python thread (about 300 launches per step plus host work) is the limit and splitting the batch doubles it; 4 actors were 1.4–1.7× faster on the same pod ([report](reports/perf-pipeline-2026-09-25.md)) | negative | — | same data as actor processes | code removed Sept. 25 (reverted after cbc3a4a) |
| `candidate_chunk 262144`, `Adam(fused=True)` | 0.3-2% | S | float-level | `ppo.py` |
| Stacked-weight forward over same-architecture Stage B opponents; static CUDA-graph rollout step | 0.5-1.5% each | M | low | after the above |
| Async actors (collect k+1 while learning k) | ≤1.05-1.1x while arms share a GPU | L | changes results | deferred |

## Engine (matters most for C3 search, fuzz and oracle tests)

| Item | Gain | Effort | Semantics | Prototype |
|---|---|---|---|---|
| Neutral-style skip (movegen feasibility, allocation removal, encoder and `make_view` bit operations are now implemented) | remaining gain needs measurement | S | bitwise | `env.cpp` / `bots.cpp` |
| `EnvConfig.single_round`, `halt_on` and `reset_env(i, deal\|seed)` | +15-20% for the batched evaluator, exact arena seed parity | M | none | `env.cpp:188-253` |

Rejected: `-march=native` (FMA contraction changes `styled_bot` results; use
x86-64-v2 if anything), contiguous env sharding, a C++ uint8 feature dtype.
Also rejected on the evening 4090 pods: a reused pinned buffer in
`RolloutBuffer.stage` (exact, but slower frozen learn in 3 of 4 pairs) and
glibc malloc tuning (mixed); see the [report](reports/perf-gpu-2026-09-24b.md).

## Evaluation

| Item | Gain | Effort | Semantics | Prototype |
|---|---|---|---|---|
| DanLM per-seat KV cache (exact hand/play memos and lazy decode are now implemented) | remaining gain needs fresh measurement | M | Q differs ≤8.5e-6 in prototype; add a top-2 gap guard | `gap-danlm-arena/variants.py` |
| Per-deal eval cache keyed on checkpoint sha, reference, seed, deals, rule profile and a gd-build plus eval-code hash | 46% of B8 crossplay repeated earlier work | S | stale results without the code hash | `eval/crossplay.py`, `eval/stage_b_baseline.py` |
| DanLM trend checkpoints at 500 deals, 1,000+ only for decisions | halves yardstick time | S | CI ±0.10 → ±0.14 levels | schedule |

## Pod operations

| Item | Gain | Effort |
|---|---|---|
| Budget runs by `max_updates`, keep `max_seconds` as a generous safety stop (B11 phase 2 stopped near 1,520 of 2,000 updates) | restores the protocol | S |
| Pull results with rsync during polls, no gzip (ratio 0.90), blocking wait on ALL_DONE | 2-16 min + ~2.5 min per experiment | S |
| Start guard and curve evaluations on TRAIN_DONE and each arm's EXIT | 12-16 min per milestone off the critical path | S-M |

## Dev loop

| Item | Gain | Effort |
|---|---|---|
| Overlap fuzz passes with other checks (bounded parallel pytest and default-off wrapper LTO are now implemented) | remaining gain needs measurement | S |
| Oracle movegen crosscheck against stored oracle digests (keyed on `sha256(gd_reference.py)`, fails rather than refreshes, so rule 1 holds); `slow` marker; ccache | 17 s → 4-8 s | S |

## v3 / belief pipeline (at the next v3 run)

Arm-specific collation now skips ignored history/memory. Remaining: uint8
transfer tables; `is_causal=True` at `belief_model.py:76`;
vectorized eval cells with a per-pass summary cache; async `save_round` in
`collect_belief.py:244` (49% of collection wall); packed mmap dataset instead
of per-round npz (348 s load per process in the earlier scan). Re-measure the
remaining end-to-end opportunity after the collation changes.
