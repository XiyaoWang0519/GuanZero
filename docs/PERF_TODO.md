# Performance backlog

Found by the 2026-09-23 scan ([report](reports/perf-scan-2026-09-23.md)) and
not yet implemented. Gains are the scan's skeptic-corrected estimates; "pod"
means a B11-like host (three arms on one RTX 4090). Prototypes and evidence
live in `.work/perf-scan-2026-09-23/` (git-ignored; `REPORT.md` there has the
full ranking and the rejected ideas).

Done on 2026-09-23: single move generation in VecEnv, the eval prune skip,
shared-reference opponent forwards, CPU thread budget (`infra/cpu_budget.py`).
In progress elsewhere: the batched VecEnv duplicate evaluator.

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
5. Learner: host minibatch gather vs GPU step (CUDA events); `candidate_chunk`
   32768 vs 262144; fused Adam.

`.work/perf-scan-2026-09-23/gap-pod-host-variance/pod_calibrate.sh` covers 1-2.
The bench needs a `--league-steady` option and a per-spec timer for 3.

## Training throughput

| Item | Gain (estimate) | Effort | Semantics | Where / prototype |
|---|---|---|---|---|
| Learner reuses the uint8 staging copy: gather buffer rows from the pinned uint8 copy, pass obs by index, build opponent features only for network rows | host copies 2.5 → 0.6 ms/step; frozen arms +5-12%, league +1-3% | S | bitwise | `train/ppo.py` Uploader, `rollout_buffer.py:246` |
| Split the first fusion Linear so the state half runs once per row | −0.06 to −0.13 s/update per arm; batched eval −8-11%; helps search | S | float-level, not bitwise (≤1.2e-6) | `train/model.py:52-93` |
| Keep rollout data on the GPU for learn (upload once per update, gather on device); shrink `buffer_candidates` (2.74 GiB per arm, ~14x oversized) | learn −0.1 to −0.5 s/update | M | bitwise with 8,192-row refresh chunks | `rollout_buffer.py:357-450`, `ppo.py` |
| League arms at `num_envs 4096 × rollout_steps 32` (same decisions per update), cap about 20 | 1.1-1.2x after the fusion work | S (config) | **changes results**: 2x staleness, slower league ramp | league configs |
| Actor processes (W = 2-4, `num_threads // W`) on main/control/frozen arms, never the pinned exploiter | 1.0-1.35x, may be negative without MPS | S (config) | stat. equivalent | bench first (item 4 above) |
| `candidate_chunk 262144`, `Adam(fused=True)` | 0.3-2% | S | float-level | `ppo.py` |
| Stacked-weight forward over same-architecture Stage B opponents; static CUDA-graph rollout step | 0.5-1.5% each | M | low | after the above |
| Async actors (collect k+1 while learning k) | ≤1.05-1.1x while arms share a GPU | L | changes results | deferred |

## Engine (matters most for C3 search, fuzz and oracle tests)

| Item | Gain | Effort | Semantics | Prototype |
|---|---|---|---|---|
| Movegen feasibility pre-checks (skip rank/type combinations the hand cannot form), encoder bit operations, `make_view` over set bits, neutral-style skip; removes the per-call allocation at `movegen.cpp:304` (CLAUDE.md rule 4) | engine about 2.5-3x beyond what is done; PPO pod 1-3% | S | bitwise (digests and oracle crosschecks) | `engine/engine-all.patch` (make `beats_reading` constexpr inline instead of copying it) |
| `EnvConfig.single_round`, `halt_on` and `reset_env(i, deal\|seed)` | +15-20% for the batched evaluator, exact arena seed parity | M | none | `env.cpp:188-253` |

Rejected: `-march=native` (FMA contraction changes `styled_bot` results; use
x86-64-v2 if anything), contiguous env sharding, a C++ uint8 feature dtype.

## Evaluation

| Item | Gain | Effort | Semantics | Prototype |
|---|---|---|---|---|
| DanLM arena: per-seat KV cache in the DanLM forward; memo `hand_calculator_v2` (clear every round); PlayIndex memo plus lazy decode | 1,000 deals 73.5 → 43-46 s | S+M | KV: Q differs ≤8.5e-6, 0 flips in 12,386 decisions (add a top-2 gap guard); memos bitwise | `gap-danlm-arena/variants.py` |
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
| `pytest -n 4 --dist worksteal` with `OMP_NUM_THREADS=1` (add pytest-xdist; fixed `-n`, watchdog tests flake under `auto`); fuzz passes in the background | check.sh 90 → ~30 s | S |
| `-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF` for `_gd_core` (full LTO costs 1.5 s per engine edit, no runtime gain) | rebuild 2.5 → 0.9 s | S |
| Oracle movegen crosscheck against stored oracle digests (keyed on `sha256(gd_reference.py)`, fails rather than refreshes, so rule 1 holds); `slow` marker; ccache | 17 s → 4-8 s | S |

## v3 / belief pipeline (at the next v3 run)

Per-arm collate with uint8 tables; `is_causal=True` at `belief_model.py:76`;
vectorized eval cells with a per-pass summary cache; async `save_round` in
`collect_belief.py:244` (49% of collection wall); packed mmap dataset instead
of per-round npz (348 s load per process). Together about 7,922 → 5,000-5,800 s
per fit set, collection 1.55x.
