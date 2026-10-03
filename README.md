# GuanZero

A self-play reinforcement-learning system for Guandan (掼蛋), a four-player Chinese climbing card game. It has three parts:

- **`gd_core`**, a C++20 rules engine with pybind11 bindings.
- A **history-Transformer policy** trained from scratch with PPO on GPU.
- A **measurement workflow** that tests every performance change with A/B trials and checks that results are bitwise identical.

Most of the work is in the training system, not in how well the agent plays. Every number below links to the report that measured it.

## Performance results

| Change | Result | Hardware | Report |
|---|---|---|---|
| CUDA Graphs + batched attention + Triton KV-cache kernel in rollout collection | **3.48x** decisions/s (1,030 → 3,587). Bitwise-identical FP32 results on all 30 replays. About 1.95x in a short end-to-end PPO loop | RTX 4090 | [history-cuda-throughput-2026-09-28](docs/reports/history-cuda-throughput-2026-09-28.md) |
| Causal SDPA + batched KV cache instead of dense attention | **1.76x** full-update throughput. Reserved GPU memory 24.74 GB → 0.145 GB | RTX 4090 | [history-stack-cuda-2026-09-26](docs/reports/history-stack-cuda-2026-09-26.md) |
| 4 actor processes sharing one GPU | **1.78x** decisions/s (1,871 → 3,324). GPU utilization 55% → 82% | RTX 4090 | [history-actor-ranks-2026-09-28](docs/reports/history-actor-ranks-2026-09-28.md) |
| Learner length groups + page-padded rollout attention + paged KV cache (opt-in) | **1.71x** per training update (149 → 87 s per 10 updates); learner 2.4x. Tier 1–2 only; GPU not yet measured | Apple M4 Pro CPU | [training-stack-refactor-2026-09-29](docs/reports/training-stack-refactor-2026-09-29.md) |
| C++ rules engine, random play, canonical move generation | **190,533** decisions/s on one core; **1,839,581** on 12 threads | Apple silicon, 14 cores | [M0](docs/reports/M0.md) |

### A rejected idea

torch.profiler showed about **300 kernel launches per rollout step**. The single Python host thread was the bottleneck, not the GPU. A two-stage in-process pipeline made collection **1.4–2.2x slower**, because splitting each batch doubled the per-decision host cost. That result is kept in [perf-pipeline-2026-09-25](docs/reports/perf-pipeline-2026-09-25.md). It led to the two changes that did work: CUDA Graphs, which cut launches per step, and separate actor processes, which add Python threads of control.

## How changes are measured

- **Same-host A/B trials** in interleaved orders (for example `ABDEEDBA`), with warmup chunks discarded and medians over independent processes.
- **Bitwise parity.** Replay digests of an optimized path are compared against the original. Deterministic CUDA algorithms stay on during these checks, because a CUDA `index_add_` was shown to be nondeterministic: 100 of 100 repeats differed, by at most 9.5e-7.
- **Profiling** with torch.profiler (kernel launches per step), cProfile (host time) and nvidia-smi sampling (utilization, memory).
- **Scope of each result.** Throughput, numerical parity and playing strength are reported separately. A faster component is not claimed to make a better player.

## Correctness

- An independent Python rules oracle (`oracle/gd_reference.py`), written separately from the engine. The engine and oracle agree on 100,000 `interpret` cases, all 37,636 `beats` pairs and 10,000 move-generation hands, with 0 mismatches.
- A property fuzzer: 10,000,008 rounds and 735,492,869 decisions with all invariants checked. 100,002 rounds run clean under UBSan.
- OpenGuanDan trace replay: 1,783,202 decisions, with one divergence class that is explained and reproduced.

Details are in [M0](docs/reports/M0.md).

## Current status

The history Transformer trains from random initialization with self-play PPO. After 44.2M decisions on one RTX 4090, it beats an earlier 71M-decision MLP (+0.87 levels per round, 89% match wins). It still loses to the strongest MLP baseline (−0.36, 22% match wins). This is a single-seed run: [history-overnight-large-2026-09-28](docs/reports/history-overnight-large-2026-09-28.md). The design and the active plan are in [docs/DESIGN.md](docs/DESIGN.md) and [docs/STAGE_C_TODO.md](docs/STAGE_C_TODO.md).

## Layout

| Path | Contents |
|---|---|
| `cpp/` | `gd_core` C++20 engine, unit tests, fuzzer |
| `python/gd/` | pybind11 package (`gd._gd_core`) |
| `oracle/` | independent Python rules oracle, used only in tests |
| `train/` | PPO trainer, history Transformer, KV caches (per-entry and paged), CUDA Graphs, Triton kernel, DDP actor ranks |
| `bench/` | throughput benchmarks |
| `docs/reports/` | one report per milestone or experiment |

## Build and test

```sh
python -m pip install -r requirements-dev.txt -r requirements-train.txt
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/cpp/tests/gd_tests          # C++ unit tests
python -m pytest -q tests oracle    # Python suites and oracle cross-checks
```

The rules are specified in [docs/RULES.md](docs/RULES.md). Training entry points and readiness are in [docs/TRAINING.md](docs/TRAINING.md).

## Play against a checkpoint

```sh
./scripts/play.sh path/to/checkpoint.pt     # opens http://127.0.0.1:8765/
```

You take the bottom seat; the checkpoint plays the other three, your partner
included, greedily and without search. The page shows only what a player at
the table knows (your hand, public plays, card counts, a tracker of the cards
not yet seen) plus, on your turn, the model's top three moves for your hand.
The server is `eval/play_vs_model.py` (one process, two torch threads by
default; `--threads`, `--port`).
