# Training and evaluation performance follow-up

Implemented a second performance pass on top of `4154ba4`, which already
includes shared-reference league inference and batched internal evaluation.
This pass removes repeated engine work, host copies and unused evaluation
inputs. It preserves model architecture, precision, training budgets, samples,
game rules and evaluation deal counts.

All measurements below are local CPU measurements. No GPU was rented.
The primary environment is Apple silicon macOS, Python 3.14 and PyTorch 2.14.0;
DanLM uses its existing Python 3.12 environment. Raw measured results are in
[the companion JSON](perf-followup-2026-09-24.json).

## Measured improvements

| Scope | Before | After | Meaning |
|---|---:|---:|---|
| Complete PPO updates, full pretrained M1 policy and B2 critic, 64 CPU environments | 16.1–16.7k decisions/s | 17.7–17.8k decisions/s | **1.082×** on median elapsed time in an ABBA comparison; real-model CPU result |
| Complete PPO updates, **tiny 16-wide model**, 256 environments | 78,078 decisions/s | 94,203 decisions/s | **1.21×** in the longer quiet run; rollout/learner overhead benchmark, not a production GPU estimate |
| VecEnv pending + step, 2,048 environments, 1 thread | 9.10–9.35 ms | 3.90–3.96 ms | About **2.3×** engine throughput; 4-thread measurements about 2.2× |
| Move generation on fixed recorded states | 2.46 µs | 0.91 µs | **2.7×** with greedy trajectories; about 3.2× with random trajectories |
| Observation encoding | 0.797 µs | 0.326–0.333 µs | About **2.4×** |
| PPO host buffer storage, 2,048 environments | 1.566 ms | 0.323 ms | **4.85×** gathering from an already populated uint8 staging copy; host-only simulation of CUDA staging |
| PPO host buffer storage, CPU float32 path | 1.566 ms | 1.402 ms | **1.12×** from removing the extra observation gather |
| Belief collation, unmasked memory arm, batch 128 | 2.221 ms | 1.472 ms | **1.51×** on synthetic binary inputs; payload 27.8 → 14.5 MB |
| Belief collation, masked memory control, batch 128 | 2.221 ms | 0.129 ms | **17.2×** for collation only; payload 27.8 → 1.1 MB |
| DanLM evaluation, quiet 20-deal ABBA comparison | 6.614 s | 6.087 s | **1.087×** throughput / **8.0% less elapsed time**, identical complete round records |

The component gains do not multiply into a whole-stack speedup. The tiny PPO
measurement used two warmup updates and 24 measured updates, 64 rollout
steps, one engine/Torch thread, one epoch and minibatches of 1,024. Both paths
performed 193 optimizer steps. The full-model CPU comparison used the same
settings except 64 environments and eight measured updates per variant;
every variant performed 17 optimizer steps. It ran before/after/after/before
with no other benchmark jobs active. The quiet engine confirmation ranged up to
2.6× at one thread; the table uses the more conservative paired measurements.
DanLM's quiet repeat is smaller but better isolated than the first 80-deal
comparison, which showed 1.20× while other work was active.

## Changes enabled by default

- **Engine:** reject infeasible rank/sequence combinations before expansion,
  cache natural-rank candidates, traverse card bitsets, calculate unseen cards
  with bit operations, and prune dominated moves in place without allocating.
  The rule comparison remains shared in `beats_reading`; action ordering stays
  identical. No native/FMA flags or rule changes were introduced.
- **PPO:** retain the existing uint8 host upload view for rollout storage and
  gather observations by index after learner/drop filtering. Storage owns its
  copies before either engine or staging buffers are reused. The device-to-host
  synchronization still precedes reuse of pinned memory.
- **Belief experiments:** flat/no-history arms omit current-history tensors;
  the unmasked memory arm still receives all its previous-round streams;
  the masked control omits those streams too. Full-history collation remains
  the public helper's default.
- **DanLM:** retain only the latest hand decomposition per seat, invalidate on
  hand/level changes, filter against the current trick every time, and decode
  legal plays lazily with a round-local cache. Attention arithmetic is untouched.
- **Development:** four bounded pytest workers, one BLAS/OpenMP thread each,
  two workers in CI, and default-off wrapper LTO. `PYTEST_WORKERS=0` restores
  serial testing; explicit `CMAKE_INTERPROCEDURAL_OPTIMIZATION=ON` enables LTO.

## Correctness and acceptance

- Complete local check: **562 Python tests passed, 2 skipped; 76 C++ cases;
  14 oracle groups; 60,000 deep-check fuzz rounds, zero failures**. The two
  skips are the CUDA-only staging check and the optional DanLM extension in
  the primary interpreter. Both optional real-DanLM observation tests passed
  separately in its Python 3.12 environment.
- Movegen differential: **78,000 calls / 2,046,379 ordered actions** exactly
  matched across all eight action-flag combinations, three rule profiles,
  every level, hand sizes 0–27 and preserved output prefixes.
- Eight VecEnv scenarios matched every batch field and completed-round digest,
  including full/canonical actions, house/ogd, styles/restyles, logging and forks.
- PPO: two rollout/learning cycles matched every stored array, model/critic
  weight, Adam state, metric and learner RNG state with the narrow host source.
- Belief: exact predictions and parameter gradients for flat, no-history,
  memory and masked memory models; exact evaluation metrics with full versus
  omitted unused inputs. Earlier-round memory tensors remain exact.
- DanLM: **112 duplicate deals / 224 round records / 18,721 decisions** matched
  across three seeds, including divergence samples in diff mode.
- Independent read-only reviews checked engine rules, PPO buffer lifetimes
  and belief information boundaries.

The complete check took 41.52 s including an engine rebuild; its parallel
Python suite took 28.92 s. This is a current duration, not a matched comparison
with the earlier scan's approximately 90-second full check. The first sandboxed
attempt could not create PyTorch's shared-memory manager for actor tests; the
complete run passed with local process permissions.

Both DMC and PPO CPU preflights completed two updates, saved, resumed to four
updates and wrote loadable checkpoints. PPO metrics remained finite. A separate
undefined-behavior sanitizer build passed all 76 C++ cases and 2,000 full-mode
deep-check fuzz rounds (150,667 decisions), with zero failures or sanitizer
diagnostics. GPU execution and Linux CI were not run in this pass.

## Reproduce

```sh
# Full correctness suite, including all fuzz passes
./scripts/check.sh

# Real-batch host storage, with exact array comparisons
OMP_NUM_THREADS=1 .venv/bin/python bench/ppo_staging.py \
  --num-envs 2048 --batches 16 --repeats 8

# Synthetic belief collation only, payload size included
OMP_NUM_THREADS=1 .venv/bin/python bench/belief_collate.py \
  --batch-size 128 --repeats 12

# Exact cached/uncached DanLM comparison; requires existing DanLM installation
PYTHONPATH=python:. OMP_NUM_THREADS=1 DANLM_ROOT=.work/external/DanLM \
  .work/external/danlm-venv/bin/python -m bench.danlm_memos \
  --checkpoint .work/runpod-b11/payload/artifacts/league.pt \
  --deals 20 --repeats 2 --seed 20260925
```

Detailed local baselines, scripts and logs are preserved under
`.work/perf-opt-2026-09-24/`, with engine reproduction instructions in its
`engine/README.md` and DanLM details in `danlm-evidence.md`. The learner storage
JSON is under `.work/perf-followup-2026-09-24/learner/`. These scratch artifacts
are git-ignored; the report, summary JSON, three new benchmark tools and
regression tests are repository files.

## Remaining work, in priority order

1. Measure the real CUDA league arm on the next already-authorized GPU run:
   host CPU quota/thread scaling, a populated opponent pool, phase timings,
   and solo versus co-running arms. This machine cannot validate those gains.
2. If learning remains a significant fraction of wall time, keep completed
   rollout data on the GPU and gather minibatches there. Account for memory
   across concurrent arms before shrinking buffer capacities.
3. DanLM per-seat attention KV caching needs numerical/tie and invalidation
   checks. The exact caches implemented here do not establish that extension.
4. For the next belief experiment, profile packed dataset loading and bounded
   async serialization; collation savings are not evidence of total fit speed.

The scan also considered splitting the policy's first fusion layer, asynchronous
actors, larger environment batches, and reduced evaluation deal counts. Those
change floating-point behavior, sample staleness, or statistical precision and
were left for separate measured experiments. See the updated
[remaining backlog](../PERF_TODO.md).
