# Training throughput refactor — October 18, 2026

## Result and scope

This refactor reduces public-history ingestion, KV-page allocation and learner
gradient-diagnostic overhead. A repository-wide architecture and hot-path audit
also found private-attention routes that bypassed the existing CUDA backward
safeguard; they now share the same dispatch helper.

Local component measurements show **4.32× public-event ingestion**, **5.57× cold
page allocation** and **1.34× actor clipping plus encoder diagnostics**. Across
24 separate-process complete-PPO timing trials, the ratios of median CPU
throughput were **+1.14% dense**, **+0.71% paged** and **+2.64% merged snapshots**.
The full-update differences are small and variable; these trials do not establish
a general training speedup or target-GPU throughput. In particular, paged-history
block results straddled zero. Cold allocator gains do not represent every append.

The full Python suite passed **1,422 tests**, with **342 skipped** and three
existing warnings. Exact original-revision comparisons passed for two PPO
updates and a further update after checkpoint resume in all three execution
paths. The native and independent oracle suites also passed, with zero failures
over 60,000 fuzz rounds.

All measured configurations, component trials, complete-update timings,
fingerprints and validation receipts are committed in
[the accompanying JSON evidence](data/training-throughput-refactor-2026-10-18.json).
The report does not depend on an ignored `.work/` directory being present.
There was no production deployment, remote training launch or playing-strength
experiment for this change.

## Source and environment

| Identity | Value |
|---|---|
| Original revision | `090f33706640367b777b67e544940bd397076303` |
| Original source SHA-256 | `29dcd81e8d384759868f1f652f551e15d3a1204d344ecc232367a8da968316c7` |
| Candidate source SHA-256 | `7117f6105a7b8cfdf68c8b70d82c337c4b21cebd07664ce8e35b3fa77d87e75e` |
| Unchanged game-dynamics digest | `10a9242cfaaba82c4331d3dcdcfa16b11fff405ff2d176a5402d8f3615dab78a` |
| Runtime | Python 3.12.14, PyTorch 2.11.0+cpu, NumPy 2.5.4 |
| Native build | Release; g++ 13.3, pybind11 3.1.0 |
| Host | Intel Xeon Platinum 8573C; Linux x86-64; nine visible CPUs, eight-CPU cgroup quota |
| Timing concurrency | One engine/PyTorch/BLAS thread; no simultaneous tests or benchmarks |
| GPU | Unavailable; CUDA/MPS execution and CUDA memory behavior unmeasured |

The candidate was measured as an uncommitted worktree on the original revision.
`infra.history_artifacts.source_identity()` hashes production sources, tests,
benchmarks and build inputs; subsequent Markdown/JSON report edits do not alter
that digest. The JSON records all 17 changed/new source-file hashes, the harness
hash and per-trial host facts. Separate original/candidate checkouts each used a
matching rebuilt native extension and the same Python environment.

## Audit and implementation

The audit followed the randomly initialized history Transformer route through
the C++ engine and binding, public event stores, collection and snapshot
dispatch, dense/batched/paged attention caches, PPO and auxiliary losses,
gradient reduction, checkpoint/source identity, evaluation entry points,
benchmark infrastructure and CI. Existing dated performance reports informed
which candidates were worth revisiting. This was an architecture and hot-path
review, not a claim that every line or optional deployment was exhaustively
validated.

### Public events cross the Python boundary in batches

`VecEnv.drain_public_batch()` returns five owned contiguous NumPy arrays instead
of one wrapped event and action array per public action. The C++ copy emits the
existing 186-byte token representation directly, explicitly zeroing private
tribute flags. It never reads hidden hands, private observations or a forced-pass
flag into the token. The scalar and batch drains consume the same queue.

`MatchEventStore.ingest_batch()` validates the complete batch, preflights match
transitions, then appends contiguous event runs with array copies.
`PublicStream` shares capacity reservation between scalar and bulk append.
Validation errors in a later run cannot partially append earlier runs. Existing
older match keys retain their original behavior, and previously returned
read-only views remain stable through append, growth and reset.

The normal history collector selects this path automatically after the extension
is rebuilt. Specialized stores retain their scalar `ingest` overrides, including
round-based habit streams; older extensions and custom environments without the
new method also use the compatibility path. The public API and lifetime contract
are documented in [PY_API.md](../PY_API.md#batched-public-history).

### KV allocation reserves the whole append's shortfall

The old append loop could repeatedly grow every layer's pool while assigning
pages to individual matches. Both ordinary and merged snapshot append now
compute their total missing pages and reserve once before assigning them.
Existing page contents and IDs stay intact. New page IDs are internal metadata;
the zero page, attention operands, full history and pool reset/high-water policy
retain their existing semantics.

For the measured cold prefill, this changed **14 pool growths to one**, reduced
copied pool storage from **369.56 MiB to 4.50 MiB**, and reserved **81.28 MiB**
instead of **95.63 MiB**. Warm appends needing no new page skip reservation.
Released-page reuse has a small measured Python overhead increase, reported
below rather than hidden in a cold-only aggregate.

### Actor clipping and encoder metrics share reductions

The learner previously computed the encoder's per-tensor norms and then
recomputed those same norms as part of actor gradient clipping.
`ActorGradientClipper` computes each actor tensor norm once for homogeneous
FP32 CPU/CUDA parameters, uses the original encoder parameter order for metrics,
and applies PyTorch's existing clipping operation with the total norm.
Other dtypes/devices retain the previous operations. The public
[`clip_grads_with_norm_` API is available in the declared PyTorch 2.9 minimum](https://docs.pytorch.org/docs/2.9/generated/torch.nn.utils.clip_grads_with_norm_.html).

Missing gradients are inspected on every call, and gradients remain in place,
including DDP buffer views. Zero, nonfinite and FP32 norm-overflow behavior retain
the existing stop condition. No wider accumulation or hidden numerical repair
was introduced. A single encoded group also passes its next-token auxiliary
inputs directly instead of copying three arrays through one-part concatenation;
multiple groups still concatenate as before.

### Every private-attention route uses the recorded CUDA safeguard

The original `_attend_by_match` path already forced math SDPA for grad-enabled
CUDA calls after the
[October 14 backward investigation](training-fixture-backward-2026-10-14.md).
Default, singleton, independent and windowed paths could bypass that guard.
They now call one `private_sdpa` helper with the same condition. CPU, no-grad
inference, public causal attention and frozen cache attention retain their
previous backend dispatch.

This is a correctness fix. It may cost time on CUDA routes that previously
selected a faster masked backend, and it is not counted as a speed improvement.
CPU route/dispatch tests pass; actual large-score CUDA forward/backward tests are
included but skipped on this host. The earlier CUDA incident evidence is not a
new GPU validation receipt for this PR.

### Boundaries and deferred candidates

Model dimensions and parameter names, FP32, full match history, complete legal
action support, rewards, PPO clipping/GAE/epochs, population rules, and token and
checkpoint schemas remain unchanged. The oracle and game-dynamics code were not
modified. Existing MLP/DanLM players remain evaluation baselines.

The audit also identified possible follow-ups in redundant greedy-choice work
inside `VecEnv.pending`, buffer indexing copies, segment operations and positional
encoding construction. These need attributable profiles and independent parity
measurements before changes are justified. They are not measured improvements
in this PR. Evaluation/search cache lifetime concerns likewise need a focused
reproducer before a repair can be claimed.

Historical negative results still apply: the
[projection candidate](training-stack-next-2026-10-13.md),
[pinned reduction comparison](training-cache-reduce-2026-10-14.md), and
[append row-index reuse](training-cache-reuse-2026-10-14.md) were not revived.
There was no precision reduction, history truncation, candidate pruning or PPO
recipe change to manufacture throughput.

## Measurements

### Components

| Component and scope | Reference median | Candidate median | Observed ratio |
|---|---:|---:|---:|
| Public event drain + ingest | 132,152 events/s | 571,460 events/s | 4.32×, median of ABBA block ratios |
| Cold KV allocation | 118.251 ms | 21.225 ms | 5.57× |
| Warm KV allocation, no new pages | 4.663 µs | 4.488 µs | 1.04× |
| Released-page reuse | 44.456 µs | 47.408 µs | 0.94×; 2.95 µs slower |
| Post-reset restored allocation | 29.598 ms | 20.597 ms | 1.44× |
| Actor clipping + encoder diagnostics | 1.240 ms | 0.922 ms | 1.34× |

Public events used 256 environments, one engine thread, seed 20261018, 32 warmup
and 128 measured steps, and three ABBA blocks. Each trial measured 44,954 events
and produced the exact same final stream digest. Engine stepping and hashing
were outside the timer. The reference is the candidate's retained scalar API;
the complete-PPO comparison below independently uses the original revision.

Page allocation used width 128, four layers/eight heads, 64 entries with target
lengths 1–512, 64-token pages, 16 initial pages, seven alternating pairs, and
1,000 calls per warm observation. The reference reproduces the previous
per-entry allocation loop. Cold/restored timings include allocating and filling
pool storage; ordinary and merged numerical behavior is checked separately.
Warm/released/restored timings are sensitive to host allocator state. The cold
copy/growth counters provide stronger causal evidence than their timing alone;
no broad steady-state gain is claimed from the other allocator phases.

Gradient measurement used a width-128, four-layer/eight-head actor with all three
auxiliary heads: 1,451,607 actor parameters, of which 793,088 belong to the public
encoder. Exact norms and clipped gradients were required before timing. Eight
alternating pairs each measured 200 calls after 20 warmups. Fixed gradients and
a nonbinding clipping threshold isolate the repeated reductions and scaling;
active clipping and optimizer behavior are verified by tests.

### Complete PPO updates across revisions

Each mode ran two ABBA blocks in separate processes: four original and four
candidate trials, 24 trials total. Each process executed four complete updates
with the first excluded as warmup. Measured work includes collection, learner
updates and update logging; initialization, checkpoint I/O and process teardown
are excluded. Timing uses an external wall clock, with collect/learn phases
retained separately. Exact-state instrumentation ran separately from timing.

All modes used width 64, two layers/four heads, eight environments, 96 steps per
update, one PPO epoch, two matches per minibatch, all three auxiliary heads,
FP32 and seed 20261018. Dense mode disables cache flags. Paged mode enables causal
SDPA, paged rollout KV cache, page-span and batched rollout/learner attention.
Snapshot mode additionally merges snapshot policies and encoder work with two
clones of the initial same-lineage policy. Those clones exercise batching, not
opponent diversity.

| Mode | Original decisions/wall-s | Candidate decisions/wall-s | Ratio of median rates | Individual ABBA block changes |
|---|---:|---:|---:|---|
| Dense history | 161.75 | 163.58 | +1.14% | +0.47%, +6.20% |
| Paged history | 1,181.24 | 1,189.60 | +0.71% | +2.99%, −0.17% |
| Merged snapshots | 920.50 | 944.83 | +2.64% | +1.96%, +3.27% |

The primary result is the ratio of the four-trial median rates, not the mean or
median of the two block ratios. All trials are retained, including one slower
dense baseline trial. Two blocks are insufficient for a reliable confidence
interval or a claim that a roughly one-percent gain generalizes.

| Mode | Original collect / learn seconds | Candidate collect / learn seconds | Learned rows per trial | Maximum prefix |
|---|---:|---:|---:|---:|
| Dense | 13.475 / 0.766 | 13.320 / 0.755 | 2,299 | 741 |
| Paged | 1.294 / 0.653 | 1.276 / 0.658 | 2,299 | 741 |
| Merged snapshots | 1.937 / 0.557 | 1.876 / 0.559 | 1,231 | 727 |

Phase values are medians of three measured updates summed within each process.
Every trial made 2,304 measured decisions. Learned-row counts and prefix coverage
matched exactly between revisions within each mode. Paged/snapshot learner phase
medians were slightly slower despite faster collection. The code removes real
component work, but its contribution to total time depends on workload and
backend; component ratios must not be multiplied into a training-speed claim.

## Correctness and validation

`bench/training_revision.py --verify` used a separate deterministic CPU workload:
width 32, one layer/four heads, four environments, 90 steps per update and two
updates with all auxiliary heads. It then saved a checkpoint, constructed a new
trainer from it and performed another real PPO update.

| Mode | Initial update samples | Samples after resume | Original/candidate comparison |
|---|---|---:|---|
| Dense | 295, 274 | 256 | Exact |
| Paged | 295, 274 | 256 | Exact |
| Merged snapshots | 143, 170 | 179 | Exact |

Compared fields include collected row and trajectory fingerprints, actor and
critic weights, both optimizers, Python/NumPy/Torch/sampler RNG states,
population, seat assignments, retained buffers/trajectories, public streams,
progress and loss metrics. Random lineage UUIDs and elapsed wall time are
excluded. This establishes matching behavior across revisions for the bounded
workload, including each revision's existing resume behavior; it does not assert
that checkpoint resume is identical to uninterrupted environment execution.

Additional checks:

- Full Python suite: **1,422 passed, 342 skipped, three warnings**, 194.33 seconds.
  Skips include CUDA and unavailable optional dependencies/fixtures. The full run
  had localhost sockets enabled for CPU Gloo distributed tests. The original
  revision's initial sandboxed comparison encountered five socket-denied cases;
  its complete DDP suite passed all 11 tests after enabling localhost sockets.
- Native C++ suite: **84 cases, zero failed assertions**. Independent oracle:
  **14 groups passed**.
- Deep-checked fuzzing: 20,000 canonical random rounds (1,469,835 decisions),
  20,000 full-action random rounds (1,502,015 decisions) and 20,000 canonical
  styled rounds (1,363,935 decisions), **zero failures in 4,335,785 decisions**.
- New focused coverage includes forced passes/exchanges/match transitions,
  batched array ownership, malformed later-run atomicity, specialized-store
  fallback, zero-page/page ownership and restored growth, missing/aliased/zero/
  nonfinite gradients, exact active clipping and Adam state, single-group
  auxiliary forward/backward equality, and private-attention dispatch routes.
- `git diff --check` passes. ASan/UBSan were not run locally; the existing Linux
  CI workflow includes native sanitizer tests and fuzzing. CUDA acceptance tests
  remain unexecuted on this host.

## Reproduction and adoption

The [training guide](../TRAINING.md#runtime-refactor-2026-10-18) contains the
component and verification entry points. Build both revisions in Release with
their own native extensions. Use one absolute path to the candidate's
`bench/training_revision.py`, change the working directory to the intended
checkout, and set its `PYTHONPATH=python:oracle:.`. Use fresh output directories
for every process, identical arguments and sequential A/B/B/A ordering.

The exact verification command's relevant arguments are:

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python:oracle:.
python /path/to/candidate/bench/training_revision.py --verify --case paged \
  --width 32 --layers 1 --heads 4 --envs 4 --steps 90 --updates 2 \
  --output .work/revision-check/paged-original
```

Repeat with `dense` and `snapshots` and the candidate checkout. Compare
`updates`, `resumed_initial_state` and `resumed_update` in the resulting JSON.
For the timed configuration, omit `--verify` and the dimension overrides; the
committed harness defaults match the complete-update configuration above.
The public-event script defaults match its measurements; pass `--heads 8` to
`bench.history_page_pool`. Gradient timings used `--device cpu` and its defaults.

Rebuild the extension to enable native batch draining. Existing checkpoint
tensor and token schemas need no conversion; source identity enforcement still
requires the documented explicit source-change adoption workflow when resuming
an old run (`--allow-source-change` where applicable). No source/engine check was
weakened. Before making a target-CUDA throughput claim, run the committed CUDA
parity tests and alternating complete-PPO measurements on the intended device,
PyTorch build, rank layout, snapshot count and prefix distribution. That evidence
is a follow-up, not a completed result of this PR.
