# Opt-in FP32 paged rollout-attention prototype — October 18, 2026

## Status and scope

**Implemented as an experimental candidate. Not CUDA-validated or production-ready.**
This follow-up targets PR #3 at `7ae0a6da5f9b80657b2205889a7913e7c4e6d06d`.
The original `train/history_paged_cache.py` blob was verified as
`92c655cb04a92a23575ef128dbe1e488e2059b22` before editing.
The development environment has PyTorch 2.10.0+cpu, no CUDA GPU, no Triton
compiler/package, and only a reconstructed subset of the repository. It cannot
establish kernel compilation, target-device numerical acceptance, complete
project integration, or a speedup. No production job/checkpoint was touched.

The [append-detail report](training-append-detail-2026-10-14.md) motivates
investigating attention and KV reads, but its instrumented measurements are not
an exclusive production-time budget. No predicted end-to-end percentage is
inferred from that table. Earlier negative cache-index-reuse results remain valid.

## Candidate

The existing paged cache writes each layer's new keys/values to
`[pool_rows, 2, heads, depth]`, then gathers a dense `[B,H,span,D]` pair and calls
masked SDPA. The prototype reads those pages directly while performing causal
attention with an online softmax. It avoids that per-layer dense KV gather and
the global attention mask in the opt-in path. New KV writes, projections,
embeddings, positional encodings, feed-forward blocks, memory writeback, and
the final top-layer memory gather are unchanged.

One Triton program processes a 16-query tile for one stream/head and traverses
64-key tiles through the stream's page table. Q may be the strided QKV view.
Dot products request `input_precision="ieee"`; probabilities and accumulators
remain FP32. There is no FP16/BF16 conversion or TF32 shortcut, dropout, window,
pruning, approximate history, private observation, custom backward, or new
training signal. Different query tiles can reread KV, and strict FP32 may be
slower than the existing library backend: those are reasons to measure, not
claims of a guaranteed win.

Both ordinary and merged-snapshot append routes use the same checked kernel.
A layout is validated from the existing host upload arrays once per append;
no `.item()`/device-to-host read is introduced into that path. Live page IDs
must be in bounds and cannot be reserved page zero. Kernel reads are masked
at context/page/tile boundaries, use wide pointer arithmetic, and never write
KV or a shared cache. Unused query outputs are zero and never stored as valid
history. Allocated pages, entry generations, per-policy isolation, reset/prune,
weight invalidation, and returned-memory ownership remain with the existing cache.

The mathematical inspiration and API references are the official
[Triton fused-attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
and [`tl.dot` precision documentation](https://triton-lang.org/main/python-api/generated/triton.language.dot.html).
The implementation must still compile and pass its own target-hardware gates.

## Explicit selection; defaults and backward are untouched

```python
cache = PagedHistoryCache(actor, attention_backend="triton")
```

Or, for a bounded experiment that constructs its own caches lazily:

```python
from train.history_paged_attention import experimental_paged_attention

with experimental_paged_attention("triton"):
    # Construct and run an isolated trainer/collector here.
    # Each new PagedHistoryCache keeps its selected backend for its lifetime.
    ...
```

Outside that scope the default is still `torch`. An explicit constructor
argument overrides the scope. Existing caches never switch backends in place;
create independent caches/trainers for the comparison. The scope is nested and
exception-safe, process-local, not a production CLI/config option, and does
**not** propagate to spawned DDP workers. A multi-process experiment must enter
its own scope in every worker and repeat the validation at that topology.

Explicit Triton requests fail on non-NVIDIA/non-CUDA devices, missing Triton,
non-FP32 inputs, and head depths outside 16/32/64/128. There is no silent eager
fallback. Gradient-enabled invocation is rejected before cache mutation;
normal cache encode/collection already runs under no_grad. Learner recomputation
and the existing private-attention math-SDPA backward safeguard are not replaced.
Counters distinguish actual custom-kernel dispatches from simply selecting a flag.
No checkpoint schema, PPO/reward/population setting, precision, or information
boundary is amended by this prototype. Source-change resume checks remain in force.

## Validation implemented and validation actually run

Local command from the reconstructed repository root:

```sh
OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_paged_attention_prototype.py \
  tests/test_paged_attention_benchmark.py \
  tests/test_paged_attention_integration.py
```

Local result: **67 passed, 41 skipped**. Forty skips are CUDA kernel cases;
one is the whole real-engine integration module, unavailable in this local
subset. Python bytecode compilation and shell syntax checks also pass. These
are not the earlier PR's full-suite counts and not GPU execution evidence.

The CPU tests check page metadata, an independent FP64 oracle, a PyTorch model
of the online recurrence, padding, future-token/stream isolation, fail-closed
selection, benchmark gates and summaries, restoration of precision settings,
and CLI refusal to call a CPU run a CUDA benchmark. **The CPU recurrence does
not execute the Triton kernel.** Static IEEE-dot checks and `compileall` also
do not compile GPU code.

Additional executable CUDA gates are supplied for:

- Direct kernel output against the original gather/SDPA and an independent
  FP64 oracle: head depths 16/32/64/128, chunks 1/27/128, long contexts to 4,097,
  ragged/empty rows, and ordinary/high-scale inputs. Repeatability and no KV
  writes are checked. Output tolerance is rtol 1e-4, atol 1e-5.
- Real ordinary and merged cache prefill/append, page growth and reuse,
  reset/prune/weight invalidation, snapshot slot gaps and shared-pool isolation.
- Real random-player self-play in ordinary/merged modes: exact choices and
  sampler RNG, exact non-logp stored fields/public streams, absolute candidate
  behavior-logp comparison at 1e-4, and chosen behavior probabilities recomputed
  from full history at 1e-4. Learner gradient recomputation must remain finite
  and match the reference within the declared test tolerance.

These CUDA/integration gates have **not been run here**. They may find compiler,
API, numerical, or integration failures; do not relax tolerances or turn a
failure into a timing result. A mature production-checkpoint replay gate and
bounded multi-rank stability run are additional prerequisites before adoption.
The provided fresh-player fixture is not that mature-checkpoint gate.

## Running the gates and measurements on Velocity

Use a separate checkout/worktree and an **existing allocated GPU session**, not
the login node or an active training directory. Use the project's working Python
environment and built native extension; the runner neither installs packages
nor submits scheduler jobs, rents compute, resets GPUs, or touches checkpoints.
It honors the allocation's `CUDA_VISIBLE_DEVICES` and uses logical device 0.
Do not broaden that variable beyond the allocation or benchmark concurrently
with production on the same GPU. A failed CUDA health check stops the script;
it does not opportunistically take another card.

Start with the acceptance gates and component benchmark:

```sh
PYTHON=.venv/bin/python bash scripts/benchmark-paged-attention.sh \
  .work/paged-attention-gpu
```

Or use the Python executable from the already configured server environment.
The output path must not exist. The script verifies CUDA/Triton and the native
extension, runs the three new test modules, and rejects **any skipped, failed,
or errored gate** before timing. It writes `gates.log`, `gates.xml`,
`component.log`, and `component.json`.

The component benchmark uses batch 61, eight heads, depth 16, chunks 1/27/128,
and longest contexts 65/1,537/4,097 with ragged rows and shuffled pages. Every
fixture passes its output gate before an exact-shape compilation/warmup.
Two ABBA blocks retain every raw timing. The eager arm precomputes row indices
and masks outside the per-layer measurement, as the original cache does once
per append: it is not charged extra setup to inflate the candidate's win.
FP64 oracle checks and extra-peak-memory probes are also outside timing.
CUDA event elapsed time can include host submission gaps/shared-device waits;
it is not labeled pure kernel-active time. Host wall time is reported separately.
Environment details, source digests, peak allocated deltas, and all regressions
are retained. This comparison excludes projections, writes, final memory gather,
and PPO, so its ratio cannot be reported as total training acceleration.

For fresh, **single-process complete PPO** trials as well:

```sh
PPO_BLOCKS=2 PYTHON=.venv/bin/python \
  bash scripts/benchmark-paged-attention.sh .work/paged-attention-full --ppo
```

This repeats the gates/component phase, then runs two ABBA blocks each for
ordinary paged history and merged snapshots: 16 independent processes total.
Each process performs 40 full updates, excluding the first 24 as warmup.
The defaults are width 128 / four layers / eight heads / FP32, 128 environments,
64 collection steps, one PPO epoch, eight matches per minibatch, and all three
auxiliary heads. Initial snapshots are clones of a newly initialized actor,
not trained production opponents. Every arm uses the same seed/configuration.
`PPO_BLOCKS=1` reduces the number of blocks, not the numerical gates.

The runner synchronizes around `trainer.update()` and includes collection,
actual learning, optimizer work, and the trainer's usual update duties; it
excludes checkpoint saving. Post-timing checks confirm the selected backend
and actual custom-kernel execution, and reject runs with no PPO learning.
`ppo-summary.json` reports raw trials/block variation and ratios of median
complete-update rates, preserving slowdowns. It refuses incomplete ABBA blocks
or mismatched source/configuration/environment. It makes no significance claim.
Late shape-specialization compilation, if encountered after warmup, is included
rather than silently discarded. Source hashes and raw update history make this
limitation inspectable.

Individual trial parameters are available through
`python -m bench.paged_attention_ppo --help`; keep both arms identical. The
fresh single-process measurement is useful engineering evidence but cannot
substitute for the trained checkpoint and the real 8-rank/shared-GPU layout.
The script does not recreate or modify that production topology.

## Acceptance decision

There is **no measured CUDA or end-to-end speedup for this candidate yet**.
Do not merge it as an enabled production optimization on the strength of CPU
recurrence checks or an isolated kernel ratio. Retain the eager default and
require successful GPU execution, real trained-policy behavior replay, finite
bounded PPO training, and repeatable complete-PPO improvement in the intended
topology. Check playing strength separately if floating-point differences cause
training trajectories to diverge. Negative measurements should remain in the
report, and a slower candidate should remain disabled or be removed.
