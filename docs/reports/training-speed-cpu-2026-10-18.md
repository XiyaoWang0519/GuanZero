# Training speed pass without a GPU (October 18, 2026)

Status: **implemented and CPU-verified; GPU throughput not measured.** No GPU
was rented, no cluster job was submitted, and the main lineage was not touched.
Every change preserves PPO/GAE/reward/population semantics, FP32, full public
history, the full candidate set and the public-only input boundary. GPU gains
below are estimates from the October 14 GPU breakdown plus CPU measurements,
op counts and byte counts; they are not measurements.

## What was done

Eight read-only scouts profiled the trainer subsystem by subsystem
(collectives, rollout host work, KV cache, actor inference, learner, buffer,
engine, whole update) against the production configuration (width 128, 4
layers, 8 heads, 8 ranks x 128 tables, 64 steps, 2 epochs, aux heads, the
October 15 option set) and the measured GPU split
([encode/wait diagnosis](training-encode-wait-2026-10-14.md),
[append detail](training-append-detail-2026-10-14.md)). Nine implementation
tracks then landed in isolated worktrees, each with its own tests, and every
merged track went through an adversarial review (one reviewer per track, a
refuting verifier per non-trivial finding). Reviews found no critical or high
defect; one medium defect (the Triton launch lacked a CUDA device guard) and
the low-severity findings were fixed (listed at the end).

Two kinds of change:

- **Default path, bitwise on CPU.** Same tokens, rows, RNG draws, losses,
  gradients and parameters as before (proved by tests that run the old and new
  code side by side, several against frozen copies of the pre-change code).
  They take effect when a run restarts on this source
  (`--allow-source-change`; the v2 engine digest is unchanged).
- **Opt-in switches.** Every one defaults to the old behaviour, is
  resume-settable (`--resume-set`), recorded in the manifest, and switchable
  between `bench.history_arms` arms. Most are tier 2 (FP32 order); `ddp_reduce`
  is bitwise with gloo but opt-in for operational reasons.

## Default-path changes (bitwise, always on)

| Area | Change | CPU evidence | Expected GPU-host effect (unmeasured) |
|---|---|---|---|
| Event drain | `VecEnv.drain_public_arrays` binding; tokens built in bulk with NumPy (layout and tribute-flag zeroing stay in `train/logs.py`); `PublicStream.extend_tokens`; per-run ingest | events_and_rounds 0.214 -> 0.057 s/update (tiny model, 128 envs) | ~2% |
| Collector rows | one uint8 cast per step, row/stream reuse in `_read_rows`/`_plan`, memoized snapshot head slots inside the frozen-weights scope, NumPy upload staging without zero-fill | host phases 0.385 -> 0.153 s/update; collect median 3.13 -> 2.82 s | 2-4% with the drain |
| Rollout buffer | growable double-buffered row storage, O(1) `compact()`, vectorized `next_iteration`/`finalize`, one value download, one-copy snapshot digest | 55 -> 35 ms per rank-update (replayed trace) | ~0.5% |
| Learn step | one actor norm pass (clip + encoder diagnostic), one backward call, one checksum collective, async per-epoch advantage moments, next minibatch collated during this backward | clip/stats ops 139 -> 21 per minibatch; parameters bitwise over 3 updates and at 3 ranks | 0.5-3% |
| Adam on resume | step counters returned to the CPU after a `map_location=cuda` load | bitwise | ~220 fewer `.item()` syncs per minibatch on every resumed run (est. 1.5-2%) |
| Learner forward | no mid-forward host-to-device copy (`aux_coefficient`), cached position table, host-planned candidate rows and next-token mask, zero response bridge without the matmul | 949 -> 870 ops per minibatch; one stream sync per minibatch removed | 0.5-5% |
| Sampling | host-planned table index/counts, no-grad softmax and Gumbel trims, no duplicate downloads | 18 fewer ops and 2 fewer downloads per vector step | 0.2-0.7% |
| KV cache | one float mask per append, page-granular memory gather, cheaper row index, uninitialized pool growth with new pages zeroed | 18.4 -> 16.1 CPU s per replayed update | 0.3-1% |
| Checkpoints | `latest.pt` and `update-N.pt` from one payload and one serialization | 61 -> 48 ms at 8.7 MB | per checkpoint on rank 0 |
| DDP startup | each rank builds its environments and collector once (`rank_streams` hook) | bitwise streams and files | startup only |

## Opt-in switches

| Option (default) | Tier | What it does | CPU evidence | Expected GPU effect (unmeasured) |
|---|---|---|---|---|
| `--ddp-reduce shm` (gloo) | bitwise vs gloo | gradient all-reduce through a `/dev/shm` mapping that adds in gloo's ring order; start-up self-check, gloo fallback | 2.5-3.6x faster per call at W=2/4; parameters sha-equal to gloo at W=2 and W=4 | 0.3-0.4 s/update (7-10%): the measured gloo all-reduce was 0.46 s/update |
| `--rollout-span-groups 2` (1) | 2 | sort each append by end and attend in up to N groups; same for learner and merged snapshot memory | padded K/V rows -31%; replayed cache work 16.1 -> 10.1 CPU s | 7-10% (calibrated on page_span) |
| `--rollout-private-kv-pages` (off) | 2 | learner pool stores `kv_proj(memory)` per token; no per-step projection of the whole history | replayed learner attention 3.1 -> 0.5 CPU s | 0.5-2% alone, 2-4% with span groups |
| `--rollout-causal-prefill` (off) | 2 | learner prefill as one causal pass per length group | CPU-neutral; logp within 4.8e-7 | 0.5-1.5% |
| `--rollout-paged-attention-kernel` (off) | 2 | Triton kernel reads K/V in place from pages up to each stream's end (appends and learner decode) | interpreter parity 4.8e-7; AOT-compiles for sm_89, FP32 FMA only, no spills | 5-9% alone, 3-6% on top of span groups; needs `bench/paged_attention_bench.py` first |
| `--learner-compact-next` (off) | 2 | next-token head and fused loss on valid positions only | ops 870 -> 781 per minibatch; rel. diff 2.5e-7 | 1-3.5% if launch-bound |
| `--adam-fused` (off) | 2 | fused Adam kernels | parameter diff ~1e-8 after 3 steps | 0.3-0.5% |
| `--rollout-trim-slack-mb N` (0) | allocator only | skip `empty_cache` while releasable cached bytes <= N MiB | decision logic over a faked allocator | 1-3% (historical trim cost 0.12-0.145 s/update) |
| `--rollout-merged-head-prefold` (off) | 2 | pre-multiplied snapshot head projections | 61 -> 53 ops per merged call | ~0.2% |

These estimates are not additive: several act on the same GPU time, and the
ranks share GPUs under MPS. If the estimates hold, the default path is worth
roughly 5-15% and the opt-in set another 15-30% of update throughput; a GPU
A/B decides.

## End-to-end CPU check

Whole `HistoryTrainer.update()` on this CPU, single process, the main
lineage's option set (`bench/cpu_profile.py`), baseline tree 05d6251 (the
October 15 source plus the profile harness) against this branch, same seeds,
processes interleaved base/new/opt over two rounds, 4 timed updates each after
the warmup (6 for the production shape, 4 for the tiny one). Means over both
rounds, seconds per update; the CPU does the model arithmetic, so the learner
column says little about a GPU.

| Workload | Arm | Wall | Collect | Learn | Mean prefix |
|---|---|---:|---:|---:|---:|
| width 128 x 4 layers, 32 tables x 64 steps | baseline | 18.45 | 9.79 | 8.64 | 900 |
| | this branch, defaults | 16.38 (-11%) | 7.51 (-23%) | 8.86 | 900 |
| | + opt-ins on CPU* | 13.23 (-28%) | 5.12 (-48%) | 8.10 | 847 |
| width 16 x 1 layer, 128 tables x 64 steps | baseline | 6.76 | 2.18 | 4.56 | 694 |
| | this branch, defaults | 6.14 (-9%) | 1.73 (-21%) | 4.40 | 694 |

The default arm's trajectories equal the baseline's (same prefixes and
decisions), as the bitwise claims require. *Opt-ins on CPU:
`rollout_span_groups=2`, `rollout_private_kv_pages`, `rollout_causal_prefill`,
`learner_compact_next`, `adam_fused`, `rollout_merged_head_prefold`
(`ddp_reduce` needs ranks; the Triton kernel is a no-op on CPU). They change
FP32 order, so their trajectories diverge and their histories were about 6%
shorter in this window; part of that arm's gain is less work, not faster
work. Each round's ordering was the same; absolute times moved about 10%
between rounds on this shared VM. Production GPU ranks spend proportionally
less in model arithmetic, so the host-side share of these savings matters more
there, but no number here is a GPU throughput.

## GPU validation plan (in this order)

1. CUDA unit tests on an L40S with the cluster's torch: the CUDA parameters of
   `tests/test_history_paged_attention.py`,
   `tests/test_history_rollout_paged_kernel.py`,
   `tests/test_history_paged_cache.py`, `tests/test_history_rollout_memory.py`,
   `tests/test_history_buffer_storage.py`, `tests/test_history_trim.py` and
   `tests/test_history_sampler_trims.py` (run with deterministic algorithms).
2. `python bench/paged_attention_bench.py --json .work/paged_attention.json`
   (kernel speed, first-call compile time, error against FP64).
3. In-run A/B on the latest checkpoint in the production layout. `--spread-gpus`
   is new in `bench.history_arms`; without it every rank lands on GPU 0:

   ```sh
   python -m bench.history_arms --resume latest.pt --output /tmp/arms \
       --device cuda --world-size 8 --spread-gpus --warmup 25 --blocks 6 \
       --block-updates 6 --arm base= --arm shm=ddp_reduce=shm
   ```

   then, one variable per arm against `base=ddp_reduce=shm`:
   `rollout_span_groups=2`, `rollout_private_kv_pages=true`,
   `rollout_causal_prefill=true`, `rollout_paged_attention_kernel=true`,
   `learner_compact_next=true`, `adam_fused=true`,
   `rollout_trim_slack_mb=<from production cuda_trim readings>`.
   The new code's default path is itself the comparison baseline: compare it
   against the October 15 source in separate processes for the bitwise
   changes.
4. Tier-2 options additionally need the real-rollout log-probability gate used
   for page-span ([stack A/B](training-stack-ab-2026-10-13.md)): identical
   choices and RNG, pairwise logp within 1e-4, full-history recompute within
   tolerance, finite gradients over 120 PPO updates.

## Proposals not implemented (need a decision or a GPU)

- Re-gate the full key/value fold of the learner's rollout attention against
  an FP64 reference instead of pairwise 1e-4: it measured +8.97% on October 13
  and failed the pairwise gate at 1.37e-4 while being closer to FP64 than the
  current path. Changing the gate is the owner's call.
- Rank layout: with 3 healthy GPUs, 8 ranks split 3/3/2 and one GPU idles at
  every collective; choose a world size divisible by the GPU count, keeping
  tables x ranks and minibatch matches x ranks constant (changes the batch
  layout record, not the learning rule).
- `num_threads=2` per rank and `GUANZERO_PINNED_UPLOAD=1`: config/env-only
  A/Bs; the pinned upload was measured without gain in September, before the
  paged cache added several blocking uploads per step.
- `learner_length_groups=3` instead of 2: cuts encoder token slots ~15% but adds
  ~275 ops per minibatch; the sign is unknown without a GPU.
- Overlapping `env.step()` + `pending()` with the step's buffer writes on a
  worker thread (bitwise, est. 0.5-1%); not built.
- Retiring legacy paths that production no longer uses (Triton cache copies,
  private CUDA graphs, the in-process arm schedule, ~1,000 lines).

## Caveats

- Rebuild the extension on the cluster (`cmake --build build`) before running
  this source: the array drain needs the new binding. Without it the collector
  silently keeps the per-event drain (same results, no speedup).
- `python/bindings.cpp` changed (new drain binding). The resume engine digest
  (v2) excludes it, but `train.tribute_data.engine_source_digest` includes it,
  so tribute/belief side-experiment datasets written before this change no
  longer pass their provenance check and must be re-collected.
- The rollout buffer keeps two row storages at peak capacity: about 40-90 MB
  more host RAM per rank.
- `--ddp-reduce shm` needs about (W + 2) x 8.6 MB of `/dev/shm` (86 MB at
  W = 8; container defaults can be 64 MB, which falls back to gloo). Its order
  model was derived on torch 2.14's gloo; the start-up self-check guards the
  cluster's torch 2.11.
- With the paged attention kernel the first update JIT-compiles several kernel
  variants per rank.
- CPU timings on this 4-core VM were taken while other agents ran; the paired
  and replayed comparisons are the evidence, absolute times are not.

Review findings fixed in this pass: the Triton kernel launched on the current
CUDA device rather than its tensors' device (a single process on a non-default
`cuda:N` would fail; production ranks set their device); fused Adam did not bump parameter version
counters (now bumped, tested); the trim slack gate counted unreleasable
fragments; stale `--ddp-reduce` help; causal prefill over-padded budget
passes; the kernel misread strided start/end views and accepted shapes that
exceed sm_89 shared memory; a cached position table under CUDA graph capture;
`bench.history_batch_host --baseline` and `bench.history_snapshot_layout
--reference-source` against pre-change sources; an untested cached `arange`.
Noted, not changed: `rollout_span_groups` > 1 also groups the merged snapshot
call's memory (the flag is new in this pass, so no earlier run used it).

## Files and tests

New modules: `train/history_shm_reduce.py`, `train/history_paged_attention.py`;
new benches: `bench/cpu_profile.py`, `bench/allreduce_bench.py`,
`bench/paged_attention_bench.py`. New test files:
`test_history_shm_reduce.py`, `test_history_learn_step.py`,
`test_history_learner_ops.py`, `test_history_buffer_storage.py`,
`test_history_host_paths.py`, `test_history_rollout_memory.py`,
`test_history_paged_attention.py`, `test_history_rollout_paged_kernel.py`,
`test_history_sampler_trims.py` (and a frozen pre-change cache copy,
`tests/history_paged_reference.py`). Final check on the merged branch
(everything `scripts/check.sh` runs): C++ engine tests 84 cases, oracle tests,
1544 Python tests passed with 414 CUDA-only skips, and the three 20,000-round
fuzzers with 0 failures. The C++ engine sources are unchanged.
