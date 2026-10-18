# Training speed deep dive — October 1, 2026

Status: code reading, the counters of the last main-lineage run and CPU
measurements on the Mac. No GPU was used. The savings named below are
estimates, not measurements, until the prepared GPU session runs.

## Where an update goes

Main lineage, updates 2724-4902 (`.work/longrun-u2623-2026-10-01/`, 4 ranks x
256 tables, RTX 4090, CUDA MPS, source `7bd7f10`), rank 0, medians:

| Phase | Seconds | Share |
|---|---:|---:|
| Collect | 7.13 | 57% |
| Learn | 5.14 | 41% |
| Other (allocator trim 0.145 s, checks, logging) | 0.21 | 2% |
| Wall | 12.55 | about 5,200 decisions/s |

Per rank and update: 9,944 learner rows, 28.7 minibatch steps of about 694
rows and 16 matches, 14 resident snapshots, mean prefix 627 and maximum 1,992
tokens. GPU utilization 85% (median) at 256 W of 450 W, memory utilization 31%;
peak allocated 3.4 GB and KV cache 1.4 GB per rank.

## Findings

1. **Snapshot public encoding is one pass per identity**, about 11 passes per
   vector step. With the learner's pass and the rebuild passes below, a step
   runs about 13.6 encode passes; encoding was about 85% of collection in the
   last direct profile (September 28). `--rollout-paged-cache
   --batch-snapshot-encoder` merges the snapshot passes into one. It is on
   `main` but has never passed on CUDA.
2. **The learner's KV cache is rebuilt lazily after every update.** A stream is
   re-encoded by the first step whose learner row reads it, in 128-token
   chunks, so the rebuild is spread over several steps and its tail runs in
   batches of one to three streams. Rank 0 counters over 2,279 updates:
   226,695 Triton append passes and 99,878 eager ones (below
   `rollout_triton_min_batch` 4), that is 143 learner passes per update for 64
   steps. 182,500 tokens are encoded per update, about 160,000 of them by this
   rebuild.
3. **`rollout_private_graphs` does nothing in production.** The learner's
   graph was replayed 49 times in 145,856 forwards. The call's
   `kv_proj(encoded)` activation (155 x 2,048 x 256 FP32, 325 MB) exceeds the
   256 MB per-policy budget, so 145,166 calls were skipped by size. One unused
   entry holds 236 MiB per rank. `--resume-set rollout_private_graphs=false`
   frees it.
4. **Learn is unexplained.** In the October 1 A/B it rose from 2.5 s to 4.8 s
   as histories refilled after a resume. On CPU 47% of the padded encoder's
   linear work and 31% of its attention work was on real tokens.
   `learner_length_groups` was measured on CUDA only with 4 matches per
   minibatch (slower); production uses 16.
5. **Measured small** (M4 Pro, CPU): host Python outside the model 3.5 ms per
   step at 256 tables and 14 snapshot identities (event ingest 1.9 ms); the
   gloo all-reduce of the 8.3 MB gradient 3.3 ms per minibatch at 4 ranks. A
   metrics line is 74 KB per rank and update and grows, because the
   population's per-snapshot counters are never pruned.

## Changes in this branch (CPU-verified)

- `--rollout-prefill-learner-cache` (opt-in, resume-overridable, tier 2 on
  learner seats): `HistoryCollector.collect` first brings every current match
  with a learner seat up to date in one batched pass per chunk
  (`BatchedHistoryCache.prefill`). Synthetic CPU check, 256 tables, 14 snapshot
  identities, maximum prefix 630: 91 learner passes per 64 steps without the
  flag, 68 with it. Stored log-probabilities differ in the last FP32 bit;
  choices, sampler state and every other stored field were identical in the
  tests.
- Paged cache: the learner has its own `KVPagePool`. `invalidate_learner_cache`
  returns its storage before PPO learns (`KVPagePool.reset`); the next
  allocation restores the previous peak in one step without a copy. This
  answers the September 29 review point that learner pages stayed reserved
  through learning. Paged plus merged-encoder training stays bitwise equal to
  `main` on CPU.
- `--profile-learn`: synchronized wall time per learn phase in
  `learn_phase_seconds` (values and GAE, batch and upload, forward, backward,
  gradient reduce, norms and clip, optimizer).
- `GUANZERO_TORCH_PROFILE_DIR` with `GUANZERO_TORCH_PROFILE_UPDATES`: the
  listed updates of a process run collect and learn under `torch.profiler` and
  write per-operator and per-kernel totals per rank.
- `bench.history_arms`: profile flags and the prefill switch in place; rows
  carry encode passes and phase times; profiled updates are left out of the
  timings; a failed rank stops the others instead of leaving them in a
  collective.
- `cache_metrics()["appends"]`: encode passes, in every metrics line.

Checks: 1,321 tests pass locally (317 CUDA-only skipped). Four CPU updates in
the plain, production-shaped and paged configurations match `main` (`2b59309`)
in every weight, optimizer state, buffer, random state and loss.

## Not known until the GPU session

Whether the merged snapshot encode and the paged cache pass on CUDA and fit in
memory with four ranks; the speed of every option above; what bounds learn
(kernel launches or padded attention). The kit `.work/speed-prep-2026-10-01/`
runs the CUDA gates, an in-run arm comparison with `bench.history_arms` under
MPS, and a kernel-level profile.
