# History trainer refactor and speed options — September 29, 2026

Status: code and CPU measurements only. No GPU was used. Nothing here is a
GPU speed claim or a playing-strength claim. The base is the production
trainer of the overnight run (`longrun-batched-2026-09-28`, merged into this
branch at `b88037f`).

## What changed

Default behaviour (tier 1: bitwise identical on CPU):

| Change | Where |
|---|---|
| `PublicStream` keeps events in contiguous growable arrays; `arrays()` and the `tokens/rounds/phases` properties are read-only O(1) views. Stream batches upload in one packed copy | `train/history_model.py` |
| One PPO loop for the base and data-parallel trainers (`epoch_batches`, `backward_minibatch`, `reduce_gradients` hooks) instead of two copies of `learn()` | `train/history_ppo.py`, `train/history_ddp.py` |
| Each minibatch's actor inputs, stored row fields and response targets go to the device in one packed upload (`SequenceRolloutBuffer.training_batch`); the critic reuses the actor's observation tensor | `train/history_rollout.py` |
| Loss terms and gradient norms come back in one transfer per minibatch instead of one `float()` per encoder parameter and per term (~60 host syncs per minibatch before); the finiteness check still runs before the optimizer step | `train/history_ppo.py` |
| `HistoryCollector.step` split from one 200-line method into named phases over two small records | `train/history_rollout.py` |

Opt-in options (all resume-overridable with `--resume-set`, recorded in the
manifest and `config_changes`):

| Option | Tier | What it does |
|---|---|---|
| `--rollout-paged-cache` | 1 on CPU | All identities' public K/V and memory live in pages of one shared pool. An encode call runs a fixed number of tensor operations (one indexed K/V write and one row gather per layer) instead of several copies per cached match, and needs no per-match pointer tables. Replaces `--rollout-triton-cache` |
| `--batch-snapshot-encoder` | 2, snapshot seats only | With merged snapshot heads and the paged cache: all snapshot identities' new public events are encoded in one pass per vector step over stacked encoder weights (was ~11 encode calls per step on the main lineage) |
| `--learner-length-groups N` | 2, learner | Each PPO minibatch is encoded in up to N groups of matches with similar length, each only as far as its rows read. Causal attention makes this the same function |
| `--rollout-page-span` | 2 | Paged cache attention keys and decision memory are padded to the next 64-token page instead of the next power of two. Not with private CUDA graphs |

`rollout_triton_cache` and `rollout_private_graphs` became resume-overridable
so a lineage that uses them can switch to the paged cache.

## Verification

- **Bitwise harness.** 6 CPU updates from the same seed, digests of every actor and critic tensor, Adam state, sampler and NumPy RNG, population and every non-timing metric, baseline tree `b88037f` against this branch: `plain`, production-shaped (`causal_sdpa`, KV cache, batched and wide rollout attention, batched learner attention, merged snapshot heads, auxiliary response head), exploration floor (T 1.3, ε 0.05) and window-8 explicit-response configurations match exactly; so do the production-shaped and exploration runs with `--rollout-paged-cache`, and with `--batch-snapshot-encoder` as well (its float noise flipped no snapshot choice in 6 updates). Two-rank DDP with the global minibatch: weights and both ranks' metrics match over 5 updates. Runs use one intra-op thread: under concurrent load, multi-threaded CPU runs of the unchanged baseline were themselves not reproducible.
- **Paged cache.** Randomized test against the per-entry cache (resets, pruning, pool growth, weight invalidation, two identities sharing a pool): outputs, counters and entry lengths equal; page accounting and the zero page checked every step. Mutation checks: dropping the zero-page mask or the BOS write fails the test.
- **Merged snapshot encoder.** Within 2e-5 of each identity's own encode (prefill chunks, fresh BOS rows, uneven rows per slot, a slot gap). A stale stacked encoder is caught: the refresh test fails if the encoder parameters are dropped from the slot signature.
- **Length groups.** Loss and every gradient within 1e-8 of the padded minibatch (both attention backends, response head on and off, batched and per-match learner attention). The plan partitions matches and rows and encodes each row's full prefix.
- **Page span.** Within 1e-5 of the per-entry cache on every real position. Most CPU SDPA calls stay bitwise; single-token appends do not.
- Suite: 1,143 passed, 290 skipped (CUDA-only) locally.

## CPU measurements

Apple M4 Pro (14 cores), torch 2.14, 4 intra-op threads, one process at a
time. Production architecture (width 128, 4 layers, 8 heads, auxiliary
response head), 64 environments, 64 steps per update, 16 matches per
minibatch, snapshot every update, production rollout flags. 22 updates from
scratch; the last 10 are measured (mean prefix 900-1,000 tokens).

| Arm | Collect (s / 10 updates) | Learn (s / 10 updates) | Total |
|---|---:|---:|---:|
| Baseline tree `b88037f` | 78.5 | 67.9 | 146.4 |
| This branch, default flags | 78.9 | 70.0 | 148.9 |
| + `--learner-length-groups 2` | 81.0 | 38.8 | 119.8 |
| + `--learner-length-groups 4` | 75.8 | 29.5 | 105.3 |
| + 4 groups, paged cache, merged encoder | 78.1 | 29.5 | 107.6 |
| + 4 groups, paged cache, merged encoder, page span | **55.1** | **31.9** | **87.0 (1.71x)** |

Arms diverge (tier 2 changes float order), so their histories differ a little
(mean prefix 886-980); single run per arm, no alternation.

- The tier-1 changes are neutral on CPU, as expected: they remove CUDA host
  syncs, launches and per-match host work.
- **Learner padding.** On real minibatches only 47% of the padded encoder's
  linear work and 31% of its attention work was on real tokens. Four length
  groups cut learn time 2.4x.
- **Page span.** CPU flash attention processes every masked key, and batches
  of ~900-token histories were padded to 2,048 keys. Page padding cut collect
  by 1.42x.
- **Paged cache and merged encoder: no CPU gain.** CPU collection is
  compute-bound (SDPA is ~60% of collect). In a host-bound proxy (155 streams
  of ~40 tokens, width 16, one thread) a paged encode takes 2.7 ms against
  5.5 ms. Whether this matters on CUDA is the open question below.
- **Launch-count proxy.** Tensor operations that would launch a kernel,
  counted at the dispatcher (views and metadata excluded), per vector step in
  steady state (64 envs, width 64, 4 layers, merged snapshot heads, 32 steps
  after the learner rebuild): per-entry cache 1,926; paged cache 1,021;
  paged cache + merged encoder 466 (4.1x fewer). The per-entry count includes
  the pack copies that production's Triton kernel already fuses, so the gain
  against the Triton path is smaller than this ratio.

## What is not known

- **CUDA speed of every option.** On the RTX 4090 the main lineage's
  collection was host-bound: 0.3 ms per cached row per call, and ~11
  snapshot encode calls per step. The paged cache and merged encoder remove
  exactly that per-row and per-call work, but that is an expectation, not a
  measurement. A parallel analysis of the same Sept 28 profile (branch
  `XiyaoWang0519/find-training-speedups`) puts the GPU at 92% utilization but
  only ~207 W, with ~3,100 kernel launches per step per rank (~2,750 in the
  snapshot encodes) and learn at ~0.1 TFLOP per minibatch: kernel-count bound,
  not FLOP bound. If so, the options that remove FLOPs (length groups, page
  span) may gain little on the GPU, while the ones that remove launches (paged
  cache, merged encoder) are the ones to test. Length groups add encoder
  launches; include `learner_length_groups=1` (prefix truncation only) and `2`
  arms.
- **Overlap with that branch.** It independently implements contiguous public
  streams, the learn-phase sync collapse and a merged snapshot encode (over the
  Triton cache instead of a page pool), plus a stable Triton pointer table and
  a CUDA MPS arm, with its own GPU A/B kit. The two branches conflict in
  `history_model.py`, `history_ppo.py`, `history_rollout.py` and
  `history_snapshot_batch.py`; only one set of those levers should be merged.
- **Memory.** The page pool grows by 25% steps and never shrinks. Its peak is
  roughly the live cache peak; allocator trims do not release it. Found in
  review: `invalidate_learner_cache` now only returns the learner's pages to
  the free list, so, unlike the per-entry cache, that memory stays reserved
  through PPO learn (on the order of hundreds of MB per rank at width 128), and
  a growth briefly holds old and new pools (about 2.25x). With four ranks on a
  GPU already at 23.6 of 24 GB, the paged-cache arm may run out of memory;
  watch peak memory in the A/B. Possible fixes: size the pool once, or give
  the learner its own pool that is freed before learn.
- **CUDA numerics.** The paged cache feeds SDPA strided key/value views. On
  CPU this is bitwise; on CUDA it may be float-order noise (tier 2) until the
  CUDA test variants pass.

## Next step on a GPU (needs a rental decision)

1. CUDA gate (about 5 minutes):
   `pytest -q tests/test_history_paged_cache.py tests/test_history_length_groups.py tests/test_history_arms_bench.py tests/test_history_snapshot_batch.py`.
2. In-run A/B from the main-lineage checkpoint. This never writes to the
   lineage; the first 25 updates refill histories, then ABBA blocks run:

   ```sh
   python -m bench.history_arms --resume latest.pt --output /tmp/arms --device cuda \
       --world-size 4 --warmup 25 --blocks 6 --block-updates 6 \
       --arm base= \
       --arm fast=rollout_triton_cache=false,rollout_private_graphs=false,rollout_paged_cache=true,rollout_page_span=true,batch_snapshot_encoder=true,learner_length_groups=4
   ```

   About 100 updates at ~16 s each is roughly 30 minutes, about $0.30 on the
   usual Vast 4090. Arms may differ only in options that switch in place
   (`bench.history_arms.SWITCHABLE`).
3. If it is faster, continue the lineage with `--resume-set` for the chosen
   options. The tier-2 options change float order only; per the acceptance
   rules they need no strength A/B, but the next 256-deal evaluation should
   name the source change.

## Review and merge (September 29, 2026)

Merged into `history-budget-2026-09-27` with two commits from
`XiyaoWang0519/find-training-speedups` that this branch did not cover: the
Triton cache-wide pointer table and the cProfile hook, plus its learner
changes ported onto the shared PPO loop (foreach encoder norms, DDP gradients
as views; bitwise on CPU against the merge, 4 configurations and 2-rank DDP).
Its PublicStream rewrite duplicated this branch's; its Triton-based
`--batch-snapshot-encode` competes with `--batch-snapshot-encoder` and stays
on that branch until the GPU A/B picks a cache. An independent review found
the default path clean (a 3-rank DDP run with an empty rank matched the base
bitwise) and raised the memory point above and the paged cache's CUDA tier.
