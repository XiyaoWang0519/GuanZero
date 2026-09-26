# History stack optimization: local preparation and verification

This document preserves the pre-approval local evidence. The subsequently
approved comparison is complete; see the [CUDA receipt](history-stack-cuda-2026-09-26.md)
for current launch hashes, repeated speed/memory results, quality limits and
verified shutdown. Optional flags remain off by default.

The completed [T4 pilot](history-t4-pilot-2026-09-26.md) spent 278.64 seconds
collecting and 21.59 seconds learning: collection was 92.8% of those two phases.
Its 24.31 GB peak allocator reservation, versus 1.55 GB peak active tensor
allocation, also required investigation before scaling. No new GPU resource
was created for this follow-up. All timings below are CPU frozen-policy
diagnostics, not CUDA training speed or playing strength.

## Changes

1. **Optional causal SDPA encoder.** `train/history_attention.py` executes the
   existing pre-norm, zero-dropout layers with their existing weights. It sends
   the causal hint directly to SDPA instead of constructing the old combined
   causal/padding mask for every batch and head. Trailing padding cannot affect
   a valid causal prefix. Both full-prefix inference and gradient recomputation
   can use this path. CUDA kernel selection and allocator savings remain to be
   measured; removing the explicit mask is not a guarantee of a particular
   fused kernel or linear-memory attention on every device.
2. **Optional batched public KV cache.** `train/history_inference.py` stores
   per-layer keys/values and public output states by policy/environment/match.
   Calls upload and encode only new public events, with a maximum prefill
   chunk of 128 tokens. Power-of-two storage growth reduces allocation shape
   churn. Capacity grows to retain the whole match; there is no history cap,
   window reduction, precision change or action filtering.
3. **Explicit invalidation and lifetime.** Learner caches are released before
   PPO computes gradients and rebuilt from the raw prefix under the new
   weights. Frozen-snapshot caches survive learner updates, but are evicted
   when no current match uses that policy/match. Ordinary in-place optimizer
   or `load_state_dict` weight changes invalidate a policy cache automatically.
   Replaced/reset public streams invalidate their entries. Direct `.data`
   mutation bypassing PyTorch version counters is unsupported. No cache is
   serialized; resumed environments start fresh. PPO always recomputes from
   raw public history with gradients, including carried rows from older rounds.
4. **Less synchronization and more useful measurements.** Candidate counts
   already known on CPU size the sampling table; ragged row expansion gets
   its known output size. The chosen log-probability is checked after its
   required CPU download, avoiding another device-side boolean synchronization.
   Optional synchronized timings separate environment work, event handling,
   input preparation, public cache/collation, actor/sampling, downloads and
   buffering. Metrics add live cache bytes, active/reserved memory,
   inactive-split peak and allocation retries. Profile timings are explicitly
   labeled and kept separate from ordinary throughput trials.

Flags on `train.history_ppo`: `--causal-sdpa`, `--rollout-kv-cache`, and
`--profile-collection`. **All default to off pending CUDA comparison.** The
architecture and checkpoint tensor names are unchanged. The original dense
path remains the control. This is a new source version; T4's immutable archive
and checkpoints remain intact, and its strict source-identity resume check has
not been weakened to silently load across code versions.

## Local evidence

A frozen mixed-policy profile, eight environments / width 64 / two layers,
spent about 0.805 of 1.001 profiled seconds in public-prefix encoding. The
64-step window invoked the actor 218 times because seats used different policy
identities. This is local diagnostic evidence locating duplicate work; it does
not establish the CUDA bottleneck split.

The repeat-order comparison used `dense → sdpa → kv → kv → sdpa → dense`, each
in a fresh process on this Mac with two Torch/BLAS threads. Twenty 64-step
chunks were collected; the first four were warm-up. Every measured case made
8,192 environment decisions and reached a 2,313-token prefix. Policy weights
were frozen, with four same-weight snapshot identities to exercise the mixed
policy dispatch and cache ownership. No CPU optimizer steps ran.

| Path | Measured collection seconds, two repeats | Collection decisions/s | Interpretation |
|---|---:|---:|---|
| Dense control | 104.682 / 102.439 | 78.3 / 80.0 | original explicit-mask encoding |
| Causal SDPA | 41.107 / 40.482 | 199.3 / 202.4 | still recomputes the complete prefix |
| KV, weights/cache retained | 5.497 / 5.473 | 1,490.2 / 1,496.9 | optimistic frozen-cache diagnostic |
| KV, learner cache invalidated each chunk | 6.572 / 6.871 | 1,246.5 / 1,192.3 | includes rebuilding every 64 vector steps |

The last row was a separate follow-up; its public encoder/collector code is
identical to the first comparison. Only the benchmark added the explicit
invalidation schedule. This is now the default in `--collect-only` mode;
`--keep-learner-cache` requests the optimistic frozen-cache control explicitly.
The rebuilt-cache diagnostic is about 15–16 times faster than the dense CPU
control for this workload, **not a predicted GPU or end-to-end training gain**.
It still excludes gradient work, actual changing weights and evolving snapshot
strength. Live cache tensor storage peaked at 52.76 MB, excluding model weights
and temporary workspaces. CUDA allocator behavior has not been measured.

All eight cases had identical per-chunk decision counts, mean-prefix and
maximum-prefix traces. Separate real mixed-policy rollout tests verify exact
sampled choices and PPO row contents; cached log-probabilities agree within
`2e-5`. Other tests cover ragged/empty prefixes, batched/reordered queries,
private query separation, prefill through 2,403 tokens without truncation,
incremental append, reset/replacement, policy separation, eviction, weight
invalidation, encoder gradient parity and save/resume with cache rebuild.

The full regression passed: **727 Python tests, 6 skipped; 81 C++ cases;
14 oracle groups; three 20,000-round deep fuzz modes with zero failures**.
Five skips require CUDA; the remaining skip is an existing optional test.
The CUDA-specific cache, long-prefix, gradient and PPO-resume tests are
declared and remain unexecuted locally.

Raw diagnostics and logs: `.work/history-stack-2026-09-26/`:
`before-profile.txt`, `cpu-mixed/comparison.json`, `cpu-rebuild/comparison.json`,
per-case metrics, `targeted-tests.log` and `full-check.log`.

## Prepared CUDA comparison

`bench.history_stack` provides bounded, alternating comparisons under the
same model, seeds, eight environments, full history, FP32 and no TF32. On CUDA
it performs real PPO work, so learner-cache rebuilding and evolving frozen
policies contribute to the measured cost. CPU mode rejects optimizer runs.
Each result preserves per-prefix throughput, unique learner rows per
collection-plus-learning second, memory and source/runtime/host identity.

The proposed kit is `.work/runpod-history-stack-2026-09-26/`:

- One RTX 4090 Secure, **at most $1.20 total, $0.80/hour including disk,
  90 minutes from creation through cleanup**. The live quote and resource
  state must be checked at creation. This is a new approval scope; the first
  T4 rental is already closed.
- CUDA parity/privacy/gradient/resume tests first; failure stops the workload.
- All-current and mixed-population workloads separately. Each compares dense,
  causal SDPA and KV in forward/reverse order: two repeats, 24 updates per
  case, four warm-up updates, 120-second limit per case.
- Two separate synchronized mixed-population profile cases (SDPA and KV),
  excluded from throughput comparisons. Public encoding moves between the
  cache/collation and actor stages, so compare their sum when interpreting
  phase timings. CPU ticks, cgroup facts and GPU telemetry are retained.
- A 45-minute remote watchdog plus independent local provider guard; health
  checks read sample progress, gradients, ratios and history/memory metrics.
  Sync every 30 seconds, verify remote hashes before normal teardown, then
  independently confirm no pods and zero hourly spend.

The plan does not automatically select or enable an optimization. CUDA
correctness, repeatable complete-update improvement, long-prefix coverage and
memory behavior must support that decision. Preserve slowdowns and insufficient
coverage as failures/limitations rather than increasing the budget or choosing
only favorable cases. No MLP assets, final-test material or provider credentials
are included in the source/payload upload.

Candidate source SHA-256:
`e42a228124dd52f4540c3247f05ecd8ecac186f47d0025f9f1216e89a3a5c2fd`.
Archive SHA-256:
`99c993fd8ad7f9531fae12670de6096463eaf388f2250673dbc66c982cd96a1f`.
No GPU result, memory fix on CUDA, or playing-strength improvement was claimed
by this preparation. The separate CUDA receipt records later engineering
evidence; it does not claim long-run playing-strength equivalence.
