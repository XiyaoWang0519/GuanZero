# Learner CUDA benchmark — September 30, 2026

Measured result: no reliable complete-PPO speedup from the reviewed learner changes on this RTX 4090 workload. Chosen-only response prediction improved the grouped fixed-batch learner probe by about 10%, but the grouped complete update improved by only 1.5%, inside process-to-process variation. The fastest complete-update setup was the ungrouped baseline. No flags were promoted and no playing-strength claim follows.

## Complete PPO measurements

Decisions/s, average of two balanced process trials per cell:

| Length groups | Baseline | New default | New chosen-only |
|---|---:|---:|---:|
| 0 | 966 | 956 | 954 |
| 4 | 726 | 726 | 736 |

Ungrouped new default: -1.0%; new chosen-only: -1.2% versus baseline. With four length groups: new default approximately unchanged; new chosen-only +1.5%. Baseline ungrouped repetitions were 981 and 951 decisions/s. These small differences do not demonstrate a robust whole-update improvement.

## Scope and controls

Same Vast machine 20082, NVIDIA RTX 4090, AMD EPYC 7B13; measured cgroup CPU quota 30.6 cores. PyTorch 2.11.0+cu128, CUDA 12.8. Same engine binary, checkpoint update 844 (SHA256 b1c5503743fe69d7d997612ffeab139cb9d5ef2ecb55e02ee2c76f259e112bd7), width 128, four layers, eight heads, auxiliary response mode, FP32 and TF32 disabled. Origin/main base 8e197db415c87e009d34bb82e63edcef169e4b86 against frozen uncommitted source, after the packing/test fixes described below.

Both sources use the checkpoint's causal SDPA, rollout KV cache, batched attention, wide projection, private CUDA graphs, Triton KV transfer, learner batched attention and batched snapshot-policy scoring. Paged cache, merged snapshot encoding and page-span padding are off; CUDA cache trimming is `auto`. Only learner length groups (0/4) and the new chosen-response flag differ between the measured arms.

The single-rank short-PPO cases use 64 environments, 64 steps/update, four matches/minibatch, two PPO epochs and the checkpoint's current/historical Transformer population. Every process resumes the same checkpoint; runs execute old/default/chosen followed by chosen/default/old, for 0 and 4 length groups. Twelve updates/process, first four excluded: 144 total updates, 96 measured updates. Growing histories are retained; ordinary CUDA trajectories can diverge. This does not establish four-rank production speed or long-run learning/strength equivalence.

The learner probe uses the same frozen 2,047 completed rows from four matches, mean public prefix 641 and maximum 1,431, with fixed parameters and no Adam steps. It includes host batch preparation, forward loss and backward, CUDA synchronization, four warmups and 24 measured repeats/process; two processes per arm. Default grouped learner rate +4.2%, chosen-only +9.9% versus the grouped baseline; chosen-only +5.4% relative to new default. Ungrouped learner rates show no gain. Per-process variation limits these estimates.

| Length groups | Baseline learner rows/s | New default | New chosen-only |
|---|---:|---:|---:|
| 0 | 33,718 | 33,191 | 33,356 |
| 4 | 19,764 | 20,601 | 21,722 |

Rates summarize two process trials per cell; raw timings and both process rates are retained in `speed-summary.json`. Parity uses strict deterministic algorithms; speed measurements use the ordinary production CUDA backend.

## Correctness and fixes found by CUDA validation

80 focused tests passed, including all eight new CUDA cases. Four cross-source frozen-batch comparisons (0/4 groups, two repeats) matched baseline losses and all actor gradients bit for bit for the default path. Chosen-only response gradients stayed within tolerance; maximum absolute difference 4.6566128730773926e-9.

Two necessary corrections were made without changing PPO, reward, GAE, canonical candidates, FP32 or public-history semantics:

- Require strict deterministic backward for parity tests and restore both prior deterministic and warn-only flags. Warning-only CUDA attention backward is nondeterministic.
- Append host-planned integer ranks after existing packed float fields. Inserting them before those fields shifts CUDA vector alignment by eight bytes for odd row counts and changed default loss/gradient bits on the large fixture. Keeping the old offsets restored bitwise parity; an odd-row packed-alignment regression test was added.

Initial failing runs and their logs remain preserved under attempt-1 and attempt-2.

## Full regression

The first complete run reported 1,437 passed, 16 skipped, 18 failed. Fifteen failures were missing training-config fixtures or TensorBoard in the remote test payload, rather than source regressions. After independently packaging the 17 config files and installing TensorBoard, the complete suite on verification machine 33405 reported 1,451 passed, 16 skipped, 4 failed.

Three failures reproduced on origin/main on the same verification GPU, with the same error in `train/history_transfers.py`: a size-one, zero-stride Long tensor cannot be reinterpreted as Byte. They affect merged snapshot encoding in the opt-in paged-cache path, which the timing matrix does not enable. The fourth failure was a small FP32 log-probability equality mismatch in the plain-versus-paged CUDA collector test; it passed in the initial complete run. A final four-process diagnostic on the original benchmark machine ran the complete paged-cache test file in baseline/current/baseline/current order. Results were 16 passed / 3 failed, 16 / 3, 15 / 4, and 16 / 3. The baseline repeat reproduced the fourth, intermittent FP32 log-probability mismatch as well. Thus all four full-suite failure locations reproduce on baseline; no new regression is established by these failures. The full GPU regression suite is still not clean.

## Lifecycle and evidence

All five instances were destroyed with provider absence confirmed. An independent final API read returned **zero instances and US$0/hour**. Combined rental time was 40.5 minutes. Compute/disk estimated from elapsed time and the quoted hourly rates totaled **US$0.526**; observed account credit decrease was **US$0.768**, including provider-posted charges beyond that estimate. The latter is an account readback, not a final invoice; delayed billing may still settle. Both remain comfortably within the US$3 ceiling.

Download verification passed with zero SHA256/size mismatches: 18 files in attempt 1, 37 in attempt 2, 141 in the benchmark, 22 in full-suite verification and 26 in the paged-cache diagnostic. Independent local destroy guards and pod-side stop guards bounded every rental. No flags were promoted and no commit was made.

Raw evidence directories (relative to repository root):

- `.work/learner-gpu-2026-09-30/download/results/`: timing matrix, `speed-summary.json`, four frozen-batch parity reports and focused/full test logs.
- `.work/learner-gpu-2026-09-30/attempt-1/` and `attempt-2/`: preserved initial failure evidence and teardown receipts.
- `.work/learner-gpu-verification-2026-09-30/download/results/`: complete fixture/dependency verification and same-GPU baseline failures.
- `.work/learner-gpu-paged-check-2026-09-30/download/results/`: alternating baseline/current paged-cache diagnostic.
- Each kit contains `kit-manifest.json`, `artifact-verification.json`, and `teardown-summary.json`.

Frozen code digests: baseline `a1f704e18f2feb839d3b3b1c07410c1636b3a5293ad04ccdd5b440dfa9f0f8d7`; changed `a66fee6e5936a1b385c17695d18b6f4024b8e79bbace0e44964fc51ec3747a34`. The tested source files still match the changed digest; documentation edits are outside the code manifest.
