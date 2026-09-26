# Batch-size recipe test: bounded RTX 4090 run, September 26, 2026 UTC

One approved RTX 4090 Secure allocation ran three fresh random-start history
Transformer arms for 300 PPO updates each, then a local CPU evaluator scored
every 10-update snapshot of every arm on the frozen 64 development deals. The
question was whether the T4 stall near −2.3 net levels/round was caused by the
tiny per-update batch (about 300 learner rows). **It was not.** A four-times
larger batch reaches the plateau faster per update and ends at the same level
per learner row; all arms plateau between −2.1 and −2.4 with entropy collapsed
below 0.5 and zero full-match wins. This is a development-curve diagnostic on
one training seed per arm, not a playing-strength result or a promotion.

## Design and identity

| Arm | Path | Environments | Learner rows/update | Updates | Trainer seconds | Learner rows | Peak allocated / reserved GB |
|---|---|---:|---:|---:|---:|---:|---:|
| A | KV cache + causal SDPA | 32 | about 1,250 | 300 | 990 | 377,250 | 0.29 / 0.57 |
| B | KV cache + causal SDPA | 8 | about 315 | 300 | 350 | 94,646 | 0.13 / 0.20 |
| C | dense reference | 8 | about 315 | 300 | 440 | 95,203 | 1.90 / 24.71 |

Everything else matched T4: width 64, two layers, four heads, 64 vector steps
per update, one PPO epoch, two matches per minibatch, learning rate 0.0003,
entropy coefficient 0.01, clip 0.2, gamma 1, GAE lambda 0.95, snapshots every
two updates, four recent eligible, snapshot probability 0.5, FP32 with TF32
disabled. All arms used seed `2026092603` (T4 used `2026092602`). No MLP,
teacher, old optimizer or replay touched the pod. Arm A's environment count
was selected on the pod by a predeclared KV sweep (16 and 32 both passed the
70% reserved-memory and 120-second update gates; 32 reserved 1% of the card).

| Identity | Value |
|---|---|
| Git base | `9e70391c5bfa664bdf003bde75c2e174313c6dd7` plus uncommitted history source |
| Source SHA-256 | `e86a9761f3db74acc9fc8e959107a9b71b3c8cb844cdcf14cab79f36aaae3008` (identical to the stack comparison) |
| Archive SHA-256 | `e24dc47edf4c46b8eb29b23433d64bae5bd80bba722ef9941d9b40cb50d04d3d` |
| Manifest SHA-256 | `a6f8fe58d6f103ede0fd39d5646409b3787f4f127b0f4c3a438877e3e7a81e15` |
| Engine digest | `606ff1e735823cefb9968a974991bfc5ca325c2ef1cb8ea5a5540f174ce84096` |
| Host | AMD EPYC 7452, usable CPU budget 10; RTX 4090; PyTorch 2.9.1+cu128 |
| Final checkpoints | A `7733e363…b2f7`, B `01160e2f…1940`, C `b24fa86b…fce7` (full hashes in the kit) |

The 98-test CUDA gate passed on the pod before any training. All 900 metric
rows are finite with positive actor/encoder/critic gradient norms; maximum
approximate KL 0.0148 (A), clipping fraction at most 0.115. Maximum prefix
2,863 tokens. Dense arm C again reserved 24.7 GB of allocator memory for
1.9 GB allocated; both KV arms stayed under 0.6 GB reserved.

## Development curves

Candidate net levels per round on the same 64 deals, two swapped legs each,
deterministic argmax, against frozen B11 main and the long-run segment-2
endpoint. Every 10-update snapshot was evaluated; the table shows a subset.
Raw results: `results/curves/` and `results/curves.json` in the kit.

| Update | A vs B11 | A vs long-run | A entropy | B vs B11 | B entropy | C vs B11 | C entropy | T4 vs B11 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | −2.992 | −3.000 | | −2.992 | | −2.992 | | −2.977 |
| 30 | −2.273 | −2.289 | 1.27 | −2.992 | 1.48 | −2.961 | 1.46 | −2.922 |
| 50 | −2.219 | −2.500 | 0.94 | −2.672 | 1.51 | −2.984 | 1.51 | −2.688 |
| 70 | −2.047 | −2.266 | 0.69 | −2.984 | 1.56 | −2.844 | 1.59 | −2.828 |
| 100 | −2.188 | −2.250 | 0.53 | −2.641 | 1.30 | −2.812 | 1.29 | −2.477 |
| 150 | −2.141 | −2.312 | 0.35 | −2.297 | 0.55 | −2.375 | 0.80 | −2.359 |
| 200 | −2.078 | −2.023 | 0.24 | −2.062 | 0.47 | −2.094 | 0.57 | −2.328 |
| 250 | −2.188 | −2.062 | 0.32 | −2.266 | 0.31 | −2.258 | 0.50 | |
| 300 | −2.250 | −2.031 | 0.31 | −2.328 | 0.30 | −2.086 | 0.48 | |

Pooled late-plateau comparison (updates 200–300, 11 snapshots per arm,
per-deal pair scores averaged across snapshots, paired whole-deal bootstrap):

| Comparison | vs B11 | vs long-run endpoint |
|---|---:|---:|
| A mean / B mean / C mean | −2.08 / −2.20 / −2.20 | −2.12 / −2.40 / −2.31 |
| A − B | +0.12 [−0.06, +0.30] | +0.28 [+0.09, +0.49] |
| B − C (KV/SDPA vs dense) | 0.00 [−0.15, +0.15] | −0.09 [−0.26, +0.09] |

At matched learner rows the arms coincide: A at update 30 (48k rows) scores
−2.27 and B at update 140 (48k rows) scores −2.31; A at update 70 (98k rows)
scores −2.05 and B at update 300 (95k rows) scores −2.33. A then spends
another 280k rows without leaving the −2.0 to −2.3 band. Full-match wins were
0/16 against each baseline for every evaluated snapshot of every arm.

Seed replicate: over updates 110–200, B and C score +0.10 [−0.03, +0.24] and
+0.09 [−0.05, +0.24] above the T4 pilot on the same deals, so the T4 curve was
not a bad-seed artifact.

## Reading

- **Batch size is not the cause of the plateau.** Four times the samples per
  update removed T4's flat first 60 updates and reached the plateau by update
  50–70, but the plateau itself moved by at most 0.1–0.3 levels and stayed far
  from parity. Per learner row the curves are indistinguishable.
- **Entropy collapse happened in every arm**, to 0.31 (A), 0.30 (B) and 0.48
  (C), and in A it happened on the larger batch, so it is not gradient noise
  from small batches. The remaining suspects are the fixed 0.01 entropy
  coefficient and exploration schedule, the width-64 two-layer capacity, the
  learning rate, and self-play population dynamics; this run does not
  separate them.
- **KV cache plus causal SDPA versus dense showed no detectable strength
  difference** at this scale on one seed (B − C within ±0.15). This is the
  first same-recipe development-curve comparison; it supports, but does not
  by itself settle, the stack report's pending strength gate.
- The three-point T4 evaluation could not show curve shape; the every-10-update
  local CPU evaluation (about 30 seconds per snapshot) can, and should stay the
  default for future bounded runs.

## Budget, artifacts and closure

Approved in chat by the user (statement recorded in `approval.json`) within the
standing scope: one RTX 4090 Secure, $1.20 cap, $0.80/hour including disk,
90 minutes. Pod `u91zka6cdmo1wt` was created at 05:07:58 UTC and independently
confirmed gone after 2,121.9 seconds (35.4 minutes). At the quoted
$0.744464/hour the estimate is **$0.4388**; the account balance moved from
$32.7518 to $32.3292, a $0.4226 observed difference. Billing line items were
not queried. The monitor SHA-256 verified **153 files before teardown**; an
independent readback found zero pods and $0 current hourly spend.

Kit: `.work/runpod-history-batch-2026-09-26/`. Receipts: `approval.json`,
`run-manifest.json`, `pod.json`, `progress.jsonl`, `verified-artifacts.json`,
`teardown.json`, `independent-provider-readback.json`, per-arm
`local/results/arm-{A,B,C}/` (metrics, population audit, every 10-update
checkpoint), `results/curves.json`, `results/analysis.json`, and the
re-runnable `evaluate_curves.sh`.

No default, checkpoint or document promotion follows from this run. Before
any further rental, decide which of the entropy/exploration, capacity or
learning-rate hypotheses to test and set a new explicit budget for it.
