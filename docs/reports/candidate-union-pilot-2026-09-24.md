# Stage C candidate-union and advantage-filter pilot — September 24, 2026

The top-32 plus eight uniformly sampled candidates did **not** establish a playing-strength gain over the top-32 control. Its head-to-head duplicate result was +0.0020 net levels per round, 95% paired bootstrap CI [−0.0353, +0.0411], on 4,000 deals. It won 490 of 1,000 full matches (49.0%, Wilson 95% CI [45.9%, 52.1%]). The advantage-filter arm clearly failed this internal comparison: −0.3284 [−0.3660, −0.2894] levels per round and 197/1,000 full-match wins (19.7%, Wilson CI [17.4%, 22.3%]). Neither experimental arm qualifies to replace the control under G7/G8. These are house-rules results, not an external DanLM or state-of-the-art claim.

## Source and protocol

Raw, complete reports: [filter evaluation](/Users/xiyaowang/Documents/Projects/GuanZero/.work/overnight-20260924/experiments-results/runs/filter/evaluation.json) and [union evaluation](/Users/xiyaowang/Documents/Projects/GuanZero/.work/overnight-20260924/experiments-results/runs/union/evaluation.json). Both report `status=complete`; their `EVAL_EXIT` markers are 0. Each retains all per-deal `pair_scores` and `results`, and 1,000 full-match records. The paired suite uses the same 4,000 generated deals, seed 2026092451, batched evaluator (256 deals per wave, two engine threads), CUDA inference, greedy, the four fixed styles, and three fixed vectors sampled from `train.styles.StyleSpace`'s heldout region. Both reports have the same control checkpoint model digest `c320e541…83782c4`; its raw suite pair scores and results, as well as all three heldout vectors, match exactly across the two files. Candidate model digests are `bf7dae9c…d8bd8e` (filter) and `457849f7…005b3569` (union). The reports contain full file SHA-256 values.

All arms started from `artifacts/b11-main.pt` with seed 2026092431 and ran for 4,500 seconds on the same GPU budget. Control used top-32 and no filter. Filter changed only `advantage_filter_quantile=0.8` and `advantage_filter_min_magnitude=0.05`. Union changed only `candidate_mode=union` and `candidate_extra=8`; `top_k` stayed 32. The frozen comparison configs are in `/Users/xiyaowang/Documents/Projects/GuanZero/.work/overnight-20260924/experiments-results/cfg/`.

## Paired opponent suite

Entries are candidate minus the shared control in net levels per round, paired by deal; 95% percentile bootstrap intervals resample whole deals. Positive favours the experimental arm.

| Opponent | Filter Δ [95% CI] | Union Δ [95% CI] |
|---|---:|---:|
| Greedy | −0.2566 [−0.2946, −0.2155] | −0.0014 [−0.0378, +0.0341] |
| Bomb-happy | −0.2714 [−0.3198, −0.2254] | −0.0056 [−0.0510, +0.0379] |
| Bomb-shy | −0.2654 [−0.3021, −0.2292] | +0.0069 [−0.0261, +0.0419] |
| High-lead | −0.0480 [−0.0683, −0.0275] | −0.0025 [−0.0201, +0.0165] |
| Low-lead | −0.2532 [−0.2833, −0.2226] | −0.0145 [−0.0389, +0.0100] |
| Heldout 0 | −0.0609 [−0.0738, −0.0479] | +0.0119 [+0.0033, +0.0210] |
| Heldout 1 | −0.0181 [−0.0311, −0.0059] | +0.0049 [−0.0065, +0.0165] |
| Heldout 2 | −0.0290 [−0.0424, −0.0164] | +0.0054 [−0.0058, +0.0163] |

The filter arm's eight paired intervals are all below zero, so its loss is broad across this suite. Union is near even in seven cells. Heldout 0 is a small positive result with an interval above zero, but it is one of eight inspected opponent cells; it does not overcome the head-to-head interval crossing zero or establish a general heldout-style improvement. Heldout 1 and 2 also trend positive but their intervals cross zero.

## Training and interpretation boundaries

The comparison matches **wall time**, not training volume. The final logs record control 1,153 updates / 151.10 million decisions; filter 1,137 / 149.01 million; union 1,088 / 142.61 million. Thus union processed about 5.6% fewer decisions than control while using the wider support, and these results cannot isolate support quality from the throughput cost at equal time. Equal time is the specified practical budget comparison, while a separate equal-decision study would answer a different question. The filter arm processed about 1.4% fewer decisions; that smaller difference does not explain away its large and consistent playing loss, but the experiment does not identify the exact failure mechanism.

The duplicate metric swaps policy assignments across the same deal and divides the pair difference by two. The full-match win rate alternates team assignment over independent match seeds; it is a separate outcome, and its signed mean net level per round should not be compared numerically with duplicate levels per round. Batched sampling uses its own fixed draw order, so the reproducibility boundary is this seed and batch configuration. The heldout vectors test a reserved style region, not an exhaustive opponent distribution. Further strength claims require the revised G3 heldout guard and repeated matched runs; this pilot supplies no external baseline result.
