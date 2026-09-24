# B11 main continuation: internal evaluation, September 24, 2026

The five-hour continuation beat its B11 starting checkpoint in the internal house-rules arena. On 4,000 swapped duplicate deals, it gained **+0.1386 net levels per round**, 95% paired bootstrap CI **[+0.0979, +0.1779]**. It won **594 of 1,000 full matches** (59.4%; Wilson 95% CI [56.3%, 62.4%]). This supports a real internal improvement over that frozen B11 baseline at this seed and evaluation setup. It does not establish strength against DanLM or another external opponent.

## Identity and completeness

The complete raw report is [main evaluation](/Users/xiyaowang/Documents/Projects/GuanZero/.work/overnight-20260924/main-results/runs/main/evaluation.json), with `status=complete` and `EVAL_EXIT=0`. It contains 4,000 duplicate `pair_scores` and results, 1,000 full-match records, and 4,000 paired scores and differences in each of eight opponent cells. The candidate model digest is `2cc50f0499125cf05e81d32154d3962372c3ca9378b10cdd9c8a85fb7dc92a99`; its checkpoint file SHA-256 is `ebb614f35f825e835c9d5b18bdc5a6efb0f79d5ba5ed9bfcdf44eca98ae139fe`. The control model digest is `8e872130189af48adb262a0b218bcc82e711b31f5804272c1dd50854b629bc3e`, and its file SHA-256 `25e9e0bf549957b9d8b8aeb426b7225259cca714ec23d77de8eeda45bef063df` matches the saved `b11-main.pt` artifact manifest. Both policies are PPO checkpoints.

The runner used seed 2026092451 for batched CUDA evaluation (batch size 256, two engine threads, two PyTorch threads). It evaluated the same 4,000 generated deals against greedy, four fixed styles, and three fixed heldout-region style vectors. The training log ends at 4,450 updates, 583.16 million decisions and 18,000.28 seconds, starting from the B11 checkpoint. These are local report and log observations; checkpoint execution and provider state were handled separately.

## Paired opponent suite

Each entry is continuation minus B11 baseline net levels per round on the same deals, with a 95% percentile bootstrap interval over whole deals.

| Opponent | Paired Δ [95% CI] |
|---|---:|
| Greedy | +0.0416 [+0.0059, +0.0785] |
| Bomb-happy | +0.0688 [+0.0254, +0.1133] |
| Bomb-shy | +0.0373 [+0.0031, +0.0710] |
| High-lead | +0.0171 [−0.0014, +0.0339] |
| Low-lead | +0.0049 [−0.0198, +0.0305] |
| Heldout 0 | −0.0053 [−0.0135, +0.0021] |
| Heldout 1 | +0.0040 [−0.0073, +0.0160] |
| Heldout 2 | +0.0009 [−0.0094, +0.0118] |

Greedy, bomb-happy and bomb-shy have positive paired intervals. The other five intervals cross zero. In particular, heldout 0's small negative point estimate is inconclusive; there is no measured heldout cell with an interval entirely below zero, but these data do not prove noninferiority for every style. The head-to-head duplicate metric swaps policy assignment across the same deal; the full-match result alternates team assignment across independent match seeds. Their numeric level-return scales differ, so the duplicate +0.1386 and full-match mean signed return +0.1866 should be read as separate measures.

This is one internal evaluation of one continuation checkpoint. Its positive head-to-head interval and absence of a clear heldout regression support continuing with this checkpoint, subject to the project's revised G3 protocol and separate external calibration. The report makes no claim of state-of-the-art play.
