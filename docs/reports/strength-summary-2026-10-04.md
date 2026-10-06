# Strength against DanLM, summary (October 4, 2026, updated with u20264, the October 5 search curve, m1 and m2 on October 6)

This collects the DanLM numbers behind the README chart
(`docs/assets/progress-m2.svg`) in one place. Every row comes from an
evaluation file listed below.

Yardstick: DanLM (`dansformer_v1_best_eval.pt`), duplicate deals through
`eval.danlm.arena`, seed 20260929, house rules, tribute fraction 0.5. Score is
net levels per round from our side; 0 would be equal strength. Intervals are 95%.

## Training curve, 4,000 deals per point

| Checkpoint | Self-play decisions | vs DanLM | 95% interval |
|---|---|---|---|
| u2623 | 171.9M | -2.045 | [-2.077, -2.012] |
| u3100 | ≈203M | -2.009 | [-2.042, -1.976] |
| u3534 | ≈232M | -2.006 | [-2.037, -1.974] |
| u3937 | ≈258M | -1.992 | [-2.024, -1.960] |
| u4309 | ≈283M | -1.989 | [-2.019, -1.955] |
| u4650 | ≈305M | -1.962 | [-1.996, -1.928] |
| u4902 | 321.3M | -1.957 | [-1.989, -1.924] |
| u6572 | ≈431M | -1.905 | [-1.939, -1.870] |
| u8060 | ≈528M | -1.791 | [-1.827, -1.756] |
| u9989 | 654.6M | -1.729 | [-1.765, -1.694] |
| u15094 (auxiliary heads) | 989.2M | -1.558 | [-1.594, -1.521] |
| u20264 (main lineage) | 1,328.0M | -1.356 | [-1.393, -1.317] |
| u27900 (m1, same lineage) | 1,828.5M | -1.151 | [-1.189, -1.113] |
| u35588 (m2, same lineage) | 2,332.3M | -0.869 | [-0.905, -0.831] |

Decision counts marked ≈ are interpolated at the measured 65.6k decisions per
update between the recorded endpoints (u2623, u4902, u9989). u15094 is the
"full" arm of [aux-heads-2026-10-02](aux-heads-2026-10-02.md), forked from
u9989; its two sibling arms scored -1.635 (no heads) and -1.649 (next-token
head only). u20264 is the constant-rate arm of
[lr-decay-weight-avg-2026-10-04](lr-decay-weight-avg-2026-10-04.md), one night
on from u15094; its learning-rate-decay sibling scored -1.357.

Reference: B11, the strongest MLP agent, -2.033 [-2.065, -2.000]. DanZero V1T,
DanLM's own reproduction of the earlier DanZero MLP agent, scores -0.483 on
990 deals ([stage-c-danlm](stage-c-danlm.md)).

## On top of u15094

| Variant | Deals | vs DanLM | Note |
|---|---|---|---|
| Plain policy | 300 | -1.640 [-1.772, -1.505] | same 300 deals as the search row |
| Test-time search | 300 | -1.342 [-1.483, -1.200] | paired gain +0.298 [+0.163, +0.443], 209 deals changed; superseded by the October 5 rerun below (-1.333) |
| Uniform average of the last 4 checkpoints | 4,000 | -1.505 [-1.542, -1.467] | no extra training; +0.098 [+0.043, +0.154] vs u15094 head to head, 2,000 paired deals |

Search configuration: triggered when any seat holds 10 or fewer cards or the
policy's top move has probability below 0.6; top 8 moves, 32 uniformly sampled
hidden-hand worlds, 10 s budget, rollouts by the policy itself in all seats.
12,003 searches over 300 deals, 1,397 overrides, about 123 s per deal on an
Apple M4 Pro.

## On top of u20264

| Variant | Deals | vs DanLM | Note |
|---|---|---|---|
| Learning-rate decay arm (u20314) | 4,000 | -1.357 [-1.392, -1.320] | +0.052 [+0.000, +0.109] vs u20264 head to head |
| Uniform average of the last 4 checkpoints | 4,000 | -1.345 [-1.382, -1.311] | +0.037 [-0.015, +0.091] vs u20264 head to head |

Test-time search has not been run on u20264; it was run on m1 (u27900), see below.

## Sample efficiency against our own MLP

B11, the strongest MLP agent, sits at the end of a lineage of about 1.02B
self-play decisions (M1 71M, B6 +102M, B8 +684M, B11 +about 165M estimated from
B8's decisions per update). The Transformer reached the same level after 172M
decisions: u2623 scores -2.045 against the reference agent versus B11's -2.033,
and head to head against B11 +0.076 [-0.009, +0.162] on 1,000 duplicate deals
([history-ablation-2026-09-29](history-ablation-2026-09-29.md)). At 654.6M
decisions u9989 beats B11 by +0.668 [+0.504, +0.824] on 256 deals.

The search gain of +0.32 (pooled, October 5 curve) is converted to training
volume with the overall slope of the curve above, +0.066 levels per round per
100M decisions (about 480M).

## Resources compared with the reference agent

Read from the reference checkpoint `ckpts/DanLM_v1/dansformer_v1_best_eval.pt`:
4,003,073 model parameters (Q head 2,363,393; Transformer blocks 885,888; hand
MLP 660,096; RoPE tables 81,920 of these are fixed buffers),
`total_transitions` 2,019,950,592, `episode_count` 33,274,126,
`train_step_count` 246,560 at batch size 8,192, 15 actors, inference on
`cuda:3` (so at least four GPUs). Its training code is not published, so the
definition of an episode is inferred: 60.7 transitions per episode is close to
our 75.3 decisions per round, so an episode is taken to be one round.

GuanZero u15094: 1,357,953 policy parameters (the exported Botzone actor;
[botzone-bot-2026-10-03](botzone-bot-2026-10-03.md)), 989,200,384 decisions and
13,134,820 rounds (`metrics.jsonl` of the aux-full segment). u20264, same
architecture: 1,328,021,504 decisions and 17,557,927 rounds (`metrics.jsonl` of
the lr-main segment), 53% of the reference agent's rounds.


## Test-time search on u9989

Same configuration and the same 300 deals, on u9989 (654.6M decisions): plain
-1.717 [-1.848, -1.585], search -1.403 [-1.533, -1.273], paired gain +0.313
[+0.173, +0.455], 224 deals changed. Source:
`search-danlm-2026-10-02/run1/summary.json`. Superseded by the October 5 rerun
below (search -1.397, gain +0.320).

## Test-time search across the curve (October 5, 2026)

Same search configuration, seed and deals as above, rerun on six checkpoints
with KV-forked rollouts on CPU (`rollout_kv_cache`, commit ef484a5). This path
gives the same search decisions as the earlier full re-encode, about 7x faster;
the earlier runs hit the 10 s budget in about 6% of searches (157 on u15094,
178 on u9989), the rerun in none. The chart's search line is this table.

| Checkpoint | Decisions | Deals | Plain | Search | Paired gain |
|---|---|---|---|---|---|
| u2623 | 171.9M | 199 | -2.123 [-2.261, -1.982] | -1.711 [-1.864, -1.555] | +0.412 [+0.261, +0.568] |
| u4902 | 321.3M | 200 | -1.845 [-2.000, -1.677] | -1.640 [-1.792, -1.482] | +0.205 [+0.018, +0.390] |
| u6572 | 430.7M | 200 | -1.920 [-2.070, -1.770] | -1.577 [-1.738, -1.410] | +0.343 [+0.160, +0.530] |
| u8060 | 528.2M | 200 | -1.765 [-1.928, -1.595] | -1.492 [-1.658, -1.323] | +0.273 [+0.090, +0.463] |
| u9989 | 654.6M | 300 | -1.717 [-1.852, -1.577] | -1.397 [-1.533, -1.258] | +0.320 [+0.178, +0.467] |
| u15094 | 989.2M | 300 | -1.640 [-1.775, -1.513] | -1.333 [-1.475, -1.190] | +0.307 [+0.167, +0.452] |

Deal 174 is excluded on u2623: its plain arm failed the mirror check, as in
the earlier run. Pooled over the six checkpoints (inverse-variance weights) the
gain is +0.317 ± 0.066. Its change from 172M to 989M decisions is -0.05 ± 0.19
(weighted linear fit; chi-squared 3.2 on 5 degrees of freedom against a
constant), so this data cannot tell whether the gain grows, shrinks or stays
flat with model strength. About 46 s per deal per worker, ten CPU workers on
an Apple M4 Pro.

### m1 (u27900), October 6, 2026

m1 is the first 500M-decision milestone of the Max-Velocity run, continued
from u20264 (`.work/velocity-milestones/m1-u27900.pt`). 4,000 deals, same
yardstick and seed: -1.151 [-1.189, -1.113] (3,998 scored, 2 excluded for
mirror failure), round win rate 27.7% against about 24.1% for u20264. Search
on the first 200 deals with the same configuration as the curve above
(32 worlds, 10 s budget, KV-forked rollouts, CPU):

| Checkpoint | Decisions | Deals | Plain | Search | Paired gain |
|---|---|---|---|---|---|
| m1 (u27900) | 1,828.5M | 200 | -1.345 [-1.500, -1.190] | -1.0225 [-1.195, -0.855] | +0.3225 [+0.138, +0.500] |

The 200-deal plain score differs from the 4,000-deal score above because the
subset is small; the chart plots the 200-deal search value as it does for the
other search points. A 128-world, 40 s variant on the same deals gave a gain of
+0.2975 [+0.155, +0.438], no better than 32 worlds. Search overrode 784 of
17,489 calls at 32 worlds and 404 at 128. Sources:
`velocity-milestones/m1-vs-danlm.json`,
`search-m1-2026-10-06/{w32,w128}/summary.json`.

### m2 (u35588), October 6, 2026

m2 is the second 500M-decision milestone of the same run
(`.work/velocity-milestones/m2-u35588.pt`, 2,332.3M decisions). 4,000 deals,
same yardstick and seed: -0.869 [-0.905, -0.831] (3,997 scored, 3 excluded
for mirror failure), round win rate 32.8%; +0.28 over m1. Search, same
configuration (32 worlds), first 200 deals:

| Checkpoint | Decisions | Deals | Plain | Search | Paired gain |
|---|---|---|---|---|---|
| m2 (u35588) | 2,332.3M | 200 | -0.9525 [-1.105, -0.7925] | -0.770 [-0.945, -0.5975] | +0.1825 [+0.0125, +0.3525] |

760 overrides in 17,293 calls, no budget exhaustion, about 40 s per deal.
Pooled over all eight checkpoints (the six above, m1 and m2; inverse-variance
weights) the search gain is +0.301 ± 0.059. Its slope with training volume is
-0.05 ± 0.08 per 1B decisions (chi-squared 5.3 on 7 degrees of freedom
against a constant): m2's lower point cannot be distinguished from a constant
gain. Sources: `velocity-milestones/m2-vs-danlm.json`,
`search-m2-2026-10-06/w32/summary.json`; chart redrawn by
`velocity-milestones/make_progress_svg.py`.

## Sources (local, under `.work/`, not committed)

- Curve: `overnight-eval-2026-10-02/danlm/*.json`,
  `longrun-u2623-danlm-2026-10-01/{u2623,u4902,b11}.json`
- Search: `search-curve-kv-2026-10-05/<checkpoint>/summary.json` (chart);
  earlier `search-danlm-2026-10-03/run1/summary.json`,
  `search-danlm-2026-10-02/run1/summary.json`
- Weight average: `weight-avg-2026-10-03/avg4-vs-danlm.json`,
  `weight-avg-2026-10-03/avg4-vs-u15094.json`
- u20264 and its variants: `lr-decay-2026-10-03/eval/*.json`
