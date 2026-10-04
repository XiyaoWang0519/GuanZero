# Strength against DanLM, summary (October 4, 2026)

This collects the DanLM numbers behind the README chart
(`docs/assets/progress-vs-danlm.svg`) in one place. Nothing here is new
measurement; every row comes from an evaluation file listed below.

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
| u15094 (auxiliary heads, main lineage) | 989.2M | -1.558 | [-1.594, -1.521] |

Decision counts marked ≈ are interpolated at the measured 65.6k decisions per
update between the recorded endpoints (u2623, u4902, u9989). u15094 is the
"full" arm of [aux-heads-2026-10-02](aux-heads-2026-10-02.md), forked from
u9989; its two sibling arms scored -1.635 (no heads) and -1.649 (next-token
head only).

Reference: B11, the strongest MLP agent, -2.033 [-2.065, -2.000]. DanZero V1T,
DanLM's own reproduction of the earlier DanZero MLP agent, scores -0.483 on
990 deals ([stage-c-danlm](stage-c-danlm.md)).

## On top of u15094

| Variant | Deals | vs DanLM | Note |
|---|---|---|---|
| Plain policy | 300 | -1.640 [-1.772, -1.505] | same 300 deals as the search row |
| Test-time search | 300 | -1.342 [-1.483, -1.200] | paired gain +0.298 [+0.163, +0.443], 209 deals changed |
| Uniform average of the last 4 checkpoints | 4,000 | -1.505 [-1.542, -1.467] | no extra training; +0.098 [+0.043, +0.154] vs u15094 head to head, 2,000 paired deals |

Search configuration: triggered when any seat holds 10 or fewer cards or the
policy's top move has probability below 0.6; top 8 moves, 32 uniformly sampled
hidden-hand worlds, 10 s budget, rollouts by the policy itself in all seats.
12,003 searches over 300 deals, 1,397 overrides, about 123 s per deal on an
Apple M4 Pro.

## Sources (local, under `.work/`, not committed)

- Curve: `overnight-eval-2026-10-02/danlm/*.json`,
  `longrun-u2623-danlm-2026-10-01/{u2623,u4902,b11}.json`
- Search: `search-danlm-2026-10-03/run1/summary.json`
- Weight average: `weight-avg-2026-10-03/avg4-vs-danlm.json`,
  `weight-avg-2026-10-03/avg4-vs-u15094.json`
