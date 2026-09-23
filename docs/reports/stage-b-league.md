# Stage B B8: league training versus a single frozen opponent

Status: complete, September 23, 2026. **Original G3: not passed** (one clause
fails, see Reading 1). **G3 as revised on September 23 by the owner: passed**
(see the section below). The league checkpoint is nonetheless the
strongest model so far by head-to-head play.

## Setup

One RTX 4090 pod (secure cloud, 32 vCPU, 125 GB), 6.45 hours including setup
and download, observed balance change $4.88. The community RTX 5090 was
unavailable and the community 4090 had no 32-vCPU host. Both arms ran at the
same time on the same host for 21,600 s, warm-started from the B6 ppo-a
checkpoint (policy, critic, optimizers), with the M1 final as the frozen
pruning reference, lr 1e-5, T 0.02, KL to M1 off, in-process rollout.
First training run under the official O1 tribute tie and without the O4
three-failure reset.

| Arm | Opponent | Updates | Environment decisions |
|---|---|---:|---:|
| league | `train/configs/league-b8.json`: M1, B6 DMC, greedy, 4 fixed styled bots, sampled styles, learner snapshots every 100 updates (argmax and T=0.02) | 5,220 | 684 M |
| frozen | the M1 final only | 8,266 | 1,083 M |

The league arm is slower per second because its network opponents need their
own forwards; equal wall clock is the compute budget.

## Head to head and against earlier models

Fresh duplicate deals, seed 20260923; mean net levels per round for the first
policy, 95% bootstrap interval over deals; matches with Wilson intervals.

| Pair | Deals | Mean [95% CI] | Matches won |
|---|---:|---|---|
| **league vs frozen** | 10,000 | **+0.087** [+0.061, +0.110] | 552/1000 [0.521, 0.583] |
| league vs ppo-a | 5,000 | +0.524 [+0.490, +0.559] | 398/500 |
| frozen vs ppo-a | 5,000 | +0.538 [+0.503, +0.575] | 408/500 |
| league vs M1 | 5,000 | +1.128 [+1.093, +1.160] | 482/500 |
| frozen vs M1 | 5,000 | +1.303 [+1.268, +1.336] | 490/500 |
| league vs B6 DMC | 5,000 | +0.681 [+0.644, +0.718] | 438/500 |
| frozen vs B6 DMC | 5,000 | +0.787 [+0.749, +0.823] | 456/500 |

Against opponents neither arm trained on specifically (2,000 deals):

| Model | vs bomb-happy | vs greedy |
|---|---|---|
| ppo-a (start) | +1.300 | +1.818 |
| frozen | +1.617 [+1.564, +1.670] | +1.952 [+1.902, +1.997] |
| league | +1.681 [+1.630, +1.734] | **+2.178** [+2.135, +2.223] |

## Rating

Weighted least-squares fit of every round-robin margin to r_a − r_b
(7 players, 21 pairs, ratings sum to 0, levels per round; fit RMSE 0.062):

| Model | Rating |
|---|---:|
| league (final) | +0.391 |
| frozen (final) | +0.372 |
| league at update 2,600 | +0.297 |
| frozen at update 4,100 | +0.283 |
| ppo-a | −0.128 |
| B6 DMC | −0.365 |
| M1 final | −0.850 |

## Reading

1. **G3 as written is not passed.** Its first clause asks the league
   checkpoint to be no worse than the single-opponent checkpoint against the
   M1 final; it is worse there (+1.13 vs +1.30). Its second clause, better
   against the styled bots and greedy, holds for greedy and is within noise
   for bomb-happy (only one styled bot was scored). The gate is reported as
   drafted, not redefined after the fact.
2. **The league checkpoint is still the stronger player.** It beats the
   frozen arm head to head with both intervals clear of even, and it is
   clearly better against greedy. The frozen arm's extra margin is
   concentrated on M1 and its direct descendants (DMC, ppo-a), the family it
   trained against; its KL from M1 grew to about 5 during training. That is
   the specialization a league is meant to prevent. M1 is therefore a poor
   yardstick from here on, as expected once models pass it by a wide margin.
3. **Both arms were still improving.** Each final checkpoint beats its own
   midpoint by about +0.10 levels per round, so six hours did not reach a
   plateau.
4. **The ranking is almost transitive** (fit RMSE 0.06 levels per round), so
   the rating table is a fair summary; the league/frozen gap in the rating is
   small (+0.02) while the direct match is +0.09.
5. **G3 should be restated** for future runs around head-to-head play and a
   broader held-out opponent set rather than M1. Proposed: the league
   checkpoint beats the single-opponent checkpoint head to head with a 95%
   interval above zero, and is not worse against a held-out suite (greedy,
   all fixed styled bots, held-out sampled styles).

## Revised G3

The owner revised G3 on September 23 (tracker, Gates). Paired per-deal
difference league minus frozen, same 4,000 deals per bot, seed 20260924:

| Held-out opponent | league | frozen | Difference [95% CI] |
|---|---:|---:|---|
| greedy | +2.171 | +2.003 | +0.168 [+0.128, +0.207] |
| bomb-happy | +1.654 | +1.595 | +0.059 [+0.012, +0.105] |
| bomb-shy | +2.292 | +2.132 | +0.160 [+0.124, +0.197] |
| high-lead | +2.868 | +2.810 | +0.058 [+0.036, +0.079] |
| low-lead | +2.631 | +2.459 | +0.172 [+0.146, +0.201] |

With the head-to-head win, the revised G3 passes: the league checkpoint is
better than the single-opponent checkpoint on every held-out opponent.

## Cross-play (partner compatibility)

The league final with a different partner, against two M1 finals, 5,000
deals (`stage-b-crossplay-b8-league.json`). "Synergy" is the mixed team
minus the midpoint of the two self-paired teams, a descriptive heuristic.

| Partner | Mixed team | Partner self-paired | Synergy |
|---|---:|---:|---:|
| league itself | +1.128 | same | 0 |
| frozen final | +1.200 | +1.303 | -0.016 |
| ppo-a | +0.960 | +0.773 | +0.009 |
| M1 | +0.601 | 0.000 | +0.037 |
| bomb-happy | +0.253 | -0.760 | +0.069 |
| greedy | +0.009 | -1.510 | +0.200 |

Every partner lands at or above the midpoint, as with ppo-a in B6: no sign of
private conventions that break with other partners. The league model lifts
weak partners more than ppo-a did (greedy +0.009 vs -0.203).

## Limitations

- One seed per arm. Throughput differs by arm (league about 32k, frozen about
  52k decisions per second); equal wall clock is the budget.
- Only bomb-happy among the styled bots was scored against both finals.
- All opponents are internal; the external benchmark gate stays unavailable.

Artifacts: `.work/runpod-b8/` (archive SHA-256 `28c64a40…85b3b2`, all
checkpoints and league snapshots, metrics, manifest with confirmed deletion).
