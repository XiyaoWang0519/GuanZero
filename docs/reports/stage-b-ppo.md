# Stage B B6: PPO versus continued DMC at equal compute

Status: complete, September 22, 2026. **Gate G2 passed.**

## Setup

One rented RTX 5090 pod (64 vCPU, 124 GB), secure cloud, 1.19 hours
including setup and download, observed balance change $1.10. Three arms ran
at the same time on the same host for 3,600 s of training each, all started
from the M1 final (`8a8e2b08dec2e739`, 34,496 updates).

| Arm | Method | Config | Updates | Environment decisions |
|---|---|---|---:|---:|
| dmc | DMC continued with M1's exact saved config | `num_envs` 4096 | 34,496 → 89,223 | 112.2 M new |
| ppo-a | PPO, pre-registered primary arm | policy lr 1e-5, T 0.02, top-k 32, KL to M1 annealed over 300 updates | 780 | 102.2 M (51.5 M learner) |
| ppo-b | PPO, exploratory | as ppo-a with policy lr 3e-5, seed 8 | 782 | 102.5 M |

PPO trained against a frozen M1 on the other team. The critic started from
the B2 perfect-information fit. DMC saw about 10% more environment
decisions than either PPO arm in the same wall-clock hour.

## Results

Fresh duplicate deals, seed 20260922, on local CPU. Mean net levels per round
for the first-named policy, with 95% bootstrap intervals over deals; matches
use Wilson intervals.

| Pair | Deals | Mean [95% CI] | Matches won [95% CI] |
|---|---:|---|---|
| ppo-a vs M1 | 5,000 | **+0.778** [+0.744, +0.811] | 456/500 [0.884, 0.934] |
| ppo-b vs M1 | 5,000 | +0.765 [+0.731, +0.800] | 445/500 [0.860, 0.915] |
| dmc vs M1 | 5,000 | +0.520 [+0.485, +0.553] | 360/500 [0.679, 0.758] |
| **ppo-a vs dmc** | 5,000 | **+0.256** [+0.219, +0.292] | 318/500 [0.593, 0.677] |
| ppo-b vs dmc | 5,000 | +0.251 [+0.216, +0.287] | not played |

Against opponents none of the arms trained against (2,000 deals each):

| Arm | vs bomb-happy | vs greedy |
|---|---|---|
| M1 final (B0 baseline) | +0.782 | +1.516 |
| dmc | +1.193 | +1.742 |
| ppo-a | **+1.285** | **+1.818** |
| ppo-b | +1.253 | +1.750 |

## Reading

1. **G2 passes.** ppo-a beats the M1 final with the duplicate interval above
   zero and the match Wilson interval above 50%, and beats DMC continued
   from the same checkpoint for the same hour on the same host, head to head.
2. **M1 was undertrained.** One more hour of DMC alone is worth +0.52 levels
   per round against M1. Part of PPO's gain over M1 is therefore simply more
   training; the PPO-versus-DMC row isolates the method effect, +0.26.
3. **The gain is not only exploitation of the training opponent.** ppo-a
   trained only against M1, yet it also beats DMC, which it never saw, and
   improves most over DMC against the bomb-happy bot.
4. **Learning rate.** The two PPO arms are within noise of each other; ppo-b
   drifted further from M1 (KL 2.3 vs 1.6) and its entropy rose from 0.32 to
   0.40 while its training win rate stalled after update 600. Use 1e-5.

## Limitations

- One seed per arm, one hour. No claim about asymptotic strength.
- PPO trained against a single fixed opponent; league training (B8) and the
  exploiter test (B9) are still needed for robustness.
- All opponents are internal. The external DanZero/SDMC gates remain
  unavailable.
- Throughput was CPU-loop bound (GPU 44-73%, ~27k decisions/s per arm on a
  64-core host); see the B5b throughput row.

Artifacts: `.work/runpod-b6/` (results archive SHA-256
`8d9664fe…6f16c2`, checkpoints, metrics, pod manifest with confirmed
deletion). Raw evaluation numbers in `stage-b-ppo.json`.
