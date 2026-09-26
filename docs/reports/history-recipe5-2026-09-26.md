# Recipe run 5: continuing the best lineages across pods, September 26, 2026 UTC

Run 4's two-epoch arms with current-policy-only self-play (SP) and 3-rank data
parallel (DP) led the development curves, but their lead over the two-epoch
reference narrowed late. This run resumed SP, DP and the same-seed control L5
from their run-4 checkpoints for another 80 minutes on the same source, and
started one new combination (SPD). **Both recipe lineages hold a +0.5..+0.8
net levels/round lead over the continued control in every window up to update
1,500, but they level off near -1.0..-1.3 against B11 (about +1.3 against
greedy): a higher plateau, not continued climbing.** Full-match wins stay at
1-2 of 16 in isolated snapshots. Development result only; no strength claim.

## Design and identity

| Arm | Lineage / change | Updates this run | Learner rows this run (all ranks) |
|---|---|---:|---:|
| SPc | continues run-4 SP (two epochs, snapshot probability 0) | 834-1,697 | 1,699,840 |
| DPc | continues run-4 DP (two epochs, 3-rank data parallel) | 680-1,366 | 2,547,952 |
| L5c | continues run-4 L5 (original control) | 958-1,954 | 1,228,580 |
| SPD | new: two epochs + snapshot probability 0 + 2-rank data parallel, seed 2026092605 | 1-725 | 2,854,152 |

Resume used the full run-4 `latest.pt` files (uploaded as hashed payload
files; SHA-256 prefixes SPc `ad14735c`, DPc `a7d8ba6c`, L5c `9bc2ad5a`). Each
arm wrote `resumed-from.json`: the pod restored updates 833 / 679 / 957 with
lineages `history_ppo-2026092605-2847cba4`, `-18a732a1`, `-cf58b612`,
identical to run 4. Resume restores actor, critic, optimizers, RNG/sampler and
the snapshot population, and restarts environments (partial rounds discarded
by design). The trainer's source/engine/token identity check passed, because
the source was frozen after run 4 (`5b05d3ef…3854`). Before packaging, ten
iCloud conflict copies (`name 2.py`, older versions of existing files) had
appeared in `train/`, `eval/` and `tests/` and changed the source identity;
they were moved unchanged to `.work/icloud-duplicates-2026-09-26/`.

Pod `r30ec2896ztpuf` (RTX PRO 4500 Blackwell Secure, EU-RO-1, AMD EPYC 7713P,
quota 27.2 CPUs, $0.724/hour) created 12:23:07 UTC; archive `d2909678…`;
manifest `15d153e7…`; pod test gate 101 passed. Final slim checkpoints: SPc
`f6b5fc8f`, DPc `2793affc`, L5c `57af6e63`, SPD `5128ebde` (prefixes).

## Development curves (run 4 + run 5 joined per lineage)

Each cell: vs B11 / vs greedy on the frozen 64 deals.

| Update | L5 (control) | SP | DP |
|---:|---|---|---|
| 600 | -1.88 / +0.52 | -1.36 / +1.02 | -1.33 / +0.98 |
| 800 | -1.98 / +0.39 | -1.40 / +1.16 | -1.18 / +1.43 |
| 1,000 | -1.88 / +0.70 | -1.12 / +1.26 | -1.11 / +1.24 |
| 1,100 | -1.88 / +0.68 | -1.15 / +1.55 | -1.18 / +1.52 |
| 1,200 | -1.71 / +0.80 | -1.21 / +1.23 | **-1.00 / +1.25** |
| 1,300 | -1.82 / +0.84 | **-0.98 / +1.33** | -1.21 / +1.16 |
| 1,400 | -1.96 / +0.88 | -1.37 / +1.70 | |
| 1,500 | -1.83 / +0.71 | -1.25 / +1.34 | |
| 1,600 | -1.76 / +0.70 | -1.25 / +1.32 | |
| 1,800 | -1.81 / +0.95 | | |
| 1,900 | -1.96 / +0.88 | | |

SPD (new, from scratch): update 100 -1.71 / +0.96, 200 -1.12 / +1.23,
300 -1.13 / +1.34, 500 -1.20 / +1.41, 600 -1.03 / +1.26, 700 -1.18 / +1.41.

Lineage windows (per-deal pair scores averaged over snapshots; paired
whole-deal bootstrap; `.work/history-recipe-kits/lineages-r4-r5.json`):

| Window | SP vs B11 | DP vs B11 | L5 vs B11 | SP - L5 | DP - L5 | DP - SP |
|---|---:|---:|---:|---:|---:|---:|
| 650-900 | -1.32 | -1.25 | -2.04 | +0.72 [+0.52, +0.93] | +0.79 [+0.56, +1.01] | +0.07 [-0.12, +0.26] |
| 950-1,200 | -1.07 | -1.05 | -1.83 | +0.75 [+0.52, +0.98] | +0.77 [+0.54, +1.01] | +0.02 [-0.14, +0.19] |
| 1,250-1,500 | -1.20 | -1.09 | -1.88 | +0.67 [+0.47, +0.89] | +0.79 [+0.51, +1.07] | +0.12 [-0.13, +0.36] |
| 1,550-1,800 | -1.30 | | -1.79 | +0.49 [+0.26, +0.73] | | |

Against the long-run endpoint and greedy the same windows give SP - L5 of
+0.45..+0.74 and +0.55..+0.73; DP - L5 of +0.45..+0.70 and +0.38..+0.70.

SPD versus run-4 SP (one variable, 2-rank data parallel): 0-300 +0.03
[-0.07, +0.12] vs B11 and +0.07 [-0.05, +0.18] vs greedy; 100-300 +0.06
[-0.06, +0.18] vs B11 and +0.22 [+0.07, +0.38] vs greedy; 350-600 +0.03
[-0.17, +0.23] vs B11. Data parallelism adds little once self-play already
doubles the learner rows.

Full-match wins (of 16, update 600 onward for the joined lineages): SP 1/16
against B11 at updates 850, 1,150, 1,250, 1,400 and 1,650 and against the
long-run endpoint at 900; DP 2/16 against B11 at 650 (run 4), 1/16 at 850, 1,050
and 1,300, and 1/16 against each baseline at 1,000; SPD 1/16 against B11 at 210,
240, 350 and 450 and against the long-run endpoint at 230; L5 none. Every
snapshot's duplicate mean remains negative against both frozen MLPs.

## Reading

- **The recipe gain is durable:** after about 1,000-1,700 updates the two recipe
  lineages still lead the control lineage by 0.5-0.8 levels/round with
  intervals well above zero, and the control has not caught up.
- **A new plateau:** SP and DP fluctuate around -1.0..-1.3 against B11 from
  update ~900 on. The best single points are SP -0.98 (update 1,300) and DP
  -1.00 (update 1,200). More of the same recipe is unlikely to reach parity
  within this budget; see the summary for next steps.
- **Engineering:** cross-pod resume of both single-process and 3-rank DDP
  lineages worked on the first attempt with identity checks intact.

No default, checkpoint or document promotion follows; no strength claim.

## Budget and closure

Pod `r30ec2896ztpuf`: created 12:23:07 UTC, confirmed gone after 5,056 s,
estimated **$1.017**; 147 files SHA-256 verified before deletion; independent
readback zero pods, $0/hour, balance $27.8538. Running total for this goal:
**$4.65** estimated (balance delta $4.46). Kit:
`.work/runpod-history-recipe5-2026-09-26/`.
