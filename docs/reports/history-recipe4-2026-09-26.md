# Recipe run 4: two epochs as the reference, one addition per arm, September 26, 2026 UTC

Two PPO epochs had passed its A/B in runs 1-3, so it became the reference
recipe for new arms. This run adds one change per arm on top of it and keeps
the original control on the same seed. **Two epochs replicated a fourth time
(this seed ended at -1.33 against B11 over updates 650-900, the control at
-2.04). Data-parallel collection (DP) and current-policy-only self-play (SP)
each added about +0.25..+0.30 early on top of two epochs, shrinking later; GAE
lambda 1.0 and critic lr 1e-3 were worse.** Development curves now sit clearly
above the old -2.1..-2.4 plateau, and some snapshots won 1-2 of 16 full matches
against B11, but every snapshot still loses on average (best -1.12 net
levels/round). That is a development result, not a strength claim.

## Design and identity

Reference = control config (batch-size kit arm A) plus 2 PPO epochs; each arm
changes one field from it except L5, which is the original control. Seed
2026092605 for all arms. 4,800 s per arm, trained on CPU.

| Arm | Change | Updates | Learner rows (all ranks) |
|---|---|---:|---:|
| B | two epochs (reference) | 816 | 1,017,956 |
| L5 | original control (1 epoch) | 957 | 1,192,932 |
| DP | two epochs + 3-rank data parallel | 679 | 2,539,638 |
| G | two epochs + GAE lambda 1.0 | 780 | 979,306 |
| C | two epochs + critic lr 1e-3 | 779 | 980,510 |
| SP | two epochs + snapshot probability 0 (current-policy copies only) | 833 | 1,638,324 |

First create request returned HTTP 500; two readbacks by manifest name found
no pod and $0/hour, and the retry created pod `j2irwcd88p1d75` (RTX PRO 4500
Blackwell Secure, EU-RO-1, AMD EPYC 7713P, quota 27.2 CPUs, $0.724/hour) at
10:54:43 UTC. Source SHA-256 `5b05d3ef…3854` (adds DDP resume support and its
test to run 2's source); archive `c454f128…`; manifest `fc6f91ea…`. Pod test
gate: 101 passed. Final slim checkpoints: B `ff779dd9`, L5 `77e33bbe`,
DP `c9b53399`, G `49882d73`, C `30e7a5ad`, SP `3cbb8d2c` (prefixes).

## Development curves

Each cell: vs B11 / vs greedy / entropy on the frozen 64 deals.

| Update | B | L5 | DP | G | C | SP |
|---:|---|---|---|---|---|---|
| 100 | -1.86/+0.55/0.64 | -2.08/+0.41/0.57 | -1.54/+0.78/0.61 | -2.18/+0.37/0.84 | -2.10/+0.61/0.77 | -1.64/+0.71/0.68 |
| 200 | -1.68/+0.77/0.65 | -2.27/+0.23/0.35 | -1.69/+1.19/0.41 | -1.98/+0.38/0.58 | -1.79/+0.83/0.52 | -1.51/+0.87/0.49 |
| 300 | -1.61/+0.77/0.47 | -2.07/+0.34/0.22 | -1.23/+0.90/0.34 | -2.01/+0.18/0.52 | -1.55/+0.70/0.46 | -1.38/+0.91/0.44 |
| 400 | -1.29/+1.31/0.39 | -1.97/+0.20/0.27 | **-1.12/+1.47**/0.27 | -1.59/+0.12/0.46 | -1.97/+0.35/0.39 | -1.38/+0.95/0.40 |
| 500 | -1.16/+0.93/0.41 | -1.81/+0.44/0.24 | -1.16/+1.18/0.35 | -1.52/+0.43/0.46 | -1.68/+0.65/0.47 | -1.24/+1.21/0.34 |
| 600 | -1.35/+1.08/0.36 | -1.88/+0.52/0.17 | -1.33/+0.98/0.29 | -1.56/+0.58/0.39 | -1.93/+0.63/0.41 | -1.36/+1.02/0.28 |
| 700 | -1.45/+0.78/0.39 | -1.96/+0.34/0.16 | | -1.62/+0.75/0.36 | -1.48/+1.08/0.31 | -1.19/+1.41/0.35 |
| 800 | -1.41/+1.03/0.44 | -1.98/+0.39/0.11 | | | | -1.40/+1.16/0.37 |

Two-epoch arms keep entropy near 0.3-0.45 late; the one-epoch control falls to
about 0.1.

## Comparisons

Against the in-pod two-epoch reference B (paired whole-deal bootstrap):

| Window | Arm - B | vs B11 | vs long-run endpoint | vs greedy |
|---|---|---:|---:|---:|
| 0-300 | DP | **+0.30 [+0.19, +0.42]** | **+0.32 [+0.22, +0.42]** | **+0.33 [+0.20, +0.47]** |
| 0-300 | SP | **+0.24 [+0.15, +0.33]** | **+0.24 [+0.15, +0.33]** | **+0.25 [+0.10, +0.39]** |
| 0-300 | G | -0.21 [-0.30, -0.12] | -0.16 [-0.26, -0.06] | -0.34 [-0.49, -0.19] |
| 0-300 | C | -0.05 [-0.16, +0.05] | +0.03 [-0.06, +0.12] | +0.06 [-0.07, +0.19] |
| 0-300 | L5 (control) | -0.19 [-0.29, -0.09] | -0.11 [-0.21, -0.00] | -0.24 [-0.42, -0.06] |
| 350-600 | DP | **+0.17 [+0.03, +0.31]** | +0.09 [-0.06, +0.24] | +0.10 [-0.07, +0.28] |
| 350-600 | SP | +0.08 [-0.11, +0.28] | +0.10 [-0.07, +0.27] | +0.08 [-0.12, +0.28] |
| 350-600 | G / C | -0.28 / -0.40 | -0.39 / -0.40 | -0.53 / -0.51 |
| 650-900 | DP (1 snapshot) | +0.13 [-0.19, +0.46] | +0.05 [-0.28, +0.38] | +0.21 [-0.11, +0.53] |
| 650-900 | SP | +0.03 [-0.19, +0.24] | +0.17 [-0.06, +0.41] | +0.13 [-0.13, +0.38] |
| 650-900 | L5 (control) | -0.71 [-0.93, -0.48] | -0.47 [-0.66, -0.27] | -0.60 [-0.87, -0.34] |

Pooled across runs 1-4 (`.work/history-recipe-kits/pooled-after-r4.json`), six
CPU control seeds versus four two-epoch seeds:

| Window | vs B11 | vs long-run endpoint | vs greedy |
|---|---:|---:|---:|
| 100-300 | +0.09 [+0.01, +0.16] | +0.05 [-0.01, +0.11] | +0.07 [-0.02, +0.16] |
| 350-600 | **+0.25 [+0.15, +0.36]** | **+0.22 [+0.13, +0.30]** | **+0.20 [+0.08, +0.32]** |
| 650-900 | **+0.37 [+0.25, +0.49]** | **+0.33 [+0.21, +0.45]** | **+0.34 [+0.22, +0.47]** |

Per-run means vs B11 over 650-900: controls -2.07, -2.04, -2.03, -1.96, -1.94,
-1.83; two epochs -1.85, -1.69, -1.58, -1.33.

Full-match wins (of 16 per baseline per snapshot): DP won 1/16 against B11 at
updates 140, 200 and 210 and 2/16 at 400 and 650; SP won 1/16 against the
long-run endpoint at 210, 250 and 270 and 1/16 against B11 at 400 and 450;
B won 1/16 against B11 at 260; L5, G and C won none. Every snapshot's
duplicate mean is still negative against both baselines.

## Reading

- **Two epochs is the established recipe change** (four seeds, never below the
  control over 0-300, +0.37 late). Its entropy stays higher than the control's.
- **DP and SP accelerate learning on top of it**, mostly early; later their
  lead over the strong B seed is small and mostly within noise. Both deliver
  more learner rows per wall-clock second (DP 2.5x, SP 1.6x). Run 5 continues
  both lineages and the control to see whether the gap survives more training.
- **GAE lambda 1.0 and critic lr 1e-3 hurt.**
- **Engineering:** the DDP arm ran three ranks at about 6.6 s per update for
  6,144 decisions; parameter checksums matched across ranks after every update.

No default, checkpoint or document promotion follows; no strength claim.

## Budget and closure

Pod `j2irwcd88p1d75`: created 10:54:43 UTC, confirmed gone after 5,078 s,
estimated **$1.022**; 313 files SHA-256 verified before deletion; independent
readback zero pods, $0/hour, balance $28.8935. Running total for this goal:
**$3.63** estimated (balance delta $3.42). Kit:
`.work/runpod-history-recipe4-2026-09-26/`.
