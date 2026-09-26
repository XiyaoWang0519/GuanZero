# Recipe run 3: replicating two PPO epochs and testing optimizer-step variants, September 26, 2026 UTC

Run 1's only positive arm was two PPO epochs; run 2 showed that 5x the samples
with the same optimizer steps gains little. This run replicates two epochs on
two seeds against two control seeds and tests four epochs, learning rate 6e-4,
one-match minibatches and pure current-policy self-play, one variable each.
**Pooled across runs, two epochs (three seeds) beat five CPU control seeds by
+0.26 [+0.11, +0.42] net levels/round against B11 over updates 650-900, and
every two-epoch seed ended above every control seed's mean in that window.**
Four epochs and a 16-snapshot population hurt; the other arms are within noise.
One two-epoch snapshot won 1 of 16 full matches against B11; that is not a
strength claim.

## Allocation history (three kits)

| Kit | Outcome | Estimated cost |
|---|---|---:|
| `runpod-history-recipe3-2026-09-26` | RTX 5090 Community pod `dgvo65fruljrb7` never received a machine or SSH; monitor deleted it after 914 s (bounded initialization), zero pods after; balance unchanged | $0.176 (ledgered conservatively) |
| `runpod-history-recipe3b-2026-09-26` | RTX 2000 Ada Secure: two create requests returned HTTP 500; each reconciled by manifest name with two readbacks (zero pods, $0/hour); nothing created | $0 |
| `runpod-history-recipe3c-2026-09-26` | RTX PRO 4500 Blackwell Secure pod `nq3gyo7f48oco7` (EU-RO-1, AMD EPYC 7763, quota 27.2 CPUs, $0.724/hour); the run below | $1.012 |

## Design and identity (kit recipe3c)

Control config = batch-size kit arm A (32 envs, KV + SDPA, lr 3e-4, entropy
0.01, 1 epoch, 2 matches/minibatch, snapshots every 2 updates, 4 recent,
p = 0.5), trained on CPU; each arm changes one field. 4,800 s per arm.

| Arm | Change | Seed | Updates | Learner rows |
|---|---|---|---:|---:|
| L4 | none | 2026092603 | 1,005 | 1,233,917 |
| L4s | none | 2026092604 | 1,086 | 1,344,256 |
| P2 | epochs 2 | 2026092603 | 840 | 1,042,777 |
| P2s | epochs 2 | 2026092604 | 818 | 1,018,040 |
| P4 | epochs 4 | 2026092603 | 633 | 785,188 |
| Q | lr 6e-4 | 2026092603 | 1,016 | 1,243,505 |
| M1 | 1 match per minibatch | 2026092603 | 1,058 | 1,296,459 |
| S | snapshot probability 0 (current-policy copies only) | 2026092603 | 1,231 | 2,413,122 |

Source SHA-256 `16bbd654…0406f` (same as run 2); archive `78229dc1…`; manifest
`93dbe99f…`; pod created 09:26:25 UTC. Pod test gate: 100 passed. Final slim
checkpoints: L4 `f057ebf1`, L4s `ccd534e9`, P2 `14da3797`, P2s `6a458ab6`,
P4 `7cd5b15a`, Q `210295ee`, M1 `44656e9a`, S `a6ba46ed` (prefixes).

## Development curves

Each cell: vs B11 / vs greedy / entropy on the frozen 64 deals.

| Update | L4 | L4s | P2 | P2s | P4 | Q | M1 | S |
|---:|---|---|---|---|---|---|---|---|
| 100 | -2.24/+0.48/0.45 | -1.94/+0.29/0.53 | -2.33/+0.47/0.70 | -2.06/+0.13/0.64 | -2.23/-0.29/0.83 | -2.30/+0.55/0.47 | -2.12/+0.06/0.60 | -2.05/+0.56/0.40 |
| 300 | -2.05/+0.41/0.38 | -1.86/+0.44/0.23 | -1.80/+0.42/0.38 | -1.81/+0.32/0.38 | -2.27/+0.23/0.56 | -1.96/+0.59/0.27 | -1.96/+0.19/0.18 | -1.80/+0.53/0.28 |
| 500 | -1.97/+0.53/0.15 | -1.92/+0.45/0.19 | -1.85/+0.70/0.37 | -1.74/+0.52/0.39 | -1.82/+0.56/0.51 | -2.14/+0.79/0.15 | -2.05/+0.48/0.19 | -1.87/+0.41/0.22 |
| 700 | -1.88/+0.25/0.16 | -1.84/+0.92/0.20 | -1.65/+0.96/0.33 | -1.70/+0.69/0.35 | | -1.84/+0.76/0.08 | -2.23/+0.38/0.18 | -1.94/+0.76/0.19 |
| 800 | -1.88/+0.42/0.14 | -2.10/+0.55/0.20 | **-1.43**/+1.05/0.31 | -1.80/+0.91/0.28 | | -1.95/+0.34/0.11 | -1.84/+1.02/0.14 | -1.88/+0.93/0.16 |

## Pooled across runs 1-3

Per run, per-deal pair scores are averaged over its snapshots in the window; a
group is the mean of its runs; brackets are a paired whole-deal bootstrap of
group minus control (training-seed variance shows in the per-run means, not in
the bracket). Control = five CPU runs of the control config (run 1 L, run 2
L2/L3, run 3 L4/L4s); two epochs = run 1 P, run 3 P2/P2s. Output:
`.work/history-recipe-kits/pooled-after-r3.json`.

| Window | Group (runs) | vs B11 minus control | vs long-run minus control | vs greedy minus control |
|---|---|---:|---:|---:|
| 100-300 | two epochs (3) | +0.01 [-0.07, +0.09] | +0.01 [-0.07, +0.08] | -0.02 [-0.12, +0.09] |
| 350-600 | two epochs (3) | **+0.14 [+0.03, +0.25]** | **+0.12 [+0.02, +0.23]** | +0.08 [-0.05, +0.20] |
| 650-900 | two epochs (3) | **+0.26 [+0.11, +0.42]** | **+0.26 [+0.13, +0.40]** | **+0.27 [+0.14, +0.40]** |
| 100-300 | four epochs (1) | -0.13 [-0.26, -0.01] | -0.09 [-0.19, +0.01] | **-0.64 [-0.82, -0.44]** |
| 100-300 | 16 recent snapshots (1) | -0.11 [-0.21, -0.00] | -0.11 [-0.21, -0.01] | -0.26 [-0.44, -0.09] |
| 650-900 | 16 recent snapshots (1) | -0.17 [-0.33, -0.00] | -0.06 [-0.20, +0.09] | **-0.48 [-0.71, -0.26]** |
| 650-900 | self-play only (1) | +0.03 [-0.10, +0.18] | +0.14 [-0.01, +0.29] | +0.17 [-0.02, +0.34] |
| 650-900 | lr 6e-4 (1) | +0.01 [-0.16, +0.19] | -0.08 [-0.21, +0.05] | -0.15 [-0.37, +0.06] |
| 650-900 | 1-match minibatches (1) | -0.09 [-0.23, +0.06] | +0.07 [-0.07, +0.22] | -0.09 [-0.32, +0.14] |
| 650-900 | entropy 0.03 (1) | +0.06 [-0.09, +0.21] | +0.16 [+0.02, +0.32] | +0.08 [-0.11, +0.26] |
| 650-900 | data parallel x6 (1) | -0.08 [-0.26, +0.11] | +0.30 [+0.09, +0.53] | +0.32 [+0.11, +0.52] |

Per-run means vs B11 over 650-900: controls -2.07, -2.03, -1.96, -1.94, -1.83;
two epochs -1.85, -1.69, -1.58.

Full-match wins: one match of 16 against B11 at P2 update 800 (duplicate score
-1.43); none in any other snapshot of this run.

## Reading

- **Two epochs is a seed-robust development improvement.** It passes the
  acceptance rule (not below the control over 0-300 in any run) and its late
  advantage is consistent across three seeds and both frozen MLPs.
- **More is not better:** four epochs is clearly worse early (entropy stays high,
  0.83 at update 100, and the greedy score lags) and only catches up to the
  control by 500; lr 6e-4 and halving the minibatch give no gain. The benefit of
  two epochs is not simply "more optimizer steps".
- **Population:** a longer window (16) hurts; current-policy-only self-play is
  neutral to slightly positive and yields twice the learner rows per decision.
- **Still no strength result:** the best points (-1.43 to -1.7 against B11) remain
  far below parity, and full-match wins are 1/16 at isolated snapshots.

## Budget and closure

Kit recipe3c: pod created 09:26:25 UTC, confirmed gone after 5,027 s,
estimated $1.012; 431 files SHA-256 verified before deletion; independent
readback zero pods, $0/hour, balance $29.9247. Running total for this goal
after run 3: **$2.61** estimated (balance delta $2.39). Artifacts:
`.work/runpod-history-recipe3c-2026-09-26/` (`results/curves.json`,
`results/analysis.json`, `results/greedy/`, `local/results/arm-*/`, receipts).
