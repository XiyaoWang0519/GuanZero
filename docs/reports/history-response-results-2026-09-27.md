# T7 results: shared versus explicit opponent-response connection

Status: complete, September 27 2026 UTC. Development-split diagnostic only; no
playing-strength promotion, final-test material unopened. Protocol, arms and
labels are in the [plan](history-response-plan-2026-09-26.md); placement in the
[device report](history-device-placement-2026-09-26.md).

## Run

- Arms A (PPO only), B (PPO + response loss, coefficient 0.1, shared features),
  C (B plus detached predicted probabilities into candidate scoring).
- Seeds 2026092801/02/03, all three arms concurrently on one RTX PRO 4500 pod
  per seed; CPU engine, CUDA model inference and learning, FP32.
- 4,800 trainer seconds per arm, 928–1,020 updates; fixed `final.pt` endpoints.
- Evaluation per endpoint and baseline: 256 duplicate deals (512 rounds) and 64
  full-match pairs, against frozen B11 main and the segment-2 raw endpoint.
- Cost: $3.84 estimated of the $6 cap, including the interrupted CPU attempt
  ($0.62). Every pod deleted; downloads verified; zero-spend readback.
- Artifacts: `.work/history-response-speed-2026-09-26/` (`comparison.json`,
  `seed-*/results/endpoint-*.json`, `ledger.json`).

## Absolute results

Net levels per round (duplicate) / full-match win rate, seeds 01 | 02 | 03.

| Arm | vs B11 main | vs segment-2 |
|---|---|---|
| A | −1.83 / 0.0% · −1.36 / 0.0% · −1.56 / 1.6% | −2.11 / 0.0% · −1.70 / 0.8% · −1.75 / 0.0% |
| B | −1.28 / 2.3% · −1.42 / 0.8% · −1.16 / 1.6% | −1.43 / 1.6% · −1.69 / 0.8% · −1.51 / 0.0% |
| C | −1.53 / 0.8% · −1.44 / 0.0% · −1.37 / 0.8% | −1.74 / 0.8% · −1.53 / 0.0% · −1.57 / 0.0% |

Every endpoint still loses on average to both baselines.

## Paired comparisons

Mean duplicate delta, 95% interval over seeds and deals.

| Comparison | vs B11 main | vs segment-2 |
|---|---|---|
| C − B (primary) | −0.16 [−0.35, +0.04] | −0.07 [−0.33, +0.19] |
| B − A | +0.29 [−0.04, +0.60] | +0.30 [−0.02, +0.65] |
| C − A | +0.14 [−0.10, +0.36] | +0.24 [+0.05, +0.41] |

Full-match win-rate deltas are all within ±0.03 and their intervals cross zero.

## Training diagnostics

Means over the last 50 updates: entropy 0.27–0.48 in every arm (collapsed, as
in the recipe campaign); explained variance 0.58–0.65; response accuracy
0.78–0.80 in B and C. A higher explained variance for B/C than A was not
consistent enough across seeds to claim.

## Reading

- The explicit connection (C) did not beat the shared representation (B); the
  point estimate is slightly negative. It is not pursued further.
- The auxiliary response loss (B) is the useful part: about +0.3 levels per round
  over PPO only, positive in two of three seeds, interval just touching zero.
  Three seeds make this exploratory, not a robust claim (T8 requires more).
- Absolute level (−1.2..−1.8 vs B11) is above the old −2.1..−2.4 plateau but
  full-match wins stay at 0–2%. Entropy collapse remains the main open problem.
- Next experiments build on arm B (`response_mode=auxiliary`). History-use
  (reduced-history) controls of T7 remain pending.

Process note: the first local evaluation stopped on a frozen engine-digest
mismatch caused by iCloud conflict copies (`search 2.h/.cpp`) matching the
engine-source glob. The identical copies were removed and the evaluation rerun.
