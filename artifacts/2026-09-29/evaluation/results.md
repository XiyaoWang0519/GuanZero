# Overnight main-lineage checkpoints vs B11 (256 duplicate deals)

| ckpt | update | decisions | entropy (31-upd mean) | vs B11 | 95% CI | match win [95% CI] | paired vs u844 | paired vs previous | wall |
|---|---:|---:|---:|---:|---|---|---|---|---|
| u0844 | 844 | 55.3M | 0.631 | -0.18 | [-0.32, -0.04] | 29% [22%, 36%] | - | - | 43 min |
| u1333 | 1333 | 87.4M | 0.603 | -0.35 | [-0.52, -0.18] | 36% [28%, 44%] | -0.17 [-0.37, +0.04] | u0844: -0.17 [-0.37, +0.04] | 48 min |
| u1612 | 1612 | 105.6M | 0.631 | -0.10 | [-0.25, +0.06] | 38% [30%, 46%] | +0.08 [-0.12, +0.29] | u1333: +0.25 [+0.03, +0.47] | 40 min |
| u1829 | 1829 | 119.9M | 0.606 | +0.02 | [-0.15, +0.20] | 51% [43%, 59%] | +0.20 [-0.01, +0.41] | u1612: +0.12 [-0.11, +0.34] | 52 min |
| u2015 | 2015 | 132.1M | 0.555 | -0.03 | [-0.20, +0.13] | 45% [37%, 53%] | +0.15 [-0.07, +0.36] | u1829: -0.05 [-0.27, +0.17] | 54 min |
| u2108 | 2108 | 138.1M | 0.528 | +0.09 | [-0.07, +0.26] | 51% [41%, 60%] | +0.27 [+0.06, +0.49] | u2015: +0.12 [-0.09, +0.34] | 41 min |
| u2201 | 2201 | 144.2M | 0.541 | +0.25 | [+0.08, +0.41] | 48% [40%, 55%] | +0.42 [+0.21, +0.63] | u2108: +0.16 [-0.05, +0.37] | 36 min |
| u2356 | 2356 | 154.4M | 0.547 | +0.14 | [-0.02, +0.31] | 49% [41%, 58%] | +0.32 [+0.12, +0.54] | u2201: -0.10 [-0.33, +0.12] | 38 min |

Paired vs u675 (44.2M, overnight-large final): u0844 +0.19 [-0.01, +0.38]; u1333 +0.01 [-0.21, +0.24]; u1612 +0.27 [+0.05, +0.48]; u1829 +0.38 [+0.17, +0.60]; u2015 +0.33 [+0.12, +0.54]; u2108 +0.45 [+0.23, +0.69]; u2201 +0.61 [+0.40, +0.82]; u2356 +0.51 [+0.29, +0.72]

Method: see results.json `method`.

## Late segment (u2108 onward; u2108, u2201, u2356)

All intervals resample the same 256 deals jointly across checkpoints (4,000 samples, seed 0). The checkpoints share deals and are consecutive in one lineage, so the pooled number is a smoothed estimate of the late plateau, not independent replication.

- Pooled late mean vs B11 (per-deal scores averaged over the 3 checkpoints, then bootstrap over deals): +0.16 [+0.06, +0.27]
- Pooled late mean minus u844: +0.34 [+0.17, +0.51]
- OLS slope of checkpoint means on update: +0.01 [-0.07, +0.10] levels/round per 100 updates (span 248 updates, so +0.03 over the span)
- Later half (u2356) minus earlier half (u2108): +0.05 [-0.16, +0.27]
- Endpoint u2356 minus mean of the other late checkpoints: -0.02 [-0.22, +0.17]
- Endpoint minus u2201 (best in-run point): -0.10 [-0.33, +0.12]
- Range of late point estimates: +0.09 (u2108) to +0.25 (u2201)

## Notes from the in-run evaluation (added after the first eight points; post hoc, exploratory)

- u844 reproduction: bit-identical to `.work/actor-ranks-2026-09-28/eval-256/main-w4.json` (same candidate sha, same source sha 14e72581..., all 256 per-deal scores and all 64 match pairs equal).
- Entropy step starts at about update 1838-1840; u1829 is the last synced checkpoint before it.
- Pooled, post hoc: mean of u2015/u2108/u2201/u2356 minus u1829 = +0.09 [-0.08, +0.27]; minus mean of u1612/u1829 = +0.15 [+0.02, +0.29]; u2356 - u1829 = +0.12 [-0.10, +0.35].
- Match win-rate 95% CIs (64 seed pairs): u1829 [0.43, 0.59], u2201 [0.40, 0.55], u2356 [0.41, 0.58].
- Wall times are with 2-4 evaluations running concurrently (4 torch threads each); single-run time will be lower.
