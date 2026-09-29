# Overnight main lineage vs B11, 256 duplicate deals (v2: adds late checkpoints)

**B11-only verification (u2201 B11-only vs `out/u2201.json`): PASS.** Per-deal scores differing: 0 of 256; match pairs differing: 0 of 64; whole b11-main report identical: True; same candidate sha True, same evaluator source True. freeze_sha256 recorded: original f77ea34dbd4bae50..., B11-only 5926f7516786f0c3... (differs by design; evaluator does not check it against a list). u2201 B11-only wall time: 25 min (concurrent 2-4) vs 36 min for the two-baseline run.

| ckpt | update | decisions | entropy (31-upd mean) | vs B11 [95% CI] | match win [95% CI] | paired vs u844 | paired vs previous | run | wall (concurrent) |
|---|---:|---:|---:|---|---|---|---|---|---|
| u0844 | 844 | 55.3M | 0.631 | -0.18 [-0.32, -0.04] | 29% [22%, 36%] | - | - | two-baseline | 43 min (2-4) |
| u1333 | 1333 | 87.4M | 0.603 | -0.35 [-0.52, -0.18] | 36% [28%, 44%] | -0.17 [-0.37, +0.04] | u0844: -0.17 [-0.37, +0.04] | two-baseline | 48 min (2-4) |
| u1612 | 1612 | 105.6M | 0.631 | -0.10 [-0.25, +0.06] | 38% [30%, 46%] | +0.08 [-0.12, +0.29] | u1333: +0.25 [+0.03, +0.47] | two-baseline | 40 min (2-4) |
| u1829 | 1829 | 119.9M | 0.606 | +0.02 [-0.15, +0.20] | 51% [43%, 59%] | +0.20 [-0.01, +0.41] | u1612: +0.12 [-0.11, +0.34] | two-baseline | 52 min (2-4) |
| u2015 | 2015 | 132.1M | 0.555 | -0.03 [-0.20, +0.13] | 45% [37%, 53%] | +0.15 [-0.07, +0.36] | u1829: -0.05 [-0.27, +0.17] | two-baseline | 54 min (2-4) |
| u2108 | 2108 | 138.1M | 0.528 | +0.09 [-0.07, +0.26] | 51% [41%, 60%] | +0.27 [+0.06, +0.49] | u2015: +0.12 [-0.09, +0.34] | two-baseline | 41 min (2-4) |
| u2201 | 2201 | 144.2M | 0.541 | +0.25 [+0.08, +0.41] | 48% [40%, 55%] | +0.42 [+0.21, +0.63] | u2108: +0.16 [-0.05, +0.37] | two-baseline | 36 min (2-4) |
| u2263 | 2263 | 148.3M | 0.559 | -0.01 [-0.18, +0.16] | 50% [44%, 57%] | +0.17 [-0.05, +0.39] | u2201: -0.26 [-0.46, -0.05] | B11-only | 23 min (1-4) |
| u2356 | 2356 | 154.4M | 0.547 | +0.14 [-0.02, +0.31] | 49% [41%, 58%] | +0.32 [+0.12, +0.54] | u2263: +0.16 [-0.06, +0.38] | two-baseline | 38 min (2-4) |
| u2418 | 2418 | 158.5M | 0.539 | -0.02 [-0.18, +0.14] | 53% [45%, 62%] | +0.16 [-0.05, +0.37] | u2356: -0.16 [-0.38, +0.07] | B11-only | 22 min (2-4) |
| u2480 | 2480 | 162.5M | 0.552 | +0.10 [-0.06, +0.27] | 48% [40%, 58%] | +0.28 [+0.05, +0.51] | u2418: +0.12 [-0.11, +0.33] | B11-only | 27 min (3-4) |
| u2542 | 2542 | 166.6M | 0.546 | +0.15 [-0.00, +0.30] | 54% [46%, 62%] | +0.33 [+0.11, +0.54] | u2480: +0.05 [-0.17, +0.27] | B11-only | 32 min (3-4) |
| u2623 | 2623 | 171.9M | 0.543 | +0.08 [-0.08, +0.24] | 60% [52%, 69%] | +0.26 [+0.03, +0.49] | u2542: -0.07 [-0.27, +0.13] | B11-only | 26 min (2-4) |

## Late segment (u2108 onward: u2108, u2201, u2263, u2356, u2418, u2480, u2542, u2623)

The checkpoints share the same 256 deals and are consecutive in one lineage, so the pooled number is a smoothed estimate of the late plateau, not independent replication. Intervals resample deals jointly.

- Pooled late mean vs B11: +0.10 [+0.02, +0.18]
- Pooled late mean minus u844: +0.27 [+0.11, +0.45]
- Trend: OLS slope -0.01 [-0.04, +0.03] levels/round per 100 updates over 515 updates (cannot distinguish from zero)
- Later half (u2418, u2480, u2542, u2623) minus earlier half (u2108, u2201, u2263, u2356): -0.04 [-0.15, +0.07] (cannot distinguish)
- Each late checkpoint minus endpoint u2623: u2108 +0.01 [-0.20, +0.23]; u2201 +0.17 [-0.05, +0.39]; u2263 -0.09 [-0.30, +0.12]; u2356 +0.07 [-0.14, +0.27]; u2418 -0.10 [-0.32, +0.14]; u2480 +0.02 [-0.19, +0.23]; u2542 +0.07 [-0.13, +0.27]

**Starting checkpoint recommendation:** no late checkpoint beats u2623 with a paired interval entirely above zero; default to the endpoint u2623.

Limits: one training lineage, one opponent (B11), 256 development deals, no final-test evaluation. Old points ran 2-4 evaluations concurrently with two baselines; new points ran B11 only with 2-4 concurrent (measured per point from start/end stamps, see column), always 4 torch threads each. Method details: results.v2.json `method`.
