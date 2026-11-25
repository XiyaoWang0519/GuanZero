# Next-coefficient endpoint evaluation — November 4, 2026

Four existing endpoints played the shared u96596 fork checkpoint on
2,000 fixed deals (seed 2026100301), twice per deal with team seats swapped.
All 4,000 rounds per arm scored. Greedy batched CUDA evaluation used
`eval.lineage_eval`, batch size 512 and KV caching. All four init files
had SHA-256 `356b47343dd7f25d44a205119ea6a9811fa25fc5cd1a4e01042b767614fee6fd`.
Deal-set SHA-256 was `e2b0c9160fc2ff550956d2cad2b8931e4fe7f60a508f64c4861013abfaaec6f0`.

| Arm | Update | Decisions | Net levels/round vs fork | Win rate |
|---|---:|---:|---:|---:|
| next-c01 | 101798 | 6,671,433,728 | +0.01825 | 50.600% |
| next-c03 | 103442 | 6,779,174,912 | +0.02250 | 51.025% |
| next-off | 107297 | 7,031,816,192 | +0.08875 | 51.075% |
| next-anneal | 104194 | 6,828,457,984 | +0.04425 | 49.950% |

Only next-off's net levels interval against the fork excludes zero.
All between-arm intervals include or touch zero. Training exposure differs
across endpoints, so these measurements do not isolate coefficient effects.

Velocity jobs 19337_0, 19337_1, 19337_2 and 19341 produced
`~/guanzero/results/gz-next-paired-19337-next-{c01,c03,off,anneal}.json`.
Task 19337_3 failed the GPU allocation guard before play and was retried as 19341.

The first four rows below give arm, checkpoint SHA-256, mean levels,
evaluator interval, wins, win fraction and evaluator source SHA-256.
Contrast rows give A, B, net level difference and interval, then win fraction
difference and interval. Arm-arm differences pair by deal against the same
fork. Arm-baseline win differences use the two opposing win fractions.
Contrast intervals use 5,000 whole-deal bootstrap replicates with seed 0
and are pointwise.

```text
next-c01 92b3c31ac6c3fccfe45c109511fd3050fc4ff9d5fffab078b4d95ec243f70a6e 0.01825 [-0.03475, 0.06875624999999998] 2024.0 0.506 3855c274b22a4e000d9a5a34151639c85b97012b5944941f7eb63c280bcb15ea
next-c03 1a8c7413df45974c0b5167512c5103545b317c446a9791f46885f2b645a2ec0b 0.0225 [-0.02725624999999999, 0.07426249999999993] 2041.0 0.51025 3855c274b22a4e000d9a5a34151639c85b97012b5944941f7eb63c280bcb15ea
next-off bbf460747ca342d4047db734f135d845341b136a58df785a6765f5e2b26940a2 0.08875 [0.0365, 0.139] 2043.0 0.51075 3855c274b22a4e000d9a5a34151639c85b97012b5944941f7eb63c280bcb15ea
next-anneal 25b23f43ff374d3a346edab251324afcc6da6f06473dc4d4438f56effc8307c9 0.04425 [-0.006762499999999978, 0.09401249999999993] 1998.0 0.4995 3855c274b22a4e000d9a5a34151639c85b97012b5944941f7eb63c280bcb15ea
next-c01 next-c03 -0.00425 [-0.07625, 0.06675] -0.00425 [-0.021, 0.0125]
next-c01 next-off -0.0705 [-0.14125, 0.0] -0.00475 [-0.02125, 0.0115]
next-c01 next-anneal -0.026 [-0.09875, 0.04575] 0.0065 [-0.01025, 0.02275]
next-c03 next-off -0.06625 [-0.13525, 0.004] -0.0005 [-0.018, 0.016]
next-c03 next-anneal -0.02175 [-0.09275, 0.049] 0.01075 [-0.0065, 0.0275]
next-off next-anneal 0.0445 [-0.02625, 0.11575] 0.01125 [-0.00525, 0.02775]
next-c01 baseline 0.01825 [-0.03375, 0.0695] 0.012 [-0.0125, 0.0355]
next-c03 baseline 0.0225 [-0.029, 0.0745] 0.0205 [-0.004, 0.045]
next-off baseline 0.08875 [0.03775, 0.13875] 0.0215 [-0.002, 0.045]
next-anneal baseline 0.04425 [-0.00575, 0.09575] -0.001 [-0.0245, 0.024]
```
