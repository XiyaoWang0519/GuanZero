# Merged snapshot inference A/B, updates 845-919

Machine 67872 (offer 50569797); CPU Intel(R) Xeon(R) Platinum 8173M CPU @ 2.00GHz, 26 usable of 112 logical CPUs (cgroup quota 26.9); GPU NVIDIA GeForce RTX 4090, 1, 16, 3135 MHz, 300.00 W. Absolute speed is comparable only on the same machine; the on/off ratio is the in-run comparison.

4 ranks; schedule `35:off,10:on,10:off,10:on,5:off,5:on`; first 2 updates of each block dropped. profiled blocks synchronize CUDA at every phase boundary; their seconds attribute time, speed comes from unprofiled blocks.

## Per arm (unprofiled blocks, settled)

| | off | on |
|---|---:|---:|
| blocks | 2 | 2 |
| wall s/update | 20.84 | 15.39 |
| decisions/s (wall) | 3,145.01 | 4,262.62 |
| collect s (slowest rank) | 14.18 | 8.87 |
| learn s (slowest rank) | 8.12 | 7.00 |
| policy calls/step (rank mean) | 12.63 | 2.00 |
| GPU util % | 87.63 | 91.66 |
| GPU mem max MiB | 21,250.00 | 23,578.00 |
| entropy | 0.6259 | 0.6170 |
| approx KL max | 0.0095 | 0.0117 |
| allocation retries (max) | 0 | 0 |

on / off decisions per wall second: **1.355**

## Blocks (settled)

| arm | profiled | updates | wall s/upd | dec/s wall | collect s | learn s | calls/step | GPU % | GPU MiB | peak GB | heads MB | retries | entropy | KL max |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off | False | 872-879 | 20.65 | 3,173 | 14.03 | 7.87 | 12.75 | 89.0 | 20,458 | 5.31 | 0.0 | 0 | 0.6336 | 0.0095 |
| on | False | 882-889 | 14.93 | 4,391 | 8.55 | 6.68 | 2.00 | 92.6 | 21,870 | 5.39 | 32.8 | 0 | 0.6215 | 0.0117 |
| off | False | 892-899 | 21.02 | 3,117 | 14.33 | 8.37 | 12.51 | 86.2 | 21,250 | 5.80 | 0.0 | 0 | 0.6181 | 0.0058 |
| on | False | 902-909 | 15.85 | 4,134 | 9.19 | 7.32 | 2.00 | 90.8 | 23,578 | 5.71 | 32.8 | 0 | 0.6125 | 0.0075 |
| off | True | 912-914 | 21.17 | 3,096 | 14.49 | 8.31 | 12.44 | 83.8 | 21,238 | 5.75 | 0.0 | 0 | 0.6119 | 0.0057 |
| on | True | 917-919 | 16.22 | 4,040 | 9.54 | 7.02 | 2.00 | 78.1 | 22,230 | 5.70 | 32.8 | 0 | 0.6058 | 0.0076 |

## Profiled blocks (synchronized; all ranks)

| arm | group | calls/step | mean rows | p10 | median | p90 | max | cache s/upd | actor s/upd |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| off | learner | 1.00 | 154.1 | 145 | 154 | 164 | 171 | 3.08 | 0.44 |
| off | snapshot | 11.44 | 8.2 | 1 | 7 | 16 | 30 | 4.07 | 5.26 |
| on | learner | 1.00 | 153.9 | 144 | 154 | 163 | 175 | 3.19 | 0.58 |
| on | snapshot | 1.00 | 93.8 | 85 | 94 | 104 | 117 | 4.55 | 0.17 |
