# Collection profile, updates 870-884

15 profiled updates, 4 ranks. profile mode synchronizes CUDA at every phase boundary: seconds attribute collection time to phases, they are not absolute speed.

## Phase seconds per update (synchronized)

| | rank 0 | rank 1 | rank 2 | rank 3 | mean |
|---|---:|---:|---:|---:|---:|
| env_pending | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| events_and_rounds | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |
| metadata_and_assignment | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| input_indexing | 0.09 | 0.08 | 0.08 | 0.08 | 0.08 |
| decision_upload | 0.13 | 0.11 | 0.11 | 0.11 | 0.12 |
| public_cache_or_collation | 7.01 | 6.89 | 7.28 | 6.88 | 7.01 |
| actor_and_sampling | 6.01 | 5.96 | 6.31 | 6.09 | 6.09 |
| decision_download | 0.07 | 0.08 | 0.06 | 0.07 | 0.07 |
| buffer_and_counters | 0.05 | 0.04 | 0.04 | 0.04 | 0.05 |
| env_step | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| unattributed (collect - phases) | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| **collect seconds** | 13.74 | 13.55 | 14.28 | 13.67 | 13.81 |
| learn seconds | 7.24 | 7.43 | 6.70 | 7.31 | 7.17 |

## Learner vs snapshot calls (seconds per update)

| | rank 0 | rank 1 | rank 2 | rank 3 | mean |
|---|---:|---:|---:|---:|---:|
| learner: public_cache_or_collation | 3.19 | 3.08 | 3.40 | 3.04 | 3.18 |
| learner: actor_and_sampling | 0.46 | 0.45 | 0.48 | 0.44 | 0.46 |
| snapshot: public_cache_or_collation | 3.82 | 3.80 | 3.88 | 3.84 | 3.84 |
| snapshot: actor_and_sampling | 5.55 | 5.50 | 5.82 | 5.65 | 5.63 |

## Policy-call batch sizes (rows per call, window total)

| rank | group | calls | calls/step | mean | median | p10 | p90 | min | max | cache ms/call | actor ms/call |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | learner | 960 | 1.00 | 155.1 | 155 | 146 | 164 | 133 | 175 | 49.79 | 7.18 |
| 0 | snapshot | 10,980 | 11.44 | 8.1 | 8 | 1 | 16 | 1 | 31 | 5.22 | 7.59 |
| 1 | learner | 960 | 1.00 | 155.0 | 155 | 145 | 164 | 132 | 180 | 48.20 | 7.08 |
| 1 | snapshot | 10,987 | 11.44 | 8.1 | 7 | 1 | 16 | 1 | 29 | 5.19 | 7.51 |
| 2 | learner | 960 | 1.00 | 153.9 | 154 | 144 | 164 | 125 | 181 | 53.16 | 7.55 |
| 2 | snapshot | 11,471 | 11.95 | 7.9 | 8 | 1 | 15 | 1 | 26 | 5.08 | 7.61 |
| 3 | learner | 960 | 1.00 | 156.4 | 156 | 148 | 166 | 133 | 179 | 47.53 | 6.92 |
| 3 | snapshot | 10,988 | 11.45 | 8.0 | 7 | 2 | 15 | 1 | 28 | 5.24 | 7.72 |

## Whole run (global fields, rank 0)

| | profiled window | unprofiled reference |
|---|---:|---:|
| updates | 870-884 | 865-869 |
| decisions/s (wall) | 3,113.59 | 3,280.27 |
| decisions/s (collect) | 4,477.55 | 4,733.84 |
| wall s/update | 21.05 | 19.98 |
| collect s (slowest rank) | 14.66 | 13.90 |
| learn s (slowest rank) | 7.82 | 7.77 |
| resident snapshots | 12-14 | 12-14 |
| mean prefix | 613.08 | 589.39 |
| GPU util % | 85.94 | 86.85 |
| GPU mem max MiB | 21,042.00 | 20,194.00 |
| GPU samples | 63 | 20 |

CUDA peak reserved per rank (GB): 5.24, 5.57, 5.55, 5.22; allocation retries max 0.
