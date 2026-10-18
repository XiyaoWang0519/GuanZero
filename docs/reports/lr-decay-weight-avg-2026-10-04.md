# Learning-rate decay and weight averaging (October 3-4, 2026)

Status: done. Both arms ran one 8 h night on RTX 5090s; evaluation on the Mac.

## Why

u15094 (989.2M decisions, -1.558 against DanLM) was trained at a constant
learning rate of 3e-4 from random initialisation. The critic-noise
measurement of October 3 found that about 69% of the critic's residual is
action-sampling noise, so the learning signal is noisy, and two recent
references (Ataraxos; Fan and Farina, arXiv 2605.19235) decay the learning
rate and evaluate an exponential moving average of the weights. Two cheap
questions followed: does averaging recent checkpoints help without any
training, and does a learning-rate decay help on top of one more night?

## What

- **Weight averaging**: a uniform average of the actor and critic weights of
  four checkpoints, about 500 updates apart, ending at the endpoint.
- **Learning-rate decay** (`e2b8862`): `HistoryPPOConfig` `lr_final`,
  `lr_decay_start`, `lr_decay_updates`, settable on resume. Both optimizers'
  rates scale linearly from `lr` to `lr_final` between the absolute updates
  `lr_decay_start` and `lr_decay_start + lr_decay_updates`, then hold, so a
  restarted job lands on the same schedule. The factor is logged as
  `lr_scale`. Off by default; `tests/test_history_ppo.py` checks the schedule,
  the resume path and the validation.
- **Night A/B from u15094**, same seed and settings (full auxiliary heads,
  fast-lg2 mode), one machine each on Vast host 406325:
  `main` at constant 3e-4, and `decay` from 3e-4 to 3e-5 over updates
  15094-19594, then held at 3e-5.

The first launch failed at setup: GCC 13 on the pod rejected
`cpp/tests/test_search.cpp`, which used `std::invalid_argument` without
`<stdexcept>` (clang on macOS accepts it). Fixed in `535764f` and checked by a
full GCC 13 build and test run in Docker before relaunching; cost $0.06.

## Results

### Weight averaging of the u15094 lineage (no training)

Checkpoints u13888, u14384, u14880, u15094.

| Comparison | Deals | Result | 95% interval |
|---|---|---|---|
| average of 4 vs u15094, head to head | 2,000 paired | +0.098 | [+0.043, +0.154] |
| average of last 2 vs u15094 | 2,000 paired | +0.081 | [+0.025, +0.136] |
| average of 4 vs DanLM | 4,000 | -1.505 | [-1.542, -1.467] |

u15094 scored -1.558 [-1.594, -1.521] on the same 4,000 deals. The arena
output holds no per-deal scores, so there is no paired interval against
DanLM; the +0.053 difference is within the two intervals.

### The night (8.2 h each, $4.01 and $4.00)

| | main | decay |
|---|---|---|
| endpoint | u20264, 1,328.0M decisions | u20314, 1,331.3M decisions |
| final rate | 3e-4 | 3e-5 (reached at u19594, held for 720 updates) |
| entropy, last 100 updates | 0.42 | 0.33 |
| critic explained variance | 0.71 | 0.74 |
| approx KL / clip fraction | 0.0048 / 0.047 | 0.0006 / 0.005 |

Head to head, 2,000 paired deals (`eval.lineage_eval`):

| Comparison | Result | 95% interval |
|---|---|---|
| main vs u15094 | +0.243 | [+0.184, +0.298] |
| decay vs u15094 | +0.321 | [+0.264, +0.379] |
| **decay vs main** | **+0.052** | [+0.000, +0.109] |
| main, average of 4 vs main endpoint | +0.037 | [-0.015, +0.091] |
| decay, average of 4 vs decay endpoint | +0.001 | [-0.051, +0.053] |

The arm averages use u18724, u19220, u19716 and the endpoint.

Against DanLM, 4,000 deals, seed 20260929:

| Model | Levels per round | 95% interval | Round win rate |
|---|---|---|---|
| u15094 (start of the night) | -1.558 | [-1.594, -1.521] | |
| **main u20264** | **-1.356** | [-1.393, -1.317] | 24.1% |
| decay u20314 | -1.357 | [-1.392, -1.320] | 23.7% |
| main, average of 4 | -1.345 | [-1.382, -1.311] | 23.9% |
| decay, average of 4 | -1.337 | [-1.374, -1.300] | 24.1% |

## Reading

1. **One more night is the large effect.** Both arms gained about +0.20 levels
   per round against DanLM over u15094, the largest single-night gain so far
   (the previous night gained +0.17). The plain policy now equals u15094 with
   test-time search (-1.342 on 300 deals). The curve is still rising.
2. **The decay is a small polish.** Head to head it is +0.05 with the lower
   end of the interval at 0; against DanLM the two arms cannot be
   distinguished. Its entropy fell from 0.42 to 0.33 and its rate ends at
   3e-5, so it is a poor point to continue training from.
3. **Averaging helps less as the weights settle.** +0.10 on the constant-rate
   u15094 lineage, +0.04 (not significant) on main and 0 on the decayed arm,
   consistent with averaging smoothing oscillation that the decay removes.

## Decision

- The main lineage continues from **main u20264**:
  `.work/lr-decay-2026-10-03/kits/main/download/results/segments/lr-main/latest.pt`.
- Learning-rate decay and checkpoint averaging are kept for producing a
  release model at the end of a lineage, not for routine nights.
- Next experiment: the variance-reduction design
  ([vrpo-design-2026-10-04](vrpo-design-2026-10-04.md)).

## Sources (local, under `.work/`, not committed)

- `weight-avg-2026-10-03/{avg4,avg2}-vs-u15094.json`, `avg4-vs-danlm.json`
- `lr-decay-2026-10-03/kits/{main,decay}/` (kits, lifecycle logs, downloads)
- `lr-decay-2026-10-03/eval/run.sh` and its `*.json` outputs
