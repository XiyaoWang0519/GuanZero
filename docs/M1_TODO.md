# M1 historical readiness record

The September 21 v1 MLP pilot and its implementation checks are historical
evidence. Current training follows [STAGE_C_TODO.md](STAGE_C_TODO.md): a new
history Transformer trained from random initialization. This page does not
schedule a larger MLP run or gate Transformer RL on a belief experiment.

RunPod validation and a 20.5-minute pilot completed under the approved
$5/two-hour cap. Artifacts are downloaded and verified; the pod is deleted.
See `reports/M1-runpod.md` for measurements, costs and remaining research gates.

| Prerequisite | Status | Evidence |
|---|---|---|
| Attribute round returns to the correct trajectory | done | Environment/match/round identifiers; parallel-env tests |
| Private auxiliary labels isolated from actor inputs | done | Hidden-count and observation-invariance tests |
| v1 two-tower model, phase heads, hidden-hand and finish heads | done | Ragged vs individual scores/gradients; full 3,078,125-parameter CPU smoke |
| DMC self-play with greedy warm start, exploration and heuristic tribute | done | Actual updates across full matches; only network play samples train |
| Bounded replay and owned copies of engine buffers | done | Reward/order, wraparound and view-lifetime tests |
| Checkpoint, optimizer/RNG recovery, graceful shutdown and snapshots | done | Atomic failure, real SIGTERM/resume, snapshot immutability tests |
| Duplicate arena, match intervals, Elo and behavior probes | done | Determinism/statistics tests and real checkpoint evaluation |
| Public history logs and matched-parameter belief experiment | done | Match-separated splits, causal-mask tests and real logged-data smoke |
| Launch image/scripts, optional durable sync, budget/time watchdog | done | Official CUDA image/bootstrap verified; artifacts retrieved; owned pod deleted and provider readback confirmed |
| CPU preflight, regression tests and updated operator guide | done | `reports/M1-preflight.md` (legacy MLP preflight) |

GPU validation on the chosen node:

1. CUDA preflight passed with full v1 model, bf16 updates and resume; the
   4,096-environment pilot completed 34,496 updates and 71.27M decisions.
2. Real GPU SIGTERM and two resumes passed. Five selected checkpoints and
   1,787 evidence files were checksum-verified locally before pod deletion.
   The node existed for 36.31 minutes, with a conservative $0.603 cost estimate.
3. Final v1 won 100/100 held-out matches against greedy and beat the five-minute
   checkpoint directly. Two play probes still failed in that pilot. Historical
   belief experiments were subsequently corrected; see [M2 evidence](M2_TODO.md).

The original external M1 strength gate was not measured because the named
baseline agents were unavailable; internal results do not replace it.
Later A2, critic, PPO and league outcomes are in [M2_TODO.md](M2_TODO.md) and
[STAGE_B_TODO.md](STAGE_B_TODO.md). Their old pretrained-player route has been
replaced by [the current design](DESIGN.md).
