# M1 readiness tracker

The first v1 DMC training run is complete and the larger Stage A run is ready.
This is an implementation gate, separate from the external playing-strength
gate in DESIGN.md section 1.2.

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
| CPU preflight, regression tests and updated operator guide | done | `reports/M1-preflight.md`, `TRAINING.md` |

GPU validation on the chosen node:

1. CUDA preflight passed with full v1 model, bf16 updates and resume; the
   4,096-environment pilot completed 34,496 updates and 71.27M decisions.
2. Real GPU SIGTERM and two resumes passed. Five selected checkpoints and
   1,787 evidence files were checksum-verified locally before pod deletion.
   The node existed for 36.31 minutes, with a conservative $0.603 cost estimate.
3. Final v1 won 100/100 held-out matches against greedy and beat the five-minute
   checkpoint directly. Two play probes still fail. The small three-seed belief
   experiment favors the flat model; v2 RL remains gated.

The original external M1 strength gate remains unmeasured because the named
baseline agents are unavailable. Internal results do not replace it. v2 RL, learned tribute, PPO,
league training, exploiter tests, search and human UI remain later milestones,
with the evidence gates and external-baseline limitation stated in DESIGN.md.

Work toward M2 is tracked in `M2_TODO.md`; the first component is the frozen-play
Stage A2 tribute experiment in `reports/M2-A2.md`.
