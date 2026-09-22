# M2 implementation and evidence tracker

Status: in progress, September 21, 2026. The external benchmark gates in
DESIGN.md are unchanged; the published agents remain unavailable. The frozen
M1 final checkpoint is the internal reference for the next experiments.

| Component | Status | Acceptance evidence |
|---|---|---|
| v2 belief gate | passed for RL experimentation | Three seeds improve flat loss by 9.1–11.1%; history adds 0.21–0.43% over no-history, with positive paired intervals |
| Shared public history cache | implemented and tested on CPU | Incremental/full-prefix parity on 331 real decisions; private queries leave shared memory unchanged |
| v2 playing policy and sequence RL | next implementation target | Batched caches, round replay, policy-head integration and equal-compute playing-strength comparison remain |
| Stage A2 counterfactual exchange labels | complete for first experiment | 4,096 positions, 52,257 branches, 293 source matches; exact clones and actor's terminal return |
| Frozen tribute/back-tribute head fit | implemented and tested | Three seeds, 1,000 updates each; match-disjoint holdout; all other tensors exactly unchanged |
| Tribute payoff arena | implemented and tested | Same play model, paired tribute-start deals, independent seeds, explicit anti-tribute/choice diagnostics |
| Stage A2 empirical decision | complete: retain heuristic | All three means negative on the same 1,000 fresh paired deals; two 95% intervals exclude zero |
| Perfect-information critic and PPO | pending | Must preserve private critic/actor separation and establish an equal-compute comparison |
| Opponent league | pending | Fixed opponents per match, bounded active models, learner-only trajectories |
| Exploiter evaluation | pending | Held-out adversarial style and regression checks |
| Original external M2 strength gate | unavailable | Cannot claim DanZero/SDMC thresholds without the named opponents |

The current work runs locally against the downloaded model. No new paid node
has been provisioned. Progress and experiment results are recorded in
`reports/M2-A2.md`, `reports/M2-belief.md`, and their JSON summaries. Current
validation: **341 Python tests pass**. C++ is unchanged since the 58-case A2
validation. A2 and the larger belief experiment are complete; M2 remains in
progress, and no playing policy has switched to v2.

The public-known-holdings encoder was corrected before A2 collection: an
outgoing back-tribute removes a previously received card's public guarantee.
Tensor dimensions are unchanged, but affected observations differ from the
old pilot. Both sides of every new comparison use the corrected engine.

Next, integrate v2 into the playing policy and DMC loop: whole-round replay,
causal training masks, batched per-environment public caches, and rebuilding
caches when published weights change. Keep the improved no-history architecture
as a control because most belief improvement came from the private pathway.
Measure playing strength at equal compute before promotion. Stage B's separate
critic, PPO, league and exploiter remain pending. The A2 artifacts are not
promoted into the frozen internal DMC reference.
