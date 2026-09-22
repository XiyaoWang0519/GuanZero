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

## Revised next steps, September 21, 2026

Decision: the v2 belief result was measured on a two-layer, width-128
prototype, 4,096 self-play rounds and 6,000 updates, with all seeds still
improving. Self-play copies have no habits (DESIGN.md 7.3), so that data
cannot show cross-round opponent modelling at all. Before any v2 or v3 RL
integration, scale the probe and give it opponents with habits. The v2
integration row above is therefore deferred behind the tasks below.
Sample diversity is a first-class requirement: styles are continuous
parameters sampled per match, never a fixed list of bots, and every probe
result must hold on held-out style regions.

| # | Task | Status | Acceptance |
|---|---|---|---|
| 1 | Style-parameterised heuristic bot in `cpp/src/bots.cpp`: one bot driven by a continuous style vector (bomb-early threshold, per-type preference weights, follow aggressiveness, lead high/low bias, partner-cooperation weight, sampling temperature). Style sampled once per match from a configurable distribution; fields live in a `BotConfig`, none hardcoded. | implemented and tested | `StyleParams`, 18 slots, in `cpp/include/gd/bots.h`; neutral style matches `greedy_bot` on 12,000 decision points; four sweeps monotone (bomb timing 0.367 to 0.731 of the hand, follow rank 0.578 to 0.737, lead rank 0.253 to 0.539, pair leads 0.027 to 0.614); 140,004 UBSan styled fuzz rounds clean; VecEnv `styled_choice` costs 6% of rollout throughput |
| 2 | Style vector and match/round index recorded per decision in `eval/collect_belief.py` logs; opponent seats driven by task 1 bots; collection of at least 100,000 rounds, whole-match splits. Training styles drawn from a sub-region, a held-out style region reserved for test. | implemented and integrated with the real styled bot | Schema 2 round-trip, once-per-match resampling, driver separation, region disjointness and coverage-report tests; `eval/style_coverage.py` writes the coverage report. Integration smoke on main: 512 rounds with the M1 checkpoint and real styled opponents, no NaN styles, every match labelled train or heldout, 128 rounds/s on one thread, so 100,000 rounds is about 13 minutes. The 100,000-round collection itself is the first step of task 3 |
| 3 | Scaled belief probe in `train/belief_experiment.py`: four layers, width 256, trained to validation plateau; flat vs no-history vs history; loss broken down by round index within the match. Report to `reports/M2-belief-scaled.md`. | runner ready, collection running | Three seeds; held-out-style test matches; paired intervals |
| 4 | v3 match-memory probe: per-seat per-round summary vectors from the task 3 stream, private query attends to them; adaptation metric (DESIGN.md 9.1 item 6) on held-out styles; memory-masked control. | pending | Adaptation gain positive with positive lower bound on held-out styles, absent in the masked control |
| 5 | Decision gate: task 3 picks the Stage B state tower (v1 vs no-history v2 vs history v2); task 4 decides whether v3 enters the Stage B league plan. Update DESIGN.md. | pending | Written decision with numbers |
| 6 | Stage B, in the order critic, PPO, league. League includes task 1 bots with per-match styles, checkpoints at several temperatures and, later, a style-conditioned learner policy (style vector as network input, small style-shaping reward) as the learned source of diversity. | pending | Rows above; exploiter test compared with and without match memory if v3 is adopted |
| 7 | Minimal `play/` logging UI, pulled forward from M3 only as far as recording human games. Human data is calibration and test only, never training. | deferred, optional | Recruiting players is the hard part; revisit once tasks 1–3 give a baseline. Behaviour histograms from any human games calibrate the task 1 sampling distribution |

Task 3 runner status, September 21: `train/belief_experiment.py` now reports
the per-cell breakdowns (round bin within the match, style region, bot- versus
policy-driven predicted seat) with paired bootstrap intervals inside each cell,
records `stop_reason` and `best_validation_step` for every fit, and can cap
resident memory with `--decisions-per-round` and validation cost with
`--validation-decisions`. Measured on this host at width 256, four layers,
batch 64 and four CPU threads, one training step costs 0.0068 s (flat),
0.0170 s (no-history) and 0.1674 s (history), so all three models for one seed
cost 0.191 s per step: 1.1 h at 20,000 steps, 2.7 h at 50,000 and 5.3 h at
100,000, or 3.2 h, 8.0 h and 15.9 h for three seeds. Fifty thousand steps is an
overnight CPU run; a hundred thousand is where a rented GPU pays for itself.
The 100,000-round styled collection is still in progress, so no scaled result
has been measured yet.

No public human Guandan game dataset was found (search, Sept. 21). Scraping
online platforms is excluded by DESIGN.md 1.3 and by the platforms' terms.
Synthetic styled opponents are the training source; humans only calibrate.

Stage B's separate critic, PPO, league and exploiter remain pending. The A2
artifacts are not promoted into the frozen internal DMC reference.
