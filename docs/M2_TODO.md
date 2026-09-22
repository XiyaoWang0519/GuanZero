# M2 implementation and evidence tracker

Status: in progress, September 21, 2026. The external benchmark gates in
DESIGN.md are unchanged; the published agents remain unavailable. The frozen
M1 final checkpoint is the internal reference for the next experiments.

Source-policy audit correction: task 3's collection used the six-update local
smoke checkpoint rather than the trained M1 GPU final (34,496 updates).
The published losses remain measurements of that collected distribution;
repeat collection with the verified final model before treating the Stage B
tower choice as established. Evidence: `reports/M2-belief-scaled.md`.

| Component | Status | Acceptance evidence |
|---|---|---|
| Scaled belief gate | complete; Stage B choice provisional | 100,000 styled rounds, three seeds: no_history improves flat by 1.345%; extra history averages 0.0249% with two wins and one loss; see scaled report |
| Shared public history cache | implemented and tested on CPU | Incremental/full-prefix parity on 331 real decisions; private queries leave shared memory unchanged |
| History playing policy and sequence RL | deferred after scaled probe | Batched caches, round replay, policy-head integration and equal-compute playing-strength comparison remain |
| Stage A2 counterfactual exchange labels | complete for first experiment | 4,096 positions, 52,257 branches, 293 source matches; exact clones and actor's terminal return |
| Frozen tribute/back-tribute head fit | implemented and tested | Three seeds, 1,000 updates each; match-disjoint holdout; all other tensors exactly unchanged |
| Tribute payoff arena | implemented and tested | Same play model, paired tribute-start deals, independent seeds, explicit anti-tribute/choice diagnostics |
| Stage A2 empirical decision | complete: retain heuristic | All three means negative on the same 1,000 fresh paired deals; two 95% intervals exclude zero |
| Perfect-information critic and PPO | pending | Must preserve private critic/actor separation and establish an equal-compute comparison |
| Opponent league | pending | Fixed opponents per match, bounded active models, learner-only trajectories |
| Exploiter evaluation | pending | Held-out adversarial style and regression checks |
| Original external M2 strength gate | unavailable | Cannot claim DanZero/SDMC thresholds without the named opponents |

The earlier A2 and prototype belief work ran locally against the downloaded
model. Task 3 completed on a separate bounded RunPod rental, now deleted, recorded in
`reports/M2-belief-scaled.md`. Earlier experiment results are recorded in
`reports/M2-A2.md`, `reports/M2-belief.md`, and their JSON summaries. Earlier full-suite
validation: **341 Python tests passed**. Task 3 target-node focused validation: **38 tests passed**. C++ is unchanged since the 58-case A2
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
| 2 | Style vector and match/round index recorded per decision in `eval/collect_belief.py` logs; opponent seats driven by task 1 bots; collection of at least 100,000 rounds, whole-match splits. Training styles drawn from a sub-region, a held-out style region reserved for test. | implemented and integrated with the real styled bot | Schema 2 round-trip, once-per-match resampling, driver separation, region disjointness and coverage-report tests; `eval/style_coverage.py` writes the coverage report. Integration smoke on main: 512 rounds with the M1 checkpoint and real styled opponents, no NaN styles, every match labelled train or heldout, 128 rounds/s on one thread, so 100,000 rounds is about 13 minutes. Full collection complete: 100,000 rounds and 10,145,726 decisions |
| 3 | Scaled belief probe in `train/belief_experiment.py`: four layers, width 256, validation-selected within a 50,000-step cap; flat vs no-history vs history; loss broken down by round index within the match. Report to `reports/M2-belief-scaled.md`. | complete within step budget | All nine fits selected step 50,000, none established a plateau; held-out evaluation, local checkpoint verification and GPU deletion complete; see `reports/M2-belief-scaled.md` |
| 4 | v3 match-memory probe: per-seat per-round summary vectors from the task 3 stream, private query attends to them; adaptation metric (DESIGN.md 9.1 item 6) on held-out styles; memory-masked control. | implemented, GPU run pending | `train/belief_memory.py` and `train/memory_experiment.py`; memory and masked control parameter identical and within 0.066% of the scaled no_history tower (4,665,318 against 4,662,246); `tests/test_memory_experiment.py`, 7 tests, covering cross-round causality both ways, public-only summaries, parameter matching, the adaptation arithmetic and a two-seed end-to-end run on held-out styles; 45 tests pass over the three belief files. Measured on this host at width 256, four layers, batch 64, four threads, 200 real matches: 0.573 s/step for `memory` and 0.0186 s/step for `memory_masked`, so one seed costs 8.2 h at 50,000 steps on CPU. No result yet |
| 5 | Decision gate: task 3 picks the Stage B state tower (v1 vs no-history v2 vs history v2); task 4 decides whether v3 enters the Stage B league plan. Update DESIGN.md. | Stage B tower provisional after source audit; v3 pending | no_history selected from task 3; cross-round memory remains a separate controlled experiment; DESIGN.md updated |
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
The 100,000-round styled collection is complete. The scaled three-seed GPU
probe is complete. Mean test losses: flat 0.400759, no_history 0.395368,
history 0.395270. History adds only 0.0249% on average and loses seed 33;
the combined history gate fails. Stage B uses no_history. No consistent
history-specific gain growth with round index was found; v3 remains separate.
All nine fits hit the 50,000-step cap, so convergence remains unestablished.
All selected weights and reports are saved under `.work/runpod-belief/`;
see `reports/M2-belief-scaled.md`. The pod is deleted; observed balance
change was about $1.28.

Task 4 runner status, September 21: `train/memory_experiment.py` is
implemented and tested, and no fit has been run on the full collection. The
memory summary is public-stream only: the schema-2 logs record nothing at a
round end, so the revealed remaining cards that DESIGN 7.3 would also pool do
not exist in the data and are omitted. The control is the same network with the
memory keys absent from the cross-attention, which makes it functionally the
no_history tower at an identical parameter count. Adaptation is reported as the
paired masked-minus-memory improvement in rounds 5 and later minus the earlier
rounds of the same matches, with a bootstrap interval over whole matches, per
round-index bin, per exact round index and as a slope; a style readout fits a
ridge from the finished-round summaries of bot-driven seats to that seat's
style vector and reports held-out R^2 per slot for both models. The gate needs
a positive adaptation improvement with a positive lower bound in every seed and
no late-round trend in the control. The CPU cost puts the three-seed run on a
GPU; the command line is in `docs/TRAINING.md`.

### Data caveat found Sept. 22

The 100,000-round styled collection behind task 3 (`collect-100k`) drove the
policy seats with `.work/m1-full-model/latest.pt`, which is a six-update local
smoke model, not the trained M1 pilot checkpoint
(`.work/runpod/artifacts/pilot/final.pt`, 34,496 updates). The task 3 tower
decision is therefore provisional: it was measured with near-random policy
seats. A re-collection with the trained checkpoint (`collect-100k-m1`, same
seeds and style space) is in progress; task 4 runs on it, and task 3 is to be
re-run on it before the Stage B tower choice is treated as final. The earlier
4,096-round probe in `reports/M2-belief.md` did use the trained checkpoint.

The task 4 GPU run is prepared under `.work/runpod-memory/` but not started:
`runpodctl` 2.14.0 has no provider-side auto-termination flag, so an
unattended pod cannot be cost-capped from the CLI alone. Options: run only
while the operator's machine stays online, or have the pod delete itself at
the end of its launch script through the credential RunPod injects into pods
(to be verified at smoke time), with a `timeout` on every training process.

No public human Guandan game dataset was found (search, Sept. 21). Scraping
online platforms is excluded by DESIGN.md 1.3 and by the platforms' terms.
Synthetic styled opponents are the training source; humans only calibrate.

Stage B's separate critic, PPO, league and exploiter remain pending. The A2
artifacts are not promoted into the frozen internal DMC reference.
