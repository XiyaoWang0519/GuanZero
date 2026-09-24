# M2 implementation and evidence tracker

Status: in progress, September 22, 2026. The external benchmark gates in
DESIGN.md are unchanged; the published agents remain unavailable. The frozen
M1 final checkpoint is the internal reference for the next experiments.

Source-policy audit correction resolved: the original task 3 collection used
the six-update local smoke checkpoint rather than the trained M1 GPU final
(34,496 updates). Its results remain historical evidence for that distribution
in [the scaled report](reports/M2-belief-scaled.md). Re-collection with the
verified final and the three-seed comparison are now complete. The corrected
evidence supports no_history for Stage B under this supervised protocol; it
does not establish playing strength. See
[the corrected report](reports/M2-belief-corrected.md).

| Component | Status | Acceptance evidence |
|---|---|---|
| Corrected scaled belief gate | complete; Stage B uses no_history under tested protocol | 100,000 styled rounds from the trained M1 final, three seeds: no_history improves flat by 1.6086%; history wins one seed and loses two, with negative pooled paired benefit; see [corrected report](reports/M2-belief-corrected.md) |
| Cross-round memory probe | complete; prototype not promoted | Overall memory benefit wins two seeds and loses one; all three paired adaptation intervals cross zero; original gate `v3_not_yet_justified`; see [memory report](reports/M2-memory.md) |
| Shared public history cache | implemented and tested on CPU | Incremental/full-prefix parity on 331 real decisions; private queries leave shared memory unchanged |
| History playing policy and sequence RL | deferred after scaled probe | Batched caches, round replay, policy-head integration and equal-compute playing-strength comparison remain |
| Stage A2 counterfactual exchange labels | complete for first experiment | 4,096 positions, 52,257 branches, 293 source matches; exact clones and actor's terminal return |
| Frozen tribute/back-tribute head fit | implemented and tested | Three seeds, 1,000 updates each; match-disjoint holdout; all other tensors exactly unchanged |
| Tribute payoff arena | implemented and tested | Same play model, paired tribute-start deals, independent seeds, explicit anti-tribute/choice diagnostics |
| Stage A2 empirical decision | complete: retain heuristic | All three means negative on the same 1,000 fresh paired deals; two 95% intervals exclude zero |
| Perfect-information critic and PPO | complete: G1, G2 passed | Critic sees hidden counts only in training; PPO beat continued DMC at equal compute; [critic](reports/stage-b-critic.md), [PPO](reports/stage-b-ppo.md) |
| Opponent league | complete: G3 passed as revised | Fixed opponents per match, learner-only trajectories; the active-model cap of 4 starved new entries (found in B11, now 16); [league](reports/stage-b-league.md) |
| Exploiter evaluation | complete: G4 passed (secondary at the margin); live exploiter G5 not passed | [exploiter](reports/stage-b-exploiter.md), [live exploiter](reports/stage-b-live-exploiter.md) |
| Original external M2 strength gate | unavailable | Cannot claim DanZero/SDMC thresholds without the named opponents |

The earlier A2 and prototype belief work ran locally against the downloaded
model. The original smoke-source Task 3 rental was deleted after artifact
retrieval; that historical run is recorded in `reports/M2-belief-scaled.md`.
The corrected Task 3 and Task 4 runs are complete. All 15 selected checkpoints
have verified SHA-256 identities and finite tensors; the 281,170,941-byte
immutable result archive is verified. The GPU rental lasted 2.706 hours, the
observed account balance decrease was $2.68559, pod deletion is confirmed, and
provider current spend is $0/hour. Evidence is in
[the completed run record](reports/M2-next-run.md). Earlier experiment results
remain in `reports/M2-A2.md`, `reports/M2-belief.md`, and their JSON summaries.
Earlier full-suite validation: **341 Python tests passed**. Current focused
belief/memory validation: **49 tests passed locally and on the GPU node**.
C++ is unchanged since the 58-case A2 validation. M2 remains in progress,
and no playing policy has switched to v2.

The public-known-holdings encoder was corrected before A2 collection: an
outgoing back-tribute removes a previously received card's public guarantee.
Tensor dimensions are unchanged, but affected observations differ from the
old pilot. Both sides of every new comparison use the corrected engine.

## Revised next steps, updated September 22, 2026

Decision: the v2 belief result was measured on a two-layer, width-128
prototype, 4,096 self-play rounds and 6,000 updates, with all seeds still
improving. Identical self-play copies do not provide controlled, stable
differences between opponent styles (DESIGN.md 7.3), so that data cannot isolate
cross-round adaptation to individual opponents. Before any v2 or v3 RL
integration, scale the probe and give it opponents with habits. The v2
integration row above is therefore deferred behind the tasks below.
Sample diversity is a first-class requirement: styles are continuous
parameters sampled per match, never a fixed list of bots, and every probe
result must hold on held-out style regions.

| # | Task | Status | Acceptance |
|---|---|---|---|
| 1 | Style-parameterised heuristic bot in `cpp/src/bots.cpp`: one bot driven by a continuous style vector (bomb-early threshold, per-type preference weights, follow aggressiveness, lead high/low bias, partner-cooperation weight, sampling temperature). Style sampled once per match from a configurable distribution; fields live in a `BotConfig`, none hardcoded. | implemented and tested | `StyleParams`, 18 slots, in `cpp/include/gd/bots.h`; neutral style matches `greedy_bot` on 12,000 decision points; four sweeps monotone (bomb timing 0.367 to 0.731 of the hand, follow rank 0.578 to 0.737, lead rank 0.253 to 0.539, pair leads 0.027 to 0.614); 140,004 UBSan styled fuzz rounds clean; VecEnv `styled_choice` costs 6% of rollout throughput |
| 2 | Style vector and match/round index recorded per decision in `eval/collect_belief.py` logs; opponent seats driven by task 1 bots; collection of at least 100,000 rounds, whole-match splits. Training styles drawn from a sub-region, a held-out style region reserved for test. | corrected collection complete with the trained M1 final | Schema 2 round-trip, once-per-match resampling, driver separation, region disjointness and coverage-report tests. `collect-100k-m1`: 100,000 rounds, 5,553,998 decisions and 18,384 recorded matches; verified source has 34,496 updates. The older smoke-source collection is retained separately |
| 3 | Scaled belief probe in `train/belief_experiment.py`: four layers, width 256, validation-selected within a 50,000-step cap; flat vs no-history vs history; loss broken down by round index within the match. Report to `reports/M2-belief-corrected.md`. | corrected rerun complete within step budget | Seven fits reached 50,000 steps; two stopped at 42,000 for validation plateau. Mean CE: flat 0.342105, no_history 0.336602, history 0.336681. Stage B uses no_history under this tested protocol; no convergence or playing-strength claim; [report](reports/M2-belief-corrected.md) |
| 4 | v3 match-memory probe: per-seat per-round summary vectors from the task 3 stream, private query attends to them; adaptation metric (DESIGN.md 9.1 item 6) on held-out styles; memory-masked control. | complete within bounded protocol | Seeds 41/42/43: overall memory benefit wins two and loses one; late benefit is positive, inconclusive, and negative respectively. All adaptation intervals cross zero; `v3_not_yet_justified`. Three of six fits reached the 50,000-step cap. Memory and masked control are parameter identical; 49 focused tests passed locally and on the target; [report](reports/M2-memory.md) |
| 5 | Decision gate: task 3 picks the Stage B state tower (v1 vs no-history v2 vs history v2); task 4 decides whether v3 enters the Stage B league plan. Update DESIGN.md. | complete for bounded experiments | Stage B continues with no_history; this v3 prototype is not promoted. Larger-scale memory and opponent-habit value are not ruled out; no playing-strength claim. DESIGN.md updated; original acceptance and external benchmark gates unchanged |
| 6 | Stage B, in the order critic, PPO, league. League includes task 1 bots with per-match styles, checkpoints at several temperatures and, later, a style-conditioned learner policy (style vector as network input, small style-shaping reward) as the learned source of diversity. | complete September 23 except the style-conditioned learner (not built); tracker [STAGE_B_TODO.md](STAGE_B_TODO.md) | G1 passed ([critic](reports/stage-b-critic.md)); G2 passed ([PPO](reports/stage-b-ppo.md)); G3 passed as revised ([league](reports/stage-b-league.md)); G4 passed, secondary at the margin ([exploiter](reports/stage-b-exploiter.md)); G5 not passed, no strength cost ([live exploiter](reports/stage-b-live-exploiter.md)). Strongest checkpoint: B11 main, +0.075 over the B8 league final. State tower stayed v1 per task 5 |
| 7 | Minimal `play/` logging UI, pulled forward from M3 only as far as recording human games. Human data is calibration and test only, never training. | deferred, optional | Recruiting players is the hard part; revisit once tasks 1–3 give a baseline. Behaviour histograms from any human games calibrate the task 1 sampling distribution |
| 8 | Stage C, added September 23 from the merged research review: DanLM external baseline under a `botzone` rules profile, top-32 pruning audit, learning-signal arms, endgame search over sampled hidden hands; history tower only if the baseline shows a gap. | pending; task breakdown in [STAGE_C_TODO.md](STAGE_C_TODO.md) | Gates G6 to G11 in `STAGE_C_TODO.md`; `DESIGN.md` v0.5 sections 7.4, 8.4, 8.5 and 9.2 |

Task 3 runner status, September 22: `train/belief_experiment.py` reports
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
The corrected 100,000-round collection and three-seed GPU probe are complete.
Mean test losses: flat 0.3421048375, no_history 0.3366016650, history
0.3366812724. no_history improves flat by 1.6086%; adding history wins seed 31
but loses seeds 32 and 33. The pooled paired no_history-minus-history loss is
-0.000081, with whole-match 95% interval [-0.000100, -0.000061], conditional
on these three fits. The original combined gate remains
`v2_not_yet_justified`, and Stage B uses no_history for this tested protocol.
Seven fits hit 50,000 steps and two stopped for plateau at 42,000; selected
weights are validation-best, and overall convergence is not established.
Excluding 35 possibly quota-truncated test matches in a post-hoc sensitivity
analysis leaves the conclusion unchanged, without changing the primary gate.
See [the corrected report](reports/M2-belief-corrected.md). Artifacts and
final runtime records live under `.work/runpod-next/`.

Historical smoke-source result: flat 0.400759, no_history 0.395368, history
0.395270; the extra history gain was 0.0249% with two wins and one loss.
All nine historical fits hit and selected 50,000 steps. Those weights and
reports remain under `.work/runpod-belief/`; its pod is deleted and observed
balance change was about $1.28. This distribution is kept separate from the
corrected run. See [the historical report](reports/M2-belief-scaled.md).

Task 4 runner status, September 22: `train/memory_experiment.py` completed
seeds 41/42/43 on the corrected collection, with a 50,000-update cap and
validation-based early stopping. Three of six fits reached the cap; the other
three stopped at 40,000 or 44,000 updates, so overall convergence is not
established. See [the memory report](reports/M2-memory.md). The
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
no late-round trend in the control. This original gate is unchanged; additive
evidence fields separately report positive late benefit, paired adaptation,
and the control's raw trend. The shared stream layers in the masked model
still train through BOS, so its style readout is not a frozen random-encoder
baseline. The command line is in `docs/TRAINING.md`.

The final mean CE is 0.336921326 for memory and 0.337168381 for the matched
masked control. Overall paired benefit wins two seeds and loses one. For
late rounds, seed 41 has a positive 95% interval, seed 42 crosses zero, and
seed 43 is negative. All three paired adaptation intervals cross zero. The
pooled overall benefit is +0.000253 [0.000219, 0.000288]; pooled late benefit
is positive too, but both are conditional on these fits and do not erase
optimizer-seed disagreement. Pooled adaptation is +0.000022
[-0.000130, +0.000173] over 862 matches present in both early and late cells.
Excluding the 35 possibly quota-truncated test matches in a post-hoc
sensitivity analysis leaves this conclusion unchanged.

The unchanged Task 4 gate is `v3_not_yet_justified`. The bounded Task 5
decision is complete: continue Stage B with no_history and do not promote
this memory prototype. Larger model/data scales and the value of opponent
habits remain open research questions. These supervised comparisons do not
establish playing strength. Task 4 sees prior-round public summaries plus
BOS, without the current-round token sequence; its contrast must not be
conflated with Task 3's current-round history test.

### Rule change Sept. 22: O1 tie and O4 reset aligned with the official rules

Owner approved, after `reports/rules-web-check.md` items 1 and 2. The `house`
profile now uses `tribute_tie = downstream` (a double tribute with equal
cards goes clockwise: the Banker's downstream seat pays the Banker and leads)
and `a_fail_limit = 0` (no three-failure reset at level A). The old values
remain available as `tribute_tie = upstream` and `a_fail_limit = 3`. The
`ogd` profile is unchanged.

Checkpoints trained before this change, M1 (`pilot/final.pt`) and B6, used
the old rules. The tie case only arises on a double win with equal tribute
cards: in a 20,000-round greedy self-play run it came up in 637 rounds
(3.2% of rounds, 9.2% of double tributes; greedy play double-wins far more
often than trained play). Dropping the reset shortens greedy-vs-greedy
matches from 10.55 to 10.26 rounds on average (2,000 matches each, longest
45 versus 23). Round-level play is unaffected apart from the tie seat;
match-level returns at A change. No retraining is required to keep using
those checkpoints, but evaluations under `house` now use the new rules.

### Data caveat found Sept. 22

The 100,000-round styled collection behind task 3 (`collect-100k`) drove the
policy seats with `.work/m1-full-model/latest.pt`, which is a six-update local
smoke model, not the trained M1 pilot checkpoint
(`.work/runpod/artifacts/pilot/final.pt`, 34,496 updates). That initial task 3
tower decision was provisional because it used near-random policy seats.
A re-collection with the trained checkpoint (`collect-100k-m1`, same
seeds and style space) completed on Sept. 22: 100,000 rounds, 18,384 matches,
5,553,998 decisions (the trained policy makes fewer, larger plays: about half
the decisions per round of the smoke run, and a bomb fraction of 10.3% versus
3.5%). Coverage report: no empty style bins, regions disjoint. The corrected
task 3 comparison is now complete and supports no_history within its tested
protocol; task 4 is also complete on the same corrected collection. The earlier
4,096-round probe in `reports/M2-belief.md` did use the trained checkpoint.

The completed corrected Task 3 / Task 4 rental is archived under
`.work/runpod-next/`. All 15 validation-selected checkpoints were checked for
SHA-256 identity and finite tensors; the immutable archive was downloaded and
verified before GPU deletion. Cleanup is confirmed and provider current spend
is $0/hour. See [the completed run record](reports/M2-next-run.md) for the
archive receipt, 2.706-hour runtime and $2.68559 observed balance decrease.

No public human Guandan game dataset was found (search, Sept. 21). Scraping
online platforms is excluded by DESIGN.md 1.3 and by the platforms' terms.
Synthetic styled opponents are the training source; humans only calibrate.

Stage B's critic, PPO, league and exploiter are complete (task 6). The A2
artifacts are not promoted into the frozen internal DMC reference.
