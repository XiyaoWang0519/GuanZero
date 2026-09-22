# Stage B implementation tracker: critic, PPO, league

Status: drafted September 22, 2026. Nothing below is implemented. This is M2
task 6 in `M2_TODO.md`, expanded so that independent sessions can each own
one task. Design source: `DESIGN.md` 8.4 (Stage B), 9.1 (arena), 9.3 and
9.4 (exploiter test). Rules of the house in `CLAUDE.md` still apply.

## Decisions carried in from M2 tasks 3 to 5

- **State tower stays v1.** Task 5 chose `no_history`. In
  `train/belief_model.py` that model is the v2 query block attending only to
  its BOS token, so it uses no public stream and no KV cache. For the playing
  policy the operational reading is: keep the `GuandanModel` MLP state tower,
  initialize from the M1 final, and do not build round-sequence RL now.
- **No v3 memory in Stage B.** The league still holds styled opponents with
  one fixed style per match, because that is the setting where memory is
  re-tested later.
- **Reference policy** is the M1 final,
  `.work/runpod/artifacts/pilot/final.pt` (34,496 updates, digest in
  `M2_TODO.md`). Never the six-update smoke model.
- **Equal-compute rule.** Every strength claim compares against a baseline
  given the same wall-clock or update budget on the same hardware.

## What already exists and is reused

| Piece | Where | Reuse |
|---|---|---|
| Two-tower model, `hidden_head`, `finish_head`, `score_candidates`, `select_actions` | `train/model.py` | policy network; play head scalar becomes the logit |
| DMC trainer, `TrainConfig`, rollout/learn alternation, checkpoint, tensorboard | `train/dmc.py` | template for the PPO loop; do not fork, factor shared parts |
| `Decision`, `ReplayBuffer` | `train/buffer.py` | replaced by an on-policy round buffer in Stage B |
| Observation with `hidden_counts` `[rows, 3, 54]` | `cpp/include/gd/env.h`, `encode_observation` | the critic's perfect-information input comes from here |
| `VecEnv::set_styles`, `styled_choice` | `cpp/include/gd/env.h`, `train/styles.py` | styled league opponents |
| Match arena, Wilson interval | `eval/arena.py` | match win rate gate |
| Duplicate deals, bootstrap CI | `eval/duplicate.py` | level-gain gate |
| Elo refresh | `eval/elo.py` | checkpoint ladder |
| `load_policy` spec strings | `eval/policies.py` | opponent pool entries |

## Gates for Stage B

1. **G1, critic quality.** On held-out M1 self-play rounds the critic's
   return MSE is below the MSE of the M1 play head's best `Q`, overall and in
   each of early, middle and late stage bins. Reported with a bootstrap CI
   over rounds.
2. **G2, PPO beats its own start.** The PPO checkpoint beats the M1 final in
   duplicate deals with a 95% interval above zero level gain, and in match
   play with a Wilson interval above 50%, at equal compute versus a DMC run
   continued from the same checkpoint for the same budget.
3. **G3, league does not regress.** The league-trained checkpoint is not
   worse than the single-opponent PPO checkpoint against the M1 final, and is
   better against the styled bot suite and the greedy bot.
4. **G4, exploitability.** An exploiter trained for a fixed budget against
   the frozen league checkpoint wins less than one trained against the
   frozen M1 final for the same budget.

A gate that fails is reported, not tuned around. The external DanZero and
SDMC thresholds stay unavailable and are not claimed.

## Tasks

Each row is sized for one session. "Depends" names the row that must be
merged first. Rows with no dependency can start now in parallel.

| # | Task | Depends | Acceptance | Status |
|---|---|---|---|---|
| B0 | **Baseline harness.** One script, `eval/stage_b_baseline.py`, that plays the M1 final against greedy, against four fixed styled bots (bomb-happy, bomb-shy, high-lead, low-lead), and against itself, in 10,000 duplicate deals and 1,000 matches each, and writes `reports/stage-b-baseline.json`. Also records rollout throughput of the current DMC loop on this host. | none | JSON with means and intervals for every pair; runtime under one hour on CPU or the script says why not | done: full run in `reports/stage-b-baseline.json`, 10,000 duplicate deals and 1,000 matches per pair, 8 min on 10 local workers. M1 final net levels/round: greedy +1.516, bomb-happy +0.782, bomb-shy +1.716, high-lead +2.738, low-lead +2.272; match wins 995, 902, 998, 1000, 1000 of 1000; self 530/1000, Wilson interval includes 50% |
| B1 | **Critic dataset.** Collect M1 self-play rounds with, per decision, the actor observation, the three `hidden_counts` rows, the phase, the seat, and the Monte Carlo round return for the actor's team (DESIGN 8.2 item 1). Whole-round grouping, match-disjoint train/val/test split, schema version recorded. Reuse `eval/collect_belief.py` conventions. | none | at least 200,000 rounds; split leakage test; a decision count per stage bin | done: `eval/collect_critic.py` (schema 1, sharded per split, `--workers` for multi-process), tests in `tests/test_critic_collection.py`; 200,000 rounds, 10,738,209 decisions, 20,380 matches from `final.pt` (34,496 updates) in 110 s on this Mac with 4 workers, data in `.work/critic-m1/` (564 MB, not committed); coverage in `docs/reports/stage-b-critic-data.json` |
| B2 | **Critic network and offline fit.** `train/critic.py`: input is the actor observation concatenated with the three hidden-count rows (or all four hands when that is cheaper to encode; pick one and say why). Output is the expected round return. MLP sized like the v1 tower. Loss MSE. Report G1 with the M1 play head's best-Q MSE as the comparison on the same test rounds. | B1 | G1 report in `reports/stage-b-critic.md`; unit tests for input assembly, no leakage of hidden counts into anything the policy sees | done, **G1 passed**: `train/critic.py` (input = observation + 3 hidden rows, byte-identical to `RolloutBuffer.critic_input`; 4 x 512 MLP; streamed shards, val early stopping, `fit`/`report` CLI), 10 tests in `tests/test_critic.py`. Test split (20,449 rounds, 1.05 M play decisions): critic MSE 2.393 vs M1 max-Q 2.789, difference −0.396 [−0.421, −0.373] with a round bootstrap; early −0.362, middle −0.464, late −0.367, all upper bounds < 0. Public-only ablation 2.885, so perfect information is worth 0.49 MSE. Fits 6.5 min on CPU. The data is on-policy for greedy M1 only. Report in `reports/stage-b-critic.md` and `.json`, weights in `.work/critic-fit/` |
| B3 | **On-policy round buffer.** Replace `ReplayBuffer` use for Stage B with a buffer that stores whole rounds from learner seats only: observation, candidate set, chosen index, behaviour log-prob, critic input, reward at round end, done. Discount 1, GAE with configurable lambda. Trivially clearable between iterations. | none | tests: advantage equals return minus value when lambda is 1; padded rounds mask correctly; no heap growth across iterations | done: `train/rollout_buffer.py` (`RolloutBuffer`, `compute_gae`), 10 tests in `tests/test_rollout_buffer.py` incl. real-`VecEnv` learner-only filter, carry-over and drop modes, array identity/nbytes unchanged over 200 iterations |
| B4 | **Policy head and candidate pruning.** Add a `policy_logits` path to `GuandanModel` that is `Q / temperature` at initialization, plus the frozen M1 top-`k` pruning (`k` around 32) with pass always kept. Sampling and log-prob over the pruned set. Play-time interface unchanged so `eval/policies.py` loads either kind. | none | tests: at temperature `t` and no training the sampled policy matches softmax of M1 Q; pruned set always contains pass; `load_policy` round-trips | done: `train/policy.py` (`PolicyConfig`, `StageBPolicy.act/evaluate/prune`, checkpoint payload `stage="ppo"`), `PrunedPolicy` and `sample[=T]:<path>` specs in `eval/policies.py`; `tests/test_stage_b_policy.py` (9 tests, incl. exact `logits == Q/t` from the M1 final); full `pytest tests` 397 passed |
| B5 | **PPO learner.** `train/ppo.py`: rollout with the B3 buffer, clipped surrogate, GAE from the B2 critic (trained jointly from here on), entropy bonus, KL penalty toward the frozen M1 policy annealed to zero, auxiliary losses kept, tribute and back-tribute steps as ordinary trajectory steps with heads started from A2 (heuristic remains default per the A2 decision; the learned heads are optional). `PPOConfig` with every knob a field. Checkpoint and resume, same layout as `dmc.py`. | B2, B3, B4 | preflight on CPU: 4 updates, checkpoint, resume; metrics: clip fraction, approx KL, explained variance, entropy | done: `train/ppo.py` (`PPOConfig`, `PPOTrainer`), `FrozenModelOpponent` in `train/opponents.py`, configs `ppo-smoke.json` and `ppo.json` (GPU draft, not run), `scripts/preflight.sh ppo` (2 updates, checkpoint, resume to 4, loads via `load_policy`), 10 tests in `tests/test_ppo.py`; full `pytest tests` 446 passed. Buffer stores the pruned set, so the learner re-scores exactly what was sampled from. Default temperature 0.02: sampled M1 vs argmax M1 over 400 duplicate deals is -2.72 at T=1, -2.12 at 0.3, -0.94 at 0.1, -0.33 at 0.05, -0.15 at 0.03, -0.11 [-0.22, +0.01] at 0.02 (entropy 0.33 nats). Sanity run on this Mac, 12 min, 16 envs, T=0.03, vs frozen M1: 3,728 updates, 3.8 M decisions, 5.3 k decisions/s, 5.2 updates/s; afterwards the argmax checkpoint scores +0.42 [+0.35, +0.50] levels/round over 1,000 duplicate deals vs the M1 final and 139/200 matches, +1.61 vs greedy. This is best-response to a fixed opponent after 12 minutes, not a G2 claim Independent re-check on main, seed 777001: sanity checkpoint vs M1 final +0.353 [+0.317, +0.388] over 5,000 duplicate deals, 359/500 matches; step-0 control exactly 0; vs bomb-happy +0.965 (M1 baseline +0.782), vs greedy +1.646 (M1 +1.516). Artifacts in `.work/ppo-sanity/`. Not G2: no equal-compute DMC arm yet. |
| B6 | **Equal-compute PPO versus DMC run.** Single opponent (the M1 final on the other team). One GPU rental: PPO from M1 for budget `T`, DMC continued from M1 for the same `T`, same seeds. Evaluate both with B0. | B0, B5 | G2 in `reports/stage-b-ppo.md`; cost and cleanup evidence as in `M2-next-run.md` | pending |
| B7 | **Opponent pool and sampler.** `train/league.py`: pool entries are `load_policy` specs plus styled-bot styles; each match draws one opponent for the whole match; sampling weight proportional to that opponent's recent win rate against the learner with a floor; at most four distinct network opponents active per rollout so inference batches; learner controls both seats of one team and only learner seats emit samples. Snapshot of the learner is added every `N` updates. | B4 | tests: one style per match, active-model cap, learner-only sampling, weight update is bounded and deterministic under seed | done: `train/league.py` (`League`, `LeagueConfig`, JSON pool file, lazy LRU model cache, `add_snapshot`/`should_snapshot`, `stats()`), 11 tests in `tests/test_league.py`. Needed an engine fix first: a match can restart inside one `pending()`, so `set_styles`/`clear_styles` now recompute the pending `styled_choice` in place (`tests/test_env_styles.py`, `cpp/tests/test_env_styles.cpp`) and the `OpponentSource` contract requires `on_match_start` before reading `styled_choice`. Reference driver and smoke `eval/league_smoke.py`: frozen M1 final vs the league, 2,048 matches (16 per env, uniform draw) in 28 s on CPU, learner wins greedy 288/292, bomb-happy 287/312 (0.920), bomb-shy 309/310, high-lead 256/256, low-lead 313/313, sampled styles 282/282, M1 at T=1 283/283; bomb-happy is the hardest, as in B0. `docs/reports/stage-b-league-smoke.json` |
| B8 | **League training run.** B5 with the B7 pool: latest and older checkpoints at two temperatures, greedy, styled bots. Same budget as B6. Evaluate against every pool member and the M1 final. | B6, B7 | G3 in `reports/stage-b-league.md`; Elo table refreshed with `eval/elo.py`; the league checkpoint is also evaluated for partner compatibility with `eval/crossplay.py` (B8x) | pending |
| B8x | **Cross-play (partner compatibility).** `eval/crossplay.py`: per-seat lineups, a team is (X at seat s, partner P at s+2), in the same duplicate deals. Against a fixed reference team (default two copies of the M1 final) report X+P, X+X and P+P mean net levels per round, `vs_self` = (X+P) − (X+X) and `partner_lift` = (X+P) − (P+P) with paired deal bootstrap CIs, and double-win rates. Partners: greedy, the four fixed styled bots, the M1 final, plus `--extra-partners` specs. | none | tests: partner really at s+2 in both legs, X+X equals `eval/duplicate.py`, results independent of `--workers`, CLI smoke | done: `eval/crossplay.py`, `play_round_seats`/`play_duplicate_teams` in `eval/duplicate.py`, 4 tests in `tests/test_crossplay.py`. Smoke, 200 deals, 6 workers, 47 s for 13 lineups, X = `.work/ppo-sanity/latest.pt`: X+X +0.44 vs M1; with M1 as partner +0.36, `vs_self` −0.08 [−0.27, +0.12]; with greedy −0.55, `vs_self` −0.99 [−1.23, −0.73], `partner_lift` +0.84. Smoke only, not a finding; full run is 5,000 deals |
| B9 | **Exploiter test.** `train/exploiter.py` or a mode of B5: freeze a target checkpoint, train a fresh PPO agent against it only, fixed budget, report win rate. Run against the M1 final and the B8 checkpoint. | B5, B8 | G4 in `reports/stage-b-exploiter.md` | pending |
| B10 | **Docs and gate update.** Update `DESIGN.md` 8.4 with what was actually built, `M2_TODO.md` task 6 status, `TRAINING.md` with the Stage B commands, and the milestone table. | B6, B8, B9 | one report per gate linked from `M2_TODO.md` | pending |

## Parallel start

Sessions that can begin immediately: B0, B1, B3, B4. B2 follows B1 within a
day. B7 follows B4. B5 waits for B2, B3, B4. The initial session verifies the
merged result, runs `./scripts/check.sh` and owns B6 onward, because those
tasks spend money.

## Not in scope

- v2 round-history tower in the playing policy, KV cache in rollout,
  sequence RL. Deferred per M2 task 5.
- v3 match memory. Re-test after B8 with a shuffled-opponent-history
  control and match-level metrics, not belief CE.
- Endgame search, human play UI, OpenGuanDan parity beyond what exists.
