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
| B0 | **Baseline harness.** One script, `eval/stage_b_baseline.py`, that plays the M1 final against greedy, against four fixed styled bots (bomb-happy, bomb-shy, high-lead, low-lead), and against itself, in 10,000 duplicate deals and 1,000 matches each, and writes `reports/stage-b-baseline.json`. Also records rollout throughput of the current DMC loop on this host. | none | JSON with means and intervals for every pair; runtime under one hour on CPU or the script says why not | pending |
| B1 | **Critic dataset.** Collect M1 self-play rounds with, per decision, the actor observation, the three `hidden_counts` rows, the phase, the seat, and the Monte Carlo round return for the actor's team (DESIGN 8.2 item 1). Whole-round grouping, match-disjoint train/val/test split, schema version recorded. Reuse `eval/collect_belief.py` conventions. | none | at least 200,000 rounds; split leakage test; a decision count per stage bin | done: `eval/collect_critic.py` (schema 1, sharded per split, `--workers` for multi-process), tests in `tests/test_critic_collection.py`; 200,000 rounds, 10,738,209 decisions, 20,380 matches from `final.pt` (34,496 updates) in 110 s on this Mac with 4 workers, data in `.work/critic-m1/` (564 MB, not committed); coverage in `docs/reports/stage-b-critic-data.json` |
| B2 | **Critic network and offline fit.** `train/critic.py`: input is the actor observation concatenated with the three hidden-count rows (or all four hands when that is cheaper to encode; pick one and say why). Output is the expected round return. MLP sized like the v1 tower. Loss MSE. Report G1 with the M1 play head's best-Q MSE as the comparison on the same test rounds. | B1 | G1 report in `reports/stage-b-critic.md`; unit tests for input assembly, no leakage of hidden counts into anything the policy sees | pending |
| B3 | **On-policy round buffer.** Replace `ReplayBuffer` use for Stage B with a buffer that stores whole rounds from learner seats only: observation, candidate set, chosen index, behaviour log-prob, critic input, reward at round end, done. Discount 1, GAE with configurable lambda. Trivially clearable between iterations. | none | tests: advantage equals return minus value when lambda is 1; padded rounds mask correctly; no heap growth across iterations | pending |
| B4 | **Policy head and candidate pruning.** Add a `policy_logits` path to `GuandanModel` that is `Q / temperature` at initialization, plus the frozen M1 top-`k` pruning (`k` around 32) with pass always kept. Sampling and log-prob over the pruned set. Play-time interface unchanged so `eval/policies.py` loads either kind. | none | tests: at temperature `t` and no training the sampled policy matches softmax of M1 Q; pruned set always contains pass; `load_policy` round-trips | pending |
| B5 | **PPO learner.** `train/ppo.py`: rollout with the B3 buffer, clipped surrogate, GAE from the B2 critic (trained jointly from here on), entropy bonus, KL penalty toward the frozen M1 policy annealed to zero, auxiliary losses kept, tribute and back-tribute steps as ordinary trajectory steps with heads started from A2 (heuristic remains default per the A2 decision; the learned heads are optional). `PPOConfig` with every knob a field. Checkpoint and resume, same layout as `dmc.py`. | B2, B3, B4 | preflight on CPU: 4 updates, checkpoint, resume; metrics: clip fraction, approx KL, explained variance, entropy | pending |
| B6 | **Equal-compute PPO versus DMC run.** Single opponent (the M1 final on the other team). One GPU rental: PPO from M1 for budget `T`, DMC continued from M1 for the same `T`, same seeds. Evaluate both with B0. | B0, B5 | G2 in `reports/stage-b-ppo.md`; cost and cleanup evidence as in `M2-next-run.md` | pending |
| B7 | **Opponent pool and sampler.** `train/league.py`: pool entries are `load_policy` specs plus styled-bot styles; each match draws one opponent for the whole match; sampling weight proportional to that opponent's recent win rate against the learner with a floor; at most four distinct network opponents active per rollout so inference batches; learner controls both seats of one team and only learner seats emit samples. Snapshot of the learner is added every `N` updates. | B4 | tests: one style per match, active-model cap, learner-only sampling, weight update is bounded and deterministic under seed | pending |
| B8 | **League training run.** B5 with the B7 pool: latest and older checkpoints at two temperatures, greedy, styled bots. Same budget as B6. Evaluate against every pool member and the M1 final. | B6, B7 | G3 in `reports/stage-b-league.md`; Elo table refreshed with `eval/elo.py` | pending |
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
