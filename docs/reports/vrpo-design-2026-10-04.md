# VRPO / Q-boosting for the history trainer: design, October 4, 2026

Status: design only. Nothing here is implemented in the trainer, and nothing
has been trained. The estimator core and its tests are the first code
(`train/q_boost.py`, `tests/test_q_boost.py`). This note replaces the earlier
chat-only list of ideas; where this note and that list disagree, this note
is the corrected one (section 6).

Source: Fan and Farina, "GAE Falls Short in Imperfect-Information Self-Play
Reinforcement Learning", arXiv 2605.19235 (read in full on Oct 4 2026,
including Algorithm 1 and Appendices E, J, K). Section numbers below refer to
that paper.

## 1. Why this is the next training-side lever

`eval/critic_noise.py` on u15094 (Oct 3, 1,184 play-phase states x 64 sampled
rollouts): the V critic's squared error against a single return is 3.25 =
2.23 action-sampling noise + 1.02 critic error. The noise share is 0.69
[0.66, 0.71] and falls through the round (0.75 with 19-27 cards, 0.62 with
11-18, 0.54 with 1-10). Monte Carlo explained variance is 0.42, and a perfect
V critic would cap at 0.60. GAE's multi-step trace carries exactly this
noise into the advantage; a better V critic cannot remove it. Q-boosting
targets the noise and nothing else, which is why it fits this measurement.

## 2. The estimator (paper section 3.2)

Centralized Q critic `Q(s, a)`, policy `pi`, `V^pi(s) = sum_a pi(a|s) Q(s, a)`
(enumerated over the legal candidates). One-step Expected SARSA residual and
the advantage:

    delta+_t = r_t + gamma * V^pi(s_{t+1}) - Q(s_t, a_t)
    A_boost_t = Q(s_t, a_t) - V^pi(s_t) + sum_{t' >= t} (lambda*gamma)^(t'-t) * delta+_t'      (3.2)

The critic target is the same trace, `Q_target_t = Q(s_t, a_t) + sum (lambda*gamma)^(t'-t) delta+_t'` (3.3),
regressed with MSE (4.3). With an exact Q every residual is zero pathwise, so
the trace adds no action-sampling noise. Theorem 3.1: unbiased for any fixed
Q when lambda = 1 and trajectories are on-policy; for lambda < 1 (the paper
uses 0.95) it is only lower-MSE than exact-V GAE when
`||Q - Q^pi||_inf < xi`. The sup-norm condition is over all candidates,
which for us means hundreds of mostly unvisited actions. Q-boosting trades
action-sampling variance for Q-critic error; our measured critic error is
already 31% of the V-critic residual.

Details that matter for an implementation:

* PPO is unchanged: same clipped surrogate (4.1), same ratio against the
  rollout policy. Only the advantage and the critic change.
* During the actor epochs the policy-expectation terms (`V^pi`) are
  recomputed with the current actor probabilities and the critic held fixed,
  then used as stop-gradient advantages. This costs one policy-weighted sum
  over critic values per row. At collection time the stored advantage must
  therefore be the one under pi_ref, and the recomputed one only changes the
  `V^pi` terms.
* Reward is terminal only, gamma = 1, as in our round returns.
* The paper's Q critic is trained for `K_critic = 4` epochs per iteration;
  ours is currently trained inside the same loop as the actor.

## 3. Mapping to our trainer

| Piece | Today | VRPO version |
|---|---|---|
| Critic | `HistoryCritic`: MLP over (obs, hidden hands), scalar head (`train/history_model.py`) | same tower plus a candidate-scoring head: `Q(s, c) = <f(state), g(cand_c)>` with its own candidate encoder (the critic shares nothing with the actor) |
| Advantage | `compute_gae` over one seat's trajectory (`train/history_rollout.py`) | `q_boost_advantages` over the same trajectories; needs `Q` of the taken action and `V^pi(s_next)` per row |
| Rows needed | chosen candidate, logp, candidate count | additionally the candidate features of each stored decision, to score all legal candidates (the learner already builds them for the actor loss) and the actor's probabilities over all candidates |
| Loss | clipped surrogate + value MSE | same surrogate with the new advantage; Q MSE against (3.3) |

Cost: scoring all candidates with the critic is another pass over the
candidate set per row (hundreds of candidates). The actor's own candidate
scoring is the benchmark; budget the critic head at well under the actor's
cost by keeping the candidate encoder small.

## 4. Trace over whose steps (the Guandan-specific choice)

The paper keeps every player's Q in one centralized critic and takes the
trace over all steps of the game, so other players' sampled actions are
averaged out too. Our buffer stores one trajectory per seat, and between a
seat's consecutive decisions three other players act. Two levels:

1. **Per-seat trace (first version).** `s_{t+1}` is the same seat's next
   decision. Removes the seat's own future action noise. The other three
   seats' sampled actions between the two decisions remain noise.
2. **Team trace over the full round (second version).** One team Q critic per
   team (teammates share the same return, so one critic replaces two), trace
   over all four seats' decisions in time order. Needs the opponent steps'
   candidates and policy probabilities stored too; the population snapshots
   act in other processes, so this is an invasive change to collection.

Measurement (`eval/critic_noise_split.py`, u15094, 1,186 play-phase states,
8 x 8 grid of continuations per state: our team's sampling on one uniform
stream per row, the opponents' on one per column; 930 s on 4 CPU workers;
raw file `.work/critic-noise-2026-10-03/split-u15094-300x4x8x8.json`). Total
within-state return variance 2.28 [2.19, 2.38], matching the 2.23 of the
earlier measurement. Two-way ANOVA shares:

| | share of variance |
|---|---|
| our team's sampling alone | 0.118 [0.107, 0.129] |
| opponents' sampling alone | 0.101 [0.091, 0.113] |
| interaction (cannot be split) | 0.781 [0.767, 0.794] |
| left if our team's sampling were removed | 0.882 [0.871, 0.893] |
| left if the opponents' sampling were removed | 0.899 [0.887, 0.909] |

By cards left (own / opponent / interaction): 19-27 cards 0.08 / 0.07 / 0.85,
11-18 cards 0.14 / 0.13 / 0.73, 1-10 cards 0.23 / 0.20 / 0.56.

Reading: the noise is not mostly one side's. One changed move reroutes the rest
of the round, so about four fifths of the variance only exists when both sides
sample. Removing our team's sampling alone would leave about 88% of the noise,
so a trace that averages out only our own future actions can remove at most
about 12% of it, and a per-seat trace (not even the partner) less. Only a trace
over all four seats' decisions can remove most of it. The simple per-seat version is therefore
unlikely to be worth a GPU night by itself; the team (four-seat) trace is the
version that targets this measurement, and it needs the opponents' candidates
and probabilities stored at collection. Caveats: "removed" here means a fixed
random stream, not the expectation Q-boosting takes, so this is a guide to
where the noise sits and not an exact prediction of the gain; 8 x 8 cells per
state; later in the round the own and opponent shares grow as the interaction
shrinks.

### 4a. What the four-seat trace changes in the code (read Oct 4 2026)

Read `train/history_rollout.py` (buffer and collector), `train/history_population.py`
and the learner path in `train/history_ppo.py`.

What exists today:

* Only rows of the `LEARNER` identity are stored (`HistoryCollector`, `acting`
  mask). With the population on, each match has one guaranteed learner seat and
  every other seat is a snapshot with probability `snapshot_probability` (0.5),
  so about 2.5 of 4 seats are stored on average. Snapshot seats act through
  frozen actors, in separate calls (and, opt-in, one merged call), and nothing
  about their decisions is kept.
* Trajectories are keyed `(env, team, round)`: GAE already runs over one
  team's rows (both teammates interleaved). So today's trace is a team trace
  over our own team's decisions, not per seat; section 4 level 1 means this.
  The opponents' decisions are not in it at all.
* Stored per learner row: obs, hidden hands of the other three seats (the
  critic's privileged input), candidate features, chosen index, logp of the
  chosen candidate only. The critic is `HistoryCritic`, an MLP over
  (obs, hidden) with a scalar head, evaluated over all rows once per update by
  `refresh_values`.
* Rounds are zero-sum between teams (`seat_return[1] = -seat_return[0]`), so
  one critic per decision, valued for the acting team, serves both teams with a
  sign flip at opponent steps.

What the four-seat trace needs, in order of cost:

1. **A Q head on the critic.** Candidate encoder plus inner product with the
   critic's state feature, shared nothing with the actor. Uses the candidate
   features already uploaded for the actor loss. The tower is an MLP, so the
   extra learn cost is small next to the history Transformer. The head must be
   created without disturbing the RNG/initialisation order of existing
   parameters (as the auxiliary heads do) and resumed from u15094 with a fresh
   head, so the Q critic needs a warm-up (critic-only updates or a GAE phase
   while it fits) before its advantages are trusted.
2. **Expected policy value `V(s) = sum_c pi(c) Q(s, c)` for every row.** Needs
   the behaviour policy's probabilities over ALL candidates of each stored
   decision. Learner rows: the collection actor's table, which collection
   computes anyway for sampling; or one extra actor forward per update over
   all rows (about a sixth more learn time). Snapshot rows: the snapshot's own
   table, which exists only inside collection, in three inference paths
   (plain, merged snapshot call, cached inference), so it has to be plumbed out
   and stored (float32 per candidate).
3. **Store the snapshot seats' play rows too:** obs, hidden, candidates, chosen
   index and the table. Rows grow by about 1.6x on average (4 / 2.5), so
   buffer memory and the host-to-device upload grow with it. Candidate
   features dominate the per-row bytes.
4. **Regroup trajectories by round instead of by team,** time-ordered across
   all four seats, and run the trace once per team over that chain with the
   sign flip. `finish_round` and the `traj` bookkeeping change; the DDP
   per-rank buffers, learner length groups and minibatch-by-match grouping do
   not need a different scheme but are all tested against the current row set.
5. **Learner cost.** The Q loss runs over all four seats' rows (1.6x rows
   through a small MLP). The actor loss stays on learner rows only. Expected
   total: learn up roughly 10-30%, collect roughly unchanged (no extra
   inference, extra copying and storage). Throughput has been strength in
   this project, so this needs a measured steady-state profile before a long
   run.

A cheaper first experiment exists. In pure self-play (`all_learner`: all four
seats are the current policy) every seat's rows are already stored and every
policy is the current actor, so items 3 and the snapshot half of item 2 vanish:
only the Q head, the per-update probability table, round-keyed trajectories
and the new advantage are needed. The cost is that both arms of the A/B must
then run pure self-play from u15094 (a different recipe from the lineage's
population), which transfers to the lineage only if the opponent pool does not
matter; the 2 x 2 capacity x pool screen (Sept 27) found no pool effect at 4M
decisions, which supports but does not prove it.

Rough effort, one person, including tests and a CUDA gate: pure-self-play
version about 1-1.5 days; adding snapshot rows to the mixed population about
1-2 more. Main risks: Q error on untaken candidates; stale-probability
bookkeeping when the actor changes between collection and the epochs; the
hidden-state dependence of Q for opponent seats' decisions (it is valued from
that seat's own view); and reviewing the interaction with every opt-in speed
path (paged cache, merged snapshot encoder, learner length groups).

## 5. Plan

Stage 0 (Mac, no cost; first deliverables in this change; the own/opponent noise split in section 4 is done):

1. `train/q_boost.py`: pure numpy `q_boost_advantages` / targets, same shape
   conventions as `compute_gae`. Tests: exact-Q gives zero-variance and
   exact advantage on a stochastic game (the paper's matching-pennies
   example); lambda = 1 unbiased for an arbitrary fixed Q; reduction to
   one-step advantage at lambda = 0; terminal handling and padding match
   `compute_gae`.
2. Offline check on a synthetic game with hidden state, before any Guandan
   data: advantage error of GAE with exact V against Q-boosting with a
   noisy Q, as a function of Q noise level, locating the break-even critic
   error. This shows what Q accuracy we must reach.

Result of item 2 (`eval/q_boost_synthetic.py`, depth 7, 4 candidates,
random softmax policy, 3 seeds x 10,000 trajectories, lambda 0.95, iid
Gaussian critic error sigma): mean squared advantage error of exact-V GAE is
0.555 against a true advantage mean square of 0.143, so the sampled trace is
about four times noisier than the signal even with a perfect V. Q-boosting
error is about 1.9 sigma^2 (0.019 at sigma 0.1, 0.469 at 0.5) and crosses
exact-V GAE near sigma 0.54, about 1.4 times the rms advantage (0.38); against
a V critic with the same sigma the crossing is near 0.8. This toy has
iid critic error and a different noise-to-signal ratio from Guandan, so it
fixes the shape of the trade-off only. The number that matters is the real
Q critic's error in units of the real advantage's rms.

Stage 1 (Mac, hours of CPU, ask the user first): the real offline check.
Fix a batch of self-play data from u15094 with candidates and hidden hands,
fit a Q critic with the (3.3) targets on CPU, and compare the advantage
estimation error of Q-boosting against GAE using rollout-based
`A(s, a) = Q(s, a) - V(s)` as truth (64 rollouts per candidate on a few
sampled candidates per state, reusing the critic-noise rollouts). Go to a GPU
only if Q-boosting is clearly closer to truth than GAE. State the expected
change and the measurement error before running (no plateau claims without
power).

Stage 1 result (Oct 4 2026; `eval/q_boost_offline.py`, `train/q_critic.py`; data
and logs in `.work/q-boost-stage1-2026-10-04`). Pure self-play from u15094; Q critic
warm-started from the V critic, fit with the four-seat Expected SARSA(0.95)
targets refreshed each epoch; truth = 48 sampled continuations after the taken
action minus 48 from the state, per point, truth noise subtracted from every
squared error.

* Critic fit. 449k decisions: validation error against realised returns 5.7%
  below the V critic, overfitting after 2 epochs. 4.5M decisions (60,000
  rounds): 10.0% below (2.767 vs 3.074), best epoch 4 of 8, overfitting after
  that. Still data-limited, but ten times the data moved it only from 5.7% to
  10%.
* True advantage of the sampled action: rms 0.19 [0.15, 0.23] levels (2,142
  points, two test sets); every estimator's rms error is 0.7-1.0.
* Squared error against truth, lambda 0.95 (2,142 points): GAE own-team 1.03;
  GAE four-seat 0.71; Q-boost own-team 0.92 (ratio to GAE 0.90 [0.85, 0.95]);
  Q-boost four-seat 0.52 (ratio 0.50 [0.47, 0.54]). But squared error is mostly
  scale: GAE own-team falls from 0.96 to 0.35 to 0.24 as lambda goes 0.95, 0.8,
  0.5 (first test set), and PPO normalises advantages per minibatch.
* Correlation with truth (what survives normalisation), difference to GAE
  own-team (0.084): Q-boost own-team -0.004 [-0.026, +0.015]; Q-boost four-seat
  +0.013 [-0.012, +0.037]; GAE four-seat +0.010 [-0.009, +0.027]. First test
  set (1,103 points, 449k-decision critic): +0.006 and +0.011, intervals about
  the same. Cannot distinguish from no gain; detectable difference at this
  sample about 0.025 (30% of the baseline correlation). Truth noise caps the
  measurable correlation near 0.5, so all estimators capture only a small
  share of the attainable ranking signal.
* By cards left, correlation with truth is about 0 for every estimator with
  19-27 cards (0.00 to 0.04) and 0.12-0.17 later.

Reading: the Q critic lowers the advantage estimator's scale and squared error,
mostly through shrinkage, but does not measurably improve how well the
estimate ranks the sampled actions, which is what the normalised PPO update
uses. This setup does not support spending the collector change (4a) or a GPU
night on Q-boosting of the sampled-action advantage. It does not test the
second-order idea of section 7: whether the Q critic ranks all candidates of a
state well enough to replace the single sampled action in the current-step
gradient; that needs the truth for every candidate of a state and is the check
to run next if the direction stays open.

Stage 2 (GPU, about one 8 h night per arm, $4-5): from u15094, arms (a)
continued main lineage, (b) per-seat Q-boosting; evaluate with the standing
yardsticks (DanLM 4,000 deals; lineage_eval new-vs-previous 2,000 deals).
Add (c) the all-candidate current-step gradient (section 7) only after (b)
shows a gain.

Risks to carry into the experiment: Q error on unvisited candidates biases
the advantage; the extra critic pass lowers throughput (and throughput has
been strength in this project); the paper reports variance reduction and
exploitability on small games plus one Dou Dizhu result, with no
Guandan evidence and no check that the estimates are more accurate in a
large game.

## 6. Corrections to the earlier chat list

* "Retrace for the replay pool": dropped. The paper's critic uses a cyclic
  replay of 64 rollouts without off-policy correction, but its ablation
  (Appendix K.1, capacities 1, 4, 16, 64 times the rollout) shows no
  degradation from removing replay. Our PPO has no replay and needs none.
* "Reference policy anchored to the old policy": not adopted. The paper's
  KL-to-uniform term (alpha = 0.1, decayed) exists for last-iterate
  convergence toward equilibrium; we have the entropy bonus. Revisit only if
  entropy collapses again.
* "Q-boosting leaves the current step sampled": imprecise. (3.2) already
  uses `Q(s_t, a_t) - V^pi(s_t)` at step t, not a sampled return. What the
  paper does not do is the expected policy gradient over all candidates at
  the current step (section 7).
* Weight averaging: the paper evaluates a policy EMA (beta = 0.999) on the
  two large games, following Sokota et al. Our checkpoint average already
  gives +0.10 vs u15094 (2,000 deals) and −1.505 vs DanLM.
* The paper decays the actor and critic learning rates and the clip
  coefficient as `base * min(1, sqrt(T_eta / T))` after `T_eta` stable
  iterations (5,000 of 40,000 on the large games), and uses Muon with a
  clip of only 0.02. Our lr-decay A/B (e2b8862) covers the first part; the
  clip and optimizer are separate, untested differences (ours: Adam,
  clip 0.2).
* The paper's batch of 2,048 trajectories and 4 actor + 4 critic epochs are
  not ours (2 epochs); do not copy hyperparameters across without an A/B.

## 7. Second-order ideas (after the first A/B)

* **All-candidate current-step gradient.** With a Q critic the advantage of
  every legal candidate is available, `A(s, c) = Q(s, c) - V^pi(s)`, so the
  current-step policy gradient can be the exact expectation
  `sum_c pi(c) A(s, c) grad log pi(c)` instead of one sampled candidate,
  at little extra compute. Cost: it relies on `Q` for candidates the
  behaviour policy never took, where the critic is least trained. Treat as a
  separate arm.
* **Team critic** (section 4, level 2).
