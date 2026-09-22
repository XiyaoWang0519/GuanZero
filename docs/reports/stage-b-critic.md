# Stage B B2: perfect-information critic and gate G1

September 22, 2026. Code: `train/critic.py`, tests: `tests/test_critic.py`.
Machine-readable summary: `stage-b-critic.json` (same directory). Checkpoints
are in `.work/critic-fit/{perfect,public}/best.pt` and are not committed.

## Result

**G1 passes.** On the held-out test split the critic's return MSE is below the
M1 play head's max-Q MSE overall and in every stage bin. The upper end of the
95% interval for the paired difference is below zero in each case.

## Setup

- **Data.** B1 dataset `.work/critic-m1`, schema 1: 200,000 rounds of M1-final
  self-play (`final.pt`, 34,496 updates, id `8a8e2b08…`). Train, val and test
  are split by match. Test has 20,449 rounds, 2,091 matches and 1,098,523
  decisions, of which 1,054,661 are play decisions.
- **Input.** The actor observation (1,849 binary planes) followed by the three
  `hidden_counts` rows for seats +1, +2 and +3, flattened to 162 counts. The
  layout is byte-identical to `RolloutBuffer.critic_input` and
  `gather()["critic_obs"]`, so B5 can feed the critic from the buffer as is.
  I chose this over encoding all four hands for two reasons. The observation
  already holds the actor's own hand and every public fact, so the hidden rows
  are the only missing information. And the engine already emits those rows at
  every decision.
- **Target.** The Monte Carlo round return for the actor's team, from -3 to +3.
  This is the same target the DMC play head was trained on.
- **Network.** MLP of 4 × 512 (the v1 state-tower size) with a linear value
  head, 1.82 M parameters. It shares no weights with `GuandanModel`.
- **Fit.** MSE loss and Adam (learning rate 5e-4, batch 4,096). Training
  streams 4 shards at a time with an in-memory shuffle. Every 500 steps the
  model is scored on the first 250k decisions of the val split, cut at a round
  boundary. Training stops early after 6 checks without improvement, and there
  is a cap of 20,000 steps and 1,080 s. The best weights are checkpointed.
  Seed 20260924, 6 torch threads, CPU only.
- **Comparison.** The M1 play head's `Q` for its argmax action, which is its
  max `Q`, is stored for each play decision. Both it and the critic are scored
  against the same returns on the same test play decisions. Tribute rows have
  no `Q` and are reported for the critics only.
- **Intervals.** Percentile bootstrap, 2,000 resamples of whole test rounds.
  All estimators are paired on the same resampled rounds. Resampling whole
  matches instead gives the same intervals to ±0.003 (`overall_ci95_matches`
  in the JSON).

## G1 table (test split, play decisions)

MSE against the round return. The variance of the play-decision return is
5.378.

| Stage bin | Decisions | Critic (perfect) | M1 max Q | Critic − M1 [95% CI] |
|---|---:|---:|---:|---:|
| overall | 1,054,661 | 2.393 [2.357, 2.432] | 2.789 [2.758, 2.822] | **−0.396** [−0.421, −0.373] |
| early | 226,574 | 3.809 [3.754, 3.868] | 4.171 [4.123, 4.220] | **−0.362** [−0.399, −0.327] |
| middle | 322,932 | 3.013 [2.960, 3.069] | 3.478 [3.434, 3.525] | **−0.464** [−0.502, −0.427] |
| late | 505,155 | 1.361 [1.334, 1.391] | 1.728 [1.701, 1.758] | **−0.367** [−0.389, −0.346] |

Tribute and back-tribute decisions (43,862 rows), scored for the critic only:
perfect 3.762 [3.698, 3.830], public-only 4.113 [4.044, 4.184].

## Ablation: public information only

This is the same network, input and fit procedure, with the 162 hidden columns
set to zero in both training and test.

| Stage bin | Public-only critic | Public − M1 [95% CI] | Public − perfect [95% CI] |
|---|---:|---:|---:|
| overall | 2.885 | +0.096 [+0.081, +0.112] | +0.492 [+0.469, +0.517] |
| early | 4.217 | +0.046 [+0.027, +0.064] | +0.408 [+0.370, +0.444] |
| middle | 3.583 | +0.106 [+0.082, +0.128] | +0.570 [+0.534, +0.609] |
| late | 1.841 | +0.113 [+0.096, +0.129] | +0.480 [+0.459, +0.502] |

Perfect information is worth about 0.49 MSE, or 9% of the return variance.
All of the critic's advantage over M1 comes from the hidden rows. Without
them, the offline critic is slightly worse than M1's max Q, which is
action-conditioned and was trained on far more samples. So G1 is a
perfect-information result, not an artefact of the fitting procedure.

## Fit and runtime

| Fit | Steps run | Best step | Best val MSE | Stop | Wall time |
|---|---:|---:|---:|---|---:|
| perfect | 8,500 | 5,500 (22.5 M samples, 2.6 epochs) | 2.475 | early stop | 295 s |
| public | 5,000 | 2,000 (8.2 M samples, 1.0 epoch) | 2.973 | early stop | 89 s |

Evaluating on the test split, including 2,000 bootstrap resamples, took 6 s.
The whole run took about 6.5 minutes, well inside the 60-minute budget. Peak
resident memory was about 1.6 GB.

## Limitations

- **On-policy for greedy M1 only.** B1 played argmax in all four seats with the
  heuristic tribute. The critic estimates the value of that policy's play. Once
  PPO samples from a softmax and changes the policy, the value function shifts,
  which is why B5 trains the critic jointly. The offline weights are a warm
  start, not a fixed critic.
- **Heavy overfitting.** On the perfect fit, train MSE was 1.48 against val
  2.81 at step 8,500. Early stopping picked 2.6 epochs. Decisions within a
  round are strongly correlated, so there are effectively about 160k
  independent samples, the number of training rounds. More rounds, dropout or
  weight decay would probably lower the MSE further. None of these were tuned:
  one configuration was run, with no hyperparameter search.
- **Fixed val subset.** Early stopping used about 250k val decisions, not the
  whole val split. The test split was touched only by the final report.
- **Unequal comparison.** M1's Q conditions on the chosen action and was
  trained online on its own replay. The critic conditions on the state plus the
  hidden hands. G1 asks whether the critic is a better baseline for
  advantages, and it is. It does not say the critic is a better policy
  evaluator in general.
- **Single seed.** Seed-to-seed variation of the fit is not measured. The
  intervals cover test sampling only.
