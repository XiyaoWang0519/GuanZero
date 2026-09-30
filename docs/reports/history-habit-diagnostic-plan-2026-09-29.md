# Cross-round habit diagnostic: plan

Status: **plan only** (September 29 2026). Nothing is implemented or run.
Phase 0 needs no rule exception; phase 1 waits for the user's approval of the
exceptions listed below.

## Why

The [history ablation](history-ablation-2026-09-29.md) found that u2623 barely
relies on earlier rounds (SWAP of earlier rounds changes 0.08% of greedy
choices on 6,122 decisions; FULL vs ROUND strength −0.010 [−0.071, +0.053]).
Two explanations are confounded:

1. The training world has nothing worth remembering. Opponents are the
   learner itself (identity 0, on average 1 + 3 × 0.5 = 2.5 seats per table,
   `train/history_population.py:74`) and the four most recent own-lineage
   snapshots, which likely play almost identically.
2. The model or training recipe cannot learn to use cross-round history
   even when it would pay.

This diagnostic plants a controlled, match-stable opponent habit to separate
the two before any money goes into opponent populations.

## Decisions recorded from the review discussion

- The diagnostic comes first; the two-lineage league and the latent-style
  population come after it, chosen by its outcome.
- A planted habit is not automatically a cross-round memory task. It must be
  shown that (a) earlier rounds add identification information beyond the
  current observation and current round, and (b) knowing the style is worth
  something. Without both, a negative result is "inconclusive", not "the
  model cannot learn".
- Stronger bias is not better: a habit visible within one round makes
  FULL ≈ ROUND for the wrong reason; a bias that cripples the opponent
  makes one generic strategy sufficient.
- All arms face the same read-only opponent pack (same base weights hash,
  same bias definition, same style distribution, matched deals). Own-lineage
  snapshots would make the arms differ in more than history visibility.
- First round changes nothing else: same network, response head, PPO recipe
  and per-round reward. No new auxiliary task, critic change or latent z.
- ROUND in training must block cross-round information at the encoder
  (reset the public stream at each round start), not only mask the final
  cross-attention: the causal self-attention would already have mixed earlier
  rounds into current-round tokens.
- Two same-team learner seats against two frozen, biased enemy seats keep
  the learner sample share near the current one; not one learner per table.
- Budget is exploratory. A first set of short runs screens for signal;
  a conclusion needs independent repeats.

## Phase 0: calibration (local CPU, no GPU, no exception needed)

Opponent policy for style z ∈ {−1, +1}, fixed for the whole match and drawn
independently of deal, seat and environment:

    pi_z(a | x) ∝ exp(l_frozen(a, x) + z · b · f(a, x))

f is a pre-declared action feature, b the bias strength. Every legal
candidate is still scored and sampled; no pruning; learner reward untouched.
z never enters the learner's observation, public tokens or caches.

Candidate axes (one axis, two styles at first):

| Axis | f(a, x) = 1 when | Trade-off |
|---|---|---|
| Voluntary pass | a is Pass and a beating candidate exists | frequent, may be identifiable within one round |
| Bomb use | a is a bomb | rarer, so accumulation across rounds likely matters more |
| Lead structure | a is a lead of a declared combination class | to be measured |

For each axis and a few b values, from u2623 self-play samples, measure:

1. Share of decisions where the bias changes the action distribution
   (TV above a threshold), and how often f is actually available.
2. Cross-round identification gain, model-free: a count-based posterior on z
   from (opportunities, feature choices). At the start of round r ≥ 2,
   compare accuracy using current-round counts only vs all earlier rounds.
   Split by whole matches.
3. Opponent strength loss vs the unbiased base (paired deals).

Pick the axis and b with the largest cross-round gain that does not
materially weaken the opponent.

## Phase 1: three short arms (single RTX 4090, needs the exceptions)

| Arm | Learner sees |
|---|---|
| ORACLE | true opponent z (private decision channel only) + current round |
| FULL | full match history |
| ROUND | current round only, stream reset at round start |

Shared: start from u2623, same frozen biased opponent pack, same seeds and
deal schedule, aligned on effective PPO samples (learner Play rows), wall
clock and cost reported separately.

Reading:

| Result | Conclusion |
|---|---|
| ORACLE ≈ ROUND | the style is not worth identifying; redesign the bias, no verdict on the model |
| ORACLE > ROUND, FULL ≈ ORACLE | history channel extracts the available value |
| ORACLE > ROUND, FULL between | partial use; the gap is the unexploited value |
| ORACLE > ROUND, FULL ≈ ROUND | not learned at this budget and recipe; check training signal before blaming architecture |

A FULL win must be backed by same-checkpoint SWAP/ROUND ablation on held-out
matches (dependence on the correct history), not only by overall strength.
Starting from u2623 tests "can a trained model acquire cross-round
adaptation"; it does not test learning it from scratch, and u2623's learned
disregard of history makes the test stricter.

## Exceptions requiring approval (diagnostic directory only)

Weights and trajectories of these runs never enter the main lineage,
populations, auxiliary labels or reward fitting.

1. Training opponents are a fixed read-only pack (u2623 base plus
   match-fixed logit bias), not own-lineage snapshots, and the style is
   hand-specified.
2. All three arms continue from u2623.
3. ORACLE reads the true z as a privileged input.

All new code sits behind flags that default to off; the existing training
path stays bitwise unchanged.

## After the diagnostic

- Positive: build real opponent diversity. Candidates: two independent
  Transformer lineages exchanging match-frozen snapshots (DESIGN.md allows
  independent seeds with one population protocol across arms; Vast machine
  20082 had a 2×RTX 4090, 64-core offer at $1.55/h on September 29), or a
  latent-style population (needs its own design: learner rows must cover
  every z, and frozen weights per match are still required).
- Negative with a valid calibration: work on the history pathway
  (per-seat history readout, an opponent-behaviour prediction head
  conditioned on the opponent's true pre-action hand with a matched
  no-history baseline, history-aware critic), one variable at a time.
- Match-level return (active probing) only after passive adaptation works.
