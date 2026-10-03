# Opponent diversity: plan (October 3, 2026)

Goal (user, Oct 3): general strength above DanLM, not a DanLM exploiter. The
v3 vision, learning opponents' playing habits, needs opponents that have
habits. Self-play against own copies has none, which is why the next-token head
alone changed nothing on Oct 3 and why the planted-habit ORACLE arm never used
the style it was given at 20M decisions (Sept 30, Oct 1).

## What a habit needs to be learnable

1. Present: a seat's style is fixed for a whole match and differs between
   matches, so there is something to identify and a reason to keep history.
2. Strong enough to matter: the styled opponent's choices must differ visibly
   from the neutral policy (phase 0 measured a count-based identification gain
   of +0.14..+0.22 for lead_single b=1 without weakening the opponent).
3. Carried by strong opponents: exploiting a weak heuristic bot teaches
   little that transfers. Styles go on Transformer snapshots, not on the
   greedy bot.

## Design

**Styled snapshots.** `StyledActor` (train/history_habit.py) already biases a
history actor's logits along one axis (pass, bomb, lead_single) with a
strength. Extend it to a continuous style vector over four axes: pass
tendency, bomb threshold, lead high/low, partner sacrifice. Each snapshot seat
of a match draws (snapshot, style vector) once per match from a training
region of style space; a held-out region is reserved for evaluation
(train/styles.py already has train/test regions). Partners can be styled too:
teammate habits matter in Guandan. Calibration on the Mac first: a styled
u15094 must lose at most 0.1 levels/round against the unstyled one on 256
deals, else the strength is lowered.

**Population.** Keep the current recent-4 snapshot pool and match-pinned
seats. Of the snapshot seats, 70% styled and 30% neutral, so the learner still
meets its own unbiased past. Heuristic styled bots enter at most 10% of seats,
as a robustness sprinkle, never more. A DanLM-style imitation model is a later
member (one style among many), not the opponent.

**Learning signal.** Match-long history is already there. Two cheap additions
in the auxiliary-head framework (Oct 2): the next-token head now predicts
opponents whose moves differ from the learner's own policy, and a fourth head
predicts each opponent seat's style vector from the decision state. Labels are
free (the sampler drew them). The style head is the direct measure of
identification: its error should fall from round 1 to round 2 and 3 of a match.

## Measuring sticks (all paired, same deals)

- A, exploitation: against styled snapshot opponents of held-out styles,
  diverse-trained model vs the control trained on the current population.
- B, general strength: against the unstyled lineage, B11 and DanLM (4,000
  deals). Must not drop; this is the guard against becoming an exploiter.
- C, mechanism: the Sept 29 SWAP ablation (swap earlier rounds' history) must
  now change decisions, and the style head's accuracy must rise across rounds
  within a match. Without C, a gain in A is luck.

Success: C positive, A positive against the control, B not worse.

## Stages and budget

| stage | where | what | cost |
|---|---|---|---|
| 0 | Mac, about one day of code | style vector for StyledActor, per-match style draws in the population, style head and labels, calibration of style strength | $0 |
| 1 | one 5090 night, two machines | from u15094: control (current population, heads as now) vs diverse (styled snapshots + style head); A/B/C in the morning | about $8 |
| 2 | later | if stage 1 passes, the diverse population becomes the lineage's; add a DanLM-style imitation member; repeat A/B/C | about $4 per night |

## Risks

- Styled opponents are weaker than neutral ones; training against them could
  cost general strength (caught by B). The calibration bound limits this.
- Heuristic-style habits may not resemble human or DanLM habits; transfer is
  untested until a DanLM-style member exists.
- One seed per arm; run-to-run noise is about 0.04 against DanLM.
- The Sept 30 result (ORACLE did not use the style at 20M decisions) may
  repeat at 330M decisions; then the conclusion is "not learnable with this
  signal", and the fallback is the explicit style head as an input rather
  than a target.
