# Stage C C1(a): candidate pruning audit

Status: complete, September 23, 2026. Local CPU, 36 seconds per checkpoint.
Script `eval/pruning_audit.py`, tests `tests/test_pruning_audit.py`.
Raw reports: [league final](stage-c-pruning-audit.json),
[frozen arm](stage-c-pruning-audit-frozen.json).

## Question

Stage B policies choose among the frozen M1's top 32 candidates plus pass
(`train/policy.py`). Does that cap bind? Two checkpoints were audited: the
B8 league final (the strongest model, 5,220 updates) and the B8 frozen arm
(8,266 updates), each playing itself in all four seats exactly as the arena
does (argmax over the pruned set, heuristic tribute, house rules, canonical
actions), one million play decisions each, seeds 20260926 and 20260927.

## How often pruning happens

| Decisions | league | frozen |
|---|---:|---:|
| all play decisions with more than 32 candidates | 7.65% | 6.25% |
| early-round leads | 84.6% | 84.1% |
| middle-round leads | 27.7% | 24.2% |
| late-round leads | 1.6% | 1.2% |
| follows, any stage | 0.01% to 6.8% | 0.01% to 3.9% |

Candidate set sizes (league): median 4, p90 25, p99 187. Leads: median 11,
p90 88, p99 423. Follows: median 4, p90 14. So the cap is a lead-phase
phenomenon, concentrated in the first third of a round; 21% of decisions
are leads and a third of those have more than 32 candidates.

## What gets pruned

Fraction of candidates removed when present, league final:

| Type | Candidates | Pruned |
|---|---:|---:|
| Straight | 2.01 M | 73.9% |
| FullHouse | 4.05 M | 61.1% |
| Plate | 0.14 M | 58.9% |
| Tube | 0.50 M | 57.3% |
| Triple | 0.49 M | 44.0% |
| Pair | 1.32 M | 24.7% |
| StraightFlush | 0.59 M | 16.3% |
| Single | 2.60 M | 13.7% |
| Bomb | 1.31 M | 11.4% (4-bomb 10%, 6-bomb 17%, 8-bomb 29%, 9-bomb 43%) |
| Pass | 0.79 M | never |

Candidates that use a wild card are pruned at 58% versus 31% for those
without. The reference removes mostly the long combinations (straights,
full houses, tubes, plates) and wild-consuming readings, and keeps the
cheap plays. Larger bombs are pruned more often than small ones as leads,
which is the reference saying "do not lead with your big bomb".

## Does the policy press against the cap

The reference rank of the action the league policy actually chose:

| | league | frozen |
|---|---:|---:|
| rank 0 (M1's own favourite) | 52.5% | 48.1% |
| median rank | 0 | 1 |
| p99 rank | 14 | 17 |
| rank 31, the last allowed | 0.029% | 0.036% |
| ranks 28 to 31 | 0.12% | 0.17% |

There is no pile-up at the boundary. The policy departs from M1 often
(half its choices are not M1's top pick), but almost always within M1's top
fifteen. This is the direct evidence on the question, and it says the cap
is not binding on the choices the policy wants to make.

## The full-set argmax, an indicator only

The policy's logits over the full canonical set, including candidates it
was never trained on:

| | league | frozen |
|---|---:|---:|
| full-set argmax outside the pruned set | 0.32% | 0.45% |
| on early-round leads | 5.0% | 7.8% |
| would be admitted by top 48 | 99.82% | 99.76% |
| would be admitted by top 64 | 99.89% | 99.85% |
| would be admitted by top 128 | 99.97% | 99.97% |

When the full-set argmax is outside, it is a single in 71% of cases
(league), and its logit exceeds the chosen action's by 8.2 on average, an
implausible margin at temperature 0.02 that marks these as untrained
extrapolations, not preferences. The audit cannot say whether any of them
is a good play; that is the training comparison C1(c).

## Reading

1. **The cap is not a measured ceiling.** The policy's choices sit far
   inside the allowed set and show no boundary pressure. Widening top-k
   would change the sampled support on about 8% of decisions, almost all
   early leads.
2. **The unknown is the untrained region**, which the audit can only
   bound: 0.3% of decisions have an untrained candidate the policy's
   logits prefer, and top 64 would admit two thirds of those.
3. **Cost of widening is small.** Top 64 doubles the policy's candidate
   compute only on the 8% of decisions where pruning is active.
4. **Recommendation.** Do not rent a GPU for C1(c) on its own. Add one
   `top_k = 64` arm to the C2 learning-signal comparison, where a control
   already exists, and treat the `candidate_union` mode as optional. Move
   on to C0 and C2.

## Limits

- Self-play of the audited checkpoint only; the candidate distribution
  under other opponents differs, though the lead-phase structure does not.
- Candidate set sizes above 512 are counted in the 512 bucket.
- One seed per checkpoint; the two checkpoints agree on every number that
  matters here, which is the replication.
