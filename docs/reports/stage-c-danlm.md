# Stage C C0: DanLM external baseline

Status: complete, September 23, 2026. **Gate G6 (calibration): a large gap in
DanLM's favour.** Our strongest checkpoint, the B8 league final, loses to
DanLM by about two levels per round and wins about one round in nine. DanLM's
own MLP reproduction of DanZero, which DanLM beats only narrowly, also beats
our model by a wide margin. All runs were local CPU; no GPU was rented.

Raw reports: [B8 league vs DanLM](stage-c-danlm-duplicate.json),
[V1T calibration](stage-c-danlm-v1t-calibration.json),
[observation replacement verification](stage-c-danlm-fastobs.json).
Rules parity findings: `RULES.md` section 13.3.

## What DanLM is

[DanLM](https://github.com/dashidhy/DanLM), tree `16ace5ad` (Sept. 4, 2026):
a causal Transformer over the tokenized public play history plus a hand MLP,
trained by DMC self-play with a next-token-prediction auxiliary loss,
model-driven tribute, no search. First on the Botzone Guandan ladder in
April 2026. Published weights (`ckpts/DanLM_v1/dansformer_v1_best_eval.pt`,
45 MB) under Apache 2.0 with a non-commercial clause; the engine, encoder and
model code are CPython 3.12 macOS binaries. Nothing is vendored: the checkout
lives in `.work/external/DanLM`, the Python 3.12 environment in
`.work/external/danlm-venv`, and a `gd` extension built for 3.12 sits beside
the 3.14 one in `python/gd/`.

## How the comparison is played

`eval/danlm/arena.py` plays every round in both engines in lockstep. DanLM's
engine is the referee: its legal plays, trick flow and finishing order decide
the score. Our engine mirrors every action so that our policy sees its own
observation encoding; every disagreement is counted. `eval/danlm/bridge.py`
converts cards, levels and plays (DanLM's rank index is the highest natural
rank of a sequence window; ours is the ace-low window index).

Deals follow DanLM's single-round protocol: random level, half the deals open
with a tribute phase from a random previous finishing order, both teams at
the round level. Each deal is played twice with the teams swapped. The score
is our team's mean net levels per round over both legs, with a bootstrap
interval over deals, plus DanLM's own metric, the fraction of rounds whose
first finisher is on our team. Tribute for our seats uses the engine
heuristic, as in every Stage B evaluation; DanLM's seats use its model.

Harness checks before the numbers were trusted:

- all-ours lockstep rounds reproduce `eval/duplicate.py` exactly on 10 of 10
  deals (finish order and returns), and the referee agrees on every one;
- B8 against random DanLM-side agents: +3.00 levels per round, 32 of 32 legs;
- DanLM against its own DanZero V1T reproduction through this harness: DanLM
  wins 57.5% of rounds (author's figure 59.6%, same protocol, different
  seeds), so the harness reproduces DanLM's published calibration.

DanLM's compiled `get_observation` spends about 40 ms per call waiting on
something inside the binary while using 1 ms of CPU. `eval/danlm/fast_obs.py`
rebuilds the identical `Observation` in pure Python and is verified field by
field against the original over 13,417 random steps with zero mismatches
(including the row order of the legal-play list, DanLM's "other unseen"
formula, which ignores the player's own played cards, and the sentinel for a
finished teammate). With it installed the 20-round diff produces the same
decisions and the same divergence tallies in 4.9 s instead of 80.6 s.
`DANLM_SLOW_OBS=1` restores the original.

## Results

Fresh deals, seed 20260929 for the main run. Positive means our team gains.

| Team A | Deals scored | Levels per round, A minus DanLM [95% CI] | A's round win rate | Legs |
|---|---:|---|---:|---:|
| **B8 league final** (`0196bd76…`) | 3,998 | **−2.065** [−2.097, −2.034] | **11.5%** | 7,996 |
| M1 final (`8a8e2b08…`) | 999 | −2.460 [−2.511, −2.407] | 6.4% | 1,998 |
| DanZero V1T, DanLM's own MLP reproduction | 990 | −0.483 [−0.560, −0.410] | 42.5% | 1,980 |

By tribute setup the B8 result is flat: −2.05 with no tribute, −2.09 single,
−2.07 double, −2.06 anti-tribute. The gap is not a tribute artefact.

Rounds excluded because the mirror could not continue: 2 of 4,000 (B8),
1 of 1,000 (M1), 10 of 1,000 (V1T), all after DanLM declared a lone wild
card at a non-level rank (`RULES.md` 13.3), which our engine does not
represent. DanLM chose such a declaration 37 times in about 640,000
decisions of the main run.

Throughput with the observation replacement and 8 worker processes: 4,000
duplicate deals in 294 s.

## Reading

1. **We are far behind, and not because of the Transformer.** The ladder is
   M1 (−2.46) < B8 league (−2.06) ≪ DanLM's MLP baseline (−0.48) < DanLM.
   Our best model is about 1.6 levels per round behind an MLP that DanLM
   beats only 57.5% to 42.5%. Architecture explains the last step of that
   ladder, not the first.
2. **The likely cause is training volume.** DanZero-style DMC runs use 160
   CPUs and one GPU for 30 days. Our M1 pilot is 34,496 updates and 71 M
   decisions; the Stage B runs add one to six GPU-hours each. The internal
   ladder (greedy → M1 → B6 → B8) was measured against opponents from the
   same short history, which is why it looked healthy.
3. **What G6 decides.** The gate opens row C5, the history tower with the
   NTP loss, since the gap is real. But the ladder above says the first
   lever is longer training with the recipe we have (C2 arms at a much
   larger budget), with DanLM as the yardstick at every checkpoint. The
   tracker ordering is revised accordingly.
4. **Rules.** No `botzone` `RuleConfig` field was needed for single rounds:
   tribute, trick flow and scoring agree with the `house` profile in every
   measured round. Match-level rules were not compared.

## Limits

- Single rounds only, as in DanLM's own evaluation. Full matches would need
  its `GuanDanGame` level rules compared first.
- One DanLM checkpoint, one seed per pairing; intervals are over deals.
- DanLM's tribute is model-driven, ours heuristic; the per-tribute breakdown
  shows this is not where the gap comes from.
- The harness excludes the rare rounds where DanLM's wild-card declaration
  cannot be mirrored; at 0.05% of rounds this cannot move the result.
