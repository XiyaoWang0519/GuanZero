# Paired per-deal evaluation, exclusion audit and checkpoint averaging (October 18, 2026)

Status: implemented and tested on CPU. **Every number here comes from
synthetic random-init checkpoints.** No real lineage checkpoint (m7, u86800,
m9), DanLM install or mini-ladder output was on this VM, so nothing below
measures playing strength. No GPU, no training job, and no change to the
`play/` or Botzone defaults.

Scope: next step 2 of the October 18 assessment. It covers persisting
per-deal arena records, an exact paired comparison tool, an audit of the
`their_choice_missing` exclusions and a uniform checkpoint-averaging utility.

## What changed

| Piece | Where | What |
|---|---|---|
| Per-deal arena records | `eval/danlm/arena.py` (`score_duplicate`, `LockstepRound.fail`) | `per_deal` holds both legs of every deal, excluded ones included: our reward, round win, decisions and divergence counts. A failed leg also gets a `failure` record: fatal kind, detail, decision index, and `trick_reading`, the reading substitution made earlier in the same trick (if any). The output also adds `excluded_deal_ids`, `failed_legs_by_kind`, `deal_set_sha256`, `checkpoint_sha256` and `source_sha256`. Headline score, CI (same bootstrap call, seed and order), win rate and `by_tribute` are computed as before |
| Deal-set identity | `eval/duplicate.py` `deal_set_sha256` | SHA-256 of the ordered hands, levels, owner, leader and previous order. Also written by `eval/lineage_eval.py`, which now states `excluded_deals: 0` |
| Paired comparison | `eval/paired.py` | A − B per deal on the deals both runs scored. Gives a whole-deal bootstrap and a normal interval, the round-win-rate difference, each run's excluded deals with reasons, a worst-case bound had those deals been scored, the unpaired standard error and the per-deal score correlation. It refuses to pair runs that differ in deal set, seed, deal count, tribute fraction, or opponent (DanLM spec or lineage baseline SHA). It also refuses arena outputs written before this change, which have no `per_deal` |
| Averaging | `train/checkpoint_average.py` | Uniform mean of N ≥ 2 `history_ppo` checkpoints of one lineage and architecture (actor and critic, float64 accumulation, stored as float32). Non-floating tensors must be equal. Refuses duplicate files and inputs that are already averages. Output: optimizer, RNG and population are empty, `averaged_from` lists each input's path, SHA, model digest and updates/decisions. `train.history_ppo` refuses to resume it |

Tests: `tests/test_paired.py` (5) and `tests/test_checkpoint_average.py` (5).
They check the arena scoring on synthetic leg records (DanLM is not
importable here), the failure record, paired means/exclusions/bound,
refusals, a lineage-eval round trip, exact averaging, that averaging two
equal-weight checkpoints plays identically to the input (score exactly 0),
and the resume refusal.

## Audit: `their_choice_missing` exclusions

1. **`eval/lineage_eval.py` cannot exclude a deal.** One `gd` engine plays
   both sides through `VecEnv`. There is no mirror and no
   `their_choice_missing` path, and a round past 2,000 decisions raises
   instead of being dropped. Every lineage result (for example the 2,000-deal
   head-to-heads) covers all its deals. The exclusion path lives only in
   `eval/danlm/arena.py`, which is where this audit looked.
2. **How the arena excludes.** A DanLM-side choice that is neither an exact
   (type, cards, key) match nor a same-(type, cards) reading in our full legal
   set ends the leg (`match_theirs`), and `run_duplicate` drops the whole
   deal. Before this change the run kept only aggregate counts and at most 20
   samples per kind across the entire run. Neither excluded deal IDs nor
   per-leg reasons were persisted. That explains why the 2026-09-26 Kaggle
   campaign needed its own raw chunks to list its 5 failed deals.
3. **Probable mechanism (static analysis, not checked against data).**
   RULES.md 13.3 records that DanLM lists a lone wild card (or wild pair)
   declared at any natural rank. Our engine has one reading, at level power
   12; checked on `gd` today (level 7, the H7 wild card: `Single`, key 12
   only). When a DanLM-side seat plays such a declaration, the adapter maps it
   to our level-power reading (`their_choice_reading`). After that the
   referee's trick top is the low declared rank, while the mirror's top is
   the level card. A later DanLM-side follow in that trick that beats the
   declared rank but not the level card is absent from our legal set, so
   `their_choice_missing` fires. Our own seats in that trick lose candidates
   instead (`our_candidate_unmatched`). This fits the recorded rates. DanLM's
   own model chose such readings 0 times in 1,592 decisions, and its runs lose
   1–4 of 4,000 deals. A rule bot that plays its lowest-ranked single would
   pick them routinely, consistent with 173–236 of 1,000 for `lalala`,
   `flush-bomb`, `hulalala` and `ez`.
4. **The exclusions are not missing at random.** They need a wild card in a
   DanLM-side hand and a particular play line. The excluded deals are
   therefore a selected subset, and the remaining sample's mean can shift in
   either direction. `eval.paired` now prints the worst-case bound. With
   17–24% excluded, that bound is ±1.0 to ±1.4 levels per round for a difference,
   so those ladder results cannot rank opponents.
5. **What this change fixes and what it does not.** The next arena run
   records, per failed leg, the fatal kind and whether a reading
   substitution preceded it in the same trick (`failure.trick_reading`).
   That directly confirms or refutes item 3 from data. The mismatch itself
   is not fixed. The two options are (a) emulate DanLM's declared-rank wild
   readings behind a `RuleConfig`/`ActionConfig` field, which per house rule
   6 first needs a RULES.md section 12 test vector, or (b) keep excluding but
   report such opponents separately. Neither was done here: the engine is
   unchanged.

## Synthetic smoke test (CPU, 4 threads)

Stand-ins for "four checkpoints around u86800": one default-size random-init
`HistoryPolicyConfig` (1.36M actor parameters). Each copy gets independent
Gaussian noise of 2% of each tensor's std, named u85312/u85808/u86304/u86800
in lineage `synthetic-smoke`. A separately seeded random init is the
baseline. Each arm played `eval.lineage_eval` on 1,000 deals, seed 2026100301
(deal set `3dd109133613302c…`), source at `bc29053`. Averaging four
8 MB checkpoints took 1.1 s.

| Arm vs synthetic baseline | Net levels / round | 95% CI |
|---|---|---|
| syn-u86800 | −0.773 | [−0.854, −0.691] |
| syn-avg4 (uniform mean of the four) | −0.750 | [−0.834, −0.664] |
| syn-u85312 | −0.666 | [−0.749, −0.585] |

| Paired (`eval.paired`, 5,000 resamples) | A − B | Bootstrap 95% | Paired SE | Unpaired SE | Correlation |
|---|---|---|---|---|---|
| avg4 − u86800 | +0.023 | [−0.066, +0.113] | 0.046 | 0.060 | 0.41 |
| u86800 − u85312 | −0.107 | [−0.205, −0.011] | 0.050 | 0.061 | 0.32 |
| u86800 − u86800 (sanity) | 0.000 | [0.000, 0.000] | 0 | 0.059 | 1.00 |

Reading: the plumbing works end to end. Pairing reduced the standard error
by 18–23% even for random policies, whose per-deal scores correlate weakly.
Nearby trained checkpoints should correlate more strongly, but that is
unmeasured. The signs and sizes of these differences say nothing about real
checkpoints.

## Limitations

- No DanLM: the arena changes are tested through `score_duplicate` on
  synthetic leg records and `LockstepRound.fail` directly, not through a real
  lockstep run.
- Older arena outputs (u86800, m7 and earlier) have no `per_deal`. Pairing
  them requires a rerun on seed 20260929 with the current arena.
- The averaging smoke test uses perturbed random inits, not consecutive
  training checkpoints. Whether averaging around u86800/m9 helps is
  unmeasured. Earlier lineage results ranged from +0.10 to 0 as weights
  settled ([October 4](lr-decay-weight-avg-2026-10-04.md)).

## Next (proposed, not run)

On a machine that has the checkpoints and DanLM: rerun u86800, m7 and avg4
around u86800/m9 on the 4,000 DanLM deals (seed 20260929) with the current
arena, then `eval.paired` each pair. Rerun one high-exclusion mini-ladder bot
and count failures with a non-null `trick_reading`.

## Artifacts (local, `.work/paired-smoke-2026-10-18/`, not committed)

`make_checkpoints.py` (generator), `ckpt/*.pt`, `average.log`,
`syn-*-vs-baseline.json`, `paired-*.json`, `lineage.log`, `paired.log`.
