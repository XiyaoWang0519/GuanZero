# Shared eval baseline: m7, u86800, avg4 (October 20, 2026)

Status: checkpoint inventory and avg4 rebuild done on cloud + Velocity;
**Mac holdout DanLM 4k completed** (seed `20261020`). Selection DanLM 4k
numbers remain the October 18 Mac receipt
([paired-eval-mac-2026-10-18.md](paired-eval-mac-2026-10-18.md)); digests
matched so selection was not re-run. No training, no resume from avg4, no
research-boundary change.

## Purpose

Freeze the shared yardstick for the October 2026 research campaign: locate
main-lineage candidates, rebuild the four-checkpoint average **avg4**, record
selection vs holdout deal identities, and report DanLM paired metrics on both
deal sets. avg4 is an **evaluation** artifact only (`averaged_from` set;
`train.history_ppo` refuses resume).

## Checkpoint inventory (Velocity, 2026-10-20)

Host: `max-velocity.uwaterloo.ca` (`ssh velocity`). Main lineage
`history_ppo-2026092801-d1d612ae`. Segment tree
`~/guanzero/runs/main/segments/lr-main/`; milestones
`~/guanzero/runs/main/milestones/`.

| Role | Path on Velocity | Updates | SHA-256 |
|---|---|---:|---|
| m7 | `runs/main/milestones/m7-u73780.pt` | 73,780 | `fa10bd1ac34c91e173902be28bb22f8d31233de50f534fa89fdb6d092d74fefb` |
| avg4 input | `runs/main/segments/lr-main/update-082460.pt` | 82,460 | `09b473d0aecff6743258656aefa474804cbf5ca1db00804d16d14b95f9a3a2d1` |
| avg4 input | `…/update-083948.pt` | 83,948 | `de7b39c363d09263231944403d55b074c0cee01ecfdb53d50cf65b136b00a855` |
| avg4 input | `…/update-085560.pt` | 85,560 | `7b0da599028332104bfdf07bc38e90c0a782c6903def53bbb38806fd769cc108` |
| u86800 | `…/update-086800.pt` | 86,800 | `639ccda15452ef101561e836f684f238cb985bfcf412edd48fb063e383d4c720` |
| m8 | `milestones/m8-u81344.pt` | 81,344 | `7f01eb72f9ee6aad7873963a96e152c418d322db9844a08c9130c332950aec8c` |
| m9 | `milestones/m9-u89032.pt` | 89,032 | `3d3296eb531793d412442bdc139234e892006635909a37d8008a279cfbd149ed` |
| m10 | `milestones/m10-u96596.pt` | 96,596 | `356b47343dd7f25d44a205119ea6a9811fa25fc5cd1a4e01042b767614fee6fd` |
| latest segment seen | `…/update-099572.pt` | 99,572 | (not hashed this pass) |

m7 and the four avg4 inputs match the Mac report byte-for-byte. Cloud copies
used for averaging live under
`.work/eval-baseline-2026-10-20/inputs/` (gitignored). Rebuilt avg4 and its
record also sit on Velocity at
`~/guanzero/eval-baseline-2026-10-20/{avg4-u86800.pt,avg4-record.json}`.

## avg4 rebuild

Command (cloud checkout at `1d7659a`, CPU torch):

```sh
PYTHONPATH=.:python .venv/bin/python -m train.checkpoint_average \
  --output .work/eval-baseline-2026-10-20/avg4-u86800.pt \
  .work/eval-baseline-2026-10-20/inputs/u82460.pt \
  .work/eval-baseline-2026-10-20/inputs/u83948.pt \
  .work/eval-baseline-2026-10-20/inputs/u85560.pt \
  .work/eval-baseline-2026-10-20/inputs/u86800.pt
```

Mac holdout pass (2026-10-20, local checkpoints, `danlm-venv` torch):

```sh
PYTHONPATH=.:python .work/external/danlm-venv/bin/python -m train.checkpoint_average \
  --output .work/eval-baseline-holdout-2026-10-20/avg4-u86800.pt \
  .work/lineage/main/u82460-5404M.pt \
  .work/lineage/main/u83948-5502M.pt \
  .work/lineage/main/u85560-5607M.pt \
  .work/velocity-milestones/u86800.pt
```

| Field | Cloud rebuild | Mac holdout rebuild | Mac 2026-10-18 |
|---|---|---|---|
| model_digest | `0ca9f30ba4717995991314ba3575c1128a10a28c6a7f74a3da71c37bec52c8e9` | same | `0ca9f30ba4717995…` |
| file SHA-256 | `21cc19d91fa299e1…` | `45367c1c5e2591cc…` | `418a4b60b8817c4a…` |
| lineage / updates | `history_ppo-2026092801-d1d612ae` / 86,800 | same | same |
| optimizer / resume | empty optimizer; `averaged_from` set; resume refused | same | eval-only |

File SHA differs because `averaged_from.inputs[].path` strings differ; actor and
critic weights match via `model_digest`. Do **not** pass avg4 to
`train.history_ppo --resume`.

Mac digest confirmation before holdout arena (file SHA for m7/u86800;
`model_digest` for avg4):

| Artifact | Check | Match |
|---|---|---|
| m7 | file `fa10bd1ac34c91e1…` | yes |
| u86800 | file `639ccda15452ef10…` | yes |
| avg4 | `model_digest` `0ca9f30ba4717995…` | yes |

Selection was **not** re-run: digests match the October 18 Mac report.

## DanLM 4k paired — selection set (seed `20260929`)

Protocol from the Mac report: `eval.danlm.arena duplicate`, 4,000 deals,
seed `20260929`, `tribute_fraction=0.5`, house rules, then `eval.paired` for
A−B. Deal-set identity
`deal_set_sha256 = 4f6952abdcb4a0fddc76803c26f9265626cff132e264b7326c6de2e1f31a80cc`
(recomputed on Mac 2026-10-20 from `eval.danlm.arena.generate_deals`).

Standing numbers from [paired-eval-mac-2026-10-18.md](paired-eval-mac-2026-10-18.md)
(not refreshed this pass):

| Model | Levels / round | 95% interval | Round win rate | Scored |
|---|---:|---|---:|---:|
| m7 u73780 | −0.334 | [−0.372, −0.297] | 43.6% | 3,992 |
| u86800 | −0.312 | [−0.350, −0.277] | 44.3% | 3,998 |
| avg4 | **−0.214** | [−0.252, −0.179] | 45.6% | 3,997 |

| Pair | Difference | Bootstrap 95% |
|---|---:|---|
| u86800 − m7 | +0.021 | [−0.025, +0.067] |
| avg4 − u86800 | **+0.097** | [+0.053, +0.141] |
| avg4 − m7 | **+0.119** | [+0.074, +0.165] |

## DanLM 4k paired — holdout set (seed `20261020`)

Mac worktree
`GuanZero-worktrees/eval-baseline-avg4-holdout` on
`cursor/eval-baseline-avg4-holdout-3f7a`, Python 3.12 `danlm-venv`, DanLM
`Developer/Projects/GuanZero/.work/external/DanLM` checkpoint
`ckpts/DanLM_v1/dansformer_v1_best_eval.pt`. Same arena settings as selection:
4,000 duplicate deals, seed **`20261020`**, `tribute_fraction=0.5`, 12 workers,
`OMP_NUM_THREADS=1`. `eval.paired` defaults (bootstrap seed 0, 5,000
resamples). Deal-set identity
`deal_set_sha256 = 059b4112c2faf6a66d0a0f138611472d24afd896cebd975132c935afbeb1871e`
(0 overlap with selection; recomputed before the runs).

| Model | Levels / round | 95% interval | Round win rate | Scored | Excluded | Wall |
|---|---:|---|---:|---:|---:|---:|
| m7 u73780 | −0.358 | [−0.395, −0.324] | 43.1% | 3,996 | 4 | 229 s |
| u86800 | −0.347 | [−0.386, −0.312] | 43.5% | 3,996 | 4 | 223 s |
| avg4 | **−0.251** | [−0.288, −0.214] | 44.9% | 3,994 | 6 | 211 s |

| Pair | Difference | Bootstrap 95% | Paired SE | Unpaired SE | Correlation | Common | Exclusion bound |
|---|---:|---|---:|---:|---:|---:|---|
| u86800 − m7 | +0.011 | [−0.037, +0.057] | 0.024 | 0.027 | 0.207 | 3,992 | [−0.001, +0.023] |
| avg4 − u86800 | **+0.096** | [+0.051, +0.140] | 0.023 | 0.027 | 0.291 | 3,991 | [+0.082, +0.109] |
| avg4 − m7 | **+0.107** | [+0.061, +0.154] | 0.024 | 0.027 | 0.228 | 3,990 | [+0.092, +0.122] |

Holdout round-win rate differences (paired): avg4 − u86800 **+0.014**
[+0.004, +0.025]; avg4 − m7 +0.018 [+0.007, +0.029]; u86800 − m7 +0.003
[−0.008, +0.014].

Every failed leg is `their_choice_missing` (same exclusion class as the
October 18 selection runs). Exclusions remain play-dependent.

**Reading (keep selection and holdout separate):** on holdout, u86800 is still
not separated from m7. avg4 remains ahead of both on levels/round; neither
avg4 − u86800 nor avg4 − m7 interval (nor exclusion bound) includes 0. The
holdout avg4 − u86800 gap (+0.096) matches the selection gap (+0.097) closely.
Absolute levels/round are a few hundredths worse on holdout than on selection
for all three models; that is deal-set variance, not a digest mismatch.

## Selection vs holdout deals

| Set | Seed | Deals | `deal_set_sha256` | Role |
|---|---:|---:|---|---|
| **Selection** | `20260929` | 4,000 | `4f6952abdcb4a0fddc76803c26f9265626cff132e264b7326c6de2e1f31a80cc` | Candidate ranking (m7 / u86800 / avg4 Mac paired Oct 18; also Velocity V1T milestone ladder) |
| **Holdout** | `20261020` | 4,000 | `059b4112c2faf6a66d0a0f138611472d24afd896cebd975132c935afbeb1871e` | Confirmation; **0** overlapping deals with selection; Mac DanLM 4k done 2026-10-20 |

## Newer main-lineage context (V1T only)

Velocity milestone ladder vs DanZero-V1T ONNX (`eval.v1t`, seed `20260929`,
4,000 deals) — not DanLM, not paired avg4:

| Checkpoint | Levels / round vs V1T | Round win rate |
|---|---:|---:|
| m7 u73780 | +0.1045 | 50.19% |
| m8 u81344 | +0.0425 | 48.79% |
| m9 u89032 | +0.16925 | 50.95% |
| m10 u96596 | +0.1195 | 50.11% |

Main training on Velocity was held (`gz-main` JobHeldUser) at probe time;
segment files continue past m10 to ~u99572. Treat m9/m10 as healthy saves for
**future** DanLM work, not as automatic replacements for the frozen avg4
selection triad without their own DanLM (and preferably holdout) scores.

## What this does / does not authorize

- **Does:** shared pointers for m7, u86800, avg4; identical avg4 weights across
  Mac/cloud; selection and holdout DanLM paired receipts; reminder that avg4
  is eval-only.
- **Does not:** claim that m9/m10 beat avg4 on DanLM; authorize resume or
  boundary changes; rent new GPU jobs; merge selection and holdout into one
  score.

## Artifacts

- Cloud (gitignored): `.work/eval-baseline-2026-10-20/`
- Velocity: `~/guanzero/eval-baseline-2026-10-20/avg4-u86800.pt`,
  `avg4-record.json`
- Mac holdout (gitignored): `.work/eval-baseline-holdout-2026-10-20/` —
  `avg4-u86800.pt`, `{m7,u86800,avg4}-vs-danlm-holdout.json`,
  `{u86800-minus-m7,avg4-minus-u86800,avg4-minus-m7}-holdout.json`
- Prior evidence: [paired-eval-mac-2026-10-18.md](paired-eval-mac-2026-10-18.md),
  [paired-eval-checkpoint-average-2026-10-18.md](paired-eval-checkpoint-average-2026-10-18.md)
