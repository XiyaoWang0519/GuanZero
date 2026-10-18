# Paired DanLM rerun: u86800, m7 and a four-checkpoint average (October 18, 2026)

Status: done on CPU. These numbers are the real main-lineage checkpoints, not
the synthetic smoke test in
[the implementation receipt](paired-eval-checkpoint-average-2026-10-18.md).
No GPU, no training, and no change to the `play/` or Botzone defaults.

## Setup

Worktree `GuanZero-worktrees/paired-eval-checkpoint-average` on
`cursor/paired-eval-checkpoint-average-46ed` (`23ef6c7`). The checkout at
`Developer/Projects/GuanZero` was left on its current branch. The engine
extension was built in the worktree for Python 3.12. DanLM is
`Developer/Projects/GuanZero/.work/external/DanLM`, checkpoint
`ckpts/DanLM_v1/dansformer_v1_best_eval.pt`. Arena settings: 4,000 duplicate
deals, seed `20260929`, `tribute_fraction=0.5`, 12 workers,
`OMP_NUM_THREADS=1`. `eval.paired` used its defaults (bootstrap seed 0,
5,000 resamples).

The four averaged checkpoints are the four saved main-lineage files whose
update counts are at or before 86,800, oldest first, so the last input is
u86800 and supplies the averaged checkpoint's metadata. Saves on disk around
this point are about 1,200–1,600 updates apart. The denser ~500-update files
used for the October 4 average, and the synthetic names u85312 / u85808 /
u86304, are not on this machine. Later saves (u87048 and after) were not
included. m9 is not a file here.

| Input | Updates | Decisions | SHA-256 |
|---|---:|---:|---|
| `.work/lineage/main/u82460-5404M.pt` | 82,460 | 5.404B | `09b473d0aecff6743258656aefa474804cbf5ca1db00804d16d14b95f9a3a2d1` |
| `.work/lineage/main/u83948-5502M.pt` | 83,948 | 5.502B | `de7b39c363d09263231944403d55b074c0cee01ecfdb53d50cf65b136b00a855` |
| `.work/lineage/main/u85560-5607M.pt` | 85,560 | 5.607B | `7b0da599028332104bfdf07bc38e90c0a782c6903def53bbb38806fd769cc108` |
| `.work/velocity-milestones/u86800.pt` | 86,800 | 5.689B | `639ccda15452ef101561e836f684f238cb985bfcf412edd48fb063e383d4c720` |

`train.checkpoint_average` wrote
`.work/paired-mac-2026-10-18/avg4-u86800.pt` in the worktree (SHA-256
`418a4b60b8817c4aaf64484a5b54773f277b925d308b98aded928d05d8387a1d`, model
digest `0ca9f30ba4717995`). m7 is `.work/velocity-milestones/m7-u73780.pt`
(SHA-256 `fa10bd1ac34c91e173902be28bb22f8d31233de50f534fa89fdb6d092d74fefb`,
73,780 updates, 4.835B decisions). All five files share lineage
`history_ppo-2026092801-d1d612ae`.

## Against DanLM

Same deal set (`deal_set_sha256`
`4f6952abdcb4a0fddc76803c26f9265626cff132e264b7326c6de2e1f31a80cc`). Levels
per round are our team's mean net level gain. The u86800 and m7 headlines,
intervals, scored-deal counts and round win rates match the October 15 logs
in `.work/velocity-milestones/` to the recorded float.

| Model | Levels / round | 95% interval | Round win rate | Scored | Excluded | Wall |
|---|---:|---|---:|---:|---:|---:|
| m7 u73780 | −0.334 | [−0.372, −0.297] | 43.6% | 3,992 | 8 | 227 s |
| u86800 | −0.312 | [−0.350, −0.277] | 44.3% | 3,998 | 2 | 231 s |
| average of 4 | **−0.214** | [−0.252, −0.179] | 45.6% | 3,997 | 3 | 227 s |

## Paired differences

A − B on deals both runs scored. The bootstrap resamples whole deals. The
exclusion bound is the range of that difference over all 4,000 deals if every
deal excluded by either run had scored at ±3 levels.

| Pair | Difference | Bootstrap 95% | Paired SE | Unpaired SE | Correlation | Common | Exclusion bound |
|---|---:|---|---:|---:|---:|---:|---|
| u86800 − m7 | +0.021 | [−0.025, +0.067] | 0.024 | 0.027 | 0.226 | 3,990 | [+0.006, +0.036] |
| average − u86800 | **+0.097** | [+0.053, +0.141] | 0.023 | 0.027 | 0.295 | 3,996 | [+0.091, +0.103] |
| average − m7 | **+0.119** | [+0.074, +0.165] | 0.024 | 0.027 | 0.243 | 3,989 | [+0.103, +0.136] |

u86800 is not separated from m7: the interval includes 0. The average is
ahead of both, and neither interval nor exclusion bound includes 0. Pairing
narrows the standard error from about 0.027 to about 0.023–0.024. Round-win
rate moves with the levels: average − u86800 is +0.013 [+0.003, +0.024];
u86800 − m7 is +0.007 [−0.004, +0.018].

## Exclusions

Every failed leg is `their_choice_missing`, and every one of the 13 failed
legs records a `trick_reading`: a DanLM-side wild single had already been
mapped to our level-power reading earlier in that trick. That is the
mechanism in the implementation receipt, now checked on these three runs.
The dropped deals still differ by checkpoint (m7 drops 8, u86800 drops 2,
the average drops 3; only deal 2850 is shared by u86800 and the average), so
the exclusions are play-dependent, not a fixed bad subset of the deal set.

## Time and files

The three arena runs and three pairings ran from 20:47:53Z to 20:59:20Z
(11 minutes 27 seconds). Averaging the checkpoints took about 2 seconds.
Outputs, gitignored under the worktree `.work/paired-mac-2026-10-18/`:

- `avg4-u86800.pt`, `avg4-record.json`
- `u86800-vs-danlm.json`, `m7-vs-danlm.json`, `avg4-vs-danlm.json`
- `u86800-minus-m7.json`, `avg4-minus-u86800.json`, `avg4-minus-m7.json`
