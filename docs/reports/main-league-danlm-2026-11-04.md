# Main and league endpoints versus DanLM — November 4, 2026

Three frozen checkpoints played the original DanLM v1 against the same
4,000 newly seeded deals (20261104, tribute fraction 0.5). Each deal
was played twice with the teams swapped. Both sides used greedy play,
DanLM refereed, and four local CPU workers ran per candidate through
`eval.danlm.arena duplicate`. The arena was the same source as the
October 30 evaluation (worktree commit eb1af98416edc04c670515d2393b65fbee17968e,
arena source SHA-256 `62182da26c65b5910f14f80e6359060a9fb3d85daa63f6c4d59714cb7296bf5b`).
DanLM checkpoint SHA-256 was `5341d7316bac1ead4098c796ea03f7e38fb47f001cc647a52be3ed0d15cd7244`.
Deal-set SHA-256 was `d556e3077538946413e8f52cafd7a785a0286ceed02a53cd39a7c188bcf93b00`.

| Candidate | Decisions | Deals scored | Net levels/round (95% CI) | Round win rate |
|---|---:|---:|---:|---:|
| main u111202 | 7,287,734,272 | 3,994 | −0.25588 [−0.29595, −0.21582] | 45.043% |
| near u130000 | 8,519,680,000 | 3,995 | −0.28260 [−0.31965, −0.24555] | 44.819% |
| filtered u130000 | 8,519,680,000 | 3,997 | −0.25294 [−0.28959, −0.21529] | 44.658% |

Main is Velocity's durable `runs/main/segments/lr-main/latest.pt` with
SHA-256 `5a0b60f7e2b9c867f40e2f830777c321a5dc2089bd10949391806a82229f2fd4`.
Both league checkpoint hashes matched their
[acceptance record](league-resume-2026-11-03.md#november-4-terminal-checkpoint-acceptance).

| A minus B | Common deals | Net levels/round (95% CI) | Win-rate gap, pp (95% CI) |
|---|---:|---:|---:|
| filtered − main | 3,991 | +0.00226 [−0.04635, +0.04949] | −0.388 [−1.516, +0.714] |
| near − main | 3,991 | −0.02744 [−0.07642, +0.02017] | −0.251 [−1.366, +0.852] |
| filtered − near | 3,992 | +0.02956 [−0.01591, +0.07665] | −0.163 [−1.240, +0.927] |

`eval.paired` resampled whole jointly scored deals, 5,000 replicates
with seed 0. Each excluded deal had a `their_choice_missing` failed leg.
All three candidates trail DanLM on net levels. No pairwise interval
excludes zero. Main and league endpoints have different training exposure,
and the earlier filtered probe-resume caveat remains. These are
single-round results through the DanLM adapter.

Raw per-deal records, logs and pairwise JSON files are in the local ignored
directory `.work/danlm-main-league-2026-11-04/`.
