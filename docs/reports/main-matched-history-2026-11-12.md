# u195450 matched versus shuffled opponent history — November 12, 2026

The correctly matched history did not improve the point estimates for
match wins or net levels per round. Both paired 95% intervals include zero.

| Metric | Matched | Shuffled | Matched minus shuffled (95% CI) |
|---|---:|---:|---:|
| Match wins | 113/128 (88.28%) | 117/128 (91.41%) | −3.13 pp [−8.59, +2.34] |
| Net levels per round | 1.0297 | 1.1373 | −0.1076 [−0.2233, +0.0133] |
| Net levels per match | 9.4688 | 10.3516 | −0.8828 [−1.7736, −0.0232] |

The first two metrics answer the primary comparison. Net levels per match
also reflects match length. The two arms played 1,177 and 1,165 rounds.
Five paired matches flipped in favor of matched history and nine flipped
in favor of shuffled history. Intervals use 10,000 bootstrap resamples of
64 seed clusters, retaining both seat assignments, with seed 0.

The frozen u195450 checkpoint played complete matches against the four
archived diverse Transformer opponents, using the same waves 8–11 and
opponent assignments as the archived trajectories used by the
[history swap probe](main-kv-swap-2026-11-12.md). Each arm used the same
match seeds, seats and per-seat sampling RNG. All 128 first-round action
hashes matched. Later play can change round state, tribute and match length
even with the same initial deal sequence.

In the shuffled arm, the learner's completed-round public token content
came from archived rounds against a different opponent policy, matching
seat orientation and favoring the same round index. The donor tokens
replace the whole table's past content for those rounds. Prefix length,
round and phase metadata and live current-round events are retained. Across
round-boundary cache rebuilds there were 5,361 replacements, 51 using a donor
from a different round index. Ten replacement attempts retained original
content for lack of a long enough donor. This intervention changes the
historical table context as a whole and cannot isolate opponent identity
from the other content in that context. The result concerns these four
opponents and this sample of matches.

The runner, logs and paired match records are in
`.work/kv-match-u195450-2026-11-12/`.
