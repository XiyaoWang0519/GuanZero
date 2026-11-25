# Main u184140 versus u195450 cross-round history swap — November 12, 2026

Both frozen checkpoints scored the same 707 archived decision contexts
from 64 match seeds, using the October 16 history-swap sample and donor
assignments (waves 8–11, control and norm16 trajectories). The probe fixed
the current-round suffix, private observation, seat, legal candidates and
total prefix length. Each actor recomputed its own history encoding.
Duplicate original inputs agreed within 1e-7 in action probabilities.

| Past prefix replacement | Mean policy TV, u184140 → u195450 | Top action flips, u184140 → u195450 |
|---|---:|---:|
| Content only | 0.630% → 0.461% | 5/707 → 3/707 |
| Content and round/phase metadata | 0.640% → 0.439% | 6/707 → 4/707 |

With 5,000 paired match-seed bootstrap replicates, the content-only TV
change was −0.169 percentage points [−0.233, −0.105]. The flip-rate
change was −0.283 percentage points [−0.993, +0.417]. The corresponding
metadata-inclusive changes were −0.201 [−0.260, −0.143] and −0.283
[−1.268, +0.561] percentage points.

The sensitivity increase required for the follow-up match test was not
observed. These archived contexts measure policy response to a history
substitution. The playing value of correctly matched opponent histories
remains unmeasured for these checkpoints.

The checkpoint hashes match the November 12 DanLM comparison.
The runner and paired per-context results are in
`.work/kv-u195450-2026-11-12/`.
