# Main u184140 versus u155248 on 20,000 fresh DanLM deals — November 10, 2026

On 19,954 jointly scored deals, u184140 gained +0.00253 net levels per
round and lost 0.073 percentage points of round wins relative to u155248.
Both paired 95% intervals include zero. The earlier fixed-set point estimates
were +0.02116 net levels per round and +0.250 percentage points. Both remain
within the new intervals.

| Metric on common deals | u155248 | u184140 | Change (95% paired bootstrap CI) |
|---|---:|---:|---:|
| Net levels per round | −0.20625 | −0.20372 | +0.00253 [−0.01854, +0.02315] |
| Round win rate | 46.281% | 46.209% | −0.073 pp [−0.576, +0.411] |

Both checkpoints were rerun from scratch on the same new 20,000-deal set
(seed 2026111001, tribute fraction 0.5). All generated hands were unique.
They had zero overlap, even ignoring seat order, with hands regenerated from
25 historical DanLM deal configurations represented in local reports and
receipts. The ordered deal-set SHA-256 was
`29b5e4759865773828be387b3b829d14d1d3814d7d7ec86279d322dd865b8562`.

The arena, original DanLM v1 weights, greedy play, heuristic tribute, DanLM
referee and two swapped-seat legs per deal matched the
[4,000-deal comparison](main-u184140-danlm-2026-11-10.md).
Both checkpoint file hashes matched that comparison. The clean arena worktree
at `eb1af98416edc04c670515d2393b65fbee17968e` ran each endpoint on local CPU
with 12 workers. u184140 took 1,031 seconds and u155248 took 1,026 seconds.
The paired intervals use 5,000 whole-deal bootstrap replicates with seed 0.
Both score and win-rate differences were independently verified from the raw
leg records.

The adapter excluded 28 u184140 deals and 21 u155248 deals for
`their_choice_missing`, with three excluded by both. Across all 20,000
deals, assigning the 46 missing paired outcomes their worst or best legal
scores bounds the net-level difference at [−0.011275, +0.016325] on this
fixed set. This is a missing-score bound.

The checkpoint gap is 1.893 billion decisions, including 1.859 billion
after the switch at u155773. These endpoints leave open whether further
training under the current recipe will improve the DanLM result. The
comparison measures single-round strength against DanLM, and the recipe
switch and additional training occurred together.

The arena outputs, paired output, input manifest, freshness audit and
verification summary are in `.work/danlm-fresh-20000-2026-11-10/`.
