# Main next-on / filtered continuation — November 8, 2026

The user authorized keeping next loss at 0.1, changing main to the
strength/behavior-filtered population, and retaining long-running automatic
continuation. This follows the
[plateau analysis](main-curve-2026-11-08.md).

## Handoff and identity

Old job 19820 received the existing USR1 graceful-stop signal, completed
with exit 0 and reason sigterm, and successfully synced its final checkpoint.
The switch starts at u155773, 10,208,739,328 decisions. A durable pre-switch
copy is preserved at
`~/guanzero/main-filtered-transition-2026-11-08/pre-switch.pt`,
SHA-256 `18a93d9479c00eefef5afbd310bfa82b072cba5f4f470f48adc086524a13b58a`.

Job 20589 resumed the same main actor, critic, optimizer and learner progress
using deployed source `src-league-resume-2026-11-03`, node-local identity
`ed082351e0662ae3`. The source's native population-schedule migration
rebuilds the pool from the resumed actor. Old filtered-arm models were not
imported. New recent snapshots and filtered archives accumulate from this
main checkpoint. Old environment trajectories are discarded on resume.

## Configuration and continuation

- next_coef=0.1, next_anneal_updates=0.
- population_archive_filter=strength_behavior, archive_every=32,
  archive_size=16, archive_share=0.5, archive_min_tv=0.05.
- Eight ranks, 128 environments and eight minibatch matches per rank.
- Two L40S GPUs, MPS off, full public history, FP32, fast-lg2 and page-span.
- No experimental update target. Each allocation is capped at 12 hours.
- Automatic successor 20590 waits afterok on 20589. Future normal
  deadline exits submit one successor after advancing finite checkpoint
  checks and successful durable sync. STOP and failure guards remain.

The initial three-GPU submission 20588 was cancelled while pending because
another user's allocation prevented immediate placement. Two GPUs were
chosen to resume promptly while retaining the global training batch.
The remote launcher is `~/guanzero/main-filtered-auto-2026-11-08.sbatch`.

## Validation and startup observation

Existing probe-resume, population and auxiliary-head CPU tests passed
24 tests with 2 skips on the deployed source. Live manifest confirms next
0.1 and the complete filtered schedule. At u155779, next_coefficient was
0.1000000015, actor gradient norm 0.41155, encoder gradient 0.08462,
value loss 0.76348 and next loss 0.34632, all finite.
This is an initial startup check. The first new checkpoint writeback and
the first filtered automatic handoff remain pending.

This is a user-selected long continuation. No concurrent unchanged-recipe
control was started, so later improvement alone cannot isolate the effect
of the filtered population from additional training.
