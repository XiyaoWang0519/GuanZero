# Exploration roadmap after the T7 entropy diagnosis (September 26, 2026)

Status: discussion record, not an implementation or result claim. Written while
the T7 campaign (`.work/history-response-speed-2026-09-26/`) was on seed 2.

Update, September 27: the queue and entropy-based success criterion below are
historical. The [Codex/Fable discussion and next steps](training-next-steps-2026-09-27.md)
recommend B control / fixed entropy 0.03 / epsilon 0.02 with three paired seeds,
plus optional unchanged-recipe continuation of all three B endpoints. Automatic
entropy and epsilon 0.10 are deferred. Use that record for the next preparation;
no new run was launched by the discussion.

## Diagnosis

All three T7 arms reached policy entropy 0.29--0.43 by update 700 and were
flat from about update 300; the omniscient critic's explained variance rose
to 0.55--0.64 and flattened at the same time. The only exploration is the
policy's own sampling (entropy coefficient 0.01); there is no hard floor.
This matches the earlier recipe campaigns, where entropy below 0.5 coincided
with the greedy plateau against B11 (-2.1 to -2.4 levels/round).

## Queue

1. **T7 finishes as planned.** Read the primary C minus B effect and whether
   all arms sit in the plateau. No changes to the running campaign.
2. **Exploration-floor arms.** Branch `fable/exploration-floor` (worktree
   `../GuanZero-exploration-floor`, commit `0a49302`) adds `rollout_epsilon`,
   `rollout_temperature` and `behaviour_weight_cap` with bitwise-identical
   defaults; see `exploration-floor-2026-09-26.md`. Arms: epsilon 0 / 0.02 /
   0.10 on the T7 winner's configuration, three seeds, same evaluation set,
   curves every ten updates. Success: entropy holds at or above 0.6 and the
   evaluation breaks the plateau. Merge with codex's branch first; user
   confirms the budget before launch.
3. **Branching rollouts (GRPO-style / TRPO "vine").** Clone the engine state at
   learner decisions, branch the top-k candidates, roll each out with the
   current policy to round end or to a critic bootstrap, and compare within the
   group. k, rollouts per branch, branch fraction and horizon are all
   configurable; the user wants k well above 4 and treats compute as
   available. Perfect-information rollouts are acceptable for value targets
   only, never as actor input. This is also the seed of endgame search.
4. **Endgame search as teacher**, late round only, once a non-plateaued base
   model exists.

## Dropped for now

- Temperature as an experiment arm: the network can rescale logits to undo
  it, and the cap-1 truncation damps it. The flag stays for later use with an
  entropy target.
- Event-style intermediate rewards (opponent passed, trick won): reward
  hacking risk; recast as auxiliary prediction targets instead.
- Curiosity / RND bonuses: the reward is not sparse and state novelty is
  noise under random deals.

## Project hygiene noted the same day

The repository has no remote and no pack files; iCloud had evicted 166 loose
git objects, which hung `git worktree add` until they were re-downloaded per
file with `brctl download`. A remote backup and moving the project out of the
iCloud-synced folder were recommended to the user.
