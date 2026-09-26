# M2 historical implementation and evidence index

Updated September 25, 2026. M2's completed work is retained here as evidence.
The current work queue is [STAGE_C_TODO.md](STAGE_C_TODO.md): Transformer
self-play from random initialization, with old MLPs used only for evaluation.
The former belief-first gate, MLP continuation order and proposed distillation
steps no longer control future work.

## Recorded results

| Work | Recorded result | Evidence |
|---|---|---|
| Initial belief prototype | Small supervised experiment; did not establish playing strength | [report](reports/M2-belief.md) |
| Smoke-source scaled probe | Collected with a six-update smoke checkpoint; distinct dataset from corrected M1 collection | [historical report](reports/M2-belief-scaled.md) |
| Corrected scaled belief probe | 100,000 rounds from trained M1 (34,496 updates). Mean test CE: flat 0.342105, no_history 0.336602, history 0.336681; history won one seed and lost two | [corrected report](reports/M2-belief-corrected.md) |
| Cross-round summary-memory probe | All three adaptation intervals crossed zero; original `v3_not_yet_justified` result retained | [memory report](reports/M2-memory.md) |
| Public history cache | Single-environment inference prototype with prefix parity checks; not a batched PPO integration | `train/history_cache.py`, [initial report](reports/M2-belief.md) |
| Counterfactual tribute experiment | Three learned heads failed to improve on the fixed heuristic on the shared paired evaluation | [A2 report](reports/M2-A2.md) |
| Perfect-information critic and PPO | Historical G1/G2 passed under their protocols | [critic](reports/stage-b-critic.md), [PPO](reports/stage-b-ppo.md) |
| League | Historical G3 passed as revised; active-model cap affected snapshot participation | [league](reports/stage-b-league.md) |
| Exploiters | G4 passed with the secondary clause at the margin; live-exploiter G5 not passed | [exploiter](reports/stage-b-exploiter.md), [live exploiter](reports/stage-b-live-exploiter.md) |
| Artifact and rental lifecycle | Recorded checkpoint/archive verification and completed teardown for those runs | [run record](reports/M2-next-run.md) |

## Interpretation that remains valid

`no_history` names the structural control in the supervised belief experiment.
The Stage B playing policy actually retained the v1 `GuandanModel` MLP;
the belief control was not transplanted into PPO. Earlier wording that
"Stage B uses no_history" should not be read as an architecture implementation.

Prediction gains and losses do not decide whether an end-to-end history
policy can improve play. The negative supervised results stand for their
datasets/objectives; they neither establish nor rule out gains from direct
history-based RL or recurrent internal computation.

Seven of nine corrected belief fits reached the configured step cap, so
convergence was not established. The memory experiment likewise remains a
bounded result. Keep the smoke-source and corrected-source datasets and all
raw reports separate. Neither dataset is a training source for the new actor.

## Reusable engineering

Public event logging and identity fields, causal encoder/cache prototypes,
owned trajectory storage, GAE/PPO utilities, privileged-label isolation,
checkpointing, evaluation adapters and lifecycle tooling can be reused after
their new contracts are verified. Reuse code rather than old learned weights
or old replay. Fixed exchange heuristics remain identical across the first
new arms; they are not play-phase opponents.

Rule corrections and source-digest caveats remain documented in
[RULES.md](RULES.md) and the original reports. Current provider state, test
counts and default checkpoint selection must not be inferred from an old
experiment's completion receipt.
