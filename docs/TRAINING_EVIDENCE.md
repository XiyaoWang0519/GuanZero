# Training implementation receipts and experiment history

Chronology moved from `TRAINING.md` on October 6, 2026. These are dated
measurements and run-specific settings, not a current launch recipe. Use
[STATUS.md](STATUS.md) for orientation, [TRAINING.md](TRAINING.md) for operations
and the [report index](reports/README.md) for the complete evidence catalog.
Relative links and original section titles are preserved. Check code and the
run manifest before reusing flags, defaults, checkpoint paths or budgets.

## Implementation status and entrypoints

T7 connection ablation is implemented with `response_mode=none|auxiliary|explicit`;
the default remains `none`. The [three-arm plan](reports/history-response-plan-2026-09-26.md)
defines the public response target, paired seeds, fixed endpoints and approved
compute cap. `python -m infra.history_response_experiment prepare --output DIR`
creates reviewable kits only. The $6 cloud campaign was corrected after same-host device measurements:
CPU environment simulation with CUDA model inference and learning delivered
1.80x concurrent decision throughput versus CPU (`--rollout-device` is optional;
default shares the learner device).
[Device-placement report](reports/history-device-placement-2026-09-26.md).
The three-seed comparison finished on September 27 UTC ($3.84 of $6):
`auxiliary` beat PPO-only by about +0.3 levels per round (exploratory, three
seeds), `explicit` was not better than `auxiliary`, and all endpoints still lose
to B11. [Results](reports/history-response-results-2026-09-27.md).

The history-RL route completed a bounded RTX 4090 cold-start/resume/population
pilot on September 26 UTC. The [T4 receipt](reports/history-t4-pilot-2026-09-26.md)
records its measured configuration and limits. Larger runs still need their
own measurement and explicit compute budget; legacy MLP configs do not apply.

| Existing entrypoint/component | Current role |
|---|---|
| `train.dmc`, `train.ppo`, M1/B8/B11 configs | Old MLP training path; not the new launch command |
| `train.behaviour_probe` | Offline behavior-cloning/next-event diagnostic; no cold-start RL or strength claim |
| `train.history_model` | T1 contract: `PublicStream`, `HistoryActor` (public-only `encode_stream`, full canonical candidate head), `HistoryCritic`, `history_ppo` checkpoint marker. CPU-verified 2026-09-25 |
| `train.history_rollout`, `train.history_ppo`, `train.history_population` | Cold-start sequence trainer with same-lineage frozen seats, match-pinned assignment, learner-only rows, atomic optimizer/RNG/population resume, source/engine/token identity checks and device-local sampler. Verified in T4: 200 CUDA updates, real resume and 34,204 historical-policy decisions |
| `train.history_ddp` | Opt-in data-parallel trainer (September 26): each rank is a full `HistoryTrainer` with its own environments; per-minibatch gradients averaged over gloo on CPU; parameter checksum verified across ranks every update; resume supported. `world_size = 1` is bitwise the base trainer. Engineering-verified; changes the effective batch, so it needs a development A/B before use as evidence ([run 2](reports/history-recipe2-2026-09-26.md)) |
| `eval.history_policy` | T3 adapter: history policies receive every public action in every evaluator and reset per match/leg; `load_policy` returns it for `history_ppo` checkpoints |
| `train.belief_experiment`, `train.memory_experiment` | Historical supervised experiments; not architecture gates |
| `scripts/preflight.sh` | Existing MLP update/resume check; cannot certify the new sequence trainer |
| `eval.arena`, `eval.duplicate`, `eval.batched`, `eval.danlm.arena` | Evaluators with history hooks (T3); `eval.probes`, `eval.tribute`, `eval.stage_b_baseline`, `eval.record_games` remain unhooked and fail explicitly with a history policy |
| `infra.runpod`, `infra.watchdog`, `infra.sync` | Existing lifecycle components whose trainer wiring must be verified |

The old `infra/run_train.sh`/bootstrap path was built around MLP configs and
preflight. Verify its actual module, config and lifecycle behavior before
reusing it for the new trainer. No proposed config field is an implemented CLI.

## Verified bounded T4 workload

The [T4 run contract](reports/history-t4-readiness-2026-09-26.md) records the
prelaunch budget and frozen evaluators; the [completed receipt](reports/history-t4-pilot-2026-09-26.md)
records the approved execution and teardown. `infra.history_pilot prepare --kit DIR`
writes an allowlisted source archive, manifest, payload/setup.sh, payload/run.sh,
config and a separate local evaluation directory. It does not allocate compute.
Preparation uses Python 3.12 at `.work/external/danlm-venv/bin/python` on this
checkout because `.venv` is absent. Run with `PYTHONPATH=python:oracle:.`.

After explicit rental approval, `infra.runpod create` allocates the one reviewed
resource and `infra.history_monitor` immediately starts an independent provider
guard, waits for SSH, verifies the uploaded archive, builds and launches the
bounded workload. Neither old preflight nor MLP training is invoked. The monitor
syncs without deletion, inspects learning metrics every 30 seconds, checks
completed artifacts against remote SHA-256 values, deletes only its owned pod
and records provider readback. Guard availability still depends on this Mac
and provider connectivity; it is not a provider-enforced TTL.

`bench.history_ppo` first measures actual growing full-match prefixes for 4/8/16
environments at width 64, two layers, FP32, no TF32 and no snapshot mixing. It
reports all-seat collection decisions/sec, unique learned rows/sec, epoch
exposures/sec, prefix lengths and CUDA peak allocated/reserved memory. A case
must reach a mean prefix of at least 720, retain half of its throughput near
144 tokens and reserve less than 70% of device memory. These are predeclared
engineering stop rules, not universal hardware performance claims. A failed
sweep ends the workload; batch KV cache work then needs full-prefix parity,
weight-version invalidation and raw-history rebuild tests before another run.

Only a passing measured configuration enters the bounded population pilot.
The first completed updates create snapshots; a save/resume exercise restores
actor, critic, optimizer, sampler and population, discards incomplete rounds
and reconstructs new histories. The current implementation has no persistent
cache. Evaluation reads only the frozen development deals through
`python -m eval.history_frozen --freeze FILE --candidate FILE --output FILE`.
It writes duplicate raw legs and same-seed full-match seat-swap pairs, with
whole-pair bootstrap intervals. There is no final-test switch.
`--backend batched --device mps` (or `cuda`) plays all deals and match slots in
VecEnv waves with the collector's KV cache. On October 1 2026 it reproduced the
scalar CPU reports of u2623, u3534, u4309 and u4650 exactly (1,024 deals, 256
match pairs) in 2-3 minutes per checkpoint on the Mac's MPS, against 30-55 minutes
scalar. Agreement within measurement error, not bit identity, is the acceptance
bar for other devices; do not mix tables from different evaluators without it.

The completed run selected eight environments: collection retained 67.2% of
its speed as mean prefix grew from 166 to 754 tokens. It completed 200 updates
and 63,948 learner training rows. All 54 remote artifacts were hash-verified,
the pod was deleted, and independent provider readback showed zero pods / $0
hourly spend. Initial/final development scores improved but full-match wins
remained 0/16 against each frozen MLP. This is no playing-strength promotion.

The population pilot reached 2,403-token prefixes and 24.31 GB allocator
reservation despite only 1.55 GB peak allocated tensor memory. The short-sweep
memory margin therefore does not certify sustained headroom. Profile before
scaling; no persistent KV cache or asymptotic scaling improvement is claimed.

## Recipe runs after T4 (September 26)

Five bounded CPU-on-pod recipe runs are summarized in
[the recipe summary](reports/history-recipe-summary-2026-09-26.md). Development
findings, all against the batch-size kit's A-arm control with in-pod controls:
two PPO epochs is a seed-robust improvement (the `HistoryPPOConfig` default is
already `epochs=2`; the T4/batch kits overrode it to 1); data-parallel
collection and current-policy-only self-play speed learning further; four
epochs, a 16-snapshot population, GAE lambda 1.0 and critic lr 1e-3 were worse.
None is a strength claim. The kit tooling (`prepare.py`, `arm.py`,
`arm_ddp.py`, `recipe.py`, `lifecycle.py`, `analyze.py`, `pooled.py`) lives in
`.work/history-recipe-kits/`; `lifecycle.py` keeps the cumulative ledger.
The small history model trains faster on the pod's CPUs than on its GPU;
training on CPU is the FP32 reference path covered by the equivalence tests.

## Speed options after the refactor (September 29)

The [refactor report](reports/training-stack-refactor-2026-09-29.md) adds
four opt-in, resume-overridable options: `--rollout-paged-cache` (one page
pool for every identity's public KV cache; bitwise on CPU; replaces
`--rollout-triton-cache`), `--batch-snapshot-encoder` (one merged snapshot
encode per step), `--learner-length-groups N` (learner encode in length
groups, each stream only as far as its rows read) and `--rollout-page-span`
(page-padded rather than power-of-two attention spans). The last three are
tier 2. Default behaviour is bitwise unchanged on CPU. On an M4 Pro CPU all
four together took 87 s per 10 production-architecture updates against
149 s; GPU speed is unmeasured. `python -m bench.history_arms` compares
options in-run from a checkpoint without touching the lineage; the report
lists the GPU gate and A/B commands.

Two learner-side changes followed (September 29). Always on, tier 1: the
batched match attention's layout (each row's rank within its match and the
slot count) is planned on the host in `training_batch` and uploaded with the
batch, removing a device bincount/argsort and one host sync per attention
call (per length group); the auxiliary mode's zero response bridge is skipped
under `no_grad`/inference, adding the same exact zero as before. Opt-in, tier
2, resume-overridable: `--learner-chosen-response` (auxiliary mode only) runs
the response head on executed candidates only, the rows its loss reads; the
policy still scores every legal candidate. `tests/test_history_learner_layout.py`
checks the layout bitwise under strict deterministic algorithms and the chosen-only
loss/gradients to FP32 tolerance. The [September 30 RTX 4090 comparison](reports/history-learner-cuda-2026-09-30.md)
passed all eight new CUDA cases and matched default losses/actor gradients bitwise
against origin/main on a frozen 2,047-row batch. Chosen-only improved the grouped
learner probe about 10%, but complete PPO updates improved only 1.5%, within
process variation; ungrouped updates showed no gain. This does not establish a
reliable whole-update speedup or playing-strength improvement. Existing packed
float-field offsets are preserved because odd-row alignment changes reduction
bits. The chosen-only flag remains opt-in.

The [October 1 host optimization](reports/training-stack-2026-10-01.md) factors
learner match IDs once per compact buffer, batches response-label calculations,
and vectorizes merged snapshot layouts. Trainer-owned collection validates KV
cache weights at the start and end of its frozen interval; standalone collectors
keep per-encode checks. Empty/singleton packed tensor transfers also handle
non-unit strides. These changes preserve the existing configuration and packed
float alignment. Local component gains range from 1.18x for batch preparation
to 14x for response labels, with exact CPU update/replay comparisons. Complete
PPO and GPU throughput gains remain unmeasured; no numeric-mode flag was promoted.

The [October 1 speed deep dive](reports/speed-deep-dive-2026-10-01.md) reads the
u2623-u4902 run's counters and adds, CPU-verified only: `--rollout-prefill-learner-cache`
(opt-in, resume-overridable, tier 2 on learner seats: the learner's public KV
cache is rebuilt for all current matches in one batched pass per chunk at the
start of each collection); a separate page pool for the learner under
`--rollout-paged-cache`, returned before every PPO update; `--profile-learn`
(synchronized learn-phase times in `learn_phase_seconds`); and a kernel-level
profile of chosen updates through `GUANZERO_TORCH_PROFILE_DIR` and
`GUANZERO_TORCH_PROFILE_UPDATES`. With every new flag off, four CPU updates
match `main` exactly in three configurations. No GPU measurement exists yet; the
prepared kit is `.work/speed-prep-2026-10-01/`.

## Auxiliary heads on the shared encoder (October 2)

`--aux-heads next,belief,outcome` (any subset; `train/history_aux.py`, report
[aux-heads-2026-10-02](reports/aux-heads-2026-10-02.md)) adds supervised heads
that read the encoder or decision state and never enter the scoring path:
`next` predicts the next public token at every encoded stream position (all
seats and phases; factorised over the token's fields, not a flat action
vocabulary), `belief` predicts for each card the relative seat of every unseen
copy from the stored hidden counts, `outcome` regresses the acting team's round
return. Their summed loss enters the actor loss with `--aux-coef` (default 0.1),
ramped linearly over `--aux-warmup-updates` updates from the heads' insertion.
Metrics: `aux_loss`, `aux_coefficient`, `next_loss`, `next_type_accuracy`,
`next_seat_accuracy`, `next_cards_exact`, `belief_loss` against
`belief_baseline` (the hand-size proportional guess that uniform determinization
implies), `outcome_loss` against `outcome_baseline` (target variance).

A trained lineage takes the heads on resume:
`--resume latest.pt --resume-set aux_heads=next,belief,outcome --resume-set
aux_coef=0.1 --resume-set aux_warmup_updates=200`. Head weights start fresh,
every other weight and its Adam moments continue, the policy is identical at
insertion (checked bitwise on u9989's weights), the change is recorded in
`config_changes` and `progress.aux_start_update`, and old population snapshots
load without head weights (snapshot digests and saved snapshots exclude heads).
Heads cannot be removed. With `aux_heads` empty the trainer is bitwise the
previous code (three single-threaded CPU updates in the production
configuration match commit `fabcb71`). CPU cost at the production architecture:
learn +4 to +6 percent, collect unchanged. Readiness: CUDA-gated and run
(Oct 3 2026): from u9989, one night, control -1.635 / next -1.649 / all three
heads **-1.558** against DanLM (±0.035; report). The heads are a production
option; a resume with `--resume-set aux_heads=...` is the launch path. The lineage
continued from the full arm's endpoint u15094 (decision of Oct 3 2026).

## Learning-rate decay (October 4)

`--resume-set lr_final=3e-05 --resume-set lr_decay_start=U --resume-set
lr_decay_updates=N` scales both optimizers' rates linearly from `lr` to
`lr_final` between absolute updates U and U+N, then holds; `lr_scale` is
logged per update; off by default (`e2b8862`). Run once (Oct 3/4, from u15094,
[report](reports/lr-decay-weight-avg-2026-10-04.md)): decay vs constant-rate
control +0.052 [+0.000, +0.109] head to head, indistinguishable against DanLM
(-1.357 vs -1.356). Kept for release models, not routine nights. **At the time of this October 4 report, the main
lineage continued from the constant-rate arm's endpoint u20264 (1,328.0M
decisions, -1.356 against DanLM),
`.work/lr-decay-2026-10-03/kits/main/download/results/segments/lr-main/latest.pt`.**

## Planted-habit diagnostic (September 30)

Diagnostic only, with the three exceptions approved on September 30 (fixed
styled opponent pack, continuation from a trained checkpoint, ORACLE reads the
true style); see the [plan](reports/history-habit-diagnostic-plan-2026-09-29.md)
and [phase 0](reports/history-habit-phase0-2026-09-30.md). `--habit-pack CKPT`
replaces the snapshot population with a frozen copy of CKPT whose logits get
`z * habit_strength * f(a)` on the `--habit-axis` feature, z = ±1 fixed per
match; the learner is one team (two seats) per match. `--habit-init CKPT`
continues from CKPT's weights and Adam moments at the configured learning rate
(lineage `habit-<view>-...`, `habit_init` recorded). `--habit-view full|round|oracle`:
ROUND reads a per-round stream from position 0 (`RoundEventStore`), ORACLE adds
a zero-initialised style embedding to the private query
(`HistoryPolicyConfig.style_input`, recorded in checkpoints only when on).
Requires `snapshot_updates 0`, no merged snapshot inference, no private graphs
for ORACLE. `eval/history_habit_eval.py` plays a checkpoint against the pack
through the same collector. `tests/test_history_habit.py` checks, per view and
with the production CPU flags, that collected and recomputed log-probabilities
agree and that the first PPO minibatch starts at ratio 1. The [phase 1a receipt](reports/history-habit-phase1a-2026-09-30.md) records
completed CUDA gates and ORACLE/ROUND training. FULL was not run; each arm had
one training seed, so the result does not establish that history is harmful.

