# Training and evaluation: Transformer self-play

Current route, September 25, 2026: [DESIGN.md](DESIGN.md) v0.6 and
[STAGE_C_TODO.md](STAGE_C_TODO.md) T0--T8. Train a randomly initialized
history Transformer through self-play RL. Old MLPs are evaluation-only.

## Implementation status and entrypoints

There is not yet a verified cold-start history-RL launch command. Do not
rename a legacy config and describe it as a Transformer run. Add executable
commands here only after T1--T4 supply the implementation and verification.

| Existing entrypoint/component | Current role |
|---|---|
| `train.dmc`, `train.ppo`, M1/B8/B11 configs | Old MLP training path; not the new launch command |
| `train.behaviour_probe` | Offline behavior-cloning/next-event diagnostic; no cold-start RL or strength claim |
| `train.history_model` | T1 contract: `PublicStream`, `HistoryActor` (public-only `encode_stream`, full canonical candidate head), `HistoryCritic`, `history_ppo` checkpoint marker. CPU-verified 2026-09-25 |
| `train.history_rollout`, `train.history_ppo` | T2 cold-start sequence trainer: match event store, sequence rollout buffer, PPO with full-prefix recompute, atomic checkpoint/resume, CLI `python -m train.history_ppo`. CPU smoke only (8 envs, 3 updates); not a pilot entrypoint until T4 records GPU throughput and a bounded run |
| `eval.history_policy` | T3 adapter: history policies receive every public action in every evaluator and reset per match/leg; `load_policy` returns it for `history_ppo` checkpoints |
| `train.belief_experiment`, `train.memory_experiment` | Historical supervised experiments; not architecture gates |
| `scripts/preflight.sh` | Existing MLP update/resume check; cannot certify the new sequence trainer |
| `eval.arena`, `eval.duplicate`, `eval.batched`, `eval.danlm.arena` | Evaluators with history hooks (T3); `eval.probes`, `eval.tribute`, `eval.stage_b_baseline`, `eval.record_games` remain unhooked and fail explicitly with a history policy |
| `infra.runpod`, `infra.watchdog`, `infra.sync` | Existing lifecycle components whose trainer wiring must be verified |

The old `infra/run_train.sh`/bootstrap path was built around MLP configs and
preflight. Verify its actual module, config and lifecycle behavior before
reusing it for the new trainer. No proposed config field is an implemented CLI.

## Local development checks

Use the existing Python environment with the matching C++ extension. Fresh
installation uses Python 3.11+, C++20, CMake 3.24+ and Ninja:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt -r requirements-train.txt
./scripts/check.sh
```

The check script covers the engine and existing Python suites. Inspect its
current options before adjusting worker counts. Local tests and bounded
plumbing checks do not establish GPU throughput or playing strength.
Sustained model training runs remotely, not on this Mac.

The Transformer-specific readiness checks must cover:

1. Fresh actor/critic/heads and optimizer states; no old MLP checkpoint or
   replay loaded, including indirect reference and pool imports.
2. Every engine-provided canonical candidate remains selectable. No MLP
   top-k, union-of-M1-candidates or reference-KL path survives in the learner.
3. Public prefixes include all observed seats and passes in order. No future
   action, opponent hand/legal list, private tribute flag or engine-private
   forced-pass flag enters the actor or shared cache.
4. Full-match history boundaries, prefix carry-over, reward attribution and
   gradients into the history encoder are correct across PPO minibatches.
5. Stored collecting policy version, candidates and loop count reproduce
   behavior log-probabilities; old-policy PPO ratios remain valid.
6. Cache/full-prefix parity, invalidation after weight updates, and isolation
   across environments, matches, seats and different snapshot weights.
7. Actual optimizer update, atomic checkpoint, interruption, resume, finite
   parameters and population-state restore for the new trainer.
8. Scalar/batched/external evaluators feed every available public event and
   reset history correctly; unsupported history policies fail explicitly.

## Freeze the run manifest before the remote pilot

These are required manifest contents, not a ready-to-use config schema:

| Area | Record |
|---|---|
| Identity | Source revision plus dirty-source digest, engine digest, architecture, tensor/token schema, random seeds |
| Initialization | Fresh actor/critic/heads; no teacher checkpoint, optimizer or old replay |
| Observations | Full raw public match history, positions/seat/round/phase, private query fields, explicit reveal policy |
| Actions | Canonical action config and complete support; all reductions declared |
| Learning | PPO/GAE/reward boundaries, optimizer and entropy schedule, auxiliary losses and their own-self-play provenance |
| Population | Current/past Transformer identities, sampling schedule, match-stable assignments, snapshot participation counters |
| Recurrence | Standard or looped, supported depths, training-depth distribution, stored per-action depth, gradient/truncation policy |
| Evaluation | Frozen MLP checkpoint hashes/settings, development seed set, unopened final seed set, checkpoint-selection rule |
| Compute | Actual GPU/CPU quota/RAM, precision, environment count, cache/replay budget, evaluation resource share |
| Lifecycle | Explicit time/cost cap, guard, save/sync interval, artifact destination and verified teardown mechanism |

Training seats use current/past Transformer versions only. The initial
population consists of current-policy copies; frozen snapshots enter once
they exist. Fixed heuristic tribute/back-tribute is a declared exchange-only
exception, identical across arms. No MLP, greedy or styled play opponent is
silently added to improve a weak cold-start curve.

## Remote GPU pilot

Choose hardware and budget for the actual history/loop workload using
[COMPUTE_GUIDE.md](COMPUTE_GUIDE.md), not the previous MLP throughput.

1. Inspect provider inventory and existing owned resources; obtain the live
   quote and effective CPU quota. Record both total budget and elapsed-time cap.
2. Provision only the intended node under its unique manifest. Reconcile an
   ambiguous create response before retrying. A machine still initializing
   is not a failed placement; wait for runtime/SSH evidence.
3. Start the independent guard and verify ownership, deadline and teardown
   scope. A local guard requires the local machine/network to remain available;
   it is not a provider-enforced TTL. Do not assume a pod-scoped credential
   can delete the pod unless that operation was verified.
4. Upload an explicit source allowlist and the run manifest. Exclude secrets,
   local credentials, `.git`, unrelated `.work` data and old training datasets.
   No MLP checkpoint is needed by the training process. If evaluation shares
   the host, keep evaluator assets and outputs outside the training manifest.
5. Verify CUDA rather than silently falling back to CPU. Build the correct
   extension and run the new trainer's bounded update/resume checks.
6. Measure history lengths, cache/activation memory, active snapshots,
   collect/learn time and decisions per second. For loops also measure latency
   at the declared depths. Reduce environment count before silently truncating
   match history. Record any declared context ablation separately.
7. Run the bounded pilot and immutable development evaluations. Continue only
   within the actual approved experiment scope and measured resource limits.

The original fixed reward is the engine's per-round team level return, with
level-A handling preserved. Full-match win rate is an independent metric.
Do not make old MLP agreement or old MLP evaluation scores an online reward.

## Checkpoint, resume and teardown

Persist model/critic/auxiliary parameters, optimizers, counters, RNG state,
config, source identities and Transformer population metadata. Save immutable
evaluation checkpoints alongside an atomically replaced latest checkpoint.
Plan save plus sync intervals to bound recoverable lost work to ten minutes.

Resume only the same verified Transformer lineage/config. Restoring weights
alone is a warm start, not exact training-state recovery. If environments
restart, explicitly discard partial trajectories and caches and reconstruct
fresh histories; if live state is restored, verify its exact consistency.
Never resume an MLP checkpoint as the new actor or reuse old-weight caches.

Sync completed files without deleting destination data. Verify source/output
manifests and hashes, selected checkpoints, logs, raw evaluation legs and
population records before planned teardown. Interrupt/failure paths must
retain the last valid checkpoint and record any incomplete artifacts.

`infra.runpod` provides inspect/status/guard/delete operations; pass an explicit
credential source and unique resource manifest. Secrets stay out of logs and
uploaded source. Provider teardown must target the owned resource and be
verified by provider readback (`confirmed_gone` for that helper), including
remaining hourly spend/resources. Stopping Python is not stopping billing.
Do not keep a failed node running indefinitely because artifact transfer failed;
follow the bounded shutdown plan and report any loss.

## Independent evaluation

Freeze MLP evaluator checkpoint hashes and their exact policy semantics before
the pilot. An old checkpoint may load its own reference inside the evaluator
to reproduce its old behavior; that reference must not enter the learner.
Retain raw paired legs, complete-match records, exclusions and runtime settings.

Existing MLP-only evaluator example (runnable with an actual compatible
checkpoint; it does not certify support for a new Transformer checkpoint):

```sh
PYTHONPATH=python:. .venv/bin/python -m eval.arena \
  --agent /absolute/path/to/legacy-checkpoint.pt --opponent greedy \
  --deals 1000 --matches 100 --seed 20260925 \
  --device cpu --output /absolute/path/to/evaluation.json
```

For the new player, add the verified history-aware invocation here after T3.
Use same-deal team swaps and full matches against fixed MLPs. Pair and bootstrap
at the deal/match level; do not count the two legs as independent deals.
Report net levels/round and complete-match win rate separately. Include
checkpoint strength versus cumulative training GPU-hours with uncertainty.

Use development seeds for monitoring. Freeze the final checkpoint, loop depth
and primary comparison before opening the final test set; do not train on
those games or expand sample size according to the observed result. Measure
failure categories through replays, rather than assigning motives from losses.

For history policies, verify public-event delivery on each chosen backend.
Batching can change RNG consumption and numerical ties; distinguish exact
deterministic parity from distributional equivalence. Match-memory policies
must not leak history between paired legs or unrelated matches.

Compare standard and looped policies at controlled total training cost and
comparable inference budgets. A same-checkpoint loop-depth curve answers a
different question from an independently trained architecture comparison;
report both. Auxiliary prediction accuracy and current-vs-current win rate
are not substitutes for independent playing strength.

DanLM remains external evaluation only. Its platform-specific runtime,
mapping failures and rule limitations are documented in
[the original calibration](reports/stage-c-danlm.md) and
[the later campaign](reports/kaggle-campaign-results-2026-09-26.md). Keep
failures/exclusions visible and do not claim the adapter makes legal sets
identical. Historical training commands/results remain in Git history and
dated reports; they are not the new experiment's launch instructions.
