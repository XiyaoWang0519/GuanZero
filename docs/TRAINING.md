# Training and evaluation: Transformer self-play

Current route, September 25, 2026: [DESIGN.md](DESIGN.md) v0.6 and
[STAGE_C_TODO.md](STAGE_C_TODO.md) T0--T8. Train a randomly initialized
history Transformer through self-play RL. Old MLPs are evaluation-only.

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

## Local development checks

The [history-stack follow-up](reports/history-stack-2026-09-26.md) adds optional
`--causal-sdpa` and `--rollout-kv-cache` paths. The
[bounded CUDA comparison](reports/history-stack-cuda-2026-09-26.md) passed
98 Python checks and measured 1.76x / 1.50x full-update KV throughput with
all-current / mixed seats. Both remain off by default: fixed-fixture numerical
equivalence and engineering throughput do not certify long-run playing quality.
Every optimization must preserve model capability, FP32, full event history,
canonical legal support and PPO/reward/population semantics. Controlled replay
and development-baseline strength checks remain the sustained quality gate.
The cache is public-only and partitioned by
policy/environment/match; learner updates invalidate it and PPO retains full
gradient recomputation. `--profile-collection` enables synchronized diagnostic
phase timers, which must be excluded from ordinary throughput comparisons.
`--batch-snapshot-policies` (off by default; `--resume-set` may switch it and
records the change) evaluates every frozen snapshot identity's rows of a
vector step in one merged actor call over stacked head weights. It is
acceptance tier 2 for snapshot seats only (FP32 reduction-order noise; the
same uniforms from the same generator in the same order); learner rows and
sampling are unchanged and snapshot seats bypass private graphs. CPU tests
pass; CUDA correctness and speed are not yet measured.
`bench.history_stack` alternates dense/SDPA/KV cases with fixed learning rules
and records full-update throughput/memory. CPU mode requires `--collect-only`
and charges a learner-cache rebuild each chunk unless explicitly disabled for
a frozen-cache diagnostic. Source changes preserve the original T4 archive;
they do not silently bypass checkpoint source identity for resume.

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
