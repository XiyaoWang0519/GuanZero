# First training run

M0 is complete; its evidence and known external-baseline limitation are in
[`reports/M0.md`](reports/M0.md). The first run trains the v1 MLP with DMC and
heuristic tribute. It does not establish the M1 playing-strength gate. CUDA
graphs, the Transformer, learned tribute, PPO, the league and endgame search
belong to later experiments and are not prerequisites for this run.

## Install and verify locally

Python 3.11 or newer, a C++20 compiler, CMake 3.24 or newer and Ninja are needed.
Keep the engine and learner in the same Python environment.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt -r requirements-train.txt
./scripts/check.sh
./scripts/preflight.sh cpu .work/preflight
```

Preflight performs actual optimizer updates, writes `latest.pt`, resumes it,
and writes another checkpoint. Its small CPU configuration verifies plumbing;
it does not estimate GPU throughput or playing strength. Use a new preflight
directory for a new check. `PY=/path/to/python` selects another interpreter.
On Linux, install a CPU wheel first if this machine has no NVIDIA GPU:

```sh
.venv/bin/python -m pip install 'torch>=2.9,<3' --index-url https://download.pytorch.org/whl/cpu
```

Direct bounded development run:

```sh
PYTHONPATH=python:. .venv/bin/python -m train.dmc \
  --config train/configs/smoke.json --run-dir .work/smoke \
  --device cpu --max-updates 4 --max-seconds 120
```

## GPU image and node preflight

```sh
docker build -f infra/Dockerfile -t guanzero:train .
docker run --rm --gpus all \
  -v "$PWD/runs:/app/runs" guanzero:train \
  ./scripts/preflight.sh cuda runs/gpu-preflight
```

The image pins PyTorch 2.9.1 with the CUDA 12.8 wheel, an explicitly supported
combination in the [PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/).
The host must expose its NVIDIA driver through the NVIDIA Container Toolkit.
The CUDA preflight refuses to fall back to CPU and is the acceptance check on
the selected node. A rented RTX 5090 has now passed CUDA training and resume
using the official `runpod/pytorch:1.2.0-cu1281-torch291-ubuntu2204` image; see
[`reports/M1-runpod.md`](reports/M1-runpod.md) for progress, measured results and
provider cleanup status. The repository's `infra/Dockerfile` has not yet been
built and validated on that node; the official-image result does not establish
that separate image-build check.

CUDA preflight uses `gpu-smoke.json`: the full v1 architecture with 256
environments and real bf16-capable inference, learning, and resume. CPU
preflight uses the small model in `smoke.json`. After CUDA preflight passes,
perform a bounded run with `m1.json` to check memory and throughput at 4,096
environments before extending the walltime.

## RunPod test-node lifecycle

`infra.runpod` uses the standard library. Supply `RUNPOD_API_KEY` through the
environment, or explicitly select a local dotenv file with `--env-file`.
The selected file overrides the inherited key; the helper reads only its
`RUNPOD_API_KEY` assignment and never scans other projects. In the examples,
`KEY_FILE` is the selected credential-file path. Keep it local and out of logs.

```sh
# inspect includes account balance, existing pods and the live GPU catalog.
python3 -m infra.runpod --env-file "$KEY_FILE" inspect --cloud SECURE

# Choose RUNPOD_GPU_ID from available inventory; use a new manifest each time.
export RUNPOD_MANIFEST=.work/runpod/pod.json
mkdir -p .work/runpod
ssh-keygen -t ed25519 -f .work/runpod/id_ed25519 -N ''
python3 -m infra.runpod --env-file "$KEY_FILE" create \
  --manifest "$RUNPOD_MANIFEST" --gpu-id "$RUNPOD_GPU_ID" \
  --ssh-public-key .work/runpod/id_ed25519.pub \
  --max-hourly-usd 1.25 --budget-usd 5 --max-hours 1.8 \
  --min-cpus 16 --min-ram-gb 32

# Start immediately in a dedicated terminal and keep this machine awake.
python3 -m infra.runpod --env-file "$KEY_FILE" guard \
  --manifest "$RUNPOD_MANIFEST"
```

The example sets an estimated $5 allowance and a 1.8-hour deadline, leaving
room inside a two-hour authorization. Check the current quote before choosing
these limits. Creation verifies availability and GPU/storage price, then
checks the allocated rate. An excessive allocated rate triggers deletion.
The unique manifest records the account, name, new pod ID and deadline;
deletion must match that identity and cannot target preexisting pods.
Creation is never automatically retried. If the response is uncertain,
inspect the existing pods for the manifest's unique name before another launch.

The local guard begins deletion 120 seconds before its deadline and polls for
confirmation. It requires a live local process, an awake machine and working
provider access; it is **not a provider-enforced TTL or guaranteed spending
cap**. Keep a supervised test short and verify the guard process remains alive.
The injected pod-scoped key was absent from the SSH environment by default.
Using it remotely returned REST HTTP 403; a GraphQL own-pod read succeeded,
but shutdown-mutation permission remains unverified. Do not assume that key
provides a working independent teardown guard or upload an account key to make
that assumption work.

In a second terminal, obtain the allocated SSH address with:

```sh
python3 -m infra.runpod --env-file "$KEY_FILE" status \
  --manifest "$RUNPOD_MANIFEST"
```

Upload an explicit allowlist of build, training, evaluation and test sources
after checking the archive contents. Exclude `.env`, private keys, `.git`,
virtual environments, `.work`, previous runs and unrelated datasets. The SSH
private key stays on the local machine. Build the engine and run CUDA preflight
inside the official image before starting the production configuration.

From the uploaded repository root, `bash infra/runpod_bootstrap.sh` installs
build/test dependencies, builds the engine, runs focused regression tests and
executes CUDA preflight. Extract cross-platform source archives with
`tar --no-same-owner` when the pod volume disallows restoring host ownership.

Retrieve the run's checkpoints, metrics, runtime records and evaluation reports
to local durable storage, and verify their completeness before teardown:

```sh
python3 -m infra.runpod --env-file "$KEY_FILE" delete \
  --manifest "$RUNPOD_MANIFEST"
python3 -m infra.runpod --env-file "$KEY_FILE" status \
  --manifest "$RUNPOD_MANIFEST"
```

Require `confirmed_gone: true`. Termination destroys the pod's container and
volume contents; merely stopping training leaves compute billing active, and
stopping the pod still incurs volume storage charges. The guard exits early
after another process records and verifies deletion. Record measurements,
artifact locations and provider cleanup evidence in the run report.

## Start a supervised run

Look up the selected node's actual rate first. Set `BUDGET_USD`,
`HOURLY_RATE_USD`, and `MAX_HOURS` explicitly; there are no default prices.
`SPENT_USD` accounts for costs already charged against this cap, including
earlier launches. Rates should include known recurring storage/network costs;
fixed fees, billing granularity and time before launch need separate allowance.

```sh
# Supply these from the approved node quote and run budget.
export BUDGET_USD HOURLY_RATE_USD MAX_HOURS
export DEVICE=cuda CONFIG=train/configs/m1.json RUN_DIR=runs/m1-first
tmux new-session -s guanzero './infra/run_train.sh'
```

Inside Docker, pass those three values using `-e` and mount `runs` on durable
storage. Allow at least 60 seconds when stopping the container so the watchdog
can forward SIGTERM and wait for a checkpoint:

```sh
docker run --gpus all --stop-timeout 90 \
  -e BUDGET_USD -e HOURLY_RATE_USD -e MAX_HOURS \
  -e RUN_DIR=runs/m1-first -v "$PWD/runs:/app/runs" guanzero:train
```

The watchdog reserves shutdown time inside the smaller of the walltime and
estimated budget allowance. It asks the trainer to save, waits up to 30 seconds
(`GRACE_SECONDS`), and kills the entire training process group if it does not
exit. Budget/walltime stops return 124; SIGINT/SIGTERM return 130/143. The JSON
audit trail is `watchdog.jsonl`. Training progress is in `metrics.jsonl`; optional
TensorBoard logs can be viewed with the TensorBoard CLI using the run directory.

The budget is an elapsed-time estimate, not a billing API. Stopping Python or a
container does not stop cloud billing. For unattended paid runs, configure and
verify provider teardown separately, and use provider-side spending limits.
No instance is provisioned, stopped or deleted by default.

## Checkpoint, resume and sync

`latest.pt` is replaced atomically. Normal exit and SIGINT/SIGTERM save a final
checkpoint; a forced kill recovers the most recent completed checkpoint. The
trainer's configured checkpoint interval must be at most 600 seconds. Resume
uses the same model/training configuration and restores learning state;
environments restart with fresh deals and partial trajectories/replay are
discarded. `--max-updates` is the cumulative target, so set it above the saved
update count to continue learning. Time/checkpoint/logging/thread settings can
change on resume; architecture and learning settings cannot. Only load
checkpoints from trusted sources.

Immutable `checkpoints/step-XXXXXXXXX.pt` snapshots are retained every 1,000
updates by default (`snapshot_updates`). Evaluate those for comparisons across
time. An unexpected learner failure preserves the last good checkpoint instead
of saving possibly invalid parameters. `max_seconds` applies to each process
invocation; the watchdog's `SPENT_USD` must include earlier invocations.

```sh
export RESUME=runs/m1-first/latest.pt
./infra/run_train.sh
```

Optional `SYNC_DEST` is a mounted durable directory or an rsync/SSH destination.
The operator supplies existing credentials; the repository never discovers or
stores provider secrets. Destination parent directories must already exist.

```sh
export SYNC_DEST=/mounted-persistent-volume/guanzero/m1-first
./infra/run_train.sh
# Or copy a stopped run explicitly:
./infra/sync.sh runs/m1-first /mounted-persistent-volume/guanzero/m1-first
```

Sync runs every 60 seconds and once after training exits. It copies completed
files using destination-side temporary files and rename, excludes checkpoint
temporary files, and never deletes destination data. For remote durability
within ten minutes, leave room for the sync interval and transfer time when
choosing the checkpoint interval (for example, 480 seconds plus 60-second sync).
A sync failure stops training and reports failure rather than continuing a
long paid run without a durable checkpoint. The final sync is attempted again.

## Optional provider teardown hook

The watchdog accepts an explicitly configured argv array, with no shell
interpretation, plus a separate opt-in. Use a narrowly scoped, tested provider
command that deletes exactly the intended instance; its credentials belong in
the provider's normal credential store. No generic provider command is guessed
here. Configure `TEARDOWN_COMMAND_JSON` and `ALLOW_TEARDOWN=1` only after this
command has been verified for the selected node. Both are required together.

Teardown runs after normal completion, training failure, budget/walltime stop,
or an external termination signal. Each hook has a bounded timeout. If final
sync fails, teardown still runs when explicitly enabled to avoid indefinite
billing, and the watchdog returns failure. A killed watchdog, lost host or
failed provider command still requires an independent provider-side limit.

Elo-stagnation shutdown is not connected to the first self-play run: it requires
a fixed evaluation ladder and periodic evaluation results. The explicit time
and estimated budget limits protect this run while those gates are measured.

## Evaluate checkpoints

Use fixed seeds and immutable snapshots. The arena evaluates the exact same
deal twice with policy teams exchanged, reports a whole-pair bootstrap
interval, and optionally plays independent full matches with a Wilson interval.
Seven behavior-probe groups are reported separately. The local random/greedy
bots are sanity baselines; they are not the unavailable OpenGuanDan agents.

```sh
export PYTHONPATH=python:.
.venv/bin/python -m eval.arena \
  --agent runs/m1-first/checkpoints/step-000001000.pt --opponent greedy \
  --deals 1000 --matches 100 --seed 20260921 \
  --output runs/m1-first/eval/step-000001000.json
.venv/bin/python -m eval.elo runs/m1-first/eval/step-*.json \
  --output runs/m1-first/eval/elo.json
```

The checkpoint identity includes its weights, so overwriting `latest.pt` cannot
merge distinct agents in Elo. Default evaluation is deterministic argmax;
`--margin 0.01` enables uniform near-best Q sampling. Keep margin and action
mode fixed across comparisons. `action_mode: "full"` in a training config
supports the planned canonical-action ablation without changing game rules.

For a checkpoint's hidden-hand accuracy, collect fresh frozen-policy games
without optimizer updates, then pass them to the arena. Use a new output
directory and a seed different from the training seed:

```sh
.venv/bin/python -m eval.collect_belief \
  --checkpoint runs/m1-first/checkpoints/step-000001000.pt \
  --output runs/m1-first/eval/belief-step-000001000 \
  --rounds 1000 --num-envs 16 --seed 20260922 --max-seconds 600
.venv/bin/python -m eval.arena \
  --agent runs/m1-first/checkpoints/step-000001000.pt --opponent greedy \
  --deals 1000 --matches 100 \
  --belief-logs runs/m1-first/eval/belief-step-000001000 \
  --output runs/m1-first/eval/step-000001000.json
```

The report verifies the collector's checkpoint identity and evaluation-only
provenance. Passing the trainer's own `belief/` directory is also possible,
but those scores are labeled diagnostics with training overlap unverified.
The collector refuses to overwrite an existing dataset and marks timed-out
collections incomplete.

## Belief experiment

The trainer writes completed-round `.npz` files in `belief/`, capped by
`log_max_rounds`, from the first `log_envs` environments. Each contains private
supervision labels separate from observations and the public history, including
forced passes. Histories include only the prefix preceding the logged decision.
Do not publish these private training labels as player observations.

```sh
.venv/bin/python -m train.belief_probe --logs runs/m1-first/belief \
  --output runs/m1-first/belief-probe.json --device cuda \
  --steps 1000 --batch-size 64 --seed 11
```

The probe holds out entire matches, compares a flat tower and causal public
Transformer with parameter counts within 2%, and reports log loss by round
stage and relative seat. It trains new supervised heads; it does not enable
Transformer RL. Repeat with independent seeds after enough distinct matches
have been logged. A two-step smoke probe is only a pipeline check.

For the larger, validation-selected architecture gate, use the corrected
private-query model and a separately trained no-history control:

```sh
PYTHONPATH=python:. .venv/bin/python -m eval.collect_belief \
  --checkpoint .work/runpod/artifacts/pilot/final.pt \
  --output .work/belief-v2/data --purpose architecture_probe \
  --rounds 4096 --num-envs 64 --threads 4 --seed 20260927 --max-seconds 1200
PYTHONPATH=python:. .venv/bin/python -m train.belief_experiment \
  --logs .work/belief-v2/data --output .work/belief-v2/experiment \
  --seeds 31 32 33 --threads 4 --max-seconds 2400
```

Use fresh output directories. This strict runner requires complete collection
provenance and matching engine sources, separates whole train/validation/test
matches, selects weights on validation only, and saves learning curves plus
match-clustered test intervals. Probe weights cannot be used as playing-policy
checkpoints. The original small runner remains available to reproduce its
historical experiment. See `reports/M2-belief.md` for the predeclared protocol.

The completed three-seed run passes the supervised v2 gate. The corrected
architecture beats flat loss by 9.1–11.1%, but history contributes only
0.21–0.43% over its no-history control. `train/history_cache.py` provides
verified incremental inference for one environment; it is not yet connected
to the playing policy or DMC replay. Keep the existing DMC checkpoint as the
playing reference until an equal-compute RL comparison establishes a gain.

## Stage A2: counterfactual tribute experiment

Use an immutable Stage A checkpoint. The collector runs the frozen play policy
through natural matches, samples rounds, and completes one cloned branch per
legal exchange candidate. It saves acting-seat features and centered terminal
returns; private snapshots stay in memory. The fitter updates only the two
exchange heads and verifies every other tensor is unchanged.

```sh
export PYTHONPATH=python:.
.venv/bin/python -m train.tribute_data \
  --checkpoint .work/runpod/artifacts/pilot/final.pt \
  --output .work/a2-data --positions 4096 --sample-fraction 1 \
  --seed 20260925 --max-seconds 600 --threads 2
.venv/bin/python -m train.tribute \
  --checkpoint .work/runpod/artifacts/pilot/final.pt \
  --dataset .work/a2-data --output .work/a2-fit \
  --steps 1000 --batch-positions 32 --seed 23 --threads 2 --max-seconds 300
.venv/bin/python -m eval.tribute \
  --agent .work/a2-fit/tribute.pt \
  --opponent .work/runpod/artifacts/pilot/final.pt \
  --deals 1000 --seed 20260926 --threads 2 --output .work/a2-eval.json
```

The example's fraction 1 labels every eligible sampled-round decision for a
bounded offline experiment; the collector default is 0.02. Use fresh output
directories. Training/holdout splitting is by complete source match. Engine
source fingerprints prevent fitting labels made with different observation
semantics. Timeouts leave incomplete collection provenance or partial fit logs;
they do not silently publish a successful artifact. Fitting checks its walltime
through loading, caching, updates and evaluation.

Ordinary DMC checkpoints keep heuristic tribute. Loading an explicit A2
checkpoint uses its learned heads for evaluation; it is not automatic promotion
of the baseline. DMC resume rejects A2's different optimizer. The tribute arena
requires exact frozen-play equality and independent fit/base/collection seeds,
and reports anti-tribute and real-choice counts. Its uniform synthetic deals
do not replace full-match or external benchmark evidence. Keep the heuristic
when payoff is within noise. See `M2_TODO.md` and `reports/M2-A2.md`.

The first completed three-seed experiment retains the heuristic: all paired
means were negative, and two intervals excluded zero. The saved A2 checkpoints
are experiment artifacts, not the default model for the next training stage.

## Initial tuning and gate boundaries

`m1.json` starts at 4,096 environments, 32 collection steps, and 64 learner
updates per phase. Replay uses compact binary features and is bounded; records
are copied before C++ buffers are reused. Greedy opponents keep the same team
for a whole match and only network play decisions produce DMC samples.
Tribute remains heuristic. The loss is round-return MSE plus hidden-hand and
finish-position cross entropy, with no discount or bootstrapping.

Watch `decisions_per_second`, `collect_seconds`, `learn_seconds`, `samples`,
`trained_samples`, losses and gradient norms on the actual GPU. These defaults
are starting values, not a measured optimal configuration. Reduce environment
count or `candidate_chunk` if memory is tight. CPU smoke timing does not predict
the 4,096-environment CUDA workload.
`runtime-NNN.json` records actual Torch/CUDA versions, device, parameter count,
and whether bf16 was enabled for each invocation, including resume.

M1 strength and the architecture decision need real training/evaluation data.
The external Rule One--Four/DanZero/SDMC/GS2 ladder is unavailable as recorded
in M0. M2/M3 gates must be revised against an agreed reference before those
stages; local greedy wins cannot be reported as those published benchmark wins.
