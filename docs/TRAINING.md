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

### Styled opponents and the style-space coverage report

Self-play copies have no habits, so a probe fitted on them cannot show
cross-round opponent modelling. `--styled` gives the other team habits: the
checkpoint drives seats `t` and `t+2`, and the continuous style-parameterised
bot drives the other two. The team `t` is drawn per match and recorded.

```sh
PYTHONPATH=python:. .venv/bin/python -m eval.collect_belief \
  --checkpoint .work/m1-full-model/latest.pt \
  --output .work/belief-styled/data --purpose architecture_probe \
  --rounds 100000 --num-envs 64 --threads 4 --seed 20260930 --max-seconds 3600 \
  --styled --style-region mixed --style-seed 101 --heldout-fraction 0.5
PYTHONPATH=python:. .venv/bin/python -m eval.style_coverage \
  --collection .work/belief-styled/data
```

| Flag | Meaning |
|---|---|
| `--styled` / `--no-styled` | Drive the opponent team with the styled bot (default off) |
| `--style-region` | `train`, `heldout` or `mixed` region of the style space |
| `--style-seed` | Seed of the per-match style draw; defaults to `--seed` |
| `--heldout-fraction` | Share of held-out styles under `--style-region mixed` |
| `--policy-team` | `random` (per match, recorded), `0` or `1` |

`train/styles.py` owns the style space: slot 0 is `bomb_threshold`, slots
1..T are per-play-type preference log-weights, then `follow_aggression`,
`lead_high_bias`, `partner_weight` and `temperature`. The held-out region is a
configurable corner of the box (by default `bomb_threshold >= 0.75` and
`follow_aggression >= 0.75`); training styles are rejection-sampled outside it,
so the two regions are disjoint by construction. Styles are drawn once per
match per seat from a generator keyed by `(style_seed, env_id, match_id)`, so a
match's styles are fixed and reproducible. The style space, region, split rule
and per-seat driver assignment are recorded in `provenance.json`, and
`matches.json` lists every sampled match with its styles, drivers and behaviour
histogram.

`eval.style_coverage` writes `coverage.json` and `coverage.md`: a histogram of
each style slot over the sampled matches, behaviour histograms (bomb timing,
play-type frequency, lead rank) per style-slot bin, and an explicit check that
no training style falls inside the held-out region.

Pass `--heldout-style-test` to `train.belief_experiment` to restrict the test
matches to the held-out style region; without it the existing seeded whole-match
split is unchanged.

### Scaled belief probe on a styled collection

The scaled probe is the same runner at four layers and width 256, trained to a
validation plateau on the 100,000-round styled collection.

```sh
PYTHONPATH=python:. .venv/bin/python -m train.belief_experiment \
  --logs .work/belief-styled/collect-100k --output .work/belief-styled/scaled \
  --seeds 31 32 33 --width 256 --layers 4 --batch-size 64 \
  --steps 50000 --min-steps 2000 --validation-interval 1000 --patience 3 \
  --decisions-per-round 20 --validation-decisions 20000 \
  --heldout-style-test --threads 4 --max-seconds 86400
```

| Flag | Meaning |
|---|---|
| `--decisions-per-round` | Keep at most this many decisions of each round (0 = all). Whole rounds and their full public token streams are always kept, so every surviving decision still has its exact history prefix |
| `--data-seed` | Seed of the per-round subsample and of the validation cap |
| `--validation-decisions` | Cap the decisions each validation pass scores (0 = all). Selection only; the test set is never subsampled |
| `--cell-bootstrap-samples` | Bootstrap resamples for the per-cell paired intervals (default 2,000; the overall interval keeps 10,000) |

Memory. Rounds stay in RAM at their on-disk dtypes, so `obs` is `uint8` at
1,849 B per decision and is cast to float32 only per batch. The full 100,000
round collection is roughly 10M decisions, about 20 GB of `obs` and `hidden`
plus 1.6 GB of tokens and 1 GB of index tuples, which does not leave headroom
inside a 32 GB budget. `--decisions-per-round 20` keeps every round but about
2M decisions, roughly 4.2 GB; 40 gives about 8.4 GB. `report.json` records the
measured `dataset_bytes` and `loaded_decisions`.

New result fields, per model and seed:

| Field | Meaning |
|---|---|
| `stop_reason` | `plateau` when patience stopped the fit, `step_cap` when it ran out of `--steps` |
| `best_validation_step` | Step of the selected weights, equal to `selected_step` |
| `validation_evaluations` | Number of validation passes the fit ran |
| `test.cells` | Test log loss per breakdown cell: `round_bin:{0,1,2-3,4-5,6+}` (match-relative round index), `style_region:{train,heldout,unknown}` and `target_driver:{bot,policy,unknown}` for the seat whose hand is predicted. Each cell reports `log_loss`, `decisions` and `test_matches` |
| `v2_vs_v1.cells`, `v2_vs_no_history.cells` | The same cells, each with a paired bootstrap 95% interval over whole test matches |

Schema 1 logs have no round index, style region or per-seat driver, so all of
their cells are labelled `unknown` and keep loading unchanged. Round bins
separate the first two rounds of a match, where an opponent model has seen the
least, and pool later rounds so the cells stay large. `target_driver` cells
score each predicted seat separately, so their `decisions` count seat targets:
three per decision.

Cost, measured on this host at `--width 256 --layers 4`, batch 64, four CPU
threads, on real histories (mean prefix 43 public tokens):

| Model | Seconds per training step | 20,000 steps | 50,000 steps | 100,000 steps |
|---|---|---|---|---|
| `v1` flat | 0.0068 | 0.04 h | 0.09 h | 0.19 h |
| `no_history` | 0.0170 | 0.09 h | 0.24 h | 0.47 h |
| `v2` history | 0.1674 | 0.93 h | 2.33 h | 4.65 h |
| all three, one seed | 0.191 | 1.1 h | 2.7 h | 5.3 h |

Three seeds therefore cost about 3.2 h, 8.0 h and 15.9 h of training at those
step counts, before validation and test passes. Fifty thousand steps on three
seeds fits an overnight CPU run; a hundred thousand is the point at which a
rented GPU pays for itself.

If `gd.STYLE_DIM` is missing the styled bot is not built. Collection then warns
loudly, plays the opponent seats greedily, records their style vectors as NaN
and sets `"styled_bot_unavailable": true` in provenance. Such a collection is a
pipeline check only and must not be used as evidence of opponent modelling.

The completed three-seed run passes the supervised v2 gate. The corrected
architecture beats flat loss by 9.1–11.1%, but history contributes only
0.21–0.43% over its no-history control. `train/history_cache.py` provides
verified incremental inference for one environment; it is not yet connected
to the playing policy or DMC replay. Keep the existing DMC checkpoint as the
playing reference until an equal-compute RL comparison establishes a gain.

### v3 match-memory probe

`train/memory_experiment.py` answers one question: can a per-seat, per-round
memory summary carried across the rounds of one match improve hidden-hand
prediction in later rounds, against opponents whose style was never seen in
training, relative to the identical model with that memory masked?

Both models are `train.belief_memory.MemoryBelief`, built on the **no_history**
query tower that task 3 selected, never on the history tower. When a round
ends, its whole public token stream is encoded by the tower's own causal
public-stream layers and mean-pooled at the tokens of each absolute seat, which
gives one summary vector per seat per finished round. At a decision in round
`r` the private query attends, in the cross-attention block the tower already
has, over the summaries of the rounds strictly before `r` of the same match,
each key tagged with an absolute-seat and a round-distance embedding. Memory is
strictly causal and strictly public: round `r` never sees round `r + 1` or
another match, and nothing private crosses a round boundary. DESIGN 7.3 would
also pool each seat's revealed remaining cards at the round end; the schema-2
collection records nothing at a round end, so those inputs are omitted and the
summaries are public-stream only.

`memory_masked` is the same network with the memory keys absent from the
cross-attention, so its query attends to the BOS token alone and it is
functionally the no_history tower. It is a deep copy, so it is parameter
identical (0.0% difference), and both towers exceed the scaled no_history
tower by exactly the tag embeddings, `(4 + memory_rounds) * width` parameters:
4,665,318 against 4,662,246 at width 256 and four layers, 0.066%. Both checks
are enforced in `matched_memory_models`.

```sh
PYTHONPATH=python:. .venv/bin/python -m train.memory_experiment \
  --logs .work/belief-styled/collect-100k --output .work/belief-memory/v3 \
  --seeds 41 --width 256 --layers 4 --batch-size 64 \
  --steps 50000 --min-steps 2000 --validation-interval 1000 --patience 3 \
  --decisions-per-round 20 --validation-decisions 20000 --memory-rounds 8 \
  --late-from 5 --heldout-style-test --device cuda --threads 8 \
  --max-seconds 86400
```

The flags of the scaled probe all carry over (`--logs`, `--output`, `--seeds`,
`--steps`, `--min-steps`, `--validation-interval`, `--patience`,
`--batch-size`, `--width`, `--layers`, `--threads`, `--learning-rate`,
`--device`, `--max-seconds`, `--decisions-per-round`, `--validation-decisions`,
`--heldout-style-test`, `--data-seed`, `--split-seed`,
`--cell-bootstrap-samples`) with the same meanings, including the guarantee
that `--decisions-per-round` keeps whole rounds and never touches a public
token stream. Summaries of earlier rounds always read the full stream of those
rounds. Four flags are new:

| Flag | Meaning |
|---|---|
| `--memory-rounds` | How many earlier rounds of the match the query may attend to, most recent first (default 8). Also the size of the round-distance embedding |
| `--late-from` | First match-relative round index counted as "late" in the adaptation metric (default 5, so rounds 0 to 4 are early) |
| `--readout-rounds`, `--readout-alpha` | Rounds sampled for the style readout and its ridge penalty |
| `--parameter-tolerance` | Relative bound on the difference to the no_history tower (default 0.001). Only a tiny test shape needs to loosen it |

Result fields in `report.json`, beyond the task 3 breakdowns, which are all
present unchanged (`test.cells` with `round_bin:*`, `style_region:*`,
`target_driver:{bot,policy}`, `target_seat:*`, `target_relation:{teammate,opponent}`
and the stage crosses):

| Field | Meaning |
|---|---|
| `runs[].models.{memory,memory_masked}.test.log_loss` | Held-out-style test log loss per model |
| `runs[].memory_vs_masked` | Paired masked-minus-memory improvement over whole test matches, with a bootstrap 95% interval, and the same paired interval inside every cell |
| `runs[].adaptation` | The DESIGN 9.1 item 6 metric in belief terms: per test match, the paired improvement in rounds with index >= `--late-from` minus the improvement in the earlier rounds of that same match, with `bootstrap_95_ci` over matches, plus `improvement_by_round_bin`, `improvement_by_round_index` and `improvement_slope_per_round` |
| `runs[].self_adaptation` | Each model's own early-minus-late loss difference, which is how the masked control is checked for a late-round trend of its own |
| `runs[].models.*.style_readout` | Ridge regression from the finished-round summary vectors of bot-driven seats to that seat's style vector, fitted on training matches, reported as `heldout_r2` per style slot. The masked control's encoder receives no gradient from its head, so its readout measures only what the initialisation captures |
| `gate`, `gate_rule`, `adaptation_supported`, `control_shows_no_trend` | The gate below, and the two conditions it is made of |

Gate. v3 is supported when the adaptation improvement is positive with a
positive bootstrap lower bound in **every** seed on held-out styles, and the
masked control shows no such late-round trend of its own. Anything else records
`v3_not_yet_justified`.

Cost, measured on this host at width 256, four layers, batch 64, four CPU
threads, on 200 real matches of the 100,000-round collection (1,065 rounds,
mean 5.3 rounds per match and 123 public tokens per round), `--memory-rounds 8`:

| Model | Seconds per training step | 20,000 steps | 50,000 steps |
|---|---|---|---|
| `memory` | 0.573 | 3.2 h | 8.0 h |
| `memory_masked` | 0.0186 | 0.10 h | 0.26 h |
| both, one seed | 0.592 | 3.3 h | 8.2 h |

The memory model is about thirty times the masked control because each batch
re-encodes every distinct earlier round it references: up to `--memory-rounds`
rounds for each of 64 decisions, each a full ~123-token stream through the
four-layer encoder. The control skips that work entirely, since masked keys
would contribute nothing and its encoder takes no gradient. One CPU seed is an
overnight run, so the three-seed probe belongs on a GPU. Sampling each batch
from fewer matches would let more decisions share the same encoded rounds, and
is the obvious optimisation if the GPU run turns out to be encoder-bound.

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

## DanLM external baseline (Stage C row C0)

DanLM plays in its own engine, which ships as CPython 3.12 macOS binaries.
It is evaluation only (non-commercial licence) and is never vendored. One-time
setup, all under `.work/`:

```sh
git clone --depth 1 https://github.com/dashidhy/DanLM.git .work/external/DanLM
uv venv .work/external/danlm-venv --python 3.12
VIRTUAL_ENV=$PWD/.work/external/danlm-venv uv pip install --python .work/external/danlm-venv/bin/python torch numpy onnxruntime pybind11 pytest
cmake -S . -B .work/build-py312 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE=$PWD/.work/external/danlm-venv/bin/python -DGD_BUILD_TESTS=OFF
cmake --build .work/build-py312 -j --target _gd_core   # lands in python/gd/ beside the 3.14 module
```

Rules diff (DanLM in every seat, both engines compared at every decision) and
the duplicate evaluation of a checkpoint, DanLM's engine refereeing:

```sh
export PYTHONPATH=python:. DANLM_ROOT=.work/external/DanLM
.work/external/danlm-venv/bin/python -m eval.danlm.arena diff --rounds 200
.work/external/danlm-venv/bin/python -m eval.danlm.arena duplicate \
  --checkpoint .work/runpod-b8/results/runs/league/run/latest.pt --deals 1000 --workers 8
.work/external/danlm-venv/bin/python -m eval.danlm.arena duplicate \
  --checkpoint "danlm:v1t:ckpts/DanZero_v3_rep_v1t/v3_rep_v1t_best_eval_001_int8.onnx" --deals 1000 --workers 4
```

About 75 s per 1,000 deals with 8 workers. `DANLM_SLOW_OBS=1` disables the
verified observation replacement (`eval/danlm/fast_obs.py`), which is 16x
slower. Tests: `tests/test_danlm_bridge.py` runs anywhere;
`tests/test_danlm_fast_obs.py` runs in the 3.12 environment and skips
elsewhere. Report: `docs/reports/stage-c-danlm.md`.
