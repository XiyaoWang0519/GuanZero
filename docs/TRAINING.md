# Training and evaluation: Transformer self-play

Operator guide, reorganized October 6, 2026. Baseline: random-initialized
history Transformer self-play; old MLPs are evaluation-only. The design
boundary is in [DESIGN.md](DESIGN.md); dated readiness and research directions
are in [STATUS.md](STATUS.md). Experiment chronology is in
[TRAINING_EVIDENCE.md](TRAINING_EVIDENCE.md), not a launch configuration.

## Experimental paged rollout attention (October 18, 2026)

The [FP32 Triton prototype and GPU acceptance guide](reports/paged-attention-prototype-2026-10-18.md)
adds explicit `PagedHistoryCache(..., attention_backend="triton")` selection for
isolated experiments. The default remains `torch`; ordinary and merged snapshot
rollout caches are covered, while learner/private-attention backward is unchanged.
CUDA compilation, numerical parity, and throughput are not yet verified. Do not
enable this in production on the basis of CPU tests. The guide includes a
fail-closed GPU gate runner and separate component/complete-PPO benchmarks.
No production CLI, checkpoint schema, or research boundary is changed.

## Choose the entrypoint

| Task | Owning module / document |
|---|---|
| Sequence self-play PPO | `train.history_config`, `train.history_ppo`, `train.history_rollout`, `train.history_population` |
| Multiple training ranks | `train.history_ddp`; check the run's actor/rank and batch semantics |
| Load a history checkpoint for evaluation | `eval.history_policy`, `eval.policies` (`history_ppo` checkpoint marker) |
| Frozen development evaluation | `eval.history_frozen`; scalar / batched backend parity is workload-specific |
| External-agent evaluation | `eval.danlm.arena`; retain failed deals, exclusions and candidate-filter statistics |
| Paired checkpoint comparison | `eval.paired` on two `eval.danlm.arena duplicate` or two `eval.lineage_eval` outputs over one deal set |
| Checkpoint weight averaging | `train.checkpoint_average`; evaluation-only output, never a resume point |
| Test-time search | `eval.search`; [search evidence](reports/strength-summary-2026-10-04.md) |
| Botzone package | `eval/botzone/`; [deployment receipt](reports/botzone-search-2026-10-04.md) |
| Legacy MLP path | `train.dmc`, `train.ppo`, M1/B8/B11 configs; historical, not the Transformer launch route |

Inspect the module's current CLI/config and checkpoint source identity before
launching. `scripts/preflight.sh` is the legacy MLP preflight. The old
`infra/run_train.sh` wiring must be checked before reusing it for a sequence
trainer. Local artifact paths and approved budgets in old reports do not
identify today's checkpoint or authorize a new rental.

Run configuration and CLI parsing live in `train.history_config`, which uses
only the standard library; `train.history_ppo` owns the training lifecycle.
`train.public_history` owns the NumPy public token schema and raw match streams,
shared by training, caches, evaluators and historical log writers.
`eval.history_events` captures engine actions and manages evaluator streams;
`eval.history_policy` runs the actor. `train.advantages` supplies the shared GAE
calculation; history rollout keeps its round-terminal, zero-bootstrap interface.
The original module imports remain available as compatibility exports.

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
pass. On one RTX 4090 the CUDA gate passed (max merged vs per-identity
log-prob difference 9.5e-7 at production size) and an in-run A/B measured
1.355x decisions/s with 4 ranks
([report](reports/history-snapshot-batching-2026-09-28.md)); no strength claim.
`rollout_trim_cuda_cache` (`--rollout-trim-cuda-cache auto|true|false`,
resume-settable and recorded) calls `torch.cuda.empty_cache()` after collect and
after learn; `auto` (default) follows the merged-snapshot arm, because with the
arm on no snapshot private graph is captured, so nothing else trims and each
rank's reserved memory ratchets. Switching the arm on also trims right after
the snapshot graphs are released. Allocator timing only: CPU tests show rows,
choices, weights and sampler bitwise unchanged; every metrics line records
reserved/allocated bytes before and after each trim and its seconds
(`cuda_trim`, `cuda_trim_seconds`). The CUDA gate passed on one RTX 4090
(bitwise apart from process-wide allocator counters) and a 30-update check kept
nvidia-smi flat at 0.12 s per update
([report](reports/history-snapshot-batching-2026-09-28.md)).
`bench.history_stack` alternates dense/SDPA/KV cases with fixed learning rules
and records full-update throughput/memory. CPU mode requires `--collect-only`
and charges a learner-cache rebuild each chunk unless explicitly disabled for
a frozen-cache diagnostic. Source changes preserve the original T4 archive;
they do not silently bypass checkpoint source identity for resume.

Python profile (diagnostic, off by default): with
`GUANZERO_CPROFILE_DIR=<dir>` in the environment, `train.history_ppo` and
every `train.history_ddp` rank run `trainer.run()` under `cProfile` and write
`<dir>/rank-<r>.prof` (`pstats` format; the single-process trainer is rank 0)
when the run returns, including after a SIGTERM/SIGINT stop, or fails. Nothing
trained or sampled changes, but profiler overhead inflates every timing field,
so a profiled run is not a throughput measurement. It covers each rank's main
thread only; read it with `python -m pstats <dir>/rank-0.prof`. C functions
are folded into their Python callers' own time; `GUANZERO_CPROFILE_BUILTINS=1`
lists them, but under Python 3.12 some torch C calls then drop the enclosing
frames (`run`, `update`, `collect`) from the call tree.

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

Private cross-attention audit and Q/K normalization (October 18, 2026;
[report](reports/private-attention-audit-2026-10-18.md)).
`python -m eval.private_attention_audit --out DIR CKPT...` collects one CPU
self-play sample of whole matches (`--sampler`, `--save-sample`/`--load-sample`
reuse it across checkpoints) and writes `audit.json` and `summary.md`:
saturation, landing, history-swap sensitivity, math-SDPA q/k gradients, Adam
state and dead ReLU units, stratified by round and tribute round. It is CPU
only and launches no training. `--private-qk-norm none|query|qk`
(`HistoryPolicyConfig.private_qk_norm`, default `none`) is an architecture
arm for a fresh lineage: `none` keeps the current lineage's architecture
record, state dict and outputs; a resume cannot change it; merged snapshot
inference (`--batch-snapshot-policies`) and the Botzone NumPy actor refuse
the other arms.

### Runtime refactor (2026-10-18)

The [October 18 implementation and measurements](reports/training-throughput-refactor-2026-10-18.md)
cover these changes:

- `VecEnv.drain_public_batch` returns owned public-token arrays for batch
  validation and stream ingestion. Rebuild the matching C++ extension with
  `cmake --build build -j` in an already configured build to enable it. Older
  extensions, custom environments and specialized event stores retain the
  scalar drain/ingest path. Event order and public/private boundaries remain
  the same.
- Ordinary and merged snapshot KV appends reserve their combined missing pages
  once. Existing pages, zero padding and the pool's reset/high-water policy are
  preserved; new physical page IDs are internal allocator metadata.
- The learner reuses actor gradient norms for encoder diagnostics and skips
  copying a next-token auxiliary input through a one-part concatenation.
  Clipping, FP32 overflow checks and auxiliary objectives remain the same.
- Every private-attention route now shares the existing math-SDPA guard for
  grad-enabled CUDA calls, including default, singleton and windowed batches.
  This extends the recorded backward safety fix; CPU, inference and public
  attention retain their previous backend dispatch.

Architecture, FP32, full match history, complete legal support, PPO/population
semantics and checkpoint/token schemas are unchanged. Local CPU verification
matched the original revision's byte fingerprints across dense, paged and
merged-snapshot cases for two PPO updates plus checkpoint resume. The report
contains suite results, component measurements and their limitations. Target
CUDA acceptance and throughput remain pending; no production deployment or
playing-strength result is recorded for this refactor.

From the repository root, use fresh output paths:

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python:oracle:.
.venv/bin/python -m bench.training_revision --verify --device cpu \
  --case paged --updates 2 --output .work/throughput-check/verify-paged
.venv/bin/python -m bench.history_public_events --output .work/throughput-check/events.json
.venv/bin/python -m bench.history_page_pool --device cpu --output .work/throughput-check/pages.json
.venv/bin/python -m bench.history_gradients --device cpu
```

Repeat verification with `--case dense` and `--case snapshots`, each in its own
output directory. For revision comparisons, run the same harness file against
each checkout's source and matching extension. `--verify` includes instrumented
state/trajectory hashing and a resumed update; omit it for separate alternating
timing trials with identical settings and warmup. Component scripts isolate
event ingestion, page allocation and gradient diagnostics, respectively; their
ratios do not substitute for complete-PPO collect/learn/wall measurements.

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

Main-lineage archive (October 6, 2026). `.work/lineage/main/` keeps one main-path
checkpoint per 100M self-play decisions plus the named milestones (u675 ...
u35588), as `u<update>-<decisions>M.pt` with `.work/lineage/index.json`
(source, decisions, sha256). `python .work/lineage/archive.py --apply` adds new
ones: local files are hard links, Velocity files are pulled over the UW VPN
and checked for lineage id and progress. Run it whenever a remote run is
checked or synced; cluster home directories are wiped each term, and
pre-u675 snapshots were lost to an earlier cleanup. Cleanup scripts must treat
`.work/lineage/` as protected. Side arms (wide, lr-decay, aux-control, ...)
share the lineage id but are not on the main path and are not archived there.

## Velocity node-local staging (October 25, 2026)

Velocity `/home` is NFS shared with other users of `max-compute-11`. Its
bandwidth is fine (cold 25–106 MB checkpoints read in about 0.1 s), but every
file open costs about 0.1 s, so Python imports from an NFS conda env starve the
node: with none of our jobs running, `import numpy` from `~/guanzero/env` took
18 s cold, and copying the 6 GB env tree crawled at 0.05 MB/s. Rules for every
GuanZero SLURM job on Velocity (scripts in `scripts/velocity/`, deployed to
`~/guanzero/bench-kit/`):

- Source `gz_job_env.sh` (training kits source its wrapper `shm_stage.sh`,
  which also stages the run dir). Never `conda activate ~/guanzero/env`.
- `/dev/shm/$USER` must be a directory owned by the job's uid with mode 0700,
  not a symlink; anything else exits 2 (`/dev/shm` is world-writable, so another
  account could otherwise plant `bin/python` or `.staged_ok`).
- The env is `/dev/shm/$USER/gz-env-local`, built from the internet by
  `build_shm_env.sh env` (uv, Python 3.12.14, `requirements-velocity.txt`, the
  cu128 torch index; about 30 s) under flock, complete only with `.staged_ok`.
  A missing env is rebuilt from the internet; a failed build fails the job.
  There is no fallback to the NFS env.
- The source tree is staged by `build_shm_env.sh src` from a tarball or
  directory into `/dev/shm/$USER/src/<name>-<key>`. The key is the sha256 of the
  tarball, or of a directory's `source-identity.json`. A changed source therefore
  gets a new stage, and a stage in use by a running job is never replaced. A
  directory with no receipt (a git checkout) gets a fresh, unshared copy each
  time. Extraction uses Python's `tarfile` `data` filter, which rejects members
  or links that escape the stage. The tree is checked file by file against
  `source-identity.json` (the trainer refuses a tree that has neither that
  receipt nor `.git`). A
  tree that ships `python/gd/_gd_core*.so` keeps it, since rebuilding could change
  the engine a run started with. Kit `*.py` files are copied to shm, and the job
  `cd`s into shm, because `python -c`/stdin put the cwd on `sys.path`.
- `PYTHONPYCACHEPREFIX`, `TMPDIR`, `XDG_CACHE_HOME`, `TORCHINDUCTOR_CACHE_DIR`,
  `TRITON_CACHE_DIR`, `CUDA_CACHE_PATH`, pip and uv caches all point into
  `/dev/shm/$USER`; `PYTHONNOUSERSITE=1`.
- The run dir lives in `/dev/shm/$USER/runs/<run>`. It is seeded from
  `~/guanzero/runs/<run>` (without the periodic `update-*.pt`; the population
  lives in `latest.pt`) and synced back every 300 s or more, plus on exit. Append-only
  `*.jsonl`/`*.log` copy only their new tail (`rsync --append`). Other files
  go to a temporary name and are renamed, so `/home` never holds a torn
  `latest.pt`. Local `update-*.pt` copies already on `/home` are pruned. Only
  the SLURM `.out` is written to `/home` directly. Seeding uses `rsync --update`,
  so a local checkpoint newer than `/home` is not rolled back. Sync runs only
  for a run dir the wrapper staged (`GZ_STAGE_RUN=1`); a kit that sets `R`
  itself keeps its own I/O.
- Sync failure stops the job. Each rsync is bounded by `GZ_SHM_SYNC_TIMEOUT`
  (600 s), and `$R/.gz-sync-status` holds the last outcome. After 3 consecutive
  failures, or 1200 s without a successful sync, the loop writes
  `$R/.gz-sync-failed`, logs `gz_job_env FATAL`, and sends TERM to the batch
  shell. The kits treat TERM as "stop cleanly, do not resubmit". The EXIT trap
  then makes one more sync attempt and exits 4, so `sacct` shows the job as
  FAILED. The shm copy stays in place; resubmit by hand once `/home` is healthy.
  Stopping bounds the work held only in node RAM to about 20 minutes, and stops
  a hammering resubmit loop against a sick NFS server.
- A CPU-only allocation hides every GPU (`CUDA_VISIBLE_DEVICES=`); a malformed
  or self-contradictory allocation (unparsable IDX, or IDX count different
  from `AllocTRES` `gres/gpu`) exits 2.
- Guards: the job exits 2 if `sys.prefix` or `torch.__file__` resolve under
  `/home`, if the env path is under `/home`, or if torch sees more GPUs than
  SLURM allocated. `CUDA_VISIBLE_DEVICES` is set to the allocated UUIDs; a
  repaired card left out of gres stays visible through the device cgroup and
  must not be used. The env's `sitecustomize.py` refuses to start any
  interpreter, including spawned workers, whose prefix, `sys.path` or
  torch/numpy resolve under `/home`. With `GZ_REQUIRE_NODE_LOCAL=1`,
  `train.history_ddp` ranks and PPO actors check again through
  `infra/node_local.py`.

Checks after a (re)submit: the `.out` shows `guard ok` with a `/dev/shm`
prefix, the allocated GPU, and a `/dev/shm` run dir. `ls -l /proc/<pid>/fd`
for our Python processes shows no `/home` paths except the `.out`. A sync
line appears every 5 minutes. Cold NFS `import numpy` is near the 18 s
zero-load baseline. Evidence:
[velocity-nfs-staging-2026-10-25](reports/velocity-nfs-staging-2026-10-25.md).

## Independent evaluation

**Yardsticks from October 3, 2026.** B11 is saturated (the lineage wins ~86% of
matches; the three Oct 3 arms were indistinguishable on 256 deals while DanLM
separated them), so a nightly report uses: the external yardstick, DanLM on
4,000 deals (±0.035); the internal yardstick, the new endpoint against the
previous lineage endpoint on 2,000 deals (`python -m eval.lineage_eval`,
batched MPS, about 7 minutes; identical checkpoints score exactly 0); and B11 on
256 deals as a one-line sanity check only. `scripts/eval_night.sh SEGMENT NAME
BASELINE.pt` runs all three plus a checkpoint curve against the baseline
(`scripts/eval_night_summary.py` prints it). First use: u15094 vs u9989 +0.198
[0.141, 0.259] on 2,000 deals; then u20264 vs u15094 +0.243 [0.184, 0.298].

**Paired records and averaging (October 18, 2026).** `eval.danlm.arena duplicate`
writes `per_deal`: both legs of every deal (our reward, round win, decisions,
divergence counts and, for a failed leg, the fatal kind, its detail and any
reading substitution earlier in that trick), plus `excluded_deal_ids`,
`failed_legs_by_kind`, `deal_set_sha256`, `checkpoint_sha256` and
`source_sha256`. Headline numbers and their bootstrap are unchanged.
`eval.lineage_eval` adds `deal_set_sha256` and `excluded_deals` (always 0:
one engine plays both sides). Two runs on the same deals, seed and opponent
(DanLM spec, or lineage baseline SHA) pair deal by deal:

```sh
python -m eval.paired u86800-danlm.json m7-danlm.json --output u86800-minus-m7.json
```

It reports A − B on the deals both runs scored (whole-deal bootstrap and
normal interval), the round-win-rate difference, the excluded deals of each
run with reasons, a worst-case bound had they been scored, and the unpaired
standard error and per-deal correlation, which show how much pairing helped.
Arena outputs from before this date have no `per_deal` and must be rerun.
`python -m train.checkpoint_average --output avg4.pt A.pt B.pt C.pt D.pt`
writes the uniform mean of the actor and critic weights of N checkpoints of
one lineage and architecture, recording its inputs under `averaged_from`. It
has no optimizer state, and `train.history_ppo` refuses to resume it. See the
[October 18 receipt](reports/paired-eval-checkpoint-average-2026-10-18.md).

**Engine digest, version 2 (October 3, 2026).** `infra.history_artifacts.engine_digest`
covers the game-dynamics sources only (`cpp/include/gd/*.h`, `cpp/src/*.cpp`
minus the evaluation-only `search.h`/`search.cpp`; `python/bindings.cpp` is out).
A resume accepts an older checkpoint when the version-2 digest recomputed from
its recorded per-file hashes matches (`engine_compatible`); frozen evaluation
files may carry either digest; the B11 freeze was rewritten with the new one.

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

## Implementation and experiment receipts

These headings remain for existing links; the detailed chronology moved.

## Implementation status and entrypoints

See [the dated record](TRAINING_EVIDENCE.md#implementation-status-and-entrypoints).

## Verified bounded T4 workload

See [the dated record](TRAINING_EVIDENCE.md#verified-bounded-t4-workload).

## Recipe runs after T4 (September 26)

See [the dated record](TRAINING_EVIDENCE.md#recipe-runs-after-t4-september-26).

## Speed options after the refactor (September 29)

See [the dated record](TRAINING_EVIDENCE.md#speed-options-after-the-refactor-september-29).

## Auxiliary heads on the shared encoder (October 2)

See [the dated record](TRAINING_EVIDENCE.md#auxiliary-heads-on-the-shared-encoder-october-2).

## Learning-rate decay (October 4)

See [the dated record](TRAINING_EVIDENCE.md#learning-rate-decay-october-4).

## Planted-habit diagnostic (September 30)

See [the dated record](TRAINING_EVIDENCE.md#planted-habit-diagnostic-september-30).
