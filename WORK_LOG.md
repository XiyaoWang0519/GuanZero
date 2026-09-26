# History recipe campaign — September 26, 2026 UTC

- Goal: lift the from-scratch history Transformer's development curve above the
  −2.1..−2.4 plateau without capability shortcuts, cumulative RunPod cap $30
  (user pre-authorized; stop new rentals at $25). Spent $4.65 over six
  allocations (plus one no-machine pod and three HTTP-500 creates, all
  reconciled); every run verified downloads, deleted its pod and read back zero
  pods / $0 per hour. Ledger: `.work/runpod-ledger-2026-09-26.json`.
- Diagnosis: the plateau is the greedy heuristic's level; a pure self-play MLP
  DMC calibration is still below greedy after 1.85M samples; the 64-wide model
  trains faster on pod CPUs than on the GPU, so all arms trained on CPU.
- Development results: two PPO epochs is seed-robust (+0.37 vs B11 late, 4 vs 6
  seeds); self-play-only and data-parallel collection speed learning on top;
  resumed lineages reach about −1.0..−1.3 vs B11 and hold +0.5..+0.8 over the
  control to update 1,500, then level off. Several knobs gave nothing or hurt.
  Full-match wins 1–2/16 at isolated snapshots; no strength promotion.
- Engineering: `train/history_ddp.py` (+ `tests/test_history_ddp.py`), data-
  parallel training with resume; kit tooling in `.work/history-recipe-kits/`.
  112 local history tests pass (6 CUDA skips); 98–101 on each pod.
- Moved ten iCloud conflict copies (`name 2.py`, older versions of existing
  files) out of `train/`, `eval/`, `tests/` into
  `.work/icloud-duplicates-2026-09-26/`; they had changed the source identity
  and would have blocked resume. Other `name 2.*` files (docs, WORK_LOG) untouched.
- Reports: `docs/reports/history-recipe{1..5}-2026-09-26.md` and
  `history-recipe-summary-2026-09-26.md`; TRAINING.md and STAGE_C_TODO.md updated.

# Batch-size recipe test — September 26, 2026 UTC

- Ran one approved RTX 4090 Secure allocation (35.4 min, est. $0.44) with three
  fresh random-start history Transformer arms for 300 updates: 32-env KV/SDPA,
  8-env KV/SDPA, 8-env dense. Evaluated every 10-update snapshot locally on the
  frozen development deals. Report: `docs/reports/history-batch-2026-09-26.md`.
- Result: the T4 plateau is not a batch-size effect; per learner row the curves
  coincide and all arms stall at -2.1..-2.4 with entropy below 0.5 and zero
  full-match wins. KV/SDPA vs dense showed no detectable difference on one seed.
- Also re-evaluated all 22 T4 pilot snapshots on CPU (kit
  `.work/runpod-history-t4-2026-09-26/results/snapshot-curve.json`): climb from
  update 40 to about 110, then no measurable gain to 200.
- No defaults, checkpoints or claims promoted. Kit:
  `.work/runpod-history-batch-2026-09-26/`.

# Current plan update — September 25, 2026

- Consolidated DESIGN.md v0.6, STAGE_C_TODO.md T0--T8 and TRAINING.md around
  randomly initialized history Transformer self-play; old MLPs are evaluation-only.
- Removed active MLP distillation, old replay/pool, teacher-reference pruning/KL
  and belief-first gates. Standard and looped Transformers share the same
  cold-start boundary, with separate history/auxiliary/compute comparisons.
- Updated the implementation inventory, performance and compute guides, plus
  repository entrypoints. Retained dated results and marked old proposals as
  historical. Existing partial implementation and legacy models remain intact.
- Documented public-history privacy, sequence PPO/cache requirements and the
  need for history-aware evaluation. Engine-private forced-pass metadata is
  not a public actor feature.
- Scope is documentation. No runtime code, checkpoint, training job or cloud
  resource was changed by this update. No new RL trainer is claimed ready.
- Verification: `git diff --check` passed; 88 local links across 11 active
  documents resolved, tables/fences were consistent, and stale-route checks
  passed. Five historical-plan notices preserve their original bodies;
  pre-existing `docs/PY_API.md` edits were unchanged. No runtime tests were
  needed for this documentation-only update.

The entries below are historical work logs, not the current task queue.

# Overnight candidate support work

Branch: `codex/overnight-candidates` in `wt-candidates`. Scope: C1(b) only.

- Added optional `candidate_mode=union`, `candidate_extra=m` to policy and PPO configs. Defaults retain the existing top-k path and RNG sequence.
- Union draws a uniform subset without replacement from actions outside frozen-reference top-k plus pass. Rollout stores the exact chosen support, action log-probability, and reference distribution on that support; learner evaluates stored candidates only.
- Increased derived buffer capacity for `top_k + candidate_extra + pass`; actor config uses existing dataclass serialization.
- Updated scalar and batched evaluation to draw union support, preserving the small-set reference skip.

Verification:

- Copied the existing Python 3.14 `gd` extension from the main checkout into this worktree's ignored `python/gd/` for tests. No build or rules change.
- `PYTHONPATH=python /Users/xiyaowang/Documents/Projects/GuanZero/.venv/bin/python -m pytest -q tests/test_candidate_union.py tests/test_stage_b_policy.py tests/test_ppo.py tests/test_ppo_actors.py tests/test_batched_evaluation.py tests/test_pruning_audit.py tests/test_ppo_throughput.py`: 62 passed, 1 skipped in 16.25s. Actor tests needed the approved elevated sandbox because the ordinary sandbox blocks `torch_shm_manager` shared memory.
- New tests measure the uniform inclusion frequencies, compare `act`/`choose` RNG states, reproduce stored behavior/reference log-probs, exercise one PPO update and checkpoint load, check zero-extra RNG identity, and pass union candidates through actor shared-memory IPC and both evaluators.

Cloud comparison proposal: clone one B11-final configuration twice with a fixed league pool and equal wall clock. Both use `top_k=32` and `warm_start_policy_override=true`; control has `candidate_mode=top_k,candidate_extra=0`, union has `candidate_mode=union,candidate_extra=8`. Freeze the pool; leave live `league_import_dir` empty. This is a short probe, not G7 acceptance.

Pending: final diff review and commit. No cloud resource or local long training started.

# Search work log — September 24, 2026

- Scope: isolated branch `codex/overnight-search` from `4940246`, worktree
  `.work/overnight-20260924/wt-search`; no edits in the main checkout.
- Read `CLAUDE.md` and `docs/RULES.md` before C++ changes. No rule changes.
- Added the C++ information-set sampler and Python binding, then a bounded
  scalar search policy with `search:<checkpoint>` loading. Existing batched
  evaluator remains unchanged.
- Built only this worktree's `build-search` with 3 jobs and used the shared
  `.venv` interpreter. Ran no training, GPU job, cloud call, or paid inference.
- Passed 81 C++ cases and 47 targeted Python tests. Scalar arena two-deal
  smoke completed. Small 64-deal B11 comparison and measured latency are in
  `docs/reports/stage-c-search-v0.md` and adjacent JSON files.
- Next work, if this arm is continued: adapt the batched evaluator or retain
  scalar, add calibrated marginal/learned beliefs, measure the required three
  budgets, and run G9's larger paired evaluation. The current interval
  includes zero; no strength claim is supported.

## Independent audit and external transfer — later September 24

- Independently recomputed all 1,000 development and 4,000 held-out internal
  pair scores from their raw legs; the held-out mean and whole-deal bootstrap
  interval match the supplied report exactly.
- Built a CPython 3.12 `gd` module in this worktree's `build-cp312`, without
  overwriting the main checkout. DanLM four-deal smoke runs for B11 and
  search:B11 each had zero mirror failures.
- Ran a new-seed, 500-pair transfer against DanLM's referee using one worker
  and one CPU thread. Raw records and paired summary are under
  `docs/reports/search-danlm-transfer-500.*`; interpretation and limits
  are in `docs/reports/stage-c-search-transfer-2026-09-24.md`.
- No model training, paid inference, GPU, or cloud job was started.

## Behaviour probe and Transformer inventory — September 25 (Claude session)

- Read-only inventory of what the history-Transformer route can reuse:
  `docs/reports/transformer-inventory-2026-09-25.md` (since rewritten by the
  v0.6 consolidation; the survey facts stand).
- Log schema 3: `train/logs.py` and `eval/collect_belief.py --candidates`
  record each decision's canonical candidate set, abstract ids and choice, and
  per-token abstract id / forced flag / phase. Old schemas and loaders unchanged.
- `train/behaviour_probe.py` plus `tests/test_behaviour_probe.py` (15 tests):
  offline BC/NTP diagnostic with candidate, fixed-vocabulary and structured
  vocabulary heads, flat control, `--window k` control, driver-split cells.
  Diagnostic only under v0.6, never a gate; results in
  `docs/reports/behaviour-probe-2026-09-25.md`.
- Found and fixed a boundary issue: the engine's `forced` flag means "only pass
  was legal"; it is removed from token inputs and pinned by a test. T2 must not
  expose it either.
- Local CPU only: two 3,000-round M1 collections (self-play and styled) and
  seven probe arms under `.work/behaviour-probe/`. No GPU, no RL, no cloud.
- `.venv` was missing; used `.work/external/danlm-venv` (3.12) with pip,
  hypothesis and pytest-xdist added. Full Python suite: 610 passed, 1 failed
  (tensorboard absent in that environment).

## T1--T3 of the v0.6 route — September 25, later (Claude session, three subagents)

- T1 `train/history_model.py` (lead): `PublicStream` (186-dim public tokens +
  round + phase, no forced bit), `HistoryActor` with public-only
  `encode_stream`, private query and full canonical candidate head,
  `HistoryCritic` on obs + hidden counts, `history_ppo` checkpoint marker.
  Adversarial tests `tests/test_history_model.py`: 68 cases pass.
- T2 `train/history_rollout.py`, `train/history_ppo.py` (subagent): match
  event store, sequence rollout buffer, PPO with full-prefix recompute,
  atomic checkpoint/resume, manifest + metrics, CLI. 14 tests pass; CPU smoke
  8 envs x 3 updates learns. Throughput falls with prefix length (2,500 to
  580 decisions/s at mean prefix 144 to 720), recorded under T4.
- T3 `eval/history_policy.py` + hooks in policies/duplicate/arena/batched/
  danlm arena (subagent): scalar event source with Python forced-pass
  resolution, exact token parity with VecEnv, per-leg/per-match resets,
  DanLM lockstep invariant. 13 tests pass; 264 evaluator tests unchanged.
- Full Python suite: 712 passed, 1 failed (tensorboard absent in the 3.12
  environment), 1 skipped. Codex's uncommitted hunks in eval/ and train/ppo.py
  verified intact. Nothing committed. No GPU, no cloud, no long training.
- Receipts written in `docs/STAGE_C_TODO.md` T1--T3 and `docs/TRAINING.md`.

## T4 bounded GPU pilot preparation — September 25 Toronto / September 26 UTC (Codex)

- Added own-lineage frozen Transformer population execution, per-match pinned
  seats and identities, learner-only PPO rows, pinned snapshot retention,
  participation counters and checkpointed population weights/RNG. Fixed the
  CPU-only sampler for CUDA training. Resume verifies source/engine/schema,
  restores optimizer and population, and explicitly drops partial trajectories
  and restarts public histories. Full-prefix recomputation remains; no KV cache
  or CUDA speedup is claimed.
- Added `bench.history_ppo` for a CUDA-only real-prefix throughput sweep and a
  stop-before-pilot gate; CUDA peak memory, collect/learn throughput and prefix
  lengths are logged. Added source packaging, reviewed manifest generation,
  separate frozen evaluation, remote workload and owned-pod monitor with an
  independent provider guard, metric health checks, artifact hashing and
  teardown readback. The old MLP preflight is not used.
- Frozen B11 main and segment-2 raw endpoint hashes and deterministic evaluation
  settings. Development deals are materialized, final-test seed material sealed
  and excluded from upload. Evaluators run locally in a separate process.
- Kit: `.work/runpod-history-t4-2026-09-26/`. Proposed rental is one RTX 4090
  Secure, max $1.20 total / $0.80 hourly / 90 minutes including startup/cleanup;
  source archive contains no checkpoints, secrets, old datasets or macOS
  metadata. No paid resource was created. Provider inspection showed no pods,
  $0 current hourly spend and a $0.74/hour Secure GPU quote (LOW availability).
- Validation: focused history tests 101 passed / 1 CUDA skip, then complete
  `PY=.work/external/danlm-venv/bin/python PYTEST_WORKERS=4 ./scripts/check.sh`
  passed: 720 Python passed / 2 skipped, 81 C++ cases, 14 oracle groups, and
  three 20,000-round deep fuzz runs with zero failures. Installed the already
  required tensorboard dependency and repaired stale CMake Python cache before
  the final check. Verified extracted source receipt against live source.
- Boundary: local implementation/packaging verified; CUDA tests, measurements,
  remote training/resume, development evaluations and paid-pod lifecycle are
  pending the user's explicit rental approval. Changes remain uncommitted.

## T4 approved GPU pilot completed — September 25 Toronto / 26 UTC

- User approved the prepared one-4090 Secure / $1.20 / 90-minute allocation.
  Reconciled an initial HTTP 500 create with no resource before retrying;
  retained the original deadline. The only allocated pod was `4k181sro9lrvmt`.
- Remote source SHA `868a2541dbf60a0d72e8f2cac2a669e8be228d8561c99dacac0dfa0774888b5a`:
  random actor/critic/heads/optimizer, no old model/data assets uploaded.
  CUDA-focused checks passed 98 / skipped 1 optional external DanLM engine.
- Measured 4/8/16 environments before selecting 8: collection 986 → 663
  decisions/s as mean prefix 166 → 754 (67.2% retained). Sixteen failed the
  reserved-memory margin. Full-prefix FP32 recomputation remains; no KV cache.
- Completed 200 updates in 317.99 trainer seconds: 1,724 rounds, 139 matches,
  63,948 learner training rows. Real resume at update 2 preserved lineage and
  progress. Published 100 snapshots; 83 historical identities acted 34,204
  times. All actor/encoder/critic tensors changed and remained finite.
- Pilot longest prefix 2,403; peak tensor allocation 1.55 GB but allocator
  reservation 24.31 GB before falling. This is a required profiling follow-up
  before scale-up, not evidence of sustained memory headroom.
- Predeclared initial/update-4/final development evaluations completed against
  both frozen MLPs, 64 duplicate deals and eight full-match pairs each. Final
  net levels versus B11 improved −2.984 → −2.328; versus long-run endpoint
  −2.992 → −2.578. Full-match wins remained 0/16 against each. No held-out
  final-test deals opened; no playing-strength promotion.
- Verified 54 downloaded files against remote SHA-256 before teardown, then
  re-audited locally. Pod confirmed gone; independent provider query: zero
  pods, $0 hourly spend. Allocation 601.22 seconds, quote-time cost estimate
  $0.1243; billing line items still unavailable, not a settled invoice.
- [Completed receipt](docs/reports/history-t4-pilot-2026-09-26.md), raw kit
  `.work/runpod-history-t4-2026-09-26/`, final checkpoint SHA
  `7f3787eacc0acad2b96afd17cfea1975bd8950cef9ca4c27427fd3612febdfd1`.
  T4 bounded pilot complete; T5 periodic evaluation/failure analysis remains
  open. Changes remain uncommitted; unrelated `.DS_Store` was preserved.

## History stack optimization — September 26

- Followed the completed T4 measurements: collection consumed 92.8% of
  collection plus learning time; long-history allocator reservation needed
  investigation. Added optional causal SDPA without the expanded padding mask,
  and batched public-only KV caches partitioned by policy/env/match. Retained
  FP32, full history, full canonical candidates and PPO/GAE/population rules.
- Learner caches are freed before learning and rebuilt under updated weights;
  frozen caches survive until their match/policy retires. Checkpoints contain
  no caches. Gradient recomputation always uses raw history. Defaults remain
  dense/no-cache until CUDA acceptance; old source-identity checks remain.
- Added per-phase diagnostic timing, allocator/cache counters and alternating
  dense/SDPA/KV benchmarking. CPU rejects optimizer runs. Frozen CPU mixed-seat
  collection, 8 envs, longest prefix 2,313: dense 102–105 s, SDPA 40–41 s,
  KV with learner rebuild each 64 steps 6.6–6.9 s for 8,192 measured decisions.
  This is a CPU diagnostic, not a CUDA training-speed claim.
- Full gate: 727 Python passed / 6 skipped, 81 C++ cases, 14 oracle groups,
  three 20,000-round deep fuzz runs, zero failures. New tests cover batched
  cache parity through 2,403 tokens, sampled choices/probabilities, private
  isolation, weight/reset invalidation, gradients and resume.
- [Report](docs/reports/history-stack-2026-09-26.md), diagnostics at
  `.work/history-stack-2026-09-26/`, prepared CUDA comparison kit at
  `.work/runpod-history-stack-2026-09-26/`. Proposed new scope: one RTX 4090
  Secure, $1.20 total / $0.80 hourly / 90 minutes including cleanup. No GPU
  resource created, no new paid run started; approval and CUDA results pending.

## History stack CUDA comparison — September 26

- User approved one RTX 4090 Secure within $1.20 / $0.80 hourly / 90 minutes,
  with the hard requirement that optimizations preserve model capability.
  FP32, full public events, architecture, all canonical actions and PPO rules
  remained fixed. Added CUDA coverage for exact mixed-policy sampled choices;
  no tolerances were relaxed. Local focused file: 7 passed / 5 CUDA skips.
- On the remote CUDA host: 98 Python checks and 81 C++ cases passed, followed
  by all 12 repeated throughput cases and two separate profiles. Across cases:
  336 updates, 172,032 decisions, 150,374 learner rows, longest prefix 2,815.
  Thirteen health polls checked progress, gradients/ratios and memory.
- Pooled full-update KV speedup: 1.76x all-current, 1.50x mixed snapshots.
  Mixed allocator peak reserved memory: 24.68 GB dense, 0.17 GB KV; peak active
  tensors 1.93 / 0.108 GB. No SDPA/KV allocator retries. Raw timing pairs,
  shorter-prefix near-ties and changed training trajectories are retained.
  Defaults remain off: bounded numerical/engineering evidence is not proof
  of sustained playing-quality equivalence; T5 remains open.
- All 83 remote artifacts hash-verified before owned-pod deletion and again
  locally. Independent provider readback: pod absent, zero pods / $0 hourly.
  Allocation 619.44 s, estimated $0.12810 (quote-based, not an invoiced bill).
  Guard/controller processes also exited. No replacement rental or extra
  budget. [Receipt](docs/reports/history-stack-cuda-2026-09-26.md), kit at
  `.work/runpod-history-stack-2026-09-26/`; source SHA `e86a9761f3db74acc9fc8e959107a9b71b3c8cb844cdcf14cab79f36aaae3008`.
