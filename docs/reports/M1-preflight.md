# M1 pre-training implementation report

September 21, 2026. The repository now supports the first Stage A v1 DMC run.
This report records the local implementation phase. Subsequent rented-GPU
validation and pilot results are in [`M1-runpod.md`](M1-runpod.md).

## Delivered

The training path now runs full matches through `VecEnv`, associates completed
rounds with environment/match/round keys, assigns each learner decision its
acting team's undiscounted return and final finish position, and learns from
owned compact feature copies. Hidden-card supervision is kept outside the
actor input. Greedy opposition is fixed by match; tribute stays heuristic.

The model has the designed four-layer 512-wide state tower, two-layer 256-wide
action tower, three phase-specific fusion heads, and both auxiliary heads:
3,078,125 parameters. CUDA has optional bf16, explicit pinned transfer staging,
candidate chunking and a static play-head path. CUDA graph capture is deferred
until profiling demonstrates its value.

Atomic checkpoints include model, optimizer, schedules/counters, Python/NumPy/
Torch/CUDA RNG state and Stage A opponent metadata. Resume restarts environments
and drops incomplete trajectories/replay. Periodic immutable snapshots permit
reproducible comparison. SIGINT/SIGTERM request a final save; unexpected learner
failure retains the last good checkpoint.

Evaluation includes duplicate rounds with paired bootstrap intervals, full
matches with Wilson intervals, content-identified Elo, and seven behavior-probe
groups. A separate frozen-checkpoint collector supplies fresh belief-evaluation
logs with provenance and no optimizer updates. Public action logs include forced passes. The offline belief experiment
compares parameter-matched flat and causal-history towers using match-held-out
data and stage/seat breakdowns; it does not enable v2 in the policy.

Launch tooling supplies a GPU image, preflight, optional checkpoint sync and a
walltime/estimated-dollar watchdog. Provider deletion is an explicit configured
hook; process termination alone cannot stop cloud billing.

## Local validation

- Release validation: 54 C++ cases and 14 standalone oracle groups passed.
  Final `.venv/bin/python -m pytest -q tests oracle`: **228 passed**.
  `scripts/check.sh` also completed 20,000 canonical and 20,000 full-mode fuzz
  rounds with no failures.
- UBSan rebuild: 54 C++ cases and 2,000 additional deep-invariant fuzz rounds
  passed. ASan retains the M0 host/toolchain limitation; Linux CI config covers it.
- Small-model actual self-play: 40 updates, 15,360 decisions, 179 completed
  rounds, 17 completed matches, 12,440 labeled learner samples.
- Full production-size model with a bounded CPU workload: six optimizer
  updates and 14 completed rounds; immutable snapshots at updates 2, 4 and 6.
- CPU preflight: two updates, resume with restored optimizer/RNG, then four
  cumulative updates. Real Trainer SIGTERM/resume and interrupted atomic-write
  recovery are also covered by regression tests.
- TensorBoard emitted real loss metrics. The watchdog's 22 subprocess tests
  include budget limits, forced termination, failed sync, disk-full logging,
  broken/closed output, teardown cleanup and signals during process creation.
- Actual saved checkpoint evaluated over duplicate deals, full matches and
  probes. Actual self-play logs passed through both belief-probe towers with
  separate match splits. Independent frozen-policy collection and checkpoint
  belief evaluation also ran successfully. These tiny experiments provide no
  strength evidence.

Local artifacts are intentionally ignored under `.work/m1-smoke/`,
`.work/m1-full-model/`, `.work/m1-match-smoke/`, `.work/final-preflight/`, and
`.work/independent-belief-smoke/`.
Reproduction commands and launch instructions are in `../TRAINING.md`.

## Remaining empirical gates

The subsequent RunPod report records the successful CUDA preflight and
4,096-environment workload/memory check, live rate, artifact retrieval and
cleanup status. The repository Dockerfile itself has not been built; the
tested node uses the official RunPod CUDA image with the bootstrap script.

External M1 strength remains unmeasured. The missing published baseline agents documented in
M0 still prevent claiming the original external ladder; random/greedy arena
results are explicitly internal. Any change to later M2/M3 gates needs an
agreed reference and real experiment results, not additional pre-run scaffolding.
