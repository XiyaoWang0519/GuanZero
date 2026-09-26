# T4: completed bounded cold-start Transformer pilot

September 25 Toronto / September 26 UTC. One approved RTX 4090 Secure run
completed 200 PPO updates, a real save/resume and Transformer snapshot play.
The development scores improved from initialization but remain far below both
frozen MLPs. This is engineering acceptance of the bounded pilot, with an
initial learning signal; it is not a playing-strength promotion or permission
for a larger rental.

## Initialization and run identity

Actor, history encoder, candidate head, independent critic and optimizer were
freshly initialized with seed `2026092602`. The initial optimizer states are
empty. Checkpoints retain `stage=history_ppo`, `init=random`, `teacher=null` and
lineage `history_ppo-2026092602-e4299c6a`. No B11/long-run weights, old replay or
teacher were uploaded. MLPs were loaded only by the separate local evaluator.

| Identity | Value |
|---|---|
| Base Git revision | `9e70391c5bfa664bdf003bde75c2e174313c6dd7` plus uncommitted T4 source |
| Launched source SHA-256 | `868a2541dbf60a0d72e8f2cac2a669e8be228d8561c99dacac0dfa0774888b5a` |
| Uploaded archive SHA-256 | `9f0fb8dd7b8ca96fefc53603c041479d61f2576f3d0a3896a712135606e91b12` |
| Engine digest | `606ff1e735823cefb9968a974991bfc5ca325c2ef1cb8ea5a5540f174ce84096` |
| Token schema | 1; complete raw public match stream, no context truncation |
| Actor / critic | width 64, 2 layers, 4 heads / independent width 256, 3 layers |
| Runtime | RTX 4090, 24 GB; PyTorch 2.9.1+cu128, CUDA 12.8; FP32, TF32 disabled |
| Host | AMD EPYC 7452; effective cgroup CPU quota 10.2, usable integer budget 10 |
| Selected pilot | 8 environments, 64 vector steps/update, 1 PPO epoch, 2 matches/minibatch |
| Optimizer and returns | learning rate 0.0003, entropy coefficient 0.01, clip 0.2; gamma 1, GAE lambda 0.95 |

Round team-level returns go to each team's last learner row, terminal at round
end, with no cross-round bootstrap. Incomplete rounds carry their original
behavior log-probabilities and policy versions until finalized. Only learner
rows enter actor/critic updates. Private hidden counts are critic inputs only.
The [prelaunch contract](history-t4-readiness-2026-09-26.md) and frozen manifest
retain the full boundary, configuration and approval scope.

## Throughput before scale selection

Three separate CUDA processes measured real growing histories before the pilot.
The sweep used seed `2026092601`, eight updates per case, no snapshot mixing.
Collection counts all-seat decisions; learning counts unique learner rows.
Combined throughput is learned rows divided by collection plus learning time.

| Environments | Mean prefix early → long | Collect decisions/s early → long | Retained | Long learn rows/s | Long combined rows/s | Peak reserved GiB | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| 4 | 171 → 760 | 656 → 532 | 81.0% | 4,880 | 479 | 1.41 | pass |
| 8 | 166 → 754 | 986 → 663 | 67.2% | 6,326 | 557 | 6.69 | pass; selected |
| 16 | 173 → 729 | 1,310 → 751 | 57.3% | 4,374 | 649 | 17.31 | fails reserved-memory gate |

The predeclared gate required at least 50% collection retention and less than
70% of device memory reserved. Eight environments had the highest combined
throughput among passing cases. Sixteen environments were faster overall but
reserved 18.59 GB of the device's reported 25.25 GB; they were not selected.
These are allocator reservations, not active tensor memory or an observed OOM.

The tested range passes without a KV cache. Every decision still recomputes
its full prefix; this does not remove the asymptotic quadratic cost or establish
the optimum card/configuration. Longer histories and population mixing require
their own measurements. Raw per-update learning and collection measurements
are in `local/results/sweep/envs-{4,8,16}/metrics.jsonl` in the kit.

## Learning, population and recovery evidence

- Reached the predeclared 200-update limit in **317.99 trainer seconds**;
  102,400 environment decisions, 1,724 completed rounds, 139 completed matches,
  64,510 collected learner rows and **63,948 finalized learner training rows**.
  The difference includes deliberately discarded partial rounds at resume and
  unfinished rows at the endpoint.
- Published 100 own-lineage frozen snapshots. Logs contain 155 unique
  `(rollout_session, env, match)` seat assignments; identities were published
  before assignment and remained match-pinned. Eight snapshots remained
  resident at the endpoint, including older pinned weights. **83 historical
  identities actually played 34,204 decisions**. Learner seats and recent
  snapshots follow the fixed 0.5 sampling rule in the manifest.
- Saved after update 2, destroyed/reconstructed the trainer and continued from
  its checkpoint. Progress and lineage matched; optimizer, sampler/RNG and
  population restore executed. Incomplete rounds, assignments and raw histories
  restart deliberately. This is recovery, not bitwise continuation of the
  interrupted environments. There is no persistent KV cache to restore.
- All 200 metric rows have finite loss/entropy/KL/clipping and positive
  actor/encoder/critic gradient norms. All 60 actor tensors, including all
  24 stream tensors, and all eight critic tensors differ from initialization;
  all endpoint parameters are finite. Snapshot digests also verify.
- Maximum approximate KL was 0.00855, maximum clipping fraction 0.10825.
  Encoder gradient norms ranged 0.000253–0.007884. Entropy decreased from
  1.544 to 0.448; its decline warrants observation in any longer experiment.
- Full-prefix length reached 2,403 tokens. Peak **allocated** memory was
  1.55 GB (1.45 GiB), while peak allocator **reserved** memory reached
  **24.31 GB (22.64 GiB)** and later dropped to 5.13 GB. The short sweep's
  70% reserve margin did not hold throughout the population pilot. No OOM
  occurred, but this run does not establish sustained memory headroom or rule
  out long-run allocator pressure. Profile this before increasing duration,
  environments or model size; preserve full-prefix parity in any KV work.

Remote CUDA-focused verification: **98 passed, 1 skipped** in 25.37 seconds.
The skip was the optional external DanLM engine package, not CUDA acceptance.
Tests cover legal finite actions, public/private isolation, prefix probability
recomputation, learner-only buffering, match-pinned seats and checkpoint
restoration. They provide tested evidence against the specified leakage paths,
not a proof against every possible information leak.

Before rental, the full local gate passed: 720 Python tests, two skips,
81 C++ cases, 14 independent oracle groups and three 20,000-round fuzz modes.
The final launcher dependency fixes also passed 50 focused lifecycle tests.
No additional model changes were made after packaging.

## Predeclared development evaluation

Initial, after-resume update 4 and final update 200 were all evaluated. No best
checkpoint was selected. Each baseline/endpoint used the same 64 fixed deals,
two swapped-team legs per deal, plus eight same-seed full-match pairs. All raw
legs and matches remain in `evaluation/{initial,after-resume,final}.json`.
Evaluation ran locally on CPU; all model training ran on the remote GPU.

| Frozen baseline | Checkpoint SHA-256 |
|---|---|
| B11 main | `25e9e0bf549957b9d8b8aeb426b7225259cca714ec23d77de8eeda45bef063df` |
| Long-run segment-2 raw endpoint | `eda2efc844d3831615b8fa99b3102591a95f9c1a6dfc03f3b0ede0479b8f6aa6` |

Baseline settings: embedded MLP reference in the evaluator, top-32 support,
temperature 0.02, deterministic argmax, margin zero, no search, FP32 and common
heuristic tribute. Transformer evaluation uses its complete canonical support.
The development-deal SHA is
`c3502f6027c8e6937b9545c19e9e50d404c71f994e04566614e8115b0a9871c9`.
Final seed material stayed sealed locally, excluded from upload; no final-test
deals were generated or evaluated. DanLM was not used as training feedback.

Scores are candidate net levels per round; brackets are 95% whole-deal
bootstrap intervals. Trainer GPU-hours exclude setup, sweep and teardown;
total allocated hours were 0.1670.

| Endpoint | Trainer GPU-hours | vs B11 | vs long-run endpoint | Full-match wins vs each |
|---|---:|---:|---:|---:|
| Random initialization | 0 | −2.984 [−3.000, −2.961] | −2.992 [−3.000, −2.977] | 0/16 |
| After resume, update 4 | 0.00099 | −2.953 [−3.000, −2.875] | −3.000 [−3.000, −3.000] | 0/16 |
| Final, update 200 | 0.08833 | −2.328 [−2.516, −2.125] | −2.578 [−2.719, −2.414] | 0/16 |

Final minus initial, paired on those same 64 deals: **+0.656 [0.469, 0.860]**
against B11 and **+0.414 [0.273, 0.578]** against the long-run endpoint.
The candidate's first-finisher team rate increased from 0 to 7.81% / 3.91%,
and opponent double-finish rate fell from 98.44% / 99.22% to 66.41% / 77.34%.
These are an initial development learning signal for one training seed; the
intervals do not measure training-seed uncertainty. All full matches were lost.
The raw all-zero full-match bootstrap intervals are degenerate, not evidence
that the true win probability is exactly zero. The update-4 all-loss duplicate
interval has the same finite-sample limitation.

## Budget, artifacts and closure

Approved scope: one RTX 4090 Secure, $1.20 total cap, $0.80/hour including disk,
90-minute deadline including setup/cleanup. The first create returned HTTP 500
without a pod ID. Two readbacks and a pre-retry reconciliation confirmed no
resource and zero spend before the same allocation was retried. The original
deadline was retained; no second paid pod or extra budget was used.

Pod `4k181sro9lrvmt` was allocated at **03:42:15 UTC** and independently
confirmed gone by **03:52:16 UTC**, a measured 601.22 seconds. At the recorded
GPU-plus-disk quote of $0.744464/hour, estimated cost is **$0.1243**. The scoped
provider billing query had not returned line items, so this is a quote-time
estimate, not a settled invoice or a claim of zero bill.

The monitor started an independent provider guard before SSH/setup and checked
sample counts, gradients, ratios, rounds, snapshot decisions and throughput
through repeated health polls. It downloaded and SHA-256 verified **54 files
before teardown**. A separate local audit reverified all 54 files, checkpoint
and snapshot identities, parameter changes and all duplicate scores from raw
legs. Provider teardown says `confirmed_gone=true`; a separate provider query
then found **zero pods and $0 current hourly spend**. Monitor, guard and local
keep-awake processes exited. The guard was not needed to enforce the deadline.

Kit: `.work/runpod-history-t4-2026-09-26/`. Important receipts are
`approval.json`, immutable `run-manifest.json`, `progress.jsonl`,
`verified-artifacts.json`, `teardown.json`, `independent-provider-readback.json`,
`provider-billing.json`, and `results/final-audit.json`. The audit can be rerun
with `PYTHONPATH=python:oracle:. .work/external/danlm-venv/bin/python
.work/runpod-history-t4-2026-09-26/results/audit_downloads.py`.

Final checkpoint: `local/results/final.pt` within that kit, SHA-256
`7f3787eacc0acad2b96afd17cfea1975bd8950cef9ca4c27427fd3612febdfd1`.
The source manifest remains the frozen prelaunch contract; approval and runtime
receipts supply the completed state without rewriting that identity.

T4's bounded engineering pilot is complete. T5 has an initial three-endpoint
development comparison, while periodic long-run evaluation, behavior/failure
analysis and training-seed uncertainty remain open. Before a larger run,
investigate the reserved-memory high-water mark and entropy trend, then set a
new explicit compute budget from those measurements.
