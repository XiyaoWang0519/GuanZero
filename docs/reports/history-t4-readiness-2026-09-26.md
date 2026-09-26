# T4 remote pilot: prelaunch contract

This document preserves the preparation state. The subsequently approved run
completed; see the [pilot receipt](history-t4-pilot-2026-09-26.md) for measured
throughput, learning/resume/population, evaluations, artifact verification and
confirmed teardown. The pending statements below describe the prelaunch state.

Prepared on September 25 Toronto / September 26 UTC. This is an engineering
cold-start/resume/population check. CUDA throughput, learning progress and
playing strength have not been established by this preparation.

## Execution order and scope

1. Obtain approval for **one RTX 4090 Secure, at most 90 minutes from create,
   $1.20 total estimated cap, $0.80/hour ceiling including storage**, minimum
   4 effective CPU cores, 24 GB host RAM, 30 GB container disk and no volume.
   Live inspection during preparation showed zero pods / $0 current hourly
   spend, RTX 4090 Secure availability LOW and GPU quote $0.74/hour. At that
   quote, conservative disk cost makes the estimate $0.7445/hour; creation
   rechecks both quote and allocated rate. No replacement-pod retries are
   authorized by this plan. Reconcile uncertain creates by unique name.
2. Immediately start the independent local provider guard, then wait for SSH.
   Missing machine metadata during initialization is not failure evidence.
   SSH wait is bounded at 15 minutes; setup is bounded at 10 minutes. Record
   actual CPU quota, GPU, runtime, compiled extension and source hashes.
3. Run CUDA update/resume/privacy checks, then throughput cases of 4/8/16
   environments. Each case has a 240-second wall bound. Width 64, two layers,
   four heads, full match prefix, FP32, TF32 disabled, one PPO epoch, 64 vector
   steps/update, two whole matches/minibatch. No snapshot population in this
   sweep. Every case records real collection/learning throughput as histories
   grow; no synthetic speed is called end-to-end training speed.
4. Require long-prefix collection throughput (mean prefix >=720) to retain
   at least 50% of throughput near 144, peak reserved memory below 70% of GPU
   memory, and nonzero finite encoder gradients. Select the passing case with
   highest long-prefix learned rows/(collect+learn) seconds. A failed sweep
   stops; implement/profile batched KV caching with rebuild after weight
   changes before more training. No history truncation or precision reduction.
5. Run that measured configuration for at most 25 minutes or 200 updates,
   within a 45-minute remote workload watchdog and the provider deadline.
   Random seed 2026092602, random actor/critic/heads, no old checkpoints or
   datasets. Save each update. Save/resume after snapshots first appear;
   preserve optimizer and RNG but deliberately discard unfinished rounds,
   old assignments and histories. Frozen snapshots come from this lineage.
6. Poll and sync every 30 seconds. Check completed rounds, PPO sample counts,
   loss/entropy/KL/clipping, actor/encoder/critic gradients, real historical
   policy decisions, growing prefixes and CUDA memory. Stop on invalid
   values, ratio instability, missing gradients, a >180-second update or stale
   pilot health. A process running by itself is not acceptance.
7. Preserve initial, after-resume and final checkpoints. Evaluate the
   predeclared endpoints against the frozen MLPs in a separate local process.
   No best-checkpoint selection. Complete raw artifact sync/hash verification
   before normal teardown; failures never extend the budget to await transfer.
   Require owned pod `confirmed_gone`, then independent pods/spend readback.

The runtime chosen here is a **measurement card**, not a claim about the most
cost-effective architecture or production environment count. Billing includes
startup and storage; a process watchdog cannot stop provider billing.
Provider contracts: [pricing](https://docs.runpod.io/pods/pricing),
[Pod lifecycle](https://docs.runpod.io/pods/manage-pods).

## Population and sequence semantics

- Identity 0 is the current learner. Every new match has one randomly selected
  guaranteed learner seat. Each remaining seat independently picks a uniformly
  sampled recent snapshot with probability 0.5; otherwise it uses the learner.
  Before snapshots exist all four seats are current copies.
- Publish every two updates containing actual optimizer work. The most recent
  four snapshots are eligible. An older snapshot remains resident while any
  match has it pinned. Match identity, rollout session, seats, publication
  update and weight SHA-256 are logged; per-policy decision counters prove
  participation. Frozen seats never enter actor or critic training rows.
- Every play seat executes a Transformer. The common engine heuristic handles
  only tribute/back-tribute. Every public play/pass/exchange stays in the match
  stream, including opponents' events. No forced-private flag, private tribute
  bits or hidden leftovers are revealed to the actor.
- Original round team level returns and GAE remain unchanged. A team reward
  is assigned to its last learner row, terminal at round end; unfinished
  rounds carry over with original behavior log-probability/version. There is
  no cross-round bootstrap. Hidden counts feed only the independent critic.
- No KV cache is implemented or claimed. The full-prefix encoder is rebuilt
  on every call, including after optimizer updates; resume starts fresh raw
  histories. T4 cache optimization remains conditional on measured CUDA data.

## Frozen evaluation

| Baseline | Checkpoint SHA-256 |
|---|---|
| B11 main | `25e9e0bf549957b9d8b8aeb426b7225259cca714ec23d77de8eeda45bef063df` |
| Latest long-run segment 2 raw endpoint | `eda2efc844d3831615b8fa99b3102591a95f9c1a6dfc03f3b0ede0479b8f6aa6` |

Both use their checkpoint's embedded MLP reference **inside the evaluator**,
top-32 support, temperature 0.02, deterministic argmax, margin 0, no search,
canonical actions, FP32 and heuristic tribute. The latest raw endpoint is
chosen in advance rather than selecting between raw/EMA on pilot results.
The older five-hour checkpoint is retained but not substituted for that
endpoint (`ebb614f35f825e835c9d5b18bdc5a6efb0f79d5ba5ed9bfcdf44eca98ae139fe`).

Development: 64 actual fixed deals from seed 2026092607, action seed
2026092608, and eight full-match seed pairs 2026092700–2026092707. Duplicate
rounds and full matches have separate reports; each pair keeps the seed and
swaps policy teams. Intervals bootstrap whole pairs. This small set measures
pilot behavior and is not a promotion test or training-seed uncertainty.
Final-test seed material is sealed with a SHA-256 commitment locally, excluded
from upload; no final-test deals are generated or evaluated. DanLM remains
external calibration only.

## Artifacts and verification boundary

Kit: `.work/runpod-history-t4-2026-09-26/`, following the prior long-run layout:
`source.tar.gz`, `run-manifest.json`, `payload/{setup,run}.sh`,
`payload/cfg/pilot.json`, `evaluation/{freeze,development}.json`, `local/`,
`results/`, and (only after approval) `pod.json`, guard/monitor/provider receipts.
Source packaging excludes `.git`, `.work`, model files, binaries, credentials,
datasets and macOS metadata. Evaluation checkpoints remain local.

Local focused verification: 101 passed, 1 CUDA-only skip. The check includes
full-prefix probability/privacy tests, frozen snapshot action execution,
learner-only buffering, immutable match seating, retained pinned snapshots,
population RNG/weights restore, CUDA sampler test declaration, throughput
stop rules, source archive inspection and artifact hash failures. Full project
regression then passed: 720 Python tests, 2 skips (CUDA-only acceptance and an
existing optional test), 81 C++ cases, 14 independent oracle groups, and three
20,000-round deep fuzz runs (canonical, full, styled), zero failures. The
additional frozen-evaluator test checks baseline/deal hashes and identical
match seeds with team swaps. Logs are saved in the kit's `results/` directory.

The missing local tensorboard test dependency was installed into the existing
Python 3.12 environment. CMake's old Python 3.14 cache entries were cleared and
the matching 3.12 extension rebuilt before the final full check. No C++ source
or model assets were changed. The source archive was unpacked separately and
its file-by-file receipt matched the live source digest.

GPU tests, throughput sweep, pilot health, endpoint evaluations and teardown
are all still pending explicit rental approval. The $1.20/$0.80/90-minute
limits are a proposal, not an already approved allocation or guaranteed
provider-enforced cap.
