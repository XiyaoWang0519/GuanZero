# History stack: bounded RTX 4090 comparison, September 26, 2026 UTC

Causal SDPA and batched public KV caching passed the CUDA equivalence tests
and improved this small, fixed-compute workload. Pooled full-update throughput
with KV was **1.76x** the dense control with all-current seats and **1.50x** with
mixed frozen snapshots. Dense allocator reservation again approached the card's
capacity; both optional paths avoided that behavior in these cases.

The user's hard condition is unchanged model capability. The changes preserve
FP32 (TF32 disabled), architecture, full public event history, the canonical
legal-action set, rewards/GAE and population rules. These measurements support
implementation equivalence within the tested numerical tolerances and an
engineering speed/memory result. **They do not establish unchanged long-run
playing strength. Both optimization flags remain off by default.**

## Scope and identity

- Approved scope: one RunPod Secure RTX 4090, at most $1.20 total,
  $0.80/hour including disk, 90 minutes including setup and cleanup.
- Actual: RTX 4090, PyTorch 2.9.1+cu128, AMD EPYC 7642 host; provider allocated
  12 vCPUs / 31 GB RAM, observed cgroup quota 10.2 CPUs. Engine/Torch threads
  both 2. The same pod ran every case in forward/reverse order.
- Randomly initialized width-64, two-layer, four-head history Transformer;
  eight environments, 64 vector steps/update, 24 updates/case, one PPO epoch,
  two matches/minibatch. First four updates excluded from timings.
- Twelve normal timing cases: dense → SDPA → KV → KV → SDPA → dense, first
  all-current, then mixed snapshots. Two separate synchronized profile cases.
  All 14 completed within the predeclared 120-second per-case limit.
- Each normal measured case made 10,240 environment decisions. Learner row
  counts and prefix distributions vary with evolving play; they are retained.
  No MLP, old optimizer/replay, or external teacher initialized these runs.
  The MLP checkpoints remain evaluation-only; no strength evaluation was run
  here, and final-test material was not opened.
- Source SHA-256: `e86a9761f3db74acc9fc8e959107a9b71b3c8cb844cdcf14cab79f36aaae3008`.
- Source archive SHA-256: `2271bc958cf5cb60e2921c8e641558ae5f0cb256cb87dae20849e18a202f86ed`.
- Approved manifest SHA-256: `c60ded50d2182870a8f1a57dcf3a4c72afad8a44a08a680cbab0f6ced57a89df`.
- Engine digest: `606ff1e735823cefb9968a974991bfc5ca325c2ef1cb8ea5a5540f174ce84096`.

Git base was `9e70391c5bfa664bdf003bde75c2e174313c6dd7` with the explicitly
hashed dirty source. This launch supersedes the preparation hashes in the
[local report](history-stack-2026-09-26.md): an additional CUDA sampled-action
parameterization and GPU telemetry were included before freezing this package.
The previous prepared package remains saved with a `preapproval` suffix.

## Correctness and health gate

Before timing, **98 Python tests passed in 26.21 seconds on the CUDA host**, with
no skips; the build also passed **81 C++ cases**. This includes CPU and CUDA
variants of the forward/gradient, ragged cache, mixed-population sampled-action,
PPO/recompute/resume and long-prefix tests, plus the existing model/rollout/
population/learner coverage.

The tests check exact sampled choices/PPO attribution in the paired mixed-seat
fixture, log-probabilities within `2e-5`, recomputation within `3e-5`, encoder
forward values at `rtol=3e-5, atol=3e-6`, gradients at `3e-4, 3e-5`, and the
2,403-token cache fixture at `4e-5, 4e-6`. Raw public events remain authoritative;
private decision observations are excluded from cache entries. Learner caches
are released before learning and rebuilt under updated weights; snapshot caches
stay policy/match-specific. Resume starts with empty caches. No tolerances were
relaxed after launch and no production code was changed during measurement.

Across all 14 cases: **336 PPO updates, 172,032 environment decisions, 150,374
learner rows**, maximum observed prefix 2,815 tokens. Thirteen monitor polls
checked metric progression and health. All inspected losses, gradients and
ratios were finite. Encoder gradient norms ranged from 0.000177 to 0.001246;
maximum approximate KL was 0.0000192, clipping fraction zero, maximum reported
ratio deviation 0.01561. These are health checks, not playing-strength metrics.

## Repeated throughput and memory

Full-update timing is recorded elapsed time at update 24 minus update 4. It
includes collection, learning and update bookkeeping/snapshot overhead, but
excludes process startup, compilation and tests. Pooled throughput divides
summed decisions or learner rows by summed seconds; speedup uses the fixed
environment-decision workload. Memory is decimal GB, peak over the two normal
cases; reserved memory is allocator reservation, not active tensor storage.

| Population | Path | Full-update decisions/s | Learner rows/s | Speedup vs dense | Peak allocated / reserved GB |
|---|---|---:|---:|---:|---:|
| All current | dense | 429.5 | 408.7 | 1.00x | 1.578 / 24.740 |
| All current | sdpa | 553.2 | 527.9 | 1.29x | 0.128 / 0.199 |
| All current | kv | 755.5 | 727.3 | 1.76x | 0.095 / 0.145 |
| Mixed snapshots | dense | 377.3 | 316.5 | 1.00x | 1.930 / 24.681 |
| Mixed snapshots | sdpa | 452.5 | 366.1 | 1.20x | 0.150 / 0.258 |
| Mixed snapshots | kv | 566.4 | 451.7 | 1.50x | 0.108 / 0.170 |

Raw repeat pairs, in repeat-index order (population 0 = all-current, 1 = mixed):

| Population | Path | Full-update seconds | Collection decisions/s | Mean prefix | Maximum prefix |
|---|---|---:|---:|---:|---:|
| 0 | dense | 21.971 / 25.715 | 517.4 / 435.9 | 798.3 / 949.1 | 2019 / 2395 |
| 0 | sdpa | 19.169 / 17.852 | 591.8 / 643.8 | 884.2 / 840.4 | 2550 / 2063 |
| 0 | kv | 13.466 / 13.640 | 891.5 / 849.6 | 784.0 / 995.7 | 2019 / 2771 |
| 1 | dense | 28.829 / 25.451 | 387.3 / 442.8 | 976.2 / 896.0 | 2657 / 1976 |
| 1 | sdpa | 23.359 / 21.901 | 481.6 / 517.5 | 1100.1 / 828.7 | 2815 / 1978 |
| 1 | kv | 18.797 / 17.360 | 611.1 / 649.8 | 828.7 / 935.4 | 1978 / 2370 |

Dense cases recorded 4/7 allocator retries with all-current seats and 3/2 with
mixed seats. All SDPA/KV cases recorded zero. Mixed KV cache tensor storage
peaked at 36.70 MB; it excludes model parameters and temporary workspaces.
These small-model measurements do not predict the memory of larger widths,
longer matches or larger populations. The large allocator-reservation drop
is partly a shape/allocation effect, not a 100-fold reduction in model size.

Within the 768–1,536 **mean-prefix** band, mixed dense collection measured
354.5 / 412.3 decisions/s; KV measured 609.9 / 615.1. The first shorter-prefix
band was essentially tied (dense 614.3, KV 613.5). Thus improvement is not
uniform in every bin. Bins contain different evolving trajectories; they are
not an isolated prefix-length sweep or an asymptotic complexity proof.

## Remaining bottleneck and evidence limits

In the separate synchronized mixed-population profile, public preparation/cache
plus actor/sampling consumed 18.81 of 20.65 collection seconds with SDPA and
15.40 of 17.12 with KV. KV reduced actor/sampling time (11.92 → 8.03 seconds),
while public cache/collation time increased (6.89 → 7.36). Pure environment
pending + step work was only about 0.33 / 0.32 seconds. Profile trajectories
also differed and synchronization perturbs timings; these cases are excluded
from the throughput table.

Forty-four coarse telemetry samples recorded GPU utilization median 22.5%,
range 0–100%, including boundaries between cases. CPU steal and cgroup throttle
increments were zero in the recorded interval. This is not a continuous GPU
trace or proof of exclusive CPU access. The next useful performance investigation
is public-data packing, policy-group dispatch and small kernel launches; these
measurements do not justify buying a larger GPU or reducing model precision,
history, legal support or PPO work.

Training is **not bit-identical** across full 24-update repetitions, including
repetitions of the same implementation. Their aggregate prefix/round/sample
traces first differ around update 7. The cause was not isolated in this run;
the report does not assign all trajectory changes to the optimizations or
claim the long-run optimization trajectories match. Fixed-fixture CUDA parity
passed, but sustained model-quality acceptance still needs controlled replay/
update analysis and a predeclared development-baseline comparison at matched
training compute. Two timing repeats and one training seed do not provide that
strength evidence. Keep the dense reference and optional flags until that gate.

The earlier full local regression passed 727 Python tests (6 skipped), 81 C++
cases, 14 oracle groups and three 20,000-round deep-fuzz modes. This turn added
CUDA parameterization to the sampled-action test: the focused local file passed
7 tests with 5 CUDA skips, followed by the complete 98-test remote gate above.
No additional full local regression was claimed for that test-only addition.

## Artifacts, shutdown and cost

Run kit: `.work/runpod-history-stack-2026-09-26/`. Immutable approval/manifest,
source archive, payload and checksums are alongside `local/results/` raw logs,
metrics, population audits, comparison results, profiles and telemetry.
`analysis.json`, `final-audit.json`, `analyze.py` and `final_audit.py` provide
reproducible post-processing and checks of the run identities and receipts.

All **83 remote files** were downloaded and SHA-256 verified **before** normal
pod deletion; the final local audit verified them again. Both monitor and an
independent provider readback confirmed pod `k1cau5ywz3uctt` absent, zero pods,
and **$0/hour current spend**. The detached guard and keep-awake controller were
also bounded by the allocation lifecycle.

Total allocation: **619.44 seconds (10.32 minutes)**. GPU rate $0.74/hour plus
conservative disk allowance gave $0.744464/hour, for an **estimated $0.12810**
total, below the approved $1.20. This is a duration/quote estimate; provider
billing line items were not queried. No replacement allocation, budget
extension, production default change or playing-strength promotion occurred.
