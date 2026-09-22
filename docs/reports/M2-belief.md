# M2 Transformer belief gate

September 21, 2026 (local date). Status: complete; supports v2 RL experimentation.
This is an architecture experiment under DESIGN §7.4, not a playing-strength
claim or a full Transformer RL implementation.

## Why repeat the first probe

The pilot used 317 rounds and six held-out matches. It also gave the history
model only a linear private projection and no post-attention MLP, while the
flat model had four nonlinear layers. That did not implement all the capacity
specified in §7.2. The archived code and results are preserved; their negative
result applies to that small prototype, not a definitive architecture choice.

The new `train/belief_model.py` adds a nonlinear private query and a residual
MLP after cross-attention. It retains a causal public stream and never sends
private features into that stream. A separately trained `no_history` model
starts with exactly the same tensors and sees only BOS throughout fitting and
evaluation. This isolates the contribution of public history from the new
private pathway. Parameter counts are 935,178 (flat) and 935,398 (history and
no-history), a 0.024% mismatch. This remains a two-layer, width-128 prototype;
the proposed four-layer, width-256 production policy has not been integrated.
Disabling history reduces active capacity and computation despite retaining
the same nominal parameters; the control is not an equal-runtime comparison.

## Predeclared protocol

The protocol was saved before collection/fitting to
`.work/belief-v2/protocol.json`. All computation uses the local CPU; no new
RunPod node was rented.

- Frozen final M1 policy, weight identity
  `8a8e2b08dec2e73998092a67cbc25c26df75b4c0d2f02fd0a791d5d16cd4f745`.
  All four seats use FP32 argmax play and heuristic tribute under house rules.
- 4,096 fresh rounds, collection seed 20260927, 64 environments with equal
  round quotas. Public exchange events are included in the stream as well as
  plays/passes; private exchange structure flags are cleared.
- Whole matches split 70/15/15 by seeded SHA-256, split seed 20260928. Training
  sees 285 matches / 2,861 rounds / 147,349 decisions. Validation sees 60
  matches / 597 rounds / 30,931 decisions. Test sees 60 matches / 638 rounds /
  32,913 decisions. Match sets are disjoint; final source matches can be partial
  because collection is bounded by round quotas.
- Fit seeds 31, 32 and 33. Each model gets the same sampled training decisions
  within a seed, Adam learning rate 0.0003, batch size 64, at most 6,000 updates.
  Check validation every 1,000 updates. Early stopping requires at least 3,000
  updates and two consecutive non-improving validations. Select the checkpoint
  with lowest validation loss, then evaluate the test set once.
- Report stage/relative-seat loss and whole-match paired bootstrap intervals
  (10,000 resamples). Overall log loss weights decisions; paired confidence
  intervals give each match equal weight. Both are identified separately.
- Support v2 RL experimentation only if all three seeds improve flat test
  log loss by at least 1%, with positive lower paired confidence bounds and at
  least 20 test matches. History must also improve the no-history control in
  every seed, with positive lower bounds in at least two seeds.

Models share one collection and test split across optimizer seeds. This is a
comparison at matched parameter counts and maximum updates, not equal runtime.
A passing belief gate still requires an equal-compute playing-strength test
after RL integration. A bounded negative result does not rule out Transformers.

## Collection and evidence

Collection completed in 12.83 seconds, producing 211,193 play decisions from
405 match groups. Every saved round passed shape/range, causal-prefix, target
cardinality, unseen-card conservation, public-known-holding and metadata checks.
The corrected known-holdings engine from Stage A2 is used throughout.

- Dataset: `.work/belief-v2/data/`.
- Dataset SHA-256: `6dc050ce9749bfea90b8ba9890c225182b86bba80650a0b227445c118c13da13`.
- Engine source: `2227731f7fc6972d2d3766211fcfc5528ee3ff96b5f21590ecb58c9359f8bc55`.
- Live metrics, reports and selected probe weights: `.work/belief-v2/experiment/`.

## Results and decision

All nine fits completed 6,000 updates. Validation selected the final checkpoint
in each case; learning curves were still improving, so this is a fixed-budget
comparison, not a claim of convergence. Collection took 12.83 seconds; the
complete fit/evaluation runner took 801.50 seconds on the local CPU.

| Fit seed | Flat test log loss | No-history test log loss | History test log loss | Reduction vs flat | Reduction vs no-history |
|---|---:|---:|---:|---:|---:|
| 31 | 0.360539 | 0.329055 | 0.327628 | 9.13% | 0.434% |
| 32 | 0.364116 | 0.326512 | 0.325812 | 10.52% | 0.214% |
| 33 | 0.366744 | 0.326616 | 0.325909 | 11.13% | 0.216% |

The following intervals use equal-weight, whole-match paired differences;
positive means lower loss for the history model. These differ slightly from
the decision-weighted loss differences in the table above.

| Fit seed | Flat minus history: 95% CI | No-history minus history: 95% CI |
|---|---|---|
| 31 | [0.032474, 0.033957] | [0.001162, 0.001897] |
| 32 | [0.037732, 0.039152] | [0.000303, 0.000995] |
| 33 | [0.040200, 0.041587] | [0.000187, 0.000911] |

**Decision: implement the v2 RL experiment.** Every seed passes both the
flat-model and history-specific checks. History also has a positive point
estimate in all nine stage/seat cells for every seed; no per-cell significance
claim is made. Most of the gain over the original MLP is also present without
history, so attributing the full 9–11% to attention would be incorrect.

This supports moving from the supervised prototype to a playing policy. It
does not show a playing-strength gain, and no baseline policy is promoted.
The per-decision probe recomputes prefixes: history training took 177–188
seconds per seed versus 13–14 seconds for flat and 26–32 seconds for no-history,
excluding validation. These timings were not isolated throughput benchmarks;
the next comparison must use an efficient sequence implementation and equal
compute. Keep the improved no-history control in that comparison.

The full metrics, stage/seat results, learning curves, checkpoint hashes and
protocol are summarized in [`M2-belief-summary.json`](M2-belief-summary.json).
Raw match-level losses and selected probe weights remain in
`.work/belief-v2/experiment/`. All seeds share the same 60 test matches; they
are three optimization runs, not three independent datasets.

## Shared history cache foundation

`train/history_cache.py` implements a bounded public KV cache for one
environment. Each public event is encoded once; all seats can query its shared
memory. `encode_private` exposes the state embedding for a future action-value
head, while `query` applies the current belief head. Private queries do not
mutate public memory. `clear()` resets to BOS at round boundaries. Ordinary
weight/device/dtype changes invalidate the cache; rebuilding it requires
replaying the public prefix under the new frozen weights. The default bound
is 160 events, excluding BOS; overflow raises instead of discarding history.

Incremental/full-prefix equality passes in float32 and float64. A check using
the trained seed-31 weights covered 331 real decisions from eight rounds,
through 87 public events, with maximum absolute logit difference
**0.000003815** (CPU FP32, tolerance 0.00001). Evidence is retained in
`.work/belief-v2/cache-parity.json`. This is a single-environment inference
foundation; batched rollout caches and round-sequence RL replay are still
pending, and no playing policy has switched to v2.

## Correctness verification

The final full Python suite passes **341 tests**. New coverage includes private/public
separation, future/padding invariance, empty history, no-history invariance,
target-seat and provenance validation, grouped splits, validation selection,
paired statistics, and incomplete timeout status. No C++ source changed in this
step; the preceding Stage A2 run passed 58 C++ test cases.

An independent audit streamed every saved round, reproduced the dataset hash
and disjoint split counts, and checked model/source/engine fingerprints. The
shared MLP helper `train/model.py` has SHA-256
`e079ac2a66b576d95d9f1153796da4a004efe7d24f4f30d6c2816b032554d51b`.

## Next implementation boundary

The remaining v2 work is the playing-policy fusion heads, whole-round replay
and causal training masks, batched rollout caches, and explicit cache rebuilds
when model weights are published. Then run a bounded equal-compute comparison
against both the v1 reference and improved no-history control. The original
external benchmark gates remain unavailable. Stage A2's heuristic decision
stands; the critic, PPO and league are separate pending M2 components.
