# T7: shared representation versus explicit response input

Status: **complete** (September 27 2026 UTC); see
[results](history-response-results-2026-09-27.md). The primary run used CPU game
simulation with CUDA model inference/training after the user's throughput
clarification and same-host measurement (1,575 decisions/s for three arms,
versus 876 for CPU and 1,359 for CPU rollout with CUDA learning). The
interrupted CPU attempt's $0.6157 estimated cost is inside the $6 cap.

See [device-placement measurements](history-device-placement-2026-09-26.md).
Current kits, execution and combined ledger are in
`.work/history-response-speed-2026-09-26/`; the original CPU artifacts are retained.

## Question and arms

Does explicitly giving the policy predicted opponent responses improve play
over training the same prediction task through a shared representation?

| Arm | PPO | Response loss | Response probabilities in policy |
|---|---|---|---|
| A | yes | no | no |
| B | yes | cross-entropy, coefficient 0.1 | no; shared features only |
| C | yes | identical to B | yes; detached probability vector |

Primary comparison: C minus B. B minus A and C minus A are secondary.
B and C have identical parameter tensors and initialization. Their policy
bridge is the same linear layer, initially zero; B gives it a zero feature,
while C gives it `softmax(response_logits) - 1/23`. All three initial policies,
shared parameter tensors and critics are identical for a given seed. Optional
head initialization preserves the base random-number stream. A remains the
default configuration; loading existing history checkpoints still works.

C retains the full original state/candidate features. The response branch
adds features; it does not replace history or remove any candidate. Predictions
are detached at this connection: the prediction head is trained by the same
supervised objective in B and C, while PPO trains the bridge and policy to use
its output. This experiment addresses that specific connection, not every
possible joint-gradient or recurrent architecture.

## What is predicted

For every executed learner Play action, use its pre-action public history,
acting player's own observation, and chosen candidate. Predict the **first
public Play event by either opponent** following that action, before the
observer acts again or the round ends. Partner actions can intervene.

The 23 classes are:

- 0: no opponent response within that horizon;
- 1–11: relative seat 1, action type Pass through JokerBomb;
- 12–22: relative seat 3, those same action types.

The first pilot predicts seat and action category, not exact cards or rank.
An engine-resolved pass has exactly the same label as a voluntary pass.
Completed-round labels come from the stored public stream; the executed
candidate must match the event at its saved prefix. Unfinished rounds cannot
be assigned a no-response label. No target seat, target event, other player's
hand/legal list, forced flag or future token enters the actor's inputs.
Only the executed candidate gets a target. The head predicts for all legal
candidates at inference, but unchosen counterfactual outcomes are never invented.

## Fixed training protocol

- Seeds: **2026092801, 2026092802, 2026092803**. Each seed has all A/B/C arms.
- Fresh width-64, two-layer, four-head actor; independent random critic.
- Two PPO epochs, lr 3e-4, entropy coefficient 0.01, minibatch 2 matches.
- 32 environments, 64 steps/update; full history, KV cache and causal SDPA.
- Original snapshot population: probability 0.5, four recent snapshots,
  published every two updates. Only current/own-lineage Transformers train.
- FP32; TF32 disabled. No MLP transfer, training seats, labels or filtering.
- Three parallel arms on the same remote pod per seed; at least 16 usable
  CPUs, 2 engine threads and 2 Torch threads per arm. The game engine remains
  on CPU. Rollout inference and learning placement are selected by same-host
  complete-update measurements, including transfers and three-arm contention.
  Both devices retain FP32 and the full public history. The new optional
  `rollout_device` setting permits CPU sampling with a CUDA learner; current
  weights synchronize before each collection and frozen snapshots are mirrored
  without changing their identities. Checkpoints retain the sampler's device.
- The interrupted CPU run is retained as an engineering artifact. All primary
  endpoints restart from the declared random seeds using the selected placement;
  CPU and CUDA segments are not combined into a single 80-minute endpoint.
- Each arm gets 4,800 trainer seconds, finishing its last complete update;
  actual time, decisions, finalized learner rows and inference cost are reported.
  Start times and shared-host contention are retained as limitations.
- Save every 50 updates or five minutes, plus the final endpoint. The primary
  endpoint is `final.pt` at the time cap, never a score-selected checkpoint.
- Hold the coefficient at 0.1 for this experiment. A null result at this setting
  does not rule out every auxiliary objective or coefficient.

## Evaluation

Each fixed endpoint is evaluated against the frozen B11 and segment-2 MLPs
in a separate CPU inference process after the paid pod is deleted. These
models and their games never train the new actor, critic or prediction head.

Use the same **256 new duplicate deals** (two seat-swapped legs each) and
**64 complete-match seed pairs** per baseline across all nine endpoints.
Preparation freezes the generated deals, match seeds, baseline hashes and
source archive. The existing sealed final test remains unopened.

Report net levels/round and full-match win rate, all three paired training-seed
effects, and paired resampling over seeds and complete deals/match pairs.
Do not count swapped legs or repeated snapshots as independent training seeds.
Three seeds provide a small replication, not a precise general estimate.
The primary comparison gets priority over exploratory secondary comparisons;
there is no automatic default/model promotion.

Training response loss/accuracy/event frequency are diagnostics. The current
kit does not establish predictive calibration on a common held-out trajectory
set or causal use of distant history. Neither improvement in training accuracy
nor a win against one baseline establishes those mechanisms. They require
separate prediction/history-use controls if the connection improves play.

## Budget and lifecycle

Approved campaign cap: **$6 total**. Three sequential allocations, at most
**$2 and 1.9 hours each**, quoted GPU-plus-disk rate at most **$0.90/hour**.
Do not create any allocation until the campaign budget is confirmed. Preserve
the earlier campaign ledger; this experiment receives its own budget record.

Authorization was received before creation. The first create returned HTTP 500;
two independent provider reads found zero pods, zero hourly spend and no balance
change before a new attempt. The original allocation cost approximately
$0.724464/hour including its 30 GB disk and had 27 usable CPUs. It was stopped
on the user's correction; its 50.99 allocated minutes cost an estimated $0.6157.
The replacement uses the same hourly rate, with its original 1.9-hour deadline
covering measurement, training, artifact download and cleanup. New allocations
have a tighter $1.75 cap, so the stopped CPU attempt remains inside $6. The campaign
controller advances only after verified downloads and independent zero-resource
readback; it stops expansion on failure and runs local inference evaluation
after all paid allocations have been deleted.

The planned Secure RTX PRO 4500 class was quoted at $0.72/hour GPU-only during
preparation, with changing availability. Recheck inventory, effective CPU
allocation and total price before creation. Availability and price are not
reserved by preparing the kit. Stay within the ceiling or stop for review.

Use the existing `infra.runpod` and `infra.history_monitor` lifecycle, with an
independent local provider guard and caffeinate. Every create requires a zero
pod/zero-spend preflight. Reconcile ambiguous creates before retrying. Verify
downloaded hashes before normal deletion, independently confirm zero pods and
zero hourly spend, and record actual allocation time/cost before the next seed.
Any failure stops expansion; save results and charge failures to the $6 cap.

## Entrypoints and implementation

`infra/history_response_experiment.py` has no provisioning or credential code.

```sh
PYTHONPATH=python:oracle:. .work/external/danlm-venv/bin/python \
  -m infra.history_response_experiment prepare \
  --output .work/history-response-2026-09-26
```

Its `workload` command refuses local research training. The owned Linux pod
must supply `POD_DEADLINE_EPOCH`; the existing monitor supplies it. Three
generated kits contain `source.tar.gz`, immutable manifests, payloads and
evaluation freezes. Preparation does not create `approval.json` or a pod.

After all three seed kits have verified downloads and provider teardown:

```sh
PYTHONPATH=python:oracle:. .work/external/danlm-venv/bin/python \
  -m infra.history_response_experiment evaluate \
  --output .work/history-response-2026-09-26
```

Core changes are `train/history_response.py`, `train/history_model.py` and
`train/history_ppo.py`. Tests cover target horizons and alignment, unfinished
round rejection, private flag independence, matched initialization, the
explicit connection's gradient boundary, causal future masking of predictions
and policy, KV/stored-probability parity, auxiliary updates and checkpoint resume.
CUDA versions of the new rollout tests must pass on the pod before training.
