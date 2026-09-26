# Overnight training and parallel development plan — September 24, 2026

> Historical record. The proposals, budgets and next steps below belong to
> this dated experiment. The active route is now random-start Transformer
> self-play, with old MLPs used only for evaluation; see
> [DESIGN.md](../DESIGN.md) and [STAGE_C_TODO.md](../STAGE_C_TODO.md).
> Preserve the recorded results and artifacts; do not launch an old plan as
> the new experiment or infer current provider state from its snapshot.

Status: execution approved and started at 2026-09-24 06:21:44 UTC.
Absolute session deadline: 2026-09-24 13:21:44 UTC. Runtime state and the shared
budget reservations are in `.work/overnight-20260924/state.json`. Live RunPod
balance at launch was USD 50.3131313317 with no existing pods or current spend.

## Scope and duration

Complete a 6–7 hour session after execution is approved. The primary agent
owns cloud resources, the shared budget ledger, integration, and final results.
Up to three GPT-6-sol subagents develop independent experimental features.
Scheduling, concurrency and experiment duration adapt to the remaining GPU
budget and the current account-wide Codex usage limits; the schedule is not fixed.

The primary GPU continues training from the B11 main checkpoint for roughly
4.5–5 hours using a fixed source snapshot and explicit opponent manifest.
Package the selected league opponents; warm_start alone does not restore the
B11 opponent pool. Measure B11 against DanLM before training: the previously
reported 11.5% round win rate belongs to B8, not B11.

## Budget

The user corrected the currency to USD. The aggregate ceiling is USD 20 across the entire session, including the main
run, every subagent experiment, setup, evaluation, storage, and cleanup.
No individual subagent may spend more than USD 15; this is an individual
ceiling within the shared USD 20 total, not an independent entitlement.
The verified starting provider balance was USD 50.3131313317. This does not
increase the USD 20 session ceiling.

Start with an approximately USD 5 reservation for the primary training and
evaluation resource, adjusted to its live quote and intended duration. Keep
the remaining allowance unallocated. Fund A/B/C only as tested implementations
and informative experiments become ready. Use expected marginal information,
available time and remaining Codex capacity to rank requests; do not split
the balance equally or rent hardware just to consume the budget. Reserve
cleanup costs before allocating any experimental extension.

The primary agent alone allocates and creates paid resources. Each reservation
must include an upper bound for the full paid interval and cleanup; outstanding
reservations count against the aggregate ceiling. Child agents request an
experiment reservation rather than renting resources independently. The primary
agent can reallocate unused envelopes without asking the user again, while
respecting both ceilings. Stop earlier if experiments are uninformative.

RunPod bills in USD; no CAD conversion is needed. Include any applicable fees
and cleanup headroom. Enforce the earlier boundary from the USD 20 aggregate
cap, each reserved experiment allowance, and the session deadline.
The last measured RTX 4090 rate was USD 0.74744/hour including storage; it is
historical evidence, not a current quote. Recheck allocation prices before use.

Start with one training GPU. Rent at most one additional shared experiment GPU
only when tested code and a bounded comparison are ready. Run its experiment
arms sequentially to avoid resource contention. Delete idle resources promptly.
If A and B are ready together with compatible settings, reserve part of the
shared reserve for one matched control and run control/A/B for equal training
time on that host. Otherwise fund each necessary control explicitly or limit
the result to a functionality check. Failed pilots do not automatically receive
their unused budget again; decide whether another attempt is informative.

## Parallel deliverables

**A — Learning signal.** Implement advantage-magnitude filtering behind a
default-off configuration flag, with a threshold fixed across an update,
correct pass handling, diagnostics, and checkpoint/resume compatibility.
Verify the unchanged default and meaningful boundary/integration cases. If
ready in time, run a short equal-budget control/treatment comparison from the
same checkpoint and opponent manifest. A short pilot does not establish G8.

**B — Candidate exploration.** Implement frozen-reference top-32 plus a small
uniform sample from the remaining legal actions, retaining legal pass. Store
the exact sampled support and behavior log-probabilities; PPO updates must not
resample support. Verify sampling, normalization, replay consistency, checkpoint
compatibility and unchanged old-mode behavior. Any GPU comparison uses its own
equal-budget control; the longer main run is not a matched control.

**C — Endgame search.** First build a safe determinization interface that uses
only the observer's own hand and public constraints. Existing VecEnv.fork copies
real hidden hands and must not be used as an information source. Implement a
bounded uniform-legal search prototype with a blueprint fallback, test card
constraints and information isolation, then compare with the same blueprint
on a bounded scalar duplicate-deal suite. Record actual latency and valid
sample rate. Full C3/G9 and learned-belief search are outside tonight's minimum.

Use separate worktrees for development. Keep the running training snapshot
immutable. The primary agent merges shared interfaces sequentially, reviews
correctness, and runs the relevant integrated checks before any cloud pilot.
All substantial training is remote; local work is development, tests and
bounded inference evaluation. Do not launch long local fitting jobs.

## Monitoring without continuous model polling

Live usage query at 2026-09-24 06:19:53 UTC reported the Codex 7-day window
4% used / 96% remaining. The provider returned no secondary window, so a
5-hour remaining allowance is unknown. The reported reset is epoch 1790832697.
These are account-wide limits and may change as other tasks run. They are
separate from the RunPod balance and do not map to a known token count.

Recheck usage at each heartbeat and before assigning substantial new model
work. Start with up to three sol agents while capacity is ample; reduce
concurrency and scope as the limiting reported window approaches the planned
15–20% integration/cleanup reserve. Finish valuable in-progress work before
opening new research directions. If capacity drops sharply or ordinary usage
is blocked, stop new model-dependent experiments and let the independent
scripts finish/stop, back up and release already-running resources. Never buy
Codex credits, redeem a reset or expand the GPU budget automatically. Do not
infer an unavailable usage window or assume the current 96% will remain.

Once execution starts, use a thread heartbeat every 30 minutes. A check reads a
compact status summary: phase, update count, latest evaluation delta, checkpoint
age, failures, paid resources, actual/estimated cost, reserved remainder, and
fresh Codex usage. Avoid repeated account queries between these decision points.
Read full logs only when diagnosing an anomaly. Do not repeatedly poll healthy
runs between heartbeats or generate conversational progress to fill waiting time.

Ordinary scripts handle frequent checkpoint saving, backups, failure detection,
time limits and teardown. Model wakeups are not the budget enforcement mechanism.
Verify ordinary-process deadline/teardown and backup behavior before long training.
The desktop must remain powered and online for local evaluation and monitoring.

Deployment evidence: provider `terminateAfter` did not release the disposable
probe at its deadline, and the provider-injected pod key rejected the own-pod
API request. Neither is treated as a verified safety mechanism. A detached
local deadline guard and separate backup supervisor, both under `caffeinate`,
enforce owned-pod cleanup independently of model availability; their provider
access still depends on this Mac's power and network. Probe deletion through
the account's local API client was confirmed. No account key was uploaded.

The main RTX 4090 is `oyzhj7bbgjoi3t`; the matched experiment RTX 4090 is
`flemmcad4fr87j`. Each allocated GPU costs USD 0.7459523809523809/hour including
the conservatively estimated storage. Reserve USD 5 per GPU and USD 0.65 for
startup/teardown probes (USD 10.65 total). Main CUDA training/resume preflight
and 27 cloud regression tests passed. Main source is immutable `4940246`;
experiment source is `cf40250` (default-off filtering and candidate union plus
the independent evaluator). The later search integration does not alter these
archives. Matched arms each receive 4,500 seconds with the same seed, explicit
league and B11 warm start; the longer main continuation is not their control.

Stay quiet while nothing actionable changes. Notify the user on material
results, completion, failure, or required action. Stop the heartbeat when all
owned resources are confirmed released and the final report is written; stop
dispatching new work at the 7-hour session deadline.

## Evaluation and morning handoff

Track intermediate checkpoint trends using development deals. Evaluate the
selected final candidate on separate held-out deals, including DanLM, the B11
start and held-out opponent styles. Target at least 4,000 duplicate deals for
final strength comparisons and include internal full-match checks, with paired
intervals and explicit limitations. Save raw outcomes and all exclusions.

Return checkpoints, tested code changes, experiment configurations, source and
artifact hashes, strength/latency results, the USD ledger, and provider
confirmation that owned resources are released. Distinguish implemented,
unit/integration-tested, pilot-tested and statistically supported improvements.
