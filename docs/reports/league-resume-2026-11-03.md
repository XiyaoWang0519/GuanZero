# Velocity league continuation — November 3, 2026

## Scope and authorization

The user authorized continuing the experiments after reviewing the recommendation
to prioritize `league-near` and `league-filtered`, and requested checking that the
earlier slowdown does not recur. Only these two arms were resumed. Main,
next-coefficient, QK-normalization and fusion/habit training remain stopped.

The five campaign STOP files explicitly record the October 26 01:53:52 UTC
user-requested pause for NFS slowness. Login recovery alone was not treated as
evidence of compute-node storage recovery.

## Source and continuation boundary

The original remote `src-campaign-2026-10-20` was preserved. An isolated copy,
`~/guanzero/src-league-resume-2026-11-03`, changes only the trainer's lazy behavior
probe initialization and adds the existing three probe-resume regressions. It
unconditionally rebinds retained archive fingerprints to the current process's
frozen inputs, publishing those inputs only after rebinding succeeds. The old
shape/count check could compare probabilities from different probes of equal
shape after a restart. Historical filtered-arm results are not retroactively
certified by this correction.

The correction was extracted from the local `codex/probe-resume-lifecycle`
worktree's October 26 implementation; its broader configuration refactor and
habit-resume changes were not imported. The remote population rebind already
computes a complete replacement mapping before assigning it.

Source identity: `3855c274b22a4e000d9a5a34151639c85b97012b5944941f7eb63c280bcb15ea`.
Both arms record the source transition through the existing resume identity
mechanism. The rules engine, FP32 precision, public full-history inputs,
architecture, reward/PPO settings and arm-specific population schedules are
preserved.

Each arm uses its own latest actor/critic/optimizer checkpoint, four ranks on one
Slurm-allocated L40S, 256 environments and 16 minibatch matches per rank. The
existing node-local staging and guarded durable synchronization remain enabled.
MPS remains off. No automatic successor submission or crash restart is enabled.
Stop at absolute update **130,000** or the job's approximately 11.73-hour training
deadline (12-hour Slurm allocation), whichever comes first. This is a bounded
continuation, not authorization for an indefinite campaign.

| Arm | Saved update | Saved cumulative decisions | Job |
|---|---:|---:|---:|
| league-near | 121,939 | 7,991,394,304 | 19007 |
| league-filtered | 123,454 | 8,090,681,344 | 19008 |

The unequal starting exposure remains a comparison limitation. Strength
evaluation should use matched exposure and fixed paired deals; training loss
and throughput are not playing-strength evidence.

## Preflight measurements

CPU preflight job **19004** completed on `max-compute-11`: **24 passed, 2 skipped**
across probe-resume, population and auxiliary-head tests. The node-local Python
environment was rebuilt after the compute-node reboot: Python 3.12.14,
Torch 2.11.0+cu128. Source staging verified all 361 recorded files; import guards
confirmed Python/Torch/native-engine mappings do not come from `/home`.

Both saved checkpoints passed SHA-256 identity and recursive finite-tensor
checks, including their serialized optimizer state. Compute-node storage tests
read the full checkpoint, wrote/fsynced an isolated evidence copy, renamed it
atomically and verified its hash:

| Arm | Checkpoint read | Write + verify |
|---|---:|---:|
| league-near | 0.143 s | 0.344 s |
| league-filtered | 0.228 s | 0.586 s |

Checkpoint hashes before resume:

- near: `7e3606bafcf780b2656543a3eb53fb4e8440d08cee3ec78fff8ce53e61966cb6`
- filtered: `9cd0ec42087a22065a2004803cffea12ac0d5da8fab173b9aff371c55264e7e4`

Both jobs were observed RUNNING and passed allocation-specific GPU health probes.
Their GPUs are distinct (allocation indices 2 and 0); the third configured GPU
was left available. Slurm reports about 1.6 TB RAM, correcting an earlier verbal
misreading of the abbreviated `sinfo` field.

## Slowdown verification

The last 100 saved pre-pause updates had median collect/learn times of
4.65/3.36 s (near) and 5.60/3.34 s (filtered). These are reference measurements
from the paused runs, not an independently established pre-incident healthy
baseline. Sequence lengths, startup and checkpoint overhead must be considered
when comparing them with the resumed run.

At 18:46:10 UTC (roughly five minutes after trainer launch), near had completed
42 new updates, reaching 121,981; filtered had completed 37, reaching 123,491.
Excluding the first five warm-up updates, median collect/learn times were
4.53/3.22 s and 5.56/3.23 s respectively, with mean prefix lengths about 558
tokens. Recent GPU samples were 93–99% (near) and 81–100% (filtered); these
are instantaneous samples, not sustained utilization averages. All observed
policy/value loss, entropy and KL fields were finite. Both remained on their
first attempt without a fallback. Manifests confirm target update 130,000 and
the recorded source transition. Filtered's rank-0 population log records
`behavior_probe_rebind` at update 123,456 for all 16 retained archive members.

Both first scheduled durable syncs succeeded (`rc=0`) in less than the shell's
one-second timing resolution, at 18:45:59 and 18:46:01 UTC. A five-second host
sample reported zero I/O wait, no blocked processes and no task of this account
in uninterruptible sleep; short-term I/O pressure was zero. No recurrence of the
NFS stall was observed in this initial window. This does not certify an entire
12-hour run or rule out intermittent storage failures. The first new periodic
training checkpoint was not yet due in this window; full checkpoint-sized
durable writes were exercised separately by preflight. Existing persistent
sync-failure guards remain active.

## November 3 amendment: automatic continuation authorized

After reviewing the measured estimate of roughly 15–17 additional hours to
update 130,000, the user explicitly authorized automatic resubmission. This
supersedes the initial no-successor setting above; the training target and
12-hour per-allocation limit are unchanged. Running jobs were not interrupted
or modified: Slurm executes its submitted script copy, so editing the original
file alone would not arm those jobs.

| Arm | Running job | Queued successor | Release dependency |
|---|---:|---:|---|
| league-near | 19007 | 19044 | afterok:19007 |
| league-filtered | 19008 | 19045 | afterok:19008 |

Both successors were verified PENDING with reason Dependency. Their isolated
`train-auto.sbatch` retains the existing source, arm configuration, allocation
guards and absolute update target. It permits another successor only after a
healthy deadline stop with a saved update below 130,000, successful durable
sync, no sync-failure marker, and no STOP marker. Successors also use afterok
dependencies, so a failing final EXIT sync prevents release. Invalid dependencies
are cancelled with `--kill-on-invalid-dep=yes`. A pending-job check prevents
stacking successors. External TERM stops do not trigger further submissions;
crash retries remain disabled. Only the two league arms are included.

## November 3, 19:11 UTC progress and handoff

Live scheduler and node-local metrics were checked at 19:11 UTC (14:11 Toronto).
The [machine-readable snapshot](data/league-progress-2026-11-03T1911Z.json)
preserves the exact counters, sampling windows and timing caveats.

| Arm | Running job | Current update | Cumulative decisions | Measured decisions/s | Seconds/update | Queued successor |
|---|---:|---:|---:|---:|---:|---:|
| league-near | 19007 | 122,167 | 8,006,336,512 | 8,206 | 7.99 | 19044 |
| league-filtered | 19048 | 123,663 | 8,104,378,368 | 9,303 | 7.04 | 19049 |

Rates are global environment decisions including collection and learning,
computed from cumulative-counter differences. Near uses 49 update intervals;
filtered uses 40 post-restart intervals, including short-history warm-up. The
filtered number is preliminary and is not a replacement for the paired +8.93%
benchmark result. Raw rate-based remaining times are approximately 17.4 hours
and 12.4 hours respectively; the latter is particularly sensitive to warm-up.
Queue delay, future sequence lengths and checkpoint overhead can extend them.

Both latest loss/KL/entropy records are finite, and metrics were less than
three seconds old. Near's durable sync succeeded at 19:11:01 UTC; filtered's
first post-restart scheduled sync succeeded (`rc=0`) in one second at
19:11:19 UTC. No recurring NFS stall was observed in these checks; intermittent
future failures remain possible and guarded. Two training GPUs are in use;
the finite speed/parity diagnostic jobs completed and released their allocation.

Current continuation rules:

- Target absolute update **130,000** in both arms, preserving their separate
  actor, critic, optimizer and population checkpoints.
- Each Slurm allocation is capped at 12 hours; healthy deadline stops may
  automatically continue through afterok-dependent successors.
- Honor STOP markers, non-finite/terminal failures and durable-sync failure
  guards. Do not change the target or silently retry crashes.
- Both arms use FP32, full public history, four ranks / one L40S and their
  existing PPO/global-batch settings. The filtered arm's only new setting is
  page-span enabled at update 123,622; its resume is recorded in the manifest.

Held training: main, next-c01/c03/off/anneal, QK-normalization and fusion/habit.
No new evaluation result or production-policy adoption is claimed.

Remaining work after the bounded training:

1. Confirm terminal jobs and durable, loadable checkpoints/optimizer states;
   verify that no further successors remain once the target is reached.
2. Compare league arms at matched training exposure on a fixed paired deal set.
   Preserve the filtered arm's historical probe-resume limitation; the November
   correction does not retroactively validate prior archive decisions.
3. Evaluate existing next-coefficient checkpoints before extending those arms.
   Keep main and other diagnostic training held pending evidence.

For a later status check, re-read Slurm and the durable run manifests/metrics;
these job IDs and estimates may have changed. Do not restart an already active
arm or submit a duplicate successor based on this dated snapshot.

## Evidence pointers

Later November 3: the [page-span speed receipt](league-page-span-2026-11-03.md)
records a validated filtered-only setting change and clean checkpointed restart.
Current filtered jobs are 19048 / 19049, replacing 19008 / 19045; near jobs
19007 / 19044 and the target/continuation guards remain unchanged.

Remote evidence: `~/guanzero/league-resume-2026-11-03/` contains preparation,
preflight, launch and monitor records, bounded launch scripts, and backups of
the original STOP marker text. Ordinary durable run logs remain under
`~/guanzero/runs/{league-near,league-filtered}/`.

Local orchestration/evidence: `.work/league-resume-2026-11-03/`. This ignored
directory is not a portable dependency. No playing-strength result or final
completion is claimed by this launch receipt.

## November 4 terminal checkpoint acceptance

Successors 19044 (near) and 19049 (filtered) exited cleanly and reported
successful final durable syncs. The Slurm queue was empty at 20:42 Toronto.
Both durable `segments/lr-main/latest.pt` files report u130000 and
8,519,680,000 decisions.

A CPU allocation on max-compute-11 loaded both files with the deployed
league-resume source (node-local source ID `ed082351e0662ae3`). Strict actor
and critic loading, Adam state loading and moment shape checks (103 actor
and 8 critic state entries per arm), recursive tensor finiteness, host RNG
restore and population snapshot digest validation all passed. Near has 13
population snapshots and 0 archive members. Filtered has 28 and 16,
respectively. The CPU check did not restore CUDA RNG or execute a resumed PPO
update.

| Arm | SHA-256 |
|---|---|
| Near | `43d849d5026101523205ff9823703659c2d5819f3bf4702aac82cdb21b8e6863` |
| Filtered | `d6d5c514682924bd7bb8ffff1cf311f06bb601e367ab904bca9ddef33e2a6bd5` |
