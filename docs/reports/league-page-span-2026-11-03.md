# League page-span validation and adoption — November 3, 2026

## Scope

The user requested speeding up the resumed league experiments. Inspection of
the actual live manifests showed `torch_threads=1` in both arms and page-span
already enabled in `league-near`, but disabled in `league-filtered`. This receipt
tests and adopts only that existing rollout-cache option in the filtered arm.
It does not alter full public history, FP32, actor architecture, canonical
actions, rewards, PPO batches, population schedules or the 130,000-update target.
Main and the other experiments remain held.

The [earlier league continuation receipt](league-resume-2026-11-03.md) owns the
probe-resume correction, source identity, storage preflight and automatic
continuation policy. Source remains its isolated
`src-league-resume-2026-11-03` copy; there is no new trainer code change here.
Page-span trims unused padding to page boundaries; it does not discard real
history positions. Earlier two-/three-GPU evidence remains in the
[October 13 report](training-startup-ecc-2026-10-13.md).

## Current-checkpoint gates

Jobs **19046** (rollout and timing) and **19047** (update-state comparison)
completed successfully on the spare Slurm-allocated L40S. The two live training
jobs continued during these isolated tests. A frozen copy of the filtered
checkpoint was used, and diagnostic updates were never saved over a live
training checkpoint.

- The existing CUDA page-span cache test passed.
- Real-checkpoint rollout comparison covered **40,602 records**: all non-logp
  fields and sampler RNG matched; stored logp and behavior-logp differences
  were zero. Full-public-history recomputation error was **3.05176e-5** in
  both arms, below the existing **1e-4** gate.
- A separate deterministic current-checkpoint comparison collected four
  rollouts and performed one actual PPO update with each setting. All **556
  compared tensors**, including actor/critic parameters, gradients, optimizer
  state and sampler RNG, matched exactly: maximum absolute difference **0**.
  Loss fields also passed the comparison. The allowed numerical tolerance was
  rtol 1e-5 / atol 1e-6; observed tensor equality was stronger. This update check
  was single-process; the timing experiment below used the live four-rank layout.

## Complete-PPO timing

One L40S, four ranks, same resumed filtered checkpoint and batch layout,
FP32, full history. After 12 warm-up updates, four paired blocks alternate
off/on and on/off. Each block has six updates per setting; its first two
updates are excluded. Total: 60 diagnostic updates, 16 retained per setting.
The historical script's unrelated projection patch was removed; only
`rollout_page_span` differs. Timings include collection and learning, but not
periodic checkpoint writes.

| Paired block | Off, seconds/update | On, seconds/update |
|---|---:|---:|
| 0 | 8.7850 | 8.2426 |
| 1 | 8.9181 | 8.0916 |
| 2 | 9.1360 | 8.3925 |
| 3 | 9.0906 | 8.2579 |
| Mean | **8.9824** | **8.2462** |

Measured throughput gain: **8.93%**, with improvement in all four pairs.
Exact enumeration of paired-block bootstrap resamples gives a descriptive
95% speed-ratio interval **[1.0746, 1.1015]**. These four blocks are not
independent training seeds or repeat jobs; the interval describes this bounded
run's block variation. Adoption required both numerical gates, at least a 5%
mean gain, and a paired-block lower bound above 1. This is engineering speed
evidence, not a playing-strength result or a guarantee for every future prefix.

## Adoption and operational state

Only filtered training was cleanly stopped at an update boundary using its
existing USR1/save/sync path. Job **19008** completed successfully; its queued
successor **19045** was held during the transition and then cancelled after
replacement submissions succeeded. New filtered job **19048** resumes the
saved checkpoint with `rollout_page_span=true`. Successor **19049** waits
`afterok:19048` with the same setting and guarded continuation policy.

Near job **19007** and successor **19044** are unchanged. Each training arm
still uses one GPU and four ranks. Update 130,000 remains the stopping point;
allocations remain capped at 12 hours. No compute was rented. The spare GPU was
released after the finite diagnostics.

Live manifest and post-restart progress verification are recorded with the
application evidence. Initial post-resume timings should not be interpreted as
steady-state throughput while histories refill. This change speeds only the
filtered arm; near already had the option enabled, so an 8.9% filtered-arm gain
does not imply an 8.9% reduction in the two-arm makespan.

## Artifacts

Remote: `~/guanzero/league-speed-2026-11-03/` holds the frozen-checkpoint receipt,
scripts, job logs, gate outputs, per-rank timings, `decision.json` and
`applied.json`. `results/update-parity/result.json` records the 556-tensor gate.
The updated continuation mapping is in
`~/guanzero/league-resume-2026-11-03/automatic-continuation.json`.

Local: `.work/league-speed-2026-11-03/` contains the scripts and retrieved
diagnostic results. These ignored artifacts exist locally but are not portable
repository dependencies. Their receipts must be checked before reusing them.
