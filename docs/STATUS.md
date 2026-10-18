# Project orientation for agents

Documentation review: October 6, 2026. This is an index of recorded status,
not a live-job dashboard or authorization to run experiments. Verify current
code, checkpoint identity and external state for the task at hand.

## Baseline and documented readiness

- Main route: randomly initialized public-history Transformer, self-play PPO,
  current/past Transformer population. Existing MLPs and DanLM are evaluation
  baselines. See [DESIGN.md](DESIGN.md) and [agent rules](../AGENTS.md).
- Sequence training, resume/population, history-aware evaluation and cache
  paths have implementation receipts. See [TRAINING.md](TRAINING.md) for
  operations and [TRAINING_EVIDENCE.md](TRAINING_EVIDENCE.md) for limits.
  T0--T8 in [STAGE_C_TODO.md](STAGE_C_TODO.md) are original milestone IDs.
- The [strength summary](reports/strength-summary-2026-10-04.md) includes the
  u20264 curve, October 5 search measurements and the October 6 m1/m2
  milestones (m2 u35588: -0.869 plain, -0.77 with search), and on October 8 m3
  u43276 (-0.665; search gain +0.13 on 800 deals) and m4 u50840 (-0.546), and on October 12 m5 u58528 (-0.497), m6 u66092
  (-0.425) and m7 u73780 (-0.334; -0.224 with search, gain +0.08 [-0.01, +0.17] on 800 deals). It is a dated summary, not
  a statement of today's latest checkpoint. The October 6
  [distillation plan](reports/search-distillation-plan-2026-10-06.md) also
  references m1/u27900; verify its artifacts before selecting that model.
- [Botzone search deployment](reports/botzone-search-2026-10-04.md) records a
  separate platform package and deadline-limited search. Its ladder sample
  and budget cannot be substituted for offline search evidence.

## Research directions and unresolved work

| Direction | Recorded state / next evidence needed |
|---|---|
| Continued training and auxiliary heads | [Experiment review](reports/experiment-review-2026-10-04.md), [auxiliary heads](reports/aux-heads-2026-10-02.md): distinguish lineage progress from multi-seed causal evidence |
| Evaluation reliability | The October 4 review identifies external-adapter exclusions and paired-record gaps; inspect current code before assuming these are still unfixed |
| History use and opponent diversity | [History ablation](reports/history-ablation-2026-09-29.md), [diversity proposal](reports/opponent-diversity-plan-2026-10-03.md); earlier-checkpoint findings do not establish current history dependence |
| Planted-habit diagnostic | [Phase 1a](reports/history-habit-phase1a-2026-09-30.md): ORACLE and ROUND trained/evaluated; FULL not run; one training seed per arm, separate diagnostic lineage |
| Cheaper search | [Fused search design](reports/fused-search-design-2026-10-05.md): explicitly design only, no implementation or measured gain claimed |
| Search distillation | [October 6 plan](reports/search-distillation-plan-2026-10-06.md): proposed; distinguishes offline Stage 1 from a Stage 2 design-boundary amendment |
| Learned tribute / back-tribute | Exchanges still use the fixed `tribute_bot` heuristic (DESIGN 1.3). [October 6 brief](reports/tribute-learning-brief-2026-10-06.md): open question for outside advice; paired DanLM data show different back-tribute choices but no measurable loss on tribute deals; the September 21 [A2 attempt](reports/M2-A2.md) was negative |
| Variance reduction / Q-boost | [VRPO design](reports/vrpo-design-2026-10-04.md), [experiment review](reports/experiment-review-2026-10-04.md): proposals/exploratory evidence, not demonstrated main-lineage gains |
| Runtime optimization | [PERF_TODO.md](PERF_TODO.md): implemented paths, measured scopes and remaining verification |

October 13 runtime evidence: [isolated stack A/B](reports/training-stack-ab-2026-10-13.md)
measured page-span complete-PPO throughput +10.28% on 2 L40S / 8 ranks,
with real-rollout parity passing. Production's 4 GPU layout remains unverified;
the main configuration is unchanged. Learner rollout projection failed its
numerical gate, so its performance measurement is diagnostic only.

October 13 startup incident: the next main slice stopped after repeated CUDA
initialization failures, including uncorrectable ECC errors with MPS off.
Current source identity passes. The bounded health check confirmed 3 of 4 allocated GPUs usable; page-span
complete-PPO throughput improved 8.50% on those 3 L40S / 8 ranks, with rollout
parity passing. Four healthy GPUs remain unverified and main training is not
restarted; its configuration remains unchanged. See the dated
[incident and validation record](reports/training-startup-ecc-2026-10-13.md)
for evidence and pending work; this is not a live queue status.

## Contract versus dated proposals

`DESIGN.md` is the September 25 baseline design; later test-time search and
multi-rank engineering have dated receipts. Its deferred-work list describes
that baseline, not a claim that those components are absent today. This
organization pass does not approve a new training signal or rewrite the
research contract. Use the distillation plan's explicit boundary discussion
for that proposed extension; do not infer approval from this index.

## Navigation and maintenance

[Documentation map](README.md) · [All reports](reports/README.md).
Update this page when recorded readiness or direction changes, linking the
owning source and its limits. Keep checkpoint and job claims dated; do not
copy a historical budget or an old session's authorization into a new launch.

October 13 follow-up [projection experiment](reports/training-stack-next-2026-10-13.md):
packed valid-token projection passed CUDA parity, but PPO stopped at update index 97
on a non-finite gradient norm. The completed subset showed no speed gain; this
candidate is not ready for adoption. Main configuration remains unchanged.

October 14 gradient diagnosis: [7220 results](reports/training-gradient-diagnosis-2026-10-13.md)
confirmed finite actor gradients overflowing the FP32 norm on the page-span base arm
(3 healthy GPUs, 8 ranks, 94 completed updates). Rank 1 already had a huge local
backward gradient; outcome-loss replay showed the largest component gradient.
The operator-level cause remains unresolved; this is not a successful repair or
throughput result. Main source/manifest unchanged; no new experiment submitted.

October 14 follow-up: [rank 1 fixture replay](reports/training-fixture-backward-2026-10-14.md)
reproduced the outcome loss on CPU with normal gradients, but device/version differ.
Isolated job 7323 compares CUDA default/math SDPA and repeatability before any repair
or cache/reduction benchmark. Main configuration remains unchanged.

October 14 update: [7323/7324 fixture evidence](reports/training-fixture-backward-2026-10-14.md)
localized erroneous gradient amplification to masked efficient SDPA backward on
PyTorch 2.11.0+cu128 / L40S. A grad-enabled private-attention math fallback passed
fixture finite-gradient and 1e-4 candidate/chosen-logp gates. Job 7325 now tests
120 full PPO updates on the isolated repaired baseline before cache/reduction
optimization; production remains unchanged and no new speedup is claimed.

October 14 stability result (supersedes the pending 7325 status above):
[7325](reports/training-fixture-backward-2026-10-14.md) completed 120 full PPO
updates on every one of 8 ranks / 3 healthy L40S; gradient metrics were finite
and per-update parameter checksums passed. This is bounded stability evidence.
[7366 cache/reduction comparison](reports/training-cache-reduce-2026-10-14.md)
is submitted and pending resources. It gates real-rollout/cache parity and
reduction equivalence before two separate full-PPO A/B tests against the repaired
page-span baseline. No new speedup measured; production remains unchanged.

October 14 final 7366 result (supersedes pending status above):
[cache/reduction evidence](reports/training-cache-reduce-2026-10-14.md) passed
rollout and reduction parity; both stages completed 8 ranks × 120 updates on
3 healthy L40S. Against repaired page-span baselines, cache throughput +1.23%
(paired block 95% interval +0.18% to +2.19%); pinned host reduction −1.02%
(interval −2.68% to +0.68%), no demonstrated gain. 114 files verified and local
analysis matches remote. No combined test or main-lineage adoption; production
source and manifest unchanged. This round is closed, with no new job submitted.

October 14 user-directed follow-up: retain cache optimization only in the
isolated baseline; do not adopt pinned host reduction. [7392 encoding/wait
diagnosis](reports/training-encode-wait-2026-10-14.md) is submitted to the existing
cluster, with passive arrival traces and a separate synchronized diagnostic.
No new speedup claim and no main-lineage changes.

October 14 evening: [7392 diagnosis](reports/training-encode-wait-2026-10-14.md)
completed on 3 healthy L40S; all 8 ranks finished passive 40 / sync 32 updates,
119 files SHA-verified. Synchronized rollout cache/collation averages 0.997s,
learner public forward 0.250s with 22.41% token-slot padding. Passive reduction
arrival spread averages 16.64ms (p95 48.57ms); this does not establish a causal
rank/GPU imbalance or a speedup. Next target is incremental cache internals;
keep cache optimization, reject pinned reduction. Main identity/manifest unchanged;
no new job, diagnostic monitor paused on closeout.

October 14 follow-up: [7452 cache internals/reuse](reports/training-cache-reuse-2026-10-14.md)
is submitted under user authorization. It profiles append/gather/metadata and
then gates exact append-row-index reuse within one encode call before full PPO
ABBA. Local CPU coverage passed; CUDA parity and speed remain unmeasured.
Main unchanged; rejected pinned reduction stays disabled.

October 14 final [7452 result](reports/training-cache-reuse-2026-10-14.md):
CUDA parity passed; all 8 ranks completed 120 PPO updates on 3 healthy L40S.
Index reuse throughput −0.435% (paired 95% interval −1.301% to +0.369%), no
proven gain despite 50.79% candidate gather hits. Do not adopt this candidate;
retain previous index_select. Profile points to append (ordinary 0.626s,
merged 0.408s inclusive) over final memory gather (0.086s), with nested timing
and synchronization caveats. 113 files verified; local analysis matches remote;
main identity/manifest unchanged. No new job; monitor paused on closeout.

October 14 user-directed [append detail diagnosis](reports/training-append-detail-2026-10-14.md):
7457 submitted; retain index_select, reject row-index reuse and pinned reduction.
Event and synchronized probes split event preparation/KV/attention/FFN. No new
candidate yet. Stop if no clear actionable bottleneck; do not keep profiling.
Main unchanged, existing cluster only.

October 15 closeout: [7457 append detail](reports/training-append-detail-2026-10-14.md)
completed both 8×32 stages on 3 healthy L40S; 103 files verified. Attention is
the largest measured leaf (~0.367s/rank/update in event and sync diagnostics),
but no clear removable/reusable work supports a new optimization. Per user stop
condition, stop exploration: no new candidate/job/profile. Keep prior isolated
index_select only; main identity/manifest unchanged. Monitor paused on closeout.

October 15 main-lineage incident and restart: [gradient blow-up report](reports/main-gradient-blowup-2026-10-15.md).
The main run's actor gradient norm exploded during u74484–77584 (masked efficient
SDPA backward on L40S), degrading the auxiliary heads; it caused the u77585 stop.
Main training resumed October 15 (job 7464) with the math-SDPA fix, index_select
cache gathers, page-span and a per-GPU health probe that skips the ECC-faulty card;
source identity rebuilt with `--allow-source-change`. Measured vs DanLM (4,000 deals):
m8 u81344 −0.376 [−0.414, −0.340], u86800 −0.312 [−0.350, −0.277] (m7 −0.334);
the model recovered, no rollback, and u86800 versus m7 is within error. This
supersedes the "production unchanged / not restarted" statements above for the main run.

October 18: [private-attention audit tooling](reports/private-attention-audit-2026-10-18.md)
implemented (`eval/private_attention_audit.py`, CPU, multi-minibatch) together with
an opt-in `private_qk_norm` architecture arm (default `none`, main lineage
unchanged, fresh lineages only). Measured only on random-init and synthetic
actors: m7/u86800/latest were not on that VM, so whether their saturation is
global remains unresolved. Next: run the audit where the checkpoints are; any
`qk` A/B lineage needs separate authorization.
