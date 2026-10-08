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
  u43276 (-0.665; search gain +0.13 on 800 deals) and m4 u50840 (-0.546). It is a dated summary, not
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
