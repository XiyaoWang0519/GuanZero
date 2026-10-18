# Documentation map for agents

Start with [../AGENTS.md](../AGENTS.md) (shared Codex / Claude Code rules),
then [STATUS.md](STATUS.md). Read only the owning documents for your task.
Updated October 6, 2026.

## Document roles

| Document | Role / when to read |
|---|---|
| [STATUS.md](STATUS.md) | Dated orientation, current directions and unresolved boundaries |
| [DESIGN.md](DESIGN.md) | Baseline architecture, information boundaries and research contract |
| [RULES.md](RULES.md) | Engine acceptance standard; read before C++ rule changes |
| [STATE_SPEC.md](STATE_SPEC.md), [MOVEGEN_SPEC.md](MOVEGEN_SPEC.md), [PY_API.md](PY_API.md) | State, action generation and Python interface contracts |
| [TRAINING.md](TRAINING.md) | Entrypoints, local checks, run manifest, resume, lifecycle and evaluation |
| [COMPUTE_GUIDE.md](COMPUTE_GUIDE.md) | Workload measurement and compute selection principles; verify live quotes |
| [PERF_TODO.md](PERF_TODO.md) | Performance follow-up and links to implemented/measured work |
| [STAGE_C_TODO.md](STAGE_C_TODO.md) | T0--T8 milestone receipts; read dates and limits, not just task labels |
| [TRAINING_EVIDENCE.md](TRAINING_EVIDENCE.md) | Historical training/implementation chronology extracted from the operator guide |
| [reports/README.md](reports/README.md) | Complete topic index of dated evidence and proposals |
| [../WORK_LOG.md](../WORK_LOG.md) | Historical development log; ends September 26, not current readiness |

## Historical records

[M0](M0_TODO.md), [M1](M1_TODO.md), [M2](M2_TODO.md) and
[Stage B](STAGE_B_TODO.md) retain old acceptance, implementation and MLP
results. Their `TODO` filenames remain for compatibility; they are not the
active queue. [OpenGuanDan parity probes](ogd_parity_probes.md) and
[profile findings](ogd_profile_findings.md) are integration evidence.

## How to resolve a disagreement

The user's current task and authorized scope guide the work. Shared agent
rules and the owning specification define the baseline contract. Actual code
and tests establish implementation; dated reports establish measured results.
A newer proposal does not supersede the contract merely by being newer. Check
a report's explicit approval/scope before using an exception. If the docs and
code disagree, investigate and record the discrepancy rather than silently
selecting a convenient interpretation.

Reports may point into ignored `.work/` directories available only on the
original checkout. Their absence on another machine does not invalidate the
historical report, but prevents independent artifact verification there.
