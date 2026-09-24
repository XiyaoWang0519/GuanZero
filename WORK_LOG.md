# Search work log — September 24, 2026

- Scope: isolated branch `codex/overnight-search` from `4940246`, worktree
  `.work/overnight-20260924/wt-search`; no edits in the main checkout.
- Read `CLAUDE.md` and `docs/RULES.md` before C++ changes. No rule changes.
- Added the C++ information-set sampler and Python binding, then a bounded
  scalar search policy with `search:<checkpoint>` loading. Existing batched
  evaluator remains unchanged.
- Built only this worktree's `build-search` with 3 jobs and used the shared
  `.venv` interpreter. Ran no training, GPU job, cloud call, or paid inference.
- Passed 81 C++ cases and 47 targeted Python tests. Scalar arena two-deal
  smoke completed. Small 64-deal B11 comparison and measured latency are in
  `docs/reports/stage-c-search-v0.md` and adjacent JSON files.
- Next work, if this arm is continued: adapt the batched evaluator or retain
  scalar, add calibrated marginal/learned beliefs, measure the required three
  budgets, and run G9's larger paired evaluation. The current interval
  includes zero; no strength claim is supported.

## Independent audit and external transfer — later September 24

- Independently recomputed all 1,000 development and 4,000 held-out internal
  pair scores from their raw legs; the held-out mean and whole-deal bootstrap
  interval match the supplied report exactly.
- Built a CPython 3.12 `gd` module in this worktree's `build-cp312`, without
  overwriting the main checkout. DanLM four-deal smoke runs for B11 and
  search:B11 each had zero mirror failures.
- Ran a new-seed, 500-pair transfer against DanLM's referee using one worker
  and one CPU thread. Raw records and paired summary are under
  `docs/reports/search-danlm-transfer-500.*`; interpretation and limits
  are in `docs/reports/stage-c-search-transfer-2026-09-24.md`.
- No model training, paid inference, GPU, or cloud job was started.
