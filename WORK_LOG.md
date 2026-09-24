# Overnight candidate support work

Branch: `codex/overnight-candidates` in `wt-candidates`. Scope: C1(b) only.

- Added optional `candidate_mode=union`, `candidate_extra=m` to policy and PPO configs. Defaults retain the existing top-k path and RNG sequence.
- Union draws a uniform subset without replacement from actions outside frozen-reference top-k plus pass. Rollout stores the exact chosen support, action log-probability, and reference distribution on that support; learner evaluates stored candidates only.
- Increased derived buffer capacity for `top_k + candidate_extra + pass`; actor config uses existing dataclass serialization.
- Updated scalar and batched evaluation to draw union support, preserving the small-set reference skip.

Verification:

- Copied the existing Python 3.14 `gd` extension from the main checkout into this worktree's ignored `python/gd/` for tests. No build or rules change.
- `PYTHONPATH=python /Users/xiyaowang/Documents/Projects/GuanZero/.venv/bin/python -m pytest -q tests/test_candidate_union.py tests/test_stage_b_policy.py tests/test_ppo.py tests/test_ppo_actors.py tests/test_batched_evaluation.py tests/test_pruning_audit.py tests/test_ppo_throughput.py`: 62 passed, 1 skipped in 16.25s. Actor tests needed the approved elevated sandbox because the ordinary sandbox blocks `torch_shm_manager` shared memory.
- New tests measure the uniform inclusion frequencies, compare `act`/`choose` RNG states, reproduce stored behavior/reference log-probs, exercise one PPO update and checkpoint load, check zero-extra RNG identity, and pass union candidates through actor shared-memory IPC and both evaluators.

Cloud comparison proposal: clone one B11-final configuration twice with a fixed league pool and equal wall clock. Both use `top_k=32` and `warm_start_policy_override=true`; control has `candidate_mode=top_k,candidate_extra=0`, union has `candidate_mode=union,candidate_extra=8`. Freeze the pool; leave live `league_import_dir` empty. This is a short probe, not G7 acceptance.

Pending: final diff review and commit. No cloud resource or local long training started.

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
