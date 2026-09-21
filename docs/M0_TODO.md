# M0 task tracker

Gate (DESIGN.md 1.2): the engine passes every test in RULES.md, replays
OpenGuanDan traces without divergence and meets the throughput target
(at least 100,000 decisions per second per core, random play, canonical mode).

| # | Task | Check | Status |
|---|---|---|---|
| 1 | Scaffold: CMake, pybind11, pytest, CI, oracle, `CLAUDE.md` | `oracle/test_gd_reference.py` passes in CI | done |
| 2 | Cards, `Hand`, text parsing and printing | round trip over all 54 ids and random hands | done |
| 3 | `power`, windows, `interpret`, `beats` | every vector of RULES.md 12; 10^5 random multisets agree with the oracle | done |
| 4 | Move generation, full mode | equals the oracle on 10^4 random small hands; sound on 27-card hands | done |
| 5 | Move generation, canonical mode | subset and best-reading invariants; candidate statistics recorded | done |
| 6 | Round and match state machine, tribute, level A, `DealSpec` | T-FLOW scenarios and tribute vectors | done |
| 7 | Random and greedy bots, fuzzer | 10^7 rounds with all invariants; 10^5 clean under ASan and UBSan | done |
| 8 | `VecEnv` and Python bindings | Python smoke test plays 1,000 matches with random choices | done |
| 9 | Benchmarks | report written, target met or the gap explained | done |
| 10 | OpenGuanDan trace logger and replay diff | 1,000 matches replay with no unexplained divergence | todo |
| 11 | Feature encoder v1 | golden tests | done |

Report: `docs/reports/M0.md`.
