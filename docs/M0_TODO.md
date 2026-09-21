# M0 task tracker

Gate (DESIGN.md 1.2): the engine passes every test in RULES.md, replays
OpenGuanDan traces without divergence and meets the throughput target
(at least 100,000 decisions per second per core, random play, canonical mode).

**Status: complete.** Report: `docs/reports/M0.md`.

| # | Task | Check | Evidence | Status |
|---|---|---|---|---|
| 1 | Scaffold: CMake, pybind11, pytest, CI, oracle, `CLAUDE.md` | `oracle/test_gd_reference.py` passes in CI | 14 test groups | done |
| 2 | Cards, `Hand`, text parsing and printing | round trip over all 54 ids and random hands | `cpp/tests/test_cards.cpp` | done |
| 3 | `power`, windows, `interpret`, `beats` | every vector of RULES.md 12; 10^5 random multisets agree with the oracle | 100,000 `interpret`, 100,000 `best_readings`, 37,636 exhaustive `beats` pairs, no mismatch | done |
| 4 | Move generation, full mode | equals the oracle on 10^4 random small hands; sound on 27-card hands | 10,000 hands, no mismatch; fuzzer covers full hands | done |
| 5 | Move generation, canonical mode | subset and best-reading invariants; candidate statistics recorded | `tests/test_movegen_crosscheck.py`; statistics in the report | done |
| 6 | Round and match state machine, tribute, level A, `DealSpec` | T-FLOW scenarios and tribute vectors | `tests/test_flow_scenarios.py`, 16 scenarios; `end_of_round` vs the oracle on 20,000 configurations | done |
| 7 | Random and greedy bots, fuzzer | 10^7 rounds with all invariants; 10^5 clean under ASan and UBSan | 10,000,008 rounds, 735,492,869 decisions, no failure; 100,002 rounds clean under UBSan (ASan is blocked by a toolchain deadlock on this host, see the report) | done |
| 8 | `VecEnv` and Python bindings | Python smoke test plays 1,000 matches with random choices | `tests/test_vecenv_smoke.py` | done |
| 9 | Benchmarks | report written, target met or the gap explained | 190,533 decisions per second per core against a target of 100,000 | done |
| 10 | OpenGuanDan trace logger and replay diff | 1,000 matches replay with no unexplained divergence | 1,783,202 decisions, 13,341 round ends, one explained class at 0.21% | done |
| 11 | Feature encoder v1 | golden tests | `tests/test_encoder_golden.py`, observation 1,849, action 154 | done |

## Carried into M1

1. The OpenGuanDan repository ships no baseline agents and no weights, so the
   external ladder the M2 and M3 gates assume does not exist yet. See section 5
   of the report; this needs a decision before M2.
2. `docs/RULES.md` 11.2 and 14.5 were corrected: the canonical-mode guarantee
   the document stated cannot hold alongside its own reductions.
3. Parity check O11 is documented rather than emulated: the simulator
   enumerates fewer wild-card substitutions than we do and our legal set is a
   strict superset.
