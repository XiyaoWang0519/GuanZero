# GuanZero agent instructions

Shared instructions for Codex and Claude Code. Codex reads this file directly;
Claude Code imports it from `CLAUDE.md`. Maintain project instructions here,
not in two independent copies. Updated October 6, 2026.

## Start here

1. Read [docs/STATUS.md](docs/STATUS.md) for dated status, unresolved questions
   and the applicable research boundary.
2. Use [docs/README.md](docs/README.md) to select only the documents relevant
   to the task. Use [docs/reports/README.md](docs/reports/README.md) for evidence.
3. Inspect the owning code, tests and `git status --short` before changing
   behavior. Reports establish what was measured at that time; they do not
   prove current implementation or a running job's state.

## Current model route and authority

The baseline is a history Transformer trained from random initialization by
self-play RL. Existing MLP players are evaluation-only: no weight/critic
transfer, imitation, old replay, MLP training opponents/partners,
frozen-reference filtering or MLP-reference KL. Training populations use
current/past Transformer versions; tribute-only heuristics are declared.
Public actor inputs and caches must not contain opponent-private information.

[DESIGN.md](docs/DESIGN.md) defines the baseline research boundary;
[TRAINING.md](docs/TRAINING.md) is the operational guide.
[STAGE_C_TODO.md](docs/STAGE_C_TODO.md) retains T0--T8 milestones and dated
receipts, rather than a complete list of today's tasks. Dated proposals are
not authorization to launch experiments, rent compute or change the boundary.
Scoped user-approved exceptions documented in reports remain scoped; do not
apply them to the main lineage. Follow the user's current instructions and
existing session authorization; do not request approval again for authorized work.

## Keeping documentation usable

- Update the owning guide when behavior changes; link measured evidence
  instead of copying experiment narratives into several guides.
- Update `docs/STATUS.md` when the documented direction or readiness changes.
  Date status claims and distinguish implemented, measured, proposed and unresolved.
- New experiment records go in `docs/reports/<topic>-YYYY-MM-DD.md`; add them
  to `docs/reports/README.md`. Record scope, configuration/source identity,
  results, limitations and artifact pointers. Label plans explicitly.
- Preserve negative results and historical receipts. Correct a historical
  result with an explicit dated correction, not a silent rewrite.
- `.work/` is ignored local evidence, not a portable dependency. Check that
  referenced artifacts exist before relying on them. Do not commit secrets,
  credentials or access-bearing URLs with reports.
- Preserve unrelated dirty files and in-progress work. Documentation edits
  alone do not establish runtime readiness, GPU speed or playing strength.

## Repository conventions

Read `docs/RULES.md` before touching anything in `cpp/`. It is the acceptance
standard: if the engine and that document disagree, one of them has a bug.
`docs/DESIGN.md` holds the system design and the milestone list.

## Layout

| Path | What |
|---|---|
| `cpp/include/gd/`, `cpp/src/` | `gd_core`, the C++20 rules engine |
| `cpp/tests/` | C++ unit tests, harness in `cpp/tests/test_util.h` |
| `python/gd/` | the pybind11 package, module `gd._gd_core` |
| `oracle/` | `gd_reference.py`, the independent Python rules oracle. Test only. |
| `tests/` | pytest and hypothesis suites, including the oracle cross-checks |
| `bench/` | throughput benchmarks |
| `eval/ogd_adapter/` | OpenGuanDan client, trace logger and replay diff |
| `docs/reports/` | dated experiment evidence and plans; indexed in its README |

## Build and test

```sh
.venv/bin/python -m pip install -r requirements-dev.txt -r requirements-train.txt
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/cpp/tests/gd_tests                 # C++ unit tests
.venv/bin/python -m pytest -q tests oracle # Python suites and oracle
./scripts/check.sh                         # everything CI runs
./scripts/preflight.sh cpu .work/preflight # legacy MLP update/resume check only
```

Current training readiness, independent checkpoint evaluation and GPU lifecycle
requirements are in `docs/TRAINING.md`. M0/M1/M2 and Stage B trackers are
historical evidence; use `docs/STATUS.md` to orient current work.

Sanitizer build. `-DGD_SANITIZE=ON` is address plus undefined; the variable
also takes an explicit `-fsanitize` list.

```sh
cmake -S . -B build-ubsan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGD_SANITIZE=undefined
cmake --build build-ubsan -j && ./build-ubsan/cpp/tests/gd_tests
UBSAN_OPTIONS=halt_on_error=1 ./build-ubsan/cpp/fuzz/gd_fuzz --rounds 100000 --threads 6
```

On macOS use `undefined` alone. AddressSanitizer deadlocks inside its own
runtime initialisation on this host, spinning in `AsanInitInternal` while dyld
shared-cache iteration re-enters malloc, before `main` runs. It is an
interceptor problem in the toolchain, not in this code. CI runs address and
undefined together on Linux, which is where ASan coverage comes from.

## Rules of the house

1. The oracle in `oracle/gd_reference.py` was written independently of the
   engine and must stay that way. Never change the oracle to make the engine
   pass. If the oracle is wrong, fix it against `docs/RULES.md` and say so.
2. Every rule decision lives behind a field of `RuleConfig` or `ActionConfig`
   (`cpp/include/gd/config.h`), never as a hardcoded constant. The `house`
   profile is the default; the `ogd` profile exists for parity testing.
3. `RoundState` and `MatchState` must stay trivially copyable: fixed-size
   arrays, no heap, no virtuals. Search clones them.
4. Nothing allocates in the rollout hot loop. Move generation appends into a
   caller-owned `std::vector<Action>` that is reused across decisions.
5. Power order and sequence order are different orderings (RULES.md 4). Mixing
   them up is the classic engine bug; name variables `power` or `window`
   accordingly.
6. New rule behaviour needs a test vector in RULES.md section 12 or a property
   in section 14 before it is implemented.

## Style

C++20, `snake_case` for functions and variables, `CamelCase` for types,
trailing `_` on private members, two-space indent, no exceptions in the hot
path. Python: 4 spaces, type hints on public functions, no dependencies in
`oracle/` beyond the standard library.
