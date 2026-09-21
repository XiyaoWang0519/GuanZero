# Guandan AI: build, test and style

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
| `docs/reports/` | one short report per milestone |

## Build and test

```sh
.venv/bin/python -m pip install -r requirements-dev.txt   # once
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/cpp/tests/gd_tests                 # C++ unit tests
.venv/bin/python -m pytest -q tests oracle # Python suites and oracle
./scripts/check.sh                         # everything CI runs
```

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
