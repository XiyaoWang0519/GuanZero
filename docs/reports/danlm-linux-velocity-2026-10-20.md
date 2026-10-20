# DanLM on Linux / Velocity: feasibility (October 20, 2026)

Status: **blocked for a native Linux path.** Investigation only; no research-boundary
change; no new DanLM paired numbers. Sibling baseline inventory:
[eval-baseline-avg4-2026-10-20.md](eval-baseline-avg4-2026-10-20.md) (PR #6)
still lists fresh DanLM 4k as Mac-only.

## Verdict

| Question | Answer |
|---|---|
| Can `eval.danlm.arena` run on Velocity (Linux) today? | **No** |
| Can we compile DanLM from the public tree on Linux? | **No** — no extension source |
| Smallest faithful Linux adapter without huge rebuild risk? | **None local** — keep Mac referee, or wait for author binaries, or build a remote-Mac service |
| What the user must provide | A macOS ARM64 host (or always-on Mac eval service), **or** Linux `cpython-312-*-linux-gnu.so` builds from the author (author has declined) |

## How GuanZero loads DanLM

Nothing from DanLM is vendored. `DANLM_ROOT` (default `.work/external/DanLM`)
is prepended to `sys.path`; `eval.danlm.arena.import_danzero` then imports:

| Import | Role in lockstep |
|---|---|
| `danzero.engine.cards` / `game` / `tribute` / `actions` | DanLM referee (legality, trick flow, finish order, rewards) |
| `danzero.eval.agents` | `EvalAgent` / `TransformerAgent` for DanLM seats |
| `danzero.model.transformer` | `TransformerConfig` + `TransformerQNetwork` for `.pt` weights |
| `eval.danlm.fast_obs` | Pure-Python drop-in for slow compiled `get_observation` |
| `eval.danlm.bridge` | Pure Python card/level/play conversion (runs anywhere) |

Loading is ordinary CPython extension import (Cython `.so`), not ctypes, JNI,
subprocess, or Botzone packaging. Our `gd` engine mirrors every action so the
policy sees GuanZero observations; DanLM remains the scoring referee.

## Upstream inventory (tree `16ace5ada4db39086bb4c8ebb372ed16cb82fab1`)

Cloned fresh on the cloud agent host, 2026-10-20:

- **12** modules as `*.cpython-312-darwin.so` (Mach-O 64-bit arm64 bundles).
- Empty `danzero/{engine,model,eval}/__init__.py`; only `encoding/__init__.py`
  and a few scripts/UI files are readable Python.
- No `setup.py` / `pyproject.toml` / Cython `.pyx` / C++ sources for those
  extensions. No GitHub releases, no Linux tags/branches, no manylinux wheels.
- README **Requirements**: Python 3.12, **macOS ARM64 (Apple Silicon)**.
- License: Apache 2.0 **plus non-commercial restriction** (academic/eval OK).
- Public weights: `ckpts/DanLM_v1/dansformer_v1_best_eval.pt` (~45 MB PyTorch);
  config and 53 state-dict tensors load on Linux with torch alone, but the
  architecture class and tokenizer/encoder live in Darwin `.so` files.
- Author reply on [issue #1](https://github.com/dashidhy/DanLM/issues/1)
  (“有没有linux环境的版本”): *no plan to maintain / will not provide a Linux
  version*; open to questions if someone reproduces the work.

Import on Linux fails before any game logic:

```text
ImportError: cannot import name 'game' from 'danzero.engine'
```

(`__init__.py` is empty; the Darwin `.so` is not a loadable ELF module.)

## Velocity probe

`ssh velocity` (`x2524wan@max-velocity.uwaterloo.ca`) succeeded. Login node is
Linux x86_64. Under `~/guanzero/` there are GuanZero trees, eval kits, and the
V1T ONNX (`evalkit/v3_rep_v1t_best_eval_001_int8.onnx`), but **no DanLM
checkout with loadable Linux extensions**. CPU inference preference is moot:
the referee and agent stack never import.

## Why compile / reimplement is out of scope

1. **No source** for engine, tokenizer, encoder, transformer, or agents —
   Cython binaries only. Rebuilding needs upstream source or a large reverse
   engineering / reimplementation project.
2. **Protocol fidelity**: standing DanLM numbers use DanLM’s engine as referee
   (`eval.danlm.arena`). Switching to `gd` as referee (as `eval.v1t.arena`
   does for ONNX V1T) would produce a **different** baseline, not a drop-in
   Linux port of the Mac paired protocol.
3. **Weights alone are insufficient**: even a pure-Python Transformer rebuild
   from the `.pt` config would still need DanLM’s tokenizer, observation
   encoding, legal-play vectors, and tribute agent logic from the missing
   modules.
4. Author will not ship Linux `.so`s; requesting again is optional, not a plan.

## Options (ordered by fidelity / effort)

| Option | Fidelity to Mac DanLM protocol | Effort / risk | Notes |
|---|---|---|---|
| **A. Keep Mac for DanLM 4k** | Exact | Zero code | Current standing numbers ([paired-eval-mac-2026-10-18.md](paired-eval-mac-2026-10-18.md)) |
| **B. Remote Mac eval service** | Exact if Mac runs full `arena` or referee+agents | Medium ops | Velocity/cloud orchestrates deals/checkpoints; Mac returns leg records. Needs an always-reachable Mac + SSH/RPC. Not implemented here. |
| **C. Author Linux wheels** | Exact | External dependency | Author declined (issue #1). |
| **D. Reimplement DanLM stack** | Uncertain until calibrated | Huge | Against “no huge upstream rebuild risk.” |
| **E. Use `eval.v1t` on Linux** | Different opponent | Already done | ONNX V1T, `gd` referee; fine for Linux ladders, **not** a DanLM substitute. |

**Smallest adapter that preserves the paired protocol:** none that runs
natively on Velocity. The practical adapter is **B** (remote Mac) or continuing
**A**. Code landed in this pass only fails closed with an explicit probe
(`eval.danlm.platform`, hooked from `import_danzero`) so Linux jobs stop with a
documented blocker instead of a blank `ImportError`.

## Code / docs in this change

- `eval/danlm/platform.py` — inventory + `require_danlm_platform()`
- `eval/danlm/arena.py` — call the probe before importing `danzero`
- `eval/danlm/__init__.py` — document Darwin-only constraint
- `tests/test_danlm_platform.py` — probe unit tests

Reproduce the probe:

```sh
DANLM_ROOT=/path/to/DanLM .venv/bin/python -m eval.danlm.platform
# exit 1 on Linux / missing Darwin extensions
```

## What the user must provide (if anything)

- **To unblock Velocity DanLM 4k with the same protocol:** either (1) run the
  existing Mac harness and sync JSON results, or (2) expose a Mac that can
  import DanLM and accept a remote-arena RPC (not built yet), or (3) obtain
  Linux Cython builds from the author despite the public decline.
- **No secrets or weights needed from the user for this investigation** —
  public clone + Velocity SSH key already available to the cloud agent.

Research boundary unchanged: DanLM remains an evaluation-only external
baseline; MLP/DanLM are not training opponents.
