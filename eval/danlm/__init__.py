"""DanLM external baseline (Stage C task C0, ``docs/STAGE_C_TODO.md``).

DanLM (https://github.com/dashidhy/DanLM) ships public weights and its own
engine as CPython 3.12 **macOS ARM64** Cython extensions
(``*.cpython-312-darwin.so``) under an Apache 2.0 licence with a
non-commercial clause. Nothing from it is vendored here: the adapter imports
it from a checkout on ``sys.path`` (``DANLM_ROOT``) inside a Python 3.12
virtual environment that also carries a ``gd`` extension built for 3.12.

There is no published Linux build and no rebuildable extension source in the
public tree; the author declined a Linux package (DanLM issue #1). Velocity /
Linux cloud therefore cannot run ``eval.danlm.arena``. Use a Mac host for the
paired DanLM protocol, or ``eval.v1t.arena`` for the Linux ONNX sibling
baseline. See ``eval.danlm.platform`` and
``docs/reports/danlm-linux-velocity-2026-10-20.md``.

``bridge`` converts cards, levels and plays between the two engines and is
pure Python, testable anywhere. ``arena`` plays rounds with both engines in
lockstep: DanLM's engine is the referee (its legality, trick flow and finish
order decide the score), ours mirrors every action so that our policy sees
its own observation encoding, and every disagreement between the two is
counted as a divergence class for the ``botzone`` rules profile work.
"""
