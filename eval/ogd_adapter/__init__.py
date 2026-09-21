"""Adapter for the OpenGuanDan reference simulator (M0 task 10).

This package never vendors code or data from the OpenGuanDan clone: it
locates the clone at runtime through the ``OGD_ROOT`` environment variable
(falling back to the path recorded in `bridge.py`) and imports its Python
bridge (`guandan-java/engine`) off `sys.path`. See docs/RULES.md section 1
and docs/DESIGN.md section 9.2 for why this adapter exists.
"""

from eval.ogd_adapter.bridge import legal_moves, make_env, ogd_root, shutdown

__all__ = ["legal_moves", "make_env", "ogd_root", "shutdown"]
