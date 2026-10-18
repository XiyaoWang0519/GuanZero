"""Make `gd` (the pybind11 package) and the oracle importable from the tests."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "python", ROOT / "oracle", ROOT):
    sys.path.insert(0, str(p))
# GD_PACKAGE_DIR: a side build of the `gd` package (python/CMakeLists.txt
# GD_PYTHON_OUTPUT_DIR) that takes precedence over python/gd.
if os.environ.get("GD_PACKAGE_DIR"):
    sys.path.insert(0, os.environ["GD_PACKAGE_DIR"])
