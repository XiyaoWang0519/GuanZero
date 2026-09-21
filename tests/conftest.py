"""Make `gd` (the pybind11 package) and the oracle importable from the tests."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "python", ROOT / "oracle", ROOT):
    sys.path.insert(0, str(p))
