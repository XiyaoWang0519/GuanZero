"""Platform requirements for importing DanLM's compiled ``danzero`` package.

DanLM (https://github.com/dashidhy/DanLM, tree ``16ace5ad``) ships the engine,
encoder, model and eval agents as CPython 3.12 Cython extensions tagged
``cpython-312-darwin`` (Mach-O ARM64). There is no published Linux ``.so``, no
extension source in the public tree, and the author closed a Linux-build
request with no plan to ship one
(https://github.com/dashidhy/DanLM/issues/1#issuecomment-4519480353).

Weights (``ckpts/DanLM_v1/dansformer_v1_best_eval.pt``) are ordinary PyTorch
state dicts, but loading them still needs ``danzero.model.transformer`` and the
engine/agents modules. ``eval.v1t.arena`` is the Linux-capable sibling baseline
(ONNX + our engine as referee); it is not a drop-in for the DanLM lockstep
protocol in ``eval.danlm.arena``.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

# Modules ``arena.import_danzero`` and ``DanLMSeats`` need at import/load time.
REQUIRED_EXTENSION_STEMS = (
    "danzero/engine/cards",
    "danzero/engine/game",
    "danzero/engine/tribute",
    "danzero/engine/actions",
    "danzero/eval/agents",
    "danzero/model/transformer",
)

SUPPORTED_PLATFORM_TAGS = ("darwin",)


def danlm_root(root: Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    env = os.environ.get("DANLM_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / ".work/external/DanLM"


def extension_platform_tag(path: Path) -> str | None:
    """Return the ABI platform tag in ``name.cpython-312-<tag>.so``, or None."""
    name = path.name
    if not name.endswith(".so"):
        return None
    # e.g. game.cpython-312-darwin.so -> darwin
    parts = name[:-3].split(".")
    if len(parts) < 2 or not parts[-1].startswith("cpython-"):
        return None
    abi = parts[-1]  # cpython-312-darwin
    bits = abi.split("-", 2)
    if len(bits) < 3:
        return None
    return bits[2]


def discover_extensions(root: Path | None = None) -> dict[str, list[Path]]:
    """Map relative stem (``danzero/engine/game``) -> matching ``.so`` paths."""
    base = danlm_root(root)
    found: dict[str, list[Path]] = {stem: [] for stem in REQUIRED_EXTENSION_STEMS}
    if not base.is_dir():
        return found
    for stem in REQUIRED_EXTENSION_STEMS:
        parent = base / Path(stem).parent
        prefix = Path(stem).name + ".cpython-"
        if not parent.is_dir():
            continue
        for path in sorted(parent.glob(f"{Path(stem).name}.cpython-*.so")):
            if path.name.startswith(prefix):
                found[stem].append(path)
    return found


def platform_report(root: Path | None = None) -> dict:
    """Structured inventory for docs/CI; does not import DanLM."""
    base = danlm_root(root)
    extensions = discover_extensions(base)
    tags: set[str] = set()
    missing = []
    for stem, paths in extensions.items():
        if not paths:
            missing.append(stem)
            continue
        for path in paths:
            tag = extension_platform_tag(path)
            if tag:
                tags.add(tag)
    host = sys.platform
    usable = bool(tags & set(SUPPORTED_PLATFORM_TAGS)) and host.startswith("darwin") and not missing
    return {
        "danlm_root": str(base),
        "root_exists": base.is_dir(),
        "host_platform": host,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "extension_tags": sorted(tags),
        "supported_tags": list(SUPPORTED_PLATFORM_TAGS),
        "missing_extensions": missing,
        "usable_here": usable,
        "blocker": None if usable else (
            "DanLM ships only cpython-312-darwin Cython extensions; no Linux "
            ".so or rebuildable source in the public tree. Author declined a "
            "Linux build (dashidhy/DanLM#1). Use a macOS ARM64 host, or keep "
            "standing Mac DanLM numbers; Linux-side ONNX baseline is eval.v1t."
        ),
    }


def require_danlm_platform(root: Path | None = None) -> Path:
    """Raise ``RuntimeError`` with a dated blocker message when DanLM cannot load."""
    report = platform_report(root)
    if report["usable_here"]:
        return Path(report["danlm_root"])
    detail = (
        f"host={report['host_platform']} python={report['python']} "
        f"tags={report['extension_tags'] or 'none'} "
        f"missing={report['missing_extensions'] or 'none'} "
        f"root={report['danlm_root']}"
    )
    raise RuntimeError(f"{report['blocker']} ({detail})")


def main() -> None:
    import json

    print(json.dumps(platform_report(), indent=2))
    raise SystemExit(0 if platform_report()["usable_here"] else 1)


if __name__ == "__main__":
    main()
