"""Portable source identities and an explicit, secret-free pilot source archive."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("train", "eval", "bench", "infra", "cpp", "python", "tests", "oracle", "scripts")
SOURCE_SUFFIXES = {".py", ".cpp", ".h", ".sh"}
TOP_FILES = ("CMakeLists.txt", "requirements-dev.txt", "requirements-train.txt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(root: Path = ROOT) -> list[Path]:
    paths = [root / name for name in TOP_FILES]
    for directory in SOURCE_DIRS:
        for path in (root / directory).rglob("*"):
            if (path.is_file() and not path.is_symlink()
                    and (path.suffix in SOURCE_SUFFIXES or path.name == "CMakeLists.txt")
                    and not any(p.startswith(".") or p == "__pycache__"
                                for p in path.relative_to(root).parts)):
                paths.append(path)
    return sorted(paths)


def source_identity(root: Path = ROOT) -> dict:
    files = {str(p.relative_to(root)): sha256(p) for p in source_files(root)}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    receipt = root / "source-identity.json"
    if receipt.exists():
        recorded = json.loads(receipt.read_text())
        if recorded["files"] != files:
            raise ValueError("source archive changed after its manifest was frozen")
        return recorded
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                          text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--",
                            *SOURCE_DIRS, *TOP_FILES], cwd=root, capture_output=True,
                           text=True, check=True).stdout
    return dict(revision=head, dirty=bool(dirty), source_sha256=digest,
                dirty_source_sha256=digest if dirty else None, files=files)


# Evaluation-only engine code: the determinization samplers of test-time search.
# Neither the trainer nor VecEnv calls them, so they are outside the engine
# digest (October 3, 2026); so is python/bindings.cpp, which only exposes the
# library. Rule, state, move-generation, encoding, bot and config sources stay in.
ENGINE_EXCLUDED = ("cpp/include/gd/search.h", "cpp/src/search.cpp")


def _is_engine_file(relative: str) -> bool:
    return ((relative.startswith("cpp/include/gd/") and relative.endswith(".h"))
            or (relative.startswith("cpp/src/") and relative.endswith(".cpp"))) \
        and relative not in ENGINE_EXCLUDED


def engine_files(root: Path = ROOT) -> list[Path]:
    paths = [*root.glob("cpp/include/gd/*.h"), *root.glob("cpp/src/*.cpp")]
    return sorted(p for p in paths if _is_engine_file(str(p.relative_to(root))))


def engine_digest_from_hashes(hashes: dict[str, str]) -> str:
    """The engine digest of a tree described by ``{relative path: sha256}``
    (``source_identity()["files"]`` of any checkpoint since the first run)."""
    digest = hashlib.sha256()
    for relative in sorted(name for name in hashes if _is_engine_file(name)):
        digest.update(relative.encode() + b"\0" + hashes[relative].encode() + b"\n")
    return digest.hexdigest()


def engine_digest(root: Path = ROOT) -> str:
    """Digest of the game-dynamics sources (version 2, October 3 2026)."""
    return engine_digest_from_hashes({str(p.relative_to(root)): sha256(p) for p in engine_files(root)})


def legacy_engine_digest(root: Path = ROOT) -> str:
    """The digest every checkpoint and freeze before October 3, 2026 recorded:
    raw bytes of all cpp headers, cpp sources and python/bindings.cpp."""
    digest = hashlib.sha256()
    for path in sorted([*root.glob("cpp/include/gd/*.h"), *root.glob("cpp/src/*.cpp"),
                        root / "python/bindings.cpp"]):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def engine_compatible(saved_run_identity: dict, root: Path = ROOT) -> bool:
    """Whether a checkpoint's recorded engine is this tree's engine: the same
    digest, or (checkpoints written before version 2) the same version-2 digest
    recomputed from the per-file source hashes the checkpoint recorded."""
    current = engine_digest(root)
    if saved_run_identity.get("engine_digest") == current:
        return True
    files = (saved_run_identity.get("source") or {}).get("files") or {}
    return bool(files) and engine_digest_from_hashes(files) == current


def pack_source(destination: Path, root: Path = ROOT) -> dict:
    identity = source_identity(root)
    with tarfile.open(destination, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for path in source_files(root):
            # Use Python tarfile: no macOS AppleDouble/xattr entries or symlinks.
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
        data = (json.dumps(identity, indent=2) + "\n").encode()
        item = tarfile.TarInfo("source-identity.json")
        item.size, item.mode = len(data), 0o644
        archive.addfile(item, io.BytesIO(data))
    return dict(**identity, archive_sha256=sha256(destination))
