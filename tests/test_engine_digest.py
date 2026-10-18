"""Engine digest v2: dynamics sources only; old checkpoints stay compatible through their file hashes."""
from pathlib import Path

from infra.history_artifacts import (ENGINE_EXCLUDED, engine_compatible, engine_digest,
                                     engine_digest_from_hashes, legacy_engine_digest,
                                     source_identity)
from infra.history_artifacts import ROOT


def fake_tree(root: Path) -> None:
    for rel, text in (("cpp/include/gd/rules.h", "rules"), ("cpp/include/gd/search.h", "search"),
                      ("cpp/src/rules.cpp", "rules impl"), ("cpp/src/search.cpp", "search impl"),
                      ("python/bindings.cpp", "bindings")):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)


def test_search_and_bindings_are_outside_the_engine_digest(tmp_path):
    fake_tree(tmp_path)
    before, legacy_before = engine_digest(tmp_path), legacy_engine_digest(tmp_path)
    for rel in (*ENGINE_EXCLUDED, "python/bindings.cpp"):
        (tmp_path / rel).write_text("changed " + rel)
    assert engine_digest(tmp_path) == before
    assert legacy_engine_digest(tmp_path) != legacy_before
    (tmp_path / "cpp/src/rules.cpp").write_text("a rule changed")
    assert engine_digest(tmp_path) != before


def test_digest_from_recorded_hashes_matches_and_migrates_old_checkpoints():
    identity = source_identity(ROOT)
    assert engine_digest_from_hashes(identity["files"]) == engine_digest(ROOT)
    old_style = dict(engine_digest="legacy-value-of-an-older-tree", source=identity)
    assert engine_compatible(old_style)
    assert engine_compatible(dict(engine_digest=engine_digest(ROOT)))
    assert not engine_compatible(dict(engine_digest="other-engine"))
    changed = dict(identity, files=dict(identity["files"], **{"cpp/src/rules.cpp": "0" * 64}))
    assert not engine_compatible(dict(engine_digest="x", source=changed))
