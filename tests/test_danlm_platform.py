"""DanLM platform probe: Linux must fail closed without Darwin extensions."""
from __future__ import annotations

from pathlib import Path

import pytest

from eval.danlm import platform as plat


def test_extension_platform_tag_parses_darwin():
    path = Path("danzero/engine/game.cpython-312-darwin.so")
    assert plat.extension_platform_tag(path) == "darwin"
    assert plat.extension_platform_tag(Path("game.py")) is None


def test_platform_report_missing_root(tmp_path, monkeypatch):
    monkeypatch.delenv("DANLM_ROOT", raising=False)
    report = plat.platform_report(tmp_path / "no-danlm")
    assert report["root_exists"] is False
    assert report["usable_here"] is False
    assert report["missing_extensions"] == list(plat.REQUIRED_EXTENSION_STEMS)
    assert "darwin" in report["blocker"].lower() or "Linux" in report["blocker"]


def test_platform_report_darwin_only_tree(tmp_path, monkeypatch):
    monkeypatch.delenv("DANLM_ROOT", raising=False)
    root = tmp_path / "DanLM"
    for stem in plat.REQUIRED_EXTENSION_STEMS:
        path = root / f"{stem}.cpython-312-darwin.so"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    report = plat.platform_report(root)
    assert report["extension_tags"] == ["darwin"]
    assert report["missing_extensions"] == []
    # On this Linux CI/cloud host the tree is still not usable.
    if report["host_platform"].startswith("darwin"):
        assert report["usable_here"] is True
    else:
        assert report["usable_here"] is False
        with pytest.raises(RuntimeError, match="darwin|Linux|DanLM"):
            plat.require_danlm_platform(root)


def test_require_fails_without_checkout(tmp_path, monkeypatch):
    monkeypatch.delenv("DANLM_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="DanLM"):
        plat.require_danlm_platform(tmp_path / "missing")
