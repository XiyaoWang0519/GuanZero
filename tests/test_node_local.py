import sys

import pytest

from infra import node_local


def test_violations_match_prefix_and_children_only(tmp_path):
    home = tmp_path / "home"
    (home / "env").mkdir(parents=True)
    (tmp_path / "homework").mkdir()
    paths = {"inside": str(home / "env"), "root": str(home),
             "sibling": str(tmp_path / "homework"), "elsewhere": "/dev/shm/env"}
    assert node_local.violations(str(home), paths) == [
        f"inside={home / 'env'}", f"root={home}"]


def test_violations_follow_symlinks(tmp_path):
    (tmp_path / "home" / "env").mkdir(parents=True)
    link = tmp_path / "shm-env"
    link.symlink_to(tmp_path / "home" / "env")
    assert node_local.violations(str(tmp_path / "home"), {"link": str(link)}) == [f"link={link}"]


def test_mapped_files_reads_absolute_paths(tmp_path):
    maps = tmp_path / "maps"
    maps.write_text("7f00-7f01 r-xp 00000000 00:2a 11 /home/u/env/lib/libstdc++.so.6.0.36\n"
                    "7f02-7f03 rw-p 00000000 00:00 0 [heap]\n"
                    "7f04-7f05 r--p 00000000 00:00 0\n")
    assert node_local.mapped_files(str(maps)) == {
        "mapped /home/u/env/lib/libstdc++.so.6.0.36": "/home/u/env/lib/libstdc++.so.6.0.36"}
    assert node_local.violations("/home", node_local.mapped_files(str(maps)))


def test_require_node_local_is_opt_in(monkeypatch):
    monkeypatch.delenv("GZ_REQUIRE_NODE_LOCAL", raising=False)
    monkeypatch.setenv("GZ_FORBID_PREFIX", "/")
    node_local.require_node_local("test")


def test_require_node_local_rejects_forbidden_prefix(monkeypatch):
    monkeypatch.setenv("GZ_REQUIRE_NODE_LOCAL", "1")
    monkeypatch.setenv("GZ_FORBID_PREFIX", sys.prefix)
    with pytest.raises(RuntimeError, match="not node-local"):
        node_local.require_node_local("test")


def test_require_node_local_accepts_other_prefix(monkeypatch, tmp_path):
    monkeypatch.setenv("GZ_REQUIRE_NODE_LOCAL", "1")
    monkeypatch.setenv("GZ_FORBID_PREFIX", str(tmp_path))
    node_local.require_node_local("test")
