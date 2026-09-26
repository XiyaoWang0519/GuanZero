"""Decision gates and artifact integrity, without cloud allocation."""
import json
from pathlib import Path
import tarfile

import pytest

from bench.history_ppo import assess
from infra.history_artifacts import pack_source, sha256, source_identity
from infra.history_monitor import verify_download
from infra.history_pilot import health


def metric(prefix=750, speed=80):
    return dict(update=5, rounds=12, update_samples=100, policy_loss=0.1,
                value_loss=0.2, entropy=1.5, approx_kl=0.01, clip_fraction=0.1,
                actor_grad_norm=0.5, encoder_grad_norm=0.1, critic_grad_norm=0.3,
                population=dict(decisions={"0": 100, "1": 200}),
                mean_prefix=prefix, decisions_per_sec=speed, collect_seconds=2,
                learn_seconds=1, learn_decisions_per_sec=100, cuda_peak_reserved_bytes=1000)


def test_throughput_gate_blocks_slow_memory_and_missing_prefix():
    early, late = metric(144, 100), metric(750, 80)
    assert assess([early, late], 10000)["passed"]
    assert not assess([early], 10000)["passed"]
    late["decisions_per_sec"] = 20
    assert not assess([early, late], 10000)["passed"]
    late["decisions_per_sec"] = 80
    assert not assess([early, late], 1000)["passed"]


def test_health_uses_learning_and_snapshot_metrics():
    line = metric()
    assert health([line])["snapshot_decisions"] == 200
    assert not health([])["healthy"]
    for key, value in [("policy_loss", float("nan")), ("encoder_grad_norm", 0),
                       ("approx_kl", 2), ("value_loss", None)]:
        bad = {**line, key: value}
        assert not health([bad])["healthy"]


def test_source_archive_and_download_hashes(tmp_path):
    archive = tmp_path / "source.tar.gz"
    identity = pack_source(archive)
    assert identity["source_sha256"] == source_identity()["source_sha256"]
    with tarfile.open(archive) as source:
        names = source.getnames()
        assert "train/history_population.py" in names
        assert "infra/history_pilot.py" in names
        assert "source-identity.json" in names
        assert all(not name.endswith((".pt", ".so", ".env")) for name in names)
        assert all(not any(p.startswith(".") for p in Path(name).parts) for name in names)
        recorded = json.load(source.extractfile("source-identity.json"))
        assert recorded["files"] == identity["files"]
    (tmp_path / "result.json").write_text('{"complete":true}')
    records = {"result.json": sha256(tmp_path / "result.json")}
    verify_download(tmp_path, records)
    (tmp_path / "result.json").write_text("truncated")
    with pytest.raises(ValueError, match="hash"):
        verify_download(tmp_path, records)
    with pytest.raises(ValueError, match="path"):
        verify_download(tmp_path, {"../outside": "x"})


def test_frozen_evaluator_hashes_and_full_match_seat_pairs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from eval import history_frozen as frozen
    from eval.duplicate import generate_deals
    from infra.history_artifacts import engine_digest

    candidate, baseline = tmp_path / "candidate.pt", tmp_path / "baseline.pt"
    candidate.write_bytes(b"candidate")
    baseline.write_bytes(b"baseline")
    deal = generate_deals(1, seed=7)[0]
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps(dict(deals=[{k: getattr(deal, k) for k in
                                          ("hands", "level", "team_levels", "owner", "leader")}],
                                   policy_seed=10, match_seeds=[91, 92])))
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(dict(engine_digest=engine_digest(),
        development=dict(file=dev.name, sha256=sha256(dev)),
        baselines=[dict(name="baseline", path=str(baseline), sha256=sha256(baseline))])))
    monkeypatch.setattr(frozen, "load_policy", lambda *a, **k: SimpleNamespace(stage="history_ppo"))
    monkeypatch.setattr(frozen, "evaluate_duplicates", lambda *a, **k: {"raw_legs": "retained"})
    calls = []
    def matches(agent, opponent, indices, seed):
        calls.append((indices, seed))
        return dict(wins=1, records=[dict(seed=seed + indices[0], team=indices[0] % 2)])
    monkeypatch.setattr(frozen, "play_matches", matches)
    report = frozen.evaluate(freeze, candidate, tmp_path / "report.json")
    assert calls == [([0], 91), ([1], 90), ([0], 92), ([1], 91)]
    assert len(report["reports"]["baseline"]["full_matches"]["pairs"]) == 2
    assert report["split"] == "development"
    baseline.write_bytes(b"changed")
    with pytest.raises(ValueError, match="baseline changed"):
        frozen.evaluate(freeze, candidate, tmp_path / "bad.json")
    dev.write_text("changed")
    with pytest.raises(ValueError, match="deal file"):
        frozen.evaluate(freeze, candidate, tmp_path / "bad.json")
