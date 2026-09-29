"""bench.history_arms: in-place arm switching, schedule and argument checks."""
import json

import pytest

from bench import history_arms
from train.history_ppo import HistoryPPOConfig, HistoryTrainer


def test_schedule_alternates_after_warmup():
    plan = history_arms.schedule(["a", "b"], 2, 3, 2)
    assert plan == ["a", "a", "a", "a", "b", "b", "b", "b", "a", "a", "a", "a", "b", "b"]


def test_arms_may_differ_only_in_switchable_options():
    history_arms.check_arms({"a": {"num_envs": 4}, "b": {"num_envs": 4, "learner_length_groups": 2}})
    with pytest.raises(ValueError):
        history_arms.check_arms({"a": {"num_envs": 4}, "b": {"num_envs": 8}})


def test_switch_rebuilds_caches_in_place_and_measures(tmp_path):
    source = tmp_path / "source"
    config = HistoryPPOConfig(width=32, layers=2, heads=4, num_envs=4, steps_per_update=24,
                              snapshot_updates=1, rollout_kv_cache=True, causal_sdpa=True,
                              batch_snapshot_policies=True, updates=2)
    trainer = HistoryTrainer(config, source)
    trainer.run()
    output = tmp_path / "arms"
    assert history_arms.main(["--resume", str(source / "latest.pt"), "--output", str(output),
                              "--device", "cpu", "--warmup", "1", "--blocks", "2",
                              "--block-updates", "2", "--arm", "base=",
                              "--arm", "fast=rollout_paged_cache=true,rollout_page_span=true,"
                                       "batch_snapshot_encoder=true,learner_length_groups=2"]) == 0
    rows = [json.loads(line) for line in (output / "arms.jsonl").open()]
    assert [r["arm"] for r in rows] == ["base"] * 3 + ["fast"] * 4 + ["base"] * 2
    assert rows[3]["cache"].get("pool_used_pages", 0) > 0 and "pool_used_pages" not in rows[-1]["cache"]
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["results"]) == {"base", "fast"}
    assert summary["results"]["fast"]["settled_updates"] == 3   # ABBA: adjacent blocks merge
    with pytest.raises(SystemExit):     # never reuse a non-empty directory
        history_arms.main(["--resume", str(source / "latest.pt"), "--output", str(output),
                           "--device", "cpu", "--arm", "a=", "--arm", "b="])


def test_rejects_duplicate_arm_names(tmp_path):
    with pytest.raises(SystemExit):
        history_arms.main(["--resume", str(tmp_path / "x.pt"), "--output", str(tmp_path / "o"),
                           "--arm", "a=", "--arm", "a=learner_length_groups=2"])
