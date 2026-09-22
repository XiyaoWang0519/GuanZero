"""Causality, information boundaries and metric arithmetic for the v3 probe."""
import json

import numpy as np
import pytest
import torch

from train.belief_experiment import load_dataset
from train.belief_memory import (attach_memory, matched_memory_models, memory_collate,
                                 memory_forward)
from train.belief_model import HistoryBelief
from train.belief_probe import count_parameters
from train.buffer import Decision
from train.logs import TOKEN_DIM, save_round
from train.memory_experiment import (adaptation_metric, least_squares_slope, run,
                                     self_adaptation, style_readout)
from train.tribute_data import engine_source_digest

OBS_DIM = 32


def make_records(matches: int = 2, rounds: int = 3, seed: int = 0) -> list[dict]:
    """Small in-memory rounds with the fields the memory pipeline reads."""
    rng = np.random.default_rng(seed)
    records = []
    for match in range(matches):
        for index in range(rounds):
            tokens = np.zeros((6, TOKEN_DIM), np.uint8)
            tokens[np.arange(6), rng.integers(0, 4, 6)] = 1
            tokens[:, 4 + rng.integers(0, 100, 6)] = 1
            obs = rng.integers(0, 2, (3, OBS_DIM)).astype(np.uint8)
            records.append({
                "group": f"match-{match}", "round_index": index,
                "obs": obs, "hidden": rng.integers(0, 3, (3, 3, 54)).astype(np.uint8),
                "seat": tokens[np.arange(3), :4].argmax(1).astype(np.int64),
                "prefix": np.arange(3, dtype=np.int64), "tokens": tokens,
                "seat_driver": np.array([1, 0, 1, 0], np.int64),
                "styles": rng.normal(size=(4, 4)).astype(np.float32)})
    return records


def items_of(records: list[dict], group: str, index: int) -> list[tuple[dict, int]]:
    record = next(r for r in records if r["group"] == group and r["round_index"] == index)
    return [(record, i) for i in range(len(record["obs"]))]


def predict(model, items, memory_rounds: int = 8) -> torch.Tensor:
    with torch.inference_mode():
        return memory_forward(model, memory_collate(items, "cpu", memory_rounds))


def make_dataset(path, *, matches=20, rounds_per_match=8, seed=3) -> None:
    """Schema-2 rounds on disk, with per-match styles and style regions."""
    rng = np.random.default_rng(seed)
    path.mkdir()
    index = 0
    for match in range(matches):
        region = "heldout" if match % 3 == 0 else "train"
        styles = np.zeros((4, 4), np.float32)
        for seat in (0, 2):
            styles[seat] = rng.normal(size=4)
        for round_index in range(rounds_per_match):
            hidden = np.zeros((3, 54), dtype=np.uint8)
            hidden[index % 3, index % 54] = 1
            obs = np.zeros(1849, dtype=np.uint8)
            obs[108 + index % 54] = 1
            for rel in range(3):
                obs[648 + 28 * rel + int(rel == index % 3)] = 1
            tokens = np.zeros((3, TOKEN_DIM), dtype=np.uint8)
            tokens[np.arange(3), np.arange(3)] = 1
            tokens[:, 4 + (index + np.arange(3)) % 100] = 1
            decisions = [Decision(obs, np.zeros(154), hidden, seat=i, phase=3, prefix=i)
                         for i in range(3)]
            meta = {"match_id": match, "round_index": round_index, "env_id": 0,
                    "styles": styles, "seat_driver": [1, 0, 1, 0], "styled": True,
                    "style_region": region}
            save_round(path / f"round-{index:08d}.npz", decisions, list(tokens),
                       f"match-{match}", meta)
            index += 1
    count = matches * rounds_per_match
    (path / "provenance.json").write_text(json.dumps(
        {"status": "complete", "purpose": "architecture_probe", "learner_updates": 0,
         "engine_source_sha256": engine_source_digest(), "action_mode": "canonical",
         "tribute_policy": "heuristic", "sampling_margin": 0, "collected_rounds": count,
         "seed": 901, "training_seed": 7, "stage": "dmc", "play_mode": "fp32_argmax",
         "checkpoint_id": "a" * 64, "requested_rounds": count,
         "collected_decisions": 3 * count, "match_groups": matches}))


def test_parameter_counts_match_the_no_history_tower_and_each_other():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    # The production shape of the scaled probe: four layers, width 256.
    models = matched_memory_models(1849, 256, 4, memory_rounds=8)
    assert set(models) == {"memory", "memory_masked"}
    sizes = {name: count_parameters(model) for name, model in models.items()}
    assert sizes["memory"] == sizes["memory_masked"]
    reference = count_parameters(HistoryBelief(1849, 256, 4, no_history=True))
    assert 4_600_000 < reference < 4_700_000
    assert sizes["memory"] - reference == (4 + 8) * 256
    assert abs(sizes["memory"] - reference) / reference < .001
    for name, value in models["memory"].state_dict().items():
        control = models["memory_masked"].state_dict()[name]
        torch.testing.assert_close(value, control, rtol=0, atol=0)
        assert value.data_ptr() != control.data_ptr()
    with pytest.raises(ValueError):
        matched_memory_models(OBS_DIM, 64, 2, memory_rounds=0)


def test_memory_is_causal_in_the_match():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    models = matched_memory_models(OBS_DIM, 16, 1, memory_rounds=4, tolerance=.5)
    records = make_records()
    attach_memory(records, 4)
    middle = items_of(records, "match-0", 1)
    before = {name: predict(model, middle, 4) for name, model in models.items()}
    assert not torch.allclose(before["memory"], before["memory_masked"])

    later = next(r for r in records if r["group"] == "match-0" and r["round_index"] == 2)
    later["tokens"] = np.roll(later["tokens"], 3, axis=0).copy()
    attach_memory(records, 4)
    for name, model in models.items():
        torch.testing.assert_close(predict(model, middle, 4), before[name], rtol=0, atol=0)

    earlier = next(r for r in records if r["group"] == "match-0" and r["round_index"] == 0)
    earlier["tokens"] = (1 - earlier["tokens"]).astype(np.uint8)
    attach_memory(records, 4)
    assert not torch.allclose(predict(models["memory"], middle, 4), before["memory"])
    torch.testing.assert_close(predict(models["memory_masked"], middle, 4),
                               before["memory_masked"], rtol=0, atol=0)

    # A different match never enters this match's memory.
    other = next(r for r in records if r["group"] == "match-1")
    other["tokens"] = np.roll(other["tokens"], 1, axis=0).copy()
    attach_memory(records, 4)
    current = predict(models["memory"], middle, 4)
    attach_memory(records, 4)
    torch.testing.assert_close(predict(models["memory"], middle, 4), current, rtol=0, atol=0)


def test_first_round_has_no_memory_and_matches_the_masked_control():
    torch.set_num_threads(1)
    torch.manual_seed(2)
    models = matched_memory_models(OBS_DIM, 16, 1, memory_rounds=4, tolerance=.5)
    records = make_records()
    attach_memory(records, 4)
    first = items_of(records, "match-0", 0)
    assert all(not r["previous"] for r in records if r["round_index"] == 0)
    torch.testing.assert_close(predict(models["memory"], first, 4),
                               predict(models["memory_masked"], first, 4))


def test_summaries_ignore_private_data():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = matched_memory_models(OBS_DIM, 16, 1, memory_rounds=4, tolerance=.5)["memory"]
    records = make_records()
    attach_memory(records, 4)
    middle = items_of(records, "match-0", 1)
    batch = memory_collate(middle, "cpu", 4)
    with torch.inference_mode():
        summary, present = model.summarise(batch["memory"]["tokens"], batch["memory"]["lengths"])
    assert summary.shape[1:] == (4, 16) and present.shape[1] == 4
    earlier = next(r for r in records if r["group"] == "match-0" and r["round_index"] == 0)
    baseline = predict(model, middle, 4)
    # Hidden hands and private observations of a finished round are labels only.
    earlier["obs"] = (1 - earlier["obs"]).astype(np.uint8)
    earlier["hidden"] = (2 - earlier["hidden"]).astype(np.uint8)
    attach_memory(records, 4)
    other = memory_collate(items_of(records, "match-0", 1), "cpu", 4)
    with torch.inference_mode():
        again, _ = model.summarise(other["memory"]["tokens"], other["memory"]["lengths"])
    torch.testing.assert_close(again, summary, rtol=0, atol=0)
    torch.testing.assert_close(predict(model, items_of(records, "match-0", 1), 4),
                               baseline, rtol=0, atol=0)


def test_adaptation_metric_arithmetic_on_a_synthetic_report():
    def evaluation(early, late, indexed):
        cells = {"adapt:early": {"matches": {g: {"log_loss": v} for g, v in early.items()}},
                 "adapt:late": {"matches": {g: {"log_loss": v} for g, v in late.items()}}}
        for index, table in indexed.items():
            cells[f"round_index:{index}"] = {
                "matches": {g: {"log_loss": v} for g, v in table.items()}}
        return {"cells": cells}

    masked = evaluation({"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 1.0},
                        {0: {"a": 1.0}, 1: {"a": 1.0}, 2: {"a": 1.0}})
    memory = evaluation({"a": 0.9, "b": 0.8}, {"a": 0.5, "b": 0.6},
                        {0: {"a": 1.0}, 1: {"a": 0.9}, 2: {"a": 0.8}})
    result = adaptation_metric(masked, memory, 7, 512)
    assert result["available"] and result["test_matches"] == 2
    assert result["mean_early_improvement"] == pytest.approx(.15)
    assert result["mean_late_improvement"] == pytest.approx(.45)
    assert result["mean_adaptation_improvement"] == pytest.approx(.3)
    low, high = result["bootstrap_95_ci"]
    assert low == pytest.approx(.2) and high == pytest.approx(.4)
    assert result["improvement_slope_per_round"] == pytest.approx(.1)
    assert set(result["improvement_by_round_index"]) == {"0", "1", "2"}
    assert result["improvement_by_round_index"]["2"]["mean_improvement"] == pytest.approx(.2)
    # The control's own early-minus-late difference is flat here.
    assert self_adaptation(masked, 7, 512)["mean_early_minus_late"] == pytest.approx(0.)
    assert self_adaptation(memory, 7, 512)["mean_early_minus_late"] == pytest.approx(.3)
    assert least_squares_slope([(1., 2.)]) is None
    assert adaptation_metric({"cells": {}}, {"cells": {}}, 7, 8)["available"] is False


def test_end_to_end_tiny_run_on_schema_two_logs(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    make_dataset(data, matches=20, rounds_per_match=8)
    report = run(data, out, seeds=(41, 42), steps=2, min_steps=1, validation_interval=1,
                 batch_size=4, width=16, layers=1, threads=1, max_seconds=120,
                 memory_rounds=4, late_from=5, heldout_style_test=True,
                 parameter_tolerance=.5,
                 cell_bootstrap_samples=32, readout_rounds=32, validation_decisions=16)
    assert report["status"] == "complete" and report["rl_enabled"] is False
    assert report["gate"] in ("supports_v3_match_memory", "v3_not_yet_justified")
    assert report["matches"] == 20 and report["rounds_per_match"]["max"] == 8
    assert all(r["style_region"] == "heldout"
               for r in load_dataset(data)[0]
               if r["group"] in report["splits"]["test"]["matches"])
    assert len(report["runs"]) == 2
    for result in report["runs"]:
        assert set(result["models"]) == {"memory", "memory_masked"}
        sizes = {name: m["parameters"] for name, m in result["models"].items()}
        assert sizes["memory"] == sizes["memory_masked"]
        for name, metrics in result["models"].items():
            payload = torch.load(out / f"{name}-s{result['seed']}.pt", weights_only=False)
            assert payload["purpose"] == "belief_probe_only"
            assert payload["selected_step"] == metrics["selected_step"]
            assert metrics["seconds_per_step"] > 0
            cells = metrics["test"]["cells"]
            assert {c for c in cells if c.startswith("target_relation:")} == {
                "target_relation:teammate", "target_relation:opponent"}
            assert "target_driver:bot" in cells and "target_driver:policy" in cells
            assert {"adapt:early", "adapt:late"} <= set(cells)
            readout = metrics["style_readout"]
            assert readout["available"] and len(readout["heldout_r2"]) == 4
        adaptation = result["adaptation"]
        assert adaptation["available"] and len(adaptation["bootstrap_95_ci"]) == 2
        assert set(adaptation["improvement_by_round_bin"]) == {"0", "1", "2-3", "4-5", "6+"}
        assert adaptation["improvement_slope_per_round"] is not None
        assert result["self_adaptation"]["memory_masked"]["available"]
        assert np.isfinite(result["memory_vs_masked"]["mean_match_log_loss_improvement"])
    with pytest.raises(FileExistsError):
        run(data, out)


def test_style_readout_reports_a_slot_per_style_and_needs_bot_seats():
    torch.set_num_threads(1)
    torch.manual_seed(5)
    model = matched_memory_models(OBS_DIM, 16, 1, memory_rounds=4, tolerance=.5)["memory"]
    records = make_records(matches=12, rounds=3, seed=9)
    result = style_readout(model, records[:27], records[27:], "cpu", max_rounds=64,
                           batch_rounds=8)
    assert result["available"] and len(result["heldout_r2"]) == 4
    # Two bot seats per round, minus any seat that acted in no token of it.
    assert 0 < result["train_samples"] <= 27 * 2 and 0 < result["test_samples"] <= 9 * 2
    for record in records:
        record["seat_driver"] = np.zeros(4, np.int64)
    assert style_readout(model, records[:27], records[27:], "cpu")["available"] is False
