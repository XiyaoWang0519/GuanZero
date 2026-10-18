"""The multi-batch private-attention audit (eval/private_attention_audit.py)."""
import json

import numpy as np
import pytest
import torch

from eval import private_attention_audit as audit
from train.history_model import HistoryPolicyConfig, fresh_player

CONFIG = HistoryPolicyConfig(width=32, layers=1, heads=4, aux_heads="outcome")


@pytest.fixture(scope="module")
def sample():
    actor, _ = fresh_player(CONFIG, 1)
    return audit.collect_sample(actor, envs=4, steps=220, seed=2, matches=4,
                                minibatch_matches=2, rows_per_match=24)


def test_sample_covers_whole_matches_and_rounds(sample):
    assert len(sample) == 2
    for batch in sample:
        assert set(audit.BATCH_FIELDS) <= set(batch)
        assert len(np.unique(batch["match_index"].numpy())) == 2
        assert (batch["prefix"] <= batch["lengths"][batch["match_index"]]).all()
    rounds = np.concatenate([b["round"].numpy() for b in sample])
    assert rounds.min() == 0 and rounds.max() >= 1


def test_tribute_round_flags_exchange_tokens():
    tokens = torch.zeros(1, 4, 186, dtype=torch.uint8)
    batch = {"rounds": torch.tensor([[0, 1, 1, 1]]), "phases": torch.tensor([[3, 1, 3, 3]]),
             "match_index": torch.tensor([0, 0, 0]), "prefix": torch.tensor([1, 2, 4]),
             "round": torch.tensor([0, 1, 1]), "tokens": tokens}
    assert audit.tribute_round(batch).tolist() == [False, True, True]


def test_landing_and_distinct_positions():
    batch = {"prefix": torch.tensor([0, 3, 3, 3, 3]), "match_index": torch.tensor([0, 0, 0, 0, 0]),
             "rounds": torch.tensor([[0, 1, 1]]), "round": torch.tensor([0, 1, 1, 1, 1])}
    captured = {"argmax": np.array([[0], [0], [1], [3], [3]])}
    marks = audit.landing(batch, captured)
    assert marks["bos"][:, 0].tolist() == [True, True, False, False, False]
    assert marks["earlier_round"][:, 0].tolist() == [False, False, True, False, False]
    assert marks["current_round"][:, 0].tolist() == [False, False, False, True, True]
    assert marks["last"][:, 0].tolist() == [False, False, False, True, True]
    assert audit.distinct_per_match(batch, captured["argmax"]).tolist() == [[3.0]]


def test_perturbations_touch_only_earlier_rounds(sample):
    actor, _ = fresh_player(CONFIG, 3)
    batch = sample[0]
    inputs = audit.batch_inputs(batch)
    base = audit.capture(actor, inputs)
    assert base["state_error"] < 1e-5
    first_round = batch["round"].numpy() == 0
    for kind in audit.PERTURBATIONS:
        other = audit.capture(actor, audit.perturbed_inputs(batch, kind, 0))
        delta = audit.sensitivity(base, other, batch)
        assert np.all(delta["tv"][first_round] < 1e-6), kind
        if kind == "identity":
            assert np.all(delta["tv"] < 1e-6) and not delta["moved"].any()
    for kind in ("swap_earlier", "shuffle_earlier"):
        perturbed = audit.perturbed_inputs(batch, kind, 0)
        for row in range(len(batch["prefix"])):
            m, j, r = (int(batch["match_index"][row]), int(perturbed.match_index[row]),
                       int(batch["round"][row]))
            n = int(perturbed.streams.lengths[j])
            rounds = batch["rounds"][m, :n]
            assert torch.equal(perturbed.streams.rounds[j, :n], rounds)
            assert torch.equal(perturbed.streams.phases[j, :n], batch["phases"][m, :n])
            kept = rounds >= r
            assert torch.equal(perturbed.streams.tokens[j, :n][kept], batch["tokens"][m, :n][kept])


def test_audit_actor_reports_every_section(sample):
    actor, _ = fresh_player(CONFIG, 4)
    result = audit.audit_actor(actor, sample)
    assert result["decisions"] == sum(len(b["prefix"]) for b in sample)
    assert result["mechanism"]["state_error_max"] < 1e-5
    assert len(result["strata"]["all"]) == CONFIG.heads
    assert {"policy", "outcome"} <= set(result["gradients"])
    assert result["gradients"]["policy"]["q_proj"]["zero_share"] == 0.0
    assert result["sensitivity"]["identity"]["tv_max"] < 1e-6
    assert set(result["dead_relu"]) == {"encoder.0", "private_block"}
    assert result["adam"] is None
    json.dumps(result, default=float)


def test_adam_state_detects_subnormal_floor():
    actor, _ = fresh_player(CONFIG, 5)
    names = [name for name, _ in actor.named_parameters()]
    state = {}
    for i, (name, p) in enumerate(actor.named_parameters()):
        v = torch.full_like(p, 1e-6)
        if name in ("q_proj.weight", "q_proj.bias"):
            v[:8] = 7e-43          # head 0 of 4 at width 32: rows 0..7, sqrt = 8.4e-22
        state[i] = {"exp_avg": torch.zeros_like(p), "exp_avg_sq": v}
    result = audit.adam_state(actor, {"state": state, "param_groups": [{"params": names}]})
    assert result["q_proj.head0"]["floor_share"] == 1.0
    assert result["q_proj.head1"]["floor_share"] == 0.0
    assert result["v_proj"]["sqrt_v_median"] == pytest.approx(1e-3)


def test_cli_writes_json(tmp_path, sample):
    path = tmp_path / "sample.pt"
    torch.save(sample[:1], path)
    audit.main(["fresh:1", "fresh:1:qk", "--out", str(tmp_path / "out"), "--load-sample",
                str(path), "--fresh-width", "32", "--fresh-layers", "1", "--fresh-heads", "4",
                "--fresh-aux-heads", "", "--no-perturb", "--adam-steps", "2"])
    result = json.loads((tmp_path / "out" / "audit.json").read_text())
    assert set(result["actors"]) == {"fresh:1", "fresh:1:qk"}
    assert "adam_probe" in result["actors"]["fresh:1:qk"]
