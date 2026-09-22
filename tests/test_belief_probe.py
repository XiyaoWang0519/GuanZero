"""History causality, match-held-out splitting, and offline probe execution."""
from pathlib import Path

import numpy as np
import torch

from train.belief_probe import HistoryBelief, collate, count_parameters, load_rounds, matched_models, run_probe
from train.buffer import Decision
from train.logs import TOKEN_DIM, save_round


def make_logs(path: Path) -> None:
    rng = np.random.default_rng(11)
    for index in range(6):
        decisions = [Decision(rng.integers(2, size=1849, dtype=np.uint8), np.zeros(154),
                              rng.integers(3, size=(3, 54), dtype=np.uint8), seat=i % 4,
                              phase=3, prefix=i) for i in range(4)]
        save_round(path / f"round-{index:08d}.npz", decisions,
                   [np.zeros(TOKEN_DIM, np.uint8) for _ in range(4)], str(index // 2))


def test_split_keeps_whole_matches_together(tmp_path):
    make_logs(tmp_path)
    train, held_out = load_rounds(tmp_path)
    assert {r["group"] for r in train}.isdisjoint({r["group"] for r in held_out})
    assert len(train) + len(held_out) == 6
    data = collate([(train[0], 0), (train[0], 2)], "cpu")
    assert data["tokens"].shape[1] == 2
    assert data["lengths"].tolist() == [0, 2]


def test_history_never_reads_future_or_padding_and_empty_prefix_is_finite():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = HistoryBelief(1849, 16, 1).eval()
    obs = torch.randn(2, 1849)
    tokens = torch.randn(2, 5, TOKEN_DIM)
    lengths = torch.tensor([0, 2])
    seats = torch.tensor([0, 1])
    with torch.inference_mode():
        expected = model(obs, tokens, lengths, seats)
        tokens[0] = torch.randn_like(tokens[0]) * 100
        tokens[1, 2:] = torch.randn_like(tokens[1, 2:]) * 100
        actual = model(obs, tokens, lengths, seats)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


def test_parameter_matched_probe_runs_on_logged_rounds(tmp_path):
    torch.set_num_threads(1)
    make_logs(tmp_path)
    models = matched_models(1849, 16, 1)
    sizes = [count_parameters(model) for model in models.values()]
    assert abs(sizes[0] - sizes[1]) / sizes[1] < .02
    report = run_probe(tmp_path, steps=1, batch_size=2, width=16, layers=1)
    assert report["train_decisions"] == 16 and report["test_decisions"] == 8
    assert all(np.isfinite(result["log_loss"]) for result in report["results"].values())
