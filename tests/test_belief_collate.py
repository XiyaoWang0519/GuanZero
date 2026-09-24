"""Removing ignored history must preserve predictions and training gradients."""
import copy

import numpy as np
import pytest
import torch

from train.belief_experiment import evaluate
from train.belief_memory import MemoryBelief, attach_memory, memory_collate, memory_forward
from train.belief_model import HistoryBelief
from train.belief_probe import FlatBelief, collate
from train.logs import TOKEN_DIM


def items():
    rng = np.random.default_rng(51)
    records = []
    for index in range(3):
        tokens = rng.integers(0, 2, (7, TOKEN_DIM), dtype=np.uint8)
        tokens[:, :4] = np.eye(4, dtype=np.uint8)[np.arange(7) % 4]
        records.append(dict(group="match", round_index=index,
                            obs=rng.integers(0, 2, (3, 1849), dtype=np.uint8),
                            hidden=rng.integers(0, 3, (3, 3, 54), dtype=np.uint8),
                            seat=np.arange(3), prefix=np.array([0, 3, 7]), tokens=tokens))
    attach_memory(records, 2)
    return [(record, row) for record in records for row in range(3)]


@pytest.mark.parametrize("architecture", ["flat", "no_history", "memory", "memory_masked"])
def test_omitting_unused_inputs_preserves_predictions_and_gradients(architecture):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    data = items()
    if architecture.startswith("memory"):
        model = MemoryBelief(1849, 16, 1, memory_rounds=2,
                             memory_masked=architecture == "memory_masked")
        old = memory_collate(data, "cpu", 2)
        new = memory_collate(data, "cpu", 2, include_history=False,
                             include_memory=not model.memory_masked)
        forward = memory_forward
        if model.memory_masked:
            assert "memory" not in new
        else:
            for name in old["memory"]:
                assert torch.equal(old["memory"][name], new["memory"][name])
    else:
        model = (FlatBelief(1849, 16) if architecture == "flat" else
                 HistoryBelief(1849, 16, 1, no_history=True))
        old = collate(data, "cpu")
        new = collate(data, "cpu", include_history=False)
        forward = lambda m, b: m(b["obs"], b["tokens"], b["lengths"], b["seat"])
    assert new["tokens"].shape == (len(data), 0, TOKEN_DIM)
    assert not new["lengths"].any()
    for key in ("obs", "hidden", "seat"):
        assert torch.equal(old[key], new[key])
    other = copy.deepcopy(model)
    model.train()
    other.train()
    before, after = forward(model, old), forward(other, new)
    assert torch.equal(before, after)
    before.square().mean().backward()
    after.square().mean().backward()
    for p, q in zip(model.parameters(), other.parameters()):
        if p.grad is None:
            assert q.grad is None
        else:
            assert torch.equal(p.grad, q.grad)


@pytest.mark.parametrize("architecture", ["flat", "no_history", "history"])
def test_belief_evaluation_matches_full_collation(architecture):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    model = (FlatBelief(1849, 16) if architecture == "flat" else
             HistoryBelief(1849, 16, 1, no_history=architecture == "no_history"))
    data = items()
    expected = evaluate(model, data, "cpu", 4, collate_fn=collate)
    assert evaluate(model, data, "cpu", 4) == expected
