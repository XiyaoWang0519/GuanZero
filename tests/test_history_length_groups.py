"""Learner length groups: the plan partitions matches and rows, every row's prefix
is encoded, and loss and gradients equal the padded minibatch up to FP32 order."""
import numpy as np
import pytest
import torch

from train.history_model import length_groups
from train.history_ppo import HistoryPPOConfig, HistoryTrainer


@pytest.mark.parametrize("groups", [1, 2, 3, 8])
def test_plan_partitions_sorted_runs(groups):
    rng = np.random.default_rng(groups)
    cited = rng.integers(0, 2400, size=17)
    parts = length_groups(cited, groups, 128)
    assert 1 <= len(parts) <= groups
    merged = np.concatenate(parts)
    assert sorted(merged.tolist()) == list(range(17))
    # Contiguous runs of the length-sorted order, longest group first.
    heads = [cited[p].max() for p in parts]
    assert heads == sorted(heads, reverse=True)
    for a, b in zip(parts, parts[1:]):
        assert cited[a].min() >= cited[b].max()
    assert len(length_groups(cited, 1, 128)) == 1


def trainer(tmp_path, **overrides):
    config = dict(width=32, layers=2, heads=4, num_envs=6, steps_per_update=80, seed=3,
                  rollout_kv_cache=True, minibatch_matches=6, snapshot_updates=1)
    config.update(overrides)
    t = HistoryTrainer(HistoryPPOConfig(**config), tmp_path)
    for _ in range(2):
        t.update()
    t.collect()
    t.buffer.finalize(t.refresh_values())
    return t


@pytest.mark.parametrize("response_mode", ["none", "auxiliary"])
@pytest.mark.parametrize("causal_sdpa", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_grouped_loss_and_gradients_match_the_padded_minibatch(tmp_path, response_mode,
                                                                causal_sdpa, batched):
    t = trainer(tmp_path, response_mode=response_mode, causal_sdpa=causal_sdpa,
                learner_batched_attention=batched)
    rows = t.buffer.samples
    batch = t.buffer.training_batch(rows, t.store, "cpu", length_groups=3, width=32)
    groups = batch.inputs.match_groups
    assert len(groups) > 1
    covered = torch.cat([g.rows for g in groups]).sort().values
    assert torch.equal(covered, torch.arange(len(rows)))
    for g in groups:
        assert (batch.inputs.match_index[g.rows] == g.matches[g.local_match]).all()
        assert int(batch.inputs.prefix[g.rows].max()) == g.tokens
    # Streams are only uploaded as far as some row reads.
    assert batch.inputs.streams.tokens.shape[1] == int(batch.inputs.prefix.max())
    results = []
    for count in (0, 3):
        t.config.learner_length_groups = count
        terms = t.minibatch_loss(rows)
        grads = torch.autograd.grad(terms["policy_total"], list(t.actor.parameters()),
                                    allow_unused=True)
        results.append((terms, grads))
    (ref, ref_grads), (grouped, grads) = results
    for key in ("policy_loss", "entropy", "approx_kl", "value_loss"):
        torch.testing.assert_close(grouped[key], ref[key], rtol=1e-5, atol=1e-6)
    for a, b in zip(grads, ref_grads):
        if b is not None:
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_config_and_resume_override():
    from train.history_ppo import RESUME_OVERRIDES, parse_resume_overrides
    with pytest.raises(ValueError):
        HistoryPPOConfig(learner_length_groups=-1)
    with pytest.raises(ValueError):
        HistoryPPOConfig(learner_length_groups=2, window=8)
    assert "learner_length_groups" in RESUME_OVERRIDES
    assert parse_resume_overrides(["learner_length_groups=4"]) == {"learner_length_groups": 4}
