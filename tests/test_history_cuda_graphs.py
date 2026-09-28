"""CUDA graph storage ownership, update boundaries and real PPO integration."""
import numpy as np
import pytest
import torch

from test_history_model import Rollout
from train.history_cuda_graphs import PrivateDecisionGraphs, storage_signature
from train.history_model import HistoryPolicyConfig, fresh_player
from train.history_ppo import HistoryPPOConfig, HistoryTrainer


def bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def test_storage_signature_distinguishes_structure_from_values():
    actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=1), 4)
    signature = storage_signature(actor)
    with torch.no_grad():
        actor.output_norm.bias.add_(0.25)
    assert storage_signature(actor) == signature
    actor.output_norm.eps *= 2
    assert storage_signature(actor) != signature
    signature = storage_signature(actor)
    actor.output_norm.bias = torch.nn.Parameter(actor.output_norm.bias.detach().clone())
    assert storage_signature(actor) != signature
    with pytest.raises(ValueError, match="CUDA"):
        PrivateDecisionGraphs(actor)


@pytest.fixture
def cuda_backend():
    if not torch.cuda.is_available():
        pytest.skip("CUDA graph acceptance")
    old = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
           torch.is_deterministic_algorithms_warn_only_enabled(),
           torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.set_num_threads(old[0])
    torch.use_deterministic_algorithms(old[1], warn_only=old[2])
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[3:]


@pytest.mark.parametrize("batched", [False, True])
def test_private_graph_observes_updated_weights_and_owns_outputs(cuda_backend, batched):
    actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=1), 4)
    actor.cuda().train()
    actor.batched_private_attention = batched
    snapshot = Rollout(num_envs=4, steps=12, seed=19, capture=(0, 11)).last
    inputs = snapshot.inputs("cuda")
    inputs.one_decision_per_stream = True
    assert torch.equal(inputs.match_index, torch.arange(inputs.decisions, device="cuda"))
    with torch.no_grad():
        encoded = actor.encode_batch(inputs.streams)
    graph = PrivateDecisionGraphs(actor, admit_after=1, max_entries=2, max_bytes=32 << 20)
    try:
        before = graph.forward(inputs, encoded)
        assert graph.stats["captures"] == 1 and len(graph.entries) == 1
        saved = before.clone()
        captures = graph.stats["captures"]
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        state = actor.decision_states(encoded.detach(), inputs)
        (state * torch.linspace(-1, 1, state.shape[1], device="cuda")).mean().backward()
        optimizer.step()
        graph.refresh()
        actual = graph.forward(inputs, encoded)
        with torch.no_grad():
            expected = actor.decision_states(encoded, inputs)
        bits(actual, expected)
        bits(before, saved)
        assert not torch.equal(actual, before)
        assert graph.stats["captures"] == captures
        assert graph.stats["hits"] >= 1
        previous = actor.output_norm.bias.data_ptr()
        actor.output_norm.bias.data = actor.output_norm.bias.detach().clone()
        assert actor.output_norm.bias.data_ptr() != previous
        graph.refresh()
        assert not graph.entries
        with torch.cuda.stream(torch.cuda.Stream()):
            with pytest.raises(ValueError, match="stream"):
                graph.forward(inputs, encoded)
    finally:
        graph.clear()


@pytest.mark.parametrize("triton", [False, True])
def test_private_graph_trainer_update_resume_and_recomputation(cuda_backend, tmp_path, triton):
    if triton:
        pytest.importorskip("triton")
    cfg = HistoryPPOConfig(width=32, layers=1, num_envs=2, steps_per_update=90,
                           epochs=1, minibatch_matches=1, snapshot_updates=1, seed=3,
                           causal_sdpa=True, rollout_kv_cache=True,
                           rollout_batched_attention=True, rollout_private_graphs=True,
                           rollout_triton_cache=triton)
    trainer = HistoryTrainer(cfg, tmp_path / "start", device="cuda")
    line = trainer.update()
    assert line["encoder_grad_norm"] > 0
    assert not trainer.collector.caches[0].entries
    assert trainer.collector.decision_graphs
    assert sum(g.stats["captures"] for g in trainer.collector.decision_graphs.values()) > 0
    if triton:
        assert sum(c.get("triton_packs", 0) for c in trainer.collector.transfer_metrics().values()) > 0
    trainer.collect()
    rows = np.flatnonzero(trainer.buffer.compact()["version"] == 1)
    with torch.no_grad():
        logp, _ = trainer.recompute_log_probs(rows)
    np.testing.assert_allclose(logp.cpu().numpy(), trainer.buffer.compact()["logp"][rows], rtol=0, atol=3e-5)
    checkpoint = trainer.save()
    restored = HistoryTrainer(cfg, tmp_path / "resume", device="cuda", resume=checkpoint)
    assert not restored.collector.decision_graphs
    bits(restored.generator.get_state(), trainer.generator.get_state())
    restored.collect()
    assert 1 in restored.collector.caches
    assert restored.learn()["encoder_grad_norm"] > 0
    # Policies no longer referenced by match assignments release their graphs.
    graph = trainer.collector.decision_graphs[0]
    trainer.collector.assignments.clear()
    trainer.collector._prune_caches()
    assert not trainer.collector.decision_graphs and not graph.entries
