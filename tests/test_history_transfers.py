"""Aligned uploads and per-vector-step downloads preserve the original fields."""
import numpy as np
import pytest
import torch

from train.history_transfers import download_tensors, runtime_settings, upload_arrays


@pytest.mark.parametrize("enabled", [False, True])
def test_runtime_settings_use_imported_upload_flag_and_current_numerics(monkeypatch, enabled):
    from train import history_transfers as transfers

    # Later environment edits do not alter the setting used by upload_arrays.
    monkeypatch.setattr(transfers, "_PINNED_CUDA_UPLOAD", enabled)
    monkeypatch.setenv("GUANZERO_PINNED_UPLOAD", "0" if enabled else "1")
    assert runtime_settings("cuda")["pinned_upload"] is enabled
    assert runtime_settings("cpu")["pinned_upload"] is False
    original = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.use_deterministic_algorithms(enabled, warn_only=enabled)
        torch.backends.cuda.matmul.allow_tf32 = not enabled
        torch.backends.cudnn.allow_tf32 = enabled
        assert runtime_settings(torch.device("cpu")) == dict(
            pinned_upload=False, deterministic_algorithms=enabled,
            deterministic_warn_only=enabled, tf32_matmul=not enabled, tf32_cudnn=enabled)
    finally:
        torch.use_deterministic_algorithms(original[0], warn_only=original[1])
        torch.backends.cuda.matmul.allow_tf32 = original[2]
        torch.backends.cudnn.allow_tf32 = original[3]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("pinned", [False, True])
def test_upload_preserves_ragged_mixed_fields_and_owns_packed_memory(device, pinned):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA transfer acceptance")
    floats = np.array([0, 0x80000000, 1, 0x7FC00123], np.uint32).view(np.float32)
    arrays = (np.arange(3, dtype=np.uint8), np.array([-(2**63), 2**63 - 1], np.int64),
              floats.reshape(2, 2).T, np.array(13, np.int64),
              np.ones((2, 186), np.uint8), np.empty((0, 3), np.int64))
    before = [array.copy() for array in arrays]
    actual = upload_arrays(arrays, device, packed=True, pinned=pinned)
    reference = upload_arrays(arrays, device, packed=False)
    for a, b, expected in zip(actual, reference, before):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.contiguous().cpu().numpy().tobytes() == expected.tobytes()
        assert b.contiguous().cpu().numpy().tobytes() == expected.tobytes()
        assert a.storage_offset() * a.element_size() % 8 == 0
    arrays[0][:] = 0
    assert actual[0].cpu().numpy().tobytes() == before[0].tobytes()
    assert upload_arrays([], device, packed=True) == ()
    assert download_tensors([], packed=True) == ()


def test_pinned_upload_setting_keeps_cpu_zero_copy_and_packed_bytes(monkeypatch):
    from train import history_transfers as transfers

    monkeypatch.setattr(transfers, "_PINNED_CUDA_UPLOAD", True)
    source = np.arange(17, dtype=np.int64)
    direct, = upload_arrays((source,), "cpu")
    packed, = upload_arrays((source,), "cpu", packed=True)
    assert direct.data_ptr() == torch.from_numpy(source).data_ptr()
    assert not direct.is_pinned() and not packed.is_pinned()
    assert packed.numpy().tobytes() == source.tobytes()
    source[:] = -1
    assert (direct == -1).all()
    assert torch.equal(packed, torch.arange(17))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pinned allocator lifetime acceptance")
def test_pinned_uploads_survive_staging_release_and_original_array_mutation(monkeypatch):
    from train import history_transfers as transfers

    # No caller changes: this exercises the process-level feature setting.
    monkeypatch.setattr(transfers, "_PINNED_CUDA_UPLOAD", True)
    stream = torch.cuda.Stream()
    received, expected = [], []
    source = np.empty(8192, dtype=np.uint8)
    floating_bits = np.array([0, 0x80000000, 1, 0x7FC00123], np.uint32).view(np.float32)
    with torch.cuda.stream(stream):
        # Keep copies queued while local staging references expire. Subsequent
        # calls exercise PyTorch's pinned allocator reuse/event handling.
        torch.cuda._sleep(10_000_000)
        for index in range(32):
            source.fill(index)
            arrays = (source, np.array(index - 16, np.int64), floating_bits.reshape(2, 2).T)
            expected.append(tuple(array.copy() for array in arrays))
            received.append(upload_arrays(arrays, "cuda"))
            source.fill(255)
    torch.cuda.current_stream().wait_stream(stream)
    for actual, reference in zip(received, expected):
        for tensor, array in zip(actual, reference):
            downloaded = tensor.cpu().numpy()
            assert downloaded.dtype == array.dtype and downloaded.shape == array.shape
            assert downloaded.tobytes() == array.tobytes()


def test_batched_attention_config_survives_population_restore():
    from train.history_model import HistoryPolicyConfig, fresh_player
    from train.history_population import HistoryPopulation
    from train.history_ppo import HistoryPPOConfig, build_parser, config_from_args

    args = build_parser().parse_args(["--output", "unused", "--rollout-batched-attention"])
    config = config_from_args(args)
    assert config.rollout_batched_attention
    assert not HistoryPPOConfig.from_payload({}).rollout_batched_attention
    actor, _ = fresh_player(HistoryPolicyConfig(width=16, layers=1), 7)
    actor.batched_private_attention = True
    pool = HistoryPopulation(actor, "attention-test", 11)
    identity = pool.snapshot(1)
    assert pool.resolve(identity).batched_private_attention
    restored = HistoryPopulation(actor, "attention-test", 11)
    restored.load_state_dict(pool.state_dict())
    assert restored.resolve(identity).batched_private_attention


def test_vector_step_packing_matches_separate_identity_transfers(monkeypatch):
    from dataclasses import asdict
    from train import history_rollout as rollout
    from train.history_model import HistoryPolicyConfig, fresh_player
    from test_history_rollout import make_env

    models = {i: fresh_player(HistoryPolicyConfig(width=16, layers=1), i + 17)[0]
              for i in (0, 1, 2)}
    outputs = []
    for packed in (False, True):
        counts = {"upload": 0, "download": 0}

        def upload(values, device):
            counts["upload"] += 1
            return upload_arrays(values, device, packed=packed)

        def download(values, **kwargs):
            counts["download"] += 1
            return download_tensors(values, packed=packed)

        with monkeypatch.context() as patch:
            patch.setattr(rollout, "upload_arrays", upload)
            patch.setattr(rollout, "_download_tensors", download)
            collector = rollout.HistoryCollector(
                make_env(4, 23), models[0], rollout.MatchEventStore(),
                rollout.SequenceRolloutBuffer(), torch.Generator().manual_seed(31),
                seat_policy=lambda env, match: [0, 1, 0, 2], resolve_policy=models.__getitem__,
                record_choices=True, kv_cache=True, temperature=0.8, epsilon=0.2)
            stats = collector.collect(24)
        assert counts == {"upload": 24, "download": 24}
        assert stats.policy_batches > stats.steps
        outputs.append((collector, asdict(stats)))
    (a, sa), (b, sb) = outputs
    assert sa == sb
    assert torch.equal(a.generator.get_state(), b.generator.get_state())
    for name, before in a.buffer.compact().items():
        after = b.buffer.compact()[name]
        assert before.dtype == after.dtype and before.shape == after.shape
        assert before.tobytes() == after.tobytes()
    assert all(x.tobytes() == y.tobytes() for x, y in zip(a.choice_log, b.choice_log))
