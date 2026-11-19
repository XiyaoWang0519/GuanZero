"""Unit tests for ``train/history_decoder.py``."""
from __future__ import annotations

from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from train.history_decoder import (
    CARD_COUNTER_OBS_KEEP_SLICES,
    PRIVATE_OBS_KEEP_SLICES,
    PRIVATE_OBS_MODES,
    REL_SEAT_BUCKETS,
    REL_STEP_BUCKETS,
    REL_STEP_EXACT,
    PrivateCrossDecoder,
    _compute_rel_buckets,
    compute_stream_pos_arrays,
    extract_stream_pos,
    infer_decision_rounds,
    mask_public_obs,
    private_obs_keep_mask,
)
from train.history_model import DecisionInputs, HistoryActor, HistoryPolicyConfig, StreamBatch
from train.public_history import TOKEN_DIM

ROOT = Path(__file__).resolve().parents[1]


def _make_synthetic_streams(
    lengths: list[int],
    *,
    max_rounds: int = 16,
    seed: int = 7,
) -> tuple[StreamBatch, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    raw_streams = []
    for n in lengths:
        tokens = np.zeros((n, TOKEN_DIM), dtype=np.uint8)
        rounds = np.zeros(n, dtype=np.int64)
        phases = np.zeros(n, dtype=np.int64)
        for t in range(n):
            seat = int((t + rng.integers(0, 4)) % 4)
            tokens[t, seat] = 1
            tokens[t, 4 + int(rng.integers(0, 146))] = 1
            tokens[t, 158 + int(rng.integers(1, 28))] = 1
            rounds[t] = min(max_rounds - 1, t // 6)
        raw_streams.append((tokens, rounds, phases))
    tokens_np, rounds_np, phases_np, lengths_np = StreamBatch.host_arrays(raw_streams)
    batch = StreamBatch(
        tokens=torch.from_numpy(tokens_np),
        rounds=torch.from_numpy(rounds_np),
        phases=torch.from_numpy(phases_np),
        lengths=torch.from_numpy(lengths_np),
    )
    return batch, tokens_np, rounds_np


def _compute_ranks(match_index: torch.Tensor, batches: int) -> tuple[torch.Tensor, int]:
    per_match = torch.bincount(match_index, minlength=batches)
    slots = int(per_match.max())
    order = torch.argsort(match_index, stable=True)
    starts = torch.cumsum(per_match, 0) - per_match
    rank = torch.empty_like(match_index)
    rank[order] = torch.arange(len(match_index), device=match_index.device) - starts[
        match_index[order]
    ]
    return rank, slots


def test_constants_and_keep_slices_match_encoder_header() -> None:
    assert PRIVATE_OBS_MODES == ("full", "private_only", "card_counter")
    assert REL_SEAT_BUCKETS == 5
    assert REL_STEP_EXACT == 32
    assert REL_STEP_BUCKETS == 39

    header = (ROOT / "cpp/include/gd/encoder.h").read_text()
    consts = {
        name: int(val)
        for name, val in re.findall(r"inline constexpr int (kObs\w+)\s*=\s*(\d+);", header)
    }
    expected_slices = (
        (consts["kObsOwnHand"], consts["kObsUnseen"]),
        (consts["kObsLevels"], consts["kObsWildUnseen"]),
        (consts["kObsWildFlags"], consts["kObsTrick"]),
        (consts["kObsPhase"], consts["kObsTribute"]),
    )
    assert PRIVATE_OBS_KEEP_SLICES == expected_slices
    expected_card_counter_slices = (
        (consts["kObsOwnHand"], consts["kObsPlayed"]),
        (consts["kObsCardsLeft"], consts["kObsLastAction"]),
        (consts["kObsPhase"], consts["kObsTribute"]),
    )
    assert CARD_COUNTER_OBS_KEEP_SLICES == expected_card_counter_slices
    assert consts["kObsDim"] == 1849

    # Verify non-overlapping, ordered, exact kept/masked counts.
    for slices, expected_kept in (
        (PRIVATE_OBS_KEEP_SLICES, 171),
        (CARD_COUNTER_OBS_KEEP_SLICES, 545),
    ):
        prev_end = -1
        kept = 0
        for start, end in slices:
            assert start > prev_end
            assert end > start
            kept += end - start
            prev_end = end
        assert kept == expected_kept
        assert consts["kObsDim"] - kept == consts["kObsDim"] - expected_kept

    mask = private_obs_keep_mask(consts["kObsDim"], torch.device("cpu"))
    assert mask.shape == (1, 1849)
    assert mask.dtype == torch.float32
    assert int(mask.sum().item()) == 171
    assert int((1.0 - mask).sum().item()) == 1678
    for start, end in PRIVATE_OBS_KEEP_SLICES:
        assert torch.all(mask[:, start:end] == 1.0)

    cc_mask = private_obs_keep_mask(consts["kObsDim"], torch.device("cpu"), mode="card_counter")
    assert cc_mask.shape == (1, 1849)
    assert cc_mask.dtype == torch.float32
    assert int(cc_mask.sum().item()) == 545
    assert int((1.0 - cc_mask).sum().item()) == 1304
    for start, end in CARD_COUNTER_OBS_KEEP_SLICES:
        assert torch.all(cc_mask[:, start:end] == 1.0)
    # Verify the counter and current trick are kept, and the history summaries are masked.
    assert torch.all(cc_mask[:, consts["kObsUnseen"] : consts["kObsPlayed"]] == 1.0)
    assert torch.all(cc_mask[:, consts["kObsCardsLeft"] : consts["kObsFinishStatus"]] == 1.0)
    assert torch.all(cc_mask[:, consts["kObsFinishStatus"] : consts["kObsLevels"]] == 1.0)
    assert torch.all(cc_mask[:, consts["kObsWildUnseen"] : consts["kObsWildFlags"]] == 1.0)
    assert torch.all(cc_mask[:, consts["kObsTrick"] : consts["kObsLastAction"]] == 1.0)
    for start, end in (
        (consts["kObsPlayed"], consts["kObsCardsLeft"]),
        (consts["kObsLastAction"], consts["kObsPhase"]),
        (consts["kObsTribute"], consts["kObsKnownHoldings"]),
        (consts["kObsKnownHoldings"], consts["kObsDim"]),
    ):
        assert torch.all(cc_mask[:, start:end] == 0.0)

    full_mask = private_obs_keep_mask(consts["kObsDim"], torch.device("cpu"), mode="full")
    assert int(full_mask.sum().item()) == 1849
    with pytest.raises(ValueError, match="private_obs_mode must be one of"):
        private_obs_keep_mask(consts["kObsDim"], torch.device("cpu"), mode="invalid")

    # Clamping behavior for smaller obs_dim values.
    assert private_obs_keep_mask(0, torch.device("cpu")).shape == (1, 0)
    small = private_obs_keep_mask(760, torch.device("cpu"), dtype=torch.float64)
    assert small.dtype == torch.float64
    assert int(small.sum().item()) == 108 + (760 - 748)


def test_mask_public_obs_behavior_and_no_host_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(123)
    n, obs_dim = 512, 1849
    obs = torch.randint(0, 2, (n, obs_dim), dtype=torch.uint8)
    seat = torch.arange(n, dtype=torch.long) % 4
    prefix = torch.arange(n, dtype=torch.long) * 3 + 1

    # prob=0.0 keeps exact obs.float().
    out_zero = mask_public_obs(obs, seat, prefix, 0.0)
    assert out_zero.dtype == torch.float32
    assert torch.equal(out_zero, obs.float())
    assert torch.equal(mask_public_obs(obs, seat, prefix, torch.tensor(-0.5)), obs.float())

    # prob=1.0 zeros all 1678 public dims and preserves all 171 private dims.
    keep = private_obs_keep_mask(obs_dim, obs.device).bool()[0]
    out_one = mask_public_obs(obs, seat, prefix, 1.0)
    assert torch.equal(out_one[:, keep], obs.float()[:, keep])
    assert torch.all(out_one[:, ~keep] == 0.0)
    assert torch.equal(mask_public_obs(obs, seat, prefix, torch.tensor(1.0)), out_one)

    # Verify explicit boundary hashes h=0 and h=65535 still get dropped when prob=1.0.
    zero_obs = torch.ones((2, obs_dim), dtype=torch.uint8)
    zero_obs[0, :108] = 0
    zero_seat = torch.tensor([0, 1], dtype=torch.long)
    zero_prefix = torch.tensor([0, 5], dtype=torch.long)
    out_edge = mask_public_obs(zero_obs, zero_seat, zero_prefix, 1.0)
    assert torch.all(out_edge[:, ~keep] == 0.0)

    # Intermediate prob=0.5 is deterministic, invariant to public perturbations, ~50% drop rate.
    out_half_a = mask_public_obs(obs, seat, prefix, 0.5)
    out_half_b = mask_public_obs(obs, seat, prefix, torch.tensor(0.5))
    assert torch.equal(out_half_a, out_half_b)

    obs_public_flipped = obs.clone()
    obs_public_flipped[:, ~keep] = 1 - obs_public_flipped[:, ~keep]
    out_half_flipped = mask_public_obs(obs_public_flipped, seat, prefix, 0.5)
    was_masked_a = torch.all(out_half_a[:, ~keep] == 0.0, dim=1)
    was_masked_b = torch.all(out_half_flipped[:, ~keep] == 0.0, dim=1)
    assert torch.equal(was_masked_a, was_masked_b)
    drop_rate = was_masked_a.float().mean().item()
    assert 0.35 < drop_rate < 0.65

    # Ensure mask_public_obs never calls Tensor.item() or bool(tensor) (CUDA host sync guard).
    def fail_sync(*_args, **_kwargs):
        raise AssertionError("mask_public_obs must not perform a host sync via .item() or bool()")

    prob_tensor = torch.tensor(0.5, dtype=torch.float32)
    monkeypatch.setattr(torch.Tensor, "item", fail_sync)
    monkeypatch.setattr(torch.Tensor, "__bool__", fail_sync)
    out_nosync = mask_public_obs(obs, seat, prefix, prob_tensor)
    monkeypatch.undo()
    assert torch.equal(out_nosync, out_half_a)

    # Empty batch edge case.
    empty_obs = torch.zeros((0, obs_dim), dtype=torch.uint8)
    empty_idx = torch.zeros(0, dtype=torch.long)
    assert mask_public_obs(empty_obs, empty_idx, empty_idx, 0.5).shape == (0, obs_dim)

    # card_counter keeps the remaining-card counter and current trick throughout warmup.
    cc_keep = private_obs_keep_mask(obs_dim, obs.device, mode="card_counter").bool()[0]
    cc_one = mask_public_obs(obs, seat, prefix, 1.0, mode="card_counter")
    assert torch.equal(cc_one[:, cc_keep], obs.float()[:, cc_keep])
    assert torch.all(cc_one[:, ~cc_keep] == 0.0)
    assert torch.equal(mask_public_obs(obs, seat, prefix, 0.0, mode="card_counter"), obs.float())
    assert torch.equal(mask_public_obs(obs, seat, prefix, 1.0, mode="full"), obs.float())

    obs_cc_masked_flipped = obs.clone()
    obs_cc_masked_flipped[:, ~cc_keep] = 1 - obs_cc_masked_flipped[:, ~cc_keep]
    cc_half_a = mask_public_obs(obs, seat, prefix, 0.5, mode="card_counter")
    cc_half_b = mask_public_obs(obs_cc_masked_flipped, seat, prefix, 0.5, mode="card_counter")
    assert torch.equal(
        torch.all(cc_half_a[:, ~cc_keep] == 0.0, dim=1),
        torch.all(cc_half_b[:, ~cc_keep] == 0.0, dim=1),
    )

    monkeypatch.setattr(torch.Tensor, "item", fail_sync)
    monkeypatch.setattr(torch.Tensor, "__bool__", fail_sync)
    cc_nosync = mask_public_obs(obs, seat, prefix, prob_tensor, mode="card_counter")
    monkeypatch.undo()
    assert torch.equal(cc_nosync, cc_half_a)

    with pytest.raises(ValueError, match="private_obs_mode must be one of"):
        mask_public_obs(obs, seat, prefix, 1.0, mode="invalid")


def test_stream_pos_arrays_extract_and_infer_rounds() -> None:
    empty_s, empty_r = compute_stream_pos_arrays(
        np.zeros((0, TOKEN_DIM), dtype=np.uint8), np.zeros(0, dtype=np.int64), max_rounds=16
    )
    assert empty_s.shape == (0,) and empty_s.dtype == np.uint8
    assert empty_r.shape == (0,) and empty_r.dtype == np.uint8

    tokens = np.zeros((4, TOKEN_DIM), dtype=np.uint8)
    for i, seat in enumerate((0, 3, 1, 2)):
        tokens[i, seat] = 1
    rounds = np.array([0, 2, 15, 22], dtype=np.int64)
    ps_ev, pr_ev = compute_stream_pos_arrays(tokens, rounds, max_rounds=16)
    assert ps_ev.dtype == np.uint8 and pr_ev.dtype == np.uint8
    np.testing.assert_array_equal(ps_ev, np.array([1, 4, 2, 3], dtype=np.uint8))
    np.testing.assert_array_equal(pr_ev, np.array([1, 3, 16, 16], dtype=np.uint8))

    # extract_stream_pos on raw StreamBatch (including length=0 and padding beyond T+1).
    lengths = [0, 5, 19]
    batch, tokens_np, rounds_np = _make_synthetic_streams(lengths, max_rounds=16, seed=11)
    encoded_len = 32  # bucketed/page-span > T + 1 (20)
    pos_seat, pos_round = extract_stream_pos(batch, encoded_len, 16, torch.device("cpu"))
    assert pos_seat.shape == (3, 32) and pos_seat.dtype == torch.int64
    assert pos_round.shape == (3, 32) and pos_round.dtype == torch.int64
    assert torch.all(pos_seat[:, 0] == 0) and torch.all(pos_round[:, 0] == 0)
    for b, length in enumerate(lengths):
        assert torch.all(pos_seat[b, length + 1 :] == 0)
        assert torch.all(pos_round[b, length + 1 :] == 0)
        if length > 0:
            ev_s, ev_r = compute_stream_pos_arrays(tokens_np[b, :length], rounds_np[b, :length], 16)
            np.testing.assert_array_equal(pos_seat[b, 1 : length + 1].numpy(), ev_s)
            np.testing.assert_array_equal(pos_round[b, 1 : length + 1].numpy(), ev_r)

    # Truncated encoded_len < T + 1.
    short_s, short_r = extract_stream_pos(batch, 4, 16, torch.device("cpu"))
    assert torch.equal(short_s, pos_seat[:, :4])
    assert torch.equal(short_r, pos_round[:, :4])

    # Precomputed pos_seat / pos_round path (slice and right-pad).
    pre_streams = SimpleNamespace(
        pos_seat=pos_seat[:, :20].to(torch.uint8),
        pos_round=pos_round[:, :20].to(torch.uint8),
    )
    ps_pad, pr_pad = extract_stream_pos(pre_streams, 32, 16, torch.device("cpu"))
    assert torch.equal(ps_pad, pos_seat) and torch.equal(pr_pad, pos_round)
    ps_slice, pr_slice = extract_stream_pos(pre_streams, 10, 16, torch.device("cpu"))
    assert torch.equal(ps_slice, pos_seat[:, :10]) and torch.equal(pr_slice, pos_round[:, :10])

    # infer_decision_rounds.
    match_idx = torch.tensor([0, 1, 2, 2], dtype=torch.long)
    prefix = torch.tensor([0, 3, 18, 50], dtype=torch.long)
    inferred = infer_decision_rounds(None, pos_round, match_idx, prefix)
    expected_r = torch.tensor(
        [
            0,
            max(0, int(pos_round[1, 3].item()) - 1),
            max(0, int(pos_round[2, 18].item()) - 1),
            0,  # clamped to index 31 which is padding 0 -> returns 0
        ],
        dtype=torch.long,
    )
    assert torch.equal(inferred, expected_r)

    explicit = torch.tensor([2, 1, 0, 5], dtype=torch.long)
    assert torch.equal(infer_decision_rounds(explicit, pos_round, match_idx, prefix), explicit)
    assert infer_decision_rounds(None, pos_round, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)).shape == (0,)


def test_step_bucket_boundaries() -> None:
    deltas = torch.tensor([0, 1, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1023, 1024, 5000])
    seq_len = int(deltas.max().item()) + 2
    prefix = torch.full((len(deltas), 1), seq_len - 1, dtype=torch.long)
    # Place a valid event at position s = prefix - delta so delta = prefix - s.
    pos_seat = torch.ones((len(deltas), seq_len), dtype=torch.long)
    pos_seat[:, 0] = 0
    pos_round = torch.ones((len(deltas), seq_len), dtype=torch.long)
    pos_round[:, 0] = 0
    seat = torch.zeros((len(deltas), 1), dtype=torch.long)
    my_round = torch.zeros((len(deltas), 1), dtype=torch.long)
    allowed, _sb, _rb, step_b = _compute_rel_buckets(
        pos_seat, pos_round, prefix, seat, my_round, max_rounds=16
    )
    assert torch.all(allowed)
    assert torch.all(step_b[:, 0] == 0)
    s_idx = prefix[:, 0] - deltas
    got = step_b[torch.arange(len(deltas)), s_idx]
    expected = torch.tensor([1, 2, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38])
    assert torch.equal(got, expected)


def test_decoder_validation_and_empty_batch() -> None:
    with pytest.raises(ValueError):
        PrivateCrossDecoder(width=0, heads=2, layers=1)
    with pytest.raises(ValueError):
        PrivateCrossDecoder(width=30, heads=4, layers=1)
    with pytest.raises(ValueError):
        PrivateCrossDecoder(width=32, heads=4, layers=0)
    with pytest.raises(ValueError):
        PrivateCrossDecoder(width=32, heads=4, layers=1, max_rounds=0)
    with pytest.raises(ValueError):
        PrivateCrossDecoder(width=32, heads=4, layers=1, query_slots=0)

    dec = PrivateCrossDecoder(width=32, heads=4, layers=2, max_rounds=8, query_slots=4)
    norm = nn.LayerNorm(32)
    out = dec(
        query=torch.zeros((0, 32)),
        attention_norm=norm,
        encoded=torch.zeros((0, 1, 32)),
        pos_seat=torch.zeros((0, 1), dtype=torch.long),
        pos_round=torch.zeros((0, 1), dtype=torch.long),
        match_index=torch.zeros(0, dtype=torch.long),
        prefix=torch.zeros(0, dtype=torch.long),
        seat=torch.zeros(0, dtype=torch.long),
        round_index=torch.zeros(0, dtype=torch.long),
    )
    assert out.shape == (0, 32)


@pytest.mark.parametrize("layers", [1, 3])
@pytest.mark.parametrize("query_slots", [1, 4])
@pytest.mark.parametrize("prefix_mode", ["zero", "multi_round"])
def test_decoder_dispatch_parity_causality_and_gradients(
    layers: int,
    query_slots: int,
    prefix_mode: str,
) -> None:
    torch.manual_seed(42 + layers * 10 + query_slots)
    width, heads, max_rounds = 32, 4, 8
    dec = PrivateCrossDecoder(width, heads, layers, max_rounds=max_rounds, query_slots=query_slots)
    norm = nn.LayerNorm(width)

    lengths = [20, 45, 72] if prefix_mode == "multi_round" else [4, 10, 12]
    streams, _, _ = _make_synthetic_streams(lengths, max_rounds=max_rounds, seed=5)
    seq_len = int(streams.tokens.shape[1]) + 1
    pos_seat, pos_round = extract_stream_pos(streams, seq_len, max_rounds, torch.device("cpu"))

    if prefix_mode == "zero":
        match_index = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.long)
        prefix = torch.zeros(len(match_index), dtype=torch.long)
        seat = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)
        round_index = torch.zeros(len(match_index), dtype=torch.long)
    else:
        match_index = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=torch.long)
        prefix = torch.tensor([0, 7, 18, 12, 41, 1, 33, 68, 72], dtype=torch.long)
        seat = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 1], dtype=torch.long)
        round_index = torch.tensor([0, 1, 3, 2, 6, 0, 5, 7, 7], dtype=torch.long)

    n = len(match_index)
    query = torch.randn(n, width)
    encoded = torch.randn(len(lengths), seq_len, width)

    # 1. Default per-match loop
    out_loop = dec(
        query, norm, encoded, pos_seat, pos_round, match_index, prefix, seat, round_index
    )
    assert out_loop.shape == (n, width)
    assert torch.isfinite(out_loop).all()

    # 2. batched_match_attention (auto rank & explicit host rank)
    out_match_auto = dec(
        query,
        norm,
        encoded,
        pos_seat,
        pos_round,
        match_index,
        prefix,
        seat,
        round_index,
        batched_match_attention=True,
    )
    rank, slots = _compute_ranks(match_index, len(lengths))
    out_match_host = dec(
        query,
        norm,
        encoded,
        pos_seat,
        pos_round,
        match_index,
        prefix,
        seat,
        round_index,
        batched_match_attention=True,
        rank=rank,
        slots=slots,
    )
    assert (out_match_auto - out_loop).abs().max().item() < 1e-5
    assert torch.equal(out_match_auto, out_match_host)

    # 3 & 4. one_decision_per_stream + batched_private_attention (independent vs wide projection)
    enc_each = encoded[match_index]
    ps_each = pos_seat[match_index]
    pr_each = pos_round[match_index]
    idx_each = torch.arange(n, dtype=torch.long)

    out_each_loop = dec(
        query,
        norm,
        enc_each,
        ps_each,
        pr_each,
        idx_each,
        prefix,
        seat,
        round_index,
        one_decision_per_stream=True,
    )
    out_batched_indep = dec(
        query,
        norm,
        enc_each,
        ps_each,
        pr_each,
        idx_each,
        prefix,
        seat,
        round_index,
        one_decision_per_stream=True,
        batched_private_attention=True,
        wide_private_projection=False,
    )
    out_batched_wide = dec(
        query,
        norm,
        enc_each,
        ps_each,
        pr_each,
        idx_each,
        prefix,
        seat,
        round_index,
        one_decision_per_stream=True,
        batched_private_attention=True,
        wide_private_projection=True,
    )
    assert (out_each_loop - out_loop).abs().max().item() < 1e-5
    assert (out_batched_indep - out_each_loop).abs().max().item() < 1e-6
    assert (out_batched_wide - out_loop).abs().max().item() < 1e-5

    # Causal prefix invariance: perturbing tokens > prefix[i] leaves row i unchanged.
    for row in range(n):
        p = int(prefix[row].item())
        if p + 1 >= seq_len:
            continue
        enc_pert = encoded.clone()
        ps_pert = pos_seat.clone()
        pr_pert = pos_round.clone()
        m = int(match_index[row].item())
        enc_pert[m, p + 1 :] += 50.0
        ps_pert[m, p + 1 :] = ((ps_pert[m, p + 1 :] + 1) % 4) + 1
        pr_pert[m, p + 1 :] = max_rounds
        out_pert = dec(
            query, norm, enc_pert, ps_pert, pr_pert, match_index, prefix, seat, round_index
        )
        assert torch.equal(out_pert[row], out_loop[row])

    # Gradients check across all dispatch modes on multi_round (all parameters active).
    if prefix_mode == "multi_round":
        for mode_kwargs in (
            {},
            {"batched_match_attention": True},
            {"one_decision_per_stream": True, "batched_private_attention": True, "wide_private_projection": False},
            {"one_decision_per_stream": True, "batched_private_attention": True, "wide_private_projection": True},
        ):
            dec.zero_grad(set_to_none=True)
            norm.zero_grad(set_to_none=True)
            q_var = query.detach().clone().requires_grad_(True)
            if mode_kwargs.get("one_decision_per_stream"):
                e_var = enc_each.detach().clone().requires_grad_(True)
                ps_in, pr_in, m_in = ps_each, pr_each, idx_each
            else:
                e_var = encoded.detach().clone().requires_grad_(True)
                ps_in, pr_in, m_in = pos_seat, pos_round, match_index
            res = dec(
                q_var, norm, e_var, ps_in, pr_in, m_in, prefix, seat, round_index, **mode_kwargs
            )
            res.pow(2).sum().backward()
            assert torch.isfinite(q_var.grad).all() and q_var.grad.abs().sum().item() > 0
            assert torch.isfinite(e_var.grad).all() and e_var.grad.abs().sum().item() > 0
            for name, param in dec.named_parameters():
                assert param.grad is not None, f"missing grad for {name} in {mode_kwargs}"
                assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"
                assert param.grad.abs().sum().item() > 0, f"zero grad for {name} in {mode_kwargs}"


def test_batched_match_padding_slot_safety() -> None:
    torch.manual_seed(99)
    width, heads, max_rounds = 32, 4, 8
    dec = PrivateCrossDecoder(width, heads, layers=2, max_rounds=max_rounds, query_slots=4)
    norm = nn.LayerNorm(width)

    # 4 matches: match 1 has 0 decisions, match 0 has 5 decisions -> lots of empty padding slots.
    streams, _, _ = _make_synthetic_streams([12, 8, 10, 15], max_rounds=max_rounds, seed=3)
    seq_len = int(streams.tokens.shape[1]) + 1
    pos_seat, pos_round = extract_stream_pos(streams, seq_len, max_rounds, torch.device("cpu"))
    match_index = torch.tensor([0, 0, 0, 0, 0, 2, 3, 3], dtype=torch.long)
    prefix = torch.tensor([0, 3, 6, 9, 12, 0, 4, 15], dtype=torch.long)
    seat = torch.tensor([0, 1, 2, 3, 0, 2, 1, 3], dtype=torch.long)
    round_index = torch.tensor([0, 0, 1, 1, 2, 0, 0, 2], dtype=torch.long)
    query = torch.randn(len(match_index), width, requires_grad=True)
    encoded = torch.randn(4, seq_len, width, requires_grad=True)

    out_ref = dec(
        query, norm, encoded, pos_seat, pos_round, match_index, prefix, seat, round_index
    )
    out_batched = dec(
        query,
        norm,
        encoded,
        pos_seat,
        pos_round,
        match_index,
        prefix,
        seat,
        round_index,
        batched_match_attention=True,
    )
    assert torch.isfinite(out_batched).all()
    assert (out_batched - out_ref).abs().max().item() < 1e-5

    out_batched.pow(2).sum().backward()
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(encoded.grad).all()
    # Match 1 had no decisions, so its gradient must be identically zero (never NaN).
    assert torch.all(encoded.grad[1] == 0.0)
    for name, param in dec.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), f"NaN/Inf in {name}"


def test_relative_seat_round_step_sensitivity() -> None:
    torch.manual_seed(77)
    width, heads, max_rounds = 32, 4, 8
    dec = PrivateCrossDecoder(width, heads, layers=2, max_rounds=max_rounds, query_slots=2)
    norm = nn.LayerNorm(width)
    seq_len = 80
    encoded = torch.randn(1, seq_len, width)
    pos_seat = torch.ones((1, seq_len), dtype=torch.long)
    pos_seat[:, 0] = 0
    pos_round = torch.ones((1, seq_len), dtype=torch.long)
    pos_round[:, 0] = 0
    query = torch.randn(1, width)
    match_idx = torch.zeros(1, dtype=torch.long)
    prefix = torch.tensor([70], dtype=torch.long)
    seat = torch.tensor([0], dtype=torch.long)
    round_idx = torch.tensor([2], dtype=torch.long)

    base = dec(query, norm, encoded, pos_seat, pos_round, match_idx, prefix, seat, round_idx)

    # Seat sensitivity (self vs partner).
    out_seat = dec(
        query, norm, encoded, pos_seat, pos_round, match_idx, prefix, torch.tensor([2]), round_idx
    )
    assert (out_seat - base).abs().max().item() > 1e-4

    # Round sensitivity (event round 1 vs round 2).
    pr_mod = pos_round.clone()
    pr_mod[:, 65:] = 2
    out_round = dec(query, norm, encoded, pos_seat, pr_mod, match_idx, prefix, seat, round_idx)
    assert (out_round - base).abs().max().item() > 1e-4

    # Step sensitivity across log2 bucket boundary (prefix=63 -> delta=62 vs prefix=64 -> delta=63/64).
    pos_seat_step = torch.zeros((1, seq_len), dtype=torch.long)
    pos_seat_step[:, 1] = 2
    pos_round_step = torch.zeros((1, seq_len), dtype=torch.long)
    pos_round_step[:, 1] = 1
    out_p32 = dec(
        query,
        norm,
        encoded,
        pos_seat_step,
        pos_round_step,
        match_idx,
        torch.tensor([32]),  # delta=31 -> bucket 32
        seat,
        round_idx,
    )
    out_p33 = dec(
        query,
        norm,
        encoded,
        pos_seat_step,
        pos_round_step,
        match_idx,
        torch.tensor([33]),  # delta=32 -> bucket 33
        seat,
        round_idx,
    )
    assert (out_p33 - out_p32).abs().max().item() > 1e-4


@pytest.mark.parametrize("query_slots", [1, 4])
def test_reset_for_migration(query_slots: int) -> None:
    torch.manual_seed(19)
    width, heads, layers, max_rounds = 32, 4, 3, 8
    dec = PrivateCrossDecoder(width, heads, layers, max_rounds=max_rounds, query_slots=query_slots)
    norm = nn.LayerNorm(width)
    r_bar = torch.randn(width)
    dec.reset_for_migration(r_bar, residual_scale=0.01)

    assert torch.equal(dec.reader_bias, r_bar)
    assert torch.all(dec.query_round.weight == 0.0)
    for layer in dec.layers:
        assert torch.all(layer.out_proj.bias == 0.0)
        assert torch.all(layer.ffn[2].bias == 0.0)
    if dec.slot_proj is not None:
        assert torch.all(dec.slot_proj.bias == 0.0)

    streams, _, _ = _make_synthetic_streams([12], max_rounds=max_rounds, seed=9)
    seq_len = int(streams.tokens.shape[1]) + 1
    pos_seat, pos_round = extract_stream_pos(streams, seq_len, max_rounds, torch.device("cpu"))
    query = torch.randn(3, width)
    encoded = torch.randn(1, seq_len, width, requires_grad=True)
    match_idx = torch.zeros(3, dtype=torch.long)
    prefix = torch.tensor([0, 5, 12], dtype=torch.long)
    seat = torch.tensor([0, 1, 2], dtype=torch.long)
    round_idx = torch.tensor([0, 1, 2], dtype=torch.long)

    out = dec(query, norm, encoded, pos_seat, pos_round, match_idx, prefix, seat, round_idx)
    target = norm(query + r_bar)
    assert (out - target).abs().max().item() < 5e-2

    out.pow(2).sum().backward()
    for name, param in dec.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), f"bad grad for {name}"
        assert param.grad.abs().sum().item() > 0, f"zero grad for {name}"


def test_history_actor_card_counter_mode_masking_and_gradients() -> None:
    torch.manual_seed(42)
    cfg = HistoryPolicyConfig(
        width=32,
        layers=1,
        heads=4,
        private_decoder_layers=2,
        private_query_slots=2,
        private_obs_mode="card_counter",
    )
    actor = HistoryActor(cfg)
    assert hasattr(actor, "obs_mask_prob")
    actor.obs_mask_prob.fill_(1.0)

    streams, _, _ = _make_synthetic_streams([6, 8], max_rounds=cfg.max_rounds, seed=11)
    n = 3
    obs = torch.rand((n, 1849), dtype=torch.float32)
    cand = torch.randn((7, 154), dtype=torch.float32)
    offsets = torch.tensor([0, 3, 5, 7], dtype=torch.long)
    inputs = DecisionInputs(
        streams=streams,
        match_index=torch.tensor([0, 0, 1], dtype=torch.long),
        prefix=torch.tensor([3, 6, 5], dtype=torch.long),
        obs=obs,
        seat=torch.tensor([0, 2, 1], dtype=torch.long),
        cand=cand,
        offsets=offsets,
        round_index=torch.tensor([0, 1, 0], dtype=torch.long),
    )

    with torch.no_grad():
        base_lp = actor.candidate_log_probs(inputs)
        # Perturb every masked public dimension; the output must be invariant.
        obs_masked_pert = obs.clone()
        for start, end in ((216, 648), (968, 1430), (1439, 1849)):
            obs_masked_pert[:, start:end] += 3.5
        pert_masked_inputs = DecisionInputs(
            streams=streams,
            match_index=inputs.match_index,
            prefix=inputs.prefix,
            obs=obs_masked_pert,
            seat=inputs.seat,
            cand=cand,
            offsets=offsets,
            round_index=inputs.round_index,
        )
        assert torch.equal(actor.candidate_log_probs(pert_masked_inputs), base_lp)

        # Perturbing the counter or current trick must affect policy outputs.
        for start, end in ((108, 216), (648, 732), (732, 748), (790, 793), (805, 968)):
            obs_counter_pert = obs.clone()
            obs_counter_pert[:, start:end] += 2.0
            counter_inputs = DecisionInputs(
                streams=streams,
                match_index=inputs.match_index,
                prefix=inputs.prefix,
                obs=obs_counter_pert,
                seat=inputs.seat,
                cand=cand,
                offsets=offsets,
                round_index=inputs.round_index,
            )
            diff = float((actor.candidate_log_probs(counter_inputs) - base_lp).abs().max().item())
            assert diff > 1e-4, f"kept card_counter slice [{start}:{end}] had zero effect"

    actor.zero_grad(set_to_none=True)
    lp = actor.candidate_log_probs(inputs)
    loss = lp.pow(2).sum()
    loss.backward()
    for module_name, module in (
        ("private", actor.private),
        ("private_decoder", actor.private_decoder),
        ("stream", actor.stream),
        ("action_tower", actor.action_tower),
        ("fusion", actor.fusion),
    ):
        assert module is not None
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, f"missing grads in {module_name}"
        assert all(torch.isfinite(g).all() for g in grads), f"non-finite grad in {module_name}"
        assert sum(float(g.abs().sum().item()) for g in grads) > 0.0, f"zero grad in {module_name}"
