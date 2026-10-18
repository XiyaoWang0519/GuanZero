"""Paged append attention read in place from ``KVPagePool`` pages (Triton).

Wired in behind the opt-in ``rollout_paged_attention_kernel`` switch
(``PagedHistoryCache(attention_kernel=True)``), where it replaces one layer's
attention in ``PagedHistoryCache._append`` and ``_merged_append`` (and, with
private K/V pages, the learner's one-query private attention):

    rows = _rows(table, ends, arange(span))                    # [B, span]
    packed = index_select(kv, 0, rows).view(B, span, 2, H, D)  # zero-padded copy
    attended = SDPA(q, K, V, attn_mask=allowed)                # [B, H, W, D]

Semantics. Stream ``b`` appends ``count_b`` positions ``start_b .. end_b - 1``,
padded to ``W = max(count)`` query rows. Query row ``i`` sees key ``j`` when
``j <= start_b + i`` and ``j < end_b``; key ``j`` is pool row
``table[b, j // P] * P + j % P`` of the layer's ``kv[rows, 2, H, D]``
(``[:, 0]`` keys, ``[:, 1]`` values). Padded rows ``i >= count_b`` see every
key ``< end_b``, so with ``end_b >= 1`` no row is fully masked; their output is
finite, equal to the reference's, and discarded by callers.

``paged_append_attention_reference`` is exactly today's path (the ``_rows``
zero-page gather plus masked ``F.scaled_dot_product_attention``) and is the
fallback. ``paged_append_attention_triton`` runs one program per (stream,
head, query tile, key split): it walks only the pages the tile can see
(``min(end_b, start_b + last row + 1)`` keys), loads each ``[P, D]`` K/V tile
straight from the pool with key-masked loads (rows at or past ``end_b`` are
never read, whatever they hold), applies both masks in registers and keeps an
FP32 online softmax (``tl.dot`` with IEEE FP32 products, no TF32; ``exp2`` with
the ``1/sqrt(D) * log2(e)`` scale folded into the logits). With more than one
key split, partial ``(acc, max, sum)`` triples go to a scratch buffer and a
second kernel combines them (flash-decoding). Every ``-inf`` case is guarded
without producing a NaN: an empty tile or split keeps ``max = -inf`` and
``sum = 0`` and gets weight 0 in the combine.

Exactness: acceptance tier 2. The function is the reference's, but the FP32
reduction order differs (online softmax, ``exp2``, split combine): the
interpreter tests measure about 1e-7 absolute against the reference on unit
logits. The integration is therefore the opt-in ``rollout_paged_attention_kernel``
switch (default off, recorded with the other inference options); it must be
gated on real-rollout log-probability parity on the target GPU before use.

Limits (the dispatcher uses the reference unless the next four hold) and the
caller's contract:
- FP32 ``q`` and ``kv`` only. Inference only: grad-requiring inputs raise.
- Triton with CUDA tensors, or Triton's CPU interpreter. Triton fixes the
  interpreter choice when it is first imported, so ``TRITON_INTERPRET=1`` must
  be set before that (tests run their interpreter cases in a subprocess).
- Head depth ``D`` and ``page_tokens`` powers of two (``tl.dot`` tile minimum
  16), ``D <= 64``, ``page_tokens <= 128`` and ``D * page_tokens <= 4096`` so
  the tiles fit sm_89 shared memory (production D=16, P=64); other shapes take
  the reference path.
- Integer ``table``/``starts``/``ends`` on the device of ``q``.
- ``0 <= start_b``, ``1 <= end_b`` and the table covers ``ceil(end_b / P)``
  pages; ``B <= 65535`` (grid dimension). The kernel clamps page walks to the
  table width for memory safety; it does not check the contract (no host sync).
- Output is a ``[B, H, W, D]`` view of ``[B, W, H, D]`` contiguous memory, so
  ``finish``'s ``transpose(1, 2).reshape`` is free.

Verified: CPU interpreter (Triton 3.8) against the reference and a float64
loop, tests/test_history_paged_attention.py, and inside the rollout cache on
real VecEnv rollouts (tests/test_history_rollout_paged_kernel.py: same
choices, learner log-probabilities within 1e-6); ahead-of-time compilation for
sm_89 with Triton 3.8 (FP32 FMA only, no MMA, no register spills; not a
launch). Unverified: compilation and launch on the cluster's torch/Triton, GPU
speed (bench/paged_attention_bench.py), the split heuristic's constants, and
Python launch overhead per layer.
"""
from __future__ import annotations

from contextlib import nullcontext

from dataclasses import dataclass
from functools import lru_cache
import inspect

import torch
from torch import Tensor
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # the reference path stays usable without Triton
    triton = tl = None

LOG2E = 1.4426950408889634
PROGRAMS_PER_SM = 4         # split keys until B*H*tiles*splits reaches this per SM (to be tuned)
MIN_PAGES_PER_SPLIT = 4     # a split shorter than this costs more in the combine than it saves
MAX_SPLITS = 16
COMBINE_BLOCK_W = 16


# ---- reference (today's path) ------------------------------------------------------

def _rows(table: Tensor, ends: Tensor, positions: Tensor, page_tokens: int) -> Tensor:
    """``PagedHistoryCache._rows``: pool row of every position, row 0 at or past ``ends``."""
    page = table[:, positions // page_tokens]
    rows = page * page_tokens + positions % page_tokens
    return torch.where(positions[None] < ends[:, None], rows, torch.zeros_like(rows))


def paged_append_attention_reference(q: Tensor, kv: Tensor, table: Tensor, starts: Tensor,
                                     ends: Tensor, page_tokens: int, *,
                                     span: int | None = None) -> Tensor:
    """Gather + masked SDPA exactly as ``PagedHistoryCache._append`` does it.

    ``span`` is the number of key positions to materialize (``_append`` uses its
    ``_span``); the default is the whole table, ``table.shape[1] * page_tokens``.
    """
    B, H, W, D = _check(q, kv, table, starts, ends, page_tokens)
    if span is None:
        span = table.shape[1] * page_tokens
    positions = torch.arange(span, device=q.device)
    rows = _rows(table, ends, positions, page_tokens).view(-1)
    packed = torch.index_select(kv, 0, rows).view(B, span, 2, H, D).permute(2, 0, 3, 1, 4)
    query = starts[:, None] + torch.arange(W, device=q.device)
    allowed = ((positions[None, None] <= query[:, :, None])
               & (positions[None, None] < ends[:, None, None]))
    return F.scaled_dot_product_attention(q, packed[0], packed[1], attn_mask=allowed[:, None],
                                          dropout_p=0.0)


def _check(q: Tensor, kv: Tensor, table: Tensor, starts: Tensor, ends: Tensor,
           page_tokens: int) -> tuple[int, int, int, int]:
    if q.dim() != 4 or kv.dim() != 4 or table.dim() != 2:
        raise ValueError("q [B, H, W, D], kv [rows, 2, H, D] and table [B, blocks] required")
    B, H, W, D = q.shape
    if kv.shape[1:] != (2, H, D) or table.shape[0] != B or table.shape[1] < 1:
        raise ValueError(f"kv {tuple(kv.shape)} / table {tuple(table.shape)} do not match q {tuple(q.shape)}")
    if starts.shape != (B,) or ends.shape != (B,):
        raise ValueError("starts and ends must be [B]")
    if page_tokens < 1 or kv.shape[0] % page_tokens:
        raise ValueError("kv rows must be whole pages")
    return B, H, W, D


# ---- Triton kernels ----------------------------------------------------------------

if triton is not None:
    @triton.jit(do_not_specialize=["W", "BLOCKS", "PAGES_PER_SPLIT"])
    def _paged_attention_kernel(
            Q, KV, TABLE, STARTS, ENDS, OUT, PART_ACC, PART_ML,
            sq_b, sq_h, sq_w, sq_d,
            skv_row, skv_kv, skv_h, skv_d,
            st_b, st_p,
            so_b, so_h, so_w, so_d,
            W, BLOCKS, PAGES_PER_SPLIT, qk_scale,
            H: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, BLOCK_Q: tl.constexpr,
            SPLITS: tl.constexpr, IEEE: tl.constexpr):
        # Heads vary fastest in launch order, so the H programs reading one
        # stream's 1 KB K/V rows (64 B per head) tend to run together.
        h = tl.program_id(0)
        b = tl.program_id(1).to(tl.int64)
        tile = tl.program_id(2) // SPLITS
        split = tl.program_id(2) % SPLITS
        start = tl.load(STARTS + b)
        end = tl.load(ENDS + b)
        qi = tile * BLOCK_Q + tl.arange(0, BLOCK_Q)
        d = tl.arange(0, D)
        j = tl.arange(0, PAGE)
        q_live = qi < W
        q = tl.load(Q + b * sq_b + h * sq_h + qi[:, None] * sq_w + d[None, :] * sq_d,
                    mask=q_live[:, None], other=0.0)
        q_pos = start + qi
        # Keys this tile can see: up to its last real row's position, and below
        # end (padded rows past count see every key < end).
        key_end = tl.minimum(end, start + tl.minimum((tile + 1) * BLOCK_Q, W))
        pages = tl.minimum((key_end + PAGE - 1) // PAGE, BLOCKS).to(tl.int32)
        first = split * PAGES_PER_SPLIT
        last = tl.minimum(first + PAGES_PER_SPLIT, pages)
        m_i = tl.full([BLOCK_Q], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_Q], tl.float32)
        acc = tl.zeros([BLOCK_Q, D], tl.float32)
        for index in range(first, last):
            page = tl.load(TABLE + b * st_b + index * st_p).to(tl.int64)
            key = index * PAGE + j
            key_ok = key < end
            base = KV + (page * PAGE + j)[:, None] * skv_row + h * skv_h + d[None, :] * skv_d
            k = tl.load(base, mask=key_ok[:, None], other=0.0)
            v = tl.load(base + skv_kv, mask=key_ok[:, None], other=0.0)
            if IEEE:
                s = tl.dot(q, tl.trans(k), input_precision="ieee")
            else:
                s = tl.dot(q, tl.trans(k), allow_tf32=False)
            visible = (key[None, :] <= q_pos[:, None]) & key_ok[None, :]
            s = tl.where(visible, s * qk_scale, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            # A row with no visible key yet keeps m = -inf; subtracting 0 instead
            # keeps every exp2 argument -inf or finite (never -inf - -inf).
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp2(m_i - m_safe)
            p = tl.exp2(s - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            if IEEE:
                pv = tl.dot(p, v, input_precision="ieee")
            else:
                pv = tl.dot(p, v, allow_tf32=False)
            acc = acc * alpha[:, None] + pv
            m_i = m_new
        if SPLITS == 1:
            out = acc / tl.where(l_i > 0.0, l_i, 1.0)[:, None]
            tl.store(OUT + b * so_b + h * so_h + qi[:, None] * so_w + d[None, :] * so_d, out,
                     mask=q_live[:, None])
        else:
            part = ((b * H + h) * SPLITS + split) * W + qi        # [B, H, SPLITS, W]
            tl.store(PART_ACC + part[:, None] * D + d[None, :], acc, mask=q_live[:, None])
            tl.store(PART_ML + part * 2, m_i, mask=q_live)
            tl.store(PART_ML + part * 2 + 1, l_i, mask=q_live)

    @triton.jit(do_not_specialize=["W"])
    def _combine_kernel(
            PART_ACC, PART_ML, OUT,
            so_b, so_h, so_w, so_d, W,
            H: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr, SPLITS_P2: tl.constexpr,
            BLOCK_W: tl.constexpr):
        bh = tl.program_id(0).to(tl.int64)
        b = bh // H
        h = bh % H
        wi = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
        s = tl.arange(0, SPLITS_P2)
        d = tl.arange(0, D)
        live = (s[:, None] < SPLITS) & (wi[None, :] < W)
        part = (bh * SPLITS + s[:, None]) * W + wi[None, :]     # [SPLITS_P2, BLOCK_W]
        m = tl.load(PART_ML + part * 2, mask=live, other=float("-inf"))
        l = tl.load(PART_ML + part * 2 + 1, mask=live, other=0.0)
        m_max = tl.max(m, 0)
        m_safe = tl.where(m_max == float("-inf"), 0.0, m_max)
        weight = tl.exp2(m - m_safe[None, :])                   # empty split: exp2(-inf) = 0
        denom = tl.sum(weight * l, 0)
        acc = tl.load(PART_ACC + part[:, :, None] * D + d[None, None, :], mask=live[:, :, None],
                      other=0.0)
        out = tl.sum(weight[:, :, None] * acc, 0) / tl.where(denom > 0.0, denom, 1.0)[:, None]
        tl.store(OUT + b * so_b + h * so_h + wi[:, None] * so_w + d[None, :] * so_d, out,
                 mask=(wi < W)[:, None])


def _interpreted(function) -> bool:
    return type(function).__name__ == "InterpretedFunction"


# Triton decorates (its own library and these kernels) for the interpreter only
# when TRITON_INTERPRET was set at import; a mix of modes cannot run.
INTERPRETED = triton is not None and _interpreted(_paged_attention_kernel)
KERNEL_AVAILABLE = triton is not None and INTERPRETED == _interpreted(getattr(tl, "zeros", None))


@lru_cache(maxsize=None)
def _ieee_dot() -> bool:
    """``tl.dot(input_precision="ieee")`` (Triton >= 3.0) or ``allow_tf32=False``."""
    try:
        return "input_precision" in inspect.signature(tl.dot).parameters
    except (TypeError, ValueError):
        return True


@lru_cache(maxsize=None)
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


# ---- launch configuration and dispatch --------------------------------------------

@dataclass(frozen=True)
class LaunchConfig:
    block_q: int
    splits: int
    pages_per_split: int
    num_warps: int


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def launch_config(batch: int, heads: int, width: int, pages: int, *,
                  sm_count: int | None = None, splits: int | None = None,
                  block_q: int | None = None, num_warps: int | None = None) -> LaunchConfig:
    """Query tile, key splits and warps for ``pages`` key pages per stream.

    Without ``splits``, keys are split only when ``batch * heads * tiles``
    programs would leave fewer than ``PROGRAMS_PER_SM`` per SM, and never below
    ``MIN_PAGES_PER_SPLIT`` pages per split. Trailing empty splits are removed.
    No ``sm_count`` (CPU interpreter) means one split unless forced.
    """
    if min(batch, heads, width, pages) < 1:
        raise ValueError("positive launch dimensions required")
    if block_q is None:
        block_q = 16 if width <= 16 else 32 if width <= 32 else 64
    if block_q < 16 or block_q & (block_q - 1):
        raise ValueError("block_q must be a power of two >= 16")
    if splits is None:
        programs = batch * heads * _cdiv(width, block_q)
        target = PROGRAMS_PER_SM * sm_count if sm_count else 0
        splits = 1
        if programs < target:
            splits = min(_cdiv(target, programs), max(pages // MIN_PAGES_PER_SPLIT, 1), MAX_SPLITS)
    splits = max(1, min(int(splits), pages))
    per_split = _cdiv(pages, splits)
    return LaunchConfig(block_q, _cdiv(pages, per_split), per_split,
                        4 if num_warps is None else int(num_warps))


def kernel_supported(device: str | torch.device, dtype: torch.dtype, depth: int,
                     page_tokens: int) -> bool:
    """The static part of ``kernel_eligible``: Triton usable for ``device``
    (CUDA, or anything under the interpreter), FP32, and a head depth and
    page size the kernel tiles: powers of two, depth in [16, 64], page size in
    [16, 128] and depth x page size at most 4096, so a 64-row query tile's K/V
    page tiles stay well inside sm_89's per-block shared memory (production:
    depth 16, 64-token pages). Larger shapes take the reference path."""
    if not KERNEL_AVAILABLE or dtype != torch.float32:
        return False
    if torch.device(device).type != "cuda" and not INTERPRETED:
        return False
    if any(n < 16 or n & (n - 1) for n in (depth, page_tokens)):
        return False
    return depth <= 64 and page_tokens <= 128 and depth * page_tokens <= 4096


def kernel_eligible(q: Tensor, kv: Tensor, table: Tensor, starts: Tensor, ends: Tensor,
                    page_tokens: int) -> bool:
    """Whether ``paged_append_attention`` would run the Triton kernel."""
    if kv.dtype != q.dtype or not kernel_supported(q.device, q.dtype, q.shape[-1], page_tokens):
        return False
    if any(t.device != q.device for t in (kv, table, starts, ends)):
        return False
    return all(t.dtype in (torch.int32, torch.int64) for t in (table, starts, ends))


def _check_inference(q: Tensor, kv: Tensor) -> None:
    if torch.is_grad_enabled() and (q.requires_grad or kv.requires_grad):
        raise RuntimeError("paged append attention is inference-only (no backward)")


def paged_append_attention_triton(q: Tensor, kv: Tensor, table: Tensor, starts: Tensor,
                                  ends: Tensor, page_tokens: int, *, max_end: int | None = None,
                                  splits: int | None = None, block_q: int | None = None,
                                  num_warps: int | None = None) -> Tensor:
    """The Triton kernel (see the module notes); no eligibility fallback.

    ``max_end`` is an optional host-side bound on ``ends`` (``_append`` knows it)
    that lets the split heuristic ignore table columns no stream reaches; it
    never changes which keys are read.
    """
    if not KERNEL_AVAILABLE:
        raise RuntimeError("Triton is unavailable or was imported in a different interpreter mode")
    _check_inference(q, kv)
    B, H, W, D = _check(q, kv, table, starts, ends, page_tokens)
    # The kernel indexes starts/ends with unit stride (views of a wider tensor
    # would otherwise be read at the wrong elements).
    starts, ends = starts.contiguous(), ends.contiguous()
    out = torch.empty(B, W, H, D, device=q.device, dtype=q.dtype).transpose(1, 2)
    if B == 0 or W == 0:
        return out
    blocks = table.shape[1]
    pages = blocks if max_end is None else max(1, min(blocks, _cdiv(int(max_end), page_tokens)))
    sm_count = None
    if q.device.type == "cuda" and not INTERPRETED:
        index = q.device.index if q.device.index is not None else torch.cuda.current_device()
        sm_count = _sm_count(index)
    config = launch_config(B, H, W, pages, sm_count=sm_count, splits=splits, block_q=block_q,
                           num_warps=num_warps)
    S = config.splits
    if S > 1:
        part_acc = torch.empty(B, H, S, W, D, device=q.device, dtype=torch.float32)
        part_ml = torch.empty(B, H, S, W, 2, device=q.device, dtype=torch.float32)
    else:
        part_acc = part_ml = out     # unused: the SPLITS == 1 branch stores straight to out
    grid = (H, B, _cdiv(W, config.block_q) * S)
    # Triton launches on the current CUDA device and its current stream; make
    # that the tensors' device, so a non-default device (cuda:1 without
    # set_device) is neither a wrong-context launch nor unordered with the
    # stream that wrote the pool.
    with (torch.cuda.device(q.device) if q.device.type == "cuda" else nullcontext()):
        _paged_attention_kernel[grid](
            q, kv, table, starts, ends, out, part_acc, part_ml,
            *q.stride(), *kv.stride(), *table.stride(), *out.stride(),
            W, blocks, config.pages_per_split, D ** -0.5 * LOG2E,
            H=H, D=D, PAGE=page_tokens, BLOCK_Q=config.block_q, SPLITS=S, IEEE=_ieee_dot(),
            num_warps=config.num_warps)
        if S > 1:
            _combine_kernel[(B * H, _cdiv(W, COMBINE_BLOCK_W))](
                part_acc, part_ml, out, *out.stride(), W,
                H=H, D=D, SPLITS=S, SPLITS_P2=triton.next_power_of_2(S),
                BLOCK_W=COMBINE_BLOCK_W, num_warps=4)
    return out


def paged_append_attention(q: Tensor, kv: Tensor, table: Tensor, starts: Tensor, ends: Tensor,
                           page_tokens: int, *, span: int | None = None,
                           max_end: int | None = None, splits: int | None = None,
                           block_q: int | None = None, num_warps: int | None = None) -> Tensor:
    """Masked append attention over paged K/V: ``q [B, H, W, D]`` -> ``[B, H, W, D]``.

    Runs the Triton kernel when ``kernel_eligible`` (CUDA, or the interpreter
    under ``TRITON_INTERPRET=1``), else ``paged_append_attention_reference``
    with ``span``. ``max_end``/``splits``/``block_q``/``num_warps`` only tune
    the kernel launch. Inference only.
    """
    _check_inference(q, kv)
    if kernel_eligible(q, kv, table, starts, ends, page_tokens):
        return paged_append_attention_triton(q, kv, table, starts, ends, page_tokens,
                                             max_end=max_end, splits=splits, block_q=block_q,
                                             num_warps=num_warps)
    return paged_append_attention_reference(q, kv, table, starts, ends, page_tokens, span=span)
