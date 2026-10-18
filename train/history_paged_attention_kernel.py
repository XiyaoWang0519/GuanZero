"""FP32 paged causal attention for short incremental query chunks.

Follows the page table directly and maintains an online softmax; no dense
K/V or global attention mask. Both dot products request IEEE FP32, with no
TF32 or FP16/BF16 probability conversion. Learner backward is untouched.
"""
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from train.history_paged_attention import PagedAttentionLayout


@triton.jit(do_not_specialize=[
    "QWIDTH", "QUERY_BLOCKS", "TABLE_COLS", "POOL_ROWS", "SQ0", "SQ1", "SQ2", "SQ3",
])
def _paged_forward(
    Q, KV, TABLE, STARTS, ENDS, OUT,
    QWIDTH, QUERY_BLOCKS, TABLE_COLS, POOL_ROWS, SQ0, SQ1, SQ2, SQ3,
    HEADS: tl.constexpr, DEPTH: tl.constexpr, PAGE: tl.constexpr,
    SCALE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    program = tl.program_id(0)
    tile = program % QUERY_BLOCKS
    batch_head = program // QUERY_BLOCKS
    batch = (batch_head // HEADS).to(tl.int64)
    head = (batch_head % HEADS).to(tl.int64)
    start = tl.load(STARTS + batch)
    end = tl.load(ENDS + batch)
    m = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, DEPTH)
    live_q = (m < QWIDTH) & (m < end - start)
    q = tl.load(Q + batch * SQ0 + head * SQ1 + m[:, None].to(tl.int64) * SQ2 + d[None, :].to(tl.int64) * SQ3,
                mask=live_q[:, None], other=0.0)
    maximum = tl.where(live_q, float("-inf"), 0.0)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    numerator = tl.zeros((BLOCK_M, DEPTH), tl.float32)
    # An entire tile beyond the valid query count has nothing to read.
    limit = tl.where(tile * BLOCK_M < end - start,
                     tl.minimum(end, start + (tile + 1) * BLOCK_M), 0)
    for key_start in range(0, limit, BLOCK_N):
        n = key_start + tl.arange(0, BLOCK_N)
        page_id = tl.load(TABLE + batch * TABLE_COLS + n // PAGE,
                          mask=n < limit, other=0)
        live_k = (n < limit) & (page_id > 0) & (page_id < POOL_ROWS // PAGE)
        row = page_id.to(tl.int64) * PAGE + n % PAGE
        offset = (row[:, None] * 2 * HEADS + head) * DEPTH + d[None, :]
        k = tl.load(KV + offset, mask=live_k[:, None], other=0.0)
        v = tl.load(KV + offset + HEADS * DEPTH, mask=live_k[:, None], other=0.0)
        score = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE
        visible = live_q[:, None] & live_k[None, :] & (n[None, :] <= start + m[:, None])
        score = tl.where(visible, score, float("-inf"))
        next_max = tl.where(live_q, tl.maximum(maximum, tl.max(score, axis=1)), 0.0)
        correction = tl.exp(maximum - next_max)
        probability = tl.exp(score - next_max[:, None])
        numerator = numerator * correction[:, None] + tl.dot(
            probability, v, input_precision="ieee")
        denominator = denominator * correction + tl.sum(probability, axis=1)
        maximum = next_max
    divisor = tl.where(live_q, denominator, 1.0)
    result = numerator / divisor[:, None]
    out_offset = ((batch * HEADS + head) * QWIDTH + m[:, None]) * DEPTH + d[None, :]
    tl.store(OUT + out_offset, result, mask=m[:, None] < QWIDTH)


def launch(q: torch.Tensor, kv: torch.Tensor, layout: "PagedAttentionLayout") -> torch.Tensor:
    """Internal launcher. The checked public wrapper owns dtype/layout validation."""
    batch, heads, width, depth = q.shape
    result = torch.empty((batch, heads, width, depth), dtype=q.dtype, device=q.device)
    query_blocks = triton.cdiv(width, 16)
    # Set the active CUDA device to match the operands, including under MPS.
    with torch.cuda.device(q.device):
        _paged_forward[(batch * heads * query_blocks,)](
            q, kv, layout.table, layout.starts, layout.ends, result,
            width, query_blocks, layout.table.shape[1], kv.shape[0], *q.stride(),
            HEADS=heads, DEPTH=depth, PAGE=layout.page_tokens,
            SCALE=depth ** -0.5, BLOCK_M=16, BLOCK_N=64,
            num_warps=4, num_stages=1, enable_fp_fusion=False,
        )
    return result
