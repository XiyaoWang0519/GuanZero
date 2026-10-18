"""Experimental FP32 paged *rollout* attention; no learner/backward replacement.

The default remains PyTorch. Explicit cache construction, or the bounded
``experimental_paged_attention`` scope, opts newly constructed caches into the
prototype. A cache retains its backend for its entire lifetime. The scope is
process-local and does not propagate into spawned DDP workers.

Layouts are append-local. Their tensors must remain unchanged until use has
finished on the current stream. ``from_host`` takes the very same host arrays
passed to the cache's existing upload, validates them without a device read,
and retains the uploaded tensors. Never reuse a layout after pool/entry changes.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import importlib.util
from typing import Iterator

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

_DEFAULT_BACKEND: ContextVar[str] = ContextVar("guanzero_paged_attention", default="torch")
DEPTHS = (16, 32, 64, 128)


def resolve_backend(backend: str | None) -> str:
    result = _DEFAULT_BACKEND.get() if backend is None else backend
    if result not in ("torch", "triton"):
        raise ValueError("paged attention backend must be 'torch' or 'triton'")
    return result


@contextmanager
def experimental_paged_attention(backend: str) -> Iterator[None]:
    """Select the backend for NEW caches in this process, restoring it on exit.

    This does not change existing caches, a checkpoint, an environment variable,
    or a production default. Build independent trainers/caches for A/B trials.
    Each spawned process must explicitly enter its own scope.
    """
    token = _DEFAULT_BACKEND.set(resolve_backend(backend))
    try:
        yield
    finally:
        _DEFAULT_BACKEND.reset(token)


def require_triton(device: torch.device, dtype: torch.dtype, depth: int) -> None:
    """Fail closed on an explicitly requested but unsupported prototype."""
    if device.type != "cuda" or torch.version.hip is not None:
        raise RuntimeError("experimental paged attention requires NVIDIA CUDA")
    if dtype != torch.float32 or depth not in DEPTHS:
        raise ValueError(f"experimental paged attention requires FP32 and depth in {DEPTHS}")
    if importlib.util.find_spec("triton") is None:
        raise RuntimeError("experimental paged attention requires the CUDA environment's Triton")


@dataclass(frozen=True)
class PagedAttentionLayout:
    table: Tensor
    starts: Tensor
    ends: Tensor
    page_tokens: int
    span: int
    query_width: int
    pool_rows: int

    @classmethod
    def from_host(cls, table: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                  table_d: Tensor, starts_d: Tensor, ends_d: Tensor, *,
                  page_tokens: int, span: int, query_width: int,
                  pool_rows: int) -> "PagedAttentionLayout":
        """Validate CPU metadata once per append, not once per layer.

        The device tensors must be unmodified uploads of the provided arrays.
        This ownership contract is satisfied by ``upload_arrays`` in the cache;
        it avoids a CUDA synchronization inside the rollout hot path.
        Use ``from_tensors`` for standalone/untrusted tensor metadata.
        """
        dimensions = (page_tokens, span, query_width, pool_rows)
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in dimensions):
            raise ValueError("positive integer layout dimensions required")
        if pool_rows % page_tokens or pool_rows < 2 * page_tokens:
            raise ValueError("pool needs complete pages including reserved page zero")
        if span >= 2**31 or query_width >= 2**31:
            raise ValueError("span and query width must fit signed int32")
        arrays = (table, starts, ends)
        if any(not isinstance(a, np.ndarray) or a.dtype != np.int64 for a in arrays):
            raise ValueError("host page metadata must be int64 NumPy arrays")
        if (table.ndim != 2 or table.shape[0] < 1
                or starts.shape != (table.shape[0],) or ends.shape != starts.shape
                or table.shape[1] != (span + page_tokens - 1) // page_tokens):
            raise ValueError("inconsistent batch/page-table shape")
        if (np.any(starts < 0) or np.any(ends < starts) or np.any(ends > span)
                or np.any(ends - starts > query_width)):
            raise ValueError("invalid append interval or context bound")
        used = np.arange(table.shape[1])[None, :] * page_tokens < ends[:, None]
        ids = table[used]
        if np.any(ids <= 0) or np.any(ids >= pool_rows // page_tokens):
            raise ValueError("live positions require allocated, in-bounds, nonzero pages")
        tensors = (table_d, starts_d, ends_d)
        if any(t.dtype != torch.int64 or t.layout != torch.strided
               or not t.is_contiguous() or tuple(t.shape) != a.shape
               or t.device != table_d.device for t, a in zip(tensors, arrays)):
            raise ValueError("uploaded metadata must match host shape/device and be contiguous int64")
        return cls(table_d, starts_d, ends_d, page_tokens, span, query_width, pool_rows)

    @classmethod
    def from_tensors(cls, table: Tensor, starts: Tensor, ends: Tensor, **kwargs
                     ) -> "PagedAttentionLayout":
        """Standalone checked entrypoint; reading CUDA metadata synchronizes.

        Construct before timed trials. Mutating these tensors invalidates the
        layout and requires fresh validation, even when the shapes do not change.
        """
        arrays = tuple(t.detach().cpu().numpy() for t in (table, starts, ends))
        return cls.from_host(*arrays, table, starts, ends, **kwargs)

    def check_operands(self, q: Tensor, kv: Tensor) -> None:
        batch = self.table.shape[0]
        if (q.ndim != 4 or kv.ndim != 4 or q.shape[0] != batch
                or q.shape[2] != self.query_width or q.shape[1] < 1 or q.shape[3] < 1
                or kv.shape != (self.pool_rows, 2, q.shape[1], q.shape[3])):
            raise ValueError("expected q[B,H,Q,D] and token-major kv[pool_rows,2,H,D]")
        if (q.layout != torch.strided or kv.layout != torch.strided
                or not kv.is_contiguous() or any(s <= 0 for s in q.stride())
                or q.device != self.table.device or kv.device != q.device
                or q.dtype != kv.dtype or q.dtype not in (torch.float32, torch.float64)):
            raise ValueError("operands require matching FP32/FP64 dtype/device and supported strides")


def dense_inputs(layout: PagedAttentionLayout) -> tuple[Tensor, Tensor]:
    """Baseline row gather and causal mask, prepared once per append."""
    positions = torch.arange(layout.span, device=layout.table.device)
    pages = layout.table[:, positions // layout.page_tokens]
    rows = pages * layout.page_tokens + positions % layout.page_tokens
    rows = torch.where(positions[None] < layout.ends[:, None], rows, 0)
    queries = torch.arange(layout.query_width, device=layout.table.device)
    allowed = ((positions[None, None] <= layout.starts[:, None, None] + queries[None, :, None])
               & (positions[None, None] < layout.ends[:, None, None]))
    return rows.reshape(-1), allowed[:, None]


def attention_reference(q: Tensor, kv: Tensor, layout: PagedAttentionLayout, *,
                        prepared: tuple[Tensor, Tensor] | None = None) -> Tensor:
    """Existing gather + SDPA arithmetic; only unused query rows are zeroed.

    FP64 is accepted as an independent numerical reference, never by the Triton
    prototype. Call under no_grad for rollout comparisons.
    """
    layout.check_operands(q, kv)
    rows, allowed = dense_inputs(layout) if prepared is None else prepared
    batch, heads, width, depth = q.shape
    packed = torch.index_select(kv, 0, rows).view(
        batch, layout.span, 2, heads, depth).permute(2, 0, 3, 1, 4)
    output = F.scaled_dot_product_attention(q, packed[0], packed[1],
                                           attn_mask=allowed, dropout_p=0.0)
    live = torch.arange(width, device=q.device)[None, :] < (layout.ends - layout.starts)[:, None]
    return torch.where(live[:, None, :, None], output, 0)


def paged_attention(q: Tensor, kv: Tensor, layout: PagedAttentionLayout) -> Tensor:
    """Launch the FP32 forward-only prototype, without silent eager fallbacks."""
    if torch.is_grad_enabled():
        raise RuntimeError("paged attention is rollout-only; enter torch.no_grad() explicitly")
    layout.check_operands(q, kv)
    require_triton(q.device, q.dtype, q.shape[-1])
    # Lazy import keeps CPU installations and default production paths unchanged.
    from train.history_paged_attention_kernel import launch
    return launch(q, kv, layout)
