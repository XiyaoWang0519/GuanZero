"""Lossless packing for the small, mixed-dtype history inference transfers."""
import os
from typing import Sequence

import numpy as np
import torch


# Read once so the experiment can switch every existing caller without adding
# an environment lookup to each upload. The default transfer stays synchronous.
_PINNED_CUDA_UPLOAD = os.environ.get("GUANZERO_PINNED_UPLOAD", "0") == "1"


def _byte_view(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.contiguous().reshape(-1)
    if flat.stride(0) != 1:
        # Empty/singleton tensors count as contiguous even with stride 0 or 2.
        # A dtype view still requires unit stride; changing it for at most one
        # element addresses the same storage without allocating or copying.
        flat = flat.as_strided(flat.shape, (1,))
    return flat.view(torch.uint8)


def runtime_settings(device: str | torch.device) -> dict[str, bool]:
    """Report live numerics and the effective default upload setting.

    The upload flag is captured at module import; reading the environment again
    here could report a setting different from the one used by existing callers.
    Explicit ``upload_arrays`` overrides remain local to that individual call.
    """
    return dict(pinned_upload=torch.device(device).type == "cuda" and _PINNED_CUDA_UPLOAD,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32)


def upload_arrays(arrays: Sequence[np.ndarray], device: str | torch.device, *,
                  packed: bool | None = None,
                  pinned: bool | None = None) -> tuple[torch.Tensor, ...]:
    """Upload a group with one copy, keeping each original dtype and shape.

    Every field starts at an eight-byte boundary so PyTorch's dtype views are
    aligned. CPU inference keeps the original zero-copy NumPy tensor views.
    For packed CUDA uploads, ``pinned=True`` stages directly in pinned memory
    and enqueues the copy on the current stream without a host wait. ``None``
    reads the process-start ``GUANZERO_PINNED_UPLOAD=1`` experiment setting.
    Consumers on another CUDA stream must establish the usual stream dependency.
    """
    device = torch.device(device)
    if packed is None:
        packed = device.type == "cuda"
    if not packed:
        return tuple(torch.from_numpy(np.asarray(array)).to(device) for array in arrays)
    if not arrays:
        return ()
    asynchronous = device.type == "cuda" and (_PINNED_CUDA_UPLOAD if pinned is None else pinned)
    host, layout, fields = pack_host(arrays, pinned=asynchronous)
    # The local staging tensor is never mutated after enqueue. PyTorch's pinned
    # allocator records the async copy's stream event before recycling storage,
    # so its Python reference can expire here without a custom host-buffer pool.
    uploaded = host.to(device, non_blocking=True) if asynchronous else host.to(device)
    return tuple(uploaded[start:end].view(dtype).reshape(shape)
                 for (start, end), (dtype, shape) in zip(layout, fields))


_TORCH_DTYPES: dict[np.dtype, torch.dtype] = {}


def _torch_dtype(dtype: np.dtype) -> torch.dtype:
    """``torch.from_numpy``'s dtype for ``dtype`` (raising as it does), memoized."""
    result = _TORCH_DTYPES.get(dtype)
    if result is None:
        result = torch.from_numpy(np.empty(0, dtype)).dtype
        _TORCH_DTYPES[dtype] = result
    return result


def pack_host(arrays: Sequence[np.ndarray], *, pinned: bool = False
              ) -> tuple[torch.Tensor, list[tuple[int, int]], list[tuple[torch.dtype, tuple]]]:
    """The host staging buffer of a packed ``upload_arrays`` call.

    Every array's bytes start at an eight-byte boundary, in order, with zero
    padding between: ``(host, [(start, end)], [(torch dtype, shape)])``. The
    bytes are those of the former per-field torch staging (``_byte_view``
    copies into ``torch.zeros``), padding included. The buffer is allocated
    uninitialized and only the alignment gaps (at most seven bytes each) are
    zeroed, since every other byte is overwritten; NumPy writes each field
    with one typed copy into a view of the buffer instead of several torch
    dispatches. ``pinned`` stages in page-locked memory (CUDA only).
    """
    arrays = [np.asarray(array) for array in arrays]
    fields = [(_torch_dtype(array.dtype), array.shape) for array in arrays]
    layout, total = [], 0
    for array in arrays:
        start = (total + 7) // 8 * 8
        total = start + array.nbytes
        layout.append((start, total))
    if pinned:
        host = torch.empty(total, dtype=torch.uint8, pin_memory=True)
        staging = host.numpy()
    else:
        staging = np.empty(total, np.uint8)
        host = torch.from_numpy(staging)
    previous = 0
    for array, (start, end) in zip(arrays, layout):
        if start > previous:
            staging[previous:start] = 0
        if end > start:
            staging[start:end].view(array.dtype).reshape(array.shape)[...] = array
        previous = end
    return host, layout, fields


def download_tensors(tensors: Sequence[torch.Tensor], *, packed: bool = True
                     ) -> tuple[np.ndarray, ...]:
    """Copy mixed-dtype results with one device-to-host transfer, bit for bit."""
    if not packed:
        return tuple(t.detach().cpu().numpy() for t in tensors)
    if not tensors:
        return ()
    dtypes = {torch.int64: np.int64, torch.float32: np.float32,
              torch.float64: np.float64, torch.bool: np.bool_}
    pieces = [_byte_view(t.detach()) for t in tensors]
    data = torch.cat(pieces).cpu().numpy()
    result, start = [], 0
    for tensor, piece in zip(tensors, pieces):
        end = start + piece.numel()
        result.append(data[start:end].view(dtypes[tensor.dtype]).reshape(tuple(tensor.shape)))
        start = end
    return tuple(result)
