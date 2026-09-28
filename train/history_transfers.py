"""Lossless packing for the small, mixed-dtype history inference transfers."""
import os
from typing import Sequence

import numpy as np
import torch


# Read once so the experiment can switch every existing caller without adding
# an environment lookup to each upload. The default transfer stays synchronous.
_PINNED_CUDA_UPLOAD = os.environ.get("GUANZERO_PINNED_UPLOAD", "0") == "1"


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
    tensors = tuple(torch.from_numpy(np.asarray(array)) for array in arrays)
    if not packed:
        return tuple(t.to(device) for t in tensors)
    if not tensors:
        return ()
    layout, total = [], 0
    for tensor in tensors:
        start = (total + 7) // 8 * 8
        total = start + tensor.numel() * tensor.element_size()
        layout.append((start, total))
    asynchronous = device.type == "cuda" and (_PINNED_CUDA_UPLOAD if pinned is None else pinned)
    host = (torch.zeros(total, dtype=torch.uint8, pin_memory=True) if asynchronous
            else torch.zeros(total, dtype=torch.uint8))
    for tensor, (start, end) in zip(tensors, layout):
        host[start:end].copy_(tensor.contiguous().reshape(-1).view(torch.uint8))
    # The local staging tensor is never mutated after enqueue. PyTorch's pinned
    # allocator records the async copy's stream event before recycling storage,
    # so its Python reference can expire here without a custom host-buffer pool.
    uploaded = host.to(device, non_blocking=True) if asynchronous else host.to(device)
    return tuple(uploaded[start:end].view(tensor.dtype).reshape(tensor.shape)
                 for tensor, (start, end) in zip(tensors, layout))


def download_tensors(tensors: Sequence[torch.Tensor], *, packed: bool = True
                     ) -> tuple[np.ndarray, ...]:
    """Copy mixed-dtype results with one device-to-host transfer, bit for bit."""
    if not packed:
        return tuple(t.detach().cpu().numpy() for t in tensors)
    if not tensors:
        return ()
    dtypes = {torch.int64: np.int64, torch.float32: np.float32,
              torch.float64: np.float64, torch.bool: np.bool_}
    pieces = [t.detach().contiguous().reshape(-1).view(torch.uint8) for t in tensors]
    data = torch.cat(pieces).cpu().numpy()
    result, start = [], 0
    for tensor, piece in zip(tensors, pieces):
        end = start + piece.numel()
        result.append(data[start:end].view(dtypes[tensor.dtype]).reshape(tuple(tensor.shape)))
        start = end
    return tuple(result)
