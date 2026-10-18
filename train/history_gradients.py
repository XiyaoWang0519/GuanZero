"""Reuse actor gradient reductions for clipping and encoder diagnostics."""
from __future__ import annotations

from collections.abc import Iterable

import torch


class ActorGradientClipper:
    """Clip an actor and retain its encoder's pre-clip squared tensor norms.

    The FP32 CPU/CUDA path computes each actor tensor's norm once. Encoder
    diagnostics reuse those norms in the encoder's parameter order; the caller
    still adds the squared FP32 values in host double precision. PyTorch's
    public ``clip_grads_with_norm_`` (available in our PyTorch 2.9 minimum)
    applies the same clipping operation as ``clip_grad_norm_``.

    Other dtypes/devices retain the original operations. In particular, the
    encoder metric's FP32 conversion must not change the actor's clipping
    dtype. Missing gradients are looked up each call, so DDP gradient views
    and ``zero_grad(set_to_none=True)`` need no cached tensor state.
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter],
                 encoder_parameters: Iterable[torch.nn.Parameter]) -> None:
        self.parameters = tuple(parameters)
        self.encoder_parameters = tuple(encoder_parameters)
        parameter_ids = {id(p) for p in self.parameters}
        if any(id(p) not in parameter_ids for p in self.encoder_parameters):
            raise ValueError("encoder parameters must belong to the actor")
        self.reuse_norms = bool(self.parameters) and all(
            p.dtype == torch.float32 and p.device == self.parameters[0].device
            and p.device.type in ("cpu", "cuda") for p in self.parameters)

    @torch.no_grad()
    def __call__(self, max_norm: float) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if not self.reuse_norms:
            grads = [p.grad.detach().float() for p in self.encoder_parameters
                     if p.grad is not None]
            encoder = (torch.stack(torch._foreach_norm(grads)) ** 2).unbind() if grads else ()
            return torch.nn.utils.clip_grad_norm_(self.parameters, max_norm), encoder

        active = [p for p in self.parameters if p.grad is not None]
        if not active:
            # Match clip_grad_norm_'s CPU scalar for an actor without gradients.
            return torch.tensor(0.0), ()
        norms = torch._foreach_norm([p.grad for p in active], 2.0)
        total = torch.linalg.vector_norm(torch.stack(norms), 2.0)
        by_parameter = {id(p): norm for p, norm in zip(active, norms)}
        encoder_norms = [by_parameter[id(p)] for p in self.encoder_parameters if p.grad is not None]
        encoder = (torch.stack(encoder_norms) ** 2).unbind() if encoder_norms else ()
        torch.nn.utils.clip_grads_with_norm_(self.parameters, max_norm, total)
        return total, encoder
