"""Causal SDPA using the existing history actor's pre-norm layer weights.

Trailing padding cannot affect valid positions in a causal stream. Passing the
causal hint directly avoids materializing a batch/head/length/length padding
mask. This changes neither the checkpoint tensors nor the attention support.
"""
from torch import Tensor
from torch.nn import functional as F


def project(layer, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    attention = layer.self_attn
    q, k, v = F.linear(layer.norm1(state), attention.in_proj_weight,
                       attention.in_proj_bias).chunk(3, dim=-1)
    shape = (*state.shape[:2], attention.num_heads, attention.head_dim)
    return tuple(x.reshape(shape).transpose(1, 2) for x in (q, k, v))


def finish(layer, state: Tensor, attended: Tensor) -> Tensor:
    attended = attended.transpose(1, 2).reshape_as(state)
    state = state + F.linear(attended, layer.self_attn.out_proj.weight,
                             layer.self_attn.out_proj.bias)
    hidden = layer.norm2(state)
    return state + layer.linear2(layer.activation(layer.linear1(hidden)))


def validate(stream) -> None:
    for layer in stream.layers:
        attention = layer.self_attn
        if (not layer.norm_first or not attention._qkv_same_embed_dim
                or attention.bias_k is not None or attention.bias_v is not None
                or attention.add_zero_attn or attention.dropout
                or layer.dropout.p or layer.dropout1.p or layer.dropout2.p):
            raise ValueError("history SDPA requires pre-norm, zero-dropout standard attention")


def causal_encode(stream, state: Tensor) -> Tensor:
    validate(stream)
    for layer in stream.layers:
        q, k, v = project(layer, state)
        state = finish(layer, state, F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True))
    return stream.norm(state) if stream.norm is not None else state
