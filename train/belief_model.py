"""Capacity-matched flat/history towers for the second offline belief gate.

The archived first probe remains in :mod:`train.belief_probe`. These towers
give the private query and attended result nonlinear capacity, as required by
DESIGN 7.2. They are an offline diagnostic, not an RL or KV-cache integration.
"""
import copy
import math

import torch
from torch import nn

from train.belief_probe import FlatBelief, count_parameters
from train.public_history import TOKEN_DIM
from train.model import mlp


class HistoryBelief(nn.Module):
    def __init__(self, obs_dim: int, width: int = 128, layers: int = 2,
                 *, no_history: bool = False) -> None:
        super().__init__()
        if obs_dim <= 0 or width <= 0 or width % 4 or layers <= 0:
            raise ValueError("positive dimensions required; width must be divisible by four")
        self.width = width
        self.no_history = no_history
        self.public = nn.Linear(TOKEN_DIM, width)
        self.bos = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(
            width, 4, width * 4, dropout=0.0, batch_first=True)
        self.stream = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.private = mlp(obs_dim, width, 2)
        self.seat = nn.Embedding(4, width)
        self.query = nn.MultiheadAttention(width, 4, dropout=0.0, batch_first=True)
        self.attention_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, width * 4), nn.ReLU(), nn.Linear(width * 4, width))
        self.output_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 3 * 54 * 3)

    def encode_public(self, tokens: torch.Tensor, lengths: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode only public prefixes; no private state or query enters here.

        Public positions are causal. The returned padding mask also hides the
        suffix from the private cross-attention. BOS remains visible for an
        empty history, including the separately trained no-history ablation.
        """
        if self.no_history:
            lengths = torch.zeros_like(lengths)
            # Avoid spending quadratic compute on masked tokens in the control.
            tokens = tokens[:, :0]
        return self.encode_stream(tokens, lengths)

    def encode_stream(self, tokens: torch.Tensor, lengths: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """Causally encode a padded batch of public token streams with BOS.

        Split out of `encode_public` so that the v3 match-memory model can run
        the same public-stream layers over the finished rounds of a match
        without going through the no-history override.
        """
        stream = torch.cat((self.bos.expand(len(tokens), -1, -1), self.public(tokens)), dim=1)
        positions = torch.arange(stream.shape[1], device=stream.device,
                                 dtype=stream.dtype)[:, None]
        frequencies = torch.exp(torch.arange(0, self.width, 2, device=stream.device,
                                             dtype=stream.dtype)
                                * (-math.log(10000.0) / self.width))
        position = torch.zeros(stream.shape[1], self.width,
                               device=stream.device, dtype=stream.dtype)
        position[:, 0::2] = torch.sin(positions * frequencies)
        position[:, 1::2] = torch.cos(positions * frequencies)
        stream = stream + position
        padding = torch.arange(stream.shape[1], device=stream.device)[None] > lengths[:, None]
        causal = torch.ones(stream.shape[1], stream.shape[1], device=stream.device,
                            dtype=torch.bool).triu(1)
        return self.stream(stream, mask=causal, src_key_padding_mask=padding), padding

    def forward(self, obs, tokens, lengths, seat):
        stream, padding = self.encode_public(tokens, lengths)
        query = (self.private(obs) + self.seat(seat))[:, None]
        attended, _ = self.query(query, stream, stream, key_padding_mask=padding,
                                 need_weights=False)
        state = self.attention_norm(query + attended)
        state = self.output_norm(state + self.feed_forward(state))
        return self.head(state[:, 0]).reshape(-1, 3, 54, 3)


def matched_models(obs_dim: int, width: int = 128, layers: int = 2
                   ) -> dict[str, nn.Module]:
    """Match flat/history parameter counts; pair the two history initializations.

    The no-history model is a separate, trainable model with identical initial
    tensors and parameter count. It cannot access any public tokens during
    training or evaluation. Equal parameters do not imply equal runtime.
    """
    history = HistoryBelief(obs_dim, width, layers)
    target = count_parameters(history)

    def flat_count(w):
        return 3 * w * w + (obs_dim + 4 + 486) * w + 486

    best = min(range(8, 4097), key=lambda w: abs(flat_count(w) - target))
    flat = FlatBelief(obs_dim, best)
    if abs(count_parameters(flat) - target) / target > 0.02:
        raise ValueError("probe parameter counts cannot be matched within 2%")
    no_history = copy.deepcopy(history)
    no_history.no_history = True
    return {"v1": flat, "v2": history, "no_history": no_history}
