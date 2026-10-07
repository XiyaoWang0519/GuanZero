"""Candidate-scoring Q critic warm-started from the trained V critic.

``Q(s, c) = V_critic(s) + <P f(s), g(c)>`` where ``f`` is the V critic's tower
(copied, then fine-tuned), ``V_critic`` its scalar head, ``P`` a linear map and
``g`` a candidate MLP whose last layer starts at zero. At initialisation
``Q(s, c) = V(s)`` for every candidate, so Q-boosting with this critic starts
exactly at GAE with the V critic and departs from it only as the candidate
term is learned. Rows are ragged: ``counts[i]`` candidates per decision,
``cand`` their stacked features.
"""
from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from train.history_model import HistoryCritic


class QCritic(nn.Module):
    def __init__(self, critic: HistoryCritic, act_dim: int, width: int | None = None) -> None:
        super().__init__()
        width = width or critic.config.critic_width
        self.tower = copy.deepcopy(critic.tower)
        self.head = copy.deepcopy(critic.head)
        self.project = nn.Linear(critic.config.critic_width, width)
        self.candidate = nn.Sequential(nn.Linear(act_dim, width), nn.ReLU(), nn.Linear(width, width))
        nn.init.zeros_(self.candidate[2].weight)
        nn.init.zeros_(self.candidate[2].bias)
        self.scale = width ** -0.5

    def state_features(self, obs: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """``(base value [n], projected feature [n, width])``."""
        state = torch.cat((obs.float(), hidden.float().reshape(len(obs), -1)), dim=-1)
        feature = self.tower(state)
        return self.head(feature).squeeze(-1), self.project(feature)

    def forward_chosen(self, obs: Tensor, hidden: Tensor, chosen_cand: Tensor) -> Tensor:
        """Q of one candidate per decision, ``chosen_cand [n, act_dim]``."""
        base, feature = self.state_features(obs, hidden)
        return base + (feature * self.candidate(chosen_cand.float())).sum(-1) * self.scale

    def forward_all(self, obs: Tensor, hidden: Tensor, cand: Tensor, counts: Tensor) -> Tensor:
        """Q of every candidate, flat in decision order (``counts`` per decision)."""
        base, feature = self.state_features(obs, hidden)
        rows = torch.repeat_interleave(torch.arange(len(counts), device=counts.device), counts)
        return base[rows] + (feature[rows] * self.candidate(cand.float())).sum(-1) * self.scale


def expected_value(q_flat: Tensor, probs_flat: Tensor, counts: Tensor) -> Tensor:
    """``sum_c pi(c) Q(c)`` per decision from flat candidate arrays."""
    rows = torch.repeat_interleave(torch.arange(len(counts), device=counts.device), counts)
    return torch.zeros(len(counts), dtype=q_flat.dtype, device=q_flat.device).index_add_(
        0, rows, q_flat * probs_flat)
