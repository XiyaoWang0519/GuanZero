"""Two towers, phase-specific action values, and training-only auxiliary heads."""
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    obs_dim: int = 1849
    act_dim: int = 154
    state_width: int = 512
    state_layers: int = 4
    action_width: int = 256
    action_layers: int = 2
    fusion_width: int = 256
    fusion_layers: int = 3

    def __post_init__(self) -> None:
        if any(value <= 0 for value in self.__dict__.values()):
            raise ValueError("model dimensions and layer counts must be positive")


def mlp(inputs: int, width: int, layers: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    for _ in range(layers):
        modules.extend((nn.Linear(inputs, width), nn.ReLU()))
        inputs = width
    return nn.Sequential(*modules)


class GuandanModel(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.state_tower = mlp(config.obs_dim, config.state_width, config.state_layers)
        self.action_tower = mlp(config.act_dim, config.action_width, config.action_layers)
        self.phase_heads = nn.ModuleDict({
            str(phase): nn.Sequential(
                mlp(config.state_width + config.action_width,
                    config.fusion_width, config.fusion_layers),
                nn.Linear(config.fusion_width, 1),
            ) for phase in (1, 2, 3)
        })
        self.hidden_head = nn.Linear(config.state_width, 3 * 54 * 3)
        self.finish_head = nn.Linear(config.state_width, 4)

    def auxiliary(self, state: Tensor) -> dict[str, Tensor]:
        return {"hidden": self.hidden_head(state).reshape(-1, 3, 54, 3),
                "finish": self.finish_head(state)}

    def _fuse(self, state: Tensor, action: Tensor, phase: Tensor,
              phase_code: int | None = None) -> Tensor:
        fused = torch.cat((state, action), dim=-1)
        if phase_code is not None:
            # Stage A uses play values only. A static head avoids CUDA's
            # dynamic boolean-indexing synchronization and unnecessary heads.
            return self.phase_heads[str(phase_code)](fused).squeeze(-1)
        result = fused.new_zeros(len(fused))
        for code in (1, 2, 3):
            mask = phase == code
            # Generic mixed phases are needed by later training stages.
            result[mask] = self.phase_heads[str(code)](fused[mask]).squeeze(-1)
        return result

    def forward(self, obs: Tensor, action: Tensor, phase: Tensor,
                phase_code: int | None = None) -> dict[str, Tensor]:
        """One chosen action per observation, used by the learner."""
        state = self.state_tower(obs)
        return {"q": self._fuse(state, self.action_tower(action), phase, phase_code),
                **self.auxiliary(state)}

    def score_candidates(self, obs: Tensor, cand: Tensor, offsets: Tensor,
                         phase: Tensor, chunk_size: int = 32768,
                         phase_code: int | None = None, state: Tensor | None = None) -> Tensor:
        """Encode states once and score a ragged batch in bounded-size chunks.
        `state`, when given, is `self.state_tower(obs)` computed by the caller
        (the learner shares it with the auxiliary heads)."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if state is None:
            state = self.state_tower(obs)
        rows = torch.repeat_interleave(torch.arange(len(obs), device=obs.device),
                                       offsets[1:] - offsets[:-1],
                                       output_size=len(cand))
        if not len(cand):
            return obs.new_empty(0)
        return torch.cat([
            self._fuse(state[rows[start:start + chunk_size]],
                       self.action_tower(cand[start:start + chunk_size]),
                       phase[rows[start:start + chunk_size]], phase_code)
            for start in range(0, len(cand), chunk_size)
        ])


def select_actions(scores: Tensor, offsets: Tensor, *, epsilon: float = 0.0,
                   margin: float = 0.0, generator: torch.Generator | None = None) -> Tensor:
    """Segment argmax, or uniform near-best / epsilon exploration, without padding."""
    if not 0 <= epsilon <= 1 or margin < 0:
        raise ValueError("epsilon must be in [0,1] and margin nonnegative")
    n = len(offsets) - 1
    rows = torch.repeat_interleave(torch.arange(n, device=scores.device),
                                   offsets[1:] - offsets[:-1], output_size=len(scores))
    best = scores.new_full((n,), -torch.inf)
    best.scatter_reduce_(0, rows, scores, reduce="amax", include_self=True)
    indices = torch.arange(len(scores), device=scores.device)
    if margin > 0:
        priority = torch.rand(len(scores), device=scores.device, generator=generator)
        priority = torch.where(scores >= best[rows] - margin, priority, -1.0)
        maxima = priority.new_full((n,), -torch.inf)
        maxima.scatter_reduce_(0, rows, priority, reduce="amax", include_self=True)
        eligible = priority == maxima[rows]
    else:
        eligible = scores == best[rows]
    choices = torch.full((n,), len(scores), device=scores.device, dtype=torch.long)
    choices.scatter_reduce_(0, rows, torch.where(eligible, indices, len(scores)),
                            reduce="amin", include_self=True)
    choices -= offsets[:-1]
    if epsilon:
        explore = torch.rand(n, device=scores.device, generator=generator) < epsilon
        random_choices = (torch.rand(n, device=scores.device, generator=generator)
                          * (offsets[1:] - offsets[:-1])).long()
        choices = torch.where(explore, random_choices, choices)
    return choices
