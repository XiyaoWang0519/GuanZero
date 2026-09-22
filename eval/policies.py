"""A small policy interface shared by the arena and scripted probes."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import random
from typing import Protocol, Sequence

import gd
import numpy as np


class Policy(Protocol):
    name: str

    def select(self, engine: gd.Engine, state: gd.MatchState,
               actions: Sequence[gd.Action], rng: random.Random) -> int:
        """Return an index in the supplied legal candidate list."""
        ...


@dataclass
class RandomPolicy:
    name: str = "random"

    def select(self, engine: gd.Engine, state: gd.MatchState,
               actions: Sequence[gd.Action], rng: random.Random) -> int:
        # Stage A uses the same tribute heuristic for every play policy.
        if state.phase != gd.Phase.Play:
            return engine.greedy(state)
        return rng.randrange(len(actions))


@dataclass
class GreedyPolicy:
    name: str = "greedy"

    def select(self, engine: gd.Engine, state: gd.MatchState,
               actions: Sequence[gd.Action], rng: random.Random) -> int:
        return engine.greedy(state)


class StyledPolicy:
    """The C++ style-parameterised heuristic bot at one fixed style vector.

    Tribute phases go to the shared tribute heuristic inside `styled_bot`.
    The per-decision seed comes from the caller's RNG, so a style with
    temperature above 0 is still reproducible under the arena seeds.
    """

    def __init__(self, style: Sequence[float], name: str = "styled") -> None:
        vector = np.asarray(style, dtype=np.float32).reshape(-1)
        if vector.shape != (int(gd.STYLE_DIM),) or not np.isfinite(vector).all():
            raise ValueError(f"style must be {int(gd.STYLE_DIM)} finite floats")
        self.style = vector
        self.name = name

    def select(self, engine: gd.Engine, state: gd.MatchState,
               actions: Sequence[gd.Action], rng: random.Random) -> int:
        return engine.styled(state, self.style, rng.getrandbits(64))


class ModelPolicy:
    """Greedy Q inference from public observation and the acting hand only.

    Ordinary DMC checkpoints retain heuristic tribute. Only explicit Stage A2
    artifacts deploy the learned tribute heads.
    """

    def __init__(self, model: object, name: str = "checkpoint", device: str = "cpu",
                 heuristic_tribute: bool = True, margin: float = 0.0) -> None:
        import torch

        if not math.isfinite(margin) or margin < 0:
            raise ValueError("sampling margin must be nonnegative and finite")
        self.model = model.to(device).eval()
        self.name = name
        self.device = torch.device(device)
        self.heuristic_tribute = heuristic_tribute
        self.margin = margin
        self.action_mode = "canonical"

    def select(self, engine: gd.Engine, state: gd.MatchState,
               actions: Sequence[gd.Action], rng: random.Random) -> int:
        import torch

        if self.heuristic_tribute and state.phase != gd.Phase.Play:
            return engine.greedy(state)
        seat = state.to_move
        obs = np.asarray(state.observation(seat), dtype=np.float32)[None, :]
        cand = np.stack([engine.encode_action(a, state, seat) for a in actions])
        with torch.inference_mode():
            scores = self.model.score_candidates(
                torch.as_tensor(obs, device=self.device),
                torch.as_tensor(cand, device=self.device),
                torch.tensor([0, len(actions)], device=self.device),
                torch.tensor([int(state.phase)], device=self.device),
            )
            if scores.shape != (len(actions),) or not torch.isfinite(scores).all():
                raise ValueError("policy must produce one finite score per legal action")
            if self.margin > 0:
                eligible = torch.nonzero(scores >= scores.max() - self.margin).flatten().tolist()
                return rng.choice(eligible)
            return int(scores.argmax().item())


def model_digest(state_dict: dict) -> str:
    """Stable identity of all model tensors, independent of checkpoint path."""
    import torch

    digest = hashlib.sha256()
    for key, value in sorted(state_dict.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}\n".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_policy(spec: str, device: str = "cpu", margin: float = 0.0) -> Policy:
    """Load 'random', 'greedy', 'styled:<name>', or a train.ckpt checkpoint.

    `<name>` is a key of train.styles.FIXED_STYLES, for example
    'styled:bomb-happy'.
    """
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("sampling margin must be nonnegative and finite")
    if spec == "random":
        return RandomPolicy()
    if spec == "greedy":
        return GreedyPolicy()
    if spec.startswith("styled:"):
        from train.styles import fixed_style

        style_name = spec[len("styled:"):]
        return StyledPolicy(fixed_style(style_name), name=spec)
    from train.ckpt import load_checkpoint
    from train.model import GuandanModel, ModelConfig

    path = Path(spec)
    checkpoint = load_checkpoint(path, device=device)
    stage = checkpoint.get("stage", "dmc")
    tribute_policy = checkpoint.get("tribute_policy", "heuristic")
    if (stage, tribute_policy) not in (("dmc", "heuristic"), ("a2", "learned")):
        raise ValueError("unsupported checkpoint stage/tribute policy marker")
    base_id = checkpoint.get("base_checkpoint_id")
    if stage == "a2" and (not isinstance(base_id, str) or len(base_id) != 64
                          or any(char not in "0123456789abcdef" for char in base_id)):
        raise ValueError("Stage A2 requires a valid base_checkpoint_id")
    action_mode = checkpoint.get("config", {}).get("action_mode", "canonical")
    if action_mode not in ("canonical", "full"):
        raise ValueError("unsupported checkpoint action_mode")
    model = GuandanModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    # latest.pt is overwritten during training. Identify the actual loaded
    # weights so historical Elo reports never conflate different checkpoints.
    # Hashing this payload also avoids racing an atomic replacement of path.
    digest = model_digest(checkpoint["model"])
    name = f"{path.name}@{digest[:16]}"
    if tribute_policy == "learned":
        name += "/tribute=learned"
    if margin:
        name += f"/margin={margin:g}"
    policy = ModelPolicy(model, name=name, device=device, margin=margin,
                         heuristic_tribute=tribute_policy == "heuristic")
    policy.checkpoint_id = digest
    policy.training_seed = checkpoint.get("config", {}).get("seed")
    policy.action_mode = action_mode
    policy.stage = stage
    policy.base_checkpoint_id = base_id
    policy.base_training_seed = checkpoint.get("base_training_seed")
    policy.collection_seed = checkpoint.get("tribute_fit", {}).get("dataset_provenance", {}).get("seed")
    return policy


def choose_action(policy: Policy, engine: gd.Engine, state: gd.MatchState,
                  rng: random.Random) -> gd.Action:
    actions = engine.legal_actions(state)
    if not actions:
        raise RuntimeError(f"no legal actions in phase {state.phase}")
    index = policy.select(engine, state, actions, rng)
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError(f"{policy.name}: action index must be an integer")
    if not 0 <= index < len(actions):
        raise ValueError(f"{policy.name}: action index {index} outside legal set")
    return actions[index]
