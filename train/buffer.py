"""Bounded replay of completed rounds; unfinished rounds never acquire targets."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Decision:
    obs: np.ndarray
    action: np.ndarray
    hidden: np.ndarray
    seat: int
    phase: int
    prefix: int = 0
    # Schema 3 only: the canonical candidate set the actor chose from, the
    # abstract id of every candidate and the index of the chosen one.
    cand: np.ndarray | None = None
    cand_abstract: np.ndarray | None = None
    choice: int = -1


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        # The v1 encoder contains only binary planes; uint8 avoids fourfold RAM use.
        self.arrays = {
            "obs": np.empty((capacity, obs_dim), np.uint8),
            "action": np.empty((capacity, act_dim), np.uint8),
            "hidden": np.empty((capacity, 3, 54), np.uint8),
            "phase": np.empty(capacity, np.int64),
            "returns": np.empty(capacity, np.float32),
            "finish": np.empty(capacity, np.int64),
        }
        self.size = 0
        self.cursor = 0

    def __len__(self) -> int:
        return self.size

    def add_round(self, decisions: list[Decision], seat_returns: list[int],
                  order: list[int]) -> None:
        positions = {seat: position for position, seat in enumerate(order)}
        if sorted(positions) != [0, 1, 2, 3]:
            raise ValueError("round result must rank all four seats")
        for item in decisions:
            i = self.cursor
            self.arrays["obs"][i] = item.obs
            self.arrays["action"][i] = item.action
            self.arrays["hidden"][i] = item.hidden
            self.arrays["phase"][i] = item.phase
            self.arrays["returns"][i] = seat_returns[item.seat]
            self.arrays["finish"][i] = positions[item.seat]
            self.cursor = (i + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, count: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if not self.size or count <= 0:
            raise ValueError("sampling requires data and a positive count")
        indices = rng.integers(self.size, size=count)
        return {key: values[indices] for key, values in self.arrays.items()}

    def clear(self) -> None:
        self.size = self.cursor = 0
