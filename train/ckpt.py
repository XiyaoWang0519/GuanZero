"""Atomic checkpoints with the model, optimizer, schedules and all learner RNGs."""
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch

SCHEMA_VERSION = 1


def rng_state(rng: np.random.Generator) -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": rng.bit_generator.state,
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict[str, Any], rng: np.random.Generator) -> None:
    random.setstate(state["python"])
    rng.bit_generator.state = state["numpy"]
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save({**payload, "schema_version": SCHEMA_VERSION}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        # Persist the directory entry as well as the file on POSIX filesystems.
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    # Only load checkpoints from this trusted training run; optimizer/RNG state
    # requires Python objects beyond the weights-only allowlist.
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported training checkpoint schema")
    required = {"model_config", "model", "optimizer", "config", "progress", "rng"}
    if not required <= payload.keys():
        raise ValueError(f"incomplete checkpoint: {required - payload.keys()}")
    return payload
