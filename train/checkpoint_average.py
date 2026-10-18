"""Uniform average of N history checkpoints of one lineage, for evaluation.

    python -m train.checkpoint_average --output avg4.pt u85312.pt u85808.pt u86304.pt u86800.pt

Every actor and critic tensor is the plain mean of the inputs' tensors,
accumulated in float64 and stored in the input dtype. The inputs must be
``history_ppo`` checkpoints of the same lineage and architecture. The result
is a ``history_ppo`` checkpoint that ``eval.policies.load_policy`` loads like
any other; it carries no optimizer, RNG or population state, records its
inputs under ``averaged_from``, and ``train.history_ppo`` refuses to resume it.
The last input supplies the remaining metadata (config, progress, identity).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from eval.policies import model_digest
from infra.history_artifacts import sha256
from train.history_model import load_history_checkpoint, save_history_checkpoint

# Per-input training state that does not survive averaging.
DROPPED_KEYS = ("population", "habit_init", "schema_version")


def average_state_dicts(states: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Elementwise uniform mean of floating tensors; other tensors must be equal."""
    if len(states) < 2:
        raise ValueError("averaging needs at least two state dicts")
    keys = list(states[0])
    for state in states[1:]:
        if list(state) != keys:
            raise ValueError("state dicts have different parameter names")
    averaged = {}
    for key in keys:
        first = states[0][key]
        for state in states[1:]:
            if state[key].shape != first.shape or state[key].dtype != first.dtype:
                raise ValueError(f"{key}: shape or dtype differs between inputs")
        if first.is_floating_point():
            total = torch.zeros(first.shape, dtype=torch.float64)
            for state in states:
                total += state[key].detach().cpu().to(torch.float64)
            averaged[key] = (total / len(states)).to(first.dtype)
        else:
            if any(not torch.equal(state[key].cpu(), first.cpu()) for state in states[1:]):
                raise ValueError(f"{key}: non-floating tensor differs between inputs")
            averaged[key] = first.detach().cpu().clone()
    return averaged


def average_checkpoints(paths: Sequence[Path], output: Path) -> dict:
    """Write the uniform average of ``paths`` to ``output``; returns its record."""
    paths = [Path(p) for p in paths]
    if len(paths) < 2:
        raise ValueError("averaging needs at least two checkpoints")
    digests = [sha256(p) for p in paths]
    if len(set(digests)) != len(digests):
        raise ValueError("the same checkpoint file is listed twice")
    payloads = [load_history_checkpoint(p, "cpu")[2] for p in paths]
    reference = payloads[-1]
    for path, payload in zip(paths, payloads):
        if payload["model_config"] != reference["model_config"]:
            raise ValueError(f"{path}: architecture differs from {paths[-1]}")
        if payload["lineage"] != reference["lineage"]:
            raise ValueError(f"{path}: lineage {payload['lineage']!r} differs from "
                             f"{reference['lineage']!r}")
        if payload.get("averaged_from"):
            raise ValueError(f"{path} is already an average")
    inputs = [dict(path=str(path), sha256=digest, model_digest=model_digest(payload["model"]),
                   updates=payload["progress"].get("updates"),
                   decisions=payload["progress"].get("decisions"))
              for path, digest, payload in zip(paths, digests, payloads)]
    averaged = {key: value for key, value in reference.items() if key not in DROPPED_KEYS}
    averaged.update(model=average_state_dicts([p["model"] for p in payloads]),
                    critic=average_state_dicts([p["critic"] for p in payloads]),
                    optimizer={}, rng={},
                    averaged_from=dict(method="uniform", count=len(paths), inputs=inputs))
    save_history_checkpoint(output, averaged)
    return dict(output=str(output), sha256=sha256(output),
                model_digest=model_digest(averaged["model"]), lineage=reference["lineage"],
                **averaged["averaged_from"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(average_checkpoints(args.checkpoints, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
