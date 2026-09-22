"""Stage A2: fit only exchange heads to centered counterfactual returns.

The resulting checkpoint is an experimental policy, not a promoted replacement.
Use eval.tribute with fresh deals to measure its payoff against heuristic tribute.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import gd
import numpy as np
import torch
from torch.nn import functional as F

from eval.policies import ModelPolicy, load_policy, model_digest
from train.ckpt import load_checkpoint, rng_state, save_checkpoint
from train.tribute_data import engine_source_digest

HEAD_PREFIXES = ("phase_heads.1.", "phase_heads.2.")


def _check_time(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("A2 fitting exceeded its walltime limit")


def load_dataset(directory: Path, base_checkpoint_id: str,
                 deadline: float | None = None) -> tuple[list[dict], dict, str]:
    provenance = json.loads((directory / "provenance.json").read_text())
    if (provenance.get("purpose") != "tribute_counterfactual_training"
            or provenance.get("status") != "complete"
            or provenance.get("base_checkpoint_id") != base_checkpoint_id
            or provenance.get("play_mode") != "fp32_argmax"
            or provenance.get("other_tribute") != "heuristic"):
        raise ValueError("dataset must be complete and use this frozen base policy")
    if provenance.get("engine_source_sha256") != engine_source_digest():
        raise ValueError("dataset engine/encoder source identity differs from this checkout")
    digest = hashlib.sha256((directory / "provenance.json").read_bytes())
    cases = []
    for path in sorted(directory.glob("position-*.npz")):
        _check_time(deadline)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
        with np.load(path, allow_pickle=False) as raw:
            case = {name: raw[name].copy() for name in
                    ("group", "obs", "cand", "phase", "seat", "returns", "advantages", "heuristic_index")}
            if int(raw["schema_version"]) != 1:
                raise ValueError("unsupported tribute dataset schema")
        n = len(case["cand"])
        if (n < 2 or case["obs"].shape != (gd.OBS_DIM,)
                or case["cand"].shape != (n, gd.ACT_DIM)
                or int(case["phase"]) not in (1, 2)
                or not 0 <= int(case["seat"]) < 4
                or not 0 <= int(case["heuristic_index"]) < n
                or case["returns"].shape != (n,) or case["advantages"].shape != (n,)
                or not all(np.isfinite(case[k]).all() for k in ("obs", "cand", "returns", "advantages"))
                or not np.allclose(case["advantages"], case["returns"] - case["returns"].mean())):
            raise ValueError(f"invalid tribute position: {path.name}")
        case["group"] = str(case["group"])
        case["phase"] = int(case["phase"])
        case["heuristic_index"] = int(case["heuristic_index"])
        cases.append(case)
    if len(cases) != provenance.get("positions") or len(cases) < 2:
        raise ValueError("dataset position count does not match completed provenance")
    return cases, provenance, digest.hexdigest()


def split_matches(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    groups = sorted({c["group"] for c in cases},
                    key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    if len(groups) < 2:
        raise ValueError("tribute fitting needs at least two distinct source matches")
    heldout = set(groups[:max(1, len(groups) // 5)])
    return ([c for c in cases if c["group"] not in heldout],
            [c for c in cases if c["group"] in heldout])


def prepare_heads(model: torch.nn.Module) -> None:
    """Transfer play fusion features, start centered outputs at zero, freeze all else."""
    for code in (1, 2):
        model.phase_heads[str(code)].load_state_dict(copy.deepcopy(model.phase_heads["3"].state_dict()))
        output = model.phase_heads[str(code)][-1]
        torch.nn.init.zeros_(output.weight)
        torch.nn.init.zeros_(output.bias)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(HEAD_PREFIXES))
    model.eval()


@torch.no_grad()
def encode_cases(model, cases: list[dict], device: torch.device,
                 deadline: float | None = None) -> list[dict]:
    cached = []
    # Cache frozen towers once; no hidden hands or outcome features enter them.
    for case in cases:
        _check_time(deadline)
        state = model.state_tower(torch.as_tensor(case["obs"], device=device, dtype=torch.float32)[None])
        actions = model.action_tower(torch.as_tensor(case["cand"], device=device, dtype=torch.float32))
        cached.append({**case, "features": torch.cat((state.expand(len(actions), -1), actions), -1).detach(),
                       "target": torch.as_tensor(case["advantages"], device=device, dtype=torch.float32)})
    return cached


@torch.inference_mode()
def evaluate_heads(model, cases: list[dict], deadline: float | None = None) -> dict:
    records = []
    for case in cases:
        _check_time(deadline)
        values = model.phase_heads[str(case["phase"])](case["features"]).squeeze(-1)
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError("non-finite tribute scores")
        choice = int(values.argmax())
        returns = case["returns"]
        heuristic = case["heuristic_index"]
        records.append({"phase": case["phase"], "huber": float(F.smooth_l1_loss(values, case["target"])),
                        "regret": float(returns.max() - returns[choice]),
                        "heuristic_regret": float(returns.max() - returns[heuristic]),
                        "advantage_over_heuristic": float(returns[choice] - returns[heuristic]),
                        "optimal": bool(returns[choice] == returns.max()),
                        "changed": choice != heuristic})
    def aggregate(rows):
        return {"positions": len(rows), **{key: float(np.mean([r[key] for r in rows])) if rows else None
                 for key in ("huber", "regret", "heuristic_regret", "advantage_over_heuristic", "optimal", "changed")}}
    return {**aggregate(records), "by_phase": {str(code): aggregate([r for r in records if r["phase"] == code])
                                               for code in (1, 2)}}


def fit(checkpoint: Path, dataset: Path, output: Path, *, steps: int = 1000,
        batch_positions: int = 32, learning_rate: float = 1e-3, seed: int = 23,
        device: str = "cpu", threads: int = 1, max_seconds: float = 600) -> dict:
    started = time.monotonic()
    if steps < 1 or batch_positions < 1 or threads < 1 or not 0 <= seed < 2**63:
        raise ValueError("invalid steps, batch positions, threads or seed")
    if any(not math.isfinite(v) or v <= 0 for v in (learning_rate, max_seconds)):
        raise ValueError("learning rate and walltime must be positive and finite")
    if output.exists():
        raise FileExistsError(f"fit output already exists: {output}")
    deadline = started + max_seconds
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    random.seed(seed)
    policy = load_policy(str(checkpoint), device=device)
    if not isinstance(policy, ModelPolicy) or not policy.heuristic_tribute:
        raise ValueError("A2 fitting requires a Stage A base checkpoint")
    _check_time(deadline)
    # Retain metadata from the same immutable base, or reject a concurrent
    # latest.pt replacement before fitting; never reload metadata at final save.
    base = load_checkpoint(checkpoint)
    if model_digest(base["model"]) != policy.checkpoint_id:
        raise ValueError("base checkpoint changed while loading; use an immutable snapshot")
    cases, provenance, dataset_digest = load_dataset(dataset, policy.checkpoint_id, deadline)
    if provenance.get("seed") == policy.training_seed:
        raise ValueError("counterfactual collection reused the base training seed")
    if provenance.get("action_mode") != getattr(policy, "action_mode", "canonical"):
        raise ValueError("base checkpoint and dataset use different action modes")
    train, test = split_matches(cases)
    if {c["phase"] for c in train} != {1, 2}:
        raise ValueError("training split must cover both exchange phases")
    model = policy.model
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    prepare_heads(model)
    train = encode_cases(model, train, policy.device, deadline)
    test = encode_cases(model, test, policy.device, deadline)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    sampler = random.Random(seed)
    output.mkdir(parents=True)
    completed = 0
    with (output / "metrics.jsonl").open("x") as stream:
        for update in range(steps):
            _check_time(deadline)
            batch = sampler.choices(train, k=batch_positions)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=policy.device)
            for code in (1, 2):
                selected = [c for c in batch if c["phase"] == code]
                if not selected:
                    continue
                features = torch.cat([c["features"] for c in selected])
                targets = torch.cat([c["target"] for c in selected])
                weights = torch.cat([torch.full_like(c["target"], 1 / (len(c["target"]) * len(batch))) for c in selected])
                values = model.phase_heads[str(code)](features).squeeze(-1)
                loss = loss + (F.smooth_l1_loss(values, targets, reduction="none") * weights).sum()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite tribute loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 10, error_if_nonfinite=True)
            optimizer.step()
            completed = update + 1
            if completed == 1 or completed % 50 == 0 or completed == steps:
                stream.write(json.dumps({"updates": completed, "huber": float(loss.detach()),
                                         "grad_norm": float(grad_norm),
                                         "elapsed_seconds": time.monotonic() - started}, allow_nan=False) + "\n")
                stream.flush()
    if not completed:
        raise TimeoutError("A2 fit finished no optimizer updates")
    frozen_unchanged = all(torch.equal(value.detach().cpu(), original[name])
                           for name, value in model.state_dict().items() if not name.startswith(HEAD_PREFIXES))
    if not frozen_unchanged:
        raise RuntimeError("A2 fit changed the frozen play model")
    report = {"stage": "a2", "status": "complete" if completed == steps else "time_limited",
              "base_checkpoint_id": policy.checkpoint_id, "dataset_sha256": dataset_digest,
              "dataset_provenance": provenance, "seed": seed, "requested_steps": steps,
              "steps": completed, "batch_positions": batch_positions, "learning_rate": learning_rate,
              "elapsed_seconds": time.monotonic() - started, "frozen_weights_unchanged": True,
              "train_matches": sorted({c["group"] for c in train}),
              "holdout_matches": sorted({c["group"] for c in test}),
              "train": evaluate_heads(model, train, deadline), "holdout": evaluate_heads(model, test, deadline),
              "promotion": "experimental_only; requires independent paired tribute arena"}
    report["elapsed_seconds"] = time.monotonic() - started
    _check_time(deadline)
    payload = {**base, "stage": "a2", "tribute_policy": "learned",
               "base_checkpoint_id": policy.checkpoint_id, "base_training_seed": policy.training_seed,
               "base_progress": base["progress"], "config": {**base["config"], "seed": seed},
               "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "rng": rng_state(np.random.default_rng(seed)),
               "progress": {"updates": completed, "elapsed_seconds": report["elapsed_seconds"]},
               "tribute_fit": report}
    save_checkpoint(output / "tribute.pt", payload)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-positions", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=600)
    args = parser.parse_args()
    print(json.dumps(fit(args.checkpoint, args.dataset, args.output, steps=args.steps,
        batch_positions=args.batch_positions, learning_rate=args.learning_rate,
        seed=args.seed, device=args.device, threads=args.threads, max_seconds=args.max_seconds), allow_nan=False))


if __name__ == "__main__":
    main()
