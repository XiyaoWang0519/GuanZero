"""Offline v1/v2 belief comparison with match-held-out, causal public histories.

This experiment does not enable Transformer RL. An improvement here is only
the first gate; equal-compute RL evaluation is still required by DESIGN 7.4.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train.public_history import TOKEN_DIM
from train.model import mlp


class FlatBelief(nn.Module):
    def __init__(self, obs_dim: int, width: int) -> None:
        super().__init__()
        self.tower = mlp(obs_dim, width, 4)
        self.head = nn.Linear(width, 3 * 54 * 3)

    def forward(self, obs, tokens, lengths, seat):
        return self.head(self.tower(obs)).reshape(-1, 3, 54, 3)


class HistoryBelief(nn.Module):
    def __init__(self, obs_dim: int, width: int = 128, layers: int = 2) -> None:
        super().__init__()
        self.width = width
        self.public = nn.Linear(TOKEN_DIM, width)
        self.bos = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, 4, width * 4, dropout=0.0, batch_first=True)
        self.stream = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.private = nn.Linear(obs_dim, width)
        self.seat = nn.Embedding(4, width)
        self.query = nn.MultiheadAttention(width, 4, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 3 * 54 * 3)

    def forward(self, obs, tokens, lengths, seat):
        # Only prefix tokens are supplied; BOS makes an empty prefix well-defined.
        stream = torch.cat((self.bos.expand(len(obs), -1, -1), self.public(tokens)), dim=1)
        positions = torch.arange(stream.shape[1], device=obs.device).float()[:, None]
        frequencies = torch.exp(torch.arange(0, self.width, 2, device=obs.device).float()
                                * (-np.log(10000.0) / self.width))
        position = torch.zeros(stream.shape[1], self.width, device=obs.device)
        position[:, 0::2] = torch.sin(positions * frequencies)
        position[:, 1::2] = torch.cos(positions * frequencies)
        stream = stream + position
        padding = torch.arange(stream.shape[1], device=obs.device)[None] > lengths[:, None]
        causal = torch.ones(stream.shape[1], stream.shape[1], device=obs.device,
                            dtype=torch.bool).triu(1)
        stream = self.stream(stream, mask=causal, src_key_padding_mask=padding)
        query = (self.private(obs) + self.seat(seat))[:, None]
        attended, _ = self.query(query, stream, stream, key_padding_mask=padding,
                                 need_weights=False)
        return self.head(self.norm(query + attended)[:, 0]).reshape(-1, 3, 54, 3)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def matched_models(obs_dim: int, width: int = 128, layers: int = 2) -> dict[str, nn.Module]:
    history = HistoryBelief(obs_dim, width, layers)
    target = count_parameters(history)
    # Parameter count for four flat layers + hidden-hand head, including biases.
    def flat_count(w):
        return 3 * w * w + (obs_dim + 4 + 486) * w + 486
    best = min(range(8, 2049), key=lambda w: abs(flat_count(w) - target))
    flat = FlatBelief(obs_dim, best)
    if abs(count_parameters(flat) - target) / target > 0.02:
        raise ValueError("probe parameter counts cannot be matched within 2%")
    return {"v1": flat, "v2": history}


def load_rounds(directory: Path, max_rounds: int = 10000) -> tuple[list[dict], list[dict]]:
    rounds = []
    for path in sorted(directory.glob("round-*.npz"))[:max_rounds]:
        with np.load(path, allow_pickle=False) as data:
            # Schema 2 only adds styled-opponent fields and schema 3 candidate
            # sets; the probe reads the same tensors from every version.
            if int(data["schema_version"]) not in (1, 2, 3):
                raise ValueError(f"unsupported log schema: {path}")
            item = {key: data[key].copy() for key in ("obs", "hidden", "seat", "prefix", "tokens")}
            item["group"] = str(data["group"])
            if np.any(item["prefix"] > len(item["tokens"])):
                raise ValueError(f"history prefix extends beyond round: {path}")
            rounds.append(item)
    groups = sorted({item["group"] for item in rounds},
                    key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    if len(groups) < 2:
        raise ValueError("belief probe needs logs from at least two distinct matches")
    held_out = set(groups[:max(1, len(groups) // 5)])
    return ([r for r in rounds if r["group"] not in held_out],
            [r for r in rounds if r["group"] in held_out])


def examples(rounds: list[dict]) -> list[tuple[dict, int]]:
    return [(record, i) for record in rounds for i in range(len(record["obs"]))]


def collate(items: list[tuple[dict, int]], device: str, *,
            include_history: bool = True) -> dict[str, torch.Tensor]:
    """Build only the inputs an arm uses; omitted history has an empty prefix."""
    lengths = (np.asarray([int(record["prefix"][i]) for record, i in items], np.int64)
               if include_history else np.zeros(len(items), np.int64))
    tokens = np.zeros((len(items), int(lengths.max(initial=0)), TOKEN_DIM), np.float32)
    if include_history:
        for row, ((record, _), length) in enumerate(zip(items, lengths)):
            tokens[row, :length] = record["tokens"][:length]
    return {
        "obs": torch.tensor(np.stack([r["obs"][i] for r, i in items]), device=device, dtype=torch.float32),
        "hidden": torch.tensor(np.stack([r["hidden"][i] for r, i in items]), device=device, dtype=torch.long),
        "seat": torch.tensor([int(r["seat"][i]) for r, i in items], device=device),
        "lengths": torch.tensor(lengths, device=device),
        "tokens": torch.tensor(tokens, device=device),
    }


@torch.inference_mode()
def evaluate(model: nn.Module, data: list, device: str, batch_size: int) -> dict:
    model.eval()
    sums = np.zeros((3, 3))
    counts = np.zeros((3, 3), dtype=np.int64)
    for start in range(0, len(data), batch_size):
        batch = collate(data[start:start + batch_size], device)
        output = model(batch["obs"], batch["tokens"], batch["lengths"], batch["seat"])
        loss = F.cross_entropy(output.reshape(-1, 3), batch["hidden"].reshape(-1),
                               reduction="none").reshape(-1, 3, 54).mean(-1).cpu().numpy()
        # Stage from the public number of cards played (all four 108-plane sets).
        played = batch["obs"][:, 216:648].sum(-1).cpu().numpy()
        stages = np.digitize(played, [36, 72])
        for stage in range(3):
            sums[stage] += loss[stages == stage].sum(0)
            counts[stage] += (stages == stage).sum()
    return {"log_loss": float(sums.sum() / counts.sum()),
            "by_stage_and_seat": {
                stage: {seat: {"log_loss": float(sums[i, j] / counts[i, j]) if counts[i, j] else None,
                               "decisions": int(counts[i, j])}
                        for j, seat in enumerate(("lho", "partner", "rho"))}
                for i, stage in enumerate(("early", "middle", "late"))}}


def run_probe(directory: Path, *, device: str = "cpu", steps: int = 1000,
              batch_size: int = 64, seed: int = 11, width: int = 128,
              layers: int = 2, max_rounds: int = 10000) -> dict:
    if steps <= 0 or batch_size <= 0 or max_rounds <= 0:
        raise ValueError("steps, batch_size and max_rounds must be positive")
    torch.manual_seed(seed)
    train, test = load_rounds(directory, max_rounds)
    train_items, test_items = examples(train), examples(test)
    if not train_items or not test_items:
        raise ValueError("both match-separated splits need decisions")
    models = matched_models(train[0]["obs"].shape[1], width, layers)
    results = {}
    for name, model in models.items():
        model.to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        sampler = random.Random(seed)  # identical sampled decisions for both towers
        for _ in range(steps):
            batch = collate(sampler.choices(train_items, k=batch_size), device)
            logits = model(batch["obs"], batch["tokens"], batch["lengths"], batch["seat"])
            loss = F.cross_entropy(logits.reshape(-1, 3), batch["hidden"].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
        results[name] = {"parameters": count_parameters(model),
                         **evaluate(model, test_items, device, batch_size)}
    return {"seed": seed, "steps": steps, "train_decisions": len(train_items),
            "test_decisions": len(test_items), "train_matches": len({r['group'] for r in train}),
            "test_matches": len({r['group'] for r in test}), "results": results,
            "v2_log_loss_improvement": results["v1"]["log_loss"] - results["v2"]["log_loss"],
            "gate": "diagnostic only; repeat across seeds before enabling v2 RL"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=10000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = run_probe(args.logs, device=args.device, steps=args.steps, batch_size=args.batch_size,
                       seed=args.seed, width=args.width, layers=args.layers, max_rounds=args.max_rounds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
