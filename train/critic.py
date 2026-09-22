"""Stage B perfect-information critic and its offline fit (STAGE_B_TODO B2).

Input. The actor observation (``gd.OBS_DIM`` binary planes) concatenated with
the three ``hidden_counts`` rows ``[3, 54]`` of seats +1, +2, +3, flattened to
162 counts in {0, 1, 2}. This is exactly the layout of
``RolloutBuffer.critic_input`` and of ``gather()["critic_obs"]`` in
``train/rollout_buffer.py``, so the PPO learner (B5) can feed this network
from the buffer with no re-encoding. It was chosen over "all four hands"
because the observation already carries the actor's own hand and every
public fact, so the three hidden rows are the only missing information; the
engine emits them per decision already and the B1 dataset stores them, which
makes the layout free to produce both offline and in rollout.

Output. One scalar: the expected round return for the actor's team, the
same target as the DMC play head (DESIGN 8.2 item 1).

Network. An MLP sized like the v1 state tower (``ModelConfig.state_width``
by ``state_layers``, 4 x 512) with a linear value head. It shares no
parameters with ``GuandanModel``; the policy never receives this input.

Fit. MSE, Adam, shards streamed from the B1 dataset a few at a time with an
in-memory shuffle, validation on a fixed whole-round subset of the val split,
early stopping with patience plus a step cap and a wall-clock cap, the best
weights checkpointed. ``hidden_mode="public"`` zeroes the hidden columns at
train and test time: same network, public information only (the ablation).

Report. ``report`` evaluates checkpoints on the TEST split against the M1
play head's max Q stored in the dataset, per stage bin, with bootstrap
intervals over whole rounds.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Callable, Iterator

import numpy as np
import torch
from torch import Tensor, nn

import gd
from eval.collect_critic import SPLITS, STAGES, load_shard, shard_paths
from train.ckpt import load_checkpoint, rng_state, save_checkpoint
from train.model import ModelConfig, mlp
from train.rollout_buffer import HIDDEN_DIM

HIDDEN_MODES = ("perfect", "public")
PLAY = int(gd.Phase.Play)
_V1 = ModelConfig()


def critic_input(obs: np.ndarray, hidden_counts: np.ndarray, *,
                 zero_hidden: bool = False) -> np.ndarray:
    """``[N, obs_dim + 162]`` uint8: the observation, then the hidden rows.

    Byte-identical to ``RolloutBuffer.critic_input``. ``zero_hidden`` keeps the
    width and zeroes the privileged columns (public-information ablation).
    """
    obs = np.asarray(obs, np.uint8)
    hidden = np.asarray(hidden_counts, np.uint8).reshape(len(obs), HIDDEN_DIM)
    if zero_hidden:
        hidden = np.zeros_like(hidden)
    return np.concatenate([obs, hidden], axis=1)


def policy_view(critic_obs: np.ndarray | Tensor, obs_dim: int = gd.OBS_DIM):
    """The columns of a critic input that the policy is allowed to see."""
    return critic_obs[:, :obs_dim]


@dataclass(frozen=True)
class CriticConfig:
    obs_dim: int = gd.OBS_DIM
    hidden_dim: int = HIDDEN_DIM
    width: int = _V1.state_width
    layers: int = _V1.state_layers

    def __post_init__(self) -> None:
        if min(self.obs_dim, self.hidden_dim, self.width, self.layers) <= 0:
            raise ValueError("critic dimensions must be positive")

    @property
    def input_dim(self) -> int:
        return self.obs_dim + self.hidden_dim


class Critic(nn.Module):
    """State value from the perfect-information input; training only."""

    def __init__(self, config: CriticConfig = CriticConfig()) -> None:
        super().__init__()
        self.config = config
        self.body = mlp(config.input_dim, config.width, config.layers)
        self.value = nn.Linear(config.width, 1)

    def forward(self, critic_obs: Tensor) -> Tensor:
        return self.value(self.body(critic_obs)).squeeze(-1)


@dataclass
class FitConfig:
    data: str = ".work/critic-m1"
    output: str = ".work/critic-fit/perfect"
    hidden_mode: str = "perfect"
    width: int = _V1.state_width
    layers: int = _V1.state_layers
    batch_size: int = 4096
    lr: float = 5e-4
    max_steps: int = 20000
    max_seconds: float = 1200.0
    eval_every: int = 500
    patience: int = 6            # evaluations without a new best before stopping
    min_delta: float = 1e-4      # val MSE improvement that counts as new best
    val_decisions: int = 250_000  # whole-round prefix of the val split
    mix_shards: int = 4          # shards shuffled together in memory
    seed: int = 20260924
    threads: int = 6

    def validate(self) -> None:
        if self.hidden_mode not in HIDDEN_MODES:
            raise ValueError(f"hidden_mode must be one of {HIDDEN_MODES}")
        for name in ("width", "layers", "batch_size", "max_steps", "eval_every",
                     "patience", "val_decisions", "mix_shards", "threads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not (self.lr > 0 and self.max_seconds > 0 and self.min_delta >= 0):
            raise ValueError("lr and max_seconds must be positive, min_delta nonnegative")


# ------------------------------------------------------------------- data
def split_paths(root: str | Path, split: str) -> list[Path]:
    """Shards of one split, from the manifest; never a path of another split."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split}")
    paths = [p for p in shard_paths(Path(root)) if p.parent.name == split]
    if not paths:
        raise FileNotFoundError(f"no {split} shards under {root}")
    return paths


def read_shard(path: Path) -> dict[str, np.ndarray]:
    """The fields the critic uses, observations still bit-packed."""
    shard = load_shard(path, unpack=False)
    if int(shard["obs_dim"]) != gd.OBS_DIM:
        raise ValueError(f"{path}: obs_dim {int(shard['obs_dim'])} != {gd.OBS_DIM}")
    if str(shard["split"]) != path.parent.name:
        raise ValueError(f"{path}: shard split {shard['split']} does not match its directory")
    lengths = np.diff(shard["round_offsets"])
    # Match identity is (worker seed, env_id, match_id); pack it into one int64.
    seed, env, match = int(shard["seed"]), shard["env_id"].astype(np.int64), shard["match_id"]
    if seed >= 2**26 or env.max(initial=0) >= 2**12 or match.max(initial=0) >= 2**24:
        raise ValueError(f"{path}: match identity does not fit the packed key")
    return {"obs_bits": shard["obs_bits"],
            "hidden": shard["hidden"].reshape(-1, HIDDEN_DIM),
            "target": shard["team_return"].astype(np.float32),
            "phase": shard["phase"], "stage": shard["stage"], "q_max": shard["q_max"],
            "round": np.repeat(np.arange(len(lengths)), lengths),
            "match": np.repeat((seed << 36) | (env << 24) | match, lengths),
            "split": np.asarray(str(shard["split"]))}


def to_tensor(obs_bits: np.ndarray, hidden: np.ndarray, *, zero_hidden: bool) -> Tensor:
    """Unpack a batch and assemble the critic input as float32."""
    obs = np.unpackbits(obs_bits, axis=1, count=gd.OBS_DIM)
    return torch.from_numpy(critic_input(obs, hidden, zero_hidden=zero_hidden)).float()


def stream_batches(paths: list[Path], batch_size: int, rng: np.random.Generator,
                   mix_shards: int) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Endless shuffled batches ``(obs_bits, hidden, target)``.

    Every epoch visits the shards in a new order, ``mix_shards`` at a time;
    rows of a group are shuffled together and the next group is read in a
    background thread. Resident memory is about two groups of packed rows.
    """
    def load(group: list[Path]) -> tuple[np.ndarray, ...]:
        parts = [read_shard(p) for p in group]
        return tuple(np.concatenate([part[k] for part in parts])
                     for k in ("obs_bits", "hidden", "target"))

    def groups() -> Iterator[list[Path]]:
        while True:
            order = rng.permutation(len(paths))
            for g in range(0, len(order), mix_shards):
                yield [paths[i] for i in order[g:g + mix_shards]]

    schedule = groups()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(load, next(schedule))
        while True:
            bits, hidden, target = future.result()
            future = pool.submit(load, next(schedule))
            perm = rng.permutation(len(target))
            for begin in range(0, len(perm) - batch_size + 1, batch_size):
                rows = np.sort(perm[begin:begin + batch_size])
                yield bits[rows], hidden[rows], target[rows]


def load_rows(paths: list[Path], max_decisions: int | None = None) -> dict[str, np.ndarray]:
    """Whole shards concatenated, stopping at the first whole round past the cap."""
    parts, total = [], 0
    for path in paths:
        part = read_shard(path)
        if parts:  # keep round ids unique across shards
            part["round"] = part["round"] + parts[-1]["round"][-1] + 1
        parts.append(part)
        total += len(part["target"])
        if max_decisions is not None and total >= max_decisions:
            break
    keys = [k for k in parts[0] if k != "split"]
    rows = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    if max_decisions is not None and total > max_decisions:
        last_round = rows["round"][max_decisions - 1]
        keep = rows["round"] <= last_round
        rows = {k: v[keep] for k, v in rows.items()}
    return rows


@torch.inference_mode()
def predict(model: Critic, obs_bits: np.ndarray, hidden: np.ndarray, *,
            zero_hidden: bool, chunk: int = 16384) -> np.ndarray:
    model.eval()
    out = np.empty(len(obs_bits), np.float32)
    for begin in range(0, len(obs_bits), chunk):
        end = begin + chunk
        out[begin:end] = model(to_tensor(obs_bits[begin:end], hidden[begin:end],
                                         zero_hidden=zero_hidden)).numpy()
    return out


# -------------------------------------------------------------------- fit
def checkpoint_payload(model: Critic, optimizer: torch.optim.Optimizer, config: FitConfig,
                       progress: dict, rng: np.random.Generator) -> dict:
    return {"stage": "critic", "hidden_mode": config.hidden_mode,
            "model_config": asdict(model.config), "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "config": asdict(config),
            "progress": progress, "rng": rng_state(rng)}


def load_critic(path: str | Path, device: str = "cpu") -> tuple[Critic, dict]:
    """Critic and its checkpoint payload. Refuses anything but a critic."""
    payload = load_checkpoint(path, device=device)
    if payload.get("stage") != "critic":
        raise ValueError(f"{path} is not a critic checkpoint")
    model = Critic(CriticConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), payload


def fit(config: FitConfig, log: Callable[[str], None] | None = None) -> dict:
    """Fit one critic; returns the fit summary also written to ``fit.json``."""
    config.validate()
    log = log or (lambda line: print(line, flush=True))
    started = time.monotonic()
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    zero_hidden = config.hidden_mode == "public"
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    train_paths = split_paths(config.data, "train")
    val = load_rows(split_paths(config.data, "val"), config.val_decisions)
    model = Critic(CriticConfig(width=config.width, layers=config.layers))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best = {"val_mse": math.inf, "step": 0}
    history: list[dict] = []
    stale, step, running, seen = 0, 0, [], 0
    stop_reason = "max_steps"
    batches = stream_batches(train_paths, config.batch_size, rng, config.mix_shards)
    for bits, hidden, target in batches:
        model.train()
        x = to_tensor(bits, hidden, zero_hidden=zero_hidden)
        loss = nn.functional.mse_loss(model(x), torch.from_numpy(target))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step += 1
        seen += len(target)
        running.append(loss.item())
        at_cap = step >= config.max_steps
        out_of_time = time.monotonic() - started >= config.max_seconds
        if step % config.eval_every == 0 or at_cap or out_of_time:
            pred = predict(model, val["obs_bits"], val["hidden"], zero_hidden=zero_hidden)
            val_mse = float(np.mean((pred - val["target"]) ** 2))
            entry = {"step": step, "samples": seen, "train_mse": float(np.mean(running)),
                     "val_mse": val_mse, "seconds": time.monotonic() - started}
            running = []
            history.append(entry)
            log(json.dumps(entry))
            if val_mse < best["val_mse"] - config.min_delta:
                best, stale = dict(entry), 0
                save_checkpoint(output / "best.pt", checkpoint_payload(
                    model, optimizer, config, dict(entry, updates=step), rng))
            else:
                stale += 1
            if stale >= config.patience:
                stop_reason = "early_stop"
                break
            if at_cap or out_of_time:
                stop_reason = "max_steps" if at_cap else "max_seconds"
                break
    batches.close()
    summary = {"config": asdict(config), "stop_reason": stop_reason, "steps": step,
               "samples": seen, "best": best, "history": history,
               "val_rounds": int(len(np.unique(val["round"]))),
               "val_decisions": int(len(val["target"])),
               "parameters": sum(p.numel() for p in model.parameters()),
               "seconds": time.monotonic() - started}
    (output / "fit.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# ----------------------------------------------------------------- report
def bootstrap(errors: dict[str, np.ndarray], counts: np.ndarray, *, cluster: np.ndarray,
              resamples: int, seed: int) -> dict[str, tuple[float, float]]:
    """Percentile 95% intervals of ratio-of-sums MSEs, resampling clusters.

    ``errors[name]`` and ``counts`` are per-decision squared errors and 0/1
    weights; they are summed per cluster (a round or a match) first, and the
    same resampled clusters are used for every name so differences are paired.
    """
    ids, cluster = np.unique(cluster, return_inverse=True)
    c = len(ids)
    n = np.bincount(cluster, weights=counts, minlength=c)
    sums = {k: np.bincount(cluster, weights=v * counts, minlength=c) for k, v in errors.items()}
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in errors}
    for begin in range(0, resamples, 100):
        size = min(100, resamples - begin)
        weight = np.stack([np.bincount(rng.integers(0, c, c), minlength=c)
                           for _ in range(size)]).astype(np.float64)
        total = weight @ n
        for k in errors:
            draws[k].append((weight @ sums[k]) / total)
    return {k: tuple(float(x) for x in np.percentile(np.concatenate(v), [2.5, 97.5]))
            for k, v in draws.items()}


def report(checkpoints: dict[str, str | Path], data: str | Path, *, resamples: int = 2000,
           seed: int = 7, threads: int = 6) -> dict:
    """G1 on the test split: critic MSE vs the M1 play head's max-Q MSE."""
    torch.set_num_threads(threads)
    test = load_rows(split_paths(data, "test"))
    target = test["target"].astype(np.float64)
    play = test["phase"] == PLAY
    q = test["q_max"].astype(np.float64)
    if not np.isfinite(q[play]).all() or not np.isnan(q[~play]).all():
        raise ValueError("expected finite max Q on play rows and NaN on tribute rows")
    predictions = {}
    for name, path in checkpoints.items():
        model, payload = load_critic(path)
        predictions[name] = predict(model, test["obs_bits"], test["hidden"],
                                    zero_hidden=payload["hidden_mode"] == "public"
                                    ).astype(np.float64)
    errors = {name: (p - target) ** 2 for name, p in predictions.items()}
    errors["m1_max_q"] = np.where(play, q - target, 0.0) ** 2
    critics = list(predictions)
    for name in critics:
        errors[f"{name}-m1_max_q"] = errors[name] - errors["m1_max_q"]
        if name != critics[0]:
            errors[f"{name}-{critics[0]}"] = errors[name] - errors[critics[0]]
    # Round ids are unique within the concatenated test rows (load_rows).
    rounds, matches = test["round"], test["match"]
    bins = {"overall": play}
    bins.update({stage: play & (test["stage"] == i) for i, stage in enumerate(STAGES)})
    table = {}
    for label, mask in bins.items():
        w = mask.astype(np.float64)
        point = {k: float((v * w).sum() / w.sum()) for k, v in errors.items()}
        ci = bootstrap(errors, w, cluster=rounds, resamples=resamples, seed=seed)
        table[label] = {"decisions": int(mask.sum()),
                        "rounds": int(len(np.unique(rounds[mask]))),
                        "mse": point, "ci95_rounds": ci}
    overall_matches = bootstrap(errors, play.astype(np.float64), cluster=matches,
                                resamples=resamples, seed=seed)
    tribute = ~play
    tribute_table = {"decisions": int(tribute.sum()),
                     "mse": {n: float(errors[n][tribute].mean()) for n in critics},
                     "ci95_rounds": bootstrap({n: errors[n] for n in critics},
                                              tribute.astype(np.float64), cluster=rounds,
                                              resamples=resamples, seed=seed)}
    first = critics[0]
    gate = all(table[b]["ci95_rounds"][f"{first}-m1_max_q"][1] < 0 for b in bins)
    return {
        "split": "test", "rounds": int(len(np.unique(rounds))),
        "decisions": int(len(target)), "play_decisions": int(play.sum()),
        "matches": int(len(np.unique(matches))),
        "target_variance_play": float(target[play].var()),
        "bootstrap": {"resamples": resamples, "unit": "round", "seed": seed,
                      "interval": "percentile 95%"},
        "checkpoints": {n: str(p) for n, p in checkpoints.items()},
        "play": table, "overall_ci95_matches": overall_matches, "tribute": tribute_table,
        "g1_critic": first,
        "g1_pass": bool(gate),
        "g1_rule": "upper 95% bound of (critic MSE - M1 max-Q MSE) below zero, "
                   "overall and in each stage bin, on play decisions of the test split",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    fit_parser = sub.add_parser("fit", help="fit one critic")
    defaults = FitConfig()
    for key, value in asdict(defaults).items():
        fit_parser.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    report_parser = sub.add_parser("report", help="G1 evaluation on the test split")
    report_parser.add_argument("--data", default=defaults.data)
    report_parser.add_argument("--critic", action="append", required=True,
                               metavar="NAME=PATH", help="first one is the G1 critic")
    report_parser.add_argument("--resamples", type=int, default=2000)
    report_parser.add_argument("--threads", type=int, default=6)
    report_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "fit":
        config = FitConfig(**{k: getattr(args, k) for k in asdict(defaults)})
        summary = fit(config)
        print(json.dumps({k: summary[k] for k in ("stop_reason", "steps", "best", "seconds")}))
    else:
        checkpoints = dict(item.split("=", 1) for item in args.critic)
        result = report(checkpoints, args.data, resamples=args.resamples, threads=args.threads)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"g1_pass": result["g1_pass"], "play": {
            b: result["play"][b]["mse"] for b in result["play"]}}, indent=2))


if __name__ == "__main__":
    main()
