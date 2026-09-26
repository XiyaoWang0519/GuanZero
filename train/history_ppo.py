"""Cold-start PPO for the history Transformer player (STAGE_C T2, DESIGN v0.6 8).

Training boundary (DESIGN 1.1): ``fresh_player`` is the only initialization
path; ``--resume`` accepts a ``history_ppo`` checkpoint of this lineage and
nothing else. There is no teacher, no old MLP, no candidate pruning and no
KL toward anything; the only old-policy comparison is PPO's ratio against
the collecting actor's stored log-probabilities.

Reward and credit assignment replicate ``train/ppo.py``: the reward of an
``(env, team, round)`` trajectory is ``RoundResult.seat_return[team]`` (the
engine's per-round team level return, +3/+2/+1 by finish order with the
opposite sign for the other team, level-A handling included), written on the
team's last stored row of that round with ``done``; every other reward is
zero; GAE uses ``gamma`` 1.0 and ``gae_lambda`` 0.95 with a terminal boundary
at round end and no bootstrap into the next round; values come from the
current critic at ``learn`` time; advantages are normalized per minibatch;
only completed trajectories train and unfinished rounds carry over.

At every epoch and minibatch the actor's log-probabilities are recomputed
from the raw public tokens of the minibatch's matches (``StreamBatch`` of
whole matches, per-row ``prefix``), so gradients flow through the stream
encoder. Minibatches are groups of whole matches, never independent rows.

Usage::

    python -m train.history_ppo --output DIR --updates N --num-envs E \\
        --steps-per-update S --seed K [--width --layers --heads --window --lr \\
        --clip --entropy --epochs --minibatch-matches --resume CKPT]
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterator
import uuid

import gd
import numpy as np
import torch
from torch.nn import functional as F

from train.ckpt import restore_rng, rng_state
from train.history_model import (STAGE, TOKEN_SCHEMA_VERSION,
                                 HistoryPolicyConfig, checkpoint_payload, count_parameters,
                                 fresh_player, load_history_checkpoint, save_history_checkpoint)
from train.history_rollout import (HistoryCollector, MatchEventStore, SequenceRolloutBuffer)
from train.logs import TOKEN_DIM

REWARD_SEMANTICS = {
    "reward": "RoundResult.seat_return[team], team = seat % 2, on the team's last stored "
              "row of the (env, team, round) trajectory; zero elsewhere",
    "terminal": "done at round end; no bootstrap across rounds (value past the last "
                "row is zero)",
    "values": "current HistoryCritic(obs, hidden_counts) at learn time",
    "trajectories": "completed rounds only; unfinished rounds carry over to the next update",
    "advantages": "GAE(gamma, gae_lambda), normalized per minibatch",
    "tribute": "engine greedy heuristic for Tribute/BackTribute rows; those rows are "
               "never PPO rows but their public exchange events are in the stream",
}


@dataclass
class HistoryPPOConfig:
    # architecture (HistoryPolicyConfig)
    width: int = 64
    layers: int = 2
    heads: int = 4
    window: int = 0
    max_rounds: int = 16
    # rollout
    num_envs: int = 16
    num_threads: int = 1
    steps_per_update: int = 64
    seed: int = 0
    # learner
    lr: float = 3e-4
    critic_lr: float | None = None       # None: same as lr
    clip: float = 0.2
    entropy: float = 0.01
    value_coef: float = 1.0
    epochs: int = 2
    minibatch_matches: int = 4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    grad_clip: float = 10.0
    normalize_advantages: bool = True
    # lifecycle
    updates: int = 1
    checkpoint_updates: int = 1
    torch_threads: int = 0               # 0 leaves torch's default

    def __post_init__(self) -> None:
        if min(self.num_envs, self.steps_per_update, self.epochs, self.minibatch_matches,
               self.checkpoint_updates) <= 0:
            raise ValueError("environment, step, epoch, minibatch and checkpoint counts "
                             "must be positive")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma must be in (0, 1] and gae_lambda in [0, 1]")
        if self.lr <= 0 or self.clip <= 0 or self.entropy < 0:
            raise ValueError("lr and clip must be positive; entropy must not be negative")

    def policy_config(self) -> HistoryPolicyConfig:
        return HistoryPolicyConfig(width=self.width, layers=self.layers, heads=self.heads,
                                   window=self.window, max_rounds=self.max_rounds)

    @classmethod
    def from_payload(cls, config: dict[str, Any], **overrides: Any) -> "HistoryPPOConfig":
        names = {f.name for f in fields(cls)}
        values = {k: v for k, v in config.items() if k in names}
        values.update(overrides)
        return cls(**values)


def segment_entropy(log_probs: torch.Tensor, rows: torch.Tensor, count: int) -> torch.Tensor:
    """Entropy of each decision's full candidate distribution, ``[count]``."""
    terms = -(log_probs.exp() * log_probs)
    return torch.zeros(count, device=log_probs.device, dtype=log_probs.dtype).index_add_(
        0, rows, terms)


def grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().float().norm() ** 2)
    return math.sqrt(total)


class HistoryTrainer:
    """Collect, learn, checkpoint. ``update()`` is one PPO iteration."""

    STAT_KEYS = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
                 "ratio_deviation", "actor_grad_norm", "encoder_grad_norm", "critic_grad_norm")

    def __init__(self, config: HistoryPPOConfig, output: str | Path, device: str = "cpu",
                 resume: str | Path | None = None) -> None:
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        if config.torch_threads:
            torch.set_num_threads(config.torch_threads)
        payload: dict[str, Any] | None = None
        if resume is not None:
            actor, critic, payload = load_history_checkpoint(resume, self.device)
            config = HistoryPPOConfig.from_payload(payload["config"], updates=config.updates)
            self.lineage = str(payload["lineage"])
        else:
            actor, critic = fresh_player(config.policy_config(), config.seed)
            self.lineage = f"{STAGE}-{config.seed}-{uuid.uuid4().hex[:8]}"
        self.config = config
        # Dropout is zero, so train mode is the same policy as eval mode; staying
        # in train mode keeps the encoder on one kernel path for both the
        # behaviour log-probabilities and the learner's recomputation.
        self.actor = actor.to(self.device).train()
        self.critic = critic.to(self.device).train()
        critic_lr = config.critic_lr if config.critic_lr is not None else config.lr
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.rng = np.random.default_rng(config.seed)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(config.seed)
        self.progress: dict[str, Any] = {"updates": 0, "decisions": 0, "rounds": 0,
                                         "matches": 0, "samples": 0, "learner_rows": 0,
                                         "elapsed_seconds": 0.0}
        if payload is not None:
            self.actor_optimizer.load_state_dict(payload["optimizer"]["actor"])
            self.critic_optimizer.load_state_dict(payload["optimizer"]["critic"])
            self.progress.update(payload["progress"])
            rng = payload["rng"]
            restore_rng(rng, self.rng)
            self.generator.set_state(rng["sampler"].cpu())
        self.env = gd.VecEnv(num_envs=config.num_envs, num_threads=config.num_threads,
                             seed=config.seed + 1000 * self.progress["updates"],
                             log_public_actions=True, log_env_limit=config.num_envs)
        self.store = MatchEventStore()
        self.buffer = SequenceRolloutBuffer()
        self.collector = HistoryCollector(self.env, self.actor, self.store, self.buffer,
                                          self.generator, self.device)
        self.prior_elapsed = float(self.progress["elapsed_seconds"])
        self.started = time.monotonic()
        self.metrics_path = self.output / "metrics.jsonl"
        self.write_manifest()

    # -- manifest ---------------------------------------------------------------

    def write_manifest(self) -> None:
        from train.tribute_data import engine_source_digest   # imports eval helpers; keep lazy
        manifest = {
            "stage": STAGE, "lineage": self.lineage, "init": "random", "teacher": None,
            "config": asdict(self.config), "model_config": asdict(self.actor.config),
            "engine_digest": engine_source_digest(),
            "token_schema": {"version": TOKEN_SCHEMA_VERSION, "dim": int(TOKEN_DIM),
                             "forced_bit": False, "private_tribute_flags": False},
            "reward": REWARD_SEMANTICS,
            "candidates": "full canonical set in engine order; every candidate selectable",
            "seats": "current-policy copies in all four seats",
            "parameters": {"actor": count_parameters(self.actor),
                           "critic": count_parameters(self.critic)},
            "torch": torch.__version__, "device": str(self.device),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # -- one iteration ----------------------------------------------------------

    def collect(self):
        return self.collector.collect(self.config.steps_per_update,
                                      version=self.progress["updates"])

    @torch.no_grad()
    def refresh_values(self, chunk: int = 4096) -> np.ndarray:
        """Current critic value of every stored row (completed or not)."""
        data = self.buffer.compact()
        n = len(self.buffer)
        values = np.zeros(n, np.float32)
        for begin in range(0, n, chunk):
            obs = torch.as_tensor(data["obs"][begin:begin + chunk], device=self.device)
            hidden = torch.as_tensor(data["hidden"][begin:begin + chunk], device=self.device)
            values[begin:begin + chunk] = self.critic(obs, hidden).float().cpu().numpy()
        return values

    def recompute_log_probs(self, rows: np.ndarray, streams=None
                            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Current-actor log-probability of each stored row's chosen candidate
        and the entropy over its full candidate set, from raw tokens."""
        inputs, chosen = self.buffer.decision_inputs(rows, self.store, self.device, streams)
        log_probs = self.actor.candidate_log_probs(inputs)
        return log_probs[chosen], segment_entropy(log_probs, inputs.rows, inputs.decisions)

    def minibatches(self) -> Iterator[np.ndarray]:
        return self.buffer.minibatches(self.config.minibatch_matches, self.rng)

    def minibatch_loss(self, rows: np.ndarray) -> dict[str, torch.Tensor]:
        cfg = self.config
        buffer = self.buffer
        log_prob, entropy = self.recompute_log_probs(rows)
        old = torch.as_tensor(buffer.compact()["logp"][rows], device=self.device)
        advantage = torch.as_tensor(buffer.advantage[rows], device=self.device)
        returns = torch.as_tensor(buffer.returns[rows], device=self.device)
        log_ratio = log_prob - old
        ratio = log_ratio.exp()
        if cfg.normalize_advantages and len(advantage) > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        clipped = torch.minimum(ratio * advantage,
                                ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advantage)
        surrogate = -clipped.mean()
        policy_total = surrogate - cfg.entropy * entropy.mean()
        data = buffer.compact()
        obs = torch.as_tensor(data["obs"][rows], device=self.device)
        hidden = torch.as_tensor(data["hidden"][rows], device=self.device)
        value_loss = F.mse_loss(self.critic(obs, hidden), returns)
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = ((ratio - 1).abs() > cfg.clip).float().mean()
            ratio_deviation = (ratio - 1).abs().max()
        return {"policy_total": policy_total, "value_total": cfg.value_coef * value_loss,
                "policy_loss": surrogate, "value_loss": value_loss, "entropy": entropy.mean(),
                "approx_kl": approx_kl, "clip_fraction": clip_fraction,
                "ratio_deviation": ratio_deviation}

    def learn(self) -> dict[str, Any]:
        cfg = self.config
        values = self.refresh_values()
        samples = self.buffer.finalize(values, cfg.gamma, cfg.gae_lambda)
        stats: dict[str, Any] = {"update_samples": samples, "minibatches": 0}
        stats.update({key: None for key in self.STAT_KEYS})
        if samples == 0:
            stats.update({"mean_reward": None, "mean_abs_reward": None, "mean_return": None,
                          "explained_variance": None})
            return stats
        rows = self.buffer.samples
        returns = self.buffer.returns[rows]
        variance = float(np.var(returns))
        stats["explained_variance"] = (1.0 - float(np.var(returns - values[rows])) / variance
                                       if variance > 0 else 0.0)
        stats["mean_return"] = float(returns.mean())
        terminal = self.buffer.done[rows]
        rewards = self.buffer.reward[rows][terminal]
        # All four seats are the learner, so the signed mean is zero-sum noise;
        # the magnitude is the levels at stake per finished trajectory.
        stats["mean_reward"] = float(rewards.mean()) if rewards.size else None
        stats["mean_abs_reward"] = float(np.abs(rewards).mean()) if rewards.size else None
        totals = {key: 0.0 for key in self.STAT_KEYS}
        count = 0
        for _ in range(cfg.epochs):
            for batch in self.minibatches():
                terms = self.minibatch_loss(batch)
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                terms["policy_total"].backward()
                terms["value_total"].backward()
                encoder = grad_norm(self.actor.stream.parameters())
                actor = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                             cfg.grad_clip))
                critic = float(torch.nn.utils.clip_grad_norm_(self.critic.parameters(),
                                                              cfg.grad_clip))
                if not (math.isfinite(actor) and math.isfinite(critic)):
                    raise FloatingPointError("non-finite gradient; stopping before the step")
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                for key in self.STAT_KEYS:
                    if key in terms:
                        totals[key] += float(terms[key].detach())
                totals["actor_grad_norm"] += actor
                totals["encoder_grad_norm"] += encoder
                totals["critic_grad_norm"] += critic
                count += 1
        stats["minibatches"] = count
        for key in self.STAT_KEYS:
            stats[key] = totals[key] / count
        self.progress["samples"] += samples
        return stats

    def update(self) -> dict[str, Any]:
        """Collect, learn, log one metrics line and advance the counters."""
        t0 = time.perf_counter()
        collected = self.collect()
        t1 = time.perf_counter()
        stats = self.learn()
        t2 = time.perf_counter()
        cited = self.buffer.next_iteration()
        pruned = self.store.prune(cited)
        self.progress["updates"] += 1
        self.progress["decisions"] += collected.decisions
        self.progress["learner_rows"] += collected.learner_rows
        self.progress["rounds"] += collected.rounds
        self.progress["matches"] += collected.matches
        self.progress["elapsed_seconds"] = (self.prior_elapsed + time.monotonic()
                                            - self.started)
        line = {
            "update": self.progress["updates"], "decisions": self.progress["decisions"],
            "rounds": self.progress["rounds"], "matches": self.progress["matches"],
            "step_decisions": collected.decisions, "step_learner_rows": collected.learner_rows,
            "step_rounds": collected.rounds,
            "round_gain": collected.gain / collected.rounds if collected.rounds else None,
            "decisions_per_sec": collected.decisions / max(t1 - t0, 1e-9),
            "collect_seconds": t1 - t0, "learn_seconds": t2 - t1,
            "mean_prefix": collected.mean_prefix, "max_prefix": collected.prefix_max,
            "store_matches": len(self.store), "store_tokens": self.store.tokens,
            "pruned_matches": pruned, "carried_rows": len(self.buffer),
            "elapsed_seconds": self.progress["elapsed_seconds"],
        }
        line.update(stats)
        with self.metrics_path.open("a") as stream:
            stream.write(json.dumps(line) + "\n")
        return line

    # -- checkpoints ------------------------------------------------------------

    def payload(self) -> dict[str, Any]:
        self.progress["elapsed_seconds"] = (self.prior_elapsed + time.monotonic()
                                            - self.started)
        rng = rng_state(self.rng)
        rng["sampler"] = self.generator.get_state()
        return checkpoint_payload(
            self.actor, self.critic, lineage=self.lineage, seed=self.config.seed,
            optimizer={"actor": self.actor_optimizer.state_dict(),
                       "critic": self.critic_optimizer.state_dict()},
            config=asdict(self.config), progress=dict(self.progress), rng=rng)

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path is not None else self.output / "latest.pt"
        save_history_checkpoint(path, self.payload())
        return path

    def run(self) -> None:
        while self.progress["updates"] < self.config.updates:
            line = self.update()
            print(json.dumps({k: line[k] for k in ("update", "decisions", "rounds",
                                                    "update_samples", "policy_loss",
                                                    "value_loss", "entropy",
                                                    "decisions_per_sec", "mean_prefix")}),
                  flush=True)
            if self.progress["updates"] % self.config.checkpoint_updates == 0:
                self.save()
        self.save()


# ---- CLI ------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--steps-per-update", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch-matches", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--checkpoint-updates", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> HistoryPPOConfig:
    names = {f.name for f in fields(HistoryPPOConfig)}
    return HistoryPPOConfig(**{k: v for k, v in vars(args).items() if k in names})


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    trainer = HistoryTrainer(config, args.output, device=args.device, resume=args.resume)
    trainer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
