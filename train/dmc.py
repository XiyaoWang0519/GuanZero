"""Synchronous full-match self-play and round-return regression (DESIGN 8.2)."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import json
import math
import os
import platform
from pathlib import Path
import random
import resource
import signal
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

import gd
from train.buffer import Decision, ReplayBuffer
from train.ckpt import load_checkpoint, restore_rng, rng_state, save_checkpoint
from train.logs import public_token, save_round
from train.model import GuandanModel, ModelConfig, select_actions


@dataclass
class TrainConfig:
    seed: int = 7
    num_envs: int = 4096
    num_threads: int = 8
    torch_threads: int = 4
    rollout_steps: int = 32
    batch_size: int = 2048
    replay_capacity: int = 262144
    min_replay: int = 2048
    learn_steps: int = 64
    max_updates: int = 100000
    max_seconds: float = 86400
    learning_rate: float = 0.0001
    hidden_weight: float = 0.1
    finish_weight: float = 0.1
    grad_clip: float = 10.0
    epsilon_start: float = 0.1
    epsilon_end: float = 0.01
    greedy_start: float = 0.5
    greedy_end: float = 0.0
    anneal_decisions: int = 100000000
    checkpoint_seconds: float = 300
    snapshot_updates: int = 1000
    candidate_chunk: int = 32768
    bf16: bool = True
    tensorboard: bool = True
    log_envs: int = 4
    log_max_rounds: int = 10000
    action_mode: str = "canonical"
    model: dict[str, int] = field(default_factory=dict)

    def validate(self) -> None:
        positive = ("num_envs", "num_threads", "torch_threads", "rollout_steps",
                    "batch_size", "replay_capacity", "min_replay", "learn_steps",
                    "max_updates", "max_seconds", "learning_rate", "grad_clip",
                    "anneal_decisions", "checkpoint_seconds", "candidate_chunk", "snapshot_updates")
        for name in positive:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.min_replay > self.replay_capacity:
            raise ValueError("min_replay exceeds replay_capacity")
        if self.checkpoint_seconds > 600:
            raise ValueError("checkpoint_seconds must be at most 600")
        for name in ("epsilon_start", "epsilon_end", "greedy_start", "greedy_end"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        for name in ("hidden_weight", "finish_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if not 0 <= self.log_envs <= self.num_envs or self.log_max_rounds < 0:
            raise ValueError("invalid probe logging limits")
        if self.action_mode not in ("canonical", "full"):
            raise ValueError("action_mode must be canonical or full")
        ModelConfig(**self.model)


def schedule(start: float, end: float, decisions: int, horizon: int) -> float:
    return start + (end - start) * min(1.0, decisions / horizon)


def tensor(array: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    array = np.ascontiguousarray(array)
    if not array.flags.writeable:
        array = array.copy()
    value = torch.from_numpy(array)
    if device.type == "cuda":
        value = value.pin_memory()
    return value.to(device=device, dtype=dtype, non_blocking=device.type == "cuda")


class Trainer:
    def __init__(self, config: TrainConfig, run_dir: str | Path, device: str = "cpu",
                 resume: str | Path | None = None) -> None:
        config.validate()
        self.config = config
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("training currently supports cpu or cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if (self.run_dir / "latest.pt").exists() and resume is None:
            raise ValueError("run directory already has a checkpoint; use --resume")
        torch.set_num_threads(config.torch_threads)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.model = GuandanModel(ModelConfig(**config.model)).to(self.device)
        if (self.model.config.obs_dim, self.model.config.act_dim) != (gd.OBS_DIM, gd.ACT_DIM):
            raise ValueError("model feature dimensions do not match the engine")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.progress: dict[str, Any] = {
            "updates": 0, "decisions": 0, "rounds": 0, "matches": 0,
            "samples": 0, "logged_rounds": 0, "elapsed_seconds": 0.0, "resumes": 0,
            "snapshot_update": 0,
            "trained_samples": 0,
        }
        if resume:
            payload = load_checkpoint(resume, self.device)
            if payload.get("stage", "dmc") != "dmc":
                raise ValueError("DMC resume requires a Stage A checkpoint; A2 uses a different optimizer")
            # Runtime limits can change; changing labels, schedules or architecture
            # while resuming an optimizer silently would invalidate comparisons.
            mutable = {"max_updates", "max_seconds", "checkpoint_seconds", "snapshot_updates", "tensorboard",
                       "torch_threads", "num_threads", "log_envs", "log_max_rounds"}
            for key, value in asdict(config).items():
                if key not in mutable and value != payload["config"].get(key):
                    raise ValueError(f"resume config differs at {key}")
            self.model.load_state_dict(payload["model"])
            self.optimizer.load_state_dict(payload["optimizer"])
            self.progress.update(payload["progress"])
            restore_rng(payload["rng"], self.rng)
            self.progress["resumes"] += 1
        self.amp = (config.bf16 and self.device.type == "cuda"
                    and torch.cuda.is_bf16_supported())
        metadata = {
            "python": platform.python_version(), "torch": torch.__version__,
            "device": str(self.device), "bf16_enabled": self.amp,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "resume_source": str(resume) if resume else None,
        }
        (self.run_dir / f"runtime-{self.progress['resumes']:03d}.json").write_text(
            json.dumps(metadata, indent=2) + "\n")
        actions = gd.ActionConfig.full() if config.action_mode == "full" else gd.ActionConfig()
        # Environments and incomplete trajectories intentionally restart on resume.
        # Draw the seed from the restored RNG instead of replaying the original deals.
        self.env = gd.VecEnv(
            config.num_envs, num_threads=config.num_threads,
            seed=int(self.rng.integers(0, 2**63)), actions=actions,
            log_public_actions=config.log_envs > 0, log_env_limit=config.log_envs,
        )
        self.env.reset()
        self.replay = ReplayBuffer(config.replay_capacity, gd.OBS_DIM, gd.ACT_DIM)
        self.pending: dict[tuple[int, int, int], list[Decision]] = {}
        self.tokens: dict[tuple[int, int, int], list[np.ndarray]] = {}
        self.greedy_teams: dict[tuple[int, int], int] = {}
        self.stop_requested = False
        self.started = self.last_save = time.monotonic()
        self.prior_elapsed = self.progress["elapsed_seconds"]
        self.prior_decisions = self.progress["decisions"]
        self.collected_candidates = 0
        self.peak_batch_candidates = 0
        self.writer = None
        if config.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.run_dir / "tensorboard"))
        (self.run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.amp else nullcontext()

    def save(self, snapshot: bool = False) -> None:
        self.progress["elapsed_seconds"] = self.prior_elapsed + time.monotonic() - self.started
        if snapshot:
            self.progress["snapshot_update"] = self.progress["updates"]
        save_checkpoint(self.run_dir / "latest.pt", {
            "model_config": asdict(self.model.config), "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(), "config": asdict(self.config),
            "progress": dict(self.progress), "rng": rng_state(self.rng),
            "league": {"stage": "A", "opponents": ["self", "greedy"]},
        })
        if snapshot:
            snapshots = self.run_dir / "checkpoints"
            snapshots.mkdir(exist_ok=True)
            target = snapshots / f"step-{self.progress['updates']:09d}.pt"
            # latest is atomically replaced on its next save; this inode remains
            # immutable and costs no duplicate disk space until then.
            if not target.exists():
                os.link(self.run_dir / "latest.pt", target)
        self.last_save = time.monotonic()

    def should_stop(self) -> bool:
        return (self.stop_requested or self.progress["updates"] >= self.config.max_updates
                or time.monotonic() - self.started >= self.config.max_seconds)

    def maintenance(self) -> None:
        snapshot = self.progress["updates"] - self.progress["snapshot_update"] >= self.config.snapshot_updates
        if snapshot or time.monotonic() - self.last_save >= self.config.checkpoint_seconds:
            self.save(snapshot=snapshot)

    def collect(self) -> None:
        cfg = self.config
        self.model.eval()
        for _ in range(cfg.rollout_steps):
            if self.should_stop():
                break
            batch = self.env.pending()
            # pending() closes rounds and may emit forced passes before opening
            # the next one. Drain BEFORE storing any next-round decisions.
            for event in self.env.drain_public_actions():
                key = (int(event.env_id), int(event.match_id), int(event.round_index))
                self.tokens.setdefault(key, []).append(public_token(event))
            for result in self.env.drain_finished_rounds():
                key = (int(result.env_id), int(result.match_id), int(result.round_index))
                decisions = self.pending.pop(key, [])
                self.replay.add_round(decisions, result.seat_return, result.order)
                tokens = self.tokens.pop(key, [])
                if key[0] < cfg.log_envs and decisions and self.progress["logged_rounds"] < cfg.log_max_rounds:
                    index = self.progress["logged_rounds"]
                    group = f"{cfg.seed}:{self.progress['resumes']}:{key[0]}:{key[1]}"
                    save_round(self.run_dir / "belief" / f"round-{index:08d}.npz",
                               decisions, tokens, group)
                    self.progress["logged_rounds"] += 1
                self.progress["rounds"] += 1
                self.progress["samples"] += len(decisions)
                if result.match_winner >= 0:
                    self.progress["matches"] += 1
                    self.greedy_teams.pop(key[:2], None)
            n = batch.rows
            if not n:
                raise RuntimeError("environment produced no pending decisions")
            epsilon = schedule(cfg.epsilon_start, cfg.epsilon_end,
                               self.progress["decisions"], cfg.anneal_decisions)
            greedy_fraction = schedule(cfg.greedy_start, cfg.greedy_end,
                                       self.progress["decisions"], cfg.anneal_decisions)
            # Keep each opponent's identity fixed for a whole match.
            network = np.asarray(batch.phase) == 3
            keys = []
            for row in range(n):
                key = (int(batch.env_id[row]), int(batch.match_id[row]),
                       int(batch.round_index[row]))
                keys.append(key)
                if key[:2] not in self.greedy_teams:
                    self.greedy_teams[key[:2]] = (int(self.rng.integers(2))
                        if self.rng.random() < greedy_fraction else -1)
                network[row] &= int(batch.seat[row]) % 2 != self.greedy_teams[key[:2]]
            # Nonlearner and tribute rows use the C++ structural heuristic.
            choices = np.array(batch.greedy_choice, dtype=np.int32, copy=True)
            if network.any():
                # Keep ragged scoring batched; no Python loop over candidates.
                with torch.inference_mode(), self.autocast():
                    offsets = tensor(batch.offsets, self.device, torch.long)
                    values = self.model.score_candidates(
                        tensor(batch.obs, self.device, torch.float32),
                        tensor(batch.cand, self.device, torch.float32), offsets,
                        tensor(batch.phase, self.device, torch.long), cfg.candidate_chunk,
                        phase_code=3)
                    if not bool(torch.isfinite(values).all()):
                        raise FloatingPointError("non-finite rollout values")
                    selected = select_actions(values.float(), offsets, epsilon=epsilon).cpu().numpy()
                choices[network] = selected[network]
            for row in np.flatnonzero(network):
                key = keys[row]
                action_index = int(batch.offsets[row]) + int(choices[row])
                self.pending.setdefault(key, []).append(Decision(
                    obs=np.array(batch.obs[row], dtype=np.uint8, copy=True),
                    action=np.array(batch.cand[action_index], dtype=np.uint8, copy=True),
                    hidden=np.array(batch.hidden_counts[row], dtype=np.uint8, copy=True),
                    seat=int(batch.seat[row]), phase=3, prefix=len(self.tokens.get(key, [])),
                ))
            candidates = int(batch.offsets[-1])
            self.env.step(choices)
            self.progress["decisions"] += n
            self.collected_candidates += candidates
            self.peak_batch_candidates = max(self.peak_batch_candidates, candidates)
            self.maintenance()

    def learn(self) -> dict[str, float]:
        cfg = self.config
        sample = self.replay.sample(cfg.batch_size, self.rng)
        batch = {name: tensor(value, self.device, torch.float32 if name in
                 ("obs", "action", "returns") else torch.long)
                 for name, value in sample.items()}
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with self.autocast():
            out = self.model(batch["obs"], batch["action"], batch["phase"], phase_code=3)
            value_loss = F.mse_loss(out["q"].float(), batch["returns"])
            belief_loss = F.cross_entropy(out["hidden"].float().reshape(-1, 3), batch["hidden"].reshape(-1))
            finish_loss = F.cross_entropy(out["finish"].float(), batch["finish"])
            loss = value_loss + cfg.hidden_weight * belief_loss + cfg.finish_weight * finish_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite learner loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip,
                                             error_if_nonfinite=True)
        self.optimizer.step()
        self.progress["updates"] += 1
        self.progress["trained_samples"] += cfg.batch_size
        return {"loss": float(loss.detach()), "value_loss": float(value_loss.detach()),
                "belief_loss": float(belief_loss.detach()), "finish_loss": float(finish_loss.detach()),
                "grad_norm": float(grad), "replay_samples": len(self.replay)}

    def metric(self, values: dict[str, float]) -> None:
        elapsed = time.monotonic() - self.started
        self.progress["elapsed_seconds"] = self.prior_elapsed + elapsed
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = rss if sys.platform == "darwin" else rss * 1024
        record = {**self.progress, **values, "wall_seconds": elapsed,
                  "decisions_per_second": (self.progress["decisions"] - self.prior_decisions) / max(elapsed, 1e-9),
                  "mean_candidates_per_decision": self.collected_candidates / max(1, self.progress["decisions"] - self.prior_decisions),
                  "peak_batch_candidates": self.peak_batch_candidates,
                  "process_peak_rss_mib": rss_bytes / 2**20,
                  "epsilon": schedule(self.config.epsilon_start, self.config.epsilon_end,
                                      self.progress["decisions"], self.config.anneal_decisions)}
        if self.device.type == "cuda":
            record.update({"cuda_allocated_mib": torch.cuda.memory_allocated(self.device) / 2**20,
                           "cuda_reserved_mib": torch.cuda.memory_reserved(self.device) / 2**20,
                           "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(self.device) / 2**20,
                           "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(self.device) / 2**20})
        with (self.run_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if self.writer:
            for key, value in record.items():
                self.writer.add_scalar(key, value, self.progress["updates"])
        print(json.dumps(record, allow_nan=False), flush=True)

    def run(self) -> dict[str, Any]:
        def stop(_signum, _frame):
            self.stop_requested = True
        handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        completed = False
        try:
            self.save(snapshot=self.progress["updates"] == 0)
            while not self.should_stop():
                collect_start = time.monotonic()
                self.collect()
                collect_seconds = time.monotonic() - collect_start
                learn_start = time.monotonic()
                metrics = {}
                if len(self.replay) >= self.config.min_replay:
                    for _ in range(self.config.learn_steps):
                        if self.should_stop():
                            break
                        metrics = self.learn()
                        self.maintenance()
                    # Bound staleness: completed samples are used in one learner
                    # phase only. Incomplete rounds keep their recorded actions.
                    self.replay.clear()
                self.metric({**metrics, "collect_seconds": collect_seconds,
                             "learn_seconds": time.monotonic() - learn_start})
            completed = True
            return self.progress
        finally:
            # On an unexpected numerical/I/O failure preserve the last known
            # checkpoint. Saving possibly poisoned parameters would defeat resume.
            try:
                if completed:
                    self.save()
            finally:
                if self.writer:
                    self.writer.close()
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="train/configs/smoke.json")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args(argv)
    data = json.loads(Path(args.config).read_text())
    for field_name in ("max_updates", "max_seconds"):
        value = getattr(args, field_name)
        if value is not None:
            data[field_name] = value
    trainer = Trainer(TrainConfig(**data), args.run_dir, args.device, args.resume)
    trainer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
