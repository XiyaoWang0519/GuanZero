"""Stage B PPO learner (DESIGN 8.4, STAGE_B_TODO B5).

One iteration ("update") is a rollout of `rollout_steps` vector steps followed
by `epochs` passes of clipped-surrogate PPO over the rounds that completed.

Rollout. A `gd.VecEnv` in which the learner controls both seats of team
`e % 2` in environment `e`. Rows of the other team go to an `OpponentSource`
(`train/opponents.py`), driven in the contract's order after every
`pending()`: drain finished rounds and report ended matches, report started
matches (changed `match_id`), and only then build `OpponentRows` from the
batch. Learner play rows are sampled by `StageBPolicy.act` over the frozen-M1
top-k-plus-pass set and stored in the B3 `RolloutBuffer` with their behaviour
log-probability, the PRUNED candidate list and the pruned index, so the
learner re-evaluates exactly the set that was sampled from. Tribute and
back-tribute rows follow `tribute_policy`: "heuristic" (default, A2 decision)
plays the engine heuristic and stores nothing, "learned" samples them from the
tribute heads and stores them as ordinary trajectory steps.

Learning. The perfect-information critic (B2) is refreshed on every completed
step with the current critic weights, then GAE with discount `gamma` and
`gae_lambda` over whole rounds, the round-end team return as the only reward.
Policy loss: clipped surrogate, minus `entropy_coef` times entropy, plus
`kl_coef` (linearly annealed to zero over `kl_anneal_updates`) times
KL(pi || pi_M1) on the pruned set, where pi_M1 = softmax(Q_M1 / temperature),
plus the DMC auxiliary hidden-hand and finish losses on the policy network.
The critic is trained jointly on MSE to the GAE returns with its own optimizer.

Checkpoints use the `dmc.py` layout (`latest.pt`, hard-linked snapshots in
`checkpoints/`, `metrics.jsonl`, `config.json`, `runtime-*.json`) and write a
Stage B payload (`stage="ppo"`) that `eval.policies.load_policy` plays. As in
`dmc.py`, environments and in-progress rounds restart on resume.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
from train.ckpt import load_checkpoint, restore_rng, rng_state, save_checkpoint
from train.critic import Critic, CriticConfig, load_critic
from train.dmc import schedule, tensor
from train.model import GuandanModel, ModelConfig
from train.opponents import FrozenModelOpponent, GreedyOpponent, OpponentRows, OpponentSource
from train.policy import (PolicyConfig, StageBPolicy, gather_segments, policy_from_payload,
                          segment_entropy, segment_log_softmax, segment_rows)
from train.rollout_buffer import HIDDEN_DIM, RolloutBuffer, RolloutBufferConfig, _ragged_index

PLAY = int(gd.Phase.Play)
M1_FINAL = ".work/runpod/artifacts/pilot/final.pt"
CRITIC_FIT = ".work/critic-fit/perfect/best.pt"


@dataclass
class PPOConfig:
    seed: int = 7
    # Initial policy (a Stage A/A2 checkpoint) and the frozen pruning/KL reference.
    init_checkpoint: str = M1_FINAL
    # B2 critic weights; empty string starts a fresh critic of critic_width x critic_layers.
    critic_init: str = CRITIC_FIT
    critic_width: int = 512
    critic_layers: int = 4
    # "frozen" (the init checkpoint, argmax Q), "frozen:<path>", or "greedy".
    opponent: str = "frozen"
    tribute_policy: str = "heuristic"
    action_mode: str = "canonical"
    num_envs: int = 1024
    num_threads: int = 8
    torch_threads: int = 4
    rollout_steps: int = 128
    epochs: int = 4
    minibatch_size: int = 4096
    clip: float = 0.2
    policy_lr: float = 1e-5
    critic_lr: float = 1e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    entropy_coef: float = 0.01
    kl_coef: float = 0.1
    kl_anneal_updates: int = 1000
    target_kl: float = 0.0         # stop the epochs early past this approx KL; 0 disables
    value_coef: float = 1.0
    hidden_weight: float = 0.1
    finish_weight: float = 0.1
    grad_clip: float = 10.0
    temperature: float = 0.02      # measured: M1 sampled at 0.02 is within noise of argmax M1 (B5 report)
    top_k: int = 32
    candidate_chunk: int = 32768
    # Rollout buffer capacities; 0 derives them from num_envs and rollout_steps.
    buffer_steps: int = 0
    buffer_candidates: int = 0
    buffer_trajectories: int = 0
    checkpoint_seconds: float = 300
    snapshot_updates: int = 100
    max_updates: int = 100000
    max_seconds: float = 86400
    tensorboard: bool = True

    def validate(self) -> None:
        positive = ("num_envs", "num_threads", "torch_threads", "rollout_steps", "epochs",
                    "minibatch_size", "clip", "kl_anneal_updates", "grad_clip", "temperature",
                    "top_k", "candidate_chunk", "checkpoint_seconds", "snapshot_updates",
                    "max_updates", "max_seconds", "critic_width", "critic_layers")
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        nonnegative = ("policy_lr", "critic_lr", "entropy_coef", "kl_coef", "target_kl",
                       "value_coef", "hidden_weight", "finish_weight", "buffer_steps",
                       "buffer_candidates", "buffer_trajectories")
        for name in nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if self.num_envs < 2:
            raise ValueError("num_envs must be at least 2 so both learner teams are covered")
        if self.clip >= 1:
            raise ValueError("clip must be in (0, 1)")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma must be in (0, 1] and gae_lambda in [0, 1]")
        if self.checkpoint_seconds > 600:
            raise ValueError("checkpoint_seconds must be at most 600")
        if self.tribute_policy not in ("heuristic", "learned"):
            raise ValueError("tribute_policy must be heuristic or learned")
        if self.action_mode not in ("canonical", "full"):
            raise ValueError("action_mode must be canonical or full")
        if not (self.opponent in ("frozen", "greedy")
                or (self.opponent.startswith("frozen:") and len(self.opponent) > 7)):
            raise ValueError("opponent must be frozen, frozen:<path> or greedy")
        if not self.init_checkpoint:
            raise ValueError("init_checkpoint is required")

    def buffer_config(self) -> RolloutBufferConfig:
        # About half of all rows are learner rows; a round in progress at the
        # iteration boundary is carried over, so leave room for one per team.
        steps = self.buffer_steps or self.num_envs * (self.rollout_steps + 128)
        return RolloutBufferConfig(
            num_envs=self.num_envs, obs_dim=gd.OBS_DIM, act_dim=gd.ACT_DIM,
            max_steps=steps,
            max_candidates=self.buffer_candidates or steps * (self.top_k + 1),
            max_trajectories=self.buffer_trajectories or self.num_envs * 2 * (self.rollout_steps // 8 + 2),
            carry_over=True)


# Fields that may change on resume: runtime limits and resources only.
MUTABLE_ON_RESUME = {"max_updates", "max_seconds", "checkpoint_seconds", "snapshot_updates",
                     "tensorboard", "torch_threads", "num_threads", "init_checkpoint", "critic_init"}


def resolve_artifact(path: str | Path) -> Path:
    """A relative `.work/...` path also resolves in the main checkout when this
    runs from a git worktree below it."""
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    for parent in Path(__file__).resolve().parents:
        if (parent / path).exists():
            return parent / path
    raise FileNotFoundError(f"{path} not found here or in any parent checkout")


def segment_sum(values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    rows = segment_rows(offsets, len(values))
    return values.new_zeros(len(offsets) - 1).index_add(0, rows, values)


def policy_terms(policy: StageBPolicy, obs: torch.Tensor, cand: torch.Tensor,
                 offsets: torch.Tensor, phase: torch.Tensor, chosen: torch.Tensor,
                 phase_code: int | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`StageBPolicy.evaluate` plus the full pruned log-probabilities (for the
    KL term) from one forward pass. Returns (log_prob, entropy, log_probs)."""
    log_probs = segment_log_softmax(policy.logits(obs, cand, offsets, phase, phase_code), offsets)
    return (gather_segments(log_probs, offsets, chosen),
            segment_entropy(log_probs, offsets), log_probs)


def reference_kl(policy: StageBPolicy, log_probs: torch.Tensor, obs: torch.Tensor,
                 cand: torch.Tensor, offsets: torch.Tensor, phase: torch.Tensor,
                 phase_code: int | None) -> torch.Tensor:
    """Per-decision KL(pi || softmax(Q_ref / t)) over the stored pruned set."""
    with torch.no_grad():
        ref = policy.reference.score_candidates(obs, cand, offsets, phase,
                                                chunk_size=policy.config.chunk_size,
                                                phase_code=phase_code)
        ref_log_probs = segment_log_softmax(ref / policy.config.temperature, offsets)
    return segment_sum(log_probs.exp() * (log_probs - ref_log_probs), offsets)


class PPOTrainer:
    def __init__(self, config: PPOConfig, run_dir: str | Path, device: str = "cpu",
                 resume: str | Path | None = None,
                 opponent: OpponentSource | None = None) -> None:
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
        policy_config = PolicyConfig(temperature=config.temperature, top_k=config.top_k,
                                     chunk_size=config.candidate_chunk)
        self.progress: dict[str, Any] = {
            "updates": 0, "optimizer_steps": 0, "decisions": 0, "learner_decisions": 0,
            "rounds": 0, "matches": 0, "learner_match_wins": 0, "samples": 0,
            "elapsed_seconds": 0.0, "resumes": 0, "snapshot_update": 0,
        }
        payload = None
        if resume:
            payload = load_checkpoint(resume, self.device)
            if payload.get("stage") != "ppo":
                raise ValueError("PPO resume requires a Stage B (ppo) checkpoint")
            for key, value in asdict(config).items():
                if key not in MUTABLE_ON_RESUME and value != payload["config"].get(key):
                    raise ValueError(f"resume config differs at {key}")
            self.policy = policy_from_payload(payload, self.device)
            if self.policy.config != policy_config:
                raise ValueError("resume policy config differs")
            self.critic = Critic(CriticConfig(**payload["critic_model_config"]))
            self.critic.load_state_dict(payload["critic"])
            self.init_source = payload.get("init_source")
            self.critic_source = payload.get("critic_source")
        else:
            init_path = resolve_artifact(config.init_checkpoint)
            init = load_checkpoint(init_path, self.device)
            if init.get("stage", "dmc") not in ("dmc", "a2"):
                raise ValueError("Stage B starts from a Stage A or A2 checkpoint")
            if init.get("config", {}).get("action_mode", "canonical") != config.action_mode:
                raise ValueError("init checkpoint action_mode differs from the config")
            if config.tribute_policy == "learned" and init.get("tribute_policy") != "learned":
                raise ValueError("learned tribute starts from an A2 checkpoint's fitted heads")
            from eval.policies import model_digest
            model = GuandanModel(ModelConfig(**init["model_config"]))
            model.load_state_dict(init["model"])
            self.policy = StageBPolicy.from_model(model, policy_config,
                                                  model_digest(init["model"])).to(self.device)
            self.init_source = str(init_path)
            if config.critic_init:
                self.critic_source = str(resolve_artifact(config.critic_init))
                self.critic, _ = load_critic(self.critic_source, str(self.device))
            else:
                self.critic_source = None
                self.critic = Critic(CriticConfig(width=config.critic_width,
                                                  layers=config.critic_layers))
        self.net = self.policy.net
        self.critic.to(self.device)
        dims = (self.net.config.obs_dim, self.net.config.act_dim, self.critic.config.input_dim)
        if dims != (gd.OBS_DIM, gd.ACT_DIM, gd.OBS_DIM + HIDDEN_DIM):
            raise ValueError("policy or critic dimensions do not match the engine")
        self.policy_optimizer = torch.optim.Adam(self.net.parameters(), lr=config.policy_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.generator = torch.Generator(device=self.device)
        if payload is not None:
            self.policy_optimizer.load_state_dict(payload["optimizer"])
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
            self.progress.update(payload["progress"])
            restore_rng(payload["rng"], self.rng)
            self.generator.set_state(payload["sampler"].cpu())
            self.progress["resumes"] += 1
        else:
            self.generator.manual_seed(int(self.rng.integers(0, 2**63)))
        self.phase_code = PLAY if config.tribute_policy == "heuristic" else None
        metadata = {
            "python": platform.python_version(), "torch": torch.__version__,
            "device": str(self.device), "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "policy_parameters": sum(p.numel() for p in self.net.parameters()),
            "critic_parameters": sum(p.numel() for p in self.critic.parameters()),
            "reference_checkpoint_id": self.policy.reference_checkpoint_id,
            "init_source": self.init_source, "critic_source": self.critic_source,
            "resume_source": str(resume) if resume else None,
        }
        (self.run_dir / f"runtime-{self.progress['resumes']:03d}.json").write_text(
            json.dumps(metadata, indent=2) + "\n")
        actions = gd.ActionConfig.full() if config.action_mode == "full" else gd.ActionConfig()
        # As in dmc.py, environments restart on resume from a seed drawn from the restored RNG.
        self.env = gd.VecEnv(config.num_envs, num_threads=config.num_threads,
                             seed=int(self.rng.integers(0, 2**63)), actions=actions)
        self.env.reset()
        self.learner_team = np.arange(config.num_envs, dtype=np.int64) % 2
        self.env_match = np.full(config.num_envs, -1, np.int64)
        self.buffer = RolloutBuffer(config.buffer_config())
        # Finish position of every seat per trajectory, for the auxiliary finish loss.
        self.traj_finish = np.zeros((self.buffer.config.max_trajectories, 4), np.int64)
        self.opponent = opponent if opponent is not None else self._default_opponent()
        self.opponent.bind(self.env, self.learner_team)
        self.bound = False
        self.stop_requested = False
        self.started = self.last_save = time.monotonic()
        self.prior_elapsed = self.progress["elapsed_seconds"]
        self.prior_decisions = self.progress["decisions"]
        self.window = {"matches": 0, "wins": 0, "rounds": 0, "learner_return": 0.0}
        self.writer = None
        if config.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.run_dir / "tensorboard"))
        (self.run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    def _default_opponent(self) -> OpponentSource:
        spec = self.config.opponent
        if spec == "greedy":
            return GreedyOpponent()
        if spec == "frozen":
            # The frozen M1 reference is already on the device and never trained.
            return FrozenModelOpponent(self.policy.reference, str(self.device),
                                       self.config.candidate_chunk)
        return FrozenModelOpponent.from_checkpoint(resolve_artifact(spec[len("frozen:"):]),
                                                   str(self.device), self.config.candidate_chunk)

    # ----------------------------------------------------------------- rollout
    def _finish_rounds(self, results) -> None:
        ended, won = [], []
        for result in results:
            env = int(result.env_id)
            team = int(self.learner_team[env])
            slot = int(self.buffer.open_traj[env, team])
            if slot >= 0:
                self.traj_finish[slot, list(result.order)] = np.arange(4)
            self.buffer.finish_round(env, int(result.match_id), int(result.round_index),
                                     result.seat_return)
            self.progress["rounds"] += 1
            self.window["rounds"] += 1
            self.window["learner_return"] += float(result.seat_return[team])
            if result.match_winner >= 0:
                learner_won = int(result.match_winner) == team
                ended.append(env)
                won.append(learner_won)
                self.progress["matches"] += 1
                self.progress["learner_match_wins"] += learner_won
                self.window["matches"] += 1
                self.window["wins"] += learner_won
        if ended:
            self.opponent.on_match_end(np.asarray(ended, np.int32), np.asarray(won, bool))

    def _start_matches(self, env_id: np.ndarray, match_id: np.ndarray) -> None:
        if not self.bound:
            # Contract: every environment starts a match right after bind.
            self.env_match.fill(-1)
            self.env_match[env_id] = match_id
            self.opponent.on_match_start(np.arange(self.config.num_envs, dtype=np.int32))
            self.bound = True
            return
        changed = match_id != self.env_match[env_id]
        if changed.any():
            envs = np.unique(env_id[changed])
            self.env_match[env_id[changed]] = match_id[changed]
            self.opponent.on_match_start(envs.astype(np.int32))

    @staticmethod
    def _ragged_rows(batch, rows: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Candidates and rebased offsets of a subset of batch rows."""
        if rows.size == len(offsets) - 1:
            return np.asarray(batch.cand), offsets
        counts = offsets[rows + 1] - offsets[rows]
        cand = np.asarray(batch.cand)[_ragged_index(offsets[rows], counts, int(counts.sum()))]
        local = np.zeros(rows.size + 1, np.int64)
        np.cumsum(counts, out=local[1:])
        return cand, local

    def _opponent_rows(self, batch, rows: np.ndarray, offsets: np.ndarray,
                       env_id: np.ndarray, seat: np.ndarray, phase: np.ndarray,
                       match_id: np.ndarray) -> OpponentRows:
        cand, local = self._ragged_rows(batch, rows, offsets)
        # styled_choice is read only now, after on_match_start (set_styles may
        # have recomputed it for rows already pending).
        return OpponentRows(
            obs=np.asarray(batch.obs)[rows], cand=cand, offsets=local.astype(np.int32),
            env_id=env_id[rows], seat=seat[rows], phase=phase[rows], match_id=match_id[rows],
            greedy_choice=np.asarray(batch.greedy_choice)[rows],
            styled_choice=np.asarray(batch.styled_choice)[rows])

    def _learner_act(self, batch, rows: np.ndarray, offsets: np.ndarray, env_id: np.ndarray,
                     seat: np.ndarray, phase: np.ndarray, match_id: np.ndarray,
                     choices: np.ndarray) -> None:
        cand, local = self._ragged_rows(batch, rows, offsets)
        obs = np.asarray(batch.obs)[rows]
        with torch.inference_mode():
            step = self.policy.act(tensor(obs, self.device, torch.float32),
                                   tensor(cand, self.device, torch.float32),
                                   tensor(local, self.device, torch.long),
                                   tensor(phase[rows], self.device, torch.long),
                                   generator=self.generator, phase_code=self.phase_code)
        keep = step.keep_index.cpu().numpy()
        choices[rows] = step.choice.cpu().numpy()
        self.buffer.add_batch(
            learner=np.ones(rows.size, bool), env_id=env_id[rows], match_id=match_id[rows],
            round_index=np.asarray(batch.round_index)[rows], seat=seat[rows], phase=phase[rows],
            obs=obs, hidden_counts=np.asarray(batch.hidden_counts)[rows],
            cand=cand[keep], offsets=step.pruned_offsets.cpu().numpy(),
            chosen=step.pruned_choice.cpu().numpy(), logp=step.log_prob.float().cpu().numpy())
        self.progress["learner_decisions"] += int(rows.size)

    def collect(self) -> None:
        self.net.eval()
        heuristic = self.config.tribute_policy == "heuristic"
        for _ in range(self.config.rollout_steps):
            if self.should_stop():
                break
            batch = self.env.pending()
            # Contract order: ended matches, then started matches, then rows.
            self._finish_rounds(self.env.drain_finished_rounds())
            n = batch.rows
            if not n:
                raise RuntimeError("environment produced no pending decisions")
            env_id = np.asarray(batch.env_id, np.int64)
            match_id = np.asarray(batch.match_id, np.int64)
            self._start_matches(env_id, match_id)
            seat = np.asarray(batch.seat, np.int64)
            phase = np.asarray(batch.phase, np.int64)
            offsets = np.asarray(batch.offsets, np.int64)
            learner = seat % 2 == self.learner_team[env_id]
            choices = np.array(batch.greedy_choice, dtype=np.int32, copy=True)
            opponent_rows = np.flatnonzero(~learner)
            if opponent_rows.size:
                picked = np.asarray(self.opponent.act(self._opponent_rows(
                    batch, opponent_rows, offsets, env_id, seat, phase, match_id)), np.int32)
                if picked.shape != opponent_rows.shape:
                    raise ValueError("opponent returned the wrong number of choices")
                choices[opponent_rows] = picked
            acting = learner & (phase == PLAY) if heuristic else learner
            learner_rows = np.flatnonzero(acting)
            if learner_rows.size:
                self._learner_act(batch, learner_rows, offsets, env_id, seat, phase,
                                  match_id, choices)
            counts = offsets[1:] - offsets[:-1]
            if ((choices < 0) | (choices >= counts)).any():
                raise ValueError("a choice lies outside its candidate list")
            self.env.step(choices)
            self.progress["decisions"] += n

    # ----------------------------------------------------------------- learner
    @torch.no_grad()
    def refresh_values(self) -> np.ndarray:
        """Critic values of every completed step, written into the buffer."""
        steps = self.buffer.pending_value_steps()
        self.critic.eval()
        chunk = max(1, self.config.minibatch_size)
        for begin in range(0, steps.size, chunk):
            part = steps[begin:begin + chunk]
            x = tensor(self.buffer.critic_input(part), self.device, torch.float32)
            self.buffer.value[part] = self.critic(x).float().cpu().numpy()
        return steps

    def kl_coef(self) -> float:
        return schedule(self.config.kl_coef, 0.0, self.progress["updates"],
                        self.config.kl_anneal_updates)

    def learn(self) -> dict[str, float]:
        cfg = self.config
        self.refresh_values()
        samples = self.buffer.finalize(cfg.gamma, cfg.gae_lambda)
        stats: dict[str, float] = {"update_samples": samples, "kl_coef": self.kl_coef()}
        if samples == 0:
            return stats
        index = self.buffer.samples[:samples]
        returns, values = self.buffer.returns[index], self.buffer.value[index]
        variance = float(np.var(returns))
        stats["explained_variance"] = (1.0 - float(np.var(returns - values)) / variance
                                       if variance > 0 else 0.0)
        stats["mean_return"] = float(returns.mean())
        stats["mean_value"] = float(values.mean())
        kl_coef = stats["kl_coef"]
        sums: dict[str, float] = {}
        count = 0
        first_ratio_deviation = None
        stopped_epoch = cfg.epochs
        self.net.train()
        self.critic.train()
        for epoch in range(cfg.epochs):
            epoch_kl, epoch_batches = 0.0, 0
            for mb in self.buffer.minibatches(cfg.minibatch_size, self.rng, self.device):
                terms = self.minibatch_loss(mb, kl_coef)
                self.policy_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                (terms["policy_total"] + cfg.value_coef * terms["value_loss"]).backward()
                policy_grad = torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), cfg.grad_clip, error_if_nonfinite=True)
                critic_grad = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), cfg.grad_clip, error_if_nonfinite=True)
                self.policy_optimizer.step()
                self.critic_optimizer.step()
                self.progress["optimizer_steps"] += 1
                values_now = {k: float(v.detach()) for k, v in terms.items() if v.dim() == 0}
                values_now["policy_grad_norm"] = float(policy_grad)
                values_now["critic_grad_norm"] = float(critic_grad)
                if first_ratio_deviation is None:
                    first_ratio_deviation = float(terms["ratio_deviation"])
                for key, value in values_now.items():
                    sums[key] = sums.get(key, 0.0) + value
                count += 1
                epoch_kl += values_now["approx_kl"]
                epoch_batches += 1
            if cfg.target_kl and epoch_kl / max(1, epoch_batches) > cfg.target_kl:
                stopped_epoch = epoch + 1
                break
        stats.update({key: value / count for key, value in sums.items()})
        stats.pop("policy_total", None)
        stats["first_ratio_max_deviation"] = first_ratio_deviation
        stats["epochs_run"] = stopped_epoch
        self.progress["samples"] += samples
        return stats

    def minibatch_loss(self, mb: dict[str, torch.Tensor], kl_coef: float) -> dict[str, torch.Tensor]:
        cfg = self.config
        log_prob, entropy, log_probs = policy_terms(
            self.policy, mb["obs"], mb["cand"], mb["offsets"], mb["phase"], mb["chosen"],
            self.phase_code)
        log_ratio = log_prob - mb["logp"]
        ratio = log_ratio.exp()
        advantage = mb["advantage"]
        if cfg.normalize_advantages and len(advantage) > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        surrogate = -torch.minimum(ratio * advantage,
                                   ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advantage).mean()
        kl_ref = reference_kl(self.policy, log_probs, mb["obs"], mb["cand"], mb["offsets"],
                              mb["phase"], self.phase_code).mean()
        state = self.net.state_tower(mb["obs"])
        aux = self.net.auxiliary(state)
        hidden_target = mb["critic_obs"][:, gd.OBS_DIM:].long().reshape(-1)
        hidden_loss = F.cross_entropy(aux["hidden"].reshape(-1, 3), hidden_target)
        steps = mb["steps"].cpu().numpy()
        finish_target = self.traj_finish[self.buffer.traj[steps], self.buffer.seat[steps]]
        finish_loss = F.cross_entropy(aux["finish"],
                                      torch.as_tensor(finish_target, device=self.device))
        value_loss = F.mse_loss(self.critic(mb["critic_obs"]), mb["returns"])
        policy_total = (surrogate - cfg.entropy_coef * entropy.mean() + kl_coef * kl_ref
                        + cfg.hidden_weight * hidden_loss + cfg.finish_weight * finish_loss)
        if not bool(torch.isfinite(policy_total)) or not bool(torch.isfinite(value_loss)):
            raise FloatingPointError("non-finite PPO loss")
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = ((ratio - 1).abs() > cfg.clip).float().mean()
            ratio_deviation = (ratio - 1).abs().max()
        return {"policy_total": policy_total, "policy_loss": surrogate, "value_loss": value_loss,
                "entropy": entropy.mean(), "kl_ref": kl_ref, "hidden_loss": hidden_loss,
                "finish_loss": finish_loss, "approx_kl": approx_kl,
                "clip_fraction": clip_fraction, "ratio_deviation": ratio_deviation}

    # -------------------------------------------------------------- bookkeeping
    def save(self, snapshot: bool = False) -> None:
        self.progress["elapsed_seconds"] = self.prior_elapsed + time.monotonic() - self.started
        if snapshot:
            self.progress["snapshot_update"] = self.progress["updates"]
        payload = self.policy.checkpoint_payload(
            optimizer=self.policy_optimizer.state_dict(), config=asdict(self.config),
            progress=dict(self.progress), rng=rng_state(self.rng),
            tribute_policy=self.config.tribute_policy)
        payload.update({
            "critic_model_config": asdict(self.critic.config), "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "sampler": self.generator.get_state(),
            "init_source": self.init_source, "critic_source": self.critic_source,
            "league": {"stage": "B", "opponents": [getattr(self.opponent, "name",
                                                           type(self.opponent).__name__)]},
        })
        save_checkpoint(self.run_dir / "latest.pt", payload)
        if snapshot:
            snapshots = self.run_dir / "checkpoints"
            snapshots.mkdir(exist_ok=True)
            target = snapshots / f"step-{self.progress['updates']:09d}.pt"
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

    def metric(self, values: dict[str, Any]) -> dict[str, Any]:
        elapsed = time.monotonic() - self.started
        self.progress["elapsed_seconds"] = self.prior_elapsed + elapsed
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = rss if sys.platform == "darwin" else rss * 1024
        window = self.window
        record = {**self.progress, **values, "wall_seconds": elapsed,
                  "decisions_per_second": (self.progress["decisions"] - self.prior_decisions) / max(elapsed, 1e-9),
                  "window_matches": window["matches"], "window_rounds": window["rounds"],
                  "learner_match_win_rate": window["wins"] / window["matches"] if window["matches"] else None,
                  "learner_round_return": window["learner_return"] / window["rounds"] if window["rounds"] else None,
                  "cumulative_learner_match_win_rate": (self.progress["learner_match_wins"] / self.progress["matches"]
                                                        if self.progress["matches"] else None),
                  "process_peak_rss_mib": rss_bytes / 2**20}
        if self.device.type == "cuda":
            record["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated(self.device) / 2**20
        self.window = {"matches": 0, "wins": 0, "rounds": 0, "learner_return": 0.0}
        with (self.run_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if self.writer:
            for key, value in record.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.writer.add_scalar(key, value, self.progress["updates"])
        print(json.dumps(record, allow_nan=False), flush=True)
        return record

    def update(self) -> dict[str, Any]:
        """One PPO iteration: rollout, learn on completed rounds, clear."""
        collect_start = time.monotonic()
        self.collect()
        collect_seconds = time.monotonic() - collect_start
        learn_start = time.monotonic()
        stats = self.learn()
        self.buffer.next_iteration()
        self.progress["updates"] += 1
        return self.metric({**stats, "collect_seconds": collect_seconds,
                            "learn_seconds": time.monotonic() - learn_start})

    def run(self) -> dict[str, Any]:
        def stop(_signum, _frame):
            self.stop_requested = True
        handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        completed = False
        try:
            self.save(snapshot=self.progress["updates"] == 0)
            while not self.should_stop():
                self.update()
                self.maintenance()
            completed = True
            return self.progress
        finally:
            try:
                if completed:
                    self.save()
            finally:
                if self.writer:
                    self.writer.close()
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)


def load_config(path: str | Path, **overrides: Any) -> PPOConfig:
    data = json.loads(Path(path).read_text())
    data.update({k: v for k, v in overrides.items() if v is not None})
    return PPOConfig(**data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="train/configs/ppo-smoke.json")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args(argv)
    config = load_config(args.config, max_updates=args.max_updates, max_seconds=args.max_seconds)
    PPOTrainer(config, args.run_dir, args.device, args.resume).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
