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
import copy
from dataclasses import asdict, dataclass, fields
import json
import math
import signal
from pathlib import Path
import sys
import time
from typing import Any, Iterator
import uuid

import gd
import numpy as np
import torch
from torch.nn import functional as F
from infra.history_artifacts import engine_digest, source_identity, sha256

from train.ckpt import restore_rng, rng_state
from train.history_model import (STAGE, TOKEN_SCHEMA_VERSION,
                                 HistoryPolicyConfig, checkpoint_payload, count_parameters,
                                 fresh_player, load_history_checkpoint, save_history_checkpoint)
from train.history_rollout import (HistoryCollector, MatchEventStore, SequenceRolloutBuffer)
from train.history_population import HistoryPopulation
from train.history_response import RESPONSE_SCHEMA, opponent_response_labels
from train.history_transfers import runtime_settings
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
    response_mode: str = "none"
    response_coef: float = 0.1
    # rollout
    num_envs: int = 16
    num_threads: int = 1
    steps_per_update: int = 64
    seed: int = 0
    causal_sdpa: bool = False           # opt-in until same-device CUDA A/B acceptance
    rollout_kv_cache: bool = False      # public-only, invalidated across learner updates
    rollout_batched_attention: bool = False  # opt-in; target-device bitwise acceptance required
    rollout_wide_projection: bool = False  # with batched attention: one q/out GEMM; FP32, not bitwise
    rollout_private_graphs: bool = False  # bounded CUDA inference graphs; sampling stays eager
    rollout_graph_budget_mb: int = 512
    rollout_graph_policy_budget_mb: int = 128  # per-policy cap inside the total budget
    rollout_triton_cache: bool = False  # lossless KV update/packing; requires Triton
    rollout_triton_min_batch: int = 1
    profile_collection: bool = False   # synchronized phase timings; diagnostic only
    rollout_device: str | None = None  # None shares the learner device
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
    # exploration floor, learner seats only: pi_b = (1 - eps) softmax(logits / T) + eps / n
    rollout_temperature: float = 1.0
    rollout_epsilon: float = 0.0
    behaviour_weight_cap: float = 1.0
    # lifecycle
    updates: int = 1
    checkpoint_updates: int = 1
    torch_threads: int = 0               # 0 leaves torch's default
    snapshot_updates: int = 2           # 0 disables for isolated throughput sweeps
    # historical opponent archive; 0 keeps only the recent snapshots
    population_archive_every: int = 0
    population_archive_size: int = 16
    population_archive_share: float = 0.5
    population_recent: int = 4
    snapshot_probability: float = 0.5

    def __post_init__(self) -> None:
        if self.rollout_graph_budget_mb < 1 or self.rollout_graph_policy_budget_mb < 1:
            raise ValueError('rollout graph memory budget must be positive')
        if self.rollout_triton_min_batch < 1:
            raise ValueError('rollout Triton minimum batch must be positive')
        if (self.rollout_private_graphs or self.rollout_triton_cache) and not self.rollout_kv_cache:
            raise ValueError('CUDA rollout optimizations require rollout_kv_cache')
        if self.rollout_wide_projection and not self.rollout_batched_attention:
            raise ValueError('rollout_wide_projection requires rollout_batched_attention')
        if self.rollout_device not in (None, 'cpu', 'cuda'):
            raise ValueError('rollout_device must be cpu, cuda, or None')
        if min(self.num_envs, self.steps_per_update, self.epochs, self.minibatch_matches,
               self.checkpoint_updates) <= 0:
            raise ValueError("environment, step, epoch, minibatch and checkpoint counts "
                             "must be positive")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma must be in (0, 1] and gae_lambda in [0, 1]")
        if self.lr <= 0 or self.clip <= 0 or self.entropy < 0:
            raise ValueError("lr and clip must be positive; entropy must not be negative")
        if (not 0 < self.rollout_temperature < math.inf or not 0 <= self.rollout_epsilon <= 1
                or not 0 < self.behaviour_weight_cap < math.inf):
            raise ValueError("rollout_temperature and behaviour_weight_cap must be positive "
                             "and finite; rollout_epsilon must be in [0, 1]")
        if (self.snapshot_updates < 0 or self.population_recent < 1
                or not 0 <= self.snapshot_probability <= 1
                or self.population_archive_every < 0 or self.population_archive_size < 1
                or not 0 <= self.population_archive_share <= 1):
            raise ValueError("invalid population schedule")
        if self.rollout_kv_cache and self.window:
            raise ValueError("rollout KV cache requires full history")
        if self.response_mode not in ("none", "auxiliary", "explicit"):
            raise ValueError("unknown response_mode")
        if not math.isfinite(self.response_coef) or self.response_coef < 0:
            raise ValueError("response_coef must be finite and nonnegative")
        if self.response_mode != "none" and self.response_coef == 0:
            raise ValueError("prediction arms require a positive response_coef")

    def policy_config(self) -> HistoryPolicyConfig:
        return HistoryPolicyConfig(width=self.width, layers=self.layers, heads=self.heads,
                                   window=self.window, max_rounds=self.max_rounds,
                                   response_mode=self.response_mode)

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
                 "ratio_deviation", "actor_grad_norm", "encoder_grad_norm", "critic_grad_norm",
                 "response_loss", "response_accuracy", "response_event_fraction",
                 "behaviour_weight_mean", "behaviour_weight_cap_fraction")

    def __init__(self, config: HistoryPPOConfig, output: str | Path, device: str = "cpu",
                 resume: str | Path | None = None, allow_source_change: bool = False) -> None:
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
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
        self.rollout_device = torch.device(config.rollout_device or self.device)
        if self.rollout_device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA rollout requested but CUDA is unavailable')
        self.run_identity = dict(engine_digest=engine_digest(), source=source_identity(),
                                 token_schema=TOKEN_SCHEMA_VERSION)
        # A lineage may continue under newer trainer source only on request, and
        # only with the same engine and token schema; the change is recorded.
        self.source_changes = list(payload.get("source_changes", [])) if payload else []
        if payload is not None and payload.get("run_identity") != self.run_identity:
            saved = payload.get("run_identity") or {}
            if not (allow_source_change
                    and saved.get("engine_digest") == self.run_identity["engine_digest"]
                    and saved.get("token_schema") == self.run_identity["token_schema"]):
                raise ValueError("resume source/engine/token identity mismatch")
            self.source_changes.append(dict(
                at_update=int(payload["progress"]["updates"]),
                previous_source_sha256=saved.get("source", {}).get("source_sha256"),
                previous_revision=saved.get("source", {}).get("revision"),
                source_sha256=self.run_identity["source"]["source_sha256"],
                revision=self.run_identity["source"]["revision"]))
        # Dropout is zero, so train mode is the same policy as eval mode; staying
        # in train mode keeps the encoder on one kernel path for both the
        # behaviour log-probabilities and the learner's recomputation.
        self.actor = actor.to(self.device).train()
        self.actor.causal_sdpa = config.causal_sdpa
        self.actor.batched_private_attention = config.rollout_batched_attention
        self.actor.wide_private_projection = config.rollout_wide_projection
        self.critic = critic.to(self.device).train()
        critic_lr = config.critic_lr if config.critic_lr is not None else config.lr
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.rng = np.random.default_rng(config.seed)
        self.generator = torch.Generator(device=self.rollout_device)
        self.generator.manual_seed(config.seed)
        self.progress: dict[str, Any] = {"updates": 0, "decisions": 0, "rounds": 0,
                                         "matches": 0, "samples": 0, "learner_rows": 0,
                                         "elapsed_seconds": 0.0}
        self.population = HistoryPopulation(self.actor, self.lineage, config.seed + 17,
                                            config.population_recent,
                                            config.snapshot_probability,
                                            config.population_archive_every,
                                            config.population_archive_size,
                                            config.population_archive_share)
        if payload is not None:
            self.actor_optimizer.load_state_dict(payload["optimizer"]["actor"])
            self.critic_optimizer.load_state_dict(payload["optimizer"]["critic"])
            self.progress.update(payload["progress"])
            rng = payload["rng"]
            if rng.get("sampler_device", "cpu") != self.rollout_device.type:
                raise ValueError("resume requires the same sampler device type")
            if "population" in payload:
                self.population.load_state_dict(payload["population"])
                self.population.prune({})  # resumed environments discard old assignments
            elif config.snapshot_updates:
                raise ValueError("population-enabled resume requires saved population state")
            restore_rng(rng, self.rng)
            self.generator.set_state(rng["sampler"].cpu())
        self.env = gd.VecEnv(num_envs=config.num_envs, num_threads=config.num_threads,
                             seed=config.seed + 1000 * self.progress["updates"],
                             log_public_actions=True, log_env_limit=config.num_envs)
        self.store = MatchEventStore()
        self.buffer = SequenceRolloutBuffer()
        self.rollout_actor = (self.actor if self.rollout_device == self.device else
                              copy.deepcopy(self.actor).to(self.rollout_device).requires_grad_(False))
        self.rollout_snapshots: dict[int, Any] = {}
        self.collector = HistoryCollector(self.env, self.rollout_actor, self.store, self.buffer,
                                          self.generator, self.rollout_device,
                                          seat_policy=self.population.assignment,
                                          resolve_policy=self.resolve_rollout_policy,
                                          assignment_log=self.population_event,
                                          kv_cache=config.rollout_kv_cache,
                                          private_graphs=config.rollout_private_graphs,
                                          private_graph_budget_mb=config.rollout_graph_budget_mb,
                                          private_graph_policy_budget_mb=config.rollout_graph_policy_budget_mb,
                                          triton_cache=config.rollout_triton_cache,
                                          triton_min_batch=config.rollout_triton_min_batch,
                                          profile=config.profile_collection,
                                          temperature=config.rollout_temperature,
                                          epsilon=config.rollout_epsilon)
        self.prior_elapsed = float(self.progress["elapsed_seconds"])
        self.started = time.monotonic()
        self.metrics_path = self.output / "metrics.jsonl"
        self.stop_requested = False
        self.resume_count = int(payload.get("resume_count", 0)) + 1 if payload else 0
        self.population_event(dict(event="resume" if payload else "start",
                                   resume_count=self.resume_count,
                                   discarded_partial_trajectories=bool(payload),
                                   source_change=bool(payload) and bool(self.source_changes)
                                   and self.source_changes[-1]["at_update"] == self.progress["updates"],
                                   cache="ephemeral; rebuilt from raw public histories" if config.rollout_kv_cache
                                         else "none; raw public histories restart with environments"))
        self.write_manifest()

    def population_event(self, event: dict) -> None:
        with (self.output / "population.jsonl").open("a") as stream:
            stream.write(json.dumps({"lineage": self.lineage,
                                     "rollout_session": getattr(self, "resume_count", 0),
                                     **event}) + "\n")

    # -- manifest ---------------------------------------------------------------

    def write_manifest(self) -> None:
        manifest = {
            "stage": STAGE, "lineage": self.lineage, "init": "random", "teacher": None,
            "config": asdict(self.config), "model_config": asdict(self.actor.config),
            "engine_digest": self.run_identity["engine_digest"],
            "source": self.run_identity["source"],
            "source_changes": self.source_changes,
            "engine_binary_sha256": sha256(Path(gd._gd_core.__file__)),
            "token_schema": {"version": TOKEN_SCHEMA_VERSION, "dim": int(TOKEN_DIM),
                             "forced_bit": False, "private_tribute_flags": False},
            "reward": REWARD_SEMANTICS,
            "response_prediction": {
                "schema": RESPONSE_SCHEMA, "mode": self.config.response_mode,
                "coefficient": self.config.response_coef,
                "target_source": "executed actions and future public events in own-lineage self-play",
                "policy_input": "detached predicted probabilities only in explicit mode",
                "target_seat_or_future_events_in_actor_input": False,
            },
            "candidates": "full canonical set in engine order; every candidate selectable",
            "seats": "match-pinned current/own-lineage snapshot seats; learner rows only",
            "population": {"snapshot_updates": self.config.snapshot_updates,
                           "recent": self.config.population_recent,
                           "snapshot_probability": self.config.snapshot_probability,
                           "archive_every": self.config.population_archive_every,
                           "archive_size": self.config.population_archive_size,
                           "archive_share": self.config.population_archive_share,
                           "guaranteed_learner_seats": 1},
            "inference": {**runtime_settings(self.rollout_device),
                          "causal_sdpa": self.rollout_actor.causal_sdpa,
                          "rollout_kv_cache": self.collector.kv_cache,
                          "rollout_batched_attention": self.rollout_actor.batched_private_attention,
                          "rollout_wide_projection": self.rollout_actor.wide_private_projection,
                          "rollout_private_graphs": self.collector.private_graphs,
                          "rollout_graph_budget_mb": self.config.rollout_graph_budget_mb,
                          "rollout_graph_policy_budget_mb": self.config.rollout_graph_policy_budget_mb,
                          "rollout_triton_cache": self.collector.triton_cache,
                          "rollout_triton_min_batch": self.collector.triton_min_batch,
                          "reuse_cache_lengths": self.collector.reuse_cache_lengths,
                          "cache_boundary": "public-only; separate policy/env/match; learner invalidated before learn"},
            "resume": "restore optimizer/RNG/population; discard partial rounds and assignments; "
                      "restart environments and public histories; no persistent cache",
            "parameters": {"actor": count_parameters(self.actor),
                           "critic": count_parameters(self.critic)},
            "torch": torch.__version__, "device": str(self.device),
            "rollout_device": str(self.rollout_device),
            "weight_transfer": "before each collection; included in collection timing",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # -- one iteration ----------------------------------------------------------

    def resolve_rollout_policy(self, identity: int):
        if self.rollout_device == self.device:
            return self.population.resolve(identity)
        if identity == 0:
            return self.rollout_actor
        if identity not in self.rollout_snapshots:
            self.rollout_snapshots[identity] = copy.deepcopy(
                self.population.resolve(identity)).to(self.rollout_device)
        return self.rollout_snapshots[identity]

    def collect(self):
        if self.rollout_actor is not self.actor:
            # Keep the learner and sampling weights identical. load_state_dict
            # changes parameter versions, invalidating the public KV cache.
            self.rollout_actor.load_state_dict({k: v.detach().to(self.rollout_device)
                                               for k, v in self.actor.state_dict().items()})
            for identity in list(self.rollout_snapshots):
                if identity not in self.population.models:
                    del self.rollout_snapshots[identity]
        self.collector.policy_decisions.clear()
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
        response_stats = {}
        if cfg.response_mode == "none":
            log_prob, entropy = self.recompute_log_probs(rows)
        else:
            from train.history_model import segment_log_softmax
            inputs, chosen = buffer.decision_inputs(rows, self.store, self.device)
            state = self.actor.decision_states(None, inputs)
            logits, response = self.actor.candidate_outputs(state, inputs.cand, inputs.offsets,
                                                            predict=True)
            all_log_probs = segment_log_softmax(logits, inputs.rows, inputs.decisions)
            log_prob = all_log_probs[chosen]
            entropy = segment_entropy(all_log_probs, inputs.rows, inputs.decisions)
            targets = torch.as_tensor(opponent_response_labels(buffer, self.store, rows),
                                      device=self.device)
            prediction = response[chosen]   # outcomes exist ONLY for executed actions
            response_stats = {"response_loss": F.cross_entropy(prediction, targets),
                              "response_accuracy": (prediction.argmax(-1) == targets).float().mean(),
                              "response_event_fraction": (targets != 0).float().mean()}
        old = torch.as_tensor(buffer.compact()["logp"][rows], device=self.device)
        behaviour = torch.as_tensor(buffer.compact()["behaviour_logp"][rows], device=self.device)
        advantage = torch.as_tensor(buffer.advantage[rows], device=self.device)
        returns = torch.as_tensor(buffer.returns[rows], device=self.device)
        log_ratio = log_prob - old
        ratio = log_ratio.exp()
        if cfg.normalize_advantages and len(advantage) > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        # Exploration floor: rows were drawn from pi_b, so each advantage carries
        # the truncated per-action weight min(cap, pi_old / pi_b). It is applied
        # AFTER normalization, so the minibatch statistics stay those of the raw
        # GAE advantages and the weight only rescales rows. With temperature 1
        # and epsilon 0, logp == behaviour_logp bitwise, the weight is exactly
        # 1.0 and this multiplication leaves the loss unchanged bit for bit.
        raw_weight = (old - behaviour).exp()
        weight = raw_weight.clamp(max=cfg.behaviour_weight_cap)
        advantage = advantage * weight
        clipped = torch.minimum(ratio * advantage,
                                ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advantage)
        surrogate = -clipped.mean()
        policy_total = surrogate - cfg.entropy * entropy.mean()
        if response_stats:
            policy_total = policy_total + cfg.response_coef * response_stats["response_loss"]
        data = buffer.compact()
        obs = torch.as_tensor(data["obs"][rows], device=self.device)
        hidden = torch.as_tensor(data["hidden"][rows], device=self.device)
        value_loss = F.mse_loss(self.critic(obs, hidden), returns)
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = ((ratio - 1).abs() > cfg.clip).float().mean()
            ratio_deviation = (ratio - 1).abs().max()
            weight_capped = (raw_weight > cfg.behaviour_weight_cap).float().mean()
        return {"policy_total": policy_total, "value_total": cfg.value_coef * value_loss,
                "policy_loss": surrogate, "value_loss": value_loss, "entropy": entropy.mean(),
                "approx_kl": approx_kl, "clip_fraction": clip_fraction,
                "ratio_deviation": ratio_deviation, "behaviour_weight_mean": weight.mean(),
                "behaviour_weight_cap_fraction": weight_capped, **response_stats}

    def learn(self) -> dict[str, Any]:
        cfg = self.config
        self.collector.invalidate_learner_cache()
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
        prefixes = self.buffer.compact()["prefix"][rows]
        stats["learn_mean_prefix"] = float(prefixes.mean())
        stats["learn_max_prefix"] = int(prefixes.max())
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
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        t0 = time.perf_counter()
        collected = self.collect()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t1 = time.perf_counter()
        collection_cache = self.collector.cache_metrics()
        stats = self.learn()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t2 = time.perf_counter()
        cited = self.buffer.next_iteration()
        pruned = self.store.prune(cited)
        self.progress["updates"] += 1
        self.population.decisions.update(self.collector.policy_decisions)
        if (stats["minibatches"] and self.config.snapshot_updates
                and self.progress["updates"] % self.config.snapshot_updates == 0):
            identity = self.population.snapshot(self.progress["updates"])
            self.population_event(dict(event="snapshot", **self.population.metadata[identity]))
        self.population.prune(self.collector.assignments)
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
            "learn_decisions_per_sec": stats["update_samples"] / max(t2 - t1, 1e-9),
            "learn_exposures_per_sec": stats["update_samples"] * self.config.epochs / max(t2 - t1, 1e-9),
            "learner_collect_decisions_per_sec": collected.learner_rows / max(t1 - t0, 1e-9),
            "mean_prefix": collected.mean_prefix, "max_prefix": collected.prefix_max,
            "store_matches": len(self.store), "store_tokens": self.store.tokens,
            "pruned_matches": pruned, "carried_rows": len(self.buffer),
            "elapsed_seconds": self.progress["elapsed_seconds"],
            "population": self.population.metrics(),
            "policy_version_min": int(self.buffer.compact()["version"].min()) if len(self.buffer) else None,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0,
            "cache_bytes": self.collector.cache_metrics()["bytes"],
            "cache": self.collector.cache_metrics(),
            "collection_cache": collection_cache,
            "private_graphs": self.collector.graph_metrics(),
            "cache_transfers": self.collector.transfer_metrics(),
            "collection_phase_seconds": collected.phase_seconds,
            "collection_profile_synchronized": self.config.profile_collection,
            "collection_policy_batches": collected.policy_batches,
            "rollout_epsilon_pick_fraction": (collected.epsilon_picks / collected.learner_rows
                                              if collected.learner_rows else None),
            "rollout_behaviour_entropy": (collected.behaviour_entropy_sum / collected.learner_rows
                                          if collected.learner_rows else None),
            "cuda_allocated_bytes": torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0,
            "cuda_reserved_bytes": torch.cuda.memory_reserved(self.device) if self.device.type == "cuda" else 0,
            "cuda_inactive_split_peak_bytes": torch.cuda.memory_stats(self.device).get("inactive_split_bytes.all.peak", 0) if self.device.type == "cuda" else 0,
            "cuda_allocation_retries": torch.cuda.memory_stats(self.device).get("num_alloc_retries", 0) if self.device.type == "cuda" else 0,
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
        rng["sampler_device"] = self.rollout_device.type
        payload = checkpoint_payload(
            self.actor, self.critic, lineage=self.lineage, seed=self.config.seed,
            optimizer={"actor": self.actor_optimizer.state_dict(),
                       "critic": self.critic_optimizer.state_dict()},
            config=asdict(self.config), progress=dict(self.progress), rng=rng)
        payload.update(population=self.population.state_dict(), run_identity=self.run_identity,
                       resume_count=self.resume_count, source_changes=self.source_changes)
        return payload

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path is not None else self.output / "latest.pt"
        save_history_checkpoint(path, self.payload())
        return path

    def run(self) -> None:
        while self.progress["updates"] < self.config.updates and not self.stop_requested:
            line = self.update()
            print(json.dumps({k: line[k] for k in ("update", "decisions", "rounds",
                                                    "update_samples", "policy_loss",
                                                    "value_loss", "entropy",
                                                    "decisions_per_sec", "mean_prefix")}),
                  flush=True)
            if self.progress["updates"] % self.config.checkpoint_updates == 0:
                self.save()
                self.save(self.output / f"update-{self.progress['updates']:06d}.pt")
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
    parser.add_argument("--response-mode", choices=("none", "auxiliary", "explicit"), default="none")
    parser.add_argument("--response-coef", type=float, default=0.1)
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
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-epsilon", type=float, default=0.0)
    parser.add_argument("--behaviour-weight-cap", type=float, default=1.0)
    parser.add_argument("--checkpoint-updates", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--snapshot-updates", type=int, default=2)
    parser.add_argument("--population-recent", type=int, default=4)
    parser.add_argument("--snapshot-probability", type=float, default=0.5)
    parser.add_argument("--population-archive-every", type=int, default=0)
    parser.add_argument("--population-archive-size", type=int, default=16)
    parser.add_argument("--population-archive-share", type=float, default=0.5)
    parser.add_argument("--causal-sdpa", action="store_true")
    parser.add_argument("--rollout-kv-cache", action="store_true")
    parser.add_argument("--rollout-batched-attention", action="store_true")
    parser.add_argument("--rollout-wide-projection", action="store_true",
                        help="with --rollout-batched-attention: one q/out projection over all "
                             "rows (FP32; reduction order differs, not bitwise)")
    parser.add_argument("--rollout-private-graphs", action="store_true")
    parser.add_argument("--rollout-graph-budget-mb", type=int, default=512)
    parser.add_argument("--rollout-graph-policy-budget-mb", type=int, default=128,
                        help="per-policy private-graph cap within the total budget")
    parser.add_argument("--rollout-triton-cache", action="store_true")
    parser.add_argument("--rollout-triton-min-batch", type=int, default=1)
    parser.add_argument("--profile-collection", action="store_true")
    parser.add_argument("--rollout-device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--allow-source-change", action="store_true",
                        help="resume under different trainer source (same engine and token schema)")
    return parser


def config_from_args(args: argparse.Namespace) -> HistoryPPOConfig:
    names = {f.name for f in fields(HistoryPPOConfig)}
    return HistoryPPOConfig(**{k: v for k, v in vars(args).items() if k in names})


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    trainer = HistoryTrainer(config, args.output, device=args.device, resume=args.resume,
                             allow_source_change=args.allow_source_change)
    def stop(signum, frame):
        trainer.stop_requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    trainer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
