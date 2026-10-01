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

import copy
from dataclasses import asdict
import json
import math
import os
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
from train.history_config import (REWARD_SEMANTICS, RESUME_OVERRIDES, HistoryPPOConfig,
                                  build_parser, config_from_args, optional_bool,
                                  parse_arm_schedule, parse_resume_overrides)
from train.history_model import (STAGE, TOKEN_SCHEMA_VERSION,
                                 HistoryPolicyConfig, checkpoint_payload, count_parameters,
                                 fresh_player, load_history_checkpoint, save_history_checkpoint)
from train.history_rollout import (HistoryCollector, MatchEventStore, SequenceRolloutBuffer)
from train.history_population import HistoryPopulation
from train.history_response import RESPONSE_SCHEMA, opponent_response_labels
from train.history_transfers import runtime_settings
from train.logs import TOKEN_DIM

def segment_entropy(log_probs: torch.Tensor, rows: torch.Tensor, count: int) -> torch.Tensor:
    """Entropy of each decision's full candidate distribution, ``[count]``."""
    terms = -(log_probs.exp() * log_probs)
    return torch.zeros(count, device=log_probs.device, dtype=log_probs.dtype).index_add_(
        0, rows, terms)


class HistoryTrainer:
    """Collect, learn, checkpoint. ``update()`` is one PPO iteration."""

    STAT_KEYS = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
                 "ratio_deviation", "actor_grad_norm", "encoder_grad_norm", "critic_grad_norm",
                 "response_loss", "response_accuracy", "response_event_fraction",
                 "behaviour_weight_mean", "behaviour_weight_cap_fraction")
    # (mean, std) of the whole data-parallel minibatch's advantages, else None
    advantage_moments: tuple[float, float] | None = None

    def __init__(self, config: HistoryPPOConfig, output: str | Path, device: str = "cpu",
                 resume: str | Path | None = None, allow_source_change: bool = False,
                 resume_overrides: dict[str, Any] | None = None) -> None:
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
            saved_config = HistoryPPOConfig.from_payload(payload["config"], updates=config.updates)
            overrides = dict(resume_overrides or {})
            unknown = set(overrides) - RESUME_OVERRIDES
            if unknown:
                raise ValueError(f"resume cannot change {sorted(unknown)}")
            config = HistoryPPOConfig.from_payload(payload["config"], updates=config.updates,
                                                   **overrides)
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
        # Resume may change how the batch is laid out over processes (and, on
        # request, the snapshot cadence); every change is recorded in the lineage.
        self.config_changes = list(payload.get("config_changes", [])) if payload else []
        if payload is not None:
            changed = {k: [getattr(saved_config, k), getattr(config, k)]
                       for k in sorted(RESUME_OVERRIDES)
                       if getattr(saved_config, k) != getattr(config, k)}
            if changed:
                self.config_changes.append(dict(at_update=int(payload["progress"]["updates"]),
                                                changes=changed))
        # Dropout is zero, so train mode is the same policy as eval mode; staying
        # in train mode keeps the encoder on one kernel path for both the
        # behaviour log-probabilities and the learner's recomputation.
        self.actor = actor.to(self.device).train()
        self.actor.causal_sdpa = config.causal_sdpa
        self.actor.batched_private_attention = config.rollout_batched_attention
        self.actor.wide_private_projection = config.rollout_wide_projection
        self.actor.batched_match_attention = config.learner_batched_attention
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
        self.session_first_update = int(self.progress["updates"])
        self.env = gd.VecEnv(num_envs=config.num_envs, num_threads=config.num_threads,
                             seed=config.seed + 1000 * self.progress["updates"],
                             log_public_actions=True, log_env_limit=config.num_envs)
        self.store = MatchEventStore()
        self.buffer = SequenceRolloutBuffer()
        self.rollout_actor = (self.actor if self.rollout_device == self.device else
                              copy.deepcopy(self.actor).to(self.rollout_device).requires_grad_(False))
        self.rollout_snapshots: dict[int, Any] = {}
        self.collector = self.make_collector()
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

    def make_collector(self) -> HistoryCollector:
        """The rollout collector over this trainer's env, store and buffer."""
        config = self.config
        return HistoryCollector(self.env, self.rollout_actor, self.store, self.buffer,
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
                                profile=self.collection_profiled(),
                                temperature=config.rollout_temperature,
                                epsilon=config.rollout_epsilon,
                                batch_snapshot_policies=self.snapshot_batching(),
                                paged_cache=config.rollout_paged_cache,
                                batch_snapshot_encoder=config.batch_snapshot_encoder,
                                page_span=config.rollout_page_span)

    def snapshot_batching(self) -> bool:
        """Merged snapshot inference for this update (the schedule's block, if any)."""
        blocks = parse_arm_schedule(self.config.batch_snapshot_policies_schedule)
        if not blocks:
            return self.config.batch_snapshot_policies
        done = int(self.progress["updates"]) - self.session_first_update
        for count, arm in blocks:
            if done < count:
                return arm
            done -= count
        return blocks[-1][1]

    def cuda_cache_trimmed(self) -> bool:
        """Trim the allocator cache in this update (explicit, or the merged arm)."""
        if self.config.rollout_trim_cuda_cache is None:
            return self.snapshot_batching()
        return bool(self.config.rollout_trim_cuda_cache)

    def trim_cuda_cache(self, point: str, trims: dict[str, dict]) -> None:
        """``torch.cuda.empty_cache()`` with reserved/allocated bytes before and
        after and the seconds it took, recorded under ``trims[point]``. Only free
        cached blocks are returned to the driver; live tensors, KV caches and
        captured graphs are untouched, so nothing computed changes. A no-op
        (zero bytes) without CUDA."""
        devices = [d for d in (self.rollout_device, self.device) if d.type == "cuda"]
        begin = time.perf_counter()
        record = dict(reserved_before=0, allocated_before=0, reserved_after=0,
                      allocated_after=0)
        if devices:
            device = devices[0]
            torch.cuda.synchronize(device)
            record.update(reserved_before=torch.cuda.memory_reserved(device),
                          allocated_before=torch.cuda.memory_allocated(device))
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            record.update(reserved_after=torch.cuda.memory_reserved(device),
                          allocated_after=torch.cuda.memory_allocated(device))
        record["seconds"] = time.perf_counter() - begin
        trims[point] = record

    def collection_profiled(self) -> bool:
        """Profile this update's collection: on, after this process's warmup updates."""
        return (self.config.profile_collection and int(self.progress["updates"])
                - self.session_first_update >= self.config.profile_collection_warmup)

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
            "config_changes": self.config_changes,
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
                          "learner_batched_attention": self.actor.batched_match_attention,
                          "learner_length_groups": self.config.learner_length_groups,
                          "rollout_graph_budget_mb": self.config.rollout_graph_budget_mb,
                          "rollout_graph_policy_budget_mb": self.config.rollout_graph_policy_budget_mb,
                          "rollout_triton_cache": self.collector.triton_cache,
                          "rollout_triton_min_batch": self.collector.triton_min_batch,
                          "rollout_paged_cache": self.collector.paged_cache,
                          "batch_snapshot_encoder": self.collector.batch_snapshot_encoder,
                          "rollout_page_span": self.collector.page_span,
                          "batch_snapshot_policies": self.config.batch_snapshot_policies,
                          "batch_snapshot_policies_schedule":
                              self.config.batch_snapshot_policies_schedule,
                          "rollout_trim_cuda_cache": self.config.rollout_trim_cuda_cache,
                          "cuda_cache_trim": "torch.cuda.empty_cache() after collect and after "
                                             "learn (None: with the merged-snapshot arm); "
                                             "allocator timing only",
                          "snapshot_batching": "merged head over stacked snapshot weights; "
                                               "per-identity public KV cache; snapshot seats "
                                               "bypass private graphs",
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
        predict = cfg.response_mode != "none"
        extra = ({"response_target": opponent_response_labels(self.buffer, self.store, rows)}
                 if predict else None)
        # Actor inputs, stored row data and response targets: one packed upload.
        batch = self.buffer.training_batch(rows, self.store, self.device, extra=extra,
                                           length_groups=cfg.learner_length_groups,
                                           width=cfg.width)
        inputs, chosen, row = batch.inputs, batch.chosen, batch.fields
        response_stats = {}
        if not predict:
            all_log_probs = self.actor.candidate_log_probs(inputs)
        else:
            from train.history_model import segment_log_softmax
            state = self.actor.decision_states(None, inputs)
            logits, response = self.actor.candidate_outputs(state, inputs.cand, inputs.offsets,
                                                            predict=True)
            all_log_probs = segment_log_softmax(logits, inputs.rows, inputs.decisions)
            targets = row["response_target"]
            prediction = response[chosen]   # outcomes exist ONLY for executed actions
            response_stats = {"response_loss": F.cross_entropy(prediction, targets),
                              "response_accuracy": (prediction.argmax(-1) == targets).float().mean(),
                              "response_event_fraction": (targets != 0).float().mean()}
        log_prob = all_log_probs[chosen]
        entropy = segment_entropy(all_log_probs, inputs.rows, inputs.decisions)
        old, behaviour = row["logp"], row["behaviour_logp"]
        advantage, returns = row["advantage"], row["returns"]
        log_ratio = log_prob - old
        ratio = log_ratio.exp()
        if self.advantage_moments is not None:   # data parallel: the cross-rank minibatch
            advantage = (advantage - self.advantage_moments[0]) / (self.advantage_moments[1] + 1e-8)
        elif cfg.normalize_advantages and len(advantage) > 1:
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
        value_loss = F.mse_loss(self.critic(inputs.obs, row["hidden"]), returns)
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

    def learn_summary(self) -> tuple[dict[str, Any], int]:
        """Values, GAE and the update's data statistics; returns ``(stats, samples)``.

        ``stats`` holds every learning key (``None`` until ``learn`` fills it).
        """
        cfg = self.config
        self.collector.invalidate_learner_cache()
        values = self.refresh_values()
        samples = self.buffer.finalize(values, cfg.gamma, cfg.gae_lambda)
        stats: dict[str, Any] = {"update_samples": samples, "minibatches": 0}
        stats.update({key: None for key in self.STAT_KEYS})
        stats.update({"mean_reward": None, "mean_abs_reward": None, "mean_return": None,
                      "explained_variance": None})
        if samples == 0:
            return stats, samples
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
        return stats, samples

    # Hooks that data-parallel training overrides (train/history_ddp.py).

    def epoch_batches(self, samples: int) -> Iterator[np.ndarray | None]:
        """One epoch's minibatch rows; ``None`` is a step without local data."""
        return self.minibatches()

    def backward_minibatch(self, rows: np.ndarray | None) -> dict[str, torch.Tensor] | None:
        """Loss terms of ``rows`` with their gradients accumulated."""
        terms = self.minibatch_loss(rows)
        terms["policy_total"].backward()
        terms["value_total"].backward()
        return terms

    def reduce_gradients(self, real: bool) -> int:
        """Number of minibatches in the gradient about to be applied (0 skips the step)."""
        return 1

    def collective(self) -> bool:
        """Whether learning runs collectives, so every step happens on every rank."""
        return False

    def learn(self) -> dict[str, Any]:
        cfg = self.config
        stats, samples = self.learn_summary()
        if samples == 0 and not self.collective():
            return stats
        # Loss terms and gradient norms stay on the device until one transfer per
        # minibatch, which also carries the finiteness check before the step.
        # Each value is the same FP32 number the per-term float() reads gave, and
        # the totals add them in the same order in double precision.
        keys = [key for key in self.STAT_KEYS
                if key not in ("actor_grad_norm", "encoder_grad_norm", "critic_grad_norm")]
        totals = {key: 0.0 for key in self.STAT_KEYS}
        count = steps = 0
        encoder_parameters = list(self.actor.stream.parameters())
        for _ in range(cfg.epochs):
            for rows in self.epoch_batches(samples):
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                terms = self.backward_minibatch(rows)
                if self.reduce_gradients(terms is not None) == 0:
                    continue
                # Per-tensor encoder norms in one foreach call rather than a few
                # kernels per tensor; squared and summed as before.
                grads = [p.grad.detach().float() for p in encoder_parameters
                         if p.grad is not None]
                encoder = (torch.stack(torch._foreach_norm(grads)) ** 2).unbind() if grads else ()
                actor = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
                critic = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
                present = [key for key in keys if terms is not None and key in terms]
                # clip_grad_norm_ returns a CPU zero when a module has no gradient.
                host = torch.stack([value.detach().to(self.device, torch.float32) for value in
                                    (actor, critic, *encoder,
                                     *(terms[key] for key in present))]).tolist()
                actor, critic = host[0], host[1]
                if not (math.isfinite(actor) and math.isfinite(critic)):
                    raise FloatingPointError("non-finite gradient; stopping before the step")
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                steps += 1
                if terms is None:
                    continue
                encoder_total = 0.0
                for value in host[2:2 + len(encoder)]:
                    encoder_total += value
                for key, value in zip(present, host[2 + len(encoder):]):
                    totals[key] += value
                totals["actor_grad_norm"] += actor
                totals["encoder_grad_norm"] += math.sqrt(encoder_total)
                totals["critic_grad_norm"] += critic
                count += 1
        # Every rank took the same optimizer steps; snapshot decisions use this count.
        stats["minibatches"] = steps
        if self.collective():
            stats["local_minibatches"] = count
        if count:
            for key in self.STAT_KEYS:
                stats[key] = totals[key] / count
        self.progress["samples"] += samples
        return stats

    def update(self) -> dict[str, Any]:
        """Collect, learn, log one metrics line and advance the counters."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        # Profiling only adds synchronized timings and counters; it never changes
        # what is collected, so switching it between updates is safe.
        self.collector.profile = self.collection_profiled()
        # Merged snapshot inference changes only how frozen snapshot seats are
        # evaluated (tier 2), so it may switch between updates as well.
        self.collector.batch_snapshot_policies = self.snapshot_batching()
        # Allocator cache trim (allocator timing only). Points: after collect,
        # so learn (the phase with the highest device readings) starts from the
        # live set rather than on top of collection's freed variable-shape
        # blocks; after learn, so collection does not sit on the learner's
        # activation cache. Switching the merged arm on releases the snapshot
        # private graphs; their pools only return to the driver on a trim.
        trim = self.cuda_cache_trimmed()
        trims: dict[str, dict] = {}
        if trim and self.collector.batch_snapshot_policies:
            if self.collector.release_snapshot_graphs():
                self.trim_cuda_cache("graphs_released", trims)
        t0 = time.perf_counter()
        collected = self.collect()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t1 = time.perf_counter()
        collection_cache = self.collector.cache_metrics()
        if trim:
            self.trim_cuda_cache("after_collect", trims)
        t1_learn = time.perf_counter()
        stats = self.learn()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t2 = time.perf_counter()
        if trim:
            self.trim_cuda_cache("after_learn", trims)
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
            "step_rounds": collected.rounds, "step_matches": collected.matches,
            "round_gain": collected.gain / collected.rounds if collected.rounds else None,
            "decisions_per_sec": collected.decisions / max(t1 - t0, 1e-9),
            "collect_seconds": t1 - t0, "learn_seconds": t2 - t1_learn,
            "learn_decisions_per_sec": stats["update_samples"] / max(t2 - t1_learn, 1e-9),
            "learn_exposures_per_sec": stats["update_samples"] * self.config.epochs / max(t2 - t1_learn, 1e-9),
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
            "collection_profile_synchronized": self.collector.profile,
            "collection_group_phase_seconds": collected.group_phase_seconds,
            "collection_policy_call_rows": {group: {str(size): count for size, count
                                                    in sorted(sizes.items())}
                                            for group, sizes in collected.policy_call_rows.items()},
            "collection_policy_batches": collected.policy_batches,
            "collection_steps": collected.steps,
            "rollout_batch_snapshot_policies": self.collector.batch_snapshot_policies,
            "rollout_trim_cuda_cache": trim,
            "cuda_trim": trims,
            "cuda_trim_seconds": sum(record["seconds"] for record in trims.values()),
            "snapshot_heads": self.collector.snapshot_head_metrics(),
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
        self.extend_metrics(line)
        with self.metrics_path.open("a") as stream:
            stream.write(json.dumps(line) + "\n")
        return line

    def extend_metrics(self, line: dict[str, Any]) -> None:
        """Hook for subclasses to add fields to the metrics line before it is written."""

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
                       resume_count=self.resume_count, source_changes=self.source_changes,
                       config_changes=self.config_changes)
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


# ---- diagnostics ----------------------------------------------------------------

CPROFILE_ENV = "GUANZERO_CPROFILE_DIR"
CPROFILE_BUILTINS_ENV = "GUANZERO_CPROFILE_BUILTINS"


def run_profiled(trainer: "HistoryTrainer", rank: int = 0) -> None:
    """``trainer.run()``; with ``GUANZERO_CPROFILE_DIR`` set, under cProfile.

    Diagnostic only and off by default: the profile of this process's main
    thread is written to ``<dir>/rank-<rank>.prof`` (``pstats`` format) when
    ``run`` returns, including after a SIGTERM/SIGINT stop request, or raises.
    It changes nothing that is trained or sampled, but the profiler's per-call
    overhead makes the run's timings unusable for throughput comparisons.

    C functions are not entries by default: their time counts in the calling
    Python function's own time. Under Python 3.12 recording them
    (``GUANZERO_CPROFILE_BUILTINS=1``) drops every enclosing frame of some
    torch C calls (``torch.save``'s zip writer, calls inside collection) from
    the call tree, so ``run``/``update``/``collect`` would be missing.
    """
    directory = os.environ.get(CPROFILE_ENV)
    if not directory:
        trainer.run()
        return
    import cProfile
    path = Path(directory) / f"rank-{rank}.prof"
    path.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile(builtins=os.environ.get(CPROFILE_BUILTINS_ENV) == "1")
    profiler.enable()
    try:
        trainer.run()
    finally:
        profiler.disable()
        profiler.dump_stats(str(path))


# ---- CLI ------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    trainer = HistoryTrainer(config, args.output, device=args.device, resume=args.resume,
                             allow_source_change=args.allow_source_change,
                             resume_overrides=parse_resume_overrides(args.resume_set))
    def stop(signum, frame):
        trainer.stop_requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    run_profiled(trainer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
