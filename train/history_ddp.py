"""Data-parallel collection for the history PPO trainer (engineering lever, opt-in).

Each rank is a complete ``HistoryTrainer`` with its own environments, event store,
rollout buffer and seat-assignment RNG. All ranks start from the same random
initialization (``fresh_player(config.seed)``) and share one lineage. Every PPO
minibatch step averages the actor and critic gradients over the ranks that had a
real minibatch at that step (``all_reduce`` over gloo), clips the averaged
gradient and applies the same Adam step, so parameters stay identical across
ranks; ``update`` verifies this with a parameter checksum. Snapshots are taken
at the same updates on every rank and are therefore identical too.

Nothing else changes: full canonical candidates, full public history, FP32,
reward/GAE semantics and match-pinned own-lineage snapshots. The effective batch
per update is ``world_size`` times larger, so this changes training dynamics and
needs a development-curve A/B before it is used as evidence. ``world_size == 1``
delegates to the base trainer unchanged; rank 0 uses the base seeds.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gd
import numpy as np
import torch
import torch.distributed as dist

from train.history_ppo import HistoryPPOConfig, HistoryTrainer, grad_norm
from train.history_rollout import HistoryCollector

RANK_SEED_STRIDE = 7919
RANK_ENV_STRIDE = 10_000_000


class HistoryDDPTrainer(HistoryTrainer):
    def __init__(self, config: HistoryPPOConfig, output: str | Path, device: str = "cpu",
                 rank: int = 0, world_size: int = 1, resume: str | Path | None = None) -> None:
        if world_size > 1 and not dist.is_initialized():
            raise RuntimeError("initialize torch.distributed before building a multi-rank trainer")
        super().__init__(config, output, device=device, resume=resume)
        self.rank, self.world_size = int(rank), int(world_size)
        if self.world_size == 1:
            return
        config = self.config
        shared = [self.lineage]
        dist.broadcast_object_list(shared, src=0)
        if resume is not None and shared[0] != self.lineage:
            raise RuntimeError("ranks resumed different lineages")
        self.lineage = shared[0]
        self.population.lineage = self.lineage
        if self.rank:
            # Rank 0 keeps the base (or exactly restored) streams; other ranks get
            # distinct deterministic streams, advanced by the update count on resume.
            updates = int(self.progress["updates"])
            offset = RANK_SEED_STRIDE * self.rank + 1000 * updates
            self.rng = np.random.default_rng(config.seed + offset)
            self.generator.manual_seed(config.seed + offset)
            self.population.rng = np.random.default_rng(config.seed + 17 + offset)
            self.env = gd.VecEnv(num_envs=config.num_envs, num_threads=config.num_threads,
                                 seed=config.seed + 1000 * updates + RANK_ENV_STRIDE * self.rank,
                                 log_public_actions=True, log_env_limit=config.num_envs)
            self.store.__init__()
            self.buffer.__init__()
            self.collector = HistoryCollector(self.env, self.actor, self.store, self.buffer,
                                              self.generator, self.device,
                                              seat_policy=self.population.assignment,
                                              resolve_policy=self.population.resolve,
                                              assignment_log=self.population_event,
                                              kv_cache=config.rollout_kv_cache,
                                              profile=config.profile_collection,
                                              temperature=config.rollout_temperature,
                                              epsilon=config.rollout_epsilon)
        self.write_manifest()
        self.population_event(dict(event="ddp", rank=self.rank, world_size=self.world_size,
                                   lineage=self.lineage, resumed=resume is not None))
        self.check_synchronized()

    # -- collectives --------------------------------------------------------------

    def parameters_all(self) -> list[torch.nn.Parameter]:
        return [*self.actor.parameters(), *self.critic.parameters()]

    def check_synchronized(self) -> None:
        local = torch.stack([p.detach().double().sum() for p in self.parameters_all()])
        high, low = local.clone(), local.clone()
        dist.all_reduce(high, op=dist.ReduceOp.MAX)
        dist.all_reduce(low, op=dist.ReduceOp.MIN)
        if not torch.equal(high, low):
            raise RuntimeError("data-parallel ranks diverged: parameter checksum mismatch")

    def reduce_gradients(self, real: bool) -> int:
        """Average the gradients of ranks that had a real minibatch; returns that count."""
        params = self.parameters_all()
        flat = torch.cat([(p.grad if (real and p.grad is not None) else torch.zeros_like(p)).reshape(-1)
                          for p in params] + [torch.ones(1, dtype=params[0].dtype, device=params[0].device)
                                              * float(real)])
        dist.all_reduce(flat)
        count = int(round(float(flat[-1])))
        if count == 0:
            return 0
        flat = flat[:-1] / count
        begin = 0
        for p in params:
            n = p.numel()
            p.grad = flat[begin:begin + n].view_as(p).clone()
            begin += n
        return count

    # -- learning -----------------------------------------------------------------

    def learn(self) -> dict[str, Any]:
        if self.world_size == 1:
            return super().learn()
        cfg = self.config
        self.collector.invalidate_learner_cache()
        values = self.refresh_values()
        samples = self.buffer.finalize(values, cfg.gamma, cfg.gae_lambda)
        stats: dict[str, Any] = {"update_samples": samples, "minibatches": 0}
        stats.update({key: None for key in self.STAT_KEYS})
        stats.update({"mean_reward": None, "mean_abs_reward": None, "mean_return": None,
                      "explained_variance": None})
        if samples:
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
            stats["mean_reward"] = float(rewards.mean()) if rewards.size else None
            stats["mean_abs_reward"] = float(np.abs(rewards).mean()) if rewards.size else None
        totals = {key: 0.0 for key in self.STAT_KEYS}
        count = steps = 0
        for _ in range(cfg.epochs):
            batches = list(self.minibatches()) if samples else []
            longest = torch.tensor([len(batches)], dtype=torch.long)
            dist.all_reduce(longest, op=dist.ReduceOp.MAX)
            for index in range(int(longest)):
                real = index < len(batches)
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                if real:
                    terms = self.minibatch_loss(batches[index])
                    terms["policy_total"].backward()
                    terms["value_total"].backward()
                if self.reduce_gradients(real) == 0:
                    continue
                encoder = grad_norm(self.actor.stream.parameters())
                actor = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip))
                critic = float(torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip))
                if not (math.isfinite(actor) and math.isfinite(critic)):
                    raise FloatingPointError("non-finite gradient; stopping before the step")
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                steps += 1
                if real:
                    for key in self.STAT_KEYS:
                        if key in terms:
                            totals[key] += float(terms[key].detach())
                    totals["actor_grad_norm"] += actor
                    totals["encoder_grad_norm"] += encoder
                    totals["critic_grad_norm"] += critic
                    count += 1
        # Every rank took the same optimizer steps; snapshot decisions use this count.
        stats["minibatches"] = steps
        stats["local_minibatches"] = count
        if count:
            for key in self.STAT_KEYS:
                stats[key] = totals[key] / count
        self.progress["samples"] += samples
        return stats

    def update(self) -> dict[str, Any]:
        line = super().update()
        if self.world_size > 1:
            self.check_synchronized()
            totals = torch.tensor([line["update_samples"], line["step_decisions"],
                                   line["step_rounds"]], dtype=torch.float64)
            dist.all_reduce(totals)
            line["global_update_samples"] = int(totals[0])
            line["global_step_decisions"] = int(totals[1])
            line["global_step_rounds"] = int(totals[2])
        return line
