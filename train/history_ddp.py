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

Actor ranks on one GPU. To keep a single-process run's batch while spreading
its host work over W processes, give each rank ``num_envs / W`` environments and
``minibatch_matches / W`` matches and set ``ddp_global_minibatch``: each step's
rank minibatches then act as one minibatch (advantages normalized over all of
its rows, gradient = row mean over all of them). What still differs from one
process is how the random streams split (per-rank environments, seat RNG and
samplers) and that each minibatch draws the same number of matches from every
rank's shard; the data distribution is the same. Collectives run on host copies
over gloo, so any number of ranks may share a GPU. ``python -m train.history_ddp
--world-size W ...`` launches the ranks; rank 0 writes checkpoints to ``--output``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
from typing import Any

import gd
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from train import history_ppo
from train.history_ppo import (HistoryPPOConfig, HistoryTrainer, config_from_args, grad_norm,
                               minibatch_scalars)

RANK_SEED_STRIDE = 7919
RANK_ENV_STRIDE = 10_000_000


class HistoryDDPTrainer(HistoryTrainer):
    def __init__(self, config: HistoryPPOConfig, output: str | Path, device: str = "cpu",
                 rank: int = 0, world_size: int = 1, resume: str | Path | None = None,
                 allow_source_change: bool = False,
                 resume_overrides: dict[str, Any] | None = None) -> None:
        if world_size > 1 and not dist.is_initialized():
            raise RuntimeError("initialize torch.distributed before building a multi-rank trainer")
        super().__init__(config, output, device=device, resume=resume,
                         allow_source_change=allow_source_change,
                         resume_overrides=resume_overrides)
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
            self.collector = self.make_collector()
        self.write_manifest()
        self.population_event(dict(event="ddp", rank=self.rank, world_size=self.world_size,
                                   lineage=self.lineage, resumed=resume is not None))
        self.check_synchronized()

    # -- collectives --------------------------------------------------------------

    def parameters_all(self) -> list[torch.nn.Parameter]:
        return [*self.actor.parameters(), *self.critic.parameters()]

    def check_synchronized(self) -> None:
        local = torch.stack([p.detach().double().sum() for p in self.parameters_all()]).cpu()
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
        # Reduced on a host copy: gloo on CUDA tensors is not relied upon, so
        # several ranks can share one GPU.
        host = flat.cpu()
        dist.all_reduce(host)
        count = int(round(float(host[-1])))
        if count == 0:
            return 0
        # A fresh tensor (the division allocates); each gradient is a view of
        # its slice, the same values as the former per-parameter clones without
        # one copy kernel per parameter. clip_grad_norm_ scales the views in
        # place, which touches disjoint slices only.
        flat = host[:-1].to(flat.device) / count
        begin = 0
        for p in params:
            n = p.numel()
            p.grad = flat[begin:begin + n].view_as(p)
            begin += n
        return count

    def global_minibatch(self, rows: np.ndarray | None) -> float:
        """With ``ddp_global_minibatch``, treat the union of this step's rank
        minibatches as one minibatch: set the advantage moments over all of its
        rows and return this rank's loss weight, so that the gradient average in
        ``reduce_gradients`` equals the gradient of the row mean over the union.
        Otherwise each rank normalizes locally and has weight 1. Every rank must
        call this at every step (``rows`` is None on a rank without a minibatch).
        """
        if not self.config.ddp_global_minibatch:
            return 1.0
        local = np.zeros(4, np.float64)
        if rows is not None:
            advantage = self.buffer.advantage[rows].astype(np.float64)
            local[:] = (len(advantage), advantage.sum(), np.square(advantage).sum(), 1.0)
        total = torch.from_numpy(local)
        dist.all_reduce(total)
        n, first, second, ranks = (float(x) for x in total)
        if rows is None or n == 0:
            return 1.0
        if self.config.normalize_advantages and n > 1:
            mean = first / n
            std = math.sqrt(max(second - n * mean * mean, 0.0) / (n - 1))
            self.advantage_moments = (mean, std)
        return len(rows) * ranks / n

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
                scale = self.global_minibatch(batches[index] if real else None)
                if real:
                    try:
                        terms = self.minibatch_loss(batches[index])
                    finally:
                        self.advantage_moments = None
                    (scale * terms["policy_total"]).backward()
                    (scale * terms["value_total"]).backward()
                if self.reduce_gradients(real) == 0:
                    continue
                # One device read per step: norms, loss statistics and the
                # finite check, before the optimizer steps (see HistoryTrainer.learn).
                encoder_norm = grad_norm(self.actor.stream.parameters(), self.device)
                actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
                critic_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
                actor, critic, encoder, scalars = minibatch_scalars(
                    actor_norm, critic_norm, encoder_norm, terms if real else {}, self.STAT_KEYS)
                if not (math.isfinite(actor) and math.isfinite(critic)):
                    raise FloatingPointError("non-finite gradient; stopping before the step")
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                steps += 1
                if real:
                    for key, value in scalars.items():
                        totals[key] += value
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
        return line

    def extend_metrics(self, line: dict[str, Any]) -> None:
        if self.world_size == 1:
            return
        # Lineage counters count every rank's data, so that a checkpoint (rank 0)
        # and a resume with another world size continue the same totals.
        local = torch.tensor([line["update_samples"], line["step_decisions"],
                              line["step_rounds"], line["step_learner_rows"],
                              line["step_matches"]], dtype=torch.float64)
        totals = local.clone()
        dist.all_reduce(totals)
        for key, index in (("samples", 0), ("decisions", 1), ("rounds", 2),
                           ("learner_rows", 3), ("matches", 4)):
            self.progress[key] += int(totals[index]) - int(local[index])
        line.update(decisions=self.progress["decisions"], rounds=self.progress["rounds"],
                    matches=self.progress["matches"])
        slowest = torch.tensor([line["collect_seconds"], line["learn_seconds"]],
                               dtype=torch.float64)
        dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
        line["global_update_samples"] = int(totals[0])
        line["global_step_decisions"] = int(totals[1])
        line["global_step_rounds"] = int(totals[2])
        line["global_decisions"] = self.progress["decisions"]
        line["global_collect_seconds"] = float(slowest[0])
        line["global_learn_seconds"] = float(slowest[1])
        line["global_decisions_per_sec"] = float(totals[1]) / max(float(slowest[0]), 1e-9)
        line["world_size"] = self.world_size

    def run(self) -> None:
        if self.world_size == 1:
            return super().run()
        while self.progress["updates"] < self.config.updates:
            # A stop seen by any rank stops every rank at the same update;
            # otherwise the others would wait in a collective forever.
            flag = torch.tensor([float(self.stop_requested)])
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if float(flag):
                break
            line = self.update()
            if self.rank == 0:
                print(json.dumps({k: line[k] for k in ("update", "global_decisions", "global_update_samples",
                                                        "policy_loss", "entropy",
                                                        "global_decisions_per_sec", "mean_prefix")}), flush=True)
                if self.progress["updates"] % self.config.checkpoint_updates == 0:
                    self.save()
                    self.save(self.output / f"update-{self.progress['updates']:06d}.pt")
        if self.rank == 0:
            self.save()
        dist.barrier()


# ---- CLI: W ranks as local processes, e.g. several actor ranks on one GPU --------

def rank_output(output: str | Path, rank: int) -> Path:
    """Rank 0 writes checkpoints and metrics to ``output`` itself, so tools that
    read a single-process run's directory work unchanged; rank r > 0 logs to
    ``output/rank-r``."""
    return Path(output) if rank == 0 else Path(output) / f"rank-{rank}"


def cuda_index(device: str) -> int | None:
    """The CUDA device index to select in a rank, or None (CPU, or bare "cuda"
    on the default device; ``torch.cuda.set_device`` rejects an index-less device)."""
    parsed = torch.device(device)
    return parsed.index if parsed.type == "cuda" else None


def _rank_main(rank: int, world: int, port: int, argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        index = cuda_index(args.device)
        if index is not None:
            torch.cuda.set_device(index)
        trainer = HistoryDDPTrainer(config_from_args(args), rank_output(args.output, rank),
                                    device=args.device, rank=rank, world_size=world,
                                    resume=args.resume,
                                    allow_source_change=args.allow_source_change,
                                    resume_overrides=history_ppo.parse_resume_overrides(
                                        args.resume_set))

        def stop(signum, frame):
            trainer.stop_requested = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        trainer.run()
    finally:
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = history_ppo.build_parser()
    parser.description = ("History PPO over --world-size local ranks. --num-envs, "
                          "--num-threads and --minibatch-matches are per rank.")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--ddp-global-minibatch", action="store_true",
                        help="normalize advantages and weight the loss over the union of "
                             "the ranks' minibatches, as one minibatch would be")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.world_size < 1:
        raise SystemExit("--world-size must be positive")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    context = mp.get_context("spawn")
    ranks = [context.Process(target=_rank_main, args=(r, args.world_size, port, argv),
                             name=f"history-rank-{r}") for r in range(args.world_size)]
    for process in ranks:
        process.start()

    def forward(signum, frame):
        for process in ranks:
            if process.is_alive():
                os.kill(process.pid, signal.SIGTERM)
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    # If any rank fails, the others would block in a collective: stop them all.
    failed = False
    while any(p.is_alive() for p in ranks):
        for process in ranks:
            process.join(timeout=1.0)
            if process.exitcode not in (None, 0) and not failed:
                failed = True
                print(f"{process.name} exited with {process.exitcode}; terminating all ranks",
                      file=sys.stderr, flush=True)
                for other in ranks:
                    if other.is_alive():
                        other.kill()
    return 0 if not failed and all(p.exitcode == 0 for p in ranks) else 1


if __name__ == "__main__":
    sys.exit(main())
