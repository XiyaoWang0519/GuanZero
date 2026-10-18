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

``ddp_reduce="shm"`` (opt-in) moves the gradient all-reduce from gloo's TCP
ring to a same-node shared-memory mapping that adds in gloo's ring order
(``train/history_shm_reduce.py``), so the sums are bitwise gloo's. A start-up
self-check against ``dist.all_reduce`` (and a ``/dev/shm`` that can hold the
mapping) gates it; otherwise the run stays on gloo and the manifest and metrics
record why. The small collectives stay on gloo.
"""
from __future__ import annotations

import argparse
from collections import deque
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
from train.history_ppo import HistoryPPOConfig, HistoryTrainer, config_from_args
from train.history_shm_reduce import ShmAllReduce

RANK_SEED_STRIDE = 7919
RANK_ENV_STRIDE = 10_000_000


class HistoryDDPTrainer(HistoryTrainer):
    def __init__(self, config: HistoryPPOConfig, output: str | Path, device: str = "cpu",
                 rank: int = 0, world_size: int = 1, resume: str | Path | None = None,
                 allow_source_change: bool = False,
                 resume_overrides: dict[str, Any] | None = None) -> None:
        if world_size > 1 and not dist.is_initialized():
            raise RuntimeError("initialize torch.distributed before building a multi-rank trainer")
        self.shm: ShmAllReduce | None = None
        self.shm_fallback: str | None = None    # why ddp_reduce=shm runs on gloo
        # Per learn: every rank's minibatch count (gathered once) and, with
        # ddp_global_minibatch, the epoch's in-flight advantage-moment all-reduces.
        self.minibatch_counts: list[int] | None = None
        self.pending_moments: deque = deque()
        self.rank, self.world_size = int(rank), int(world_size)
        super().__init__(config, output, device=device, resume=resume,
                         allow_source_change=allow_source_change,
                         resume_overrides=resume_overrides)
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
        self.check_synchronized()
        if self.config.ddp_reduce == "shm":
            self.shm_reducer()       # map and self-check at start-up
        self.write_manifest()
        self.population_event(dict(event="ddp", rank=self.rank, world_size=self.world_size,
                                   lineage=self.lineage, resumed=resume is not None,
                                   gradient_reduce=self.gradient_reduce()))

    def close(self) -> None:
        """Release the shared-memory mapping (if any); idempotent."""
        if self.shm is not None:
            self.shm.close()
            self.shm = None

    # -- collectives --------------------------------------------------------------

    def parameters_all(self) -> list[torch.nn.Parameter]:
        return [*self.actor.parameters(), *self.critic.parameters()]

    def check_synchronized(self) -> None:
        """Every rank's per-tensor parameter checksums are equal: their maximum
        and minimum over ranks agree, from one MAX over ``[x, -x]`` (negation is
        exact, so ``-max(-x) == min(x)``)."""
        local = torch.stack([p.detach().double().sum() for p in self.parameters_all()]).cpu()
        both = torch.cat([local, -local])
        dist.all_reduce(both, op=dist.ReduceOp.MAX)
        if not torch.equal(both[:len(local)], -both[len(local):]):
            raise RuntimeError("data-parallel ranks diverged: parameter checksum mismatch")

    def shm_reducer(self) -> ShmAllReduce | None:
        """The shared-memory gradient reducer, or None when it is unavailable.

        Created on first use (collective: every rank reaches it at the same step,
        also after a live switch) and kept only if its sums equal
        ``dist.all_reduce``'s bitwise on seeded probes of the real size. Any
        failure (no room in /dev/shm, a rank cannot map it, a mismatch) is the
        same on every rank: the run continues on gloo and records why.
        """
        numel = sum(p.numel() for p in self.parameters_all()) + 1
        if self.shm is None and self.shm_fallback is None:
            try:
                shm = ShmAllReduce(self.rank, self.world_size, numel)
            except (RuntimeError, ValueError) as error:
                self.shm_fallback = f"unavailable: {error}"
            else:
                mismatches = shm.self_check()
                if mismatches:
                    shm.close()
                    self.shm_fallback = (f"self-check: {mismatches} probe elements differ "
                                         "from dist.all_reduce")
                else:
                    self.shm = shm
            if self.shm_fallback is not None:
                if self.rank == 0:
                    print(f"ddp_reduce=shm {self.shm_fallback}; reducing gradients over gloo",
                          file=sys.stderr, flush=True)
                self.write_manifest()
        if self.shm is not None and self.shm.numel != numel:
            raise RuntimeError("the parameter count changed under the shared-memory reducer")
        return self.shm

    def gradient_reduce(self) -> str:
        """The transport this rank's gradient all-reduce uses now."""
        if self.world_size > 1 and self.config.ddp_reduce == "shm" and self.shm is not None:
            return "shm"
        return "gloo"

    def reduce_record(self) -> dict[str, Any]:
        return dict(super().reduce_record(), ddp_reduce_active=self.gradient_reduce(),
                    ddp_reduce_fallback=self.shm_fallback)

    def reduce_gradients(self, real: bool) -> int:
        """Average the gradients of ranks that had a real minibatch; returns that count.

        One flat vector per rank: every gradient (zeros on a rank without a real
        minibatch), then the real flag, so the sum's last element is the count.
        """
        if self.world_size == 1:
            return super().reduce_gradients(real)
        params = self.parameters_all()
        device = params[0].device
        shm = self.shm_reducer() if self.config.ddp_reduce == "shm" else None
        if shm is not None:
            total = shm.all_reduce(lambda slot: self.fill_gradient_slot(slot, params, real))
        else:
            if real:
                flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                                  for p in params]
                                 + [torch.ones(1, dtype=params[0].dtype, device=device)])
            else:
                flat = torch.zeros(sum(p.numel() for p in params) + 1, dtype=params[0].dtype,
                                   device=device)
            # Reduced on a host copy: gloo on CUDA tensors is not relied upon, so
            # several ranks can share one GPU.
            total = flat.cpu()
            dist.all_reduce(total)
        count = int(round(float(total[-1])))
        if count == 0:
            return 0
        # The division allocates a fresh tensor (never a view of the host or
        # shared buffer), so each gradient can be a view of its slice instead of
        # a per-parameter copy; clipping scales the views in place over disjoint
        # slices.
        flat = total[:-1].to(device) / count
        begin = 0
        for p in params:
            n = p.numel()
            p.grad = flat[begin:begin + n].view_as(p)
            begin += n
        return count

    @staticmethod
    def fill_gradient_slot(slot: torch.Tensor, params: list[torch.nn.Parameter],
                           real: bool) -> None:
        """This rank's flat gradient and real flag, written straight into its slot
        of the shared mapping (one device-to-host copy, no host staging tensor)."""
        if not real:
            slot.zero_()
            return
        parts = [(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                 for p in params]
        n = slot.numel() - 1
        if parts[0].device.type == "cpu":
            torch.cat(parts, out=slot[:n])
        else:
            slot[:n].copy_(torch.cat(parts))
        slot[n] = 1.0

    def local_moments(self, rows: np.ndarray | None) -> np.ndarray:
        """``[rows, sum, sum of squares, 1]`` of this rank's minibatch advantages
        in float64 (zeros on a step without a minibatch)."""
        local = np.zeros(4, np.float64)
        if rows is not None:
            advantage = self.buffer.advantage[rows].astype(np.float64)
            local[:] = (len(advantage), advantage.sum(), np.square(advantage).sum(), 1.0)
        return local

    def global_minibatch(self, rows: np.ndarray | None) -> float:
        """With ``ddp_global_minibatch``, treat the union of this step's rank
        minibatches as one minibatch: set the advantage moments over all of its
        rows and return this rank's loss weight, so that the gradient average in
        ``reduce_gradients`` equals the gradient of the row mean over the union.
        Otherwise each rank normalizes locally and has weight 1. Inside ``learn``
        the step's moment all-reduce was issued by ``epoch_batches`` and is only
        awaited here; otherwise every rank must call this at the same step and
        it all-reduces synchronously (``rows`` is None on a rank without a
        minibatch).
        """
        if not self.config.ddp_global_minibatch:
            return 1.0
        if self.pending_moments:
            size, total, work = self.pending_moments.popleft()
            if size != (None if rows is None else len(rows)):
                raise RuntimeError("advantage moments were issued for different minibatch rows")
            work.wait()
        else:
            total = torch.from_numpy(self.local_moments(rows))
            dist.all_reduce(total)
        n, first, second, ranks = (float(x) for x in total)
        if rows is None or n == 0:
            return 1.0
        if self.config.normalize_advantages and n > 1:
            mean = first / n
            std = math.sqrt(max(second - n * mean * mean, 0.0) / (n - 1))
            self.advantage_moments = (mean, std)
        return len(rows) * ranks / n

    # -- learning: the base PPO loop with collective hooks -------------------------

    def collective(self) -> bool:
        return self.world_size > 1

    def learn(self) -> dict[str, Any]:
        if self.world_size > 1:
            self.minibatch_counts = None
            self.pending_moments.clear()
        return super().learn()

    def epoch_batches(self, samples: int):
        """This rank's minibatches, padded with ``None`` to the longest rank's count
        so every rank takes the same optimizer steps.

        Every epoch of one learn has the same number of minibatches per rank
        (whole match groups, fixed during learn), so the counts are gathered
        once per learn. With ``ddp_global_minibatch`` the epoch's per-step
        float64[4] moment all-reduces are issued here, asynchronously and in
        step order: the same collectives in the same order as one blocking call
        per step, so the sums are bitwise the same; ``global_minibatch`` waits.
        """
        if self.world_size == 1:
            return super().epoch_batches(samples)
        batches = list(self.minibatches()) if samples else []
        if self.minibatch_counts is None:
            counts = [torch.zeros(1, dtype=torch.long) for _ in range(self.world_size)]
            dist.all_gather(counts, torch.tensor([len(batches)], dtype=torch.long))
            self.minibatch_counts = [int(count) for count in counts]
        elif len(batches) != self.minibatch_counts[self.rank]:
            raise RuntimeError("a rank's minibatch count changed within one learn")
        longest = max(self.minibatch_counts)
        padded = [batches[i] if i < len(batches) else None for i in range(longest)]
        if self.config.ddp_global_minibatch:
            if self.pending_moments:
                raise RuntimeError("advantage moments of the previous epoch were not consumed")
            for rows in padded:
                total = torch.from_numpy(self.local_moments(rows))
                self.pending_moments.append((None if rows is None else len(rows), total,
                                             dist.all_reduce(total, async_op=True)))
        return padded

    def backward_minibatch(self, rows: np.ndarray | None, batch=None
                           ) -> dict[str, torch.Tensor] | None:
        if self.world_size == 1:
            return super().backward_minibatch(rows, batch)
        scale = self.global_minibatch(rows)
        self.learn_mark("advantage_reduce")
        if rows is None:
            return None
        try:
            terms = self.minibatch_loss(rows, batch)
        finally:
            self.advantage_moments = None
        # One backward call over both disjoint graphs (bitwise, see the base).
        (scale * terms["policy_total"] + scale * terms["value_total"]).backward()
        self.learn_mark("backward")
        return terms

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
        line["ddp_reduce"] = self.gradient_reduce()

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
                    self.save(copies=[self.output / f"update-{self.progress['updates']:06d}.pt"])
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


def rank_cuda_index(device: str, rank: int, spread_gpus: bool, gpu_count: int) -> int | None:
    """The CUDA index a rank selects. With ``spread_gpus`` and a bare ``cuda`` device the
    ranks take the visible GPUs in turn (rank r -> r % gpu_count); an explicit ``cuda:N``
    is kept for every rank. The device itself stays bare ``cuda`` (the current device)
    so the rollout/learner device comparisons in the trainer are unchanged."""
    index = cuda_index(device)
    if spread_gpus and index is None and torch.device(device).type == "cuda":
        return rank % max(1, gpu_count)
    return index


def _rank_main(rank: int, world: int, port: int, argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        index = rank_cuda_index(args.device, rank, args.spread_gpus,
                                torch.cuda.device_count() if torch.cuda.is_available() else 0)
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
        try:
            history_ppo.run_profiled(trainer, rank)
        finally:
            trainer.close()
    finally:
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = history_ppo.build_parser()
    parser.description = ("History PPO over --world-size local ranks. --num-envs, "
                          "--num-threads and --minibatch-matches are per rank.")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--spread-gpus", action="store_true",
                        help="with --device cuda, rank r uses visible GPU r %% device_count "
                             "(collectives stay on host copies, so nothing else changes)")
    parser.add_argument("--ddp-global-minibatch", action="store_true",
                        help="normalize advantages and weight the loss over the union of "
                             "the ranks' minibatches, as one minibatch would be")
    parser.add_argument("--ddp-reduce", choices=("gloo", "shm"), default="gloo",
                        help="gradient all-reduce transport: gloo, or a same-node /dev/shm "
                             "mapping with a fixed rank-order sum (bitwise equal to gloo at "
                             "two ranks; FP32 order differs beyond)")
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
