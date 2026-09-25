"""Synchronous rollout actor processes for Stage B PPO (STAGE_B_TODO B5b).

`PPOConfig.actor_processes = W > 0` splits the `num_envs` environments into
W shards, each stepped by its own process with its own `gd.VecEnv`, its own
copy of the policy and frozen reference on the trainer's device, its own
opponent and its own sampling generator. A single rollout process is bound by
one Python thread (batch assembly, copies, kernel launches, buffer writes);
W processes run those in parallel on a many-core host.

The learner process keeps everything else. Every shard's `RolloutBuffer`
lives in shared memory: an actor writes its rounds there, the learner reads
them in place (critic values, GAE, minibatches over the union of shards), so
no trajectory data is pickled. Per update:

1. the learner publishes the policy weights into shared tensors;
2. every actor loads them, clears its shard's finished rounds
   (`next_iteration`, carrying rounds in progress), and collects
   `rollout_steps` vector steps;
3. the learner waits for all actors, then learns.

Actors never run while the learner trains, so every step collected in an
update was drawn from exactly the weights the learner starts that update
with: actors add no policy lag to the in-process path (tests/test_ppo_actors.py
bounds the first-epoch importance ratio of those steps). As in-process, a
round still open at an update boundary is carried over (`carry_over`), and
its earlier steps were drawn from the previous weights: with rounds of about
70 decisions against 64-step rollouts, about half of the trained samples are
one update old, and their ratios start away from 1. What differs from the in-process path
under the same seed is only which random streams drive what: the shards'
environment seeds and sampling generators are drawn from the trainer RNG,
and minibatches are permuted over the union of shards. The distribution of
the data and of the updates is the same.

Opponents are built per actor from the config spec (`frozen`,
`frozen:<path>`, `greedy`, `league:<pool.json>`); each actor drives its own
`OpponentSource` in the contract's order for its shard. An opponent object
passed to `PPOTrainer` is not supported here.

League (`opponent = "league:<pool.json>"`, B8). Every actor owns a `League`
for its shard (own sampler RNG, own model cache, own `max_active_models` cap,
so its network opponents batch within the actor: up to W x cap distinct
models can be live across the run). The learner owns the authoritative
league, which is never bound to an environment. At each synchronous update
boundary:

1. the learner broadcasts its league state (entries in order, sampling
   EMA and tallies, snapshot list, no RNG) with the collect command; each
   actor adopts it (`League.load_state_dict`), so every shard samples from
   the same weight table and sees every snapshot added since the last
   boundary (entries are matched by name, so a match already running
   against an evicted snapshot finishes against it);
2. during the rollout each actor updates its local copy from its own
   match results only, exactly as in-process, and logs (entry, learner won);
3. the learner applies all logs to the authoritative league in actor order
   (`League.apply_results`), which is the same update the in-process league
   makes for the same results in that order. With one actor the two paths
   hold identical weights at every boundary; with W actors a shard does not
   see the other shards' results until the next boundary.

Snapshots are written by the learner after the update (`PPOTrainer.
league_snapshot`) and reach the actors at the next boundary. Checkpoints hold
the authoritative league state plus every actor's sampler RNG; a resume with
the same number of actors restores those, otherwise the actor samplers are
reseeded from the trainer RNG.
"""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import signal
import time
import traceback
from typing import Any

import numpy as np
import torch

import gd

PROGRESS_KEYS = ("decisions", "learner_decisions", "rounds", "matches", "learner_match_wins")


def shard_sizes(num_envs: int, actors: int) -> list[int]:
    if actors <= 0 or num_envs % actors or num_envs // actors < 2:
        raise ValueError("num_envs must split into actor_processes shards of at least 2")
    return [num_envs // actors] * actors


def shared_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().to("cpu", copy=True).share_memory_()
            for k, v in module.state_dict().items()}


def publish(module: torch.nn.Module, shared: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for key, value in module.state_dict().items():
            shared[key].copy_(value)


def buffer_bytes(config) -> int:
    from train.rollout_buffer import RolloutBuffer
    return sum(a.nbytes for a in RolloutBuffer(config).storage().values())


def check_shared_memory(needed: int) -> None:
    """Shard buffers live in POSIX shared memory (/dev/shm on Linux). Docker
    gives containers 64 MiB there unless told otherwise, and running out
    kills a process with SIGBUS mid-rollout, so check up front."""
    shm = Path("/dev/shm")
    if not shm.is_dir():
        return
    stat = os.statvfs(shm)
    free = stat.f_bavail * stat.f_frsize
    if free < needed * 1.1:
        raise RuntimeError(
            f"actor_processes needs {needed / 2**30:.2f} GiB of /dev/shm for the shard buffers, "
            f"{free / 2**30:.2f} GiB free; enlarge it (docker --shm-size) or lower "
            "num_envs/buffer_candidates")


def shard_specs(trainer, sizes: list[int], league_rng: list | None) -> list[dict]:
    """One `Actor` spec per shard. Seeds are drawn from the trainer RNG in
    shard order (environment, generator, then league sampler), so a seed
    fixes every shard."""
    from train.ppo import resolve_artifact

    cfg = trainer.config
    opponent = cfg.opponent
    if opponent.startswith("frozen:"):
        opponent = "frozen:" + str(resolve_artifact(opponent[len("frozen:"):]))
    specs = []
    for index, size in enumerate(sizes):
        buffer_config = type(cfg)(**{**asdict(cfg), "num_envs": size}).buffer_config()
        spec = {
            "config": asdict(cfg), "index": index, "num_envs": size,
            "device": str(trainer.device), "opponent": opponent,
            "follow_fallback": trainer.follow_fallback(),
            "env_seed": int(trainer.rng.integers(0, 2**63)),
            "generator_seed": int(trainer.rng.integers(0, 2**63)),
            "model_config": asdict(trainer.policy.net.config),
            "policy_config": asdict(trainer.policy.config),
            "reference_checkpoint_id": trainer.policy.reference_checkpoint_id,
            "phase_code": trainer.phase_code,
            "buffer_config": asdict(buffer_config),
            "league": None,
        }
        if trainer.league is not None:
            league_config = asdict(trainer.league_config)
            league_config["seed"] = int(trainer.rng.integers(0, 2**63))
            spec["league"] = {"config": league_config, "entries": trainer.league_entries,
                              "external": sorted(trainer.league_external),
                              "rng": None if league_rng is None else league_rng[index]}
        specs.append(spec)
    return specs


def merge_replies(trainer, buffers: list, replies: list[dict]) -> dict[str, float]:
    """Fold the shards' collect replies into the learner: buffer counters,
    progress, window and (in shard order) the league results."""
    for buffer, reply in zip(buffers, replies):
        buffer.n_steps, buffer.n_cand, buffer.n_traj = reply["counters"]
        buffer.n_samples = 0
        buffer.finalized = False
        buffer.staged = None
        for key in PROGRESS_KEYS:
            trainer.progress[key] += reply["progress"][key]
        for key, value in reply["window"].items():
            trainer.window[key] += value
    if trainer.league is not None:
        # Shard order, each shard's results in match-end order.
        for reply in replies:
            trainer.league.apply_results(reply["league_results"])
        info = [reply["league_info"] for reply in replies]
        trainer.actor_league_info = {
            "league/actor_active_models_max": float(max(i["active"] for i in info)),
            "league/actor_cached_models_max": float(max(i["cached"] for i in info)),
            "league/forward_calls": float(sum(i["forward_calls"] for i in info))}
    seconds = [reply["seconds"] for reply in replies]
    return {"actor_collect_seconds_max": max(seconds),
            "actor_collect_seconds_min": min(seconds)}


class ActorPool:
    """Learner-side handle of the actor processes."""

    def __init__(self, trainer, sizes: list[int], league_rng: list | None = None) -> None:
        from train.rollout_buffer import RolloutBuffer, RolloutBufferConfig

        ctx = torch.multiprocessing.get_context("spawn")
        self.net_weights = shared_state(trainer.policy.net)
        reference_weights = shared_state(trainer.policy.reference)
        specs = shard_specs(trainer, sizes, league_rng)
        configs = [RolloutBufferConfig(**spec["buffer_config"]) for spec in specs]
        check_shared_memory(sum(buffer_bytes(c) for c in configs))
        self.buffers = []
        self.connections = []
        self.processes = []
        for index, (spec, buffer_config) in enumerate(zip(specs, configs)):
            buffer, tensors = RolloutBuffer.shared(buffer_config)
            parent, child = ctx.Pipe()
            process = ctx.Process(target=actor_main, daemon=True, name=f"ppo-actor-{index}",
                                  args=(child, spec, self.net_weights, reference_weights, tensors))
            process.start()
            child.close()
            self.buffers.append(buffer)
            self.connections.append(parent)
            self.processes.append(process)
        for connection in self.connections:
            self._receive(connection)   # "ready"

    def _receive(self, connection) -> Any:
        status, payload = connection.recv()
        if status == "error":
            self.close()
            raise RuntimeError(f"PPO actor failed:\n{payload}")
        return payload

    def collect(self, trainer, steps: int) -> dict[str, float]:
        publish(trainer.policy.net, self.net_weights)
        league_state = (None if trainer.league is None
                        else trainer.league.state_dict(include_rng=False))
        for connection in self.connections:
            connection.send(("collect", (steps, league_state, trainer.learner_reference_all)))
        replies = [self._receive(connection) for connection in self.connections]
        return merge_replies(trainer, self.buffers, replies)

    def request(self, command: str) -> list[Any]:
        for connection in self.connections:
            connection.send((command, None))
        return [self._receive(connection) for connection in self.connections]

    def close(self) -> None:
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
        self.connections, self.processes = [], []


class LocalPool:
    """`PPOConfig.rollout_pipeline`: the shards of `ActorPool`, built from the
    same specs, but held in this process and stepped in turn.

    Each round of the loop resumes every shard's `collect_steps` once: a
    shard copies back its previous step's results, steps its engine, reads
    the next pending batch and launches that batch's forwards, then yields.
    On CUDA every shard runs on its own stream and synchronises only it, so
    the device works through one shard's forwards while the host serves the
    next. Elsewhere the shards simply alternate. The data, the league
    semantics and the no-lag property are those of the actor processes; only
    the host scheduling differs, so wall time is the only thing it changes.

    Stream safety: a shard's pinned upload buffers and host arrays belong to
    its own `Uploader`, rewritten only after that shard's stream was
    synchronised by its own copy back (`to_host`), as in the one-shard path.
    Weights are loaded on the default stream in `Actor.begin`, which every
    shard stream waits for before its first launch."""

    def __init__(self, trainer, sizes: list[int], league_rng: list | None = None) -> None:
        from train.rollout_buffer import RolloutBuffer, RolloutBufferConfig

        self.net_weights = {k: v.detach().to("cpu", copy=True)
                            for k, v in trainer.policy.net.state_dict().items()}
        reference_weights = {k: v.detach().to("cpu", copy=True)
                             for k, v in trainer.policy.reference.state_dict().items()}
        self.device = trainer.device
        self.buffers, self.actors, self.streams = [], [], []
        for spec in shard_specs(trainer, sizes, league_rng):
            buffer = RolloutBuffer(RolloutBufferConfig(**spec["buffer_config"]))
            tensors = {k: torch.from_numpy(a) for k, a in buffer.storage().items()}
            actor = Actor(spec, self.net_weights, reference_weights, tensors)
            self.buffers.append(buffer)
            self.actors.append(actor)
            self.streams.append(torch.cuda.Stream(self.device)
                                if self.device.type == "cuda" else None)
        self.first = True

    def collect(self, trainer, steps: int) -> dict[str, float]:
        publish(trainer.policy.net, self.net_weights)
        league_state = (None if trainer.league is None
                        else trainer.league.state_dict(include_rng=False))
        argument = (steps, league_state, trainer.learner_reference_all)
        for actor, stream in zip(self.actors, self.streams):
            actor.begin(argument, self.first)
            if stream is not None:
                stream.wait_stream(torch.cuda.current_stream(self.device))
        self.first = False
        running = [(actor.collector.collect_steps(), stream)
                   for actor, stream in zip(self.actors, self.streams)]
        while running:
            live = []
            for steps_left, stream in running:
                try:
                    if stream is None:
                        next(steps_left)
                    else:
                        with torch.cuda.stream(stream):
                            next(steps_left)
                except StopIteration:
                    continue
                live.append((steps_left, stream))
            running = live
        for stream in self.streams:
            if stream is not None:
                torch.cuda.current_stream(self.device).wait_stream(stream)
        return merge_replies(trainer, self.buffers, [actor.end() for actor in self.actors])

    def request(self, command: str) -> list[Any]:
        return [getattr(actor, command)() for actor in self.actors]

    def close(self) -> None:
        pass


def actor_main(connection, spec: dict, net_weights: dict, reference_weights: dict,
               tensors: dict) -> None:
    """Entry point of one actor process."""
    # The learner handles SIGINT/SIGTERM and closes the actors itself.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        actor = Actor(spec, net_weights, reference_weights, tensors)
        connection.send(("ok", "ready"))
        first = True
        while True:
            command, argument = connection.recv()
            if command == "close":
                break
            if command == "collect":
                connection.send(("ok", actor.collect_once(argument, first)))
                first = False
            elif command == "weights_digest":
                connection.send(("ok", actor.weights_digest()))
            elif command == "league_rng":
                connection.send(("ok", actor.league_rng()))
            elif command == "league_state":
                connection.send(("ok", actor.league_state()))
            else:
                raise ValueError(f"unknown actor command {command}")
    except (EOFError, KeyboardInterrupt):
        pass
    except BaseException:  # report anything else to the learner
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass


class Actor:
    """One shard: a `RolloutCollector` over its own environments."""

    def __init__(self, spec: dict, net_weights: dict, reference_weights: dict,
                 tensors: dict) -> None:
        from train.model import GuandanModel, ModelConfig
        from train.opponents import FrozenModelOpponent, config_opponent
        from train.policy import PolicyConfig, StageBPolicy
        from train.ppo import PPOConfig, RolloutCollector, Uploader, shares_opponent_forward
        from train.rollout_buffer import RolloutBuffer, RolloutBufferConfig

        cfg = PPOConfig(**{**spec["config"], "num_envs": spec["num_envs"]})  # this shard
        torch.set_num_threads(cfg.torch_threads)
        device = torch.device(spec["device"])
        model_config = ModelConfig(**spec["model_config"])
        net, reference = GuandanModel(model_config), GuandanModel(model_config)
        net.load_state_dict(net_weights)
        reference.load_state_dict(reference_weights)
        policy = StageBPolicy(net, reference, PolicyConfig(**spec["policy_config"]),
                              spec["reference_checkpoint_id"]).to(device)
        net.eval()
        self.net_weights = net_weights
        collector = RolloutCollector()
        collector.config = cfg
        collector.device = device
        collector.policy = policy
        collector.net = net
        collector.phase_code = spec["phase_code"]
        actions = gd.ActionConfig.full() if cfg.action_mode == "full" else gd.ActionConfig()
        size = spec["num_envs"]
        collector.env = gd.VecEnv(size, num_threads=cfg.num_threads, seed=spec["env_seed"],
                                  actions=actions)
        collector.env.reset()
        collector.learner_team = np.arange(size, dtype=np.int64) % 2
        collector.env_match = np.full(size, -1, np.int64)
        collector.buffer = RolloutBuffer(RolloutBufferConfig(**spec["buffer_config"]),
                                         arrays={k: t.numpy() for k, t in tensors.items()})
        opponent_spec = spec["opponent"]
        self.league = None
        collector.league_fused = False
        if spec["league"] is not None:
            from train.league import League, LeagueConfig

            setup = spec["league"]
            opponent = League(setup["entries"], LeagueConfig.from_dict(setup["config"]))
            opponent.external = frozenset(setup["external"])
            opponent.record_results = True
            if setup["rng"] is not None:
                opponent.rng.bit_generator.state = setup["rng"]
            collector.league_fused = bool(opponent.external)
            self.league = opponent
        else:
            opponent = config_opponent(opponent_spec, policy.reference, str(device),
                                       cfg.candidate_chunk, fallback=spec["follow_fallback"])
        collector.opponent = opponent
        opponent.bind(collector.env, collector.learner_team)
        collector.fused_opponent = (cfg.fast_rollout and isinstance(opponent, FrozenModelOpponent)
                                    and opponent.model is policy.reference)
        collector.shared_opponent = shares_opponent_forward(cfg, collector.phase_code, opponent,
                                                            collector.fused_opponent)
        collector.upload = Uploader(device)
        collector.timers = None
        collector.generator = torch.Generator(device=device)
        collector.generator.manual_seed(spec["generator_seed"])
        collector.progress = {key: 0 for key in PROGRESS_KEYS}
        collector.window = {"matches": 0, "wins": 0, "rounds": 0, "learner_return": 0.0}
        collector.bound = False
        collector.should_stop = lambda: False
        self.collector = collector

    def collect_once(self, argument: tuple, first: bool) -> dict:
        self.begin(argument, first)
        self.collector.collect()
        return self.end()

    def begin(self, argument: tuple, first: bool) -> None:
        """Prepare a collect: the learner's league state and weights, the
        shard's finished rounds cleared."""
        steps, league_state, learner_reference_all = argument
        c = self.collector
        c.learner_reference_all = learner_reference_all
        self.start = time.monotonic()
        if self.league is not None:
            self.league.load_state_dict(league_state)   # the learner's weights and snapshots
            self.league.pop_results()
        if not first:
            c.buffer.next_iteration()
        c.net.load_state_dict(self.net_weights)
        self.before = dict(c.progress)
        c.window = {"matches": 0, "wins": 0, "rounds": 0, "learner_return": 0.0}
        c.config.rollout_steps = steps

    def end(self) -> dict:
        """The reply to the learner for the collect since `begin`."""
        c = self.collector
        reply = {"counters": (c.buffer.n_steps, c.buffer.n_cand, c.buffer.n_traj),
                 "progress": {k: c.progress[k] - self.before[k] for k in PROGRESS_KEYS},
                 "window": dict(c.window), "seconds": time.monotonic() - self.start}
        if self.league is not None:
            reply["league_results"] = self.league.pop_results()
            reply["league_info"] = {"active": len(self.league.active),
                                    "cached": len(self.league.cache),
                                    "forward_calls": self.league.forward_calls}
        return reply

    def league_rng(self) -> dict | None:
        return None if self.league is None else self.league.rng.bit_generator.state

    def league_state(self) -> dict | None:
        """This actor's league state (tests: snapshots reach actors at a boundary)."""
        return None if self.league is None else self.league.state_dict()

    def weights_digest(self) -> float:
        with torch.no_grad():
            return float(sum(p.double().sum() for p in self.collector.net.parameters()))
