"""Bounded, frozen-policy collection diagnostics and exact replay hashes.

Run one case per process and compare the same case across source revisions.
Randomly initialized, distinct snapshot weights exercise the production seat
sampler; these snapshots do not represent a trained opponent population.
No optimizer or critic is constructed and no weights are updated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import cProfile
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pstats
import subprocess
import time

import gd
import gd._gd_core as engine
import numpy as np
import torch

from infra.cpu_budget import host_facts
from infra.history_artifacts import ROOT, engine_digest, sha256, source_files
from train.history_model import HistoryActor, HistoryPolicyConfig
from train.history_population import HistoryPopulation, weights_digest
from train.history_rollout import HistoryCollector, MatchEventStore, SequenceRolloutBuffer
from train.history_transfers import runtime_settings


class ExactDigest:
    """Length-framed names, metadata and bytes; preserve dtype and shape."""

    def __init__(self):
        self.digest = hashlib.sha256()

    def raw(self, value: bytes):
        self.digest.update(len(value).to_bytes(8, "little"))
        self.digest.update(value)

    def json(self, name: str, value):
        self.raw(name.encode())
        self.raw(json.dumps(value, sort_keys=True, separators=(",", ":"),
                            allow_nan=False).encode())

    def array(self, name: str, value):
        value = np.ascontiguousarray(value)
        self.json(name, {"dtype": value.dtype.str, "shape": value.shape})
        self.raw(value.tobytes())

    def hexdigest(self):
        return self.digest.hexdigest()


def initialized_actor(config, seed, causal_sdpa, device="cpu", batched_attention=False,
                      wide_projection=False):
    torch.manual_seed(seed)
    actor = HistoryActor(config).float().train().requires_grad_(False)
    actor.causal_sdpa = causal_sdpa
    actor.batched_private_attention = batched_attention
    actor.wide_private_projection = wide_projection
    return actor.to(device)


def make_population(args, config):
    """Use the normal assignment algorithm with reproducible distinct weights."""
    device = getattr(args, "device", "cpu")
    batched_attention = getattr(args, "batched_private_attention", False)
    wide = getattr(args, "wide_private_projection", False)
    if wide and not batched_attention:
        raise ValueError("--wide-private-projection requires --batched-private-attention")
    actor = initialized_actor(config, args.seed, args.causal_sdpa, device, batched_attention, wide)
    count = args.identities
    if count is None:
        count = {"current": 0, "recent": 4, "wide": 16}[args.case]
    if args.case == "current" and count:
        raise ValueError("the current case requires --identities 0")
    if args.case != "current" and count < 1:
        raise ValueError("population cases require at least one snapshot identity")
    recent = count if args.case == "recent" else min(args.recent, max(1, count))
    population = HistoryPopulation(
        actor, "history-host-frozen", args.seed + 17, recent=max(1, recent),
        snapshot_probability=args.snapshot_probability,
        archive_every=1 if args.case == "wide" else 0,
        archive_size=max(1, count), archive_share=args.archive_share)
    initial = {0: weights_digest(actor.state_dict())}
    for index in range(1, count + 1):
        identity = population.snapshot(index)
        seeded = initialized_actor(config, args.seed + 100003 * index, args.causal_sdpa,
                                   device, batched_attention, wide)
        population.models[identity].load_state_dict(seeded.state_dict())
        initial[identity] = weights_digest(population.models[identity].state_dict())
        population.metadata[identity]["sha256"] = initial[identity]
    if len(set(initial.values())) != len(initial):
        raise AssertionError("snapshot initialization produced duplicate weights")
    return actor, population, initial


def chunk_hashes(collector, stats, population):
    """Hash all replay-visible arrays and state before releasing this chunk."""
    parts = {}
    digest = ExactDigest()
    for step, choices in enumerate(collector.choice_log):
        digest.array(str(step), choices)
    parts["choices"] = digest.hexdigest()

    digest = ExactDigest()
    for name, value in sorted(collector.buffer.compact().items()):
        digest.array(name, value)
    for name in ("samples", "value", "reward", "done", "advantage", "returns"):
        digest.array(name, getattr(collector.buffer, name))
    digest.json("trajectories", [asdict(t) for t in collector.buffer.trajectories])
    digest.json("open", sorted((list(key), value) for key, value in collector.buffer.open.items()))
    parts["buffer"] = digest.hexdigest()

    digest = ExactDigest()
    for key, stream in sorted(collector.store.streams.items()):
        digest.json("stream", [list(key), stream.match_id, stream.generation])
        for name, value in zip(("tokens", "rounds", "phases"), stream.arrays()):
            digest.array(name, value)
    digest.json("current", sorted(collector.store.current.items()))
    parts["public_streams"] = digest.hexdigest()

    digest = ExactDigest()
    digest.array("sampler", collector.generator.get_state().cpu().numpy())
    digest.array("torch_global", torch.get_rng_state().numpy())
    if collector.generator.device.type == "cuda":
        digest.array("cuda_global", torch.cuda.get_rng_state(collector.device).cpu().numpy())
    digest.json("population_rng", population.rng.bit_generator.state)
    parts["generator"] = digest.hexdigest()

    digest = ExactDigest()
    digest.json("assignments", [(list(k), v.tolist()) for k, v in sorted(collector.assignments.items())])
    digest.json("seat_matches", sorted(population.seat_matches.items()))
    digest.json("policy_decisions", sorted(collector.policy_decisions.items()))
    parts["population"] = digest.hexdigest()

    digest = ExactDigest()
    digest.json("stats", {k: v for k, v in asdict(stats).items() if k != "phase_seconds"})
    parts["stats"] = digest.hexdigest()
    digest = ExactDigest()
    digest.json("parts", parts)
    return dict(parts=parts, sha256=digest.hexdigest())


def source_receipt():
    # Also works in a git archive without .git or a source-identity.json file.
    files = {str(path.relative_to(ROOT)): sha256(path) for path in source_files()}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              text=True, capture_output=True)
    return dict(revision=revision.stdout.strip() if revision.returncode == 0 else None,
                source_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
                engine_source_sha256=engine_digest(), engine_binary_sha256=sha256(Path(engine.__file__)))


def run_case(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diagnostics require an available CUDA device")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = HistoryPolicyConfig(width=args.width, layers=args.layers, heads=args.heads,
                                 response_mode=args.response_mode)
    actor, population, initial_weights = make_population(args, config)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    store = MatchEventStore()
    assignments = []
    env = gd.VecEnv(num_envs=args.envs, num_threads=args.engine_threads,
                    seed=args.seed, log_public_actions=True, log_env_limit=args.envs)
    collector = HistoryCollector(
        env, actor, store, SequenceRolloutBuffer(), generator, args.device,
        seat_policy=population.assignment,
        resolve_policy=population.resolve, record_choices=True,
        assignment_log=assignments.append, kv_cache=args.kv_cache, profile=args.profile,
        private_graphs=getattr(args, "private_graphs", False),
        triton_cache=getattr(args, "triton_cache", False),
        triton_min_batch=getattr(args, "triton_min_batch", 1),
        temperature=args.temperature, epsilon=args.epsilon)
    # One small Python counter per actual inference call, shared by all revisions.
    batch_counts = Counter()
    for identity in initial_weights:
        model = population.resolve(identity)
        method_name = "explore" if identity == 0 else "act"
        original = getattr(model, method_name)

        def counted(*call_args, _identity=identity, _original=original, **kwargs):
            batch_counts[_identity] += 1
            return _original(*call_args, **kwargs)

        setattr(model, method_name, counted)

    profiler = cProfile.Profile() if args.cprofile else None
    rows = []
    run_digest = ExactDigest()
    run_digest.json("weights", initial_weights)
    for chunk in range(args.chunks):
        collector.policy_decisions.clear()
        collector.choice_log.clear()
        batch_counts.clear()
        # Match the production learner-update cache boundary with frozen weights.
        collector.invalidate_learner_cache()
        measured = chunk >= args.warmup
        if profiler and measured:
            profiler.enable()
        if args.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        stats = collector.collect(args.steps, version=chunk)
        if args.device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        if profiler and measured:
            profiler.disable()
        hashes = chunk_hashes(collector, stats, population)
        run_digest.json(str(chunk), hashes)
        inferred = sum(collector.policy_decisions.values())
        row = dict(chunk=chunk, measured=measured, collect_seconds=seconds,
                   decisions_per_second=stats.decisions / seconds,
                   stats=asdict(stats), mean_prefix=stats.mean_prefix,
                   inference_decisions=inferred,
                   mean_policy_batch=inferred / stats.policy_batches if stats.policy_batches else 0,
                   policies={str(identity): dict(decisions=collector.policy_decisions.get(identity, 0),
                                               batches=batch_counts[identity],
                                               mean_batch=collector.policy_decisions.get(identity, 0)
                                               / batch_counts[identity] if batch_counts[identity] else 0)
                             for identity in initial_weights},
                   collection_cache=collector.cache_metrics(),
                   private_graphs=collector.graph_metrics(), exact=hashes)
        row["cache_transfers"] = collector.transfer_metrics()
        if args.device == "cuda":
            row.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        rows.append(row)
        # Deliberately discard diagnostic rows; no GAE, critic or optimizer work.
        collector.buffer = SequenceRolloutBuffer()
        store.prune(set())
    final_weights = {identity: weights_digest(population.resolve(identity).state_dict())
                     for identity in initial_weights}
    if initial_weights != final_weights:
        raise AssertionError("frozen policy weights changed")
    measured_rows = rows[args.warmup:]
    seconds = sum(row["collect_seconds"] for row in measured_rows)
    decisions = sum(row["stats"]["decisions"] for row in measured_rows)
    inferred = sum(row["inference_decisions"] for row in measured_rows)
    batches = sum(row["stats"]["policy_batches"] for row in measured_rows)
    phases = Counter()
    for row in measured_rows:
        phases.update(row["stats"]["phase_seconds"])
    result = dict(
        schema=1, case=args.case, device=args.device, dtype="float32", torch=torch.__version__,
        gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
        host=host_facts(), source=source_receipt(), model=asdict(config),
        inference=dict(runtime_settings(collector.device), causal_sdpa=actor.causal_sdpa,
                       rollout_kv_cache=collector.kv_cache,
                       rollout_batched_attention=actor.batched_private_attention,
                       rollout_wide_projection=actor.wide_private_projection,
                       rollout_private_graphs=collector.private_graphs,
                       rollout_triton_cache=collector.triton_cache,
                       rollout_triton_min_batch=collector.triton_min_batch,
                       reuse_cache_lengths=collector.reuse_cache_lengths),
        settings={k: v for k, v in vars(args).items() if k != "output"},
        population=population.config(), initial_weights=initial_weights,
        final_weights=final_weights, seat_assignments=assignments, chunks=rows,
        exact_sha256=run_digest.hexdigest(),
        summary=dict(measured_chunks=len(measured_rows), collect_seconds=seconds,
                     decisions=decisions, decisions_per_second=decisions / seconds,
                     inference_decisions=inferred, policy_batches=batches,
                     mean_policy_batch=inferred / batches if batches else 0,
                     max_prefix=max(row["stats"]["prefix_max"] for row in measured_rows),
                     phase_seconds=dict(phases)),
        claim=f"{args.device.upper()} frozen-policy diagnostic; no training or playing-strength claim",
        limitations=["Synthetic distinct initial weights; population has not been trained.",
                     "Buffer and replay hashing occur outside collection timings; choice recording and batch counters are timed.",
                     "Phase profiling and cProfile perturb throughput; compare unprofiled runs for speed.",
                     "Exact digest covers every chunk, including warmup; excludes timings and cache internals.",
                     "Pending engine events after the final step are drained on the next step, as in production."])
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if profiler:
        profiler.dump_stats(args.output / "collection.prof")
        with (args.output / "profile.txt").open("w") as output:
            pstats.Stats(profiler, stream=output).strip_dirs().sort_stats("cumulative").print_stats(80)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("current", "recent", "wide"), default="recent")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batched-private-attention", action="store_true")
    parser.add_argument("--wide-private-projection", action="store_true",
                        help="one q/out projection over all rows (FP32, not bitwise)")
    parser.add_argument("--private-graphs", action="store_true")
    parser.add_argument("--triton-cache", action="store_true")
    parser.add_argument("--triton-min-batch", type=int, default=1)
    parser.add_argument("--identities", type=int, help="snapshot identities, excluding current policy 0")
    parser.add_argument("--recent", type=int, default=4, help="recent tail size for the wide case")
    parser.add_argument("--snapshot-probability", type=float, default=0.5)
    parser.add_argument("--archive-share", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026092701)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--engine-threads", type=int, default=1)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--response-mode", choices=("none", "auxiliary", "explicit"), default="auxiliary")
    parser.add_argument("--causal-sdpa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kv-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--chunks", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cprofile", action="store_true")
    args = parser.parse_args(argv)
    if (min(args.envs, args.torch_threads, args.engine_threads, args.steps, args.chunks, args.recent) < 1
            or not 0 <= args.warmup < args.chunks or (args.identities is not None and args.identities < 0)):
        parser.error("positive sizes and chunks greater than nonnegative warmup required")
    if not np.isfinite(args.temperature) or args.temperature <= 0 or not 0 <= args.epsilon <= 1:
        parser.error("positive finite temperature and epsilon in [0, 1] required")
    if (args.output / "result.json").exists():
        parser.error("result.json already exists; choose a fresh output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    result = run_case(args)
    print(json.dumps({name: result[name] for name in ("case", "summary", "exact_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
