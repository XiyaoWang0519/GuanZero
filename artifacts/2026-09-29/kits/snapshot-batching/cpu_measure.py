"""CPU-only measurement: merged snapshot inference on vs off (does NOT predict CUDA).

Production architecture (width 128, 4 layers, 8 heads, auxiliary response),
KV cache on, 64 tables, 12 snapshot identities plus the learner with the
population's seat pattern (one learner anchor, other seats a snapshot with
probability 0.5). Both arms replay the same seeds; after WARMUP steps the next
STEPS steps are timed (wall) and the following COUNT steps are run under a
dispatch counter that counts backend ATen operator calls (a proxy for CUDA
kernel launches) split into the learner/snapshot actor and cache phases.
"""
import argparse
import json
import sys
import time

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, "tests")
from train import history_rollout as rollout
from train.history_model import HistoryPolicyConfig, fresh_player
from test_history_rollout import make_env


class Count(TorchDispatchMode):
    """Backend ATen calls; ``actor`` counts those inside snapshot actor calls."""

    def __init__(self):
        super().__init__()
        self.ops = 0
        self.actor = 0
        self.inside = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops += 1
        self.actor += self.inside
        return func(*args, **(kwargs or {}))


def counted(counter, function):
    def wrapper(*args, **kwargs):
        outer, counter.inside = counter.inside, True
        try:
            return function(*args, **kwargs)
        finally:
            counter.inside = outer
    return wrapper


def run(merge, args):
    torch.manual_seed(0)
    config = HistoryPolicyConfig(width=128, layers=4, heads=8, action_width=128,
                                 fusion_width=128, response_mode="auxiliary")
    policies = {i: fresh_player(config, 100 + i)[0] for i in range(args.snapshots + 1)}
    for actor in policies.values():
        actor.causal_sdpa = True
    rng = np.random.default_rng(7)

    def seats(env, match):
        anchor = int(rng.integers(4))
        return [0 if s == anchor or rng.random() < 0.5 else int(rng.integers(1, args.snapshots + 1))
                for s in range(4)]

    collector = rollout.HistoryCollector(
        make_env(args.envs, 23), policies[0], rollout.MatchEventStore(),
        rollout.SequenceRolloutBuffer(), torch.Generator().manual_seed(31), seat_policy=seats,
        resolve_policy=policies.__getitem__, kv_cache=True, batch_snapshot_policies=merge,
        record_choices=True)
    collector.collect(args.warmup)
    begin = time.perf_counter()
    stats = collector.collect(args.steps)
    wall = time.perf_counter() - begin
    collector.profile = True
    profiled = collector.collect(args.steps)
    collector.profile = False
    import train.history_snapshot_batch as batch
    from train.history_model import HistoryActor
    counter = Count()
    patches = [(batch, "merged_log_probs"), (batch, "merged_sample"), (HistoryActor, "act")]
    saved = [getattr(owner, name) for owner, name in patches]
    for owner, name in patches:
        setattr(owner, name, counted(counter, getattr(owner, name)))
    try:
        with counter:
            counted_stats = collector.collect(args.count)
    finally:
        for (owner, name), value in zip(patches, saved):
            setattr(owner, name, value)
    return dict(merge=merge, steps=args.steps, wall_s_per_step=wall / args.steps,
                decisions_per_s=stats.decisions / wall,
                policy_calls_per_step=stats.policy_batches / stats.steps,
                snapshot_rows_per_call={k: sum(s * n for s, n in v.items()) / sum(v.values())
                                        for k, v in profiled.policy_call_rows.items()},
                calls_per_step={k: sum(v.values()) / profiled.steps
                                for k, v in profiled.policy_call_rows.items()},
                profiled_group_ms_per_step={g: {p: 1e3 * s / profiled.steps for p, s in v.items()}
                                            for g, v in profiled.group_phase_seconds.items()},
                aten_ops_per_step=counter.ops / counted_stats.steps,
                snapshot_actor_aten_ops_per_step=counter.actor / counted_stats.steps,
                choices_digest=hash(b"".join(c.tobytes() for c in collector.choice_log)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--snapshots", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    results = [run(merge, args) for merge in (False, True, False, True)]
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
