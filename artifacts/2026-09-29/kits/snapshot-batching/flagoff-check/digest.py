"""Digest of collector and trainer behaviour with merged snapshot inference OFF.

Run once with PYTHONPATH at the parent commit (8303769) and once at the branch;
equal digests show the default path is bitwise unchanged on this machine.
"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, "tests")
from train import history_rollout as rollout
from train.history_model import HistoryPolicyConfig, fresh_player
from train.history_ppo import HistoryPPOConfig, HistoryTrainer
from test_history_rollout import make_env

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
digest = hashlib.sha256()
records = {}


def add(name, value):
    part = hashlib.sha256(value).hexdigest()
    records[name] = part
    digest.update(part.encode())


config = HistoryPolicyConfig(width=32, layers=1, heads=4, action_width=16, fusion_width=24,
                             response_mode="auxiliary")
for kv_cache in (False, True):
    for temperature, epsilon in ((1.0, 0.0), (0.8, 0.2)):
        policies = {i: fresh_player(config, 40 + i)[0] for i in range(4)}
        collector = rollout.HistoryCollector(
            make_env(6, 23), policies[0], rollout.MatchEventStore(),
            rollout.SequenceRolloutBuffer(), torch.Generator().manual_seed(31),
            seat_policy=lambda env, match: [0, 1 + env % 3, 2, 1 + (env + match) % 3],
            resolve_policy=policies.__getitem__, record_choices=True, kv_cache=kv_cache,
            temperature=temperature, epsilon=epsilon)
        collector.collect(40)
        name = f"collector kv={kv_cache} T={temperature} eps={epsilon}"
        add(name + " choices", b"".join(c.tobytes() for c in collector.choice_log))
        add(name + " rows", b"".join(v.tobytes()
                                     for _, v in sorted(collector.buffer.compact().items())))
        add(name + " sampler", collector.generator.get_state().numpy().tobytes())
        add(name + " decisions", json.dumps(sorted(collector.policy_decisions.items())).encode())

with tempfile.TemporaryDirectory() as tmp:
    base = dict(width=16, layers=1, heads=4, num_envs=4, steps_per_update=30, epochs=1,
                minibatch_matches=1, snapshot_updates=1, seed=5, causal_sdpa=True,
                rollout_kv_cache=True, snapshot_probability=1.0)
    start = HistoryTrainer(HistoryPPOConfig(**base), Path(tmp) / "start")
    for _ in range(3):
        start.update()
    checkpoint = start.save()
    trainer = HistoryTrainer(HistoryPPOConfig(updates=6), Path(tmp) / "resumed",
                             resume=checkpoint)
    for _ in range(3):
        line = trainer.update()
    add("trainer actor", b"".join(p.detach().numpy().tobytes()
                                  for p in trainer.actor.parameters()))
    add("trainer sampler", trainer.generator.get_state().numpy().tobytes())
    add("trainer counters", json.dumps({k: line[k] for k in (
        "decisions", "rounds", "update_samples", "policy_loss", "entropy",
        "collection_policy_batches")}).encode())
print(json.dumps(dict(total=digest.hexdigest(), parts=records), indent=1))
