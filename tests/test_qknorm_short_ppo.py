"""Guard instrumentation and learner-only migration without collecting or training."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bench.qknorm_short_ppo import migrate_learner, watch_clipping
from test_qknorm_plasticity import SmallReader, saved_state


class Reader(SmallReader):
    def auxiliary_parameter_names(self):
        return set()


def test_learner_migration_leaves_initial_frozen_pool_and_critic_unchanged():
    actor = Reader()
    with torch.no_grad():
        for p in actor.parameters():
            p.mul_(100)
    state = saved_state(actor)
    optimizer = torch.optim.Adam(actor.parameters())
    optimizer.load_state_dict(state)
    critic = nn.Linear(4, 1)
    snapshot = copy.deepcopy(actor)
    trainer = SimpleNamespace(actor=actor, actor_optimizer=optimizer, critic=critic,
        rollout_actor=actor, collector=SimpleNamespace(actor=actor),
        population=SimpleNamespace(models={1: snapshot}))
    pool_before = copy.deepcopy(snapshot.state_dict())
    critic_before = copy.deepcopy(critic.state_dict())
    q_before = actor.q_proj.weight.detach().clone()
    record = migrate_learner(trainer, {"optimizer": {"actor": state}})
    assert record["initial_population_unchanged"]
    assert not torch.equal(actor.q_proj.weight, q_before)
    assert all(torch.equal(v, pool_before[k]) for k, v in snapshot.state_dict().items())
    assert all(torch.equal(v, critic_before[k]) for k, v in critic.state_dict().items())


def test_clip_watcher_preserves_native_active_clipping_and_restores_function():
    actor = nn.Linear(2, 1)
    for p in actor.parameters():
        p.grad = torch.full_like(p, 20.)
    previous = torch.nn.utils.clip_grad_norm_
    records = []
    with watch_clipping(SimpleNamespace(actor=actor), records):
        norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.)
    assert torch.nn.utils.clip_grad_norm_ is previous
    assert float(norm) > 10 and len(records) == 1
    assert records[0]["clipped"] and records[0]["after"] <= 10.0001


def test_clip_watcher_restores_on_exception():
    previous = torch.nn.utils.clip_grad_norm_
    with pytest.raises(RuntimeError):
        with watch_clipping(SimpleNamespace(actor=nn.Linear(2, 1)), []):
            raise RuntimeError("diagnostic aborted")
    assert torch.nn.utils.clip_grad_norm_ is previous
