"""T7 response labels, information/gradient boundaries and B/C isolation."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from train.history_model import HistoryPolicyConfig, PublicStream, fresh_player, load_history_checkpoint
from train.history_ppo import HistoryPPOConfig, HistoryTrainer
from train.history_response import PLAY_ACTION_TYPES, RESPONSE_CLASSES, opponent_response_labels
from train.history_rollout import MatchEventStore
from train.logs import TOKEN_DIM


def action(kind=1):
    result = np.zeros(154, np.uint8)
    result[108 + kind] = 1
    return result


def add(stream, seat, kind=1, rnd=0, private_flags=False):
    token = np.zeros(TOKEN_DIM, np.uint8)
    token[seat] = token[158 + 10] = 1
    token[4:158] = action(kind)
    event = SimpleNamespace(seat=seat, cards_left=10, round_index=rnd, phase=0,
                            encoded_action=token[4:158].copy(), forced=private_flags)
    # An event's private engine flag must never affect the public target.
    stream.append(event)


def synthetic(following, complete=True):
    store = MatchEventStore()
    stream = store.stream(0, 0)
    add(stream, 3, 2)                  # past opponent action, not the target
    add(stream, 0, 1)                  # executed candidate at prefix=1
    for event in following:
        add(stream, **event)
    data = {key: np.array([value]) for key, value in
            dict(env=0, match=0, round=0, seat=0, prefix=1, phase=0,
                 traj=0, cand_start=0, chosen=0).items()}
    data['cand'] = np.stack([action(1)])
    buffer = SimpleNamespace(compact=lambda: data,
                             trajectories=[SimpleNamespace(complete=complete)])
    return buffer, store


@pytest.mark.parametrize('following,expected', [
    ([dict(seat=1, kind=0)], 1),
    ([dict(seat=2, kind=3), dict(seat=3, kind=7)], 1 + PLAY_ACTION_TYPES + 7),
    ([dict(seat=1, kind=10), dict(seat=3, kind=1)], 11),
    ([dict(seat=2), dict(seat=0), dict(seat=1)], 0),
    ([dict(seat=1, rnd=1)], 0),
    ([], 0),
])
def test_public_target_horizon(following, expected):
    buffer, store = synthetic(following)
    assert opponent_response_labels(buffer, store, np.array([0])).tolist() == [expected]


def test_no_private_flags_or_unfinished_absence_labels():
    a = synthetic([dict(seat=1, kind=0, private_flags=False)])
    b = synthetic([dict(seat=1, kind=0, private_flags=True)])
    assert np.array_equal(opponent_response_labels(*a, np.array([0])),
                          opponent_response_labels(*b, np.array([0])))
    buffer, store = synthetic([], complete=False)
    with pytest.raises(ValueError, match='completed round'):
        opponent_response_labels(buffer, store, np.array([0]))


def test_misaligned_executed_action_rejected():
    buffer, store = synthetic([dict(seat=1)])
    buffer.compact()['cand'][0] = action(2)
    with pytest.raises(ValueError, match='stored action'):
        opponent_response_labels(buffer, store, np.array([0]))


def test_matched_initialization_and_explicit_connection():
    cfg = HistoryPolicyConfig(width=16, layers=1, heads=2, action_width=16, fusion_width=16)
    models = [fresh_player(replace(cfg, response_mode=mode), 19)
              for mode in ('none', 'auxiliary', 'explicit')]
    base, auxiliary, explicit = [a for a, _ in models]
    for actor, critic in models[1:]:
        for name, value in base.state_dict().items():
            assert torch.equal(value, actor.state_dict()[name])
        for name, value in models[0][1].state_dict().items():
            assert torch.equal(value, critic.state_dict()[name])
    for name, value in auxiliary.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name])
    assert sum(p.numel() for p in auxiliary.parameters()) == sum(p.numel() for p in explicit.parameters())
    state, cand, offsets = torch.randn(2, 16), torch.randn(7, 154), torch.tensor([0, 3, 7])
    original = base.candidate_logits(state, cand, offsets)
    assert torch.equal(original, auxiliary.candidate_logits(state, cand, offsets))
    assert torch.equal(original, explicit.candidate_logits(state, cand, offsets))
    with torch.no_grad():
        auxiliary.response_bridge.weight.normal_()
        explicit.response_bridge.weight.copy_(auxiliary.response_bridge.weight)
    assert torch.equal(original, auxiliary.candidate_logits(state, cand, offsets))
    assert not torch.allclose(original, explicit.candidate_logits(state, cand, offsets))
    # Policy gradients cannot turn the predictor into arbitrary reward features.
    explicit.candidate_logits(state, cand, offsets).sum().backward()
    assert explicit.response_bridge.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in explicit.response_head.parameters())
    # Both modes still train their shared features via the same prediction task.
    auxiliary.zero_grad(set_to_none=True)
    state = state.requires_grad_()
    _, prediction = auxiliary.candidate_outputs(state, cand, offsets, predict=True)
    torch.nn.functional.cross_entropy(prediction, torch.arange(7)).backward()
    assert state.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in auxiliary.response_head.parameters())


@pytest.mark.parametrize('mode', ['auxiliary', 'explicit'])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_real_rollout_labels_probability_parity_and_resume(tmp_path, mode, device):
    cfg = HistoryPPOConfig(width=16, layers=1, heads=2, num_envs=2, num_threads=1,
                           steps_per_update=90, seed=29, torch_threads=1, epochs=1,
                           minibatch_matches=1, response_mode=mode,
                           rollout_kv_cache=True, causal_sdpa=True)
    trainer = HistoryTrainer(cfg, tmp_path/'run', device=device)
    trainer.collect()
    trainer.buffer.finalize(trainer.refresh_values())
    rows = trainer.buffer.samples
    assert len(rows) > 0
    targets = opponent_response_labels(trainer.buffer, trainer.store, rows)
    assert ((0 <= targets) & (targets < RESPONSE_CLASSES)).all()
    assert (targets > 0).any()
    data = trainer.buffer.compact()
    with torch.no_grad():
        logs, _ = trainer.recompute_log_probs(rows)
    assert np.max(np.abs(logs.cpu().numpy() - data['logp'][rows])) < 1e-5
    # Recompute with the future cut off: neither labels nor their metadata enter
    # the actor. Use one row so there is a unique causal prefix for the stream.
    one = rows[:1]
    row = int(one[0])
    key = (int(data['env'][row]), int(data['match'][row]))
    old = trainer.store.stream(*key)
    truncated = PublicStream(old.match_id)
    prefix = int(data['prefix'][row])
    for i in range(prefix):
        truncated.append_token(old.tokens[i], old.rounds[i], old.phases[i])
    with torch.no_grad():
        full, _ = trainer.recompute_log_probs(one)
        short, _ = trainer.recompute_log_probs(one, {key: truncated})
        full_inputs, _ = trainer.buffer.decision_inputs(one,trainer.store,device)
        short_inputs, _ = trainer.buffer.decision_inputs(one,trainer.store,device,{key:truncated})
        predictions = []
        for inputs in (full_inputs,short_inputs):
            state = trainer.actor.decision_states(None,inputs)
            predictions.append(trainer.actor.candidate_outputs(state,inputs.cand,inputs.offsets,
                                                               predict=True)[1])
    torch.testing.assert_close(full, short, rtol=0, atol=1e-5)
    torch.testing.assert_close(*predictions,rtol=0,atol=1e-5)
    before = {n:p.clone() for n,p in trainer.actor.response_head.named_parameters()}
    stats = trainer.learn()
    assert np.isfinite(stats['response_loss']) and stats['response_loss'] > 0
    assert 0 <= stats['response_accuracy'] <= 1
    assert any(not torch.equal(before[n],p) for n,p in trainer.actor.response_head.named_parameters())
    trainer.save()
    actor, _, payload = load_history_checkpoint(tmp_path/'run/latest.pt')
    assert actor.config.response_mode == mode
    resumed = HistoryTrainer(cfg, tmp_path/'resumed', resume=tmp_path/'run/latest.pt', device=device)
    for name, value in trainer.actor.state_dict().items():
        assert torch.equal(value, resumed.actor.state_dict()[name])
    assert payload['config']['response_coef'] == cfg.response_coef


@pytest.mark.parametrize('mode,coef', [('bad',.1), ('explicit',0), ('auxiliary',float('nan'))])
def test_invalid_prediction_config(mode, coef):
    with pytest.raises(ValueError):
        HistoryPPOConfig(response_mode=mode, response_coef=coef)
