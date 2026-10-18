"""Planted-habit oracle value: styled table, estimators and rollout plumbing."""
import numpy as np
import torch

from bench.history_habit_value import StyledTable, node_metrics
from bench.history_value_tasks import rollout
from train.history_habit import style_feature
from train.history_model import segment_log_softmax
from tests.test_history_value_tasks import fixture, small_actor


def styled_pair(actor, axis, b):
    out = {}
    for t, z in enumerate((-1, 1)):
        m = StyledTable(actor.config)
        m.load_state_dict(actor.state_dict())
        out[t] = m.set_style(axis, b, z).eval().requires_grad_(False)
    return out


def test_styled_table_applies_the_tilt_once():
    import gd
    from bench.history_value_tasks import stream_from
    from train.history_model import DecisionInputs, StreamBatch
    actor = small_actor()
    state, arrays, _ = fixture()
    engine = gd.Engine()
    legal = engine.legal_actions(state)
    sb = StreamBatch.from_arrays([arrays], 'cpu')
    inputs = DecisionInputs(sb, torch.zeros(1, dtype=torch.long), sb.lengths,
                            torch.tensor(np.asarray(state.observation(0))[None]), torch.tensor([0]),
                            torch.tensor(np.asarray([engine.encode_action(a, state, 0) for a in legal])),
                            torch.tensor([0, len(legal)]), one_decision_per_stream=True)
    with torch.inference_mode():
        base = actor.candidate_log_probs(inputs)
        for axis in ('pass', 'bomb', 'lead_single'):
            pair = styled_pair(actor, axis, 2.0)
            for t, z in enumerate((-1, 1)):
                got = pair[t]._log_prob_table(inputs, None, len(legal))[0]
                want = segment_log_softmax(base + z * 2.0 * style_feature(inputs, axis),
                                           inputs.rows, inputs.decisions)
                torch.testing.assert_close(got, want.float())


def test_metrics_separate_type_difficulty_from_response_value():
    rng = np.random.default_rng(0)
    noise = lambda: rng.normal(0, 1e-3, size=(2, 3, 64))
    # Type 1 is harder for every action by the same amount: no value, no interaction.
    q = np.array([[.6, .5, .4], [.4, .3, .2]])[:, :, None]
    m = node_metrics(q + noise(), q + noise())
    assert abs(m['plugin']) < 1e-2 and abs(m['crossfit']) < 1e-2
    assert abs(m['interaction_sq']) < 1e-4 and m['type_effect'] < -.19
    assert not m['top_differs']
    # Types prefer different actions: oracle value 0.1 for both estimators.
    q = np.array([[.6, .4, .4], [.4, .6, .4]])[:, :, None]
    m = node_metrics(q + noise(), q + noise())
    assert abs(m['plugin'] - .1) < 1e-2 and abs(m['crossfit'] - .1) < 1e-2
    assert m['interaction_sq'] > .01 and m['top_differs']


def test_crossfit_does_not_reward_pure_selection_noise():
    rng = np.random.default_rng(1)
    plug, cross = [], []
    for _ in range(400):
        qa, qb = rng.binomial(1, .5, size=(2, 2, 6, 16)).astype(float)
        m = node_metrics(qa, qb)
        plug.append(m['plugin'])
        cross.append(m['crossfit'])
    assert np.mean(plug) > .02            # max selection inflates the plug-in
    assert abs(np.mean(cross)) < .01      # independent scoring does not


def test_styled_rollout_runs_through_the_cached_path():
    actor = small_actor()
    state, arrays, cand = fixture()
    pair = styled_pair(actor, 'pass', 8.0)
    q = rollout(actor, pair, state, arrays, cand, np.arange(1, 9))
    assert q.shape == (2, len(cand), 8, 2)
    assert np.isin(q[..., 0], [0, 1]).all()
