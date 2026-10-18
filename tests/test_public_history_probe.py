from types import SimpleNamespace

import numpy as np

from bench.public_history_analysis import Tables, public_rows, softmax
from bench.public_history_probe import PublicCollector, PublicTap
from bench.public_history_residual import adjust


def encoded(kind):
    a = np.zeros(154, np.uint8)
    a[108+kind] = 1
    if kind:
        a[0] = 1
    return a


def test_public_tap_never_reads_forced_action_or_private_properties(tmp_path):
    class Event:
        env_id = round_index = step = match_id = seat = 0
        cards_left = 26
        phase = 3
        encoded_action = encoded(1)
        @property
        def forced(self):
            raise AssertionError("forced is not public")
        @property
        def action(self):
            raise AssertionError("use only redacted public encoding")
    tap = PublicTap(SimpleNamespace(drain_public_actions=lambda:[Event()]))
    tap.drain_public_actions()
    tap.save(tmp_path/"public.npz")
    with np.load(tmp_path/"public.npz", allow_pickle=False) as data:
        assert set(data.files) == {"public_meta", "public_actions", "public_round_results"}
        assert data["public_actions"].shape == (1, 146)


def test_collection_does_not_even_read_hidden_counts():
    class Batch:
        rows = 1
        offsets = [0, 1]
        env_id = match_id = round_index = seat = greedy_choice = [0]
        phase = [3]
        obs = np.zeros((1, 1849))
        cand = np.zeros((1, 154))
        @property
        def hidden_counts(self):
            raise AssertionError("do not read privileged data")
    c = PublicCollector.__new__(PublicCollector)
    c.assignment = lambda e,m: [0]*4
    c.store = SimpleNamespace(stream=lambda e,m:SimpleNamespace(prefix=0))
    c._assignments_changed = False
    r = c._read_rows(Batch())
    assert r.hidden.shape == (1, 0) and r.acting.tolist() == [True]


def test_context_is_pre_action_and_public_passes_close_trick():
    metadata = np.asarray([[0,0,i,s,26 if s == 0 else 27,3]
                           for i,s in enumerate([0,1,2,3,0])])
    actions = np.asarray([encoded(t)[:146] for t in [1,0,0,0,1]])
    rows = list(public_rows(metadata, actions, [], team=1))
    assert [r[1] for r in rows] == [1,0,1]  # enemy seat 2 also publicly passes
    changed = metadata.copy()
    changed[0,4] = 1
    changed_rows = list(public_rows(changed, actions, [], team=1))
    assert rows[0] == changed_rows[0], "post-action hand count must not be a predictor"


def test_public_history_formula_detects_synthetic_style_without_label_input():
    tables = Tables()
    tables.prefix_counts[:] = 1
    for identity in range(4):
        tables.prefix_counts[:, identity, :, identity] += 1000
    tables.finish()
    for identity in range(4):
        history = np.zeros((2,4))
        history[:,identity] = 40
        posterior = softmax(tables.history_logits(0, history))
        assert posterior.argmax() == identity and posterior[identity] > .99


def test_indistinguishable_public_styles_produce_no_identity_information():
    tables = Tables()
    tables.prefix_counts[:] = np.arange(1,5)
    tables.finish()
    history = np.asarray([[1,20,3,4],[20,1,8,9]])
    assert np.allclose(softmax(tables.history_logits(1,history)), .25)


def test_direct_residual_adjustment_is_normalized_and_zero_residual_is_identity():
    p = np.asarray([.1,.3,.2,.4])
    assert np.allclose(adjust(p,np.zeros(4),10),p)
    q = adjust(p,np.asarray([-100,100,0,0]),10)
    assert np.isfinite(q).all() and (q > 0).all() and np.isclose(q.sum(),1)
