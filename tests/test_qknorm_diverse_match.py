from types import SimpleNamespace

import numpy as np
import pytest

from bench.qknorm_diverse_match import AblationCollector,PublicTrace


@pytest.mark.parametrize("round_streams",[False,True])
def test_only_learner_prefix_is_round_limited_without_reading_hidden_counts(round_streams):
    class Batch:
        rows=2
        offsets=[0,1,2]
        env_id=[0,1]
        match_id=round_index=[2,2]
        seat=[0,1]
        phase=[3,3]
        greedy_choice=[0,0]
        obs=np.zeros((2,1849))
        cand=np.zeros((2,154))
        @property
        def hidden_counts(self):
            raise AssertionError("privileged data must not be read")
    c=AblationCollector.__new__(AblationCollector)
    c.assignment=lambda e,m:[0,47241,0,47241]
    c.store=SimpleNamespace(stream=lambda e,m:SimpleNamespace(prefix=99+e),
        round_stream=lambda e,m,k:SimpleNamespace(prefix=7))
    c._assignments_changed=False
    c.round_streams=round_streams
    rows=c._read_rows(Batch())
    assert rows.prefix.tolist()==([7,100] if round_streams else [99,100])
    assert rows.hidden.shape==(2,0)


def test_public_trajectory_hash_uses_only_public_fields_and_detects_card_changes():
    class Event:
        env_id=match_id=round_index=seat=0
        phase=3
        cards_left=26
        encoded_action=np.zeros(154,np.uint8)
        @property
        def forced(self):
            raise AssertionError("forced flag is not public")
    event=Event()
    env=SimpleNamespace(drain_public_actions=lambda:[event])
    first=PublicTrace(env,1)
    first.drain_public_actions()
    event.encoded_action=event.encoded_action.copy()
    event.encoded_action[1]=1
    second=PublicTrace(env,1)
    second.drain_public_actions()
    assert first.counts.tolist()==second.counts.tolist()==[1]
    assert first.hashes[0].hexdigest()!=second.hashes[0].hexdigest()
