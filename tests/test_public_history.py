"""Public history stays usable without model, training-log or engine dependencies."""
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from train.public_history import TOKEN_DIM, PublicStream, public_token

ROOT = Path(__file__).resolve().parents[1]


def run_isolated(script: str) -> None:
    done = subprocess.run([sys.executable, "-I", "-c", textwrap.dedent(script), str(ROOT)],
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr


def test_public_schema_import_and_stream_use_need_no_engine_or_training_runtime():
    run_isolated("""
        import importlib.abc
        import sys
        from types import SimpleNamespace

        sys.path.insert(0, sys.argv[1])

        class NoTrainingRuntime(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'torch', 'gd'}:
                    raise RuntimeError(f'public schema imported {fullname}')
                if fullname in {'train.buffer', 'train.logs', 'train.history_model'}:
                    raise RuntimeError(f'public schema imported {fullname}')

        sys.meta_path.insert(0, NoTrainingRuntime())
        import numpy as np
        from train.public_history import TOKEN_DIM, TOKEN_SCHEMA_VERSION, PublicStream, public_token

        assert TOKEN_DIM == 186 and TOKEN_SCHEMA_VERSION == 1
        event = SimpleNamespace(seat=2, encoded_action=np.zeros(154, np.uint8),
                                cards_left=27, round_index=0, phase=3)
        stream = PublicStream(8)
        stream.append(event)
        np.testing.assert_array_equal(stream.tokens[0], public_token(event))
        assert stream.match_id == 8 and stream.prefix == 1
        assert stream.rounds.tolist() == [0] and stream.phases.tolist() == [3]
        stream.reset(9)
        assert stream.match_id == 9 and stream.prefix == 0
    """)


def test_evaluator_event_capture_and_stream_store_need_no_policy_model():
    run_isolated("""
        import importlib.abc
        import sys
        from types import SimpleNamespace

        sys.path[:0] = [sys.argv[1], sys.argv[1] + '/python']

        class NoPolicyModel(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] == 'torch':
                    raise RuntimeError(f'event plumbing imported {fullname}')
                if fullname in {'train.buffer', 'train.logs', 'train.history_model',
                                'eval.history_policy'}:
                    raise RuntimeError(f'event plumbing imported {fullname}')

        sys.meta_path.insert(0, NoPolicyModel())
        import gd
        import numpy as np
        from eval.history_events import HistoryStreamStore, apply_and_observe, explicit_passes

        engine, state = gd.Engine(), gd.MatchState()
        engine.new_match(state, 7)
        events = []
        listener = SimpleNamespace(observe=events.append)
        with explicit_passes(engine, [listener]):
            assert not engine.auto_pass
            event = apply_and_observe(engine, state, engine.legal_actions(state)[0], [listener])
        assert engine.auto_pass and events == [event]
        store = HistoryStreamStore(2)
        store.sync(np.array([0, 1]), np.array([4, 5]))
        envelope = SimpleNamespace(env_id=1, match_id=5, seat=event.seat, phase=event.phase,
                                   round_index=event.round_index, cards_left=event.cards_left,
                                   encoded_action=event.encoded_action)
        assert store.ingest([envelope]) == 1 and store.prefix(1) == 1
        old = store.streams[1].tokens
        streams, rows = store.select_streams(np.array([1, 0, 1]))
        assert rows.tolist() == [1, 0, 1] and len(streams) == 2
        store.sync(np.array([1]), np.array([6]))
        assert store.prefix(1) == 0 and store.streams[1].match_id == 6
        assert len(old) == 1 and store.ingested == 1
    """)


def test_public_history_compatibility_exports_keep_the_same_objects():
    from eval import history_events, history_policy
    from train import history_model, logs, public_history

    assert history_model.PublicStream is public_history.PublicStream
    assert history_model.public_token is logs.public_token is public_history.public_token
    assert history_model.TOKEN_SCHEMA_VERSION == public_history.TOKEN_SCHEMA_VERSION
    assert history_model.TOKEN_DIM == logs.TOKEN_DIM == public_history.TOKEN_DIM
    for name in ("PublicEvent", "HistoryStreamStore", "needs_history", "history_listeners",
                 "apply_and_observe", "resolve_forced_passes", "explicit_passes"):
        assert getattr(history_policy, name) is getattr(history_events, name)


def test_public_token_and_stream_never_read_private_or_forced_fields():
    encoded = np.zeros(154, np.uint8)
    encoded[7] = 1
    encoded[146:154] = 1

    class PublicOnlyEvent:
        seat, cards_left, round_index, phase = 1, 26, 2, 1
        encoded_action = encoded

        def __getattr__(self, name):
            raise AssertionError(f'public history read nonpublic field {name}')

    event = PublicOnlyEvent()
    token = public_token(event)
    stream = PublicStream()
    stream.append(event)
    np.testing.assert_array_equal(stream.tokens[0], token)
    assert token[:4].tolist() == [0, 1, 0, 0]
    assert token[4 + 7] == 1 and token[158 + 26] == 1
    assert not token[150:158].any() and encoded[146:154].all()
    encoded[:] = 0
    token[:] = 0
    assert stream.tokens[0, 4 + 7] == 1
    assert stream.rounds.tolist() == [2] and stream.phases.tolist() == [1]


def test_stream_views_keep_their_contents_across_append_growth_and_reset():
    stream = PublicStream(3)
    snapshots = []

    def retain_snapshot():
        views = stream.arrays()
        snapshots.append((views, tuple(array.copy() for array in views)))

    def check_snapshots():
        for views, copies in snapshots:
            for view, original in zip(views, copies):
                np.testing.assert_array_equal(view, original)
                assert not view.flags.writeable
                if view.size:
                    with pytest.raises(ValueError, match='read-only'):
                        view.flat[0] = 0

    retain_snapshot()   # empty views also stay empty
    for i in range(65):
        encoded = np.zeros(154, np.uint8)
        encoded[i % 146] = 1
        stream.append(SimpleNamespace(seat=i % 4, encoded_action=encoded, cards_left=27,
                                      round_index=i // 8, phase=3))
        if i in (0, 2, 63):
            retain_snapshot()
        check_snapshots()
    assert stream.prefix == 65   # crossed the initial 64-event capacity
    generation = stream.generation
    stream.reset(4)
    stream.append(SimpleNamespace(seat=0, encoded_action=np.zeros(154, np.uint8), cards_left=26,
                                  round_index=0, phase=3))
    assert stream.match_id == 4 and stream.generation == generation + 1
    check_snapshots()
