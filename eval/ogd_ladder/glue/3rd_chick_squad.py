"""3rd-chick-squad: ``state.State()`` + ``action.Action()``; act messages go to
``Action.parse(msg, mate_pos)`` ("增加了一个新参数mate_pos，表示队友的位置"), so
``mate_pos = (myPos + 2) % 4`` from the ``beginning`` notify; ``solve`` derives
the opponents as ``mate_pos + 1`` and ``mate_pos + 3``.

``action.py`` does ``from test import tsolve``. The team's ``test.py`` is not
in the release and ``tsolve`` is never called; without it the import resolves
to the standard library's ``test`` package and fails. A stub module that only
provides ``tsolve`` (raising if ever called) stands in for it while
``action`` is imported."""
import sys
import types

from state import State


def _import_action():
    saved = sys.modules.get("test")
    stub = types.ModuleType("test")

    def tsolve(*_args, **_kwargs):
        raise RuntimeError("tsolve came from the team's missing test.py")

    stub.tsolve = tsolve
    sys.modules["test"] = stub
    try:
        from action import Action
    finally:
        if saved is None:
            sys.modules.pop("test", None)
        else:
            sys.modules["test"] = saved
    return Action


Action = _import_action()


class Glue:
    def __init__(self) -> None:
        self.state = State()
        self.action = Action()
        self.mate_pos = None

    def handle(self, msg: dict):
        if msg.get("stage") == "beginning":
            self.mate_pos = (msg["myPos"] + 2) % 4
        try:
            self.state.parse(msg)
        except KeyError:
            pass                      # stages the bot's State does not know
        if "actionList" in msg:
            return self.action.parse(msg, self.mate_pos)
        return None
