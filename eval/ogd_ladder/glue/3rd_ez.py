"""3rd-ez: root ``state.State()`` + root ``action.Action()``; ``Action.parse(msg)``.

``action.py`` is the only action module and its ``parse(msg)`` takes just the
message: it reads the seat from ``data1.txt``, which ``State.notify_begin``
writes, so ``State.parse`` must see every message (the beginning one first).
Protocol fix: ``Myfunc1014.getindex``/``lasthand.getindex`` are swapped for an
order-insensitive card-list match (see ``_compat.unordered_getindex``).
"""
import lasthand
import Myfunc1014
from action import Action
from state import State

from eval.ogd_ladder.glue import _compat

Myfunc1014.Myfunc1014.getindex = _compat.unordered_getindex
lasthand.lasthand.getindex = _compat.unordered_getindex


class Glue:
    def __init__(self) -> None:
        self.state = State()
        self.action = Action()

    def handle(self, msg: dict):
        msg = _compat.fix_beginning(msg)
        try:
            self.state.parse(msg)
        except KeyError:
            pass                      # stages the bot's State does not know
        if "actionList" in msg:
            return self.action.parse(msg)
        return None
