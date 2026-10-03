"""part-369: root ``state.State()`` + root ``action.Action()``; ``Action.parse(msg)``.

``action.py`` is the only action module and its ``parse(msg)`` takes just the
message: it reads the seat from ``data1.txt``, which ``State.notify_begin``
writes, so ``State.parse`` must see every message (the beginning one first).
"""
from action import Action
from state import State


class Glue:
    def __init__(self) -> None:
        self.state = State()
        self.action = Action()

    def handle(self, msg: dict):
        try:
            self.state.parse(msg)
        except KeyError:
            pass                      # stages the bot's State does not know
        if "actionList" in msg:
            return self.action.parse(msg)
        return None
