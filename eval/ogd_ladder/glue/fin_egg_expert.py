"""fin-egg-expert (吃蛋能手): ``state.State(logFile)`` + ``my_action.MyAction(logFile)``;
``MyAction.parse(msg, state)``.

``action.py`` is DanLM's one-line wrapper re-exporting ``MyAction``. Both
classes take the client's open log file (only printed to; a ``str`` would be
turned into an unbounded StringIO, so we pass ``/dev/null``). ``MyAction``
reads the hand grouping (``myOrderedCards``) and teammate/opponent tracking
the ``State`` builds from every message, so ``state.parse`` runs first.
"""
import os

from my_action import MyAction
from state import State


class Glue:
    def __init__(self) -> None:
        log = open(os.devnull, "w")
        self.state = State(log)
        self.action = MyAction(log)

    def handle(self, msg: dict):
        try:
            self.state.parse(msg)
        except KeyError:
            pass                      # stages the bot's State does not know
        if "actionList" in msg:
            return self.action.parse(msg, self.state)
        return None
