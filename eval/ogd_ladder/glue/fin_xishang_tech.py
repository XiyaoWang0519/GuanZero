"""fin-xishang-tech: ``state.State()`` + ``action.Action()``; act messages go to
``Action.parse(msg, myPos)``. That ``action.py`` is DanLM's wrapper around the
decision code of the team's ``client_v5_1.py`` ("extract decision logic ...
into Action.parse(msg, myPos)"); ``Util2`` uses ``myPos`` to tell the
partner's lead (``(myPos + 2) % 4``) from an opponent's, so it is the seat
number from the ``beginning`` notify."""
from action import Action
from state import State


class Glue:
    def __init__(self) -> None:
        self.state = State()
        self.action = Action()
        self.pos = None

    def handle(self, msg: dict):
        if msg.get("stage") == "beginning":
            self.pos = msg["myPos"]
        try:
            self.state.parse(msg)
        except KeyError:
            pass                      # stages the bot's State does not know
        if "actionList" in msg:
            return self.action.parse(msg, self.pos)
        return None
