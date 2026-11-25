"""2nd-no-ai: ``state.State()`` + ``action.Action()``. This bot decides inside
``State.parse``: ``act_play`` (FreePlay/RestrictedPlay) and ``act_back``
(actBack) leave the chosen move in ``state.retValue``, and ``Action`` maps it
back to an index with ``GetIndexFromPlay(msg, retValue)`` /
``GetIndexFromBack(msg, retValue)``. ``act_tribute`` computes nothing, so
tribute goes to ``Action.parse(msg)`` (the template's random pick over the
tribute options)."""
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
        if "actionList" not in msg:
            return None
        stage = msg.get("stage")
        if stage == "play":
            return self.action.GetIndexFromPlay(msg, self.state.retValue)
        if stage == "back":
            return self.action.GetIndexFromBack(msg, self.state.retValue)
        return self.action.parse(msg)
