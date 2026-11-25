"""fin-caiji (caiji): ``state.State()`` + ``action.Action()`` (which is
``strong_rule.Action``), entry ``Action.parse(msg)``: a one-ply greedy scorer
over ``actionList`` that reads only the act message itself (``handCards``,
``curRank``, ``indexRange``). The State keeps no history the Action uses; it
is fed every message as the 2020 client template did."""
from action import Action
from state import State


class Glue:
    def __init__(self) -> None:
        self.state = State()
        self.action = Action()

    def handle(self, msg: dict):
        self.state.parse(msg)
        if "actionList" not in msg:
            return None
        return self.action.parse(msg)
