"""3rd-hulalala (``new_version``): ``state.State()`` + ``action.Action()``,
entry ``Action.rule_parse(msg, mypos, remaincards, history,
remain_cards_classbynum, pass_num, my_pass_num, tribute_result)`` -- the
1st-lalala interface (same State, no ``name`` argument); every argument after
``msg`` is the State attribute of the same name (``_myPos``, ``remain_cards``,
``history``, ``remain_cards_classbynum``, ``pass_num``, ``my_pass_num``,
``tribute_result``)."""
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
        s = self.state
        return self.action.rule_parse(msg, s._myPos, s.remain_cards, s.history,
                                      s.remain_cards_classbynum, s.pass_num,
                                      s.my_pass_num, s.tribute_result)
