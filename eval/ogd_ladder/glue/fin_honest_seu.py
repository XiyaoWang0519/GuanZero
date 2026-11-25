"""fin-honest-seu (东大老实人): ``state.State(name)`` + ``action.Action(name)``,
entry ``Action.rule_parse(msg, mypos, playcards, remaincards, history,
pass_num, my_pass_num, back_pos)``; the arguments are the State attributes
``_myPos``, ``play_cards``, ``remain_cards``, ``history``, ``pass_num``,
``my_pass_num`` and ``back_pos``. ``back_pos`` is only set by the *back
notify*, which arrives after the bot's own back act, so at a back decision it
holds the previous round's value (``None`` in the first one, where the bot's
own ``except`` picks index 1). That is the bot's behaviour, kept as is. The
``passive`` network is loaded by ``Action`` but never called (``nw_model`` has
no caller), so play is purely rule based."""
from action import Action
from state import State


class Glue:
    def __init__(self) -> None:
        self.state = State("client")
        self.action = Action("client")

    def handle(self, msg: dict):
        self.state.parse(msg)
        if "actionList" not in msg:
            return None
        s = self.state
        return self.action.rule_parse(msg, s._myPos, s.play_cards, s.remain_cards, s.history,
                                      s.pass_num, s.my_pass_num, s.back_pos)
