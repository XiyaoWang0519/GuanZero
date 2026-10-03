"""fin-go-go-go: ``state.State()`` + ``action.Action()``; act messages go to
``Action.parse(msg)`` (the template signature, unchanged by the team: all
decisions come from ``msg["actionList"]`` and ``msg["handCards"]`` via
``tools.random_aciton``/``parse_pass``). The message is passed as the bot
received it, so the bot's in-place edits of ``actionList`` stay its own."""
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
