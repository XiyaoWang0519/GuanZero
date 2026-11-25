"""2nd-egg-pancake: ``state.State()`` + ``action.Action()``; act messages go to
``Action.parse_AI(msg, myPos)``, the "AI-assisted" entry (``parse`` is the
template's uniform-random stub). ``check_message(msg, pos)`` compares
``greaterPos`` with ``pos`` and ``(pos + 2) % 4``, so ``pos`` is the seat
number from the ``beginning`` notify. Tribute is unhandled by the AI and
falls back to the bot's own ``randint``, as in the competition."""
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
            return self.action.parse_AI(msg, self.pos)
        return None
