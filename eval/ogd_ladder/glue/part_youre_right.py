"""part-youre-right (你说的都对): root ``state.State()`` + ``AIAction_back.AIAction()``;
``AIAction.parse(msg, restCards, episode_rounds, agent_pos)``.

``action.py`` is DanLM's wrapper re-exporting ``AIAction``, the only strategy
module. ``agent_pos`` is our seat (``myPos`` of the beginning message; the
strategy uses it to tell partner from opponents). ``restCards`` ("agent剩余手牌")
is the hand in the act message and ``episode_rounds`` ("回合数") counts this
seat's decisions in the current episode; the strategy only prints those two.
"""
from AIAction_back import AIAction
from state import State


class Glue:
    def __init__(self) -> None:
        self.state = State()
        self.action = AIAction()
        self.pos = None
        self.rounds = 0

    def handle(self, msg: dict):
        if msg.get("stage") == "beginning":
            self.pos = msg["myPos"]
            self.rounds = 0
        try:
            self.state.parse(msg)
        except KeyError:
            pass                      # stages the bot's State does not know
        if "actionList" in msg:
            self.rounds += 1
            return self.action.parse(msg, msg["handCards"], self.rounds, self.pos)
        return None
