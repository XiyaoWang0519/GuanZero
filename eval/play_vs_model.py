"""Play Guandan in the browser against a history checkpoint.

You take seat 0 (bottom); the checkpoint plays the other three seats, so
your partner (seat 2, top) is the model too. Play is greedy and without
search, exactly ``HistoryPolicy`` as the evaluators run it, tribute
included (the engine's tribute heuristic for the model seats). The page
shows only what a player at the table would know: your hand, the public
plays, card counts and a tracker of the cards not yet seen. On your turn it
also shows the model's top moves for your hand. The model
sees one match-long public stream, fed every applied action of every seat,
your plays and forced passes included.

Your own moves come from a second engine in full action mode, so any
concrete card choice is accepted, not only the canonical representative the
model chooses from. The model never reads your hand.

Usage:
    PYTHONPATH=python:. .venv/bin/python -m eval.play_vs_model --checkpoint u9989.pt
    ./scripts/play.sh [checkpoint]      # builds gd_core if needed, opens the browser
"""
from __future__ import annotations

import argparse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import random
import threading
from typing import Any, Sequence
import webbrowser

import gd

from .history_policy import HistoryPolicy, apply_and_observe, load_history_policy
from .record_games import RANKS, TYPE_ZH, card_json, play_order, sorted_cards, type_zh

HUMAN = 0
PLAY = int(gd.Phase.Play)
TRIBUTE_PHASES = (int(gd.Phase.Tribute), int(gd.Phase.BackTribute))
SEAT_ZH = ("你", "下家", "对家", "上家")
FINISH_ZH = ("头游", "二游", "三游", "末游")
UI = Path(__file__).with_name("play_ui") / "index.html"


SUIT_SYM = {"S": "♠", "H": "♥", "C": "♣", "D": "♦", "": ""}


def card_text(card: dict) -> str:
    return {"BJ": "小王", "RJ": "大王"}.get(card["r"], card["r"] + SUIT_SYM[card["s"]])


def card_out(card: int, level: int) -> dict:
    out = card_json(card, level)
    out["id"] = card
    return out


def action_cards(action: gd.Action, level: int) -> list[dict]:
    return [card_out(c, level) for c in play_order(action, level)]


def action_label(action: gd.Action) -> str:
    if action.is_pass:
        return "不出"
    if action.type in TYPE_ZH:
        return type_zh(action)
    return {"Tribute": "进贡", "BackTribute": "还贡"}.get(action.type, action.type)


class Game:
    """One match: the human at seat 0, the policy at seats 1 to 3."""

    def __init__(self, policy: HistoryPolicy, seed: int, top_k: int = 3) -> None:
        rules = gd.RuleConfig.house()
        self.policy = policy
        self.engine = gd.Engine(rules)
        self.full = gd.Engine(rules, gd.ActionConfig.full())
        self.engine.auto_pass = False
        self.full.auto_pass = False
        self.seed = seed
        self.top_k = top_k
        self.rng = random.Random(seed)
        self.state = gd.MatchState()
        self.engine.new_match(self.state, seed)
        policy.start_match(seed)
        self.rounds: list[dict] = []
        self.prev_order: list[int] | None = None
        self.start_round()

    # -- round bookkeeping ----------------------------------------------------

    def start_round(self) -> None:
        self.table: list[dict | None] = [None] * 4
        self.log: list[dict] = []
        self.result: dict | None = None
        self.saw_tribute = False
        self.anti_logged = False
        self.played: Counter = Counter()
        self.advice_cache: tuple[int, dict] | None = None
        self.level = int(self.state.level)

    def next_round(self) -> list[dict]:
        if self.result is None or self.result["match_over"]:
            raise ValueError("the round is not over, or the match is")
        self.engine.begin_round(self.state)
        self.start_round()
        return self.advance()

    # -- applying actions -----------------------------------------------------

    def record(self, seat: int, action: gd.Action, forced: bool = False) -> dict:
        state = self.state
        phase = int(state.phase)
        new_trick = phase == PLAY and bool(state.top_is_open)
        before = [Counter(state.hand(s)) for s in range(4)]
        apply_and_observe(self.engine, state, action, [self.policy], forced=forced)
        event: dict[str, Any] = {"seat": seat, "forced": forced}
        if phase in TRIBUTE_PHASES:
            self.saw_tribute = True
            kind = "进贡" if phase == int(gd.Phase.Tribute) else "还贡"
            moves = []
            after = [Counter(state.hand(s)) for s in range(4)]
            gained = [after[s] - before[s] for s in range(4)]
            for src in range(4):
                for card, n in (before[src] - after[src]).items():
                    for _ in range(n):
                        dst = next(s for s in range(4) if gained[s][card] > 0)
                        gained[dst][card] -= 1
                        moves.append({"from": src, "to": dst, "card": card_out(card, self.level)})
            event.update(kind="tribute", label=kind, moves=moves)
            if seat == HUMAN or moves:
                text = "、".join(f"{SEAT_ZH[m['from']]}→{SEAT_ZH[m['to']]} "
                                f"{card_text(m['card'])}" for m in moves)
                self.log.append({"seat": seat, "text": f"{kind}：{text or '（已选择）'}"})
        else:
            if new_trick:
                self.table = [None] * 4
            shown = {"pass": bool(action.is_pass), "label": action_label(action),
                     "cards": action_cards(action, self.level)}
            self.played.update(action.cards)
            self.table[seat] = shown
            event.update(kind="play", new_trick=new_trick, **shown)
            if not forced:
                text = shown["label"]
                if not action.is_pass:
                    text += " " + " ".join(card_text(c) for c in shown["cards"])
                self.log.append({"seat": seat, "text": text})
        event["table"] = list(self.table)
        event["left"] = [len(state.hand(s)) for s in range(4)]
        event["order"] = list(state.order)
        return event

    def model_choice(self, actions: Sequence[gd.Action]) -> tuple[int, list[tuple[int, float]]]:
        """Greedy choice of the policy and its top candidates as (index, prob)."""
        import torch

        policy, state = self.policy, self.state
        policy.verify_stream(state)
        if policy.heuristic_tribute and int(state.phase) != PLAY:
            return int(self.engine.greedy(state)), []
        inputs = policy.decision_inputs(self.engine, state, actions)
        with torch.inference_mode():
            log_probs = policy.actor.candidate_log_probs(inputs).float().cpu()
        probs = log_probs.exp().tolist()
        ranked = sorted(range(len(probs)), key=lambda i: -probs[i])
        return ranked[0], [(i, probs[i]) for i in ranked[:self.top_k]]

    def advance(self) -> list[dict]:
        """Play forced passes and model seats until the human must act or the round ends."""
        events: list[dict] = []
        state = self.state
        if (not self.anti_logged and int(state.phase) == PLAY and self.prev_order is not None
                and not self.saw_tribute):
            banker, follower, third, dweller = self.prev_order
            payers = [third, dweller] if follower == (banker + 2) % 4 else [dweller]
            self.log.append({"seat": None, "text": "抗贡：" + "、".join(SEAT_ZH[s] for s in payers)})
            events.append({"kind": "anti", "payers": payers})
        self.anti_logged = True
        while int(state.phase) != int(gd.Phase.RoundEnd):
            actions = self.engine.legal_actions(state)
            seat = int(state.to_move)
            if int(state.phase) == PLAY and len(actions) == 1 and actions[0].is_pass:
                events.append(self.record(seat, actions[0], forced=True))
                continue
            if seat == HUMAN:
                return events
            choice, top = self.model_choice(actions)
            events.append(self.record(seat, actions[choice]))
        events.append(self.finish_round())
        return events

    def finish_round(self) -> dict:
        state = self.state
        level_before = list(state.levels)
        result = self.engine.end_round(state)
        order = list(result.order)
        self.prev_order = order
        self.result = {
            "order": order,
            "winning_team": int(result.winning_team),
            "gain": int(result.gain),
            "levels_before": [RANKS[x] for x in level_before],
            "levels_after": [RANKS[x] for x in result.levels],
            "match_over": int(result.match_winner) >= 0,
            "match_winner": int(result.match_winner) if int(result.match_winner) >= 0 else None,
        }
        self.rounds.append(dict(self.result))
        mine = int(result.winning_team) == HUMAN % 2
        self.log.append({"seat": None, "text": (
            f"本局结束：{'我方' if mine else '对方'}升 {int(result.gain)} 级，"
            + "，".join(f"{FINISH_ZH[i]} {SEAT_ZH[s]}" for i, s in enumerate(order)))})
        return {"kind": "round_end", "result": self.result}

    # -- the human seat -------------------------------------------------------

    def human_actions(self) -> list[gd.Action]:
        if int(self.state.phase) == int(gd.Phase.RoundEnd) or int(self.state.to_move) != HUMAN:
            raise ValueError("it is not your turn")
        return self.full.legal_actions(self.state)

    def play(self, cards: Sequence[int], option: int | None = None) -> dict:
        actions = self.human_actions()
        want = sorted(int(c) for c in cards)
        if Counter(want) - Counter(self.state.hand(HUMAN)):
            raise ValueError("你没有这些牌")
        matches = [a for a in actions if sorted(a.cards) == want]
        readings: dict[tuple, gd.Action] = {}
        for a in matches:
            readings.setdefault((a.type, str(a.key), int(a.bomb_size)), a)
        options = list(readings.values())
        if not options:
            return {"error": "这手牌不能出" if want else "现在不能不出"}
        if len(options) > 1 and option is None:
            return {"options": [{"index": i, "label": f"{action_label(a)}（{self.key_zh(a)}）"}
                                for i, a in enumerate(options)]}
        action = options[option or 0]
        events = [self.record(HUMAN, action)]
        events += self.advance()
        return {"events": events}

    def key_zh(self, action: gd.Action) -> str:
        cards = play_order(action, self.level)
        if action.type in ("Straight", "Tube", "Plate", "StraightFlush"):
            return f"{card_json(cards[0], self.level)['r']} 起"
        return f"{len(cards)} 张"

    def advice(self) -> dict:
        """The model's top moves for your hand at this decision (cached per position)."""
        if int(self.state.to_move) != HUMAN or self.result is not None:
            raise ValueError("it is not your turn")
        key = int(self.state.hash())
        if self.advice_cache and self.advice_cache[0] == key:
            return self.advice_cache[1]
        actions = self.engine.legal_actions(self.state)
        choice, top = self.model_choice(actions)
        if not top:
            top = [(choice, 1.0)]
        out = {"hints": [{"label": action_label(actions[i]), "p": round(p, 3),
                          "cards": action_cards(actions[i], self.level),
                          "ids": [int(c) for c in actions[i].cards]} for i, p in top]}
        self.advice_cache = (key, out)
        return out

    def unseen(self) -> list[dict]:
        """Per rank, the copies neither in your hand nor played this round."""
        mine = Counter(self.state.hand(HUMAN))
        rows = []
        for rank in range(13):
            ids = range(4 * rank, 4 * rank + 4)
            gone = sum(mine[c] + self.played[c] for c in ids)
            wild = 4 * rank + 1
            rows.append({"r": RANKS[rank], "left": 8 - gone, "level": rank == self.level,
                         "wild": 2 - mine[wild] - self.played[wild] if rank == self.level else None})
        for card, name in ((52, "小王"), (53, "大王")):
            rows.append({"r": name, "left": 2 - mine[card] - self.played[card],
                         "level": False, "wild": None})
        return rows

    # -- view -----------------------------------------------------------------

    def view(self) -> dict:
        state = self.state
        phase = int(state.phase)
        your_turn = (phase != int(gd.Phase.RoundEnd) and self.result is None
                     and int(state.to_move) == HUMAN)
        can_pass = False
        if your_turn and phase == PLAY:
            can_pass = any(a.is_pass for a in self.engine.legal_actions(state))
        team_levels = list(state.levels)
        return {
            "seed": self.seed,
            "round": len(self.rounds) + (0 if self.result else 1),
            "level": RANKS[self.level],
            "owner": int(state.owner) if int(state.owner) >= 0 else None,
            "team_levels": [RANKS[x] for x in team_levels],
            "phase": {PLAY: "play", int(gd.Phase.Tribute): "tribute",
                      int(gd.Phase.BackTribute): "back"}.get(phase, "end"),
            "to_move": int(state.to_move),
            "your_turn": your_turn,
            "can_pass": can_pass,
            "hand": [card_out(c, self.level) for c in sorted_cards(state.hand(HUMAN), self.level)],
            "left": [len(state.hand(s)) for s in range(4)],
            "order": list(state.order),
            "table": list(self.table),
            "log": self.log[-80:],
            "result": self.result,
            "unseen": self.unseen(),
            "advice": self.advice() if your_turn else None,
            "history": self.rounds,
            "model": self.policy.name,
        }


class Server:
    def __init__(self, policy: HistoryPolicy) -> None:
        self.policy = policy
        self.lock = threading.Lock()
        self.game: Game | None = None

    def handle(self, path: str, body: dict) -> dict:
        with self.lock:
            if path == "/api/new":
                seed = body.get("seed")
                seed = int(seed) if seed not in (None, "") else random.randrange(1, 2**31)
                self.game = Game(self.policy, seed)
                return {"events": self.game.advance(), "view": self.game.view()}
            game = self.game
            if game is None:
                return {"view": None}
            out: dict[str, Any] = {}
            if path == "/api/state":
                pass
            elif path == "/api/play":
                out = game.play(body.get("cards", []), body.get("option"))
            elif path == "/api/next":
                out = {"events": game.next_round()}
            else:
                raise KeyError(path)
            out["view"] = game.view()
            return out


def make_handler(server: Server):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:   # keep the terminal quiet
            pass

        def send(self, code: int, payload: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self.send(200, UI.read_bytes(), "text/html; charset=utf-8")
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                out = server.handle(self.path, body)
                code = 200
            except KeyError:
                out, code = {"error": "unknown endpoint"}, 404
            except ValueError as exc:
                out, code = {"error": str(exc)}, 400
            self.send(code, json.dumps(out, ensure_ascii=False, default=_json_default)
                      .encode("utf-8"), "application/json; charset=utf-8")

    return Handler


def _json_default(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(type(value).__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="history_ppo checkpoint (.pt)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2,
                        help="torch CPU threads; small by default so the machine stays responsive")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(max(1, args.threads))
    policy = load_history_policy(args.checkpoint, device=args.device)
    server = Server(policy)
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), make_handler(server))
    except OSError as exc:
        raise SystemExit(f"cannot listen on {args.host}:{args.port} ({exc}); "
                         "is another copy running? try --port") from None
    url = f"http://{args.host}:{args.port}/"
    print(f"model {policy.name}\nopen {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
