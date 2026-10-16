"""Play a full Guandan match in the browser against the latest Transformer.

One human sits at seat 0; the partner (seat 2) and both opponents (seats 1
and 3) are the same history checkpoint, each deciding from its own private
observation plus the shared public action stream (the evaluation setup of
``eval.history_policy``). Tribute and back-tribute of the model seats use the
engine heuristic, as in training.

The human's legal list comes from a second engine with suit de-duplication
off, so any concrete card choice the player selects can be matched exactly;
the model seats keep the canonical list they were trained on.

Usage:
    PYTHONPATH=python:oracle:. .work/external/danlm-venv/bin/python -m play.server \
        --checkpoint .work/velocity-milestones/m7-u73780.pt --port 18688
"""
from __future__ import annotations

import argparse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
import secrets
import threading
import time
from typing import Any
from urllib.parse import urlparse

import gd
import numpy as np

from eval.history_policy import HistoryPolicy, apply_and_observe, load_history_policy

HUMAN = 0
PLAY = int(gd.Phase.Play)
TRIBUTE = int(gd.Phase.Tribute)
BACK = int(gd.Phase.BackTribute)
ROUND_END = int(gd.Phase.RoundEnd)
RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
TYPE_ZH = {
    "Single": "单张", "Pair": "对子", "Triple": "三张", "FullHouse": "三带二",
    "Straight": "顺子", "Tube": "三连对", "Plate": "钢板", "Bomb": "炸弹",
    "StraightFlush": "同花顺", "JokerBomb": "天王炸",
}
SEQUENCES = ("Straight", "Tube", "Plate", "StraightFlush")
WEB = Path(__file__).resolve().parent / "web"
GAME_TTL = 6 * 3600          # idle games are dropped after six hours
MAX_GAMES = 64


def card_power(card: int, level: int) -> int:
    rank = 13 if card == 52 else 14 if card == 53 else card // 4
    return gd.power(rank, level)


def play_order(action: gd.Action, level: int) -> list[int]:
    """Display order of a play: sequence order for sequences, else power order."""
    cards = list(action.cards)
    if action.type not in SEQUENCES:
        return sorted(cards, key=lambda c: (card_power(c, level), c))
    ace_low = int(action.key) == 0
    return sorted(cards, key=lambda c: (-1 if ace_low and c // 4 == 12 else c // 4, c))


def describe(action: gd.Action) -> str:
    name = TYPE_ZH[action.type]
    if action.type == "Bomb":
        name = f"{int(action.bomb_size)}炸"
    return name


def reading_label(action: gd.Action, level: int) -> str:
    """How a wild-card reading differs from another reading of the same cards."""
    cards = play_order(action, level)
    if action.type in SEQUENCES:
        # Window w starts at rank index w - 1; window 0 is the ace-low one.
        low = int(action.key) - 1
        span = {"Straight": 5, "StraightFlush": 5, "Tube": 3, "Plate": 2}[action.type]
        first = "A" if low < 0 else RANKS[low]
        return f"{describe(action)} {first}-{RANKS[low + span - 1]}"
    return f"{describe(action)}（{len(cards)}张）"


class Game:
    """One match: engine state, model stream and the event log the browser replays."""

    def __init__(self, server: "PlayServer", seed: int) -> None:
        self.server = server
        self.id = secrets.token_urlsafe(9)
        self.seed = seed
        self.rng = random.Random(seed)
        self.engine = server.new_engine(canonical=True)
        self.full = server.new_engine(canonical=False)
        self.state = gd.MatchState()
        self.engine.new_match(self.state, seed)
        self.model = HistoryPolicy(server.actor, name="model", device="cpu")
        self.model.start_match(0)
        self.events: list[dict] = []
        self.round_no = 1
        self.finished: list[int] = []
        self.trick: dict[int, dict] = {}
        self.last_result: dict | None = None
        self.touched = time.time()
        self.events.append({"kind": "round_start", "round": self.round_no, **self.level_info()})
        self.advance()

    # -- public view ------------------------------------------------------------

    def level_info(self) -> dict:
        st = self.state
        return {"level": RANKS[st.level], "levels": [RANKS[x] for x in st.levels],
                "owner": int(st.owner)}

    def view(self, since: int = 0) -> dict:
        st = self.state
        level = st.level
        phase = int(st.phase)
        pending = None
        if self.waiting_for_human():
            if phase == PLAY:
                acts = self.full.legal_actions(st)
                pending = {"kind": "play", "can_pass": any(a.is_pass for a in acts),
                           "lead": bool(st.top_is_open)}
            else:
                acts = self.full.legal_actions(st)
                pending = {"kind": "tribute" if phase == TRIBUTE else "back",
                           "choices": sorted({int(a.cards[0]) for a in acts})}
        return {
            "id": self.id,
            "seed": self.seed,
            "round": self.round_no,
            "phase": phase,
            "to_move": int(st.to_move) if phase in (PLAY, TRIBUTE, BACK) else -1,
            "hand": sorted(st.hand(HUMAN), key=lambda c: (-card_power(c, level), c)),
            "counts": [len(st.hand(s)) for s in range(4)],
            "finished": list(self.finished),
            "trick": {str(s): v for s, v in self.trick.items()},
            "pending": pending,
            "result": self.last_result,
            "match_over": int(st.winner) >= 0,
            "winner": int(st.winner),
            "events": self.events[since:],
            "event_count": len(self.events),
            **self.level_info(),
        }

    # -- flow -------------------------------------------------------------------

    def waiting_for_human(self) -> bool:
        st = self.state
        return int(st.phase) in (PLAY, TRIBUTE, BACK) and int(st.to_move) == HUMAN

    def apply(self, action: gd.Action, forced: bool = False) -> None:
        st = self.state
        phase = int(st.phase)
        seat = int(st.to_move)
        level = st.level
        if phase == PLAY:
            new_trick = bool(st.top_is_open)
            if new_trick:
                self.trick = {}
            apply_and_observe(self.engine, st, action, [self.model], forced=forced)
            cards = play_order(action, level) if not action.is_pass else []
            event = {"kind": "play", "seat": seat, "pass": bool(action.is_pass),
                     "forced": forced, "cards": cards, "new_trick": new_trick,
                     "type": None if action.is_pass else describe(action),
                     "left": len(st.hand(seat))}
            self.trick[seat] = {"pass": bool(action.is_pass), "cards": cards,
                                "type": event["type"]}
            if not action.is_pass and len(st.hand(seat)) == 0:
                self.finished.append(seat)
                event["out"] = len(self.finished)
            self.events.append(event)
        else:
            before = [Counter(st.hand(s)) for s in range(4)]
            apply_and_observe(self.engine, st, action, [self.model], forced=forced)
            after = [Counter(st.hand(s)) for s in range(4)]
            kind = "tribute" if phase == TRIBUTE else "back"
            gained = [after[s] - before[s] for s in range(4)]
            for src in range(4):
                for card in (before[src] - after[src]).elements():
                    # Equal double-tribute cards: each loser pays its upstream
                    # neighbour (RULES.md 9.2), so try that seat first.
                    for dst in [(src + 3) % 4] + [s for s in range(4) if s != (src + 3) % 4]:
                        if gained[dst][card] > 0:
                            gained[dst][card] -= 1
                            self.events.append({"kind": kind, "from": src, "to": dst,
                                                "card": int(card)})
                            break

    def advance(self) -> None:
        """Run model seats and forced passes until the human must act or the round ends."""
        st = self.state
        guard = 0
        while int(st.phase) != ROUND_END:
            guard += 1
            if guard > 5000:
                raise RuntimeError("round did not end")
            phase = int(st.phase)
            acts = self.engine.legal_actions(st)
            if phase == PLAY and len(acts) == 1 and acts[0].is_pass:
                self.apply(acts[0], forced=True)
                continue
            if int(st.to_move) == HUMAN:
                if phase != PLAY and len(acts) == 1:
                    self.apply(acts[0])      # a single legal tribute card needs no click
                    continue
                return
            with self.server.lock:
                index = self.model.select(self.engine, st, acts, self.rng)
            self.apply(acts[index])
        self.finish_round()

    def finish_round(self) -> None:
        st = self.state
        order_before = list(self.finished)
        result = self.engine.end_round(st)
        order = [int(s) for s in result.order]
        self.last_result = {
            "round": self.round_no,
            "order": order,
            "winning_team": int(result.winning_team),
            "gain": int(result.gain),
            "levels": [RANKS[x] for x in result.levels],
            "match_winner": int(result.match_winner),
            "hands": [sorted(st.hand(s), key=lambda c: (-card_power(c, st.level), c))
                      for s in range(4)],
            "went_out": order_before,
        }
        self.events.append({"kind": "round_end", **self.last_result})

    def next_round(self) -> None:
        st = self.state
        if int(st.phase) != ROUND_END or int(st.winner) >= 0:
            raise ValueError("the round is not over")
        self.engine.begin_round(st)
        self.round_no += 1
        self.finished = []
        self.trick = {}
        self.last_result = None
        self.events.append({"kind": "round_start", "round": self.round_no, **self.level_info()})
        if int(st.phase) == PLAY and self.round_no > 1:
            self.events.append({"kind": "anti_tribute"})
        self.advance()

    # -- human input ------------------------------------------------------------

    def human_candidates(self, cards: list[int]) -> list[gd.Action]:
        want = Counter(int(c) for c in cards)
        if want - Counter(self.state.hand(HUMAN)):
            raise ValueError("你没有这些牌")
        out = []
        seen = set()
        for action in self.full.legal_actions(self.state):
            if action.is_pass or Counter(action.cards) != want:
                continue
            key = (action.type, int(action.key), int(action.bomb_size))
            if key not in seen:
                seen.add(key)
                out.append(action)
        return out

    def human_play(self, cards: list[int], reading: int | None) -> dict:
        if not self.waiting_for_human():
            raise ValueError("还没轮到你")
        st = self.state
        phase = int(st.phase)
        if phase == PLAY and not cards:
            passes = [a for a in self.full.legal_actions(st) if a.is_pass]
            if not passes:
                raise ValueError("你必须出牌")
            self.apply(passes[0])
            self.advance()
            return {"ok": True}
        if phase != PLAY:
            if len(cards) != 1:
                raise ValueError("请选择一张牌")
            match = [a for a in self.full.legal_actions(st) if int(a.cards[0]) == int(cards[0])]
            if not match:
                raise ValueError("这张牌不能用来进贡/还贡" if phase == TRIBUTE else "还贡只能还 10 以下的非级牌")
            self.apply(match[0])
            self.advance()
            return {"ok": True}
        options = self.human_candidates(cards)
        if not options:
            raise ValueError("这手牌不能这样出")
        if len(options) > 1 and reading is None:
            return {"ok": False, "choose": [{"index": i, "label": reading_label(a, st.level)}
                                            for i, a in enumerate(options)]}
        action = options[reading or 0]
        self.apply(action)
        self.advance()
        return {"ok": True}

    def hints(self) -> list[dict]:
        """Every canonical legal play for the human seat with the model's probability.

        This is the policy the model itself would sample from in this seat,
        over the canonical list it was trained on; sorted most likely first.
        """
        if not self.waiting_for_human() or int(self.state.phase) != PLAY:
            return []
        st = self.state
        acts = self.engine.legal_actions(st)
        with self.server.lock:
            import torch

            inputs = self.model.decision_inputs(self.engine, st, acts)
            with torch.inference_mode():
                logp = self.model.actor.candidate_log_probs(inputs).double().cpu().numpy()
        prob = np.exp(logp - logp.max())
        prob /= prob.sum()
        order = np.argsort(-prob, kind="stable")
        return [{"cards": play_order(acts[i], st.level) if not acts[i].is_pass else [],
                 "pass": bool(acts[i].is_pass),
                 "type": "不出" if acts[i].is_pass else describe(acts[i]),
                 "prob": float(prob[i])} for i in order]


class PlayServer:
    def __init__(self, checkpoint: str) -> None:
        import torch

        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        policy = load_history_policy(checkpoint, device="cpu")
        self.actor = policy.actor
        self.model_name = Path(checkpoint).stem
        self.lock = threading.Lock()        # model inference
        self.games_lock = threading.Lock()  # game table and per-game moves
        self.games: dict[str, Game] = {}

    @staticmethod
    def new_engine(canonical: bool) -> gd.Engine:
        if canonical:
            engine = gd.Engine(gd.RuleConfig.house())
        else:
            cfg = gd.ActionConfig()
            cfg.suit_dedup = False
            engine = gd.Engine(gd.RuleConfig.house(), cfg)
        engine.auto_pass = False
        return engine

    def new_game(self) -> Game:
        with self.games_lock:
            now = time.time()
            for gid in [g for g, v in self.games.items() if now - v.touched > GAME_TTL]:
                del self.games[gid]
            if len(self.games) >= MAX_GAMES:
                oldest = min(self.games.values(), key=lambda g: g.touched)
                del self.games[oldest.id]
        game = Game(self, random.SystemRandom().randrange(1, 2**31))
        with self.games_lock:
            self.games[game.id] = game
        return game

    def game(self, gid: str) -> Game:
        game = self.games.get(gid)
        if game is None:
            raise KeyError(gid)
        game.touched = time.time()
        return game


def make_handler(server: PlayServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            if not self.path.startswith("/api/state"):
                super().log_message(fmt, *args)

        def send_json(self, body: dict, status: int = 200) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                path = "/index.html"
            target = (WEB / path.lstrip("/")).resolve()
            if WEB not in target.parents or not target.is_file():
                self.send_error(404)
                return
            data = target.read_bytes()
            kind = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                    ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}.get(
                        target.suffix, "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if path == "/api/new":
                    game = server.new_game()
                    self.send_json({**game.view(), "model": server.model_name})
                    return
                game = server.game(str(body.get("id", "")))
                since = int(body.get("since", 0))
                with server.games_lock:
                    if path == "/api/state":
                        out: dict = {}
                    elif path == "/api/play":
                        cards = [int(c) for c in body.get("cards", [])]
                        reading = body.get("reading")
                        out = game.human_play(cards, None if reading is None else int(reading))
                    elif path == "/api/next":
                        game.next_round()
                        out = {"ok": True}
                    elif path == "/api/hint":
                        self.send_json({"hints": game.hints()})
                        return
                    else:
                        self.send_error(404)
                        return
                    self.send_json({**out, **game.view(since), "model": server.model_name})
            except KeyError:
                self.send_json({"error": "对局不存在（服务器可能重启过），请开新局", "lost": True}, 404)
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 400)

    return Handler


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=".work/velocity-milestones/m7-u73780.pt")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18688)
    args = parser.parse_args(argv)
    server = PlayServer(args.checkpoint)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(server))
    print(f"model {server.model_name}; open http://localhost:{args.port}/", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
