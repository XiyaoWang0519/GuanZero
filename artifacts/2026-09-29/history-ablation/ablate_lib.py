"""History-input ablation of a history_ppo checkpoint (analysis only).

Run with PYTHONPATH=<export b8c0c54>/python:<export>/oracle:<export>:<this dir>.
Nothing in the export is edited: the conditions are subclasses of the export's
HistoryPolicy that change only what reaches the policy's own PublicStream.

  full  : unchanged HistoryPolicy behaviour (whole match stream).
  round : the stream is cleared whenever a new round starts (first event of a
          new round index, or a decision in a new round before any event of
          it). Current-round tokens keep their true round index embedding but
          sit at positions 1.. after BOS.
  none  : observe() stores nothing; every decision sees BOS only.

ProbePolicy plays exactly like `full` and, at every play-phase decision it is
asked for, also evaluates the none / round / swap streams on the same state
without acting on them (teacher-forced comparison).
  swap  : earlier-round tokens replaced by the same number of earlier-round
          tokens from a DIFFERENT match (donor pool), current round unchanged.
"""
from __future__ import annotations

import math
import random
from typing import Sequence

import gd
import numpy as np
import torch

from eval.history_policy import (PLAY, HistoryPolicy, apply_and_observe, explicit_passes,
                                 history_listeners, resolve_forced_passes)
from eval.duplicate import RoundScore
from eval.policies import choose_action
from train.history_model import PublicStream

ROOT = "/Users/xiyaowang/Developer/Projects/GuanZero/"
CKPT = ROOT + ".work/longrun-batched-eval-2026-09-29/ckpt/u2623.pt"
B11 = ROOT + ".work/runpod-longrun-2026-09-25/payload/artifacts/b11-main.pt"


class AblatedPolicy(HistoryPolicy):
    def __init__(self, base: HistoryPolicy, mode: str) -> None:
        assert mode in ("full", "round", "none")
        super().__init__(base.actor, name=f"{base.name}/{mode}", device="cpu", sample=False,
                         heuristic_tribute=base.heuristic_tribute)
        self.mode = mode
        self.withheld = 0
        self.prefix_log: list[tuple[int, int]] = []   # (round_index, prefix) per play decision

    def observe(self, event) -> None:
        if not self.started:
            raise RuntimeError("observe() before start_match()")
        if self.mode == "none":
            self.withheld += 1
            return
        if (self.mode == "round" and self.stream.rounds
                and self.stream.rounds[-1] != int(event.round_index)):
            self.stream.reset(self.stream.match_id)
        self.stream.append(event)

    def select(self, engine, state, actions, rng) -> int:
        if (self.mode == "round" and self.stream.rounds
                and self.stream.rounds[-1] != int(state.round_index)):
            self.stream.reset(self.stream.match_id)
        if int(state.phase) == PLAY:
            self.prefix_log.append((int(state.round_index), self.stream.prefix))
        return super().select(engine, state, actions, rng)


def load_base():
    from eval.policies import load_policy
    return load_policy(CKPT, "cpu")


def load_b11():
    from eval.policies import load_policy
    return load_policy(B11, "cpu", margin=0.0)


# ---- matches ------------------------------------------------------------------

def play_match(a, b, a_team: int, seed: int, max_rounds: int = 200) -> dict:
    """eval.arena.play_matches for one match with explicit team; per-round records."""
    engine = gd.Engine()
    policies = (a, b) if a_team == 0 else (b, a)
    state = gd.MatchState()
    engine.new_match(state, seed)
    for listener in history_listeners(policies):
        listener.start_match()
    rounds = []
    match = a_team   # play_matches(index=a_team, seed=seed-a_team) reproduces this match
    while state.winner < 0:
        if len(rounds) >= max_rounds:
            raise RuntimeError("runaway match")
        from eval.duplicate import play_round
        score = play_round(engine, state, policies, seed + match * 1000003 + len(rounds))
        rounds.append({"net": score.net_gain(a_team), "first": score.winning_team == a_team,
                       "double": score.double_win(a_team),
                       "opp_double": score.double_win(1 - a_team),
                       "order": list(score.order), "decisions": score.decisions})
        if state.winner < 0:
            engine.begin_round(state)
    return {"seed": seed, "a_team": a_team, "winner": int(state.winner),
            "a_won": int(state.winner) == a_team, "rounds": rounds}


# ---- teacher-forced probe -----------------------------------------------------

def tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


class ProbePolicy(HistoryPolicy):
    """Plays as FULL; records FULL/none/round/swap candidate distributions."""

    def __init__(self, base: HistoryPolicy, donors: list[dict] | None = None,
                 describe=None) -> None:
        super().__init__(base.actor, name=f"{base.name}/probe", device="cpu", sample=False,
                         heuristic_tribute=base.heuristic_tribute)
        self.donors = donors or []
        self.records: list[dict] = []
        self.describe = describe
        self.event_desc: list[tuple[int, int, str]] = []   # (round, seat, text)
        self.match_tag = ""
        self.team = 0
        self.cur_level = 0

    def start_match(self, match_id: int = -1) -> None:
        super().start_match(match_id)
        self.event_desc = []

    def observe(self, event) -> None:
        super().observe(event)
        if self.describe is not None and event.action is not None:
            self.event_desc.append((int(event.round_index), int(event.seat),
                                    self.describe(event.action, self.cur_level)))

    def _table(self, engine, state, actions, stream) -> np.ndarray:
        saved = self.stream
        self.stream = stream
        try:
            inputs = self.decision_inputs(engine, state, actions)
            with torch.inference_mode():
                table = self.actor._log_prob_table(inputs, None, None)
        finally:
            self.stream = saved
        return table[0, :len(actions)].double().exp().numpy()

    def _sub_stream(self, keep) -> PublicStream:
        s = PublicStream(self.stream.match_id)
        for t, r, p in zip(self.stream.tokens, self.stream.rounds, self.stream.phases):
            if keep(r):
                s.append_token(t, r, p)
        return s

    def _swap_stream(self, current: int) -> tuple[PublicStream | None, int]:
        own_earlier = sum(1 for r in self.stream.rounds if r < current)
        if current == 0 or own_earlier == 0:
            return None, -1
        key = hash((self.match_tag, current)) & 0xFFFFFFFF
        order = list(range(len(self.donors)))
        random.Random(key).shuffle(order)
        for i in order:
            d = self.donors[i]
            if d["team"] != self.team or d["tag"] == self.match_tag:
                continue
            idx = np.nonzero(d["rounds"] < current)[0]
            if len(idx) < own_earlier:
                continue
            idx = idx[-own_earlier:]
            s = PublicStream(self.stream.match_id)
            for j in idx:
                s.append_token(d["tokens"][j], int(d["rounds"][j]), int(d["phases"][j]))
            for t, r, p in zip(self.stream.tokens, self.stream.rounds, self.stream.phases):
                if r == current:
                    s.append_token(t, r, p)
            assert s.prefix == self.stream.prefix
            return s, i
        return None, -1

    def select(self, engine, state, actions, rng) -> int:
        if int(state.phase) != PLAY:
            return super().select(engine, state, actions, rng)
        choice = super().select(engine, state, actions, rng)
        current = int(state.round_index)
        seat = int(state.to_move)
        p_full = self._table(engine, state, actions, self.stream)
        assert int(p_full.argmax()) == choice, "probe FULL table disagrees with policy choice"
        p_none = self._table(engine, state, actions, PublicStream(self.stream.match_id))
        round_stream = self._sub_stream(lambda r: r == current)
        p_round = (p_full if round_stream.prefix == self.stream.prefix
                   else self._table(engine, state, actions, round_stream))
        swap_stream, donor = self._swap_stream(current)
        p_swap = None if swap_stream is None else self._table(engine, state, actions, swap_stream)
        hands = [len(state.hand(s)) for s in range(4)]
        chosen = actions[choice]
        rec = {
            "tag": self.match_tag, "round": current, "seat": seat, "level": int(state.level),
            "lead": bool(state.top_is_open), "n_legal": len(actions),
            "prefix": self.stream.prefix, "round_events": round_stream.prefix,
            "own_left": hands[seat], "partner_left": hands[(seat + 2) % 4],
            "opp_left": [hands[(seat + 1) % 4], hands[(seat + 3) % 4]],
            "pass": bool(chosen.is_pass), "type": None if chosen.is_pass else str(chosen.type),
            "has_nonpass": any(not a.is_pass for a in actions),
            "arg": {"full": choice, "none": int(p_none.argmax()), "round": int(p_round.argmax()),
                    "swap": None if p_swap is None else int(p_swap.argmax())},
            "pmax_full": float(p_full.max()),
            "tv": {"none": tv(p_full, p_none), "round": tv(p_full, p_round),
                   "swap": None if p_swap is None else tv(p_full, p_swap),
                   "swap_vs_round": None if p_swap is None else tv(p_swap, p_round)},
            "donor": donor,
        }
        if self.describe is not None:
            idx = sorted(set(np.argsort(-p_full)[:3]) | {rec["arg"]["none"], rec["arg"]["round"]}
                         | ({rec["arg"]["swap"]} if p_swap is not None else set()))
            rec["cands"] = [{"i": int(i), "a": self.describe(actions[i], int(state.level)),
                             "full": float(p_full[i]), "none": float(p_none[i]),
                             "round": float(p_round[i]),
                             "swap": None if p_swap is None else float(p_swap[i])} for i in idx]
            from review_lib import cards_str
            rec["hand"] = cards_str(list(state.hand(seat)), int(state.level))
            rec["history_len"] = len(self.event_desc)
        self.records.append(rec)
        return choice
