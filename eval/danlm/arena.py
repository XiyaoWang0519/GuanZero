"""Play rounds against DanLM with both engines in lockstep.

Run inside the Python 3.12 environment that has DanLM's dependencies and a
``gd`` extension built for 3.12 (``docs/STAGE_C_TODO.md`` row C0)::

    PYTHONPATH=python:. DANLM_ROOT=.work/external/DanLM \\
      .work/external/danlm-venv/bin/python -m eval.danlm.arena diff --rounds 200
    PYTHONPATH=python:. DANLM_ROOT=.work/external/DanLM \\
      .work/external/danlm-venv/bin/python -m eval.danlm.arena duplicate \\
      --checkpoint .work/runpod-b8/results/runs/league/run/latest.pt --deals 400

DanLM's engine is the referee: its legal plays, trick flow and finish order
decide the outcome. Our engine mirrors every action so that our policy sees
its own observation encoding. Every point where the two engines disagree is
counted as a divergence class; a round whose mirror cannot continue is
reported as ``mirror_failed`` and excluded from the score, with its count.

``diff`` plays DanLM (or ``random``) in all four seats and compares the full
legal sets at every decision; ``duplicate`` seats our policy and DanLM as
teams, each deal played twice with the teams swapped.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np

import gd

from eval.danlm.bridge import (BIG_JOKER, NormalizedPlay, PlayIndex, card_ours_to_theirs,
                               card_theirs_to_ours, decode_play, hand_ours_to_theirs,
                               level_ours_to_theirs, normalize_action)
from eval.duplicate import bootstrap_interval

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DANLM_ROOT = ROOT / ".work/external/DanLM"
DEFAULT_DANLM_CHECKPOINT = "ckpts/DanLM_v1/dansformer_v1_best_eval.pt"
SAMPLE_LIMIT = 20


def danlm_root() -> Path:
    return Path(os.environ.get("DANLM_ROOT", DEFAULT_DANLM_ROOT))


def import_danzero():
    """Import DanLM's compiled package from ``DANLM_ROOT``; never vendored."""
    root = str(danlm_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from danzero.engine import cards, game, tribute  # noqa: WPS433
    from danzero.eval import agents  # noqa: WPS433
    from eval.danlm import fast_obs

    # DanLM's compiled get_observation spends about 40 ms per call waiting;
    # the verified pure-Python replacement (fast_obs.verify, zero mismatches
    # over 13,417 steps) builds the identical Observation in 0.2 ms.
    if os.environ.get("DANLM_SLOW_OBS") != "1":
        fast_obs.install()
    return cards, game, tribute, agents


class DanLMSeats:
    """Four DanLM ``EvalAgent`` instances sharing one loaded model."""

    def __init__(self, spec: str = DEFAULT_DANLM_CHECKPOINT, device: str = "cpu") -> None:
        cards, game, tribute, agents = import_danzero()
        self.spec = spec
        if not spec.endswith(".pt"):
            # random, bot:<name>, v1t:<path>, mlp:<path> or a bare .onnx: DanLM's
            # own factory, one agent per seat, paths resolved under DANLM_ROOT.
            prefix, _, path = spec.rpartition(":")
            if path and not path.startswith("/") and (path.endswith(".onnx") or path.endswith(".pt")):
                spec = f"{prefix}:{danlm_root() / path}" if prefix else str(danlm_root() / path)
            self.agents = [agents.create_agent(spec, device) for _ in range(4)]
            self.model_sha256 = None
            return
        import dataclasses
        import hashlib

        import torch
        from danzero.model.transformer import TransformerConfig, TransformerQNetwork  # noqa: WPS433

        path = Path(spec)
        if not path.is_absolute():
            path = danlm_root() / path
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        raw = checkpoint.get("model_config") or checkpoint.get("config", {})
        valid = {f.name for f in dataclasses.fields(TransformerConfig)}
        config = TransformerConfig(**{k: v for k, v in raw.items() if k in valid})
        model = TransformerQNetwork(config)
        model.load_state_dict(checkpoint.get("model_state_dict") or checkpoint["model"])
        model.eval()
        with path.open("rb") as stream:
            self.model_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        self.agents = [agents.create_agent_from_model(model, "transformer", device=device,
                                                      max_seq_len=config.max_seq_len)
                       for _ in range(4)]


@dataclass
class Divergences:
    counts: dict[str, int] = field(default_factory=dict)
    samples: dict[str, list[str]] = field(default_factory=dict)

    def add(self, kind: str, detail: str = "", n: int = 1) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + n
        bucket = self.samples.setdefault(kind, [])
        if detail and len(bucket) < SAMPLE_LIMIT:
            bucket.append(detail)

    def merge(self, other: "Divergences") -> None:
        for kind, n in other.counts.items():
            self.add(kind, n=n)
            for detail in other.samples.get(kind, []):
                bucket = self.samples.setdefault(kind, [])
                if len(bucket) < SAMPLE_LIMIT:
                    bucket.append(detail)


@dataclass
class RoundRecord:
    status: str                    # "ok" or "mirror_failed"
    seats: tuple[str, str, str, str]
    finish_order: list[int]        # DanLM's, the referee
    rewards: list[int]             # DanLM's per-seat reward
    our_order: list[int] | None    # gd's order when the mirror completed
    our_returns: list[int] | None
    decisions: int
    tribute: str                   # "none", "single", "double", "anti"
    divergences: Divergences


class LockstepRound:
    """One round: DanLM's engine referees, gd mirrors, seats choose."""

    def __init__(self, deal: gd.DealSpec, seat_kind: tuple[str, str, str, str],
                 our_policy, danlm: DanLMSeats | None, rng: random.Random, *,
                 rules: gd.RuleConfig | None = None, diff: bool = False,
                 round_id: str = "") -> None:
        self.cards, self.game, self.tribute, _ = import_danzero()
        self.deal, self.seat_kind, self.rng = deal, seat_kind, rng
        self.our_policy, self.danlm, self.diff, self.round_id = our_policy, danlm, diff, round_id
        rules = rules or gd.RuleConfig.house()
        self.engine_full = gd.Engine(rules, gd.ActionConfig.full())
        self.engine_canon = gd.Engine(rules, gd.ActionConfig())
        # Forced passes stay explicit so both engines ask the same seat.
        self.engine_full.auto_pass = False
        self.engine_canon.auto_pass = False
        self.state = gd.MatchState()
        self.engine_full.set_deal(self.state, deal)
        self.level = int(deal.level)
        self.level_t = level_ours_to_theirs(self.level)
        self.div = Divergences()
        self.decisions = 0
        self.tribute_kind = "none"

    # -- helpers ---------------------------------------------------------

    def their_agent(self, seat: int):
        """The DanLM-side agent for a seat; ``danlm`` may map seat kinds to sets."""
        kind = self.seat_kind[seat]
        if isinstance(self.danlm, dict):
            return self.danlm[kind].agents[seat]
        return self.danlm.agents[seat]

    def hand_t(self, seat: int) -> np.ndarray:
        return hand_ours_to_theirs(self.state.hand(seat))

    def note(self, kind: str, detail: str = "", n: int = 1) -> None:
        self.div.add(kind, f"{self.round_id}: {detail}" if detail else "", n)

    def apply_ours(self, action) -> None:
        self.engine_full.apply(self.state, action)

    # -- tribute ---------------------------------------------------------

    def run_tribute(self) -> tuple[list, np.ndarray | None]:
        """Drive gd's tribute phases, asking DanLM agents for their seats.

        Returns DanLM ``TributeRecord`` list and the anti-tribute matrix.
        """
        if not has_tribute(self.deal):
            return [], None
        order = [int(s) for s in self.deal.prev_order]
        p1, p2, p3, p4 = order
        is_double = (p3 - p4) % 4 == 2
        TributeRecord = self.tribute.TributeRecord
        if self.state.phase == gd.Phase.Play:
            # gd decided anti-tribute. Reconstruct DanLM's records.
            self.tribute_kind = "anti"
            holders = [s for s in ((p3, p4) if is_double else (p4,))
                       if self.state.hand(s).count(BIG_JOKER) >= 2]
            if not holders and is_double:
                holders = [p4, p3]
            anti = np.zeros((4, 54), np.float32)
            records = []
            for seat in holders:
                anti[seat, BIG_JOKER] += 2.0 if len(holders) == 1 else 1.0
                records.append(TributeRecord(seat, seat, -1))
            if len(holders) == 1:
                records.append(TributeRecord(holders[0], holders[0], -1))
            if self.state.to_move != p1:
                self.note("tribute_leader_mismatch", f"anti: ours {self.state.to_move} theirs {p1}")
            return records, anti
        self.tribute_kind = "double" if is_double else "single"
        gives: dict[int, int] = {}          # payer -> DanLM card
        give_records: list = []
        receivers: dict[int, int] = {}      # payer -> receiver
        steps = 0
        while self.state.phase in (gd.Phase.Tribute, gd.Phase.BackTribute) and steps < 8:
            steps += 1
            seat = int(self.state.to_move)
            actions = self.engine_canon.legal_actions(self.state)
            legal_ours = {int(a.cards[0]) for a in actions}
            giving = self.state.phase == gd.Phase.Tribute
            if giving:
                legal_t = self.tribute.tribute_give_legal_cards(self.hand_t(seat), self.level_t)
            else:
                legal_t = self.tribute.tribute_back_legal_cards(self.hand_t(seat), self.level_t)
            legal_t_ours = {card_theirs_to_ours(c) for c in legal_t}
            if legal_t_ours != legal_ours:
                self.note("tribute_legal_mismatch",
                          f"{'give' if giving else 'back'} seat {seat} ours {sorted(legal_ours)} "
                          f"theirs {sorted(legal_t_ours)}")
            if self.seat_kind[seat] == "ours":
                index = self.our_policy.select(self.engine_canon, self.state, actions, self.rng)
                action = actions[index]
            else:
                if giving:
                    receiver = None if is_double else p1
                    card_t = self.their_agent(seat).select_tribute_give(
                        self.hand_t(seat), legal_t, is_double, receiver)
                else:
                    back_to = next((payer for payer, r in receivers.items() if r == seat), p4)
                    card_t = self.their_agent(seat).select_tribute_back(
                        self.hand_t(seat), legal_t, back_to, give_records)
                card = card_theirs_to_ours(int(card_t))
                action = next((a for a in actions if int(a.cards[0]) == card), None)
                if action is None:
                    self.note("tribute_choice_illegal_in_ours", f"seat {seat} card {card_t}")
                    action = actions[self.engine_canon.greedy(self.state)]
            card_t = card_ours_to_theirs(int(action.cards[0]))
            self.apply_ours(action)
            self.decisions += 1
            if giving:
                gives[seat] = card_t
                if not is_double:
                    receivers[seat] = p1
                    give_records.append(TributeRecord(seat, p1, card_t))
                elif len(gives) == 2:
                    receivers.update(self.double_receivers(gives, p1, p2, p3, p4))
                    for payer in (p4, p3):
                        give_records.append(TributeRecord(payer, receivers[payer], gives[payer]))
            else:
                back_to = next((payer for payer, r in receivers.items() if r == seat), p4)
                give_records.append(TributeRecord(seat, back_to, card_t))
        if self.state.phase != gd.Phase.Play:
            self.note("tribute_phase_stuck", f"phase {self.state.phase}")
        # Referee leader by DanLM's rule.
        if is_double:
            higher = max((p4, p3), key=lambda p: self.cards.single_card_power(gives[p], self.level_t))
            leader_t = next(p for p in (p4, p3) if receivers[p] == p1)
            if self.cards.single_card_power(gives[p4], self.level_t) != \
                    self.cards.single_card_power(gives[p3], self.level_t):
                leader_t = higher
        else:
            leader_t = p4
        if int(self.state.to_move) != leader_t:
            self.note("tribute_leader_mismatch", f"ours {self.state.to_move} theirs {leader_t}")
        return give_records, None

    def double_receivers(self, gives: dict[int, int], p1: int, p2: int, p3: int, p4: int) -> dict[int, int]:
        """DanLM's pairing: higher card to first place; equal cards go upstream."""
        power4 = self.cards.single_card_power(gives[p4], self.level_t)
        power3 = self.cards.single_card_power(gives[p3], self.level_t)
        if power4 == power3:
            return {p4: (p4 + 3) % 4, p3: (p3 + 3) % 4}
        return {p4: p1, p3: p2} if power4 > power3 else {p4: p2, p3: p1}

    # -- play ------------------------------------------------------------

    def play(self) -> RoundRecord:
        for seat in range(4):
            if self.seat_kind[seat] != "ours":
                self.their_agent(seat).reset(seat, self.level_t)
        records, anti = self.run_tribute()
        if has_tribute(self.deal):
            for seat in range(4):
                if self.seat_kind[seat] != "ours":
                    self.their_agent(seat).notify_tribute(records, anti)
        hands_t = [self.hand_t(seat).copy() for seat in range(4)]
        for seat in range(4):
            if self.seat_kind[seat] != "ours":
                self.their_agent(seat).notify_start(hands_t[seat].copy())
        team_levels = tuple(level_ours_to_theirs(int(v)) for v in self.deal.team_levels)
        rnd = self.game.GuanDanRound(level=self.level_t, hands=[h.copy() for h in hands_t],
                                     first_player=int(self.state.to_move), team_levels=team_levels)
        status = "ok"
        obs = rnd.get_observation()
        guard = 0
        while obs is not None and not rnd.done:
            guard += 1
            if guard > 2000:
                self.note("runaway_round")
                status = "mirror_failed"
                break
            player = int(obs.player)
            index = PlayIndex(obs.legal_plays, self.level)
            theirs_only_pass = len(index.plays) == 1 and index.plays[0].is_pass
            if self.state.phase != gd.Phase.Play:
                self.note("round_end_mismatch", f"gd ended first, theirs at {player}")
                status = "mirror_failed"
                break
            ours_actions = self.engine_full.legal_actions(self.state)
            if player != int(self.state.to_move):
                ours_only_pass = len(ours_actions) == 1 and ours_actions[0].is_pass
                if theirs_only_pass:
                    obs = self.step_theirs(rnd, obs, index.plays.index(next(p for p in index.plays if p.is_pass)))
                    continue
                if ours_only_pass:
                    self.apply_ours(ours_actions[0])
                    continue
                self.note("to_move_mismatch", f"theirs {player} ours {self.state.to_move}")
                status = "mirror_failed"
                break
            if self.diff:
                self.compare_legal_sets(index, ours_actions)
            if self.seat_kind[player] != "ours":
                choice = int(self.their_agent(player).select_play(obs, rnd))
                play = index.plays[choice]
                action = self.match_theirs(play, ours_actions)
                if action is None:
                    status = "mirror_failed"
                    break
                self.apply_ours(action)
            else:
                candidates = self.engine_canon.legal_actions(self.state)
                rows, usable = [], []
                for action in candidates:
                    row, how = index.find(normalize_action(action))
                    if row is None:
                        self.note("our_candidate_unmatched", f"{action}")
                        continue
                    rows.append((row, how))
                    usable.append(action)
                if not usable:
                    self.note("our_candidates_all_unmatched")
                    status = "mirror_failed"
                    break
                pick = self.our_policy.select(self.engine_canon, self.state, usable, self.rng)
                choice, how = rows[pick]
                if how == "reading":
                    self.note("our_choice_reading", f"{usable[pick]} -> {index.plays[choice]}")
                self.apply_ours(usable[pick])
            self.decisions += 1
            obs = self.step_theirs(rnd, obs, choice)
        finish = list(rnd.finish_order)
        rewards = [int(self.game.compute_reward(finish, p)) for p in range(4)] if rnd.done else [0] * 4
        our_order = our_returns = None
        if status == "ok":
            if not rnd.done:
                self.note("their_round_not_done")
                status = "mirror_failed"
            elif self.state.phase != gd.Phase.RoundEnd:
                self.note("round_end_mismatch", "theirs ended first")
                status = "mirror_failed"
            else:
                result = self.engine_full.end_round(self.state)
                our_order, our_returns = list(result.order), list(result.seat_return)
                if our_order[:len(finish)] != finish and our_order != finish:
                    self.note("finish_order_mismatch", f"ours {our_order} theirs {finish}")
                if our_returns != rewards:
                    self.note("reward_mismatch", f"ours {our_returns} theirs {rewards}")
        return RoundRecord(status, self.seat_kind, finish, rewards, our_order, our_returns,
                           self.decisions, self.tribute_kind, self.div)

    def step_theirs(self, rnd, obs, choice: int):
        play = obs.legal_plays[choice]
        new_obs = rnd.step(choice, obs)
        new_trick = (new_obs is not None and bool(new_obs.is_leading)
                     and int(new_obs.player) == int(rnd.state.lead_player))
        for seat in range(4):
            if self.seat_kind[seat] != "ours":
                self.their_agent(seat).observe_action(int(obs.player), play, new_trick)
        return new_obs

    def match_theirs(self, play: NormalizedPlay, ours_actions):
        exact = [a for a in ours_actions if normalize_action(a) == play]
        if exact:
            return exact[0]
        same_cards = [a for a in ours_actions if a.type == play.type
                      and tuple(sorted(int(c) for c in a.cards)) == play.cards]
        if same_cards:
            self.note("their_choice_reading", f"{play} -> {same_cards[0]}")
            return same_cards[0]
        self.note("their_choice_missing", f"{play}")
        return None

    def compare_legal_sets(self, index: PlayIndex, ours_actions) -> None:
        theirs = set(index.plays)
        ours = {normalize_action(a) for a in ours_actions}
        ours_cards = {(p.type, p.cards) for p in ours}
        theirs_cards = {(p.type, p.cards) for p in theirs}
        for play in theirs - ours:
            kind = "reading" if (play.type, play.cards) in ours_cards else "cards"
            self.note(f"legal_theirs_only/{play.type}/{kind}",
                      f"{play}" if kind == "cards" or self.rng.random() < 0.05 else "")
        for play in ours - theirs:
            kind = "reading" if (play.type, play.cards) in theirs_cards else "cards"
            self.note(f"legal_ours_only/{play.type}/{kind}",
                      f"{play}" if kind == "cards" or self.rng.random() < 0.05 else "")
        self.note("decisions_compared")


# -- deals -----------------------------------------------------------------

def generate_deals(count: int, seed: int, tribute_fraction: float = 0.5) -> list[gd.DealSpec]:
    """Like ``eval.duplicate.generate_deals`` plus DanLM's random tribute setup."""
    if count < 1 or not 0 <= tribute_fraction <= 1:
        raise ValueError("positive deal count and tribute fraction in [0, 1] required")
    rng = random.Random(seed)
    deals = []
    for _ in range(count):
        deck = [card for card in range(54) for _ in range(2)]
        rng.shuffle(deck)
        deal = gd.DealSpec()
        deal.hands = [sorted(deck[seat * 27:(seat + 1) * 27]) for seat in range(4)]
        deal.level = rng.randrange(13)
        deal.team_levels = [deal.level, deal.level]
        deal.owner = -1
        if rng.random() < tribute_fraction:
            order = list(range(4))
            rng.shuffle(order)
            deal.prev_order = order
        else:
            deal.leader = rng.randrange(4)
        deals.append(deal)
    return deals


def has_tribute(deal: gd.DealSpec) -> bool:
    """True when the deal opens in its tribute phase (a full previous order)."""
    return sorted(int(s) for s in deal.prev_order) == [0, 1, 2, 3]


def deal_to_json(deal: gd.DealSpec) -> dict:
    return {"hands": [list(h) for h in deal.hands], "level": int(deal.level),
            "team_levels": list(deal.team_levels), "leader": int(deal.leader),
            "prev_order": list(deal.prev_order) if has_tribute(deal) else []}


def deal_from_json(payload: dict) -> gd.DealSpec:
    deal = gd.DealSpec()
    deal.hands = payload["hands"]
    deal.level = payload["level"]
    deal.team_levels = payload["team_levels"]
    deal.owner = -1
    if len(payload["prev_order"]) == 4:
        deal.prev_order = payload["prev_order"]
    else:
        deal.leader = payload["leader"]
    return deal


# -- workers ---------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _init_worker(checkpoint: str | None, danlm_spec: str, torch_threads: int) -> None:
    import torch

    torch.set_num_threads(torch_threads)
    sets = {"danlm": DanLMSeats(danlm_spec)}
    _WORKER["policy"] = None
    if checkpoint and checkpoint.startswith("danlm:"):
        # Team A is another DanLM-side agent (their MLP reproduction, a bot,
        # random): a calibration of the harness against DanLM's own numbers.
        sets["danlm_a"] = DanLMSeats(checkpoint[len("danlm:"):])
    elif checkpoint:
        from eval.policies import load_policy

        _WORKER["policy"] = load_policy(checkpoint)
    _WORKER["danlm"] = sets


def _run_rounds(job: dict) -> list[dict]:
    """Play the rounds of one job; returns JSON-ready records."""
    danlm, policy = _WORKER["danlm"], _WORKER["policy"]
    out = []
    for item in job["rounds"]:
        deal = deal_from_json(item["deal"])
        rng = random.Random(item["seed"])
        record = LockstepRound(deal, tuple(item["seats"]), policy, danlm, rng,
                               diff=job["diff"], round_id=item["id"]).play()
        out.append({"id": item["id"], "status": record.status, "seats": list(record.seats),
                    "finish_order": record.finish_order, "rewards": record.rewards,
                    "our_order": record.our_order, "our_returns": record.our_returns,
                    "decisions": record.decisions, "tribute": record.tribute,
                    "divergences": {"counts": record.divergences.counts,
                                    "samples": record.divergences.samples}})
    return out


def run_jobs(rounds: list[dict], *, checkpoint: str | None, danlm_spec: str, workers: int,
             diff: bool, torch_threads: int = 1, chunk: int = 8) -> list[dict]:
    jobs = [{"rounds": rounds[i:i + chunk], "diff": diff} for i in range(0, len(rounds), chunk)]
    if workers <= 1:
        _init_worker(checkpoint, danlm_spec, torch_threads)
        return [r for job in jobs for r in _run_rounds(job)]
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(workers, mp_context=context, initializer=_init_worker,
                             initargs=(checkpoint, danlm_spec, torch_threads)) as pool:
        results = list(pool.map(_run_rounds, jobs))
    return [r for group in results for r in group]


def merge_divergences(records: list[dict]) -> dict:
    total = Divergences()
    for record in records:
        d = Divergences(dict(record["divergences"]["counts"]),
                        {k: list(v) for k, v in record["divergences"]["samples"].items()})
        total.merge(d)
    return {"counts": dict(sorted(total.counts.items())), "samples": total.samples}


# -- modes -----------------------------------------------------------------

def run_diff(rounds: int, seed: int, danlm_spec: str, workers: int, tribute_fraction: float) -> dict:
    deals = generate_deals(rounds, seed, tribute_fraction)
    items = [{"id": f"diff-{i}", "deal": deal_to_json(deal), "seed": seed + i,
              "seats": ["danlm"] * 4} for i, deal in enumerate(deals)]
    started = time.monotonic()
    records = run_jobs(items, checkpoint=None, danlm_spec=danlm_spec, workers=workers, diff=True)
    elapsed = time.monotonic() - started
    statuses = {}
    for r in records:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    tributes = {}
    for r in records:
        tributes[r["tribute"]] = tributes.get(r["tribute"], 0) + 1
    return {"mode": "diff", "rounds": rounds, "seed": seed, "danlm": danlm_spec,
            "tribute_fraction": tribute_fraction, "elapsed_seconds": elapsed,
            "decisions": sum(r["decisions"] for r in records), "statuses": statuses,
            "tribute_kinds": tributes, "divergences": merge_divergences(records)}


def run_duplicate(checkpoint: str, deals: int, seed: int, danlm_spec: str, workers: int,
                  tribute_fraction: float, bootstrap_samples: int = 2000) -> dict:
    deal_specs = generate_deals(deals, seed, tribute_fraction)
    items = []
    team_a = "danlm_a" if checkpoint.startswith("danlm:") else "ours"
    for i, deal in enumerate(deal_specs):
        payload = deal_to_json(deal)
        items.append({"id": f"deal-{i}/a", "deal": payload, "seed": seed + i,
                      "seats": [team_a, "danlm", team_a, "danlm"]})
        items.append({"id": f"deal-{i}/b", "deal": payload, "seed": seed + i,
                      "seats": ["danlm", team_a, "danlm", team_a]})
    started = time.monotonic()
    records = run_jobs(items, checkpoint=checkpoint, danlm_spec=danlm_spec, workers=workers, diff=False)
    elapsed = time.monotonic() - started
    by_id = {r["id"]: r for r in records}
    per_deal, wins, legs, excluded = [], 0, 0, 0
    per_deal_tribute: dict[str, list[float]] = {}
    for i, deal in enumerate(deal_specs):
        a, b = by_id[f"deal-{i}/a"], by_id[f"deal-{i}/b"]
        if a["status"] != "ok" or b["status"] != "ok":
            excluded += 1
            continue
        ours_a, ours_b = a["rewards"][0], b["rewards"][1]
        per_deal.append((ours_a + ours_b) / 2)
        per_deal_tribute.setdefault(a["tribute"], []).append((ours_a + ours_b) / 2)
        wins += int(a["finish_order"][0] % 2 == 0) + int(b["finish_order"][0] % 2 == 1)
        legs += 2
    if checkpoint.startswith("danlm:"):
        name, checkpoint_id = checkpoint, None
    else:
        from eval.policies import load_policy

        policy = load_policy(checkpoint)
        name, checkpoint_id = policy.name, getattr(policy, "checkpoint_id", None)
    summary = {"mode": "duplicate", "checkpoint": checkpoint, "checkpoint_name": name,
               "checkpoint_id": checkpoint_id, "danlm": danlm_spec,
               "deals": deals, "seed": seed, "tribute_fraction": tribute_fraction,
               "elapsed_seconds": elapsed, "scored_deals": len(per_deal),
               "excluded_deals_mirror_failed": excluded,
               "levels_per_round": float(np.mean(per_deal)) if per_deal else None,
               "levels_per_round_ci95": list(bootstrap_interval(per_deal, seed, bootstrap_samples))
               if len(per_deal) > 1 else None,
               "round_win_rate": wins / legs if legs else None, "round_legs": legs,
               "by_tribute": {k: {"deals": len(v), "levels_per_round": float(np.mean(v))}
                              for k, v in per_deal_tribute.items()},
               "divergences": merge_divergences(records),
               "reading": "levels_per_round is our team's mean net level gain per round over "
                          "both legs of each deal (DanLM's engine scores); round_win_rate is "
                          "the fraction of legs whose first finisher is on our team, DanLM's "
                          "single-round metric"}
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("diff", "duplicate"):
        p = sub.add_parser(name)
        p.add_argument("--danlm", default=DEFAULT_DANLM_CHECKPOINT,
                       help="DanLM checkpoint relative to DANLM_ROOT, 'random' or 'bot:<name>'")
        p.add_argument("--seed", type=int, default=20260928)
        p.add_argument("--workers", type=int, default=1)
        p.add_argument("--tribute-fraction", type=float, default=0.5)
        p.add_argument("--output", default=None)
    sub.choices["diff"].add_argument("--rounds", type=int, default=200)
    sub.choices["duplicate"].add_argument(
        "--checkpoint", required=True,
        help="our load_policy spec for team A, or danlm:<spec> for a DanLM-side agent")
    sub.choices["duplicate"].add_argument("--deals", type=int, default=400)
    args = parser.parse_args(argv)
    if args.mode == "diff":
        report = run_diff(args.rounds, args.seed, args.danlm, args.workers, args.tribute_fraction)
        output = args.output or "docs/reports/stage-c-danlm-diff.json"
    else:
        report = run_duplicate(args.checkpoint, args.deals, args.seed, args.danlm, args.workers,
                               args.tribute_fraction)
        output = args.output or "docs/reports/stage-c-danlm-duplicate.json"
    report["danlm_root"] = str(danlm_root())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "divergences"}, indent=2))
    print("divergence counts:", json.dumps(report["divergences"]["counts"], indent=2))
    print("written", output)


if __name__ == "__main__":
    main()
