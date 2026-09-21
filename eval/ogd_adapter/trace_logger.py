"""Drives four random players through the OpenGuanDan simulator and logs
complete matches (level 2 until one team passes A, or the shuffle limit) to
newline-delimited JSON, one JSON object per event.

Usage:
    python -m eval.ogd_adapter.trace_logger --matches 1000 --out <dir> --seed 0

Resumable: a match whose output file already exists (and is non-empty) is
skipped, so a killed run can be restarted with the same command.

Event schema (one JSON object per line):
  deal:          hands (4 lists of card strings), round level, first_leader
                 (filled in once the post-tribute leader is known), team
                 levels, owner_seat/owner_team (None for round 1).
  act:           seat, phase (play/tribute/back), hand at that moment, the
                 full actionList offered, chosen_index, chosen_action.
  tribute:       the (payer, receiver, card) tuples, as broadcast.
  anti-tribute:  anti_num, anti_pos, as broadcast.
  back:          the (payer, receiver, card) back-tribute tuples.
  episodeOver:   finishing order, remaining hands (double win), round level.
  gameResult:    winning_team, final team levels.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from eval.ogd_adapter import bridge

# Messages broadcast identically to all four seats: log once (seat 0) rather
# than four times.
_BROADCAST_STAGES = {"episodeOver", "tribute", "back", "anti-tribute", "gameResult", "gameOver"}


class _Recorder:
    """Turns the raw `Msg(seat, body)` stream of one match into trace events."""

    def __init__(self, match_id: int) -> None:
        self.match_id = match_id
        self.round_idx = -1
        self.events: list[dict[str, Any]] = []
        self.next_idx: int | None = None

        self._hands: dict[int, list[str]] = {}
        self._last_team_levels: dict[str, str] = {"0": "2", "1": "2"}
        self._deal_event_idx: int | None = None
        self._leader_pending = False
        self._deal_pending = False
        self._pending_owner_seat: int | None = None

    def process(self, msgs, env: Any) -> None:
        self.next_idx = None
        for seat, body in msgs:
            mtype = body.get("type")
            stage = body.get("stage")
            if mtype == "notify":
                self._on_notify(seat, stage, body, env)
            elif mtype == "act":
                self._on_act(seat, stage, body)

    def _on_notify(self, seat: int, stage: str, body: dict, env: Any) -> None:
        if stage == "beginning":
            # NOTE: `notify_beginning`'s curRank/selfRank/oppoRank are raw
            # integer rank indices (table.py passes `self.__rank`/`player.rank`
            # directly), unlike every other message, which passes the
            # stringified rank through `Card.RANKS[...]`. We only use this
            # message for `handCards` (a list of card strings either way) and
            # read the level/team levels off the next `act` message instead,
            # which is consistently typed.
            self._hands[seat] = body["handCards"]
            if len(self._hands) == 4:
                self._deal_pending = True
                self._pending_owner_seat = getattr(env, "rank_belongs", -1)
            return
        if stage in _BROADCAST_STAGES and seat != 0:
            return  # identical broadcast to every seat; keep one copy
        if stage == "episodeOver":
            self.events.append({
                "event": "episodeOver", "match": self.match_id, "round": self.round_idx,
                "level": body["curRank"], "order": body["order"], "rest_cards": body["restCards"],
            })
        elif stage == "tribute":
            self.events.append({
                "event": "tribute", "match": self.match_id, "round": self.round_idx,
                "result": body["result"],
            })
        elif stage == "back":
            self.events.append({
                "event": "back", "match": self.match_id, "round": self.round_idx,
                "result": body["result"],
            })
        elif stage == "anti-tribute":
            self.events.append({
                "event": "anti-tribute", "match": self.match_id, "round": self.round_idx,
                "anti_num": body["antiNum"], "anti_pos": body["antiPos"],
            })
        elif stage == "gameResult":
            winning_team = 0 if body["victoryNum"][0] > 0 else 1
            self.events.append({
                "event": "gameResult", "match": self.match_id,
                "winning_team": winning_team,
                "final_ranks": dict(self._last_team_levels),
            })
        # "play" (notify_play broadcast) and "gameOver" carry no information
        # beyond what the "act" and "gameResult" events already capture.

    def _on_act(self, seat: int, stage: str, body: dict) -> None:
        if self._deal_pending:
            self._flush_deal(seat, body)
        n = body["indexRange"] + 1
        idx = random.randrange(0, n) if n > 0 else 0
        action_list = body["actionList"]
        chosen = action_list[idx]
        self.events.append({
            "event": "act", "match": self.match_id, "round": self.round_idx,
            "seat": seat, "phase": stage, "hand": body["handCards"],
            "action_list": action_list, "chosen_index": idx, "chosen_action": chosen,
        })
        if self._leader_pending and stage == "play" and self._deal_event_idx is not None:
            self.events[self._deal_event_idx]["first_leader"] = seat
            self._leader_pending = False
        self.next_idx = idx

    def _flush_deal(self, acting_seat: int, act_body: dict) -> None:
        self.round_idx += 1
        level = act_body["curRank"]
        own_team = acting_seat % 2
        opp_team = 1 - own_team
        team_levels = {str(own_team): act_body["selfRank"], str(opp_team): act_body["oppoRank"]}
        self._last_team_levels = team_levels
        owner_seat = self._pending_owner_seat
        self.events.append({
            "event": "deal", "match": self.match_id, "round": self.round_idx,
            "level": level,
            "hands": [self._hands.get(p, []) for p in range(4)],
            "team_levels": team_levels,
            "owner_seat": None if owner_seat is None or owner_seat < 0 else owner_seat,
            "owner_team": None if owner_seat is None or owner_seat < 0 else owner_seat % 2,
            "first_leader": None,
        })
        self._deal_event_idx = len(self.events) - 1
        self._leader_pending = True
        self._deal_pending = False
        self._hands = {}


def play_match(match_id: int, seed: int) -> list[dict[str, Any]]:
    """Drive one full match (level 2 to a match win or the shuffle limit)."""
    random.seed(seed)
    env = bridge.make_env()
    rec = _Recorder(match_id)
    fn = env.start
    pending_idx: int | None = None
    while True:
        msgs = fn({"actIndex": pending_idx}) if pending_idx is not None else fn()
        rec.process(msgs, env)
        if env.results is not None:
            break
        pending_idx = rec.next_idx
        fn = env.loop
    return rec.events


def _write_events(path: Path, events: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev, separators=(",", ":")))
            f.write("\n")
    tmp.replace(path)


def run(matches: int, out_dir: Path, seed: int, progress_every: int = 25) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    done = 0
    skipped = 0
    for i in range(matches):
        path = out_dir / f"match_{i:05d}.jsonl"
        if path.exists() and path.stat().st_size > 0:
            skipped += 1
            continue
        events = play_match(match_id=i, seed=seed + i)
        _write_events(path, events)
        done += 1
        if done % progress_every == 0:
            elapsed = time.perf_counter() - start
            rate = done / elapsed if elapsed > 0 else 0.0
            print(f"[{i + 1}/{matches}] generated={done} skipped={skipped} "
                  f"elapsed={elapsed:.1f}s rate={rate:.2f} matches/s", flush=True)
    elapsed = time.perf_counter() - start
    total_bytes = sum(p.stat().st_size for p in out_dir.glob("match_*.jsonl"))
    print(f"done: generated={done} skipped={skipped} elapsed={elapsed:.1f}s "
          f"trace_dir={out_dir} total_size={total_bytes / 1e6:.1f}MB")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matches", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out_dir = args.out or bridge.ogd_traces_dir()
    try:
        run(args.matches, out_dir, args.seed)
    finally:
        bridge.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
