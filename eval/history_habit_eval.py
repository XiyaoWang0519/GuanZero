"""Planted-habit diagnostic: a learner checkpoint against the styled opponent pack.

  python -m eval.history_habit_eval CKPT --pack BASE --matches N --seed S --out OUT.npz
      [--view full|round|oracle] [--axis lead_single --strength 1.0] [--envs 16]

Runs the training collector itself (``HistoryCollector`` with the pack's seat
assignment, the view's streams and the style input), so evaluation reads
exactly what training read: one learner team (two seats) per match, both other
seats the pack at the match's z. The learner samples from its own softmax
(temperature 1, as in collection); the pack samples its tilted distribution.
Nothing is trained. Writes one record per finished round of every match that
finished: env, match, round, z, learner team, the team's round return
(``seat_return``, what PPO optimizes) and the signed level gain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import gd
import numpy as np
import torch

from train.history_habit import HABIT_VIEWS, STYLE_AXES, HabitPack, RoundEventStore
from train.history_model import load_history_checkpoint
from train.history_rollout import HistoryCollector, MatchEventStore, SequenceRolloutBuffer


def evaluate(checkpoint: str | Path, pack: str | Path, view: str, axis: str, strength: float,
             matches: int, envs: int, seed: int, steps_per_chunk: int = 256) -> dict:
    actor, _, payload = load_history_checkpoint(checkpoint, "cpu")
    actor.eval()
    if view == "oracle" and not actor.config.style_input:
        raise ValueError("the oracle view needs a style-input checkpoint")
    if actor.config.style_input and view != "oracle":
        raise ValueError("a style-input checkpoint is evaluated in the oracle view")
    habit = HabitPack(actor, "eval", seed + 17, pack, axis, strength, "cpu")
    teams: dict[tuple[int, int], tuple[int, int]] = {}

    def seats(env: int, match: int) -> list[int]:
        assignment = habit.assignment(env, match)
        teams[(int(env), int(match))] = (assignment.index(0) % 2, habit.style(env, match))
        return assignment

    env = gd.VecEnv(num_envs=envs, num_threads=1, seed=seed, log_public_actions=True,
                    log_env_limit=envs)
    store = RoundEventStore() if view == "round" else MatchEventStore()
    generator = torch.Generator().manual_seed(seed)
    collector = HistoryCollector(env, actor, store, SequenceRolloutBuffer(), generator, "cpu",
                                 seat_policy=seats, resolve_policy=habit.resolve, kv_cache=True,
                                 round_streams=view == "round",
                                 row_style=habit.style if view == "oracle" else None)
    rounds: list[tuple] = []
    finished: set[tuple[int, int]] = set()
    started = time.time()
    with torch.inference_mode():
        while len(finished) < matches:
            collector.buffer = SequenceRolloutBuffer()    # nothing is trained
            collector.collect(steps_per_chunk)
            for r in collector.results:
                key = (int(r.env_id), int(r.match_id))
                team, z = teams[key]
                rounds.append((*key, int(r.round_index), z, team, float(r.seat_return[team]),
                               int(r.gain) if int(r.winning_team) == team else -int(r.gain)))
                if r.match_winner >= 0:
                    finished.add(key)
            store.prune(set())
            habit.prune(collector.assignments)
    kept = [r for r in rounds if (r[0], r[1]) in finished]
    columns = ("env", "match", "round", "z", "team", "ret", "gain")
    record = {name: np.asarray([r[i] for r in kept]) for i, name in enumerate(columns)}
    meta = dict(checkpoint=str(checkpoint), lineage=payload.get("lineage"),
                update=payload.get("progress", {}).get("updates"), view=view,
                pack=habit.pack, seed=seed, envs=envs, matches=len(finished),
                rounds=len(kept), seconds=time.time() - started)
    return dict(record, meta=meta)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--pack", required=True)
    parser.add_argument("--view", choices=HABIT_VIEWS, default=None,
                        help="default: the checkpoint's recorded habit view, else full")
    parser.add_argument("--axis", choices=STYLE_AXES, default="lead_single")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--matches", type=int, required=True)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    view = args.view
    if view is None:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False).get("habit")
        view = saved["view"] if saved else "full"
    result = evaluate(args.checkpoint, args.pack, view, args.axis, args.strength,
                      args.matches, args.envs, args.seed)
    meta = result.pop("meta")
    np.savez_compressed(args.out, meta=json.dumps(meta), **result)
    ret = result["ret"]
    print(json.dumps({**{k: meta[k] for k in ("view", "matches", "rounds", "seconds")},
                      "mean_return": float(ret.mean()) if len(ret) else None,
                      "mean_gain": float(result["gain"].mean()) if len(ret) else None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
