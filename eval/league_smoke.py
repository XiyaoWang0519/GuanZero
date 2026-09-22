"""League smoke run: a frozen learner against `train.league.League` on CPU.

Also the reference driver for the `OpponentSource` call order that the PPO
loop (B5) follows; see `drive`. Example::

    .venv/bin/python -m eval.league_smoke \\
        --learner .work/runpod/artifacts/pilot/final.pt --matches 400

No training happens here. The learner plays greedy Q (heuristic tribute); the
league's opponents are the default smoke pool unless `--pool` names a file.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from typing import Callable

import numpy as np

import gd
from train.league import League, LeagueConfig, TorchModel
from train.opponents import OpponentRows

LearnerAct = Callable[[object, np.ndarray], np.ndarray]


def opponent_rows(batch, index: np.ndarray) -> OpponentRows:
    """The `OpponentRows` of batch rows `index`. Read after on_match_start."""
    offsets = np.asarray(batch.offsets, np.int64)
    starts, sizes = offsets[index], offsets[index + 1] - offsets[index]
    sub = np.zeros(len(index) + 1, np.int64)
    np.cumsum(sizes, out=sub[1:])
    flat = np.repeat(starts - sub[:-1], sizes) + np.arange(sub[-1])
    return OpponentRows(
        obs=np.asarray(batch.obs)[index], cand=np.asarray(batch.cand)[flat],
        offsets=sub.astype(np.int32), env_id=np.asarray(batch.env_id)[index],
        seat=np.asarray(batch.seat)[index], phase=np.asarray(batch.phase)[index],
        match_id=np.asarray(batch.match_id)[index],
        greedy_choice=np.asarray(batch.greedy_choice)[index],
        styled_choice=np.asarray(batch.styled_choice)[index])


def drive(env, league, learner_team: np.ndarray, learner_act: LearnerAct, *,
          matches: int | None = None, steps: int | None = None,
          on_step: Callable[[object, np.ndarray, np.ndarray], None] | None = None,
          until: Callable[[], bool] | None = None) -> int:
    """Play until `matches` matches finished, `steps` steps or `until()`;
    returns the number of finished matches.

    Call order per step: pending(); drain rounds and report finished matches
    (on_match_end); on_match_start for every environment whose match_id
    changed; only then build OpponentRows (styled_choice is refreshed by the
    restyle) and act; step. `env` must be reset before the call.
    """
    team = np.asarray(learner_team, np.int64)
    league.bind(env, team)
    league.on_match_start(np.arange(len(team)))
    last = np.zeros(len(team), np.int64)
    done = step = 0
    while ((matches is None or done < matches) and (steps is None or step < steps)
           and (until is None or not until())):
        batch = env.pending()
        ended, won = [], []
        for result in env.drain_finished_rounds():
            if result.match_winner >= 0:
                e = int(result.env_id)
                ended.append(e)
                won.append(int(result.match_winner) == int(team[e]))
        if ended:
            league.on_match_end(np.asarray(ended), np.asarray(won))
            done += len(ended)
        env_id = np.asarray(batch.env_id, np.int64)
        match_id = np.asarray(batch.match_id, np.int64)
        changed = match_id != last[env_id]
        if changed.any():
            league.on_match_start(env_id[changed])
            last[env_id[changed]] = match_id[changed]
        learner = np.asarray(batch.seat) % 2 == team[env_id]
        choice = np.empty(batch.rows, np.int32)
        opponent = np.flatnonzero(~learner)
        if len(opponent):
            choice[opponent] = league.act(opponent_rows(batch, opponent))
        mine = np.flatnonzero(learner)
        if len(mine):
            choice[mine] = learner_act(batch, mine)
        if on_step is not None:
            on_step(batch, learner, choice)
        env.step(choice)
        step += 1
    return done


class FirstMatches:
    """Delegating `OpponentSource` that tallies each environment's first
    `per_env` matches only. Counting whichever matches finish first would
    favour short, lopsided matches; a fixed quota per environment does not."""

    def __init__(self, league: League, num_envs: int, per_env: int) -> None:
        self.league, self.per_env = league, per_env
        self.finished = np.zeros(num_envs, np.int64)
        self.opened: list[list[str]] = [[] for _ in range(num_envs)]
        self.tally: dict[str, list[int]] = {e.name: [0, 0] for e in league.entries}

    def bind(self, env, learner_team: np.ndarray) -> None:
        self.league.bind(env, learner_team)

    def on_match_start(self, env_ids: np.ndarray) -> None:
        self.league.on_match_start(env_ids)
        for e in np.asarray(env_ids).reshape(-1):
            self.opened[int(e)].append(self.league.assigned[int(e)].name)

    def act(self, rows: OpponentRows) -> np.ndarray:
        return self.league.act(rows)

    def on_match_end(self, env_ids: np.ndarray, learner_won: np.ndarray) -> None:
        self.league.on_match_end(env_ids, learner_won)
        for e, won in zip(np.asarray(env_ids).reshape(-1), np.asarray(learner_won).reshape(-1)):
            e = int(e)
            if self.finished[e] < self.per_env:
                row = self.tally.setdefault(self.opened[e][self.finished[e]], [0, 0])
                row[0] += 1
                row[1] += int(bool(won))
            self.finished[e] += 1

    def complete(self) -> bool:
        return bool((self.finished >= self.per_env).all())


def model_learner(model: TorchModel) -> LearnerAct:
    """Greedy network learner on play rows, heuristic tribute elsewhere."""
    play = int(gd.Phase.Play)

    def act(batch, index: np.ndarray) -> np.ndarray:
        choice = np.asarray(batch.greedy_choice)[index].astype(np.int32)
        phase = np.asarray(batch.phase)[index]
        rows = index[phase == play] if model.heuristic_tribute else index
        if len(rows):
            sub = opponent_rows(batch, rows)
            local = model.choose(sub.obs, sub.cand, sub.offsets.astype(np.int64), sub.phase)
            choice[np.searchsorted(index, rows)] = local
        return choice
    return act


def default_pool(learner: str) -> list[dict]:
    return [{"spec": "greedy"},
            {"spec": "styled:bomb-happy"}, {"spec": "styled:bomb-shy"},
            {"spec": "styled:high-lead"}, {"spec": "styled:low-lead"},
            {"spec": "sampled-style"},
            {"spec": f"sample=1:{learner}", "name": "learner/T=1"}]


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return float("nan"), float("nan")
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--learner", required=True)
    parser.add_argument("--pool", help="pool JSON file; default: bots plus the learner at T=1")
    parser.add_argument("--matches", type=int, default=400)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--uniform-mix", type=float, default=1.0,
                        help="1.0 spreads games evenly for per-entry estimates")
    parser.add_argument("--json", help="write the per-entry table here")
    args = parser.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)
    if args.pool:
        league = League.from_file(args.pool, seed=args.seed, uniform_mix=args.uniform_mix)
    else:
        league = League(default_pool(args.learner),
                        LeagueConfig(seed=args.seed, uniform_mix=args.uniform_mix))
    learner = TorchModel(args.learner, "cpu", seed=args.seed)
    env = gd.VecEnv(num_envs=args.envs, num_threads=args.threads, seed=args.seed)
    env.reset()
    team = np.arange(args.envs, dtype=np.int64) % 2
    per_env = max(1, math.ceil(args.matches / args.envs))
    source = FirstMatches(league, args.envs, per_env)
    started = time.monotonic()
    drive(env, source, team, model_learner(learner), until=source.complete)
    elapsed = time.monotonic() - started
    stats = league.stats()
    table = []
    for name, (games, wins) in source.tally.items():
        low, high = wilson(wins, games)
        table.append({"entry": name, "games": games, "learner_wins": wins,
                      "learner_win_rate": wins / games if games else None,
                      "wilson95": [low, high],
                      "weight": stats.get(f"league/{name}/weight")})
    done = per_env * args.envs
    print(f"{done} matches ({per_env} per env) in {elapsed:.0f} s, learner {learner.name}")
    print(f"{'entry':<24}{'games':>7}{'wins':>7}{'rate':>8}  wilson95")
    for row in table:
        rate = row["learner_win_rate"]
        print(f"{row['entry']:<24}{row['games']:>7}{row['learner_wins']:>7}"
              f"{(rate if rate is not None else float('nan')):>8.3f}  "
              f"[{row['wilson95'][0]:.3f}, {row['wilson95'][1]:.3f}]")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"matches": done, "seconds": elapsed, "learner": learner.name,
                       "entries": table}, handle, indent=2)


if __name__ == "__main__":
    main()
