"""Evaluation-only VecEnv runner; scalar duplicate/arena remain the reference.

Batching changes stochastic draw order. Fixed seeds and batch configuration
replay, but sampled policies need distributional comparisons with scalar runs.
Deterministic policies retain candidate order/tie breaking (subject to floating
point differences between batch-1 and batch-N inference).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Iterable

import gd
import numpy as np
import torch

from .duplicate import DuplicateScore, RoundScore
from .policies import GreedyPolicy, ModelPolicy, Policy, PrunedPolicy, RandomPolicy, StyledPolicy
from train.model import select_actions
from train.policy import sample_segments, segment_log_softmax

PLAY = int(gd.Phase.Play)


@dataclass(frozen=True)
class EvalConfig:
    backend: str = "batched"
    batch_size: int = 256  # deals (two slots each), or full matches
    engine_threads: int = 1

    def __post_init__(self):
        if self.backend not in ("batched", "scalar"):
            raise ValueError("backend must be batched or scalar")
        if self.batch_size < 1 or self.engine_threads < 1:
            raise ValueError("eval batch size and engine threads must be positive")

    def metadata(self) -> dict:
        return {"backend": self.backend, "batch_size": self.batch_size,
                "engine_threads": self.engine_threads,
                "sampling_note": "batched random draws differ from scalar; compare distributions"}


def add_eval_arguments(parser) -> None:
    parser.add_argument("--eval-backend", choices=("batched", "scalar"), default="batched")
    parser.add_argument("--eval-batch-size", type=int, default=256,
                        help="maximum duplicate deals or full matches per VecEnv wave")
    parser.add_argument("--engine-threads", type=int, default=1, help="VecEnv threads per worker")


def config_from_args(args) -> EvalConfig:
    return EvalConfig(args.eval_backend, args.eval_batch_size, args.engine_threads)


def gather(obs, cand, offsets, index):
    """Compact selected ragged rows without changing candidate order."""
    starts = offsets[index].astype(np.int64)
    sizes = offsets[index + 1].astype(np.int64) - starts
    sub = np.concatenate(([0], np.cumsum(sizes)))
    flat = np.repeat(starts - sub[:-1], sizes) + np.arange(sub[-1])
    return obs[index], cand[flat], sub


class BatchActor:
    def __init__(self, policy: Policy, seed: int):
        # Exact types avoid silently ignoring a custom subclass's select().
        if type(policy) not in (GreedyPolicy, RandomPolicy, StyledPolicy, ModelPolicy, PrunedPolicy):
            raise TypeError(f"unsupported batched policy {type(policy).__name__}; use scalar backend")
        self.policy = policy
        self.rng = np.random.default_rng(seed % (1 << 64))
        self.generator = None
        if isinstance(policy, ModelPolicy):
            self.generator = torch.Generator(device=policy.device).manual_seed(seed % (1 << 63))

    def act(self, batch, index):
        policy = self.policy
        if isinstance(policy, StyledPolicy):
            return np.asarray(batch.styled_choice)[index]
        out = np.asarray(batch.greedy_choice)[index].copy()
        if isinstance(policy, GreedyPolicy):
            return out
        phases = np.asarray(batch.phase)
        rows = index[phases[index] == PLAY] if getattr(policy, "heuristic_tribute", True) else index
        if not len(rows):
            return out
        positions = np.searchsorted(index, rows)
        if isinstance(policy, RandomPolicy):
            offsets = np.asarray(batch.offsets)
            out[positions] = self.rng.integers(offsets[rows + 1] - offsets[rows])
            return out
        obs, cand, off = gather(np.asarray(batch.obs), np.asarray(batch.cand),
                                np.asarray(batch.offsets), rows)
        device = policy.device
        o = torch.from_numpy(obs).to(device)
        c = torch.from_numpy(cand).to(device)
        offsets = torch.from_numpy(off).to(device)
        phase = torch.as_tensor(phases[rows].astype(np.int64), device=device)
        code = PLAY if policy.heuristic_tribute else None
        with torch.inference_mode():
            keep = torch.arange(len(c), device=device)
            pruned_offsets = offsets
            if isinstance(policy, PrunedPolicy):
                # Only rows above top_k need reference inference. Smaller rows
                # keep all candidates, including pass, in their original order.
                big = np.flatnonzero(np.diff(off) > policy.stage_b.config.top_k)
                if len(big):
                    bo, bc, bf = gather(obs, cand, off, big)
                    kept, _ = policy.stage_b.prune(
                        torch.from_numpy(bo).to(device), torch.from_numpy(bc).to(device),
                        torch.from_numpy(bf).to(device), phase[big], code)
                    mask = np.ones(len(cand), bool)
                    local = np.zeros(len(bc), bool)
                    local[kept.cpu().numpy()] = True
                    row_ids = np.repeat(np.arange(len(rows)), np.diff(off))
                    mask[np.isin(row_ids, big)] = local
                    indices = np.flatnonzero(mask)
                    counts = np.bincount(row_ids[indices], minlength=len(rows))
                    keep = torch.as_tensor(indices, device=device)
                    pruned_offsets = torch.as_tensor(np.concatenate(([0], np.cumsum(counts))), device=device)
                scores = policy.stage_b.logits(o, c[keep], pruned_offsets, phase, code)
            else:
                scores = policy.model.score_candidates(o, c, offsets, phase, phase_code=code)
            if scores.shape != (len(keep),) or not torch.isfinite(scores).all():
                raise ValueError("policy must produce one finite score per legal action")
            if policy.margin:
                boundaries = pruned_offsets.cpu().numpy()
                values = scores.cpu().numpy()
                choice = torch.as_tensor([
                    self.rng.choice(np.flatnonzero(values[a:b] >= values[a:b].max() - policy.margin))
                    for a, b in zip(boundaries[:-1], boundaries[1:])], device=device)
            elif isinstance(policy, PrunedPolicy) and policy.sample:
                choice = sample_segments(segment_log_softmax(scores, pruned_offsets),
                                         pruned_offsets, self.generator)
            else:
                choice = select_actions(scores, pruned_offsets)
            choice = keep[pruned_offsets[:-1] + choice] - offsets[:-1]
        out[positions] = choice.cpu().numpy()
        return out


def _waves(items: Iterable, size: int):
    iterator = iter(items)
    while wave := list(islice(iterator, size)):
        yield wave


def _play(env, seats, seed, *, matches=False, max_rounds=1000, max_decisions=2000):
    """Yield (slot, RoundScore, match_winner); ignore restarted/completed slots."""
    policies = []
    assignment = np.empty((len(seats), 4), np.int32)
    for e, lineup in enumerate(seats):
        for seat, policy in enumerate(lineup):
            if not any(policy is p for p in policies):
                policies.append(policy)
            assignment[e, seat] = next(i for i, p in enumerate(policies) if policy is p)
    actors = [BatchActor(p, seed + i) for i, p in enumerate(policies)]
    if any(isinstance(p, StyledPolicy) for p in policies):
        styles = np.tile(np.asarray(gd.StyleParams.neutral().to_array(), np.float32), (len(seats), 4, 1))
        for e, lineup in enumerate(seats):
            for s, p in enumerate(lineup):
                if isinstance(p, StyledPolicy):
                    styles[e, s] = p.style
        env.set_styles(styles)
    done = np.zeros(len(seats), bool)
    decisions = np.zeros(len(seats), np.int64)
    rounds = np.zeros(len(seats), np.int64)
    while not done.all():
        batch = env.pending()
        for result in env.drain_finished_rounds():
            e = result.env_id
            if done[e] or result.match_id != 0:
                continue
            score = RoundScore(tuple(result.order), result.winning_team, result.gain,
                               tuple(result.seat_return), int(decisions[e]))
            decisions[e] = 0
            rounds[e] += 1
            done[e] = not matches or result.match_winner >= 0
            yield e, score, result.match_winner
            if not done[e] and rounds[e] >= max_rounds:
                raise RuntimeError(f"match slot {e} exceeded {max_rounds} rounds")
        if done.all():
            break
        ids, seat = np.asarray(batch.env_id), np.asarray(batch.seat)
        live = ~done[ids]
        if np.any(decisions[ids[live]] >= max_decisions):
            raise RuntimeError(f"round exceeded {max_decisions} decisions")
        choices = np.asarray(batch.greedy_choice).copy()
        owners = assignment[ids, seat]
        for i, actor in enumerate(actors):
            rows = np.flatnonzero(live & (owners == i))
            if len(rows):
                choices[rows] = actor.act(batch, rows)
        decisions[ids[live]] += 1
        env.step(choices)


def play_duplicate_batch(deals: Iterable[gd.DealSpec], team: tuple[Policy, Policy],
                         opponents: tuple[Policy, Policy], seed: int = 0,
                         config: EvalConfig = EvalConfig(), max_decisions: int = 2000) -> list[DuplicateScore]:
    if max_decisions < 1:
        raise ValueError("max_decisions must be positive")
    scores = []
    for wave in _waves(deals, config.batch_size):
        env = gd.VecEnv(2 * len(wave), config.engine_threads, seed % (1 << 64))
        env.reset([deal for deal in wave for _ in range(2)])
        seats = [(team[0], opponents[0], team[1], opponents[1]),
                 (opponents[0], team[0], opponents[1], team[1])] * len(wave)
        results = [None] * len(seats)
        for e, score, _ in _play(env, seats, seed + len(scores), max_decisions=max_decisions):
            results[e] = score
        scores.extend(DuplicateScore(results[i], results[i + 1]) for i in range(0, len(results), 2))
    return scores


def play_matches_batch(agent: Policy, opponent: Policy, indices: Iterable[int], seed: int = 0,
                       max_rounds: int = 1000, config: EvalConfig = EvalConfig()) -> dict:
    from .arena import MATCH_COUNTERS
    if max_rounds < 1:
        raise ValueError("match round bound must be positive")
    totals = dict.fromkeys(MATCH_COUNTERS, 0)
    totals["records"] = []
    for wave in _waves(indices, config.batch_size):
        env = gd.VecEnv(len(wave), config.engine_threads, seed % (1 << 64))
        env.reset(match_seeds=[seed + m for m in wave])
        seats = [(agent, opponent, agent, opponent) if m % 2 == 0 else
                 (opponent, agent, opponent, agent) for m in wave]
        rounds = [0] * len(wave)
        records = [None] * len(wave)
        for e, score, winner in _play(env, seats, seed + wave[0], matches=True, max_rounds=max_rounds):
            team = wave[e] % 2
            rounds[e] += 1
            totals["rounds"] += 1
            totals["bankers"] += score.winning_team == team
            totals["double_wins"] += score.double_win(team)
            totals["opponent_doubles"] += score.double_win(1 - team)
            totals["awarded"] += score.gain if score.winning_team == team else 0
            totals["net"] += score.net_gain(team)
            if winner >= 0:
                won = winner == team
                totals["matches"] += 1
                totals["wins"] += won
                records[e] = {"seed": seed + wave[e], "agent_team": team, "winner": winner,
                              "agent_won": won, "rounds": rounds[e]}
        totals["records"].extend(records)
    return totals
