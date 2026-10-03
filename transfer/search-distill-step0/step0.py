"""Search-distillation step 0: label quality of cheap search configs, then their play strength.

Run from the search worktree (.work/search-trigger-2026-10-02/tree), which is only read:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:oracle:. ../../external/danlm-venv/bin/python \\
        ../../search-distill-step0-2026-10-03/step0.py labels --positions 2000 --workers 3
    ... step0.py duel --config C3 --deals 200 --workers 3
    ... step0.py summarize

Part A (``labels``): u9989 greedy self-play on fresh deals (generate_deals seed 20261003,
deal indices 0..). At play decisions with >= 2 legal actions where the blueprint gives its
own greedy choice p < 0.6, positions are sampled (prob 0.5, at most 8 per round). Labels are
computed inline (history stream intact), then the game continues with the baseline action.
Each position gets Q over one fixed candidate set (the greedy choice plus the next 7 by
probability) under

    REF  32 worlds, greedy u9989 rollouts to round end
    C1    8 worlds, rollouts to round end
    C2   32 worlds, rollouts until the current trick ends (lead resets), leaf = critic
    C3    8 worlds, C2 truncation
    REF2 32 worlds to round end, new seeds (deals with index % 6 == 0): the noise ceiling

Every config draws its own worlds (independent RNG streams). Values are the root seat's
``seat_return`` (levels; its team's return). A critic leaf is HistoryCritic(obs, hidden) of
the seat to move at the leaf (the new leader), the value of that seat's team (the trainer's
target, gamma 1); it is negated when the leader is on the other team. REF branches also log
the critic at their first lead reset next to their final return (calibration, and a
same-worlds truncated estimate ``REF_T``).

Part B (``duel``): fresh deals (indices 10000.. of the same generator), both legs,
u9989 + search(config, unsure trigger) vs plain u9989. Greedy plain-vs-plain is exactly
zero on a duplicate deal, so the score is the gain of search.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
import json
import math
import multiprocessing
from pathlib import Path
import random
import time

import numpy as np

HERE = Path(__file__).resolve().parent
CHECKPOINT = str((HERE / "../longrun-u4902-2026-10-02/download/results/segments/main-u4902/latest.pt").resolve())
SEED = 20261003
DUEL_OFFSET = 10000
CONFIGS = {"REF": (32, False), "C1": (8, False), "C2": (32, True), "C3": (8, True)}
THRESHOLD, TOP, MARGIN = 0.6, 8, 0.5
BATCH = 64

_W: dict = {}


# ---- shared machinery -----------------------------------------------------------

def initialize(device: str, two_policies: bool) -> None:
    import torch
    torch.set_num_threads(1)
    import gd
    from eval.policies import load_policy
    from train.history_model import load_history_checkpoint
    base = load_policy(CHECKPOINT, device=device)
    base.actor.causal_sdpa = True
    base.actor.batched_private_attention = True
    _, critic, _ = load_history_checkpoint(Path(CHECKPOINT), "cpu")
    _W.update(gd=gd, base=base, critic=critic.eval(), device=device)
    if two_policies:
        plain = load_policy(CHECKPOINT, device=device)
        plain.actor.causal_sdpa = True
        plain.actor.batched_private_attention = True
        _W["plain"] = plain


def leaf_features(state, seat):
    """Critic input of the seat to move, and +1/-1: same team as the root seat or not."""
    gd = _W["gd"]
    mover = int(state.to_move)
    obs = np.asarray(state.observation(mover), dtype=np.float32)
    hidden = np.stack([np.bincount(state.hand((mover + rel) % 4), minlength=54)
                       for rel in (1, 2, 3)]).astype(np.float32)
    return obs, hidden, (1.0 if mover % 2 == seat % 2 else -1.0)


def critic_values(features):
    import torch
    if not features:
        return np.zeros(0)
    obs = torch.as_tensor(np.stack([f[0] for f in features]))
    hidden = torch.as_tensor(np.stack([f[1] for f in features]))
    sign = np.asarray([f[2] for f in features])
    with torch.inference_mode():
        v = _W["critic"](obs, hidden).float().numpy()
    return sign * v


def rollout_chunk(base, engine, sample, actions, candidates, seat, memo, truncate, probe):
    """eval.search_rollout.rollout_values without a deadline, plus the trick-end leaf.

    Returns per-branch (value, leaf) where value is the real return or, when truncating,
    None with a leaf feature tuple; with ``probe`` a full rollout also returns the leaf
    features of its first lead reset (or None if the round ended inside the first trick).
    """
    gd = _W["gd"]
    from eval.history_policy import apply_and_observe, explicit_passes, resolve_forced_passes
    branches, policies = [], []
    for state in sample:
        branches.append(gd.MatchState.deserialize(state.serialize()))
        policy = copy.copy(base)
        policy.stream = copy.deepcopy(base.stream)
        policies.append(policy)
    n = len(branches)
    values, leaves = [None] * n, [None] * n
    active = list(range(n))
    with explicit_passes(engine, policies):
        for i, candidate in enumerate(candidates):
            apply_and_observe(engine, branches[i], actions[candidate], [policies[i]])
        while active:
            rows, legal_sets, streams, observations, seats, encoded = [], [], [], [], [], []
            offsets, pending, keys, duplicates, unique = [0], [], [], [], {}
            for i in active:
                branch, policy = branches[i], policies[i]
                resolve_forced_passes(engine, branch, [policy])
                if branch.phase == gd.Phase.RoundEnd:
                    values[i] = float(engine.end_round(branch).seat_return[seat])
                    continue
                if branch.top_is_open and leaves[i] is None and (truncate or probe):
                    leaves[i] = leaf_features(branch, seat)
                    if truncate:
                        continue
                legal = engine.legal_actions(branch)
                pending.append(i)
                if len(legal) == 1:
                    apply_and_observe(engine, branch, legal[0], [policy])
                    continue
                acting = int(branch.to_move)
                obs = np.asarray(branch.observation(acting), dtype=np.float32)
                cand = np.asarray([engine.encode_action(a, branch, acting) for a in legal],
                                  dtype=np.float32)
                key = (tuple(a.tobytes() for a in policy.stream.arrays()), obs.tobytes(), cand.tobytes())
                if key in memo:
                    apply_and_observe(engine, branch, legal[memo[key]], [policy])
                    continue
                if key in unique:
                    duplicates.append((i, legal, unique[key]))
                    continue
                unique[key] = len(rows)
                keys.append(key)
                rows.append(i)
                legal_sets.append(legal)
                streams.append(policy.stream)
                seats.append(acting)
                observations.append(obs)
                encoded.extend(cand)
                offsets.append(offsets[-1] + len(legal))
            if rows:
                inputs = base.batch_inputs(streams, np.arange(len(rows)), np.asarray(seats),
                                           np.stack(observations), np.asarray(encoded, dtype=np.float32),
                                           np.asarray(offsets))
                inputs.one_decision_per_stream = True
                choices = base.act(inputs)
                for key, choice in zip(keys, choices):
                    if len(memo) < 4096:
                        memo[key] = int(choice)
                selections = list(zip(rows, legal_sets, choices))
                selections.extend((i, legal, choices[row]) for i, legal, row in duplicates)
                for i, legal, choice in selections:
                    apply_and_observe(engine, branches[i], legal[int(choice)], [policies[i]])
                if base.device.type == "mps":
                    import torch
                    del inputs
                    if torch.mps.driver_allocated_memory() > 4 * 1024**3:
                        torch.mps.empty_cache()
            active = pending
    return values, leaves


def world_matrix(base, engine, state, actions, candidates, seat, rng, worlds, truncate, probe=False):
    """[worlds, candidates] values; with ``probe`` also the critic at each branch's first
    lead reset ([worlds, candidates], NaN where the round ended in the first trick)."""
    k = len(candidates)
    memo = {}
    samples = [state.determinize_uniform(seat, rng.getrandbits(64)) for _ in range(worlds)]
    flat_states = [w for w in samples for _ in candidates]
    flat_cands = candidates * worlds
    values, leaves = [], []
    for start in range(0, len(flat_states), BATCH):
        v, l = rollout_chunk(base, engine, flat_states[start:start + BATCH], actions,
                             flat_cands[start:start + BATCH], seat, memo, truncate, probe)
        values.extend(v)
        leaves.extend(l)
    present = [i for i, l in enumerate(leaves) if l is not None]
    crit = np.full(len(leaves), np.nan)
    crit[present] = critic_values([leaves[i] for i in present])
    if truncate:
        out = np.asarray([v if v is not None else crit[i] for i, v in enumerate(values)], dtype=np.float64)
        return out.reshape(worlds, k), None
    out = np.asarray(values, dtype=np.float64).reshape(worlds, k)
    return out, (crit.reshape(worlds, k) if probe else None)


def candidate_set(base, engine, state, actions):
    import torch
    with torch.inference_mode():
        log_probs = base.actor.candidate_log_probs(
            base.decision_inputs(engine, state, actions)).float().cpu().numpy()
    return log_probs


def decide(means, margin=MARGIN):
    """Index into candidates of the action the search plays (0 = baseline)."""
    winner = int(np.argmax(means))
    return winner if winner and means[winner] >= means[0] + margin else 0


def world_rng(*parts) -> random.Random:
    return random.Random(hash_parts(parts))


def hash_parts(parts) -> int:
    import hashlib
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little")


def r4(a):
    return np.round(np.asarray(a, dtype=np.float64), 4).tolist()


# ---- Part A ---------------------------------------------------------------------

def label_deal(index: int) -> dict:
    gd = _W["gd"]
    from eval.duplicate import generate_deals
    from eval.history_policy import apply_and_observe, explicit_passes, resolve_forced_passes
    base = _W["base"]
    deal = generate_deals(index + 1, SEED)[index]
    engine = gd.Engine(gd.RuleConfig.house())
    state = gd.MatchState()
    engine.set_deal(state, deal)
    base.start_match()
    pick = random.Random(hash_parts(("pick", SEED, index)))
    play_rng = random.Random(SEED + index)
    positions, decisions, unsure, started = [], 0, 0, time.perf_counter()
    with explicit_passes(engine, [base]):
        while state.phase != gd.Phase.RoundEnd:
            resolve_forced_passes(engine, state, [base])
            if state.phase == gd.Phase.RoundEnd:
                break
            actions = engine.legal_actions(state)
            baseline = base.select(engine, state, actions, play_rng)
            decisions += 1
            if state.phase == gd.Phase.Play and len(actions) >= 2:
                log_probs = candidate_set(base, engine, state, actions)
                p = float(math.exp(log_probs[baseline]))
                if p < THRESHOLD:
                    unsure += 1
                    if len(positions) < 8 and pick.random() < 0.5:
                        positions.append(label_position(engine, state, actions, baseline,
                                                        log_probs, index, len(positions), decisions))
            apply_and_observe(engine, state, actions[baseline], [base])
    result = engine.end_round(state)
    return {"deal": index, "decisions": decisions, "unsure": unsure,
            "seat_return": list(result.seat_return), "positions": positions,
            "wall_seconds": time.perf_counter() - started}


def label_position(engine, state, actions, baseline, log_probs, deal, slot, decision) -> dict:
    base = _W["base"]
    seat = int(state.to_move)
    others = sorted((i for i in range(len(actions)) if i != baseline), key=lambda i: -log_probs[i])
    candidates = [baseline] + others[:TOP - 1]
    counts = [len(state.hand(s)) for s in range(4)]
    record = {"deal": deal, "slot": slot, "decision": decision, "seat": seat,
              "baseline_probability": float(math.exp(log_probs[baseline])),
              "legal": len(actions), "candidates": len(candidates),
              "cand_prob": r4(np.exp(log_probs[candidates])), "own_cards": counts[seat],
              "min_cards": min(counts), "lead": bool(state.top_is_open),
              "events": int(base.events_seen), "configs": {}}
    names = list(CONFIGS) + (["REF2"] if deal % 6 == 0 else [])
    for name in names:
        worlds, truncate = CONFIGS["REF" if name == "REF2" else name]
        rng = world_rng("worlds", SEED, deal, slot, name)
        t0 = time.perf_counter()
        values, probe = world_matrix(base, engine, state, actions, candidates, seat, rng,
                                     worlds, truncate, probe=(name == "REF"))
        record["configs"][name] = {"values": r4(values), "seconds": time.perf_counter() - t0}
        if probe is not None:
            # Same worlds, truncated: critic at the first lead reset, else the real return.
            record["configs"]["REF_T"] = {"values": r4(np.where(np.isnan(probe), values, probe)),
                                          "seconds": None}
            record["calibration"] = {"critic": r4(probe), "ended_first_trick": int(np.isnan(probe).sum())}
    return record


def run_labels(args) -> None:
    out = HERE / "partA"
    out.mkdir(exist_ok=True)
    raw = out / "raw.jsonl"
    rows = [json.loads(l) for l in raw.read_text().splitlines() if l.strip()] if raw.exists() else []
    done = {r["deal"] for r in rows}
    total = sum(len(r["positions"]) for r in rows)
    start, pending, next_deal = time.perf_counter(), set(), 0
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize, initargs=(args.device, False)) as pool, \
            raw.open("a") as stream:
        while True:
            while total < args.positions and len(pending) < args.workers:
                while next_deal in done:
                    next_deal += 1
                pending.add(pool.submit(label_deal, next_deal))
                next_deal += 1
            if not pending:
                break
            finished, pending = wait(pending, timeout=120, return_when=FIRST_COMPLETED)
            for future in finished:
                row = future.result()
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                stream.flush()
                total += len(row["positions"])
            print(json.dumps({"elapsed": round(time.perf_counter() - start), "positions": total,
                              "pending": len(pending)}), flush=True)


# ---- Part B ---------------------------------------------------------------------

class StepSearch:
    """u9989 + search at unsure decisions (p(greedy) < 0.6), one fixed config."""

    needs_history = True

    def __init__(self, base, config: str, seed: int):
        self.blueprint, self.config = base, config
        self.name = f"step0-search:{config}"
        self.action_mode = "canonical"
        self.rng = random.Random(seed)
        self.calls = self.triggered = self.overrides = 0
        self.seconds: list[float] = []

    def start_match(self, match_id: int = -1) -> None:
        self.blueprint.start_match(match_id)

    def observe(self, event) -> None:
        self.blueprint.observe(event)

    @property
    def events_seen(self) -> int:
        return self.blueprint.events_seen

    def select(self, engine, state, actions, rng) -> int:
        gd = _W["gd"]
        self.calls += 1
        baseline = self.blueprint.select(engine, state, actions, rng)
        if state.phase != gd.Phase.Play or len(actions) < 2:
            return baseline
        log_probs = candidate_set(self.blueprint, engine, state, actions)
        if math.exp(log_probs[baseline]) >= THRESHOLD:
            return baseline
        self.triggered += 1
        t0 = time.perf_counter()
        others = sorted((i for i in range(len(actions)) if i != baseline), key=lambda i: -log_probs[i])
        candidates = [baseline] + others[:TOP - 1]
        worlds, truncate = CONFIGS[self.config]
        values, _ = world_matrix(self.blueprint, engine, state, actions, candidates,
                                 int(state.to_move), self.rng, worlds, truncate)
        choice = decide(values.mean(0))
        self.seconds.append(time.perf_counter() - t0)
        if choice:
            self.overrides += 1
        return candidates[choice]


def duel_deal(job) -> dict:
    index, configs = job
    gd = _W["gd"]
    from eval.duplicate import generate_deals, play_duplicate
    deal = generate_deals(DUEL_OFFSET + index + 1, SEED)[DUEL_OFFSET + index]
    out = {"deal": index, "arms": {}}
    for config in configs:
        policy = StepSearch(_W["base"], config, hash_parts(("duel", SEED, index, config)))
        t0 = time.perf_counter()
        score = play_duplicate(deal, policy, _W["plain"], seed=SEED + DUEL_OFFSET + index)
        out["arms"][config] = {"net": score.levels_per_round,
                               "legs": [score.first.seat_return[0], score.swapped.seat_return[0]],
                               "calls": policy.calls, "triggered": policy.triggered,
                               "overrides": policy.overrides, "search_seconds": r4(policy.seconds),
                               "wall_seconds": time.perf_counter() - t0}
    return out


def run_duel(args) -> None:
    out = HERE / "partB"
    out.mkdir(exist_ok=True)
    raw = out / "raw.jsonl"
    configs = args.config
    done: dict[int, set] = {}
    if raw.exists():
        for line in raw.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.setdefault(row["deal"], set()).update(row["arms"])
    jobs = [(i, [c for c in configs if c not in done.get(i, set())]) for i in range(args.deals)]
    jobs = [j for j in jobs if j[1]]
    start, pending, queue = time.perf_counter(), set(), iter(jobs)
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize, initargs=(args.device, True)) as pool, \
            raw.open("a") as stream:
        while True:
            while len(pending) < args.workers:
                job = next(queue, None)
                if job is None:
                    break
                pending.add(pool.submit(duel_deal, job))
            if not pending:
                break
            finished, pending = wait(pending, timeout=120, return_when=FIRST_COMPLETED)
            for future in finished:
                stream.write(json.dumps(future.result(), separators=(",", ":")) + "\n")
                stream.flush()
            print(json.dumps({"elapsed": round(time.perf_counter() - start),
                              "lines": len(raw.read_text().splitlines())}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("labels")
    a.add_argument("--positions", type=int, default=2000)
    b = sub.add_parser("duel")
    b.add_argument("--config", nargs="+", required=True, choices=list(CONFIGS))
    b.add_argument("--deals", type=int, default=200)
    for p in (a, b):
        p.add_argument("--workers", type=int, default=3)
        p.add_argument("--device", default="mps")
    args = parser.parse_args()
    if args.workers > 3:
        parser.error("at most 3 workers on this Mac")
    {"labels": run_labels, "duel": run_duel}[args.command](args)


if __name__ == "__main__":
    main()
