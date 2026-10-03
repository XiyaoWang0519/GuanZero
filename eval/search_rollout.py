"""Batched independent search branches, with actor inference on its own device."""
from __future__ import annotations

import copy
import time

import gd
import numpy as np

from .history_policy import HistoryPolicy, apply_and_observe, explicit_passes, resolve_forced_passes


def rollout_values(search, engine, sample, actions, candidates, seat, deadline, memo):
    """Simulate a chunk of candidate branches; unfinished branches have no value."""
    base = search.blueprint
    if type(base) is not HistoryPolicy or base.sample:
        raise TypeError("batched search requires a greedy HistoryPolicy")
    branches, policies = [], []
    for state, candidate in zip(sample, candidates):
        branch = gd.MatchState.deserialize(state.serialize())
        policy = copy.copy(base)
        policy.stream = copy.deepcopy(base.stream)
        branches.append(branch)
        policies.append(policy)
    values = [None] * len(branches)
    steps = [0] * len(branches)
    active = list(range(len(branches)))
    with explicit_passes(engine, policies):
        for i, candidate in enumerate(candidates):
            apply_and_observe(engine, branches[i], actions[candidate], [policies[i]])
        while active and time.perf_counter() < deadline:
            rows, legal_sets, streams, observations, seats, encoded_actions = [], [], [], [], [], []
            offsets = [0]
            pending = []
            keys, duplicates, unique = [], [], {}
            for i in active:
                branch, policy = branches[i], policies[i]
                resolve_forced_passes(engine, branch, [policy])
                if branch.phase == gd.Phase.RoundEnd:
                    values[i] = float(engine.end_round(branch).seat_return[seat])
                    continue
                if search.config.max_rollout_steps and steps[i] >= search.config.max_rollout_steps:
                    continue
                policy.verify_stream(branch)
                legal = engine.legal_actions(branch)
                pending.append(i)
                if len(legal) == 1:
                    apply_and_observe(engine, branch, legal[0], [policy])
                    steps[i] += 1
                    continue
                acting = int(branch.to_move)
                obs = np.asarray(branch.observation(acting), dtype=np.float32)
                cand = np.asarray([engine.encode_action(a, branch, acting) for a in legal], dtype=np.float32)
                key = (tuple(a.tobytes() for a in policy.stream.arrays()), obs.tobytes(), cand.tobytes())
                if key in memo:
                    apply_and_observe(engine, branch, legal[memo[key]], [policy])
                    steps[i] += 1
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
                encoded_actions.extend(cand)
                offsets.append(offsets[-1] + len(legal))
            if rows:
                inputs = base.batch_inputs(streams, np.arange(len(rows)), np.asarray(seats),
                    np.stack(observations), np.asarray(encoded_actions, dtype=np.float32), np.asarray(offsets))
                inputs.one_decision_per_stream = True
                choices = base.act(inputs)
                if time.perf_counter() >= deadline:
                    break
                for key, choice in zip(keys, choices):
                    if len(memo) < 4096:
                        memo[key] = int(choice)
                selections = list(zip(rows, legal_sets, choices))
                selections.extend((i, legal, choices[row]) for i, legal, row in duplicates)
                for i, legal, choice in selections:
                    apply_and_observe(engine, branches[i], legal[int(choice)], [policies[i]])
                    steps[i] += 1
                if base.device.type == 'mps':
                    # Variable sequence/candidate shapes otherwise retain large
                    # transient Metal allocations over a long evaluation.
                    import torch
                    del inputs
                    if torch.mps.driver_allocated_memory() > 4 * 1024**3:
                        torch.mps.empty_cache()
            active = pending
    return values


def batched_world_values(search, engine, state, actions, candidates, seat, rng, deadline):
    """Only whole worlds (every root candidate completed) contribute to means."""
    count = len(candidates)
    totals = np.zeros(count, dtype=np.float64)
    completed = 0
    memo = {}  # exact visible input -> greedy action, scoped to frozen search
    worlds_per_wave = max(1, search.config.rollout_batch_size // count)
    while completed < search.config.max_worlds and time.perf_counter() < deadline:
        n = min(worlds_per_wave, search.config.max_worlds - completed)
        worlds = [state.determinize_uniform(seat, rng.getrandbits(64)) for _ in range(n)]
        samples = [world for world in worlds for _ in candidates]
        choices = candidates * n
        values = []
        for start in range(0, len(samples), search.config.rollout_batch_size):
            end = start + search.config.rollout_batch_size
            if time.perf_counter() >= deadline:
                break
            values.extend(rollout_values(search, engine, samples[start:end], actions,
                                         choices[start:end], seat, deadline, memo))
        # Retain only a completed prefix of worlds, as the scalar reference does.
        done = 0
        for w in range(n):
            row = values[w * count:(w + 1) * count]
            if len(row) != count or any(v is None for v in row):
                break
            totals += row
            done += 1
        completed += done
        if done < n:
            break
    return totals.tolist(), completed
