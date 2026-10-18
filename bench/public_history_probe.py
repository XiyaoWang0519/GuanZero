"""Frozen-policy public-event collection; no trainer, optimizer, or private probe fields."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import gd
import numpy as np
import torch

import bench.qknorm_match_ablation as matches
from train.history_model import load_history_checkpoint, PLAY_PHASE
from train.history_population import weights_digest
from train.history_rollout import _StepRows, MatchEventStore, SequenceRolloutBuffer


class PublicTap:
    """Observer receives public events only, never observations/candidates/forced flags."""
    def __init__(self, env):
        self.env = env
        self.meta = []
        self.actions = []
        self.rounds = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def drain_public_actions(self):
        events = self.env.drain_public_actions()
        for event in events:
            if int(event.match_id) != 0:
                continue
            encoded = np.asarray(event.encoded_action, dtype=np.uint8)
            assert not encoded[146:].any(), "public exchange flags must be redacted"
            self.meta.append((int(event.env_id), int(event.round_index), int(event.step),
                              int(event.seat), int(event.cards_left), int(event.phase)))
            self.actions.append(encoded[:146].copy())
        return events

    def drain_finished_rounds(self):
        results = self.env.drain_finished_rounds()
        for r in results:
            if int(r.match_id) == 0:
                self.rounds.append((int(r.env_id), int(r.round_index), int(r.round_level),
                                    int(r.next_owner), *map(int, r.levels)))
        return results

    def save(self, path):
        np.savez_compressed(path, public_meta=np.asarray(self.meta, np.int32).reshape(-1, 6),
            public_actions=np.asarray(self.actions, np.uint8).reshape(-1, 146),
            public_round_results=np.asarray(self.rounds, np.int32).reshape(-1, 6))


class PublicCollector(matches.EvalCollector):
    last = None

    def __init__(self, env, *args, **kwargs):
        self.tap = PublicTap(env)
        super().__init__(self.tap, *args, **kwargs)
        PublicCollector.last = self

    def _read_rows(self, batch):
        # Actors legitimately use their own obs/candidates to play. Hidden counts are
        # not needed even by the actors; do not read that batch property at all.
        n = int(batch.rows)
        if not n:
            raise RuntimeError("no pending decisions")
        offsets = np.asarray(batch.offsets, np.int64)
        r = _StepRows(env_id=np.asarray(batch.env_id, np.int64),
            match_id=np.asarray(batch.match_id, np.int64),
            round_index=np.asarray(batch.round_index, np.int64),
            seat=np.asarray(batch.seat, np.int64), phase=np.asarray(batch.phase, np.int64),
            offsets=offsets, counts=offsets[1:]-offsets[:-1], obs=np.asarray(batch.obs),
            cand=np.asarray(batch.cand), hidden=np.empty((n, 0), np.uint8),
            choices=np.array(batch.greedy_choice, np.int32, copy=True),
            logp=np.zeros(n, np.float32), behaviour_logp=np.zeros(n, np.float32),
            prefix=np.zeros(n, np.int64), identities=np.zeros(n, np.int64))
        for i in range(n):
            r.identities[i] = self.assignment(r.env_id[i], r.match_id[i])[r.seat[i]]
            r.prefix[i] = self.store.stream(r.env_id[i], r.match_id[i]).prefix
        r.acting = (r.identities == 0) & (r.phase == PLAY_PHASE)
        if self._assignments_changed:
            self._prune_caches()
            self._assignments_changed = False
        return r


def load_players(base):
    endpoints = base / "qknorm-short-ppo-2026-10-16/remote/runs"
    actors = {}
    for arm in ("control", "norm16"):
        a, critic, payload = load_history_checkpoint(endpoints/arm/"endpoint.pt", "cpu")
        actors[arm] = a.eval().requires_grad_(False)
        del critic, payload
    parent_path = base/"qknorm-mechanism-2026-10-16/remote/qk-both/500M-update-094488.pt"
    a, critic, parent = load_history_checkpoint(parent_path, "cpu")
    del a, critic
    opponents = {}
    pool = parent["population"]["snapshots"]
    metadata = {}
    for identity in sorted(pool)[-4:]:
        record = pool[identity]
        assert weights_digest(record["model"]) == record["metadata"]["sha256"]
        a = copy.deepcopy(actors["control"])
        missing, unexpected = a.load_state_dict(record["model"], strict=False)
        assert not unexpected and set(missing) <= a.auxiliary_parameter_names()
        opponents[identity] = a.eval().requires_grad_(False)
        metadata[identity] = record["metadata"]
    return actors, opponents, metadata


def collect_wave(args):
    actors, opponents, metadata = load_players(args.base)
    output = args.output/f"wave-{args.wave:02d}"
    output.mkdir(parents=True, exist_ok=False)
    seeds = np.random.default_rng(16102671).integers(1, 2**63-1, size=(16, 16))
    schedule = np.repeat(seeds[args.wave], 2).tolist()
    original = matches.EvalCollector
    matches.EvalCollector = PublicCollector
    try:
        for arm, actor in actors.items():
            result = matches.run_wave(actor, opponents, schedule, "cpu", "full", 20000)
            PublicCollector.last.tap.save(output/f"{arm}-public.npz")
            previous = args.base/"qknorm-match-ablation-2026-10-16/results"/f"wave-{args.wave:02d}/waves.jsonl"
            reference = next(json.loads(s) for s in previous.read_text().splitlines()
                             if json.loads(s)["arm"] == arm and json.loads(s)["view"] == "full")
            assert result["records"] == reference["records"], "public tap changed match trajectory"
            (output/f"{arm}-matches.json").write_text(json.dumps(result, indent=2)+"\n")
            print(json.dumps(dict(arm=arm, matches=32, public_events=len(PublicCollector.last.tap.meta),
                                  seconds=result["seconds"], prior_match_parity=True)), flush=True)
    finally:
        matches.EvalCollector = original
    (output/"complete.json").write_text(json.dumps(dict(complete=True, opponents=metadata,
        source_sha=matches.sha(__file__), model_sha=matches.sha(__import__("train.history_model", fromlist=["x"]).__file__)))+"\n")


def collect_questions(args):
    _, opponents, metadata = load_players(args.base)
    output = args.output/"questions"
    output.mkdir(parents=True, exist_ok=False)
    seeds = np.random.default_rng(16102677).integers(1, 2**63-1, size=256)
    counts = np.zeros((256, 4, 2, 3), np.int32)
    labels = np.zeros((256, 4, 64), np.int8)
    started = time.monotonic()
    identities = sorted(opponents)
    for q, seed in enumerate(seeds):
        env = gd.VecEnv(num_envs=256, num_threads=1, seed=71,
                        log_public_actions=True, log_env_limit=256)
        env.reset(match_seeds=[int(seed)]*256)
        policy_seeds = np.random.default_rng([16102678, q]).integers(1, 2**63-1, size=64)
        collector = PublicCollector(env, opponents[identities[0]], MatchEventStore(),
            SequenceRolloutBuffer(), device="cpu", kv_cache=True,
            seat_policy=lambda e, m: [identities[e//64]]*4,
            resolve_policy=opponents.__getitem__, seeds=np.tile(policy_seeds, 4).tolist())
        collector.started = True
        with torch.inference_mode():
            collector.collect(1)
        events = env.drain_public_actions()
        assert len(events) == 256
        for event in events:
            # Only the already-played public action is examined, never candidates.
            assert event.round_index == 0 and int(event.phase) == PLAY_PHASE and event.step == 0
            encoded = np.asarray(event.encoded_action)
            action_type = int(encoded[108:121].argmax())
            assert action_type != 0
            category = 0 if action_type == 1 else 1 if action_type in (8, 9, 10) else 2
            e = int(event.env_id)
            i, draw = divmod(e, 64)
            counts[q, i, draw//32, category] += 1
            labels[q, i, draw] = category
        if q % 32 == 31:
            print(json.dumps(dict(questions=q+1, seconds=time.monotonic()-started)), flush=True)
    np.savez_compressed(output/"public_answers.npz", counts=counts, labels=labels)
    (output/"complete.json").write_text(json.dumps(dict(complete=True, questions=256,
        public_answers=int(counts.sum()), calibration_draws=32, heldout_draws=32,
        identical_deal_per_question=True, random_draws_paired_across_opponents=True,
        opponents=metadata, source_sha=matches.sha(__file__), seconds=time.monotonic()-started))+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path(".work"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wave", type=int)
    p.add_argument("--questions", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.questions:
        collect_questions(args)
    else:
        assert args.wave is not None and 0 <= args.wave < 16
        collect_wave(args)


if __name__ == "__main__":
    main()
