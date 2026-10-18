"""Frozen endpoint FULL/ROUND complete-match evaluation. Never constructs a trainer."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import gd
import numpy as np
import torch

from train.history_habit import ROUND_KEY_STRIDE, RoundEventStore
from train.history_model import DecisionInputs, load_history_checkpoint
from train.history_population import weights_digest
from train.history_rollout import HistoryCollector, MatchEventStore, SequenceRolloutBuffer


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class MatchSampler:
    """Independent seat streams per predeclared match seed; no batch-order RNG."""
    def __init__(self, seeds):
        self.rng = [[np.random.default_rng(np.random.SeedSequence([int(s), seat, 161026]))
                     for seat in range(4)] for s in seeds]

    def draw(self, envs, seats):
        return np.asarray([self.rng[int(e)][int(s)].random() for e, s in zip(envs, seats)])


def inverse_cdf(table, uniform):
    probabilities = table.double().exp()
    probabilities /= probabilities.sum(1, keepdim=True)
    cdf = probabilities.cumsum(1)
    cdf[:, -1] = 1.
    choice = (cdf < uniform[:, None]).sum(1)
    chosen = table.gather(1, choice[:, None])[:, 0]
    if not torch.isfinite(chosen).all():
        raise FloatingPointError("nonfinite selected action probability")
    return choice, chosen


class EvalCollector(HistoryCollector):
    """Reuse public-history/cache inference; skip finished slots and all replay storage."""
    def __init__(self, *args, seeds, **kwargs):
        super().__init__(*args, **kwargs)
        self.done = np.zeros(len(seeds), bool)
        self.sampler = MatchSampler(seeds)
        self.first_round = [hashlib.sha256() for _ in seeds]
        self.play_decisions = np.zeros(len(seeds), np.int64)

    def _drain(self, stats):
        super()._drain(stats)
        for r in self.results:
            if int(r.match_id) == 0 and r.match_winner >= 0:
                self.done[int(r.env_id)] = True

    def _plan(self, r):
        r.phase[self.done[r.env_id]] = -1
        if self.round_streams and 0 in self.caches:
            self.caches[0].prune({(int(e), int(m)*ROUND_KEY_STRIDE+int(k))
                                 for e, m, k in zip(r.env_id, r.match_id, r.round_index)
                                 if not self.done[e]})
        return super()._plan(r)

    def _infer_group(self, group, fields, r, stats, mark):
        actor = self.actor if group.identity == 0 else self.resolve_policy(group.identity)
        stream_batch, encoded = self._encode_streams(group, actor, fields[1], r.prefix)
        inputs = DecisionInputs(streams=stream_batch, match_index=fields[0], prefix=fields[1],
            obs=fields[2], seat=fields[3], cand=fields[4], offsets=fields[5],
            candidate_rows=fields[6], one_decision_per_stream=group.one_decision_per_stream)
        table = actor._log_prob_table(inputs, encoded, group.max_candidates)
        uniform = torch.as_tensor(self.sampler.draw(r.env_id[group.rows], r.seat[group.rows]),
                                  device=self.device)
        choice, logp = inverse_cdf(table, uniform)
        self._count_call(stats, "learner" if group.identity == 0 else "snapshot", [group])
        if group.identity != 0:
            return choice, logp.float()
        return choice, logp.float(), logp.float(), choice.new_zeros(()), logp.new_zeros(())

    def _store(self, r, stats):
        if ((r.choices < 0) | (r.choices >= r.counts)).any():
            raise ValueError("invalid legal action index")
        for e, k, s, c, n in zip(r.env_id, r.round_index, r.seat, r.choices, r.counts):
            if not self.done[e]:
                self.play_decisions[e] += 1
                if k == 0:
                    self.first_round[e].update(np.asarray([s, c, n], dtype="<i8").tobytes())


def run_wave(actor, opponents, seeds, device, view, max_steps):
    count = len(seeds)
    ids = sorted(opponents)
    teams = np.arange(count) % 2
    opponent_ids = np.asarray([ids[(e//2) % len(ids)] for e in range(count)])
    def assignment(e, m):
        return [0 if s % 2 == teams[e] else int(opponent_ids[e]) for s in range(4)]
    env = gd.VecEnv(num_envs=count, num_threads=1, seed=71,
                    log_public_actions=True, log_env_limit=count)
    env.reset(match_seeds=seeds)
    store = RoundEventStore() if view == "round" else MatchEventStore()
    collector = EvalCollector(env, actor, store, SequenceRolloutBuffer(), device=device,
        seat_policy=assignment, resolve_policy=opponents.__getitem__, kv_cache=True,
        round_streams=view == "round", seeds=seeds)
    collector.started = True  # preserve explicit reset(match_seeds); no collector reset
    returns = np.zeros(count)
    gains = np.zeros(count)
    rounds = np.zeros(count, np.int64)
    records = {}
    started = time.monotonic()
    with torch.inference_mode():
        for step in range(max_steps):
            collector.collect(1)
            for r in collector.results:
                e = int(r.env_id)
                if e in records or int(r.match_id) != 0:
                    continue
                returns[e] += float(r.seat_return[teams[e]])
                gains[e] += int(r.gain) * (1 if r.winning_team == teams[e] else -1)
                rounds[e] += 1
                if r.match_winner >= 0:
                    collector.done[e] = True
                    records[e] = dict(env=e, seed=int(seeds[e]), team=int(teams[e]),
                        opponent=int(opponent_ids[e]), win=int(r.match_winner == teams[e]),
                        rounds=int(rounds[e]), total_return=float(returns[e]),
                        total_gain=float(gains[e]), decisions=int(collector.play_decisions[e]),
                        first_round_sha=collector.first_round[e].hexdigest())
            if collector.done.all():
                return dict(records=[records[e] for e in range(count)], steps=step+1,
                            seconds=time.monotonic()-started)
            if step % 128 == 0:
                store.prune(set())
            if step and step % 1024 == 0:
                print(json.dumps(dict(view=view, step=step, finished=int(collector.done.sum()),
                                      seconds=time.monotonic()-started)), flush=True)
    raise RuntimeError(f"incomplete wave: {collector.done.sum()}/{count}; no censoring allowed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--norm16", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--waves", type=int, default=16)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=16102671)
    parser.add_argument("--wave-start", type=int, default=0)
    parser.add_argument("--wave-count", type=int)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    assert args.envs % 8 == 0 and args.waves > 0
    torch.set_num_threads(1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    actors = {}
    for arm in ("control", "norm16"):
        actor, critic, payload = load_history_checkpoint(getattr(args, arm), "cpu")
        actors[arm] = actor.eval().requires_grad_(False).to(args.device)
        del critic, payload
    _, critic, parent = load_history_checkpoint(args.parent, "cpu")
    del critic
    pool = parent["population"]["snapshots"]
    eligible = sorted(pool)[-4:]
    opponents = {}
    for i in eligible:
        record = pool[i]
        assert weights_digest(record["model"]) == record["metadata"]["sha256"]
        actor = copy.deepcopy(actors["control"])
        missing, unexpected = actor.load_state_dict(record["model"], strict=False)
        assert not unexpected and set(missing) <= actor.auxiliary_parameter_names()
        opponents[i] = actor.eval().requires_grad_(False)
    meta = dict(endpoints={a: sha(getattr(args, a)) for a in actors}, parent=sha(args.parent),
        opponents={i: pool[i]["metadata"] for i in eligible}, waves=args.waves, envs=args.envs,
        seed=args.seed, matches_per_condition=args.waves*args.envs,
        model_source_sha=sha(__import__("train.history_model", fromlist=["x"]).__file__),
        runner_sha=sha(__file__), torch=torch.__version__, device=args.device,
        pairing="explicit match seeds duplicated with team swap; independent seat RNG streams",
        sampling="temperature 1 inverse CDF; fixed tribute heuristic; no optimizer or training")
    (output/"metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    seeds = np.random.default_rng(args.seed).integers(1, 2**63-1,
                                                    size=(args.waves, args.envs//2))
    with (output/"waves.jsonl").open("w") as stream:
        stop = args.waves if args.wave_count is None else args.wave_start + args.wave_count
        assert 0 <= args.wave_start < stop <= args.waves
        for wave in range(args.wave_start, stop):
            base_seeds = seeds[wave]
            schedule = np.repeat(base_seeds, 2).tolist()
            first_round = {}
            for arm, view in (("control", "full"), ("control", "round"),
                              ("norm16", "full"), ("norm16", "round")):
                result = run_wave(actors[arm], opponents, schedule, args.device,
                                  view, args.max_steps)
                result.update(wave=wave, arm=arm, view=view)
                hashes = [r["first_round_sha"] for r in result["records"]]
                if view == "full":
                    first_round[arm] = hashes
                else:
                    assert hashes == first_round[arm], "first-round FULL/ROUND action mismatch"
                stream.write(json.dumps(result)+"\n")
                stream.flush()
                print(json.dumps({k:v for k,v in result.items() if k != "records"}), flush=True)
    (output/"complete.json").write_text(json.dumps(dict(complete=True))+"\n")


if __name__ == "__main__":
    main()
