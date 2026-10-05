"""Decision-for-decision check of ``eval.v1t.player`` against DanLM's compiled V1T agent.

macOS only (DanLM's binaries). Every round runs in ``eval.danlm.arena``'s
lockstep harness with the COMPILED V1T agent in all four seats: DanLM's
engine referees, ours mirrors, and the pure player hears every mirrored action
through the history protocol. At every decision of the compiled agent
(play, tribute give, tribute back) the pure player decides on our engine's
state first; then the compiled agent decides while its ONNX calls are
recorded. Compared per decision:

* ``state``: the 884-float state part of the model input, bit for bit;
* ``legal``: the set of 80-dim play rows (DanLM's order is not reproduced;
  outputs are row-independent, so only the set matters);
* ``q``: the Q value of every common row, bit for bit (and max abs diff);
* ``choice``: the chosen 80-dim play (tribute: the chosen card).

    PYTHONPATH=python:oracle:. .work/external/danlm-venv/bin/python -m eval.v1t.verify \\
        --deals 400 --seed 7 --tribute-fraction 0.75 --output docs/reports/v1t-verify.json

A round is tagged ``after_declaration`` from the first time the compiled
agent declares a lone wild card at another rank (a DanLM-only play our engine
plays at the level rank): from then on the two engines' tricks differ, so the
later decisions of that round are counted separately, not dropped.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import numpy as np

from eval.v1t import encoder as enc
from eval.v1t.player import V1TModel, V1TPlayer, default_model_path

SAMPLES = 20


class Recorder:
    """Stands in for the compiled agent's ``OnnxEvalModel``; keeps the last call."""

    def __init__(self, model) -> None:
        self.model = model
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def __call__(self, batch):
        out = self.model(batch)
        self.calls.append((batch.detach().cpu().numpy().copy(), out.detach().cpu().numpy().copy()))
        return out

    def eval(self):
        return self


class Verifier:
    def __init__(self, player: V1TPlayer) -> None:
        self.player = player
        self.counts: Counter = Counter()
        self.samples: dict[str, list] = {}
        self.max_q_diff = 0.0
        self.round = None
        self.round_id = ""
        self.declared = False
        self.steps: list[dict] | None = None     # fixture capture of the current round

    def note(self, key: str, detail=None) -> None:
        self.counts[key] += 1
        if detail is not None:
            bucket = self.samples.setdefault(key, [])
            if len(bucket) < SAMPLES:
                bucket.append(f"{self.round_id}: {detail}")

    def compare(self, phase: str, mine, recorder: Recorder, chosen_theirs: np.ndarray) -> None:
        tag = f"{phase}/after_declaration" if self.declared else phase
        self.counts[f"decisions/{tag}"] += 1
        if len(recorder.calls) != 1:
            self.note(f"model_calls_not_one/{tag}", len(recorder.calls))
            return
        rows, q = recorder.calls[0]
        state_ok = bool((rows[:, :enc.DIM_STATE] == rows[0, :enc.DIM_STATE]).all()) and \
            np.array_equal(rows[0, :enc.DIM_STATE], mine.rows[0, :enc.DIM_STATE])
        if not state_ok:
            diff = np.nonzero(rows[0, :enc.DIM_STATE] != mine.rows[0, :enc.DIM_STATE])[0]
            self.note(f"state_mismatch/{tag}", f"dims {diff[:12].tolist()}")
        theirs = {r.tobytes(): i for i, r in enumerate(rows[:, enc.DIM_STATE:])}
        ours = {r.tobytes(): i for i, r in enumerate(mine.rows[:, enc.DIM_STATE:])}
        only_t, only_o = theirs.keys() - ours.keys(), ours.keys() - theirs.keys()
        if only_t or only_o:
            self.note(f"legal_mismatch/{tag}", f"theirs_only {len(only_t)} ours_only {len(only_o)}")
        common = theirs.keys() & ours.keys()
        if state_ok and common:
            qt = np.array([q[theirs[k]] for k in common])
            qo = np.array([mine.q[ours[k]] for k in common])
            diff = float(np.max(np.abs(qt - qo)))
            self.max_q_diff = max(self.max_q_diff, diff)
            if not np.array_equal(qt, qo):
                self.note(f"q_not_bitwise/{tag}", f"max diff {diff:.3g}")
        if np.array_equal(chosen_theirs, mine.vector):
            self.counts[f"choice_equal/{tag}"] += 1
        else:
            self.note(f"choice_mismatch/{tag}",
                      f"theirs {np.nonzero(chosen_theirs)[0].tolist()} "
                      f"ours {np.nonzero(mine.vector)[0].tolist()}")
        if int((q == q.max()).sum()) > 1:
            self.note(f"q_tie/{tag}", "tie at the maximum")
        if mine.declared or (phase == "play" and self._is_declaration(chosen_theirs)):
            self.declared = True
            self.note("declared_wild_choice", np.nonzero(chosen_theirs)[0].tolist())
        if self.steps is not None:
            self.steps.append({"decision": fixture_entry(phase, rows, q, chosen_theirs)})

    def _is_declaration(self, vector: np.ndarray) -> bool:
        """A lone wild (or wild pair) whose rank bit is not the level rank."""
        level = self.player.round.level
        wild_t = level                     # DanLM heart = suit 0: card = rank
        cards = np.nonzero(vector[:enc.DIM_CARDS])[0].tolist()
        rank = int(vector[enc.DIM_CARDS + 11:].argmax())
        return cards == [wild_t] and rank != level

    def wrap(self, agent) -> None:
        recorder = Recorder(agent._model)
        agent._model = recorder
        select_play, give, back = agent.select_play, agent.select_tribute_give, agent.select_tribute_back

        def play(obs, rnd):
            mine = self.player.decide(self.round.state)
            recorder.calls.clear()
            index = select_play(obs, rnd)
            self.compare("play", mine, recorder, np.asarray(obs.legal_plays[index], np.float32))
            return index

        def tribute(fn, phase):
            def call(hand, legal, *args):
                mine = self.player.decide(self.round.state)
                if mine.legal_cards != sorted(int(c) for c in legal):
                    self.note(f"tribute_legal_mismatch/{phase}", f"theirs {sorted(legal)} ours {mine.legal_cards}")
                recorder.calls.clear()
                card = int(fn(hand, legal, *args))
                self.compare(phase, mine, recorder, enc.single_play(card))
                return card
            return call

        agent.select_play = play
        agent.select_tribute_give = tribute(give, "give")
        agent.select_tribute_back = tribute(back, "back")


def sparse(vector: np.ndarray) -> list:
    """[indices, values] of the nonzero entries."""
    nz = np.nonzero(vector)[0]
    return [nz.tolist(), vector[nz].astype(float).tolist()]


def fixture_entry(phase: str, rows: np.ndarray, q: np.ndarray, chosen: np.ndarray) -> dict:
    """Reference rows (compiled agent's model input) and Q values of one decision."""
    return {"phase": phase, "state": sparse(rows[0, :enc.DIM_STATE]),
            "actions": [sparse(a) for a in rows[:, enc.DIM_STATE:]],
            "q": q.astype(float).tolist(),
            "chosen": sparse(chosen)}


def run(deals: int, seed: int, tribute_fraction: float, model_path: str | None,
        fixture: bool = False) -> tuple[dict, list]:
    from eval.danlm import arena

    spec = f"v1t:{Path(model_path or default_model_path()).resolve()}"
    seats = arena.DanLMSeats(spec)
    player = V1TPlayer(V1TModel(model_path))
    verifier = Verifier(player)
    for agent in seats.agents:
        verifier.wrap(agent)
    deal_specs = arena.generate_deals(deals, seed, tribute_fraction)
    wanted = {"none", "single", "double", "anti"} if fixture else set()
    fixture_rounds: list[dict] = []
    observe = player.observe

    def logged_observe(event):
        if verifier.steps is not None:
            a = event.action
            verifier.steps.append({"action": [a.type, sorted(int(c) for c in a.cards),
                                              None if a.type in ("Pass", "JokerBomb") else int(a.key)]})
        observe(event)

    player.observe = logged_observe
    statuses: Counter = Counter()
    tributes: Counter = Counter()
    divergences: Counter = Counter()
    started = time.monotonic()
    for i, deal in enumerate(deal_specs):
        lockstep = arena.LockstepRound(deal, ("danlm",) * 4, player, seats, random.Random(seed + i),
                                       round_id=f"deal-{i}")
        verifier.round, verifier.round_id, verifier.declared = lockstep, f"deal-{i}", False
        verifier.steps = [] if wanted else None
        player.begin_round(deal, lockstep.state)
        record = lockstep.play()
        kind = record.tribute
        if verifier.steps is not None and kind in wanted and record.status == "ok" \
                and not verifier.declared and not record.divergences.counts:
            wanted.discard(kind)
            fixture_rounds.append({"id": f"deal-{i}", "tribute": kind, "deal": arena.deal_to_json(deal),
                                   "steps": verifier.steps})
        statuses[record.status] += 1
        tributes[record.tribute] += 1
        divergences.update(record.divergences.counts)
    elapsed = time.monotonic() - started
    counts = dict(sorted(verifier.counts.items()))
    decided = {k.split("/", 1)[1]: v for k, v in counts.items() if k.startswith("decisions/")}
    equal = {k.split("/", 1)[1]: v for k, v in counts.items() if k.startswith("choice_equal/")}
    report = {
        "deals": deals, "seed": seed, "tribute_fraction": tribute_fraction,
        "model": str(Path(model_path or default_model_path())), "elapsed_seconds": elapsed,
        "rounds": dict(statuses), "tribute_kinds": dict(tributes),
        "lockstep_divergences": dict(divergences),
        "decisions": decided, "choices_equal": equal,
        "decisions_total": sum(decided.values()), "choices_equal_total": sum(equal.values()),
        "max_abs_q_diff_common_rows": verifier.max_q_diff,
        "counts": counts, "samples": verifier.samples,
    }
    return report, fixture_rounds


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--deals", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tribute-fraction", type=float, default=0.75)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fixture", type=Path, default=None,
                        help="also write one round per tribute kind (reference rows, Q values "
                             "and the applied actions) here, gzip JSON, for tests/test_v1t.py")
    args = parser.parse_args(argv)
    report, fixture = run(args.deals, args.seed, args.tribute_fraction, args.model,
                          bool(args.fixture))
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    if args.fixture:
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        import gzip

        payload = {"model": "v3_rep_v1t_best_eval_001_int8.onnx", "seed": args.seed,
                   "tribute_fraction": args.tribute_fraction, "rounds": fixture}
        with gzip.open(args.fixture, "wt") as stream:
            json.dump(payload, stream, separators=(",", ":"))
    print(json.dumps({k: v for k, v in report.items() if k not in ("samples",)}, indent=2))
    for key, items in report["samples"].items():
        print(key, items[:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
