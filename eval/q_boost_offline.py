"""Stage 1 of docs/reports/vrpo-design-2026-10-04.md: can a fitted Q critic give
better advantage estimates than GAE with the trained V critic?

Pure self-play from one checkpoint (all four seats the current policy, sampled).
Three steps, each a subcommand:

    collect   play rounds and store every play decision: observation, hidden
              hands, all candidates, the policy's probabilities, the choice.
              Test rounds also get ground truth at random decisions: K sampled
              continuations after the taken action (Q) and K from the state (V).
    fit       train a candidate-scoring Q critic (train.q_critic, warm-started
              from the V critic) with the multi-step Expected SARSA(lambda)
              targets over the four-seat chain, refreshed every epoch.
    evaluate  on the test points compare the advantage estimates (GAE with V,
              own-team and four-seat; Q-boosting, own-team and four-seat)
              against the rollout truth, after subtracting the truth's own
              sampling variance.

Rounds are zero-sum between teams, so the acting team's Q at an opponent's
decision enters the other team's chain with a sign flip. See the design note
for what each estimator is.
"""
from __future__ import annotations

import argparse
import copy
import json
from multiprocessing import get_context
from pathlib import Path
import pickle
import random
import time

import gd
import numpy as np

from eval.critic_noise import PLAY, ROUND_END, _Listener, hidden_counts, make_deal
from eval.critic_noise_split import SplitWorker
from eval.history_policy import apply_and_observe, explicit_passes, resolve_forced_passes
from train.history_rollout import compute_gae
from train.q_boost import q_boost_advantages

ACT_DIM = int(gd.ACT_DIM)
OBS_DIM = int(gd.OBS_DIM)


def unpack(rows: np.ndarray, dim: int) -> np.ndarray:
    """Binary features are stored bit-packed (8x smaller); rounds stored unpacked pass through."""
    return rows if rows.shape[1] == dim else np.unpackbits(rows, axis=1, count=dim)
SPLIT_SEED = {"train": 1_000_000, "val": 2_000_000, "test": 3_000_000}


def sample_rows(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One draw per row of a padded ``[n, width]`` probability table."""
    cumulative = probs.cumsum(1)
    draws = rng.random(len(probs)) * cumulative[:, -1]
    picks = (cumulative <= draws[:, None]).sum(1)
    return np.minimum(picks, probs.shape[1] - 1)


class OfflineWorker(SplitWorker):
    def __init__(self, checkpoint: str, threads: int, device: str = "cpu") -> None:
        super().__init__(checkpoint, threads)
        if device != "cpu":
            from eval.history_policy import HistoryPolicy
            self.policy = HistoryPolicy(self.policy.actor, sample=True, device=device)

    def to_end(self, branches: list, streams: list, origin: int, rng: np.random.Generator
               ) -> list[float]:
        listeners = [_Listener(s) for s in streams]
        returns = [None] * len(branches)
        active = list(range(len(branches)))
        guard = 0
        while active:
            guard += 1
            if guard > 2000:
                raise RuntimeError("rollout does not terminate")
            for i in list(active):
                resolve_forced_passes(self.engine, branches[i], [listeners[i]])
                if int(branches[i].phase) == ROUND_END:
                    returns[i] = float(self.engine.end_round(branches[i]).seat_return[origin])
                    active.remove(i)
            if not active:
                break
            table = self.table([branches[i] for i in active], [streams[i] for i in active])
            picks = sample_rows(table, rng)
            for row, i in enumerate(active):
                actions = self.engine.legal_actions(branches[i])
                apply_and_observe(self.engine, branches[i], actions[int(picks[row])], [listeners[i]])
        return returns

    def truth(self, state, stream, choice: int, k: int, rng: np.random.Generator) -> dict:
        """K continuations forcing ``choice`` (Q) and K from the state (V), for the mover's team."""
        origin = int(state.to_move)
        branches = [gd.MatchState.deserialize(state.serialize()) for _ in range(2 * k)]
        streams = [copy.deepcopy(stream) for _ in range(2 * k)]
        for i in range(k):
            listener = _Listener(streams[i])
            action = self.engine.legal_actions(branches[i])[choice]
            apply_and_observe(self.engine, branches[i], action, [listener])
        returns = np.asarray(self.to_end(branches, streams, origin, rng), np.float64)
        q, v = returns[:k], returns[k:]
        return {"mean_q": float(q.mean()), "var_q": float(q.var(ddof=1)),
                "mean_v": float(v.mean()), "var_v": float(v.var(ddof=1)), "k": k}

    def play(self, job: dict) -> dict:
        from train.history_model import PublicStream

        seed = job["seed"]
        rng = random.Random(seed)
        nprng = np.random.default_rng(seed)
        state = gd.MatchState()
        self.engine.set_deal(state, make_deal(rng, job["tribute_fraction"]))
        stream = PublicStream()
        listener = _Listener(stream)
        seats, obs, hidden, counts, cand, probs, chosen, cards_left, points = (
            [], [], [], [], [], [], [], [], [])
        with explicit_passes(self.engine, [listener]):
            while True:
                resolve_forced_passes(self.engine, state, [listener])
                phase = int(state.phase)
                if phase == ROUND_END:
                    break
                actions = self.engine.legal_actions(state)
                if phase != PLAY:
                    apply_and_observe(self.engine, state, actions[self.engine.greedy(state)], [listener])
                    continue
                seat = int(state.to_move)
                row = self.table([state], [stream])[0][:len(actions)]
                choice = int(sample_rows(row[None], nprng)[0])
                if job["point_rate"] and len(actions) > 1 and nprng.random() < job["point_rate"]:
                    found = self.truth(state, stream, choice, job["truth_rollouts"], nprng)
                    found["index"] = len(seats)
                    points.append(found)
                seats.append(seat)
                obs.append(np.asarray(state.observation(seat), np.uint8))
                hidden.append(hidden_counts(state, seat).astype(np.uint8).reshape(-1))
                counts.append(len(actions))
                cand.append(np.stack([np.asarray(self.engine.encode_action(a, state, seat), np.uint8)
                                      for a in actions]))
                probs.append(row.astype(np.float32))
                chosen.append(choice)
                cards_left.append(len(state.hand(seat)))
                apply_and_observe(self.engine, state, actions[choice], [listener])
        result = self.engine.end_round(state)
        team = [float(result.seat_return[0]), float(result.seat_return[1])]
        return {"seat": np.asarray(seats, np.int8), "obs": np.packbits(np.stack(obs), axis=1),
                "hidden": np.stack(hidden),
                "counts": np.asarray(counts, np.int32),
                "cand": np.packbits(np.concatenate(cand), axis=1),
                "probs": np.concatenate(probs), "chosen": np.asarray(chosen, np.int32),
                "cards_left": np.asarray(cards_left, np.int16), "team_return": team,
                "points": points, "seed": seed}


_WORKER: OfflineWorker | None = None


def _init(checkpoint: str, threads: int) -> None:
    global _WORKER
    _WORKER = OfflineWorker(checkpoint, threads)


def _play(job: dict) -> dict:
    return _WORKER.play(job)


def collect(args: argparse.Namespace) -> None:
    jobs = [{"seed": SPLIT_SEED[args.split] + args.seed_offset + i, "tribute_fraction": 0.5,
             "point_rate": args.point_rate, "truth_rollouts": args.truth_rollouts}
            for i in range(args.rounds)]
    started = time.perf_counter()
    rounds = []
    with get_context("spawn").Pool(args.workers, _init, (args.checkpoint, 1)) as pool:
        for i, round_ in enumerate(pool.imap(_play, jobs, chunksize=1), 1):
            rounds.append(round_)
            if i % 25 == 0 or i == len(jobs):
                print(json.dumps({"split": args.split, "rounds": i, "of": len(jobs),
                                  "decisions": int(sum(len(r["seat"]) for r in rounds)),
                                  "points": int(sum(len(r["points"]) for r in rounds)),
                                  "elapsed": round(time.perf_counter() - started, 1)}), flush=True)
    zero_sum = sum(abs(r["team_return"][0] + r["team_return"][1]) < 1e-9 for r in rounds)
    print(json.dumps({"zero_sum_rounds": int(zero_sum), "rounds": len(rounds)}), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump({"checkpoint": args.checkpoint, "args": vars(args), "rounds": rounds}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


# ---- estimators ---------------------------------------------------------------------

def round_estimates(round_: dict, q_chosen: np.ndarray, v_pi: np.ndarray, v_base: np.ndarray,
                    lam: float) -> dict[str, np.ndarray]:
    """Advantage estimates for every decision of one round, in the mover's own team's view.

    ``q_chosen``: Q of the taken candidate, ``v_pi``: policy expectation of Q,
    ``v_base``: the V critic's value of the state.
    """
    seat = round_["seat"].astype(np.int64)
    n = len(seat)
    team_of = seat % 2
    out = {name: np.zeros(n) for name in ("mc", "gae_own", "gae_four", "qboost_own", "qboost_four")}
    for team in (0, 1):
        reward = round_["team_return"][team]
        mine = np.flatnonzero(team_of == team)
        if not len(mine):
            continue
        sign = np.where(team_of == team, 1.0, -1.0)
        # own-team chain (the trainer's trajectory): that team's rows only
        last = np.zeros((1, len(mine)))
        last[0, -1] = reward
        dones = np.zeros((1, len(mine)), bool)
        dones[0, -1] = True
        gae, _ = compute_gae(v_base[mine][None], last, dones, 1.0, lam)
        out["gae_own"][mine] = gae[0]
        boost, _ = q_boost_advantages(q_chosen[mine][None], v_pi[mine][None], last, dones, 1.0, lam)
        out["qboost_own"][mine] = boost[0]
        out["mc"][mine] = reward - v_base[mine]
        # four-seat chain: every decision, opponents' values sign-flipped
        full = np.zeros((1, n))
        full[0, -1] = reward
        done_full = np.zeros((1, n), bool)
        done_full[0, -1] = True
        gae4, _ = compute_gae((sign * v_base)[None], full, done_full, 1.0, lam)
        out["gae_four"][mine] = gae4[0][mine]
        boost4, _ = q_boost_advantages((sign * q_chosen)[None], (sign * v_pi)[None], full,
                                       done_full, 1.0, lam)
        out["qboost_four"][mine] = boost4[0][mine]
    return out


# ---- fitting and evaluation ---------------------------------------------------------

def load_rounds(paths: list[str]) -> list[dict]:
    rounds = []
    for path in paths:
        with open(path, "rb") as handle:
            rounds.extend(pickle.load(handle)["rounds"])
    return rounds


def forward_round(model, critic, round_: dict, torch, chunk: int = 4096):
    """Q of the taken candidate, ``sum pi Q`` and the V critic's value, per decision."""
    from train.q_critic import expected_value

    n = len(round_["seat"])
    starts = np.concatenate([[0], np.cumsum(round_["counts"])])
    q_chosen = np.zeros(n)
    v_pi = np.zeros(n)
    v_base = np.zeros(n)
    with torch.no_grad():
        for begin in range(0, n, chunk):
            end = min(begin + chunk, n)
            obs = torch.as_tensor(unpack(round_["obs"][begin:end], OBS_DIM))
            hidden = torch.as_tensor(round_["hidden"][begin:end])
            counts = torch.as_tensor(round_["counts"][begin:end].astype(np.int64))
            lo, hi = starts[begin], starts[end]
            cand = torch.as_tensor(unpack(round_["cand"][lo:hi], ACT_DIM))
            probs = torch.as_tensor(round_["probs"][lo:hi])
            q_flat = model.forward_all(obs, hidden, cand, counts)
            v_pi[begin:end] = expected_value(q_flat, probs.double().to(q_flat.dtype), counts).numpy()
            picks = torch.as_tensor(starts[begin:end] - lo + round_["chosen"][begin:end])
            q_chosen[begin:end] = q_flat[picks].numpy()
            v_base[begin:end] = critic(obs, hidden).numpy()
    return q_chosen, v_pi, v_base


def fit(args: argparse.Namespace) -> None:
    import torch
    from train.history_model import load_history_checkpoint
    from train.q_critic import QCritic

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    _, critic, _ = load_history_checkpoint(args.checkpoint, "cpu")
    critic.eval()
    model = QCritic(critic, ACT_DIM)
    train_rounds = load_rounds(args.train)
    val_rounds = load_rounds(args.val)
    sizes = [len(r["seat"]) for r in train_rounds]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    total = int(offsets[-1])
    print(json.dumps({"train_rounds": len(train_rounds), "train_decisions": total,
                      "val_rounds": len(val_rounds)}), flush=True)
    obs = np.concatenate([r["obs"] for r in train_rounds])          # packed rows
    hidden = torch.as_tensor(np.concatenate([r["hidden"] for r in train_rounds]))
    chosen_cand = []
    for r in train_rounds:
        starts = np.concatenate([[0], np.cumsum(r["counts"])])[:-1]
        chosen_cand.append(r["cand"][starts + r["chosen"]])
    chosen_cand = np.concatenate(chosen_cand)                        # packed rows
    optimizer = torch.optim.Adam([
        {"params": model.tower.parameters(), "lr": args.lr * 0.3},
        {"params": list(model.head.parameters()) + list(model.project.parameters())
         + list(model.candidate.parameters()), "lr": args.lr}])
    history = []
    best = (float("inf"), None, -1)

    def validation() -> dict:
        errors, base_errors, q_all, v_all, g_all = [], [], [], [], []
        for r in val_rounds:
            q, vpi, vb = forward_round(model, critic, r, torch)
            ret = np.asarray([r["team_return"][int(s) % 2] for s in r["seat"]])
            errors.append((q - ret) ** 2)
            base_errors.append((vb - ret) ** 2)
            g_all.append(ret)
        errors, base_errors, g_all = map(np.concatenate, (errors, base_errors, g_all))
        return {"q_mse": float(errors.mean()), "v_mse": float(base_errors.mean()),
                "return_var": float(g_all.var())}

    for epoch in range(args.epochs + 1):
        started = time.perf_counter()
        if epoch:                 # epoch 0 is the untrained (== GAE) starting point
            targets = np.zeros(total, np.float32)
            for index, r in enumerate(train_rounds):
                q, vpi, _ = forward_round(model, critic, r, torch)
                seat = r["seat"].astype(np.int64)
                n = len(seat)
                for team in (0, 1):
                    sign = np.where(seat % 2 == team, 1.0, -1.0)
                    reward = np.zeros((1, n))
                    reward[0, -1] = r["team_return"][team]
                    done = np.zeros((1, n), bool)
                    done[0, -1] = True
                    _, tgt = q_boost_advantages((sign * q)[None], (sign * vpi)[None], reward, done,
                                                1.0, args.lam)
                    mine = seat % 2 == team
                    targets[offsets[index]:offsets[index + 1]][mine] = tgt[0][mine]
            target_t = torch.as_tensor(targets)
            model.train()
            order = torch.randperm(total)
            running = 0.0
            for begin in range(0, total, args.batch):
                idx = order[begin:begin + args.batch]
                rows = idx.numpy()
                pred = model.forward_chosen(torch.as_tensor(unpack(obs[rows], OBS_DIM)), hidden[idx],
                                            torch.as_tensor(unpack(chosen_cand[rows], ACT_DIM)))
                loss = ((pred - target_t[idx]) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                running += float(loss.detach()) * len(idx)
            model.eval()
            train_loss = running / total
        else:
            train_loss = float("nan")
        row = {"epoch": epoch, "train_loss": train_loss, **validation(),
               "seconds": round(time.perf_counter() - started, 1)}
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["q_mse"] < best[0]:
            best = (row["q_mse"], copy.deepcopy(model.state_dict()), epoch)
    model.load_state_dict(best[1])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "best_epoch": best[2], "history": history,
                "args": vars(args)}, args.output)
    print(json.dumps({"best_epoch": best[2], "best_val_q_mse": best[0]}), flush=True)


def evaluate(args: argparse.Namespace) -> None:
    import torch
    from train.history_model import load_history_checkpoint
    from train.q_critic import QCritic

    torch.set_num_threads(args.threads)
    _, critic, _ = load_history_checkpoint(args.checkpoint, "cpu")
    critic.eval()
    model = QCritic(critic, ACT_DIM)
    saved = torch.load(args.qcritic, map_location="cpu") if args.qcritic else None
    if saved:
        model.load_state_dict(saved["model"])
    model.eval()
    rows = []
    for r in load_rounds(args.test):
        if not r["points"]:
            continue
        q, vpi, vb = forward_round(model, critic, r, torch)
        est = round_estimates(r, q, vpi, vb, args.lam)
        for p in r["points"]:
            t = p["index"]
            noise = (p["var_q"] + p["var_v"]) / p["k"]
            rows.append({"truth": p["mean_q"] - p["mean_v"], "truth_noise": noise,
                         "truth_q_noise": p["var_q"] / p["k"], "truth_v_noise": p["var_v"] / p["k"],
                         "q_err": q[t] - p["mean_q"], "vbase_err": vb[t] - p["mean_v"],
                         "vpi_err": vpi[t] - p["mean_v"], "cards_left": int(r["cards_left"][t]),
                         **{name: float(est[name][t]) for name in est}})
    if not rows:
        raise SystemExit("no truth points in the test data")
    names = ("mc", "gae_own", "gae_four", "qboost_own", "qboost_four")

    def column(key: str, idx: np.ndarray) -> np.ndarray:
        return np.asarray([rows[i][key] for i in idx])

    def stats(idx: np.ndarray) -> dict:
        truth, noise = column("truth", idx), column("truth_noise", idx)
        out = {"points": int(len(idx)), "truth_rms": float(np.sqrt(max((truth ** 2 - noise).mean(), 0)))}
        for name in names:
            diff = column(name, idx) - truth
            out[f"{name}_mse"] = float((diff ** 2 - noise).mean())
            out[f"{name}_corr"] = float(np.corrcoef(column(name, idx), truth)[0, 1])
        for key, noise_key in (("q_err", "truth_q_noise"), ("vbase_err", "truth_v_noise"),
                               ("vpi_err", "truth_v_noise")):
            out[f"{key}_rms"] = float(np.sqrt(max((column(key, idx) ** 2 - column(noise_key, idx)).mean(), 0)))
        return out

    rng = np.random.default_rng(args.seed)
    everyone = np.arange(len(rows))
    point = stats(everyone)
    boots = [stats(rng.integers(0, len(rows), len(rows))) for _ in range(500)]
    ci = {key: [float(np.percentile([b[key] for b in boots], 2.5)),
                float(np.percentile([b[key] for b in boots], 97.5))] for key in point if key != "points"}
    ratios = {}
    for name in names:
        values = [b[f"{name}_mse"] / b["gae_own_mse"] for b in boots]
        ratios[name] = {"point": point[f"{name}_mse"] / point["gae_own_mse"],
                        "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]}
    # Paired bootstrap of the scale-free quality: PPO normalises advantages per
    # minibatch, so correlation with the truth matters more than mean squared error.
    corr_diff = {}
    for name in names:
        diffs = []
        for _ in range(1000):
            idx = rng.integers(0, len(rows), len(rows))
            truth = column("truth", idx)
            diffs.append(np.corrcoef(column(name, idx), truth)[0, 1]
                         - np.corrcoef(column("gae_own", idx), truth)[0, 1])
        corr_diff[name] = {"point": float(point[f"{name}_corr"] - point["gae_own_corr"]),
                           "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    cards = column("cards_left", everyone)
    strata = {}
    for label, low, high in (("19-27", 19, 27), ("11-18", 11, 18), ("1-10", 1, 10)):
        idx = np.flatnonzero((cards >= low) & (cards <= high))
        if len(idx) > 30:
            strata[label] = stats(idx)
    report = {"args": vars(args), "point": point, "ci95": ci, "mse_over_gae_own": ratios,
              "by_cards_left": strata, "trained_q": bool(saved),
              "corr_minus_gae_own": corr_diff, "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"point": point, "mse_over_gae_own": ratios}, indent=1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--split", choices=sorted(SPLIT_SEED), required=True)
    c.add_argument("--rounds", type=int, required=True)
    c.add_argument("--point-rate", type=float, default=0.0)
    c.add_argument("--truth-rollouts", type=int, default=48)
    c.add_argument("--seed-offset", type=int, default=0)
    c.add_argument("--workers", type=int, default=6)
    c.add_argument("--output", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--checkpoint", required=True)
    f.add_argument("--train", nargs="+", required=True)
    f.add_argument("--val", nargs="+", required=True)
    f.add_argument("--epochs", type=int, default=12)
    f.add_argument("--lr", type=float, default=3e-4)
    f.add_argument("--lam", type=float, default=0.95)
    f.add_argument("--batch", type=int, default=1024)
    f.add_argument("--threads", type=int, default=8)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--output", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--qcritic")
    e.add_argument("--test", nargs="+", required=True)
    e.add_argument("--lam", type=float, default=0.95)
    e.add_argument("--threads", type=int, default=8)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    {"collect": collect, "fit": fit, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
