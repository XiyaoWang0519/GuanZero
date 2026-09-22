"""Validation-selected, match-held-out Transformer belief experiment.

The original small probe is preserved in belief_probe.py. This stricter runner
uses only provenance-verified, frozen-policy architecture-probe collections.
It never selects checkpoints or hyperparameters using the final test set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from train.belief_model import matched_models
from train.belief_probe import collate, count_parameters, examples
from train.logs import TOKEN_DIM
from train.tribute_data import engine_source_digest


def check_time(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("belief experiment exceeded its cooperative walltime bound")


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_dataset(directory: Path, deadline: float | None = None) -> tuple[list[dict], dict, str]:
    provenance = json.loads((directory / "provenance.json").read_text())
    if (provenance.get("status") != "complete"
            or provenance.get("purpose") != "architecture_probe"
            or provenance.get("learner_updates") != 0):
        raise ValueError("complete frozen-policy architecture_probe provenance is required")
    if provenance.get("engine_source_sha256") != engine_source_digest():
        raise ValueError("belief labels require the current engine source fingerprint")
    if (provenance.get("action_mode") != "canonical"
            or provenance.get("tribute_policy") != "heuristic"
            or provenance.get("sampling_margin") != 0 or provenance.get("stage") != "dmc"
            or provenance.get("play_mode") != "fp32_argmax"):
        raise ValueError("expected canonical frozen argmax play and heuristic exchanges")
    identity = provenance.get("checkpoint_id", "")
    if not isinstance(identity, str) or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("missing checkpoint weight identity")
    if (not isinstance(provenance.get("seed"), int)
            or not isinstance(provenance.get("training_seed"), int)
            or provenance["seed"] == provenance["training_seed"]):
        raise ValueError("distinct known collection and base training seeds are required")
    rounds = []
    digest = hashlib.sha256()
    for path in sorted(directory.glob("round-*.npz")):
        check_time(deadline)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
        with np.load(path, allow_pickle=False) as data:
            version = int(data["schema_version"])
            if version not in (1, 2):
                raise ValueError("unsupported belief schema")
            r = {k: data[k].copy() for k in ("obs", "hidden", "seat", "prefix", "tokens")}
            r["group"] = str(data["group"])
            # Schema 2 adds styled-opponent labels; schema 1 rounds stay loadable.
            r["schema_version"] = version
            r["style_region"] = str(data["style_region"]) if version >= 2 else "unknown"
            r["driver"] = data["driver"].copy() if version >= 2 else np.zeros(len(r["seat"]), np.int64)
        n = len(r["obs"])
        if (not n or r["obs"].shape != (n, 1849)
                or r["hidden"].shape != (n, 3, 54)
                or r["seat"].shape != (n,) or r["prefix"].shape != (n,)
                or r["tokens"].ndim != 2 or r["tokens"].shape[1] != TOKEN_DIM):
            raise ValueError(f"invalid belief shapes: {path}")
        for key in ("obs", "hidden", "seat", "prefix", "tokens"):
            if not np.isfinite(r[key]).all() or not np.issubdtype(r[key].dtype, np.integer):
                raise ValueError(f"belief arrays must be finite integers: {path}")
        if (np.any(r["obs"] < 0) or np.any(r["obs"] > 1)
                or np.any(r["hidden"] < 0) or np.any(r["hidden"] > 2)
                or np.any(r["seat"] < 0) or np.any(r["seat"] > 3)
                or np.any(r["prefix"] < 0) or np.any(r["prefix"] >= len(r["tokens"]))
                or np.any(np.diff(r["prefix"]) <= 0)
                or np.any(r["tokens"] < 0) or np.any(r["tokens"] > 1)):
            raise ValueError(f"invalid belief values or history prefixes: {path}")
        # Public tokens must never contain private exchange-structure flags.
        if r["tokens"][:, 150:158].any():
            raise ValueError("private exchange flags leaked into public history")
        if (not np.all(r["tokens"][:, :4].sum(1) == 1)
                or not np.array_equal(r["tokens"][r["prefix"], :4].argmax(1), r["seat"])):
            raise ValueError("decision prefix must precede its own public action")
        # Relative hidden hands partition the observable unseen-card multiset.
        if not np.array_equal(r["hidden"].sum(1), r["obs"][:, 108:162] + r["obs"][:, 162:216]):
            raise ValueError("hidden counts disagree with observable unseen cards")
        cards_left = r["obs"][:, 648:732].reshape(n, 3, 28)
        if (not np.all(cards_left.sum(2) == 1)
                or not np.array_equal(r["hidden"].sum(2), cards_left.argmax(2))
                or np.any(r["hidden"] < r["obs"][:, 1687:1849].reshape(n, 3, 54))):
            raise ValueError("relative hidden hands disagree with public seat counts or known holdings")
        rounds.append(r)
    if (len(rounds) != provenance.get("collected_rounds") or not rounds
            or len(rounds) != provenance.get("requested_rounds")
            or sum(len(r["obs"]) for r in rounds) != provenance.get("collected_decisions")
            or len({r["group"] for r in rounds}) != provenance.get("match_groups")):
        raise ValueError("round count disagrees with complete provenance")
    return rounds, provenance, digest.hexdigest()


def split_rounds(rounds: list[dict], split_seed: int,
                 heldout_styles: bool = False) -> dict[str, list[dict]]:
    """Whole-match split by seeded SHA-256, optionally restricted by style region.

    With `heldout_styles` the test set is drawn only from matches whose styles
    came from the held-out style region, and training never sees them; the
    seeded hash still orders the split so it stays reproducible.
    """
    regions = {}
    for r in rounds:
        regions.setdefault(r["group"], set()).add(r.get("style_region", "unknown"))
    groups = sorted({r["group"] for r in rounds},
                    key=lambda g: hashlib.sha256(f"{split_seed}:{g}".encode()).digest())
    if len(groups) < 10:
        raise ValueError("experiment needs at least ten distinct source matches")
    size = max(1, int(len(groups) * .15))
    if heldout_styles:
        reserved = [g for g in groups if regions[g] == {"heldout"}]
        rest = [g for g in groups if regions[g] != {"heldout"}]
        if len(reserved) < size:
            raise ValueError("not enough held-out-style matches for the test split")
        test, validation = set(reserved[:size]), set(rest[:size])
        return {"train": [r for r in rounds
                          if r["group"] not in test | validation
                          and regions[r["group"]] != {"heldout"}],
                "validation": [r for r in rounds if r["group"] in validation],
                "test": [r for r in rounds if r["group"] in test]}
    test, validation = set(groups[:size]), set(groups[size:2 * size])
    return {"train": [r for r in rounds if r["group"] not in test | validation],
            "validation": [r for r in rounds if r["group"] in validation],
            "test": [r for r in rounds if r["group"] in test]}


@torch.inference_mode()
def evaluate(model, items: list, device: str, batch_size: int,
             deadline: float | None = None) -> dict:
    model.eval()
    sums, counts = np.zeros((3, 3)), np.zeros((3, 3), dtype=np.int64)
    matches: dict[str, list[float]] = {}
    for start in range(0, len(items), batch_size):
        check_time(deadline)
        part = items[start:start + batch_size]
        batch = collate(part, device)
        logits = model(batch["obs"], batch["tokens"], batch["lengths"], batch["seat"])
        losses = F.cross_entropy(logits.reshape(-1, 3), batch["hidden"].reshape(-1),
                                 reduction="none").reshape(-1, 3, 54).mean(-1).cpu().numpy()
        if not np.isfinite(losses).all():
            raise FloatingPointError("non-finite belief evaluation loss")
        stages = np.digitize(batch["obs"][:, 216:648].sum(-1).cpu().numpy(), [36, 72])
        for stage in range(3):
            sums[stage] += losses[stages == stage].sum(0)
            counts[stage] += (stages == stage).sum()
        for (record, _), loss in zip(part, losses.mean(1)):
            entry = matches.setdefault(record["group"], [0., 0])
            entry[0] += float(loss)
            entry[1] += 1
    return {"log_loss": float(sums.sum() / counts.sum()), "decisions": len(items),
            "matches": {g: {"log_loss": total / n, "decisions": n}
                        for g, (total, n) in sorted(matches.items())},
            "by_stage_and_seat": {
                stage: {seat: {"log_loss": float(sums[i, j] / counts[i, j]) if counts[i, j] else None,
                               "decisions": int(counts[i, j])}
                        for j, seat in enumerate(("lho", "partner", "rho"))}
                for i, stage in enumerate(("early", "middle", "late"))}}


def paired_improvement(reference: dict, candidate: dict, seed: int, samples: int = 10000) -> dict:
    groups = sorted(reference["matches"])
    if set(groups) != set(candidate["matches"]):
        raise ValueError("paired evaluation requires identical test matches")
    delta = []
    for g in groups:
        if reference["matches"][g]["decisions"] != candidate["matches"][g]["decisions"]:
            raise ValueError("paired match decisions differ")
        delta.append(reference["matches"][g]["log_loss"] - candidate["matches"][g]["log_loss"])
    values = np.asarray(delta)
    rng = np.random.default_rng(seed)
    boot = np.concatenate([values[rng.integers(len(values), size=(min(256, samples-i), len(values)))].mean(1)
                           for i in range(0, samples, 256)])
    return {"micro_log_loss_improvement": reference["log_loss"] - candidate["log_loss"],
            "relative_micro_improvement": 1 - candidate["log_loss"] / reference["log_loss"],
            "mean_match_log_loss_improvement": float(values.mean()),
            "bootstrap_95_ci": np.quantile(boot, [.025, .975]).tolist(),
            "test_matches": len(groups), "bootstrap_samples": samples,
            "sampling_unit": "whole paired match, equal weight per match"}


def run(directory: Path, output: Path, *, seeds=(31, 32, 33), split_seed=20260928,
        steps=6000, min_steps=3000, validation_interval=1000, patience=2,
        batch_size=64, width=128, layers=2, learning_rate=.0003,
        device="cpu", threads=4, max_seconds=2400, heldout_style_test=False) -> dict:
    if (not seeds or len(set(seeds)) != len(seeds) or min(steps, min_steps, validation_interval,
            patience, batch_size, threads) <= 0 or min_steps > steps
            or not np.isfinite(learning_rate) or learning_rate <= 0
            or not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("invalid experiment budget or seeds")
    if output.exists():
        raise FileExistsError(f"experiment output already exists: {output}")
    output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + max_seconds
    config = dict(seeds=list(seeds), split_seed=split_seed, steps=steps, min_steps=min_steps,
                  validation_interval=validation_interval, patience=patience, batch_size=batch_size,
                  width=width, layers=layers, learning_rate=learning_rate, device=device,
                  threads=threads, max_seconds=max_seconds,
                  heldout_style_test=bool(heldout_style_test))
    report = {"schema_version": 1, "status": "running", "config": config, "runs": [],
              "gate": "pending", "rl_enabled": False,
              "source_sha256": {name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                                  for name in ("train/belief_experiment.py", "train/belief_model.py",
                                               "train/belief_probe.py", "train/logs.py")}}
    write_json(output / "report.json", report)
    try:
        torch.set_num_threads(threads)
        rounds, provenance, fingerprint = load_dataset(directory, deadline)
        if set(seeds) & {provenance["seed"], provenance["training_seed"]}:
            raise ValueError("fit seeds must differ from collection and base training seeds")
        splits = split_rounds(rounds, split_seed, heldout_styles=bool(heldout_style_test))
        data = {name: examples(records) for name, records in splits.items()}
        report.update(dataset_sha256=fingerprint, collection=provenance,
                      splits={name: {"matches": sorted({r['group'] for r in records}),
                                     "rounds": len(records), "decisions": len(data[name])}
                              for name, records in splits.items()})
        write_json(output / "report.json", report)
        for seed in seeds:
            check_time(deadline)
            torch.manual_seed(seed)
            models = matched_models(rounds[0]["obs"].shape[1], width, layers)
            result = {"seed": seed, "models": {}}
            for name, model in models.items():
                model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                sampler = random.Random(seed)
                best, best_step, stale = float("inf"), 0, 0
                selected = None
                curve = []
                model_started = time.monotonic()
                learn_seconds = 0.
                for step in range(1, steps + 1):
                    check_time(deadline)
                    update_started = time.monotonic()
                    model.train()
                    batch = collate(sampler.choices(data["train"], k=batch_size), device)
                    logits = model(batch["obs"], batch["tokens"], batch["lengths"], batch["seat"])
                    loss = F.cross_entropy(logits.reshape(-1, 3), batch["hidden"].reshape(-1))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
                    optimizer.step()
                    learn_seconds += time.monotonic() - update_started
                    if step % validation_interval and step != steps:
                        continue
                    metric = evaluate(model, data["validation"], device, batch_size, deadline)
                    entry = {"step": step, "last_train_batch_loss": float(loss.detach()),
                             "validation_log_loss": metric["log_loss"], "learn_seconds": learn_seconds}
                    curve.append(entry)
                    with (output / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps({"seed": seed, "model": name, **entry}) + "\n")
                    if metric["log_loss"] < best:
                        best, best_step, stale = metric["log_loss"], step, 0
                        selected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    else:
                        stale += 1
                    if step >= min_steps and stale >= patience:
                        break
                model.load_state_dict(selected)
                # Test data is first accessed here, after validation-only selection.
                final_test = evaluate(model, data["test"], device, batch_size, deadline)
                path = output / f"{name}-s{seed}.pt"
                temporary = path.with_suffix(".tmp")
                torch.save({"purpose": "belief_probe_only", "architecture": name, "model": selected,
                            "config": config, "seed": seed, "selected_step": best_step,
                            "dataset_sha256": fingerprint}, temporary)
                temporary.replace(path)
                result["models"][name] = {"parameters": count_parameters(model),
                    "steps_run": step, "selected_step": best_step, "validation_log_loss": best,
                    "early_stopped": step < steps, "best_at_budget_end": best_step == steps,
                    "learn_seconds": learn_seconds, "elapsed_seconds": time.monotonic() - model_started,
                    "learning_curve": curve, "test": final_test, "checkpoint": str(path),
                    "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                write_json(output / f"seed-{seed}-partial.json", result)
                model.to("cpu")
            result["v2_vs_v1"] = paired_improvement(result["models"]["v1"]["test"],
                                                      result["models"]["v2"]["test"], split_seed)
            result["v2_vs_no_history"] = paired_improvement(result["models"]["no_history"]["test"],
                                                      result["models"]["v2"]["test"], split_seed)
            report["runs"].append(result)
            write_json(output / "report.json", report)
        primary = all(r["v2_vs_v1"]["test_matches"] >= 20
                      and r["v2_vs_v1"]["relative_micro_improvement"] >= .01
                      and r["v2_vs_v1"]["bootstrap_95_ci"][0] > 0 for r in report["runs"])
        history = all(r["v2_vs_no_history"]["mean_match_log_loss_improvement"] > 0 for r in report["runs"])
        history &= sum(r["v2_vs_no_history"]["bootstrap_95_ci"][0] > 0 for r in report["runs"]) >= 2
        check_time(deadline)
        report["gate"] = "supports_v2_rl_experiment" if len(seeds) >= 3 and primary and history else "v2_not_yet_justified"
        report["status"] = "complete"
        report["interpretation"] = ("Shared dataset and test matches across optimizer seeds. Matched parameters and maximum "
            "updates, not equal wall-clock compute. Positive belief evidence still needs an equal-compute RL comparison. "
            "A negative bounded result is not proof that Transformers cannot help.")
    except BaseException as error:
        report["status"] = "incomplete"
        report["error_type"] = type(error).__name__
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "report.json", report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[31, 32, 33])
    for name, default in (("split-seed", 20260928), ("steps", 6000), ("min-steps", 3000),
                          ("validation-interval", 1000), ("patience", 2), ("batch-size", 64),
                          ("width", 128), ("layers", 2), ("threads", 4)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--learning-rate", type=float, default=.0003)
    p.add_argument("--max-seconds", type=float, default=2400)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--heldout-style-test", action="store_true",
                   help="restrict test matches to the held-out style region")
    args = vars(p.parse_args())
    args["directory"] = args.pop("logs")
    result = run(**args)
    print(json.dumps({k: result[k] for k in ("status", "gate", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
