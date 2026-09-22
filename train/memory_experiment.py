"""v3 match-memory belief probe: memory against a memory-masked control.

Question. Can a per-seat, per-round memory summary carried across the rounds
of one match improve hidden-hand prediction in later rounds, against opponents
whose style was never seen in training, relative to the identical model with
that memory masked?

Design. Both models are `train.belief_memory.MemoryBelief` towers built on the
no_history query tower that task 3 selected, at matched parameter counts
(within 0.1% of the scaled no_history tower; the masked control is parameter
identical, the memory keys are simply absent from its cross-attention). Data
is the task 2 styled collection grouped into matches by the collection group,
which encodes `(env_id, match_id)`, and ordered by `round_index`. Training
styles come from the train region and, with `--heldout-style-test`, every test
match comes from the held-out style region, exactly as in the scaled run.
Earlier-round summaries always read the full, never subsampled public token
stream of those rounds.

Results, in `report.json`:

* `test.log_loss` per model, with the task 3 cell breakdowns, including
  `target_relation:*` (teammate/opponent) and `target_driver:*` (bot/policy).
* `adaptation`: the DESIGN 9.1 item 6 metric in belief terms. Per test match,
  the paired masked-minus-memory log-loss improvement in rounds with index
  `>= --late-from` minus the same improvement in the earlier rounds of that
  same match, with a bootstrap 95% interval over matches; plus the improvement
  per round bin (0, 1, 2-3, 4-5, 6+), the improvement per exact round index and
  the least-squares slope of improvement against round index.
* `self_adaptation` per model: that model's own early-minus-late loss
  difference, which is what shows whether the masked control has a late-round
  trend of its own.
* `style_readout`: a ridge regression fitted on training matches from the
  finished-round summary vectors of bot-driven seats to that seat's style
  vector, reported as held-out R^2 per style slot for both models. The masked
  control's summaries are computed but never used by its head. Its shared
  stream layers still train through BOS, so this is a learned no-history
  baseline, not a frozen random encoder baseline.

Gate. v3 is supported when the adaptation improvement is positive with a
positive bootstrap lower bound in every seed on held-out styles, and the
masked control shows no such late-round trend of its own.
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

from train.belief_experiment import (ROUND_BINS, check_time, dataset_bytes, evaluate,
                                     load_dataset, paired_cells, paired_improvement,
                                     split_rounds, write_json)
from train.belief_memory import (attach_memory, matched_memory_models, memory_collate,
                                 memory_forward)
from train.belief_probe import count_parameters, examples
from train.logs import DRIVER_BOT

# Exact round-index cells stop here; later rounds are rare and pooled by bin.
ROUND_INDEX_CAP = 16


def memory_cells(late_from: int):
    """Name the extra decision-level cells the adaptation metric needs."""
    def cells(record: dict, index: int) -> tuple[str, ...]:
        round_index = int(record.get("round_index", -1))
        if round_index < 0:
            return ("round_index:unknown", "adapt:unknown")
        phase = "late" if round_index >= late_from else "early"
        return (f"round_index:{min(round_index, ROUND_INDEX_CAP)}", "adapt:" + phase)
    return cells


def bootstrap(values: np.ndarray, seed: int, samples: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = np.concatenate([
        values[rng.integers(len(values), size=(min(256, samples - i), len(values)))].mean(1)
        for i in range(0, samples, 256)])
    return np.quantile(draws, [.025, .975]).tolist()


def cell_matches(evaluation: dict, name: str) -> dict[str, float]:
    """Per-match log loss inside one cell, or an empty table if it is absent."""
    cell = evaluation.get("cells", {}).get(name)
    if not cell or "matches" not in cell:
        return {}
    return {group: value["log_loss"] for group, value in cell["matches"].items()}


def least_squares_slope(points: list[tuple[float, float]]) -> float | None:
    """Ordinary least-squares slope of y on x, or None with fewer than two x."""
    if len({x for x, _ in points}) < 2:
        return None
    x = np.asarray([p[0] for p in points], float)
    y = np.asarray([p[1] for p in points], float)
    return float(np.polyfit(x, y, 1)[0])


def adaptation_metric(reference: dict, candidate: dict, seed: int, samples: int = 10000
                      ) -> dict:
    """Late-minus-early paired improvement of `candidate` over `reference`.

    `reference` is the masked control and `candidate` the memory model. Both
    scored the identical test decisions, so a match that appears in the early
    and the late cell of both evaluations contributes one paired number:
    (masked - memory) late minus (masked - memory) early. The bootstrap unit is
    a whole match, as everywhere else in this experiment.
    """
    tables = {phase: (cell_matches(reference, "adapt:" + phase),
                      cell_matches(candidate, "adapt:" + phase))
              for phase in ("early", "late")}
    groups = sorted(set.intersection(*[set(t) for pair in tables.values() for t in pair]))
    if not groups:
        return {"available": False, "reason": "no match has both early and late rounds",
                "test_matches": 0}
    early = np.asarray([tables["early"][0][g] - tables["early"][1][g] for g in groups])
    late = np.asarray([tables["late"][0][g] - tables["late"][1][g] for g in groups])
    values = late - early
    result = {"available": True, "test_matches": len(groups),
              "mean_adaptation_improvement": float(values.mean()),
              "bootstrap_95_ci": bootstrap(values, seed, samples),
              "mean_late_improvement": float(late.mean()),
              "mean_early_improvement": float(early.mean()),
              "late_bootstrap_95_ci": bootstrap(late, seed + 1, samples),
              "early_bootstrap_95_ci": bootstrap(early, seed + 2, samples),
              "bootstrap_samples": samples,
              "sampling_unit": "whole paired match, equal weight per match"}
    by_bin, by_index = {}, {}
    for _, _, label in ROUND_BINS:
        pair = (cell_matches(reference, "round_bin:" + label),
                cell_matches(candidate, "round_bin:" + label))
        shared = sorted(set(pair[0]) & set(pair[1]))
        if shared:
            by_bin[label] = {"mean_improvement":
                             float(np.mean([pair[0][g] - pair[1][g] for g in shared])),
                             "test_matches": len(shared)}
    for index in range(ROUND_INDEX_CAP + 1):
        pair = (cell_matches(reference, f"round_index:{index}"),
                cell_matches(candidate, f"round_index:{index}"))
        shared = sorted(set(pair[0]) & set(pair[1]))
        if shared:
            by_index[str(index)] = {"mean_improvement":
                                    float(np.mean([pair[0][g] - pair[1][g] for g in shared])),
                                    "test_matches": len(shared)}
    result["improvement_by_round_bin"] = by_bin
    result["improvement_by_round_index"] = by_index
    result["improvement_slope_per_round"] = least_squares_slope(
        [(float(k), v["mean_improvement"]) for k, v in by_index.items()])
    return result


def self_adaptation(evaluation: dict, seed: int, samples: int = 10000) -> dict:
    """One model's own early-minus-late loss difference, over whole matches.

    A positive value means that model is already better in later rounds. The
    rounds themselves can differ in difficulty, so this raw trend is a
    diagnostic; the paired adaptation metric subtracts the control's trend.
    The originally declared gate still records its stricter no-positive-trend
    requirement separately.
    """
    early, late = cell_matches(evaluation, "adapt:early"), cell_matches(evaluation, "adapt:late")
    groups = sorted(set(early) & set(late))
    if not groups:
        return {"available": False, "test_matches": 0}
    values = np.asarray([early[g] - late[g] for g in groups])
    return {"available": True, "test_matches": len(groups),
            "mean_early_minus_late": float(values.mean()),
            "bootstrap_95_ci": bootstrap(values, seed, samples),
            "bootstrap_samples": samples}


def memory_evidence(result: dict) -> dict:
    """Expose paired evidence separately from the predeclared legacy gate.

    Increasing relative benefit alone need not mean a useful memory: it can
    also mean that an initially harmful memory is less harmful in late rounds.
    Report the late benefit and its interval independently. None denotes an
    unavailable contrast, not evidence of absence.
    """
    adaptation = result["adaptation"]
    control = result["self_adaptation"]["memory_masked"]
    available = bool(adaptation.get("available"))
    return {
        "paired_adaptation_supported": (
            adaptation["mean_adaptation_improvement"] > 0
            and adaptation["bootstrap_95_ci"][0] > 0) if available else None,
        "paired_late_benefit_supported": (
            adaptation["mean_late_improvement"] > 0
            and adaptation["late_bootstrap_95_ci"][0] > 0) if available else None,
        "masked_positive_raw_late_trend": (
            control["mean_early_minus_late"] > 0
            and control["bootstrap_95_ci"][0] > 0) if control.get("available") else None,
        "interpretation": (
            "Paired adaptation subtracts the masked control's raw early-to-late trend. "
            "A raw control trend can reflect round difficulty; failure to detect one "
            "does not establish equivalence. The original gate is retained unchanged. "
            "A memory benefit alone does not isolate opponent habits from other public "
            "match information; an opponent-history shuffle or an arena test is still needed.")}


@torch.inference_mode()
def style_readout(model, train_rounds: list[dict], test_rounds: list[dict], device: str,
                  *, alpha: float = 1.0, max_rounds: int = 512, batch_rounds: int = 16,
                  seed: int = 0, deadline: float | None = None) -> dict:
    """Ridge readout of a bot seat's style vector from its round summary.

    Fitted on training matches and reported as R^2 per style slot on the test
    matches, which under `--heldout-style-test` come from the held-out style
    region. Only bot-driven seats with a finite recorded style contribute.
    """
    model.eval()

    def features(rounds: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        chosen = rounds
        if len(chosen) > max_rounds:
            chosen = random.Random(seed).sample(chosen, max_rounds)
        rows, targets = [], []
        for start in range(0, len(chosen), batch_rounds):
            check_time(deadline)
            part = chosen[start:start + batch_rounds]
            length = max(len(r["tokens"]) for r in part)
            tokens = np.zeros((len(part), length, part[0]["tokens"].shape[1]), np.float32)
            lengths = np.zeros(len(part), np.int64)
            for row, record in enumerate(part):
                tokens[row, :len(record["tokens"])] = record["tokens"]
                lengths[row] = len(record["tokens"])
            summary, present = model.summarise(torch.tensor(tokens, device=device),
                                               torch.tensor(lengths, device=device))
            summary = summary.float().cpu().numpy()
            present = present.cpu().numpy()
            for row, record in enumerate(part):
                styles = np.asarray(record.get("styles"))
                drivers = np.asarray(record.get("seat_driver"))
                for seat in range(4):
                    if (drivers.shape != (4,) or int(drivers[seat]) != DRIVER_BOT
                            or styles.ndim != 2 or not np.isfinite(styles[seat]).all()
                            or not present[row, seat]):
                        continue
                    rows.append(summary[row, seat])
                    targets.append(styles[seat].astype(np.float64))
        if not rows:
            return np.zeros((0, 1)), np.zeros((0, 1))
        return np.stack(rows).astype(np.float64), np.stack(targets)

    train_x, train_y = features(train_rounds)
    test_x, test_y = features(test_rounds)
    if len(train_x) < train_x.shape[1] // 8 + 2 or not len(test_x):
        return {"available": False, "train_samples": int(len(train_x)),
                "test_samples": int(len(test_x)),
                "reason": "too few bot-driven seats with recorded styles"}
    mean_x, mean_y = train_x.mean(0), train_y.mean(0)
    centred = train_x - mean_x
    gram = centred.T @ centred + alpha * np.eye(centred.shape[1])
    weights = np.linalg.solve(gram, centred.T @ (train_y - mean_y))
    predicted = (test_x - mean_x) @ weights + mean_y
    residual = ((test_y - predicted) ** 2).sum(0)
    total = ((test_y - test_y.mean(0)) ** 2).sum(0)
    r2 = [None if t <= 1e-12 else float(1 - s / t) for s, t in zip(residual, total)]
    try:
        from train import styles as style_lib
        names = list(style_lib.slot_names(train_y.shape[1]))
    except Exception:
        names = [f"slot_{i}" for i in range(train_y.shape[1])]
    return {"available": True, "alpha": alpha, "slots": names,
            "heldout_r2": r2, "train_samples": int(len(train_x)),
            "test_samples": int(len(test_x)),
            "mean_heldout_r2": float(np.mean([v for v in r2 if v is not None]))
            if any(v is not None for v in r2) else None}


def run(directory: Path, output: Path, *, seeds=(41, 42, 43), split_seed=20260928,
        steps=6000, min_steps=3000, validation_interval=1000, patience=2,
        batch_size=64, width=256, layers=4, learning_rate=.0003, device="cpu",
        threads=4, max_seconds=2400, heldout_style_test=False, decisions_per_round=0,
        data_seed=20260930, cell_bootstrap_samples=2000, validation_decisions=0,
        memory_rounds=8, late_from=5, readout_rounds=512, readout_alpha=1.0,
        parameter_tolerance=.001) -> dict:
    """Fit the memory model and its masked control and write `report.json`."""
    if (not seeds or len(set(seeds)) != len(seeds)
            or min(steps, min_steps, validation_interval, patience, batch_size, threads,
                   cell_bootstrap_samples, memory_rounds, readout_rounds) <= 0
            or min_steps > steps or late_from <= 0 or decisions_per_round < 0
            or validation_decisions < 0 or not np.isfinite(learning_rate) or learning_rate <= 0
            or not np.isfinite(readout_alpha) or readout_alpha <= 0
            or not np.isfinite(parameter_tolerance) or parameter_tolerance <= 0
            or not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("invalid experiment budget or seeds")
    if output.exists():
        raise FileExistsError(f"experiment output already exists: {output}")
    output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + max_seconds
    config = dict(seeds=list(seeds), split_seed=split_seed, steps=steps, min_steps=min_steps,
                  validation_interval=validation_interval, patience=patience,
                  batch_size=batch_size, width=width, layers=layers,
                  learning_rate=learning_rate, device=device, threads=threads,
                  max_seconds=max_seconds, heldout_style_test=bool(heldout_style_test),
                  decisions_per_round=decisions_per_round, data_seed=data_seed,
                  cell_bootstrap_samples=cell_bootstrap_samples,
                  validation_decisions=validation_decisions, memory_rounds=memory_rounds,
                  late_from=late_from, readout_rounds=readout_rounds,
                  readout_alpha=readout_alpha, parameter_tolerance=parameter_tolerance)
    root = Path(__file__).resolve().parents[1]
    report = {"schema_version": 1, "status": "running", "config": config, "runs": [],
              "gate": "pending", "rl_enabled": False,
              "gate_rule": ("v3 is supported when the adaptation improvement (masked-minus-memory "
                            "loss, rounds >= late_from minus the earlier rounds of the same "
                            "matches) is positive with a positive bootstrap lower bound in "
                            "every seed on held-out styles, and the masked control shows no "
                            "such late-round trend of its own."),
              "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in ("train/memory_experiment.py",
                                             "train/belief_memory.py",
                                             "train/belief_experiment.py",
                                             "train/belief_model.py", "train/belief_probe.py",
                                             "train/logs.py")}}
    write_json(output / "report.json", report)
    cells = memory_cells(late_from)

    def collate_fn(items, target):
        return memory_collate(items, target, memory_rounds)

    try:
        torch.set_num_threads(threads)
        rounds, provenance, fingerprint = load_dataset(
            directory, deadline, decisions_per_round=decisions_per_round, data_seed=data_seed)
        if set(seeds) & {provenance["seed"], provenance["training_seed"]}:
            raise ValueError("fit seeds must differ from collection and base training seeds")
        matches = attach_memory(rounds, memory_rounds)
        splits = split_rounds(rounds, split_seed, heldout_styles=bool(heldout_style_test))
        data = {name: examples(records) for name, records in splits.items()}
        if validation_decisions and len(data["validation"]) > validation_decisions:
            data["validation"] = random.Random(data_seed).sample(
                data["validation"], validation_decisions)
        report.update(dataset_sha256=fingerprint, collection=provenance,
                      dataset_bytes=dataset_bytes(rounds),
                      loaded_decisions=sum(len(r["obs"]) for r in rounds),
                      matches=len(matches),
                      rounds_per_match={"mean": len(rounds) / len(matches),
                                        "max": max(len(v) for v in matches.values())},
                      splits={name: {"matches": sorted({r["group"] for r in records}),
                                     "rounds": len(records),
                                     "decisions": sum(len(r["obs"]) for r in records),
                                     "decisions_used": len(data[name])}
                              for name, records in splits.items()})
        write_json(output / "report.json", report)
        for seed in seeds:
            check_time(deadline)
            torch.manual_seed(seed)
            models = matched_memory_models(rounds[0]["obs"].shape[1], width, layers,
                                           memory_rounds=memory_rounds,
                                           tolerance=parameter_tolerance)
            result = {"seed": seed, "models": {}}
            for name, model in models.items():
                model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                sampler = random.Random(seed)  # identical sampled batches for both models
                best, best_step, stale = float("inf"), 0, 0
                selected = None
                path = output / f"{name}-s{seed}.pt"
                curve = []
                model_started = time.monotonic()
                learn_seconds = 0.
                for step in range(1, steps + 1):
                    check_time(deadline)
                    update_started = time.monotonic()
                    model.train()
                    batch = collate_fn(sampler.choices(data["train"], k=batch_size), device)
                    logits = memory_forward(model, batch)
                    loss = F.cross_entropy(logits.reshape(-1, 3), batch["hidden"].reshape(-1))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
                    optimizer.step()
                    learn_seconds += time.monotonic() - update_started
                    if step % validation_interval and step != steps:
                        continue
                    metric = evaluate(model, data["validation"], device, batch_size, deadline,
                                      collate_fn=collate_fn, forward_fn=memory_forward)
                    entry = {"step": step, "last_train_batch_loss": float(loss.detach()),
                             "validation_log_loss": metric["log_loss"],
                             "learn_seconds": learn_seconds}
                    curve.append(entry)
                    with (output / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps({"seed": seed, "model": name, **entry}) + "\n")
                    if metric["log_loss"] < best:
                        best, best_step, stale = metric["log_loss"], step, 0
                        selected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        temporary = path.with_suffix(".tmp")
                        torch.save({"purpose": "belief_probe_only", "architecture": name,
                                    "model": selected, "config": config, "seed": seed,
                                    "selected_step": best_step,
                                    "dataset_sha256": fingerprint}, temporary)
                        temporary.replace(path)
                    else:
                        stale += 1
                    if step >= min_steps and stale >= patience:
                        break
                model.load_state_dict(selected)
                # Test data is first accessed here, after validation-only selection.
                final_test = evaluate(model, data["test"], device, batch_size, deadline,
                                      collate_fn=collate_fn, forward_fn=memory_forward,
                                      extra_cells=cells)
                readout = style_readout(model, splits["train"], splits["test"], device,
                                        alpha=readout_alpha, max_rounds=readout_rounds,
                                        seed=data_seed, deadline=deadline)
                result["models"][name] = {
                    "parameters": count_parameters(model), "steps_run": step,
                    "selected_step": best_step, "validation_log_loss": best,
                    "best_validation_step": best_step, "validation_evaluations": len(curve),
                    "stop_reason": "plateau" if step < steps else "step_cap",
                    "early_stopped": step < steps, "best_at_budget_end": best_step == steps,
                    "learn_seconds": learn_seconds,
                    "seconds_per_step": learn_seconds / max(1, step),
                    "elapsed_seconds": time.monotonic() - model_started,
                    "learning_curve": curve, "test": final_test,
                    "style_readout": readout, "checkpoint": str(path),
                    "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                write_json(output / f"seed-{seed}-partial.json", result)
                model.to("cpu")
            check_time(deadline)
            base = result["models"]["memory_masked"]["test"]
            candidate = result["models"]["memory"]["test"]
            result["memory_vs_masked"] = paired_improvement(base, candidate, split_seed)
            result["memory_vs_masked"]["cells"] = paired_cells(base, candidate, split_seed,
                                                               cell_bootstrap_samples)
            result["adaptation"] = adaptation_metric(base, candidate, split_seed,
                                                     cell_bootstrap_samples)
            result["self_adaptation"] = {
                name: self_adaptation(metrics["test"], split_seed, cell_bootstrap_samples)
                for name, metrics in result["models"].items()}
            result["evidence"] = memory_evidence(result)
            # The per-cell match tables are only needed for the pairing above.
            for metrics in result["models"].values():
                for cell in metrics["test"]["cells"].values():
                    cell["test_matches"] = len(cell.pop("matches"))
            report["runs"].append(result)
            write_json(output / "report.json", report)
        adaptation_supported = all(
            r["adaptation"].get("available")
            and r["adaptation"]["mean_adaptation_improvement"] > 0
            and r["adaptation"]["bootstrap_95_ci"][0] > 0 for r in report["runs"])
        control_flat = all(
            not r["self_adaptation"]["memory_masked"].get("available")
            or r["self_adaptation"]["memory_masked"]["bootstrap_95_ci"][0] <= 0
            for r in report["runs"])
        report["adaptation_supported"] = bool(adaptation_supported)
        report["control_shows_no_trend"] = bool(control_flat)
        check_time(deadline)
        report["gate"] = ("supports_v3_match_memory"
                          if len(seeds) >= 2 and adaptation_supported and control_flat
                          else "v3_not_yet_justified")
        report["status"] = "complete"
        report["interpretation"] = (
            "Shared dataset, splits and sampled batches across the two models and across "
            "optimizer seeds. Parameters are matched within 0.1% and the control differs "
            "only by the absence of the memory keys, so the contrast is memory, not "
            "capacity. Summaries are built from public tokens alone: the collection "
            "records no revealed remaining cards at a round end, so that part of "
            "DESIGN 7.3 is not implemented. A belief-loss adaptation gain is not a "
            "playing-strength result; DESIGN 9.1 item 6 still needs an arena run.")
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
    p.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    for name, default in (("split-seed", 20260928), ("steps", 6000), ("min-steps", 3000),
                          ("validation-interval", 1000), ("patience", 2), ("batch-size", 64),
                          ("width", 256), ("layers", 4), ("threads", 4),
                          ("decisions-per-round", 0), ("data-seed", 20260930),
                          ("cell-bootstrap-samples", 2000), ("validation-decisions", 0),
                          ("memory-rounds", 8), ("late-from", 5), ("readout-rounds", 512)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--learning-rate", type=float, default=.0003)
    p.add_argument("--readout-alpha", type=float, default=1.0)
    p.add_argument("--parameter-tolerance", type=float, default=.001)
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
