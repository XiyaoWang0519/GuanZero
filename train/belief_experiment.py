"""Validation-selected, match-held-out Transformer belief experiment.

The original small probe is preserved in belief_probe.py. This stricter runner
uses only provenance-verified, frozen-policy architecture-probe collections.
It never selects checkpoints or hyperparameters using the final test set.

Result breakdowns. Besides the overall held-out log loss and the loss by stage
of round and relative seat, every fit reports `cells`: the test loss inside the
match-relative round bins ``0``, ``1``, ``2-3``, ``4-5`` and ``6+`` (habits need
several rounds of a match to show, so early rounds are separated one by one and
later ones pooled), inside the style region of the test match (``train``,
``heldout`` or ``unknown``) and by whether the seat whose hand is predicted is
bot-driven or policy-driven. Stage, relative seat, teammate/opponent relation,
and stage-by-seat/relation cells provide the same paired comparisons.
Schema 1 logs carry no round index, style region or
per-seat driver, so their cells are labelled ``unknown`` and stay loadable. The
paired flat-vs-history and no-history-vs-history differences are bootstrapped
inside each cell with whole matches as the resampling unit.

Memory. Rounds are held in RAM at their on-disk dtypes: the v1 encoder is
binary, so `obs` stays `uint8` (1,849 B per decision) instead of being widened
to float32 (7,396 B), and batches are cast to float32 only in `collate`. A
100,000-round styled collection is about 10M decisions, so the resident cost is
roughly 10M x (1849 + 162) B = 20 GB plus about 1.6 GB of public tokens and 1 GB
of index tuples - too close to a 32 GB budget to be safe. `--decisions-per-round`
caps how many decisions of each round are kept, deterministically and without
dropping any round, so the public token stream and history prefixes of every
kept decision remain exact. At the collection's ~100 decisions per round, a cap
of 20 loads about 2M decisions, roughly 4.2 GB, and a cap of 40 about 8.4 GB.
`dataset_bytes` in the report records the measured resident size.

Plateau cost. Validation runs every `--validation-interval` steps and stops the
fit once `--patience` consecutive validations fail to improve, so `--steps` can
be set far above the expected plateau. `--validation-decisions` caps how many
validation decisions each of those evaluations scores, which keeps a long run
affordable; the test set is never subsampled. Each fit records `stop_reason`
(``plateau`` or ``step_cap``) and `best_validation_step`.
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
from train.belief_probe import FlatBelief, collate, count_parameters, examples
from train.logs import TOKEN_DIM
from train.tribute_data import engine_source_digest


def check_time(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("belief experiment exceeded its cooperative walltime bound")


DRIVER_LABELS = {0: "policy", 1: "bot"}
DRIVER_UNKNOWN = -1
# Match-relative round bins. Early rounds are separated because an opponent
# model has seen the least there; later rounds are pooled to keep cells large.
ROUND_BINS = ((0, 0, "0"), (1, 1, "1"), (2, 3, "2-3"), (4, 5, "4-5"), (6, None, "6+"))


def round_bin(index: int) -> str:
    """Bin a match-relative round index; schema 1 logs have no index at all."""
    if index < 0:
        return "unknown"
    for low, high, label in ROUND_BINS:
        if index >= low and (high is None or index <= high):
            return label
    return "unknown"


def driver_label(code: int) -> str:
    """Name the driver of a seat: the frozen policy, a styled bot, or unknown."""
    return DRIVER_LABELS.get(int(code), "unknown")


def dataset_bytes(rounds: list[dict]) -> int:
    """Resident bytes of the loaded round arrays, at their on-disk dtypes."""
    keys = ("obs", "hidden", "seat", "prefix", "tokens", "driver")
    return int(sum(r[key].nbytes for r in rounds for key in keys))


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_dataset(directory: Path, deadline: float | None = None, *,
                 decisions_per_round: int = 0, data_seed: int = 20260930
                 ) -> tuple[list[dict], dict, str]:
    """Load, validate and fingerprint one collection, optionally subsampled.

    A positive `decisions_per_round` keeps at most that many decisions of each
    round, chosen without replacement from a generator keyed by `data_seed` and
    the file name, so the choice is reproducible. Whole rounds are always kept
    and the public token stream is never subsampled, so the history prefix of
    every surviving decision is still exact. Validation and the provenance
    decision count both run on the full round, before any row is dropped.
    """
    if decisions_per_round < 0:
        raise ValueError("decisions_per_round must not be negative")
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
    total_decisions = 0
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
            r["round_index"] = int(data["round_index"]) if version >= 2 else -1
            r["round_bin"] = round_bin(r["round_index"])
            r["seat_driver"] = (data["seat_driver"].copy() if version >= 2
                                else np.full(4, DRIVER_UNKNOWN, np.int64))
            # Kept for the v3 style readout; never an input to any belief model.
            r["styles"] = (data["styles"].copy() if version >= 2
                           else np.full((4, 1), np.nan, np.float32))
            r["env_id"] = int(data["env_id"]) if version >= 2 else -1
            r["match_id"] = int(data["match_id"]) if version >= 2 else -1
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
        total_decisions += n
        if decisions_per_round and n > decisions_per_round:
            seed = int.from_bytes(hashlib.sha256(
                f"{data_seed}:{path.name}".encode()).digest()[:8], "big")
            keep = np.sort(np.random.default_rng(seed).choice(
                n, decisions_per_round, replace=False))
            for key in ("obs", "hidden", "seat", "prefix", "driver"):
                r[key] = r[key][keep].copy()
        rounds.append(r)
    if (len(rounds) != provenance.get("collected_rounds") or not rounds
            or len(rounds) != provenance.get("requested_rounds")
            or total_decisions != provenance.get("collected_decisions")
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


def accumulate(cells: dict, name: str, group: str, value: float) -> None:
    entry = cells.setdefault(name, {}).setdefault(group, [0., 0])
    entry[0] += value
    entry[1] += 1


@torch.inference_mode()
def evaluate(model, items: list, device: str, batch_size: int,
             deadline: float | None = None, *, collate_fn=None, forward_fn=None,
             extra_cells=None) -> dict:
    """Held-out loss overall, by stage and seat, and in the task 3 cells.

    `cells` carries one entry per breakdown cell with its own per-match table,
    so `paired_cells` can bootstrap a paired difference inside each cell with
    whole matches as the resampling unit. Round-bin and style-region cells
    and stage cells average a decision's three hidden hands; target-driver,
    target-seat/relation and stage-by-seat/relation cells score each predicted seat on its
    own, so their `decisions` count seat targets.

    `collate_fn` and `forward_fn` let a model with extra inputs (the v3 match
    memory) reuse this breakdown unchanged; `extra_cells(record, index)` names
    further decision-level cells, which is how the v3 runner adds its exact
    round-index and early/late adaptation cells.
    """
    model.eval()
    sums, counts = np.zeros((3, 3)), np.zeros((3, 3), dtype=np.int64)
    matches: dict[str, list[float]] = {}
    cells: dict[str, dict[str, list[float]]] = {}
    for start in range(0, len(items), batch_size):
        check_time(deadline)
        part = items[start:start + batch_size]
        batch = (collate(part, device,
                         include_history=not (isinstance(model, FlatBelief)
                                              or getattr(model, "no_history", False)))
                 if collate_fn is None else collate_fn(part, device))
        logits = (model(batch["obs"], batch["tokens"], batch["lengths"], batch["seat"])
                  if forward_fn is None else forward_fn(model, batch))
        losses = F.cross_entropy(logits.reshape(-1, 3), batch["hidden"].reshape(-1),
                                 reduction="none").reshape(-1, 3, 54).mean(-1).cpu().numpy()
        if not np.isfinite(losses).all():
            raise FloatingPointError("non-finite belief evaluation loss")
        stages = np.digitize(batch["obs"][:, 216:648].sum(-1).cpu().numpy(), [36, 72])
        for stage in range(3):
            sums[stage] += losses[stages == stage].sum(0)
            counts[stage] += (stages == stage).sum()
        seats = batch["seat"].cpu().numpy()
        for row, ((record, index), loss) in enumerate(zip(part, losses)):
            group, mean = record["group"], float(loss.mean())
            entry = matches.setdefault(group, [0., 0])
            entry[0] += mean
            entry[1] += 1
            accumulate(cells, "round_bin:" + record.get("round_bin", "unknown"), group, mean)
            accumulate(cells, "style_region:" + record.get("style_region", "unknown"), group, mean)
            stage = ("early", "middle", "late")[stages[row]]
            accumulate(cells, "stage:" + stage, group, mean)
            for name in (() if extra_cells is None else extra_cells(record, index)):
                accumulate(cells, name, group, mean)
            drivers = record.get("seat_driver")
            for j in range(3):
                # Relative target j is lho, partner then rho of the acting seat.
                code = (DRIVER_UNKNOWN if drivers is None
                        else drivers[(int(seats[row]) + 1 + j) % 4])
                accumulate(cells, "target_driver:" + driver_label(code), group, float(loss[j]))
                relative_seat = ("lho", "partner", "rho")[j]
                accumulate(cells, "target_seat:" + relative_seat, group, float(loss[j]))
                accumulate(cells, f"stage:{stage}|target_seat:{relative_seat}", group, float(loss[j]))
                relation = "teammate" if j == 1 else "opponent"
                accumulate(cells, "target_relation:" + relation, group, float(loss[j]))
                accumulate(cells, f"stage:{stage}|target_relation:{relation}", group, float(loss[j]))
    return {"log_loss": float(sums.sum() / counts.sum()), "decisions": len(items),
            "matches": {g: {"log_loss": total / n, "decisions": n}
                        for g, (total, n) in sorted(matches.items())},
            "by_stage_and_seat": {
                stage: {seat: {"log_loss": float(sums[i, j] / counts[i, j]) if counts[i, j] else None,
                               "decisions": int(counts[i, j])}
                        for j, seat in enumerate(("lho", "partner", "rho"))}
                for i, stage in enumerate(("early", "middle", "late"))},
            "cells": {name: {
                "log_loss": sum(t for t, _ in table.values()) / sum(n for _, n in table.values()),
                "decisions": sum(n for _, n in table.values()),
                "matches": {g: {"log_loss": t / n, "decisions": n}
                            for g, (t, n) in sorted(table.items())}}
                for name, table in sorted(cells.items())}}


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


def paired_cells(reference: dict, candidate: dict, seed: int, samples: int = 2000) -> dict:
    """Paired bootstrap inside every breakdown cell the two evaluations share.

    Both models score the identical test decisions, so a cell present in both
    always holds the same matches; a cell that somehow does not is skipped
    rather than paired across different denominators.
    """
    paired = {}
    for name in sorted(set(reference["cells"]) & set(candidate["cells"])):
        ref, cand = reference["cells"][name], candidate["cells"][name]
        if set(ref["matches"]) != set(cand["matches"]):
            continue
        paired[name] = paired_improvement(ref, cand, seed, samples)
    return paired


def run(directory: Path, output: Path, *, seeds=(31, 32, 33), split_seed=20260928,
        steps=6000, min_steps=3000, validation_interval=1000, patience=2,
        batch_size=64, width=128, layers=2, learning_rate=.0003,
        device="cpu", threads=4, max_seconds=2400, heldout_style_test=False,
        decisions_per_round=0, data_seed=20260930, cell_bootstrap_samples=2000,
        validation_decisions=0) -> dict:
    if (not seeds or len(set(seeds)) != len(seeds) or min(steps, min_steps, validation_interval,
            patience, batch_size, threads, cell_bootstrap_samples) <= 0 or min_steps > steps
            or decisions_per_round < 0 or validation_decisions < 0
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
                  heldout_style_test=bool(heldout_style_test),
                  decisions_per_round=decisions_per_round, data_seed=data_seed,
                  cell_bootstrap_samples=cell_bootstrap_samples,
                  validation_decisions=validation_decisions)
    report = {"schema_version": 1, "status": "running", "config": config, "runs": [],
              "gate": "pending", "rl_enabled": False,
              "source_sha256": {name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                                  for name in ("train/belief_experiment.py", "train/belief_model.py",
                                               "train/belief_probe.py", "train/logs.py")}}
    write_json(output / "report.json", report)
    try:
        torch.set_num_threads(threads)
        rounds, provenance, fingerprint = load_dataset(
            directory, deadline, decisions_per_round=decisions_per_round, data_seed=data_seed)
        if set(seeds) & {provenance["seed"], provenance["training_seed"]}:
            raise ValueError("fit seeds must differ from collection and base training seeds")
        splits = split_rounds(rounds, split_seed, heldout_styles=bool(heldout_style_test))
        data = {name: examples(records) for name, records in splits.items()}
        # A plateau needs many validations, so the validation set may be capped.
        # The test set is never subsampled; only model selection sees this cap.
        if validation_decisions and len(data["validation"]) > validation_decisions:
            data["validation"] = random.Random(data_seed).sample(
                data["validation"], validation_decisions)
        report.update(dataset_sha256=fingerprint, collection=provenance,
                      dataset_bytes=dataset_bytes(rounds),
                      loaded_decisions=sum(len(r["obs"]) for r in rounds),
                      splits={name: {"matches": sorted({r['group'] for r in records}),
                                     "rounds": len(records),
                                     "decisions": sum(len(r["obs"]) for r in records),
                                     "decisions_used": len(data[name])}
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
                path = output / f"{name}-s{seed}.pt"
                curve = []
                model_started = time.monotonic()
                learn_seconds = 0.
                for step in range(1, steps + 1):
                    check_time(deadline)
                    update_started = time.monotonic()
                    model.train()
                    batch = collate(sampler.choices(data["train"], k=batch_size), device,
                                    include_history=name == "v2")
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
                        # Publish every validation improvement before another update or test.
                        temporary = path.with_suffix(".tmp")
                        torch.save({"purpose": "belief_probe_only", "architecture": name, "model": selected,
                                    "config": config, "seed": seed, "selected_step": best_step,
                                    "dataset_sha256": fingerprint}, temporary)
                        temporary.replace(path)
                    else:
                        stale += 1
                    if step >= min_steps and stale >= patience:
                        break
                model.load_state_dict(selected)
                # Test data is first accessed here, after validation-only selection.
                final_test = evaluate(model, data["test"], device, batch_size, deadline)
                result["models"][name] = {"parameters": count_parameters(model),
                    "steps_run": step, "selected_step": best_step, "validation_log_loss": best,
                    "best_validation_step": best_step, "validation_evaluations": len(curve),
                    "stop_reason": "plateau" if step < steps else "step_cap",
                    "early_stopped": step < steps, "best_at_budget_end": best_step == steps,
                    "learn_seconds": learn_seconds, "elapsed_seconds": time.monotonic() - model_started,
                    "learning_curve": curve, "test": final_test, "checkpoint": str(path),
                    "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                write_json(output / f"seed-{seed}-partial.json", result)
                model.to("cpu")
            for reference, key in (("v1", "v2_vs_v1"), ("no_history", "v2_vs_no_history")):
                check_time(deadline)
                base, candidate = result["models"][reference]["test"], result["models"]["v2"]["test"]
                result[key] = paired_improvement(base, candidate, split_seed)
                result[key]["cells"] = paired_cells(base, candidate, split_seed,
                                                    cell_bootstrap_samples)
            # The per-cell match tables are only needed for the pairing above.
            for metrics in result["models"].values():
                for cell in metrics["test"]["cells"].values():
                    cell["test_matches"] = len(cell.pop("matches"))
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
                          ("width", 128), ("layers", 2), ("threads", 4),
                          ("decisions-per-round", 0), ("data-seed", 20260930),
                          ("cell-bootstrap-samples", 2000), ("validation-decisions", 0)):
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
