"""Cross-play: does a model cooperate only with copies of itself? (STAGE_B_TODO B8x)

Every other evaluation seats the same policy at both seats of a team. Here a
team is a lineup `(a, b)`: `a` at seat `s`, its partner `b` at seat `s + 2`.
Each deal is played twice with the teams swapping seats (eval/duplicate.py),
so hands and leader are identical and the score stays paired.

For a model X, a list of partners P and a fixed reference team R (default two
copies of the M1 final), on one shared deal set:

- `X+P`: lineup (X, P) against R, mean net levels per round;
- `X+X`: lineup (X, X) against R on the same deals (equals eval/duplicate.py);
- `P+P`: lineup (P, P) against R on the same deals;
- `vs_self = (X+P) - (X+X)`: what X loses when its partner is not itself;
- `partner_lift = (X+P) - (P+P)`: whether X lifts or drags that partner.

Both differences are per deal, with a 95% percentile bootstrap over deals.
Deal `i` uses play seed `seed + i` in every lineup, so results do not depend
on the worker count.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np

from .duplicate import bootstrap_interval, generate_deals, play_duplicate_teams, summarize_duplicates
from .policies import Policy, load_policy
from .stage_b_baseline import _init_worker, file_sha256, host_info, resolve_path

DEFAULT_REFERENCE = Path(".work/runpod/artifacts/pilot/final.pt")
DEFAULT_OUTPUT = Path("docs/reports/stage-b-crossplay.json")
# "m1" stands for the reference checkpoint path (the M1 final by default).
DEFAULT_PARTNERS: tuple[str, ...] = (
    "greedy", "styled:bomb-happy", "styled:bomb-shy", "styled:high-lead",
    "styled:low-lead", "m1",
)
SCRIPTED = ("random", "greedy")


def canonical_spec(spec: str, m1: str | None = None) -> str:
    """Resolve checkpoint paths (also through the main checkout) and the `m1` alias."""
    if spec == "m1":
        if m1 is None:
            raise ValueError("the m1 alias needs a reference checkpoint path")
        return m1
    if spec in SCRIPTED or spec.startswith("styled:"):
        return spec
    head, path = "", spec
    if spec.startswith("sample:") or spec.startswith("sample="):
        head, _, path = spec.partition(":")
        head += ":"
    return head + str(resolve_path(Path(path)))


def describe(spec: str, policy: Policy) -> dict:
    if spec in SCRIPTED:
        return {"spec": spec, "name": policy.name, "kind": spec}
    if spec.startswith("styled:"):
        from train.styles import FIXED_STYLES

        return {"spec": spec, "name": policy.name, "kind": "styled",
                "style_changes_from_neutral": FIXED_STYLES[spec[len("styled:"):]]}
    path = Path(spec.partition(":")[2] if spec.startswith("sample") else spec)
    return {"spec": spec, "name": policy.name, "kind": "checkpoint",
            "stage": getattr(policy, "stage", None),
            "model_digest": getattr(policy, "checkpoint_id", None),
            "file_sha256": file_sha256(path), "training_seed": getattr(policy, "training_seed", None)}


# Per-process caches: each pool worker loads a policy and the deal set once.
_POLICIES: dict[tuple[str, str], Policy] = {}
_DEALS: dict[tuple[int, int], list] = {}


def _policy(spec: str, device: str) -> Policy:
    if (spec, device) not in _POLICIES:
        _POLICIES[(spec, device)] = load_policy(spec, device)
    return _POLICIES[(spec, device)]


def _run_chunk(task: tuple) -> tuple[int, list]:
    lineup, reference, device, deals, seed, start, stop = task
    team = (_policy(lineup[0], device), _policy(lineup[1], device))
    opponents = (_policy(reference[0], device), _policy(reference[1], device))
    if (deals, seed) not in _DEALS:
        _DEALS[(deals, seed)] = generate_deals(deals, seed)
    deal_set = _DEALS[(deals, seed)]
    return start, [play_duplicate_teams(deal_set[i], team, opponents, seed + i)
                   for i in range(start, stop)]


def _chunks(total: int, workers: int) -> list[tuple[int, int]]:
    size = max(1, -(-total // max(1, workers * 4)))
    return [(start, min(start + size, total)) for start in range(0, total, size)]


def evaluate_lineup(lineup: tuple[str, str], reference: tuple[str, str], deals: int, seed: int,
                    bootstrap_samples: int, device: str = "cpu",
                    pool: ProcessPoolExecutor | None = None, workers: int = 1) -> dict:
    """Lineup (a at s, b at s+2) against the reference lineup, over `deals` duplicate deals.

    The report keeps `pair_scores` (per deal, in deal order) for paired differences.
    """
    tasks = [(lineup, reference, device, deals, seed, a, b) for a, b in _chunks(deals, workers)]
    started = time.monotonic()
    parts = list(pool.map(_run_chunk, tasks)) if pool else [_run_chunk(task) for task in tasks]
    scores = [score for _, part in sorted(parts, key=lambda item: item[0]) for score in part]
    report = summarize_duplicates(scores, seed, bootstrap_samples)
    report.pop("results")
    report["lineup"] = list(lineup)
    report["seconds"] = time.monotonic() - started
    return report


def paired_difference(a: dict, b: dict, seed: int, bootstrap_samples: int) -> dict:
    diff = np.asarray(a["pair_scores"]) - np.asarray(b["pair_scores"])
    return {"mean": float(diff.mean()), "bootstrap_95_ci": list(bootstrap_interval(diff, seed, bootstrap_samples)),
            "deals": int(diff.size)}


def _public(result: dict) -> dict:
    keep = ("lineup", "deals", "rounds", "mean_net_levels_per_round", "bootstrap_95_ci",
            "double_win_rate", "opponent_double_win_rate", "banker_rate", "seconds")
    return {key: result[key] for key in keep}


def run_crossplay(model: str, partners: tuple[str, ...], reference: str, deals: int, seed: int,
                  bootstrap_samples: int = 2000, device: str = "cpu", threads: int = 1,
                  workers: int = 1, log=print) -> dict:
    """`model`, `partners` and `reference` are load_policy specs (`m1` = `reference`)."""
    import torch

    if deals < 1 or bootstrap_samples < 1 or threads < 1 or workers < 1:
        raise ValueError("require deals, bootstrap samples, threads and workers > 0")
    started = time.monotonic()
    torch.set_num_threads(threads)
    reference = canonical_spec(reference)
    model = canonical_spec(model, reference)
    partner_specs: list[tuple[str, str]] = []
    for raw in partners:
        spec = canonical_spec(raw, reference)
        if spec not in [s for _, s in partner_specs]:
            partner_specs.append((raw, spec))
    ref_team = (reference, reference)
    report = {
        "schema_version": 1, "task": "STAGE_B_TODO B8x", "protocol": "internal-house",
        "rules": "gd.RuleConfig.house()",
        "model": describe(model, load_policy(model, device)),
        "reference_team": {"lineup": list(ref_team), **describe(reference, load_policy(reference, device))},
        "seating": ("lineup (a, b): a at seat s, b at seat s + 2; s = 0 in leg 1 and 1 in the swapped "
                    "leg; the reference team takes the other two seats"),
        "seeds": {"deal_seed": seed, "play_seed_rule": "deal i uses seed + i in every lineup",
                  "bootstrap_seed": seed},
        "requested": {"deals": deals, "bootstrap_samples": bootstrap_samples,
                      "partners": list(partners)},
        "host": host_info(),
        "definitions": {
            "mean_net_levels_per_round": "(lineup net level gain in leg 1 + in leg 2) / 2, vs the reference team",
            "X+P": "lineup (model, partner)", "X+X": "lineup (model, model)", "P+P": "lineup (partner, partner)",
            "vs_self": "(X+P) - (X+X), per deal", "partner_lift": "(X+P) - (P+P), per deal",
            "intervals": "95% percentile bootstrap over whole deals",
            "double_win_rate": "fraction of legs where the lineup finishes first and second",
        },
        "partners": {},
    }
    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_init_worker, initargs=(threads,))
    cache: dict[tuple[str, str], dict] = {}

    def lineup(a: str, b: str) -> dict:
        if (a, b) not in cache:
            cache[(a, b)] = evaluate_lineup((a, b), ref_team, deals, seed, bootstrap_samples,
                                            device, pool, workers)
            result = cache[(a, b)]
            log(f"[crossplay] ({Path(a).name}, {Path(b).name}) vs reference: "
                f"{result['mean_net_levels_per_round']:+.4f} CI [{result['bootstrap_95_ci'][0]:+.4f}, "
                f"{result['bootstrap_95_ci'][1]:+.4f}] ({result['seconds']:.0f}s)")
        return cache[(a, b)]

    try:
        self_pair = lineup(model, model)
        report["self_pair"] = _public(self_pair)
        for raw, spec in partner_specs:
            xp, pp = lineup(model, spec), lineup(spec, spec)
            report["partners"][raw] = {
                "partner": describe(spec, load_policy(spec, device)),
                "X+P": _public(xp), "P+P": _public(pp),
                "vs_self": paired_difference(xp, self_pair, seed, bootstrap_samples),
                "partner_lift": paired_difference(xp, pp, seed, bootstrap_samples),
            }
            entry = report["partners"][raw]
            log(f"  {raw}: vs_self {entry['vs_self']['mean']:+.4f} "
                f"CI [{entry['vs_self']['bootstrap_95_ci'][0]:+.4f}, {entry['vs_self']['bootstrap_95_ci'][1]:+.4f}]"
                f"; partner_lift {entry['partner_lift']['mean']:+.4f}")
    finally:
        if pool is not None:
            pool.shutdown()
    report["runtime"] = {"seconds": time.monotonic() - started, "lineups_played": len(cache),
                         "workers": workers, "torch_threads_per_worker": threads}
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="load_policy spec of X, usually a checkpoint path")
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE),
                        help="spec of the opposing team's two seats (default: the M1 final)")
    parser.add_argument("--partners", nargs="+", default=list(DEFAULT_PARTNERS),
                        help="partner specs; 'm1' is the reference checkpoint")
    parser.add_argument("--extra-partners", nargs="*", default=[],
                        help="extra partner specs (checkpoint paths, sample=<T>:<path>) appended to --partners")
    parser.add_argument("--deals", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1),
                        help="evaluation processes; results do not depend on it")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads per worker")
    parser.add_argument("--out", "--output", dest="output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.deals < 1 or args.bootstrap_samples < 1 or args.threads < 1 or args.workers < 1:
        parser.error("require deals, bootstrap samples, threads and workers > 0")
    report = run_crossplay(args.model, tuple(args.partners + args.extra_partners), args.reference,
                           args.deals, args.seed, args.bootstrap_samples, args.device,
                           args.threads, args.workers)
    report["command"] = ["python", "-m", "eval.crossplay", *(argv if argv is not None else sys.argv[1:])]
    output = args.output if args.output.is_absolute() else Path.cwd() / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Cross-play report written to {output}")


if __name__ == "__main__":
    main()
