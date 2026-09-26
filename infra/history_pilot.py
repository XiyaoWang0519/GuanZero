"""Prepare a reviewable T4 kit, or run its CUDA-only workload on an owned pod.

Preparation is local/read-only with respect to the provider. Provisioning and
the independent provider guard remain explicit infra.runpod operations.
"""
from __future__ import annotations

import argparse
import gc
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import time

from infra.history_artifacts import ROOT, engine_digest, pack_source, sha256, source_identity


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(kit: Path) -> None:
    import torch
    from eval.duplicate import generate_deals
    from train.history_ppo import HistoryPPOConfig, REWARD_SEMANTICS

    kit.mkdir(parents=True, exist_ok=True)
    payload, evaluation = kit / "payload", kit / "evaluation"
    payload.mkdir(exist_ok=True)
    evaluation.mkdir(exist_ok=True)
    for name in ("local", "results"):
        (kit / name).mkdir(exist_ok=True)
    dev_path = evaluation / "development.json"
    if not dev_path.exists():
        seed = 2026092607
        deals = generate_deals(64, seed)
        write_json(dev_path, dict(seed=seed, policy_seed=2026092608,
                                  deals=[{k: getattr(d, k) for k in
                                          ("hands", "level", "team_levels", "owner", "leader")}
                                         for d in deals],
                                  match_seeds=[2026092700 + i for i in range(8)]))
    # The bytes are generated and committed once; neither preparation nor the
    # evaluator parses a test seed or generates a final-test deal.
    sealed = kit / "sealed-final-seed.bin"
    if not sealed.exists():
        with sealed.open("xb") as stream:
            os.chmod(sealed, 0o600)
            stream.write(secrets.token_bytes(32))
    baseline_paths = {
        "b11-main": ROOT / ".work/runpod-longrun-2026-09-25/payload/artifacts/b11-main.pt",
        "longrun-segment2-raw-endpoint": ROOT / ".work/runpod-seg2-2026-09-25/local/cont/final.pt",
    }
    baselines = []
    for name, path in baseline_paths.items():
        value = torch.load(path, map_location="cpu", weights_only=False)
        baselines.append(dict(name=name, path=str(path), sha256=sha256(path),
                              policy_config=value["policy_config"], stage=value["stage"],
                              training_seed=value["config"]["seed"],
                              action_mode=value["config"]["action_mode"],
                              tribute_policy=value.get("tribute_policy", "heuristic"),
                              settings=dict(sample=False, margin=0.0, precision="fp32",
                                            search=False, embedded_reference="evaluation only")))
    freeze = dict(engine_digest=engine_digest(), baselines=baselines,
                  development=dict(file=dev_path.name, sha256=sha256(dev_path)),
                  final_test=dict(commitment_sha256=sha256(sealed), opened=False,
                                  protocol="256-bit seed material sealed locally; no final-test evaluation in T4"),
                  selection="initial, after resume, and final bounded endpoint; no best-checkpoint search",
                  danlm="external calibration only; not a training gate, seat or reward")
    freeze_path = evaluation / "freeze.json"
    if freeze_path.exists() and json.loads(freeze_path.read_text()) != freeze:
        raise ValueError("existing evaluation freeze differs; refusing to replace it")
    write_json(freeze_path, freeze)
    source = pack_source(kit / "source.tar.gz")
    config = HistoryPPOConfig(width=64, layers=2, heads=4, num_envs=4, num_threads=2,
                              steps_per_update=64, epochs=1, minibatch_matches=2,
                              updates=200, torch_threads=2, seed=2026092602,
                              snapshot_updates=2, population_recent=4)
    write_json(payload / "cfg" / "pilot.json", asdict(config))
    manifest = dict(
        id=kit.name, status="prepared_awaiting_user_rental_approval",
        prepared_at=datetime.now(timezone.utc).isoformat(), source=source,
        engine_digest=engine_digest(), architecture=asdict(config.policy_config()),
        token_schema=dict(version=1, dimensions=186, round_index=True, phase=True,
                          forced_bit=False, private_tribute_flags=False, public_leftover_reveals=False),
        initialization=dict(actor="random", critic="independent random", heads="random",
                            teacher=None, old_optimizer=None, old_replay=None),
        observations="raw full-match public history; own obs/seat query; hidden counts critic only",
        actions="full canonical engine set, no reductions beyond canonical rule configuration",
        learning=dict(config=asdict(config), reward=REWARD_SEMANTICS,
                      entropy_schedule="constant 0.01", auxiliary_losses=[],
                      partial_rounds="carried across updates with original behaviour logp/version; discarded on resume"),
        population=dict(snapshot_updates=2, eligible_recent=4, snapshot_probability=0.5,
                        assignments="one guaranteed current seat; each other seat samples a recent snapshot with p=0.5",
                        pinning="per match, including across rounds/updates; retired snapshots retained while pinned",
                        audit="population.jsonl seat ids/snapshot hashes plus per-id decision counters"),
        recurrence=dict(kind="standard", loops=1, truncation="none", cache="none, full-prefix recomputation"),
        evaluation=dict(freeze_sha256=sha256(freeze_path), **freeze,
                        process="separate local CPU evaluator; assets are excluded from training upload"),
        compute=dict(provider="RunPod", proposed_gpu="NVIDIA GeForce RTX 4090", cloud="SECURE",
                     gpu_count=1, min_vcpu=4, min_ram_gb=24, disk_gb=30, volume_gb=0,
                     actual_quota="record host-facts.json after SSH", precision="fp32; TF32 disabled",
                     num_envs="selected from 4/8/16 by CUDA sweep, no old MLP default",
                     resident_snapshots="recent 4 plus snapshots pinned by active matches",
                     evaluation_gpu_share=0),
        throughput_gate=dict(prefixes="actual growing prefixes near 144 and >=720; retain all update metrics",
                             min_collection_retention=0.5, max_gpu_memory_fraction=0.7,
                             selection="highest long-prefix end-to-end learned rows/sec among passing cases",
                             failure="stop; implement/profile batched KV cache with weight-update rebuild before further training"),
        lifecycle=dict(proposed_budget_usd=1.2, max_hours_from_create=1.5, max_hourly_usd=0.8,
                       workload_seconds=2700, sweep_case_seconds=240, pilot_seconds=1500,
                       save_every_updates=1, sync_seconds=60, max_metric_stall_seconds=180,
                       max_recoverable_loss_seconds=600, shutdown_reserve_seconds=180,
                       guards="local infra.runpod guard plus remote process watchdog; not provider-enforced TTL",
                       credentials="provider key stays local, never uploaded",
                       teardown="hash-verify final artifacts, then owned pod DELETE and confirmed_gone; independent pods/spend query",
                       sync_failure="retry within remaining cap; guard deletes at deadline even if transfer fails",
                       artifact_destination=str(kit / "local")),
        acceptance="legal finite actions; encoder/critic updates; actual snapshot participation; optimizer/RNG/population resume; privacy tests; throughput/memory receipts",
        claim="bounded cold-start engineering and learning diagnostic, no playing-strength claim",
    )
    write_json(kit / "run-manifest.json", manifest)
    shutil.copy2(kit / "run-manifest.json", payload / "run-manifest.json")
    for script in ("setup", "run"):
        shutil.copy2(ROOT / "infra" / f"history_{script}.sh", payload / f"{script}.sh")
    print(json.dumps(dict(kit=str(kit), source_sha256=source["source_sha256"],
                          archive_sha256=source["archive_sha256"], approval="required")))


def health(lines: list[dict]) -> dict:
    if not lines:
        return dict(healthy=False, reason="no metrics")
    latest = lines[-1]
    numeric = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
               "actor_grad_norm", "encoder_grad_norm", "critic_grad_norm")
    for line in lines:
        if line["update_samples"] > 0:
            if any(line.get(k) is None or not math.isfinite(line[k]) for k in numeric):
                return dict(healthy=False, reason="non-finite/missing learner metrics")
            if min(line[k] for k in ("actor_grad_norm", "encoder_grad_norm", "critic_grad_norm")) <= 0:
                return dict(healthy=False, reason="optimizer has no actor/encoder/critic gradient")
            if line["approx_kl"] > 0.5 or line["clip_fraction"] > 0.95:
                return dict(healthy=False, reason="PPO ratio instability")
    return dict(healthy=True, update=latest["update"], rounds=latest["rounds"],
                learned_rows=sum(r["update_samples"] for r in lines),
                snapshot_decisions=sum(v for k, v in latest["population"]["decisions"].items() if int(k) != 0),
                mean_prefix=latest["mean_prefix"], collect_dps=latest["decisions_per_sec"],
                learn_dps=latest["learn_decisions_per_sec"],
                gpu_peak_bytes=latest["cuda_peak_reserved_bytes"])


def remote(manifest_path: Path, output: Path, deadline_epoch: float) -> None:
    import torch
    from infra.cpu_budget import host_facts
    from train.history_ppo import HistoryPPOConfig, HistoryTrainer

    manifest = json.loads(manifest_path.read_text())
    if source_identity()["source_sha256"] != manifest["source"]["source_sha256"]:
        raise ValueError("manifest/source mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing CPU fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=True)
    facts = host_facts()
    write_json(output / "host-facts.json", facts)
    if facts["usable_cpus"] < manifest["compute"]["min_vcpu"]:
        raise RuntimeError("effective CPU quota below the manifest minimum")
    if time.time() + 900 > deadline_epoch:
        raise RuntimeError("insufficient remaining time for sweep and safe sync")
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_history_model.py",
                    "tests/test_history_rollout.py", "tests/test_history_ppo.py",
                    "tests/test_history_population.py", "tests/test_history_eval.py"], check=True)
    subprocess.run([sys.executable, "-m", "bench.history_ppo", "--output", str(output / "sweep"),
                    "--case-seconds", str(manifest["lifecycle"]["sweep_case_seconds"])], check=True)
    selection = json.loads((output / "sweep/selection.json").read_text())
    config = HistoryPPOConfig(**manifest["learning"]["config"])
    config.num_envs = selection["selected"]["envs"]
    write_json(output / "resolved-config.json", asdict(config))
    trainer = HistoryTrainer(config, output / "pilot", device="cuda")
    trainer.save(output / "initial.pt")
    end = min(time.time() + manifest["lifecycle"]["pilot_seconds"], deadline_epoch - 180)
    lines, resumed = [], False
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopped and time.time() < end and trainer.progress["updates"] < config.updates:
        started = time.monotonic()
        line = trainer.update()
        lines.append(line)
        receipt = health(lines)
        receipt["update_seconds"] = time.monotonic() - started
        write_json(output / "health.json", receipt)
        print(json.dumps(receipt), flush=True)
        if not receipt["healthy"] or receipt["update_seconds"] > 180:
            raise RuntimeError("pilot health/throughput gate failed; retaining last valid checkpoint")
        trainer.save()
        if trainer.progress["updates"] in (2, 4) or trainer.progress["updates"] % 10 == 0:
            trainer.save(output / f"update-{trainer.progress['updates']:06d}.pt")
        if trainer.progress["updates"] >= 2 and trainer.population.models and not resumed:
            checkpoint = trainer.save()
            before = dict(trainer.progress)
            lineage = trainer.lineage
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            trainer = HistoryTrainer(config, output / "pilot", device="cuda", resume=checkpoint)
            if trainer.progress != before or trainer.lineage != lineage:
                raise RuntimeError("resume progress/lineage mismatch")
            resumed = True
            write_json(output / "resume.json", dict(restored=True, progress=before,
                                                     discarded_partial_rounds=True,
                                                     histories_rebuilt=True, persistent_cache=False))
    trainer.save(output / "final.pt")
    receipt = health(lines)
    receipt.update(resumed=resumed, complete=bool(resumed and receipt.get("learned_rows", 0) > 0
                                                 and receipt.get("snapshot_decisions", 0) > 0),
                   claim="engineering pilot only; no strength claim")
    write_json(output / "receipt.json", receipt)
    if not receipt["complete"]:
        raise RuntimeError("pilot ended without required learning/resume/snapshot participation")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--kit", type=Path, required=True)
    run = sub.add_parser("remote")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.kit.resolve())
    else:
        remote(args.manifest, args.output, args.deadline_epoch)


if __name__ == "__main__":
    main()
