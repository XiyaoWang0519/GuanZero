"""Fixed-budget exploration screen and T7 arm-B continuation.

Screen: three arms on the T7 arm-B recipe (auxiliary response loss, two PPO
epochs), each trained from random initialization for exactly ``UPDATES``
updates of ``DECISIONS_PER_UPDATE`` all-seat decisions, three new paired seeds:
control (entropy 0.01), entropy (0.03) and epsilon (0.02 learner-seat floor,
temperature 1, weight cap 1). Continuation: the three T7 arm-B lineages resume
from their own ``train/latest.pt`` and train ``CONTINUATION_UPDATES`` more
updates with the recipe unchanged. They resume under this (newer) source with
``allow_source_change``: the weights, engine and token schema must match, and
the new trainer's defaults reproduce the T7 sampler and loss bit for bit.

prepare: local freeze only; no provider operations, no approval.
workload / preflight / job: an owned Linux pod with a provider deadline.
evaluate: local CPU inference on downloaded fixed endpoints; never trains.
No subcommand creates a pod or reads provider credentials.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from infra.history_artifacts import ROOT, engine_digest, pack_source, sha256, source_identity
from infra.history_pilot import write_json

SEEDS = (2026092721, 2026092722, 2026092723)
ARMS = {"control": {}, "entropy": {"entropy": 0.03}, "epsilon": {"rollout_epsilon": 0.02}}
PRIMARY = (("entropy", "control"), ("epsilon", "control"))
UPDATES = 2048
DECISIONS_PER_UPDATE = 2048          # 32 environments x 64 steps, all four seats
SNAPSHOT_UPDATES = 128               # weights-only snapshots for diagnostics
LATEST_UPDATES = 32                  # resumable latest.pt cadence
DIAGNOSTIC_UPDATES = (512, 1024, 1536)
ALIGNMENT_UPDATE = 1024              # about the T7 arm-B endpoints (1.93M-2.05M decisions)
CONTINUATION_UPDATES = 1024
MIN_EFFECT = 0.3                     # levels/round; T7 paired seed noise 0.2-0.35
T7 = ROOT / ".work/history-response-speed-2026-09-26"
T7_SEEDS = (2026092801, 2026092802, 2026092803)
T7_SOURCE_SHA256 = "d730650da1edd3ad360e1d9bb1babc798772fcf4296e6726b81c01875b8700dd"
TESTS = [f"tests/test_history_{name}.py" for name in
         ("model", "rollout", "ppo", "population", "inference", "eval", "response",
          "devices", "exploration", "budget_experiment")]
SPIKE_WINDOW = 10                    # ratio spikes are fatal only as a burst in this window
SPIKE_BURST = 3
HOURLY_USD = 0.73                    # RTX PRO 4500 Blackwell, secure cloud, T7 ledger
# Lifecycle caps are provisional until the preflight measures throughput.
CAPS = {"screen": (3.60, 4.2), "continuation": (1.90, 2.5), "preflight": (0.60, 0.8)}


def rebase(path: str) -> Path:
    """Baseline paths frozen before the ~/Developer move, re-rooted at ROOT."""
    marker = "/.work/"
    return ROOT / path[path.index(marker) + 1:] if marker in path else Path(path)


def screen_config(seed: int) -> dict:
    """The T7 arm-B configuration with a new seed and the fixed update budget."""
    manifest = json.loads((T7 / f"seed-{T7_SEEDS[0]}/run-manifest.json").read_text())
    config = dict(manifest["config"], response_mode="auxiliary", seed=seed, updates=UPDATES,
                  checkpoint_updates=SNAPSHOT_UPDATES)
    if config["num_envs"] * config["steps_per_update"] != DECISIONS_PER_UPDATE:
        raise ValueError("decision budget does not match the T7 collection shape")
    return config


def freeze_evaluation(directory: Path) -> dict:
    """Development freeze (the T7 deals) plus a smaller diagnostic subset."""
    t7 = T7 / f"seed-{T7_SEEDS[0]}/evaluation"
    old = json.loads((t7 / "freeze.json").read_text())
    baselines = []
    for spec in old["baselines"]:
        path = rebase(spec["path"])
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"evaluation baseline digest mismatch: {spec['name']}")
        baselines.append(dict(spec, path=str(path)))
    directory.mkdir(parents=True)
    shutil.copy2(t7 / "development.json", directory / "development.json")
    if sha256(directory / "development.json") != old["development"]["sha256"]:
        raise ValueError("T7 development deals changed")
    dev = json.loads((directory / "development.json").read_text())
    write_json(directory / "diagnostic.json", dict(dev, deals=dev["deals"][:64],
                                                    match_seeds=dev["match_seeds"][:16]))
    common = dict(engine_digest=engine_digest(), final_test=old["final_test"])
    write_json(directory / "freeze.json", dict(
        common, baselines=baselines,
        development=dict(file="development.json", sha256=sha256(directory / "development.json")),
        selection=f"final.pt at exactly {UPDATES} updates; never select by scores"))
    write_json(directory / "diagnostic-freeze.json", dict(
        common, baselines=[b for b in baselines if b["name"] == "b11-main"],
        development=dict(file="diagnostic.json", sha256=sha256(directory / "diagnostic.json")),
        selection="predeclared intermediate snapshots; diagnostic only"))
    return {name: sha256(directory / name) for name in ("freeze.json", "diagnostic-freeze.json")}


def run_script(budget_usd: float, max_hours: float, command: str, hourly_usd: float = 0.90) -> str:
    return ("#!/usr/bin/env bash\nset -euo pipefail\ncd /workspace/GuanZero\n"
            ': "${POD_DEADLINE_EPOCH:?owned provider deadline required}"\n'
            'export PYTHONPATH="$PWD/python:$PWD/oracle:$PWD"\n'
            "export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NVIDIA_TF32_OVERRIDE=0\n"
            f"exec python -m infra.watchdog --budget-usd {budget_usd:.2f} --hourly-rate-usd {hourly_usd:.2f} "
            f"--max-hours {max_hours:.2f} --grace-seconds 30 "
            f"--log-file /workspace/results/watchdog.jsonl -- {command}\n")


def prepare(root: Path) -> dict:
    """Freeze the screen kits, the continuation kit and the preflight; approval pending."""
    from train.history_ppo import HistoryPPOConfig

    if root.exists():
        raise ValueError("refusing to overwrite an existing experiment")
    if source_identity()["source_sha256"] == T7_SOURCE_SHA256:
        raise ValueError("the screen needs the exploration-floor source, not the T7 source")
    root.mkdir(parents=True)
    freeze = freeze_evaluation(root / "evaluation")
    source = pack_source(root / "source.tar.gz")
    kits = []

    def kit_dir(name: str) -> tuple[Path, Path]:
        kit = root / name
        (kit / "payload").mkdir(parents=True)
        shutil.copy2(root / "source.tar.gz", kit / "source.tar.gz")  # the monitor uploads this
        shutil.copy2(ROOT / "infra/history_setup.sh", kit / "payload/setup.sh")
        return kit, kit / "payload"

    common = dict(source=source, tests=TESTS)
    # Screen: one pod per seed, the three arms concurrently (paired on hardware).
    for seed in SEEDS:
        kit, payload = kit_dir(f"screen-{seed}")
        config = screen_config(seed)
        jobs = {}
        for name, override in ARMS.items():
            values = dict(config, **override)
            HistoryPPOConfig(**values)  # validate now, not on the pod
            jobs[name] = dict(kind="fresh", config=values, target_updates=UPDATES)
        (payload / "run.sh").write_text(run_script(*CAPS["screen"], (
            "python -m infra.history_budget_experiment workload "
            "--manifest /workspace/payload/run-manifest.json --output /workspace/results")))
        kits.append(write_manifest(kit, dict(id=f"history-budget-screen-{seed}", kind="screen",
                                             seed=seed, jobs=jobs, lifecycle=lifecycle("screen"),
                                             **common)))
    # Continuation: one pod, the three T7 lineages concurrently.
    kit, payload = kit_dir("continuation")
    jobs = {}
    for seed in T7_SEEDS:
        verified = json.loads((T7 / f"seed-{seed}/verified-artifacts.json").read_text())
        latest = T7 / f"seed-{seed}/local/results/arm-B/train/latest.pt"
        if sha256(latest) != verified["arm-B/train/latest.pt"]:
            raise ValueError(f"T7 arm-B latest.pt changed: {seed}")
        shutil.copy2(latest, payload / f"t7-{seed}-latest.pt")
        jobs[f"b{seed}"] = dict(kind="resume", resume=f"t7-{seed}-latest.pt",
                                extra_updates=CONTINUATION_UPDATES, allow_source_change=True,
                                t7_source_sha256=T7_SOURCE_SHA256,
                                t7_final_sha256=verified["arm-B/final.pt"])
    (payload / "run.sh").write_text(run_script(*CAPS["continuation"], (
        "python -m infra.history_budget_experiment workload "
        "--manifest /workspace/payload/run-manifest.json --output /workspace/results")))
    kits.append(write_manifest(kit, dict(id="history-budget-continuation", kind="continuation",
                                         jobs=jobs, lifecycle=lifecycle("continuation"), **common)))
    # Preflight: one short pod before any paid run is approved.
    kit, payload = kit_dir("preflight")
    shutil.copy2(root / f"screen-{SEEDS[0]}/run-manifest.json", payload / "screen-manifest.json")
    shutil.copy2(root / "continuation/run-manifest.json", payload / "continuation-manifest.json")
    for seed in T7_SEEDS:
        shutil.copy2(root / f"continuation/payload/t7-{seed}-latest.pt", payload / f"t7-{seed}-latest.pt")
    (payload / "run.sh").write_text(run_script(*CAPS["preflight"], (
        "python -m infra.history_budget_experiment preflight --updates 60 "
        "--manifest /workspace/payload/screen-manifest.json "
        "--continuation /workspace/payload/continuation-manifest.json "
        "--output /workspace/results")))
    kits.append(write_manifest(kit, dict(id="history-budget-preflight", kind="preflight",
                                         lifecycle=lifecycle("preflight"), **common)))
    campaign = dict(
        status="prepared; preflight and paid execution require a confirmed new budget",
        prepared_at=datetime.now(timezone.utc).isoformat(),
        source=dict(revision=source["revision"], dirty=source["dirty"],
                    source_sha256=source["source_sha256"],
                    archive_sha256=source["archive_sha256"]),
        seeds=list(SEEDS), arms=ARMS, updates=UPDATES,
        decisions_per_update=DECISIONS_PER_UPDATE,
        decision_budget=UPDATES * DECISIONS_PER_UPDATE,
        primary_comparisons=[f"{a} minus {b}" for a, b in PRIMARY],
        primary_metric="net levels per round, 256 duplicate deals (512 rounds) vs b11-main",
        secondary=["full-match win rate over 64 pairs vs b11-main",
                   "both metrics vs longrun-segment2-raw-endpoint"],
        endpoint=f"final.pt at exactly {UPDATES} updates; never select by scores",
        diagnostics=dict(snapshots=list(DIAGNOSTIC_UPDATES), set="first 64 deals, 16 match pairs, b11-main",
                         alignment=f"control update {ALIGNMENT_UPDATE} on the full set vs the T7 arm-B endpoints"),
        continuation=dict(lineages=list(T7_SEEDS), extra_updates=CONTINUATION_UPDATES,
                          comparison="continued endpoint minus the T7 endpoint, same deals, per lineage and pooled"),
        decision_rule=(f"report 95% and Bonferroni 97.5% paired seed-and-deal intervals for both "
                       f"primary comparisons; call an arm better only if the 97.5% interval excludes "
                       f"zero; a point estimate under {MIN_EFFECT} levels/round is inconclusive at "
                       f"three seeds; no automatic promotion"),
        final_test="existing sealed final test remains unopened",
        kits=[str(k) for k in kits], evaluation_freeze=freeze)
    write_json(root / "campaign.json", campaign)
    return campaign


def prepare_resume(root: Path, kit_name: str, seconds_per_update: float = 5.2,
                   hourly_usd: float = 0.90, unverified: bool = False) -> Path:
    """A follow-up kit resuming every incomplete job of a stopped kit from its latest.pt.

    Targets stay absolute (the original update budgets). The downloaded
    latest.pt files must match the verified download hashes. The kit runs the
    current source, so resume allows a recorded source change.
    """
    segments = sorted(root.glob(f"{kit_name}-r*"), key=lambda p: int(p.name.rsplit("-r", 1)[1]))
    previous = segments[-1] if segments else root / kit_name
    plan = json.loads((previous / "run-manifest.json").read_text())
    # A pod torn down mid-run (e.g. the monitor lost the network) has no
    # verified download; its synced latest.pt files were written atomically on
    # the pod and are then identified by their local hash instead.
    verified = (None if unverified else
                json.loads((previous / "verified-artifacts.json").read_text()))
    kit = root / f"{kit_name}-r{len(segments) + 1}"
    payload = kit / "payload"
    payload.mkdir(parents=True)
    source = pack_source(kit / "source.tar.gz")
    shutil.copy2(ROOT / "infra/history_setup.sh", payload / "setup.sh")
    jobs, remaining = {}, 0
    for name, job in plan["jobs"].items():
        results = previous / "local/results" / name
        receipt = results / "receipt.json"
        if receipt.exists() and json.loads(receipt.read_text())["complete"]:
            continue
        device = json.loads((results / "device.json").read_text())
        latest = results / "train/latest.pt"
        if verified is not None and sha256(latest) != verified[f"{name}/train/latest.pt"]:
            raise ValueError(f"downloaded latest.pt does not match its verified hash: {name}")
        shutil.copy2(latest, payload / f"{name}-latest.pt")
        import torch
        start = int(torch.load(latest, map_location="cpu", weights_only=False)["progress"]["updates"])
        jobs[name] = dict(kind="resume", resume=f"{name}-latest.pt",
                          target_updates=int(device["target_updates"]), allow_source_change=True,
                          resumed_from=dict(kit=previous.name, updates=start,
                                            latest_sha256=sha256(latest),
                                            download_verified=verified is not None))
        remaining = max(remaining, int(device["target_updates"]) - start)
    if not jobs:
        raise ValueError("every job of this kit is complete")
    hours = remaining * seconds_per_update / 3600 + 0.35
    budget = round(hours * hourly_usd * 1.1 + 0.05, 2)
    (payload / "run.sh").write_text(run_script(budget, hours * 1.25, (
        "python -m infra.history_budget_experiment workload "
        "--manifest /workspace/payload/run-manifest.json --output /workspace/results"),
        hourly_usd=hourly_usd))
    write_manifest(kit, dict(id=f"history-budget-{kit.name}", kind=plan["kind"] + "-resume",
                             jobs=jobs, source=source, tests=plan["tests"],
                             lifecycle=dict(proposed_budget_usd=budget,
                                            max_hours_from_create=round(hours * 1.25, 2),
                                            max_hourly_usd=hourly_usd)))
    return kit


def job_results(root: Path, kit_name: str, job: str) -> list[Path]:
    """Result directories of one job, first segment first (resume kits follow)."""
    kits = [root / kit_name] + sorted(root.glob(f"{kit_name}-r*"),
                                      key=lambda p: int(p.name.rsplit("-r", 1)[1]))
    return [k / "local/results" / job for k in kits if (k / "local/results" / job).exists()]


def job_file(root: Path, kit_name: str, job: str, name: str) -> Path:
    found = [d / name for d in job_results(root, kit_name, job) if (d / name).exists()]
    if len(found) != 1:
        raise ValueError(f"expected exactly one {name} for {kit_name}/{job}, found {len(found)}")
    return found[0]


def lifecycle(kind: str) -> dict:
    budget, hours = CAPS[kind]
    return dict(proposed_budget_usd=budget, max_hours_from_create=hours, max_hourly_usd=0.90,
                status="provisional until the preflight measures throughput")


def write_manifest(kit: Path, manifest: dict) -> Path:
    manifest.update(status="prepared; approval pending",
                    prepared_at=datetime.now(timezone.utc).isoformat(),
                    engine_digest=engine_digest(), training_device="cuda",
                    decisions_per_update=DECISIONS_PER_UPDATE,
                    compute=dict(proposed_gpu="NVIDIA RTX PRO 4500 Blackwell", cloud="SECURE",
                                 minimum_usable_cpus=16, precision="FP32; TF32 disabled",
                                 placement="CPU engine; CUDA model inference and learning"),
                    payload_files={p.name: sha256(p) for p in sorted((kit / "payload").iterdir())})
    manifest["lifecycle"]["artifact_destination"] = str(kit / "local/results")
    write_json(kit / "run-manifest.json", manifest)
    shutil.copy2(kit / "run-manifest.json", kit / "payload/run-manifest.json")
    return kit


# ---- pod side -------------------------------------------------------------------

def ratio_spike(line: dict) -> bool:
    return line["update_samples"] > 0 and (line["approx_kl"] > 0.5 or line["clip_fraction"] > 0.95)


def gate(recent: list[dict]) -> dict:
    """``infra.history_pilot.health`` with PPO ratio spikes recorded, not fatal alone.

    approx_kl is the mean of (r - 1) - log r, so one rare row whose ratio jumps
    dominates it. That happened in the epsilon arm (uniform picks with pi_old
    near 1e-5: approx_kl 0.5 to 201, ratio up to 15,000) while clip fraction,
    entropy, losses and gradient norms stayed normal; control and entropy arms
    never exceeded approx_kl 0.2. A spike therefore fails only as a burst of
    ``SPIKE_BURST`` within ``SPIKE_WINDOW`` updates; a clip fraction above
    0.95, non-finite metrics or missing gradients still fail. Every spike is
    listed in the report.
    """
    from infra.history_pilot import health

    spikes = [l for l in recent if ratio_spike(l)]
    burst = any(sum(1 for o in spikes if 0 <= o["update"] - l["update"] < SPIKE_WINDOW)
                >= SPIKE_BURST for l in spikes)
    kept = [l for l in recent if not ratio_spike(l) or l["clip_fraction"] > 0.95]
    check = health(recent if burst else kept or recent[-1:])
    if burst:
        check.update(healthy=False, reason="repeated PPO ratio spikes")
    elif not kept:
        check = dict(healthy=True, update=recent[-1]["update"])
    check["ratio_spikes"] = [dict(update=l["update"], approx_kl=l["approx_kl"],
                                  clip_fraction=l["clip_fraction"],
                                  ratio_deviation=l["ratio_deviation"])
                             for l in spikes]
    return check


def require_pod() -> None:
    if sys.platform != "linux" or "POD_DEADLINE_EPOCH" not in os.environ:
        raise RuntimeError("research training requires an owned Linux pod with a provider deadline")


def build_trainer(job: dict, output: Path, payload_dir: Path, target: int | None = None):
    import torch
    from train.history_ppo import HistoryPPOConfig, HistoryTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if job["kind"] == "fresh":
        config = HistoryPPOConfig(**dict(job["config"], updates=target or job["target_updates"]))
        return HistoryTrainer(config, output / "train", device="cuda"), target or job["target_updates"]
    source = payload_dir / job["resume"]
    start = int(torch.load(source, map_location="cpu", weights_only=False)["progress"]["updates"])
    goal = target or job.get("target_updates") or start + job["extra_updates"]
    # The saved config is restored in full; only the update target changes.
    trainer = HistoryTrainer(HistoryPPOConfig(updates=goal), output / "train", device="cuda",
                             resume=source,
                             allow_source_change=bool(job.get("allow_source_change")))
    if trainer.progress["updates"] != start or trainer.resume_count < 1:
        raise RuntimeError("resume did not restore the saved progress")
    return trainer, goal


def run_job(job: dict, output: Path, payload_dir: Path, target: int | None = None) -> dict:
    """Train one job to its update target; checkpoints stay resumable on failure."""
    import torch
    from train.history_model import save_history_checkpoint
    output.mkdir(parents=True, exist_ok=True)
    trainer, goal = build_trainer(job, output, payload_dir, target)
    write_json(output / "device.json", dict(
        device=str(trainer.device), sampler_device=str(trainer.generator.device),
        rollout_device=str(trainer.rollout_device),
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        torch=torch.__version__, cuda=torch.version.cuda, tf32=False,
        start_updates=trainer.progress["updates"], target_updates=goal))
    write_json(output / "resolved-config.json", trainer.config.__dict__)

    def snapshot(path: Path) -> None:
        payload = trainer.payload()
        payload["optimizer"] = {}
        payload.pop("population", None)
        save_history_checkpoint(path, payload)

    if trainer.progress["updates"] == 0:
        snapshot(output / "initial.pt")
    interrupted: list[str] = []
    signal.signal(signal.SIGTERM, lambda *_: interrupted.append("SIGTERM"))
    signal.signal(signal.SIGINT, lambda *_: interrupted.append("SIGINT"))
    deadline = float(os.environ["POD_DEADLINE_EPOCH"]) - 300
    recent: list[dict] = []
    start = time.monotonic()
    while trainer.progress["updates"] < goal and not interrupted:
        if time.time() > deadline:
            trainer.save()
            raise RuntimeError("provider deadline before the update budget; latest.pt is resumable")
        tick = time.monotonic()
        line = trainer.update()
        if line["step_decisions"] != DECISIONS_PER_UPDATE:
            trainer.save()
            raise RuntimeError("an update collected a different number of decisions")
        recent = (recent + [line])[-120:]
        check = gate(recent)
        check.update(epoch=time.time(), elapsed_seconds=time.monotonic() - start,
                     update_seconds=time.monotonic() - tick,
                     gpu_allocated_bytes=torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                     gpu_reserved_bytes=torch.cuda.memory_reserved() if torch.cuda.is_available() else 0)
        write_json(output / "health.json", check)
        if not check["healthy"] or check["update_seconds"] > 150:
            trainer.save()
            raise RuntimeError("health/throughput gate failed")
        update = trainer.progress["updates"]
        if update % SNAPSHOT_UPDATES == 0:
            snapshot(output / f"update-{update:06d}.pt")
        if update % LATEST_UPDATES == 0:
            trainer.save()
    trainer.save()
    complete = trainer.progress["updates"] == goal and not interrupted
    if complete:
        snapshot(output / "final.pt")
    receipt = dict(complete=complete, reason=interrupted[-1] if interrupted else "update_budget",
                   elapsed_seconds=time.monotonic() - start, progress=trainer.progress,
                   resume_count=trainer.resume_count, target_updates=goal,
                   final_sha256=sha256(output / "final.pt") if complete else None)
    write_json(output / "receipt.json", receipt)
    if not complete:
        raise RuntimeError("job interrupted before its update budget")
    return receipt


def check_source(plan: dict) -> None:
    if source_identity()["source_sha256"] != plan["source"]["source_sha256"]:
        raise ValueError("frozen source mismatch")


def supervise(commands: dict[str, list[str]], output: Path, stale_seconds: float = 170) -> None:
    """Run the jobs concurrently; stop all on the first failure or stale health."""
    children = {}
    try:
        for name, command in commands.items():
            with (output / f"{name}.log").open("w") as log:
                children[name] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                  start_new_session=True)
        while any(p.poll() is None for p in children.values()):
            if time.time() > float(os.environ["POD_DEADLINE_EPOCH"]) - 240:
                raise RuntimeError("provider cleanup deadline")
            for name, child in children.items():
                if child.poll() not in (None, 0):
                    raise RuntimeError(f"job {name} exited with {child.returncode}")
                path = output / name / "health.json"
                if path.exists() and child.poll() is None:
                    state = json.loads(path.read_text())
                    if not state["healthy"] or time.time() - path.stat().st_mtime > stale_seconds:
                        raise RuntimeError(f"job {name} health/staleness failure")
            write_json(output / "health.json", dict(healthy=True, phase="training", epoch=time.time()))
            time.sleep(10)
        for name, child in children.items():
            if child.returncode:
                raise RuntimeError(f"job {name} exited with {child.returncode}")
    finally:
        for child in children.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children.values():
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def job_command(manifest_path: Path, name: str, output: Path, target: int | None = None) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve()), "job", "--manifest", str(manifest_path),
               "--job", name, "--output", str(output)]
    return command + (["--target", str(target)] if target is not None else [])


def workload(manifest_path: Path, output: Path) -> None:
    from infra.cpu_budget import host_facts, usable_cpus

    require_pod()
    plan = json.loads(manifest_path.read_text())
    check_source(plan)
    if usable_cpus() < plan["compute"]["minimum_usable_cpus"]:
        raise RuntimeError("insufficient effective CPU quota for three concurrent jobs")
    for name, digest in plan["payload_files"].items():
        if name != "run-manifest.json" and sha256(manifest_path.parent / name) != digest:
            raise ValueError("payload digest mismatch")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "host-facts.json", host_facts())
    write_json(output / "health.json", dict(healthy=True, phase="tests"))
    with (output / "tests.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "pytest", "-q", *plan["tests"]], stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=300)
    supervise({name: job_command(manifest_path, name, output / name) for name in plan["jobs"]},
              output)
    receipts = {name: json.loads((output / name / "receipt.json").read_text())
                for name in plan["jobs"]}
    if not all(r["complete"] for r in receipts.values()):
        raise RuntimeError("a job is incomplete")
    write_json(output / "receipt.json", dict(complete=True, jobs=list(plan["jobs"]),
                                              claim="evaluation pending"))


def throughput(metrics: Path, skip: int) -> dict:
    lines = [json.loads(l) for l in metrics.read_text().splitlines()][skip:]
    collect = sum(l["collect_seconds"] for l in lines) / len(lines)
    learn = sum(l["learn_seconds"] for l in lines) / len(lines)
    return dict(updates=len(lines), collect_seconds=collect, learn_seconds=learn,
                update_seconds=collect + learn,
                all_seat_decisions_per_second=DECISIONS_PER_UPDATE / (collect + learn),
                peak_reserved_bytes=max(l["cuda_peak_reserved_bytes"] for l in lines),
                entropy=lines[-1]["entropy"],
                epsilon_pick_fraction=[l.get("rollout_epsilon_pick_fraction") for l in lines][-1],
                behaviour_weight_mean=[l.get("behaviour_weight_mean") for l in lines][-1])


def preflight(manifest_path: Path, continuation_path: Path | None, output: Path, updates: int) -> None:
    """Short concurrent screen run, fresh-arm resume, and real CUDA resume of T7 lineages.

    Nothing produced here is an experiment result.
    """
    from infra.cpu_budget import host_facts

    require_pod()
    plan = json.loads(manifest_path.read_text())
    check_source(plan)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "host-facts.json", host_facts())
    with (output / "tests.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "pytest", "-q", *plan["tests"]], stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=300)
    screen = output / "screen"
    screen.mkdir()
    started = time.time()
    supervise({n: job_command(manifest_path, n, screen / n, updates) for n in plan["jobs"]}, screen)
    report = dict(screen_wall_seconds=time.time() - started,
                  screen={n: throughput(screen / n / "train/metrics.jsonl", updates // 3)
                          for n in plan["jobs"]})
    # Fresh-arm resume: continue each screen job for 4 updates from its latest.pt.
    resumed = output / "screen-resume"
    resumed.mkdir()
    for name in plan["jobs"]:
        job = dict(kind="resume", resume=str(screen / name / "train/latest.pt"), extra_updates=4)
        receipt = run_job(job, resumed / name, Path("/"))
        report.setdefault("screen_resume", {})[name] = dict(
            start=updates, end=receipt["progress"]["updates"], resume_count=receipt["resume_count"])
    if continuation_path is None:
        write_json(output / "preflight.json", report)
        write_json(output / "receipt.json", dict(complete=True, claim="preflight only; not a result"))
        return
    # T7 lineages: real CUDA resume under this source, three concurrently.
    continuation = json.loads(continuation_path.read_text())
    lineage = output / "continuation"
    lineage.mkdir()
    started = time.time()
    supervise({name: job_command(continuation_path, name, lineage / name) + ["--extra", "20"]
               for name in continuation["jobs"]}, lineage)
    report["continuation_wall_seconds"] = time.time() - started
    report["continuation"] = {}
    for name in continuation["jobs"]:
        receipt = json.loads((lineage / name / "receipt.json").read_text())
        manifest = json.loads((lineage / name / "train/manifest.json").read_text())
        report["continuation"][name] = dict(receipt=receipt, source_changes=manifest["source_changes"],
                                            **throughput(lineage / name / "train/metrics.jsonl", 5))
    write_json(output / "preflight.json", report)
    write_json(output / "receipt.json", dict(complete=True, claim="preflight only; not a result"))


# ---- evaluation -----------------------------------------------------------------

def paired(high: list[list[float]], low: list[list[float]], rng, samples: int = 10000) -> dict:
    """Paired seed-and-deal bootstrap of mean(high - low); rows are seeds."""
    import numpy as np

    d = np.asarray(high) - np.asarray(low)
    boot = []
    for _ in range(samples):
        seeds = rng.integers(0, len(d), len(d))
        deals = rng.integers(0, d.shape[1], d.shape[1])
        boot.append(d[seeds][:, deals].mean())
    return dict(mean_delta=float(d.mean()), seed_deltas=[float(v.mean()) for v in d],
                ci_95=np.quantile(boot, [0.025, 0.975]).tolist(),
                ci_97_5=np.quantile(boot, [0.0125, 0.9875]).tolist())


def evaluate(root: Path) -> None:
    """Fixed endpoints on the full set, predeclared snapshots on the diagnostic set."""
    import numpy as np
    import torch
    from eval.history_frozen import evaluate as evaluate_frozen

    torch.set_num_threads(2)
    campaign = json.loads((root / "campaign.json").read_text())
    freeze, diagnostic = root / "evaluation/freeze.json", root / "evaluation/diagnostic-freeze.json"
    for path in (freeze, diagnostic):
        if sha256(path) != campaign["evaluation_freeze"][path.name]:
            raise ValueError("evaluation freeze changed after preparation")
    out = root / "results"
    out.mkdir(exist_ok=True)

    def frozen_evaluate(freeze_path: Path, checkpoint: Path, output: Path) -> dict:
        """Reuse a report of the same checkpoint under the same freeze."""
        if output.exists():
            report = json.loads(output.read_text())
            if (report.get("candidate_sha256") == sha256(checkpoint)
                    and report.get("freeze_sha256") == sha256(freeze_path)):
                return report
        return evaluate_frozen(freeze_path, checkpoint, output)

    def endpoint(kit_name: str, job: str, name: str) -> dict:
        checkpoint = job_file(root, kit_name, job, "final.pt")
        receipt = json.loads((checkpoint.parent / "receipt.json").read_text())
        if not receipt["complete"] or sha256(checkpoint) != receipt["final_sha256"]:
            raise ValueError(f"fixed endpoint incomplete or modified: {checkpoint}")
        return frozen_evaluate(freeze, checkpoint, out / name)["reports"]

    from eval.history_entropy import measure
    screen, entropy = {}, {}
    for seed in SEEDS:
        kit = f"screen-{seed}"
        screen[seed] = {arm: endpoint(kit, arm, f"{seed}-{arm}-final.json") for arm in ARMS}
        for arm in ARMS:
            for update in DIAGNOSTIC_UPDATES:
                snapshot = job_file(root, kit, arm, f"update-{update:06d}.pt")
                frozen_evaluate(diagnostic, snapshot, out / f"{seed}-{arm}-update-{update:06d}.json")
                entropy[f"{seed}/{arm}/{snapshot.name}"] = measure(snapshot)
            entropy[f"{seed}/{arm}/final.pt"] = measure(job_file(root, kit, arm, "final.pt"))
        frozen_evaluate(freeze, job_file(root, kit, "control", f"update-{ALIGNMENT_UPDATE:06d}.pt"),
                        out / f"{seed}-control-update-{ALIGNMENT_UPDATE:06d}.json")
    # Predeclared entropy strata, diagnostic only (on-policy self-play, fixed seed).
    write_json(out / "entropy-strata.json", entropy)
    rng = np.random.default_rng(2026092724)
    baselines = list(next(iter(screen.values()))["control"])
    summary = dict(primary=campaign["primary_comparisons"], decision_rule=campaign["decision_rule"],
                   comparisons={}, continuation={})
    for high, low in PRIMARY:
        summary["comparisons"][f"{high}-{low}"] = {b: dict(
            duplicate=paired([screen[s][high][b]["duplicates"]["pair_scores"] for s in SEEDS],
                             [screen[s][low][b]["duplicates"]["pair_scores"] for s in SEEDS], rng),
            full_match=paired([[p["win_rate"] for p in screen[s][high][b]["full_matches"]["pairs"]]
                               for s in SEEDS],
                              [[p["win_rate"] for p in screen[s][low][b]["full_matches"]["pairs"]]
                               for s in SEEDS], rng)) for b in baselines}
    if job_results(root, "continuation", f"b{T7_SEEDS[0]}"):
        continued = {seed: endpoint("continuation", f"b{seed}", f"b{seed}-continued.json")
                     for seed in T7_SEEDS}
        for b in baselines:
            before, after = [], []
            for seed in T7_SEEDS:
                old = json.loads((T7 / f"seed-{seed}/results/endpoint-B.json").read_text())
                new = continued[seed]
                before.append(old["reports"][b]["duplicates"]["pair_scores"])
                after.append(new[b]["duplicates"]["pair_scores"])
            summary["continuation"][b] = paired(after, before, rng)
        write_json(out / "continuation-entropy-strata.json",
                   {f"b{seed}": measure(job_file(root, "continuation", f"b{seed}", "final.pt"))
                    for seed in T7_SEEDS})
    write_json(root / "comparison.json", summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "resume", "workload", "job", "preflight",
                                            "evaluate"])
    parser.add_argument("--kit")
    parser.add_argument("--seconds-per-update", type=float, default=5.2)
    parser.add_argument("--hourly-usd", type=float, default=0.90)
    parser.add_argument("--unverified", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--continuation", type=Path)
    parser.add_argument("--job")
    parser.add_argument("--target", type=int)
    parser.add_argument("--extra", type=int)
    parser.add_argument("--updates", type=int, default=60)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.output.resolve())
    elif args.command == "resume":
        print(prepare_resume(args.output.resolve(), args.kit, args.seconds_per_update,
                             args.hourly_usd, args.unverified))
    elif args.command == "evaluate":
        evaluate(args.output.resolve())
    elif args.manifest is None:
        parser.error("--manifest is required")
    elif args.command == "workload":
        workload(args.manifest, args.output)
    elif args.command == "preflight":
        preflight(args.manifest, args.continuation, args.output, args.updates)
    else:
        require_pod()
        plan = json.loads(args.manifest.read_text())
        job = dict(plan["jobs"][args.job])
        if args.extra is not None:
            job["extra_updates"] = args.extra
        run_job(job, args.output, args.manifest.parent, args.target)


if __name__ == "__main__":
    main()
