"""Bounded Vast.ai lifecycle: CPU strength evaluation of overnight main-lineage checkpoints.
Never prints secrets.

Adapted from .work/longrun-batched-2026-09-28/lifecycle.py (25 min SSH wait with status
logging, destroy-when-the-run-never-started, ordered machine list with datacenter /
verified / reliability filters, balance check, supervisord start, read-only pod guard
verify, rsync, verified download before destroy, absence confirmation, refreeze,
recover, destroy, caffeinate). The workload is eval_pod.py: run_point.py (the unchanged
reference script) on each checkpoint, 4 torch threads each, in parallel as the pod's
actual CPU quota allows. Evaluator source: git b8c0c54, source identity 14e72581...

    python lifecycle.py freeze            # build payload/ + SHA256s (no network)
    python lifecycle.py refreeze          # re-hash after a kit-only edit (payload kept)
    python lifecycle.py dry-run [--offline]
    python lifecycle.py launch --yes-spend
    python lifecycle.py status | sync | destroy
    python lifecycle.py recover --yes-spend   # only after a pod-side stop (see README)
    python lifecycle.py summarize         # re-render results.md from download/

Timeline from create (seconds): SSH by 1500 (25 min) or destroy; the run must be started
on the pod by 4800 (1 h 20 m) or destroy; evaluations are killed at 7500 (2 h 05 m);
local sync loop ends at 7680; the local guard destroys at 8400 (2 h 20 m); the pod stops
itself at 8520 (2 h 22 m) or 20 min after DONE; hard cap 9000 (2 h 30 m).
Worst case at the $0.80/h ceiling: 0.80 x 8520 s = $1.893 + traffic <= 1.3 GB x
$0.035/GB = $0.046 -> $1.94 < $2.00.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PAYLOAD = ROOT / "payload"
SECRETS = Path("/Users/xiyaowang/Developer/Projects/GuanZero-throughput/.work/"
               "vast-throughput-2026-09-28/secrets")
API_KEY = SECRETS / "vast_api_key"
SSH_KEY = SECRETS / "id_ed25519"
MANIFEST = ROOT / "instance.json"
LABEL = "guanzero-eval-cloud-20260929"
# Ordered preference, chosen by CPU (the GPU is never used). Found 2026-09-29 with the
# filters below; all Vast Secure Cloud (hosting_type 1), verified, amd64:
#   62158  RTX 5070 Ti, EPYC 7V13, 64 effective CPUs, 129 GB, rel 0.9991, $0.552/h (Texas)
#   38639  A100 SXM4,   EPYC 7713, 32 effective CPUs, 129 GB, rel 0.9992, $0.676/h (Slovenia)
#   33405  RTX 4090,    EPYC 7B13, 32 effective CPUs, 113 GB, rel 0.9993, $0.743/h (Maryland)
#   20082  RTX 4090,    EPYC 7B13, 32 effective CPUs, 129 GB, rel 0.9989, $0.783/h (Maryland)
MACHINE_IDS = (62158, 38639, 33405, 20082)
MIN_RELIABILITY = 0.99
MIN_CPU_CORES_EFFECTIVE = 28
MIN_CPU_RAM_MB = 32000
MIN_CUDA = 12.8                   # the image is CUDA 12.8; an older driver may not start it
MAX_HOURLY_USD = 0.80
MAX_INET_USD_PER_GB = 0.035       # both directions
TRAFFIC_GB_BOUND = 1.3            # ~0.93 GB upload + results (a few MB) + margin
BALANCE_MARGIN_USD = 0.50
IMAGE = "vastai/pytorch:cuda-12.8.1-auto"
DISK_GB = 50
RUNNING_WAIT_SECONDS = 1500       # SSH reachability wait from create
WAIT_LOG_SECONDS = 60
STOPPED_POLLS_ABORT = 3
UPLOAD_TIMEOUT_SECONDS = 1800
LATEST_START_SECONDS = 4800       # later than this, evaluations cannot finish: destroy
EVAL_STOP_SECONDS = 7500
SYNC_STOP_SECONDS = 7680
GUARD_DESTROY_SECONDS = 8400
REMOTE_STOP_SECONDS = 8520
HARD_CAP_SECONDS = 9000
COST_CAP_USD = 2.00
DONE_GRACE_SECONDS = 1200
SYNC_INTERVAL_SECONDS = 60
REMOTE_STARTED = ROOT / "remote-started.json"
# Priority order: endpoint, the two references with local results, then newest first.
# With fewer than 7 x 4 usable CPUs the tail waits for a free slot.
POINTS = ("u2623", "u0844", "u2201", "u2542", "u2480", "u2418", "u2263")
REFERENCES = ("u0844", "u2201")
SOURCE_REVISION = "b8c0c543ff20d9933029d4f5d3384d5a4d3edde7"
EXPECT_SOURCE_SHA256 = "14e72581d628409032acf88b646f7eb7a1ded7e954fa06d2e367bcce67889270"
FREEZE = REPO / ".work/history-budget-2026-09-27/evaluation/freeze.json"
FREEZE_SHA256 = "f77ea34dbd4bae50dfb088b441a1c46868d7c0b1b013205975655517de3f7760"
LOCAL_EVAL = REPO / ".work/longrun-batched-eval-2026-09-29"
REFERENCE_RUN_POINT = REPO / ".work/actor-ranks-2026-09-28/eval-256/run_point.py"
RUN_DIR = REPO / ".work/longrun-batched-2026-09-28/download/results/segments/main-overnight"
CHECKPOINTS = {
    "u0844": (LOCAL_EVAL / "ckpt/u0844.pt",
              "b1c5503743fe69d7d997612ffeab139cb9d5ef2ecb55e02ee2c76f259e112bd7"),
    "u2201": (RUN_DIR / "update-002201.pt",
              "8dd52066c0cfa469"),       # prefix; full digest recorded at freeze
    "u2263": (RUN_DIR / "update-002263.pt", None),
    "u2418": (RUN_DIR / "update-002418.pt", None),
    "u2480": (RUN_DIR / "update-002480.pt", None),
    "u2542": (RUN_DIR / "update-002542.pt", None),
    "u2623": (RUN_DIR / "latest.pt", "6fd3b94d9ce8f4ae"),
}
KIT_SCRIPTS = ("eval_pod.py", "remote_setup.sh", "remote_start.sh")


def worst_case_usd(hourly: float, inet_per_gb: float) -> float:
    """Billing until the pod stops itself (the later of the two stops) plus traffic."""
    return hourly * REMOTE_STOP_SECONDS / 3600 + inet_per_gb * TRAFFIC_GB_BOUND


assert worst_case_usd(MAX_HOURLY_USD, MAX_INET_USD_PER_GB) <= COST_CAP_USD
assert (RUNNING_WAIT_SECONDS < LATEST_START_SECONDS < EVAL_STOP_SECONDS < SYNC_STOP_SECONDS
        < GUARD_DESTROY_SECONDS < REMOTE_STOP_SECONDS < HARD_CAP_SECONDS)


# ---- provider API (read-only except create/destroy/stop/start) ----------------------

def api(method, path, body=None):
    key = API_KEY.read_text().strip()
    request = urllib.request.Request(
        "https://console.vast.ai" + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Vast HTTP {exc.code}: " + exc.read().decode().replace(key, "[REDACTED]")) from None


def save(path, value):
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def log(event, **values):
    record = dict(time=time.time(), event=event, **values)
    with (ROOT / "lifecycle.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def instances():
    data = api("GET", "/api/v1/instances/")
    if data.get("next_token"):
        raise RuntimeError("Unexpected pagination; refuse incomplete ownership check")
    return data["instances"]


def balance():
    try:
        user = api("GET", "/api/v0/users/current/")
        return {k: user.get(k) for k in ("credit", "balance")}
    except Exception as error:
        return {"error": str(error)[:200]}


def owned(manifest):
    matches = [r for r in instances() if int(r["id"]) == int(manifest["instance_id"])]
    if not matches:
        return None
    row = matches[0]
    if row.get("label") != LABEL or int(row["machine_id"]) != manifest["machine_id"]:
        raise RuntimeError("Instance ownership check failed")
    return row


def public(row):
    return {k: row.get(k) for k in (
        "id", "machine_id", "label", "actual_status", "cur_state", "intended_status", "dph_total",
        "disk_space", "gpu_name", "num_gpus", "cpu_cores_effective", "cpu_ram", "ssh_host",
        "ssh_port", "public_ipaddr", "ports", "start_date", "image_uuid", "image_runtype",
        "status_msg", "inet_up_cost", "inet_down_cost", "inet_up_billed", "inet_down_billed")}


def destroy(manifest, reason):
    if owned(manifest) is None:
        log("absent_confirmed", instance_id=manifest["instance_id"])
        return True
    result = api("DELETE", f"/api/v0/instances/{manifest['instance_id']}/", {})
    log("destroy_requested", instance_id=manifest["instance_id"], reason=reason, result=result)
    for _ in range(12):
        if owned(manifest) is None:
            log("absent_confirmed", instance_id=manifest["instance_id"])
            return True
        time.sleep(5)
    raise RuntimeError("Destroy requested but instance absence not yet confirmed")


def stopped(row) -> bool:
    return (row.get("intended_status") == "stopped"
            or row.get("actual_status") in ("exited", "stopped", "offline"))


def run_started(manifest) -> bool:
    try:
        return int(json.loads(REMOTE_STARTED.read_text())["instance_id"]) == int(manifest["instance_id"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def verified_before() -> bool:
    try:
        return bool(json.loads((ROOT / "artifact-verification.json").read_text()).get("verified"))
    except (OSError, ValueError):
        return False


def teardown(manifest, reason):
    """Destroy, unless the run was started on the pod, the pod already stopped itself
    (billing only storage) and the results were never downloaded and verified: then
    keep it for `recover`. Before the run started there is nothing on the pod, so a
    stopped or unreachable instance is destroyed."""
    row = owned(manifest)
    if row is not None and run_started(manifest) and stopped(row) and not verified_before():
        note = ("The pod stopped itself before its results were downloaded and verified "
                "(the Mac was probably asleep or offline). The stopped instance keeps its "
                "disk and bills storage only. Run `lifecycle.py recover --yes-spend` to "
                "restart it, download, verify and destroy it, or `lifecycle.py destroy` to "
                "give the results up. Partial results already synced are in download/results/.\n")
        (ROOT / "RECOVER.md").write_text(note)
        log("left_stopped_for_recovery", instance_id=manifest["instance_id"], reason=reason,
            actual_status=row.get("actual_status"), intended_status=row.get("intended_status"))
        return "left_stopped"
    return destroy(manifest, reason)


def guard():
    """Detached: destroy at the deadline or on a rate above the ceiling."""
    manifest = json.loads(MANIFEST.read_text())
    log("guard_started", pid=os.getpid(), deadline=manifest["guard_destroy_epoch"])
    while time.time() < manifest["guard_destroy_epoch"]:
        try:
            row = owned(manifest)
            if row is None:
                log("guard_instance_absent")
                return
            save(ROOT / "last-instance.json", public(row))
            rate = row.get("dph_total")
            if rate is not None and float(rate) > manifest["max_hourly_usd"]:
                destroy(manifest, "hourly rate exceeded")
                return
        except Exception as exc:
            log("guard_check_error", error=str(exc)[:300])
        time.sleep(max(0, min(30, manifest["guard_destroy_epoch"] - time.time())))
    for attempt in range(20):
        try:
            teardown(manifest, "guard deadline")
            return
        except Exception as exc:
            log("guard_destroy_retry", attempt=attempt, error=str(exc)[:300])
            time.sleep(10)
    raise RuntimeError("Independent guard could not confirm teardown")


def qualified(row, machine_id):
    return (int(row["machine_id"]) == machine_id and int(row["num_gpus"]) == 1
            and float(row["dph_total"]) <= MAX_HOURLY_USD
            and int(row.get("hosting_type") or 0) == 1 and row.get("verification") == "verified"
            and float(row.get("reliability2") or 0) >= MIN_RELIABILITY
            and float(row.get("cpu_cores_effective") or 0) >= MIN_CPU_CORES_EFFECTIVE
            and float(row.get("cpu_ram") or 0) >= MIN_CPU_RAM_MB
            and row.get("cpu_arch", "amd64") == "amd64"
            and float(row.get("cuda_max_good") or 0) >= MIN_CUDA
            and float(row.get("inet_up_cost") or 0) <= MAX_INET_USD_PER_GB
            and float(row.get("inet_down_cost") or 0) <= MAX_INET_USD_PER_GB)


def search():
    """Offers of the first machine in MACHINE_IDS that has a qualifying one."""
    report, raw = {}, {}
    chosen = (None, [])
    for machine_id in MACHINE_IDS:
        query = {"machine_id": {"eq": machine_id}, "num_gpus": {"eq": 1},
                 "rentable": {"eq": True}, "rented": {"eq": False}, "verified": {"eq": True},
                 "datacenter": {"eq": True}, "type": "on-demand", "limit": 20,
                 "allocated_storage": DISK_GB}
        data = api("POST", "/api/v0/bundles/", query)
        raw[str(machine_id)] = data
        rows = data.get("offers", [])
        offers = sorted((r for r in rows if qualified(r, machine_id)),
                        key=lambda r: float(r["dph_total"]))
        report[str(machine_id)] = dict(returned=len(rows), qualified=len(offers),
                                       dph_total=[round(float(r["dph_total"]), 4) for r in offers],
                                       cpu_cores_effective=[r.get("cpu_cores_effective") for r in offers])
        if offers and chosen[0] is None:
            chosen = (machine_id, offers)
    save(ROOT / "offers.json", raw)
    return chosen[0], chosen[1], report


def create_body():
    return {"client_id": "me", "image": IMAGE, "disk": DISK_GB, "label": LABEL,
            "runtype": "ssh_direct", "env": {}, "onstart": "entrypoint.sh",
            "cancel_unavail": True, "target_state": "running"}


def create(offer):
    if MANIFEST.exists():
        raise RuntimeError("An owned manifest already exists; refuse duplicate rent")
    if any(r.get("label") == LABEL for r in instances()):
        raise RuntimeError("Matching instance already exists")
    if int(offer["machine_id"]) not in MACHINE_IDS or int(offer["num_gpus"]) != 1:
        raise RuntimeError("Wrong offer")
    if float(offer["dph_total"]) > MAX_HOURLY_USD:
        raise RuntimeError("Offer rate exceeds the price ceiling")
    started = time.time()
    manifest = dict(machine_id=int(offer["machine_id"]), offer_id=int(offer["id"]), label=LABEL,
                    created_at_epoch=started, offer_dph_total=float(offer["dph_total"]),
                    offer_cpu_cores_effective=offer.get("cpu_cores_effective"),
                    latest_start_epoch=started + LATEST_START_SECONDS,
                    eval_stop_epoch=started + EVAL_STOP_SECONDS,
                    sync_stop_epoch=started + SYNC_STOP_SECONDS,
                    guard_destroy_epoch=started + GUARD_DESTROY_SECONDS,
                    remote_stop_epoch=started + REMOTE_STOP_SECONDS,
                    hard_cap_epoch=started + HARD_CAP_SECONDS,
                    max_hourly_usd=MAX_HOURLY_USD, cost_cap_usd=COST_CAP_USD,
                    disk_gb=DISK_GB, image=IMAGE)
    save(ROOT / "create-intent.json", manifest)
    result = api("PUT", f"/api/v0/asks/{int(offer['id'])}/", create_body())
    if not result.get("success") or not result.get("new_contract"):
        raise RuntimeError("Creation failed: " + str(result)[:500])
    manifest["instance_id"] = int(result["new_contract"])
    save(MANIFEST, manifest)
    with (ROOT / "guard.log").open("a") as output:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "guard"],
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    save(ROOT / "guard-receipt.json", {"pid": process.pid,
                                       "deadline_epoch": manifest["guard_destroy_epoch"]})
    log("created", **manifest, guard_pid=process.pid)
    return manifest


# ---- kit freeze and checks ---------------------------------------------------------

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pack_b8c0c54(destination: Path, check_dir: Path) -> dict:
    """source.tar.gz from `git archive b8c0c54` (the evaluator source of the eight
    finished points): the source files of infra.history_artifacts.source_files plus a
    source-identity.json receipt, so source_identity() on the pod needs no git and
    fails loudly if any source file changes. Also extracts it to check_dir."""
    with tempfile.TemporaryDirectory() as temp:
        tree = Path(temp) / "tree"
        tree.mkdir()
        archive = subprocess.run(["git", "archive", SOURCE_REVISION], cwd=REPO,
                                 capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(tree, filter="data")
        # The tree's own code defines the file set and the digest.
        code = ("import sys, json; sys.path.insert(0, sys.argv[1]);"
                "from infra.history_artifacts import source_files, sha256, engine_digest, ROOT;"
                "from pathlib import Path; import hashlib; r = Path(sys.argv[1]);"
                "assert ROOT == r.resolve(), ROOT;"
                "f = {str(p.relative_to(r)): sha256(p) for p in source_files(r)};"
                "print(json.dumps(dict(files=f, engine_digest=engine_digest(r), source_sha256="
                "hashlib.sha256(json.dumps(f, sort_keys=True).encode()).hexdigest())))")
        result = json.loads(subprocess.run([sys.executable, "-c", code, str(tree.resolve())],
                                           capture_output=True, text=True, check=True,
                                           cwd=temp).stdout)
        if result["source_sha256"] != EXPECT_SOURCE_SHA256:
            raise RuntimeError(f"b8c0c54 source identity {result['source_sha256']} != recorded "
                               f"{EXPECT_SOURCE_SHA256}")
        identity = dict(revision=SOURCE_REVISION, dirty=False,
                        source_sha256=result["source_sha256"], dirty_source_sha256=None,
                        files=result["files"])
        with tarfile.open(destination, "w:gz", format=tarfile.PAX_FORMAT) as out:
            for name in sorted(result["files"]):
                out.add(tree / name, arcname=name, recursive=False)
            data = (json.dumps(identity, indent=2) + "\n").encode()
            item = tarfile.TarInfo("source-identity.json")
            item.size, item.mode = len(data), 0o644
            out.addfile(item, io.BytesIO(data))
    if check_dir.exists():
        shutil.rmtree(check_dir)
    check_dir.mkdir(parents=True)
    with tarfile.open(destination) as stream:
        stream.extractall(check_dir, filter="data")
    return dict(revision=SOURCE_REVISION, source_sha256=identity["source_sha256"],
                engine_digest=result["engine_digest"], source_files=len(identity["files"]),
                archive_sha256=sha256(destination))


def place(source: Path, target: Path):
    """Hard link (no extra disk) or copy; never modifies the source."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def freeze():
    freeze_data = json.loads(FREEZE.read_text())
    if sha256(FREEZE) != FREEZE_SHA256:
        raise RuntimeError("freeze.json changed")
    if sha256(ROOT / "run_point.py") != sha256(REFERENCE_RUN_POINT) or \
            sha256(ROOT / "run_point.py") != sha256(LOCAL_EVAL / "run_point.py"):
        raise RuntimeError("run_point.py differs from the reference script")
    if PAYLOAD.exists():
        shutil.rmtree(PAYLOAD)
    PAYLOAD.mkdir()
    source = pack_b8c0c54(PAYLOAD / "source.tar.gz", ROOT / "local-src")
    # Local-only (never uploaded): the macOS extension of the b8c0c54 export, so the
    # local smoke and summarize.py can import the packed tree. The pod builds its own.
    for so in (LOCAL_EVAL / "src-b8c0c54/python/gd").glob("_gd_core*.so"):
        shutil.copy2(so, ROOT / "local-src/python/gd" / so.name)
    if source["engine_digest"] != freeze_data["engine_digest"]:
        raise RuntimeError("engine digest of b8c0c54 differs from freeze.json")
    for name in ("run_point.py", *KIT_SCRIPTS):
        shutil.copy2(ROOT / name, PAYLOAD / name)
    # Evaluation assets: freeze.json, its development file and every baseline it names.
    assets = {"freeze.json": dict(local=str(FREEZE), pod=str(FREEZE), sha256=FREEZE_SHA256)}
    dev = FREEZE.parent / freeze_data["development"]["file"]
    if sha256(dev) != freeze_data["development"]["sha256"]:
        raise RuntimeError("development file changed")
    assets["development.json"] = dict(local=str(dev), pod=str(dev), sha256=sha256(dev))
    for spec in freeze_data["baselines"]:
        path = Path(spec["path"])
        if sha256(path) != spec["sha256"]:
            raise RuntimeError(f"baseline {spec['name']} changed")
        assets[f"{spec['name']}.pt"] = dict(local=str(path), pod=str(path), sha256=spec["sha256"])
    for name, meta in assets.items():
        place(Path(meta["local"]), PAYLOAD / "assets" / name)
    points = []
    for name in POINTS:
        path, expect = CHECKPOINTS[name]
        digest = sha256(path)
        if expect and not digest.startswith(expect):
            raise RuntimeError(f"{name}: {path} sha256 {digest[:16]} != expected {expect[:16]}")
        place(path, PAYLOAD / "ckpt" / f"{name}.pt")
        points.append(dict(name=name, update=int(name[1:]), source_path=str(path), sha256=digest,
                           bytes=path.stat().st_size, reference=name in REFERENCES))
    for ref in REFERENCES:
        local = json.loads((LOCAL_EVAL / "out" / f"{ref}.json").read_text())
        mine = next(p for p in points if p["name"] == ref)
        if local["candidate_sha256"] != mine["sha256"]:
            raise RuntimeError(f"{ref}: checkpoint differs from the one evaluated locally")
        if local["evaluation_source_sha256"] != EXPECT_SOURCE_SHA256:
            raise RuntimeError(f"{ref}: local result has another source identity")
    import torch, numpy   # the local evaluation environment (danlm-venv)
    expected = dict(source_sha256=EXPECT_SOURCE_SHA256, source_revision=SOURCE_REVISION,
                    engine_digest=freeze_data["engine_digest"], freeze_pod_path=str(FREEZE),
                    freeze_sha256=FREEZE_SHA256, torch_version=torch.__version__.split("+")[0],
                    numpy_version=numpy.__version__, python_version=sys.version.split()[0],
                    threads_per_eval=4, points=points, assets=assets)
    save(PAYLOAD / "expected.json", expected)
    files = sorted(str(p.relative_to(PAYLOAD)) for p in PAYLOAD.rglob("*")
                   if p.is_file() and p.name != "kit-files.sha256")
    digests = {name: sha256(PAYLOAD / name) for name in files}
    (PAYLOAD / "kit-files.sha256").write_text("".join(f"{d}  {n}\n" for n, d in digests.items()))
    sizes = {name: (PAYLOAD / name).stat().st_size for name in files}
    manifest = dict(frozen_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    source=source, payload_files=digests, payload_bytes=sizes,
                    payload_total_bytes=sum(sizes.values()),
                    kit_files_sha256=sha256(PAYLOAD / "kit-files.sha256"),
                    lifecycle_sha256=sha256(Path(__file__).resolve()),
                    remote_guard_template_sha256=sha256(ROOT / "remote_guard.template.py"),
                    summarize_sha256=sha256(ROOT / "summarize.py"),
                    machine_ids=list(MACHINE_IDS), label=LABEL, image=IMAGE,
                    max_hourly_usd=MAX_HOURLY_USD, min_reliability=MIN_RELIABILITY,
                    min_cpu_cores_effective=MIN_CPU_CORES_EFFECTIVE, min_cpu_ram_mb=MIN_CPU_RAM_MB,
                    timeline_seconds=timeline())
    save(ROOT / "kit-manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("payload_files",)}, indent=2))


def timeline():
    return dict(ssh_wait=RUNNING_WAIT_SECONDS, latest_start=LATEST_START_SECONDS,
                eval_stop=EVAL_STOP_SECONDS, sync_stop=SYNC_STOP_SECONDS,
                guard_destroy=GUARD_DESTROY_SECONDS, remote_stop=REMOTE_STOP_SECONDS,
                hard_cap=HARD_CAP_SECONDS, cost_cap_usd=COST_CAP_USD)


def refreeze():
    """Re-hash kit scripts after a kit-only edit without rebuilding the payload data."""
    old = json.loads((ROOT / "kit-manifest.json").read_text())
    for name in KIT_SCRIPTS:
        shutil.copy2(ROOT / name, PAYLOAD / name)
    files = sorted(str(p.relative_to(PAYLOAD)) for p in PAYLOAD.rglob("*")
                   if p.is_file() and p.name != "kit-files.sha256")
    digests = {name: sha256(PAYLOAD / name) for name in files}
    changed = {n: d for n, d in digests.items() if old["payload_files"].get(n) != d}
    if any(not n.endswith(tuple(KIT_SCRIPTS)) for n in changed):
        raise RuntimeError(f"payload data changed ({sorted(changed)}); use freeze")
    (PAYLOAD / "kit-files.sha256").write_text("".join(f"{d}  {n}\n" for n, d in digests.items()))
    history = list(old.get("refreezes", []))
    history.append(dict(at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        changed=sorted(changed), previous_lifecycle_sha256=old.get("lifecycle_sha256"),
                        previous_machine_ids=old.get("machine_ids")))
    manifest = dict(old, payload_files=digests, kit_files_sha256=sha256(PAYLOAD / "kit-files.sha256"),
                    lifecycle_sha256=sha256(Path(__file__).resolve()),
                    remote_guard_template_sha256=sha256(ROOT / "remote_guard.template.py"),
                    summarize_sha256=sha256(ROOT / "summarize.py"),
                    machine_ids=list(MACHINE_IDS), timeline_seconds=timeline(), refreezes=history)
    save(ROOT / "kit-manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "payload_files"}, indent=2))


def verify_kit(full=True):
    manifest = json.loads((ROOT / "kit-manifest.json").read_text())
    if sha256(Path(__file__).resolve()) != manifest["lifecycle_sha256"]:
        raise RuntimeError("lifecycle.py changed after freeze (refreeze)")
    if sha256(ROOT / "remote_guard.template.py") != manifest["remote_guard_template_sha256"]:
        raise RuntimeError("remote guard template changed after freeze")
    if sha256(PAYLOAD / "kit-files.sha256") != manifest["kit_files_sha256"]:
        raise RuntimeError("kit-files.sha256 changed after freeze")
    if [int(m) for m in manifest["machine_ids"]] != list(MACHINE_IDS):
        raise RuntimeError("kit manifest machine list differs from MACHINE_IDS")
    if manifest["source"]["source_sha256"] != EXPECT_SOURCE_SHA256:
        raise RuntimeError("payload source identity is not 14e72581...")
    listed = {}
    for line in (PAYLOAD / "kit-files.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        listed[name] = digest
    present = {str(p.relative_to(PAYLOAD)) for p in PAYLOAD.rglob("*")
               if p.is_file() and p.name != "kit-files.sha256"}
    if present != set(listed):
        raise RuntimeError(f"payload files differ from the frozen list: {sorted(present ^ set(listed))}")
    if full:
        for name, digest in listed.items():
            if sha256(PAYLOAD / name) != digest:
                raise RuntimeError(f"payload/{name} changed after freeze")
    for name in KIT_SCRIPTS:
        if sha256(ROOT / name) != listed[name]:
            raise RuntimeError(f"{name} edited after freeze (refreeze)")
    return manifest


def check_secrets():
    for path in (API_KEY, SSH_KEY):
        if not path.exists():
            raise RuntimeError(f"missing secret file {path}")
        mode = path.stat().st_mode & 0o777
        if mode & 0o077:
            raise RuntimeError(f"{path.name} permissions {oct(mode)}; require 0600")
    return {"api_key": "present, 0600", "ssh_key": "present, 0600"}


def render_remote(manifest, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    text = (ROOT / "remote_guard.template.py").read_text()
    text = text.replace("__INSTANCE__", str(int(manifest["instance_id"])))
    text = text.replace("__DEADLINE__", repr(float(manifest["remote_stop_epoch"])))
    text = text.replace("__DONE_GRACE__", repr(float(DONE_GRACE_SECONDS)))
    (directory / "remote_guard.py").write_text(text)
    py_compile.compile(str(directory / "remote_guard.py"), doraise=True)
    (directory / "deadlines.env").write_text(
        f"EVAL_DEADLINE_EPOCH={manifest['eval_stop_epoch']:.3f}\n"
        f"REMOTE_STOP_EPOCH={manifest['remote_stop_epoch']:.3f}\n"
        f"KIT_MACHINE_ID={int(manifest['machine_id'])}\n"
        f"KIT_OFFER_ID={int(manifest.get('offer_id', 0))}\n"
        f"KIT_DPH_TOTAL={float(manifest.get('offer_dph_total', 0.0)):.6f}\n"
        f"CREATED_EPOCH={float(manifest.get('created_at_epoch', 0.0)):.3f}\n")
    digests = {name: sha256(directory / name) for name in ("remote_guard.py", "deadlines.env")}
    (directory / "generated.sha256").write_text("".join(f"{d}  {n}\n" for n, d in digests.items()))
    return digests


# ---- ssh helpers -------------------------------------------------------------------

def ssh_target(row):
    ports = (row.get("ports") or {}).get("22/tcp") or []
    if row.get("public_ipaddr") and ports:
        return row["public_ipaddr"].strip(), int(ports[0]["HostPort"])
    return row["ssh_host"], int(row["ssh_port"])


def ssh_base(port):
    return ["-i", str(SSH_KEY), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={ROOT / 'known_hosts'}", "-o", "ServerAliveInterval=30"]


def ssh(target, command, timeout=120):
    host, port = target
    return subprocess.run(["ssh", *ssh_base(port), "-p", str(port), f"root@{host}", command],
                          capture_output=True, text=True, timeout=timeout)


def rsh(port):
    return "ssh " + " ".join(ssh_base(port)) + f" -p {port}"


def rsync(target, final=False):
    host, port = target
    destination = ROOT / "download" / "results"
    destination.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["rsync", "-az", "--timeout=60", "--exclude=.*", "-e", rsh(port),
                             f"root@{host}:/workspace/results/", str(destination) + "/"],
                            capture_output=True, text=True, timeout=600)
    finished = sorted(p.stem for p in (destination / "out").glob("u*.time")
                      if "wall_seconds" in p.read_text()) if (destination / "out").exists() else []
    log("sync", final=final, returncode=result.returncode, stderr=result.stderr[-500:],
        points_finished=finished)
    return result.returncode == 0


def upload(target):
    host, port = target
    for source in (str(PAYLOAD) + "/", str(ROOT / "generated") + "/"):
        result = subprocess.run(["rsync", "-a", "--partial", "--timeout=120", "-e", rsh(port),
                                 source, f"root@{host}:/workspace/kit/"],
                                capture_output=True, text=True, timeout=UPLOAD_TIMEOUT_SECONDS)
        if result.returncode != 0:
            raise RuntimeError("upload failed: " + result.stderr[-300:])


def verify_download():
    results = ROOT / "download" / "results"
    manifest_path = results / "results-manifest.json"
    if not manifest_path.exists():
        report = dict(verified=False, reason="no results-manifest.json (run not DONE)")
    else:
        manifest = json.loads(manifest_path.read_text())
        bad = [name for name, meta in manifest.items()
               if not (results / name).exists() or sha256(results / name) != meta["sha256"]
               or (results / name).stat().st_size != meta["bytes"]]
        report = dict(verified=not bad, files=len(manifest), mismatched=bad[:50],
                      manifest_sha256=sha256(manifest_path))
    save(ROOT / "artifact-verification.json", report)
    log("verify_download", **{k: v for k, v in report.items() if k != "mismatched"},
        mismatched=len(report.get("mismatched", [])))
    return report


def summarize_local():
    """Render results.md/json from whatever is in download/results (partial is fine)."""
    result = subprocess.run([sys.executable, str(ROOT / "summarize.py"),
                             "--pod-results", str(ROOT / "download" / "results"),
                             "--out", str(ROOT)], capture_output=True, text=True, timeout=600)
    log("summarize", returncode=result.returncode, stderr=result.stderr[-500:])
    return result.returncode == 0


# ---- commands ----------------------------------------------------------------------

def dry_run(offline: bool):
    kit = verify_kit()
    secrets = check_secrets()
    for script in ("remote_setup.sh", "remote_start.sh"):
        subprocess.run(["bash", "-n", str(ROOT / script)], check=True)
    py_compile.compile(str(ROOT / "eval_pod.py"), doraise=True)
    py_compile.compile(str(ROOT / "summarize.py"), doraise=True)
    now = time.time()
    fake = dict(instance_id=99999999, machine_id=MACHINE_IDS[0], offer_id=0, created_at_epoch=now,
                eval_stop_epoch=now + EVAL_STOP_SECONDS, remote_stop_epoch=now + REMOTE_STOP_SECONDS)
    rendered = render_remote(fake, ROOT / "dry-run-generated")
    report = dict(machine_ids=list(MACHINE_IDS), kit_source=kit["source"],
                  payload_total_mb=round(kit["payload_total_bytes"] / 1e6, 1),
                  secrets=secrets, rendered=rendered, manifest_exists=MANIFEST.exists(),
                  points=list(POINTS), timeline_seconds=timeline(),
                  worst_case_usd_at_ceiling=round(worst_case_usd(MAX_HOURLY_USD, MAX_INET_USD_PER_GB), 3),
                  create_body=create_body())
    blocked = []
    if MANIFEST.exists():
        blocked.append("instance.json exists: a previous launch is not closed (launch refuses)")
    if not offline:
        rows = instances()
        report["provider_instances"] = len(rows)
        report["matching_label"] = [dict(id=r["id"], actual_status=r.get("actual_status"),
                                         intended_status=r.get("intended_status"))
                                    for r in rows if r.get("label") == LABEL]
        report["balance"] = balance()
        machine, offers, per_machine = search()
        report["search"] = per_machine
        report["would_rent_machine"] = machine
        report["offers"] = [{k: r.get(k) for k in (
            "id", "machine_id", "host_id", "gpu_name", "dph_total", "cpu_name", "cpu_cores",
            "cpu_cores_effective", "cpu_ghz", "cpu_ram", "disk_space", "inet_up", "inet_down",
            "reliability2", "hosting_type", "verification", "cuda_max_good", "driver_version",
            "geolocation", "inet_up_cost", "inet_down_cost", "storage_cost")} for r in offers]
        report["would_create_offer"] = offers[0]["id"] if offers else None
        if offers:
            o = offers[0]
            report["worst_case_usd_at_offer"] = round(worst_case_usd(
                float(o["dph_total"]), max(float(o.get("inet_up_cost") or 0),
                                           float(o.get("inet_down_cost") or 0))), 3)
        else:
            blocked.append("no qualifying offer on the pinned machines right now")
        report["balance_check"] = balance_check(report["balance"])
        if not report["balance_check"]["ok"]:
            blocked.append("credit below worst case + margin")
        if report["matching_label"]:
            blocked.append("an instance with this kit's label exists (launch refuses)")
    report["stopped_before"] = "PUT /api/v0/asks/{offer}/ (create) -- dry run, nothing rented"
    report["launch_blocked_by"] = blocked
    save(ROOT / "dry-run.json", report)
    print(json.dumps(report, indent=2))
    if blocked:
        raise SystemExit("dry run passed its other checks, but launch would refuse: " + "; ".join(blocked))


def balance_check(value: dict) -> dict:
    needed = worst_case_usd(MAX_HOURLY_USD, MAX_INET_USD_PER_GB) + BALANCE_MARGIN_USD
    credit = value.get("credit")
    return dict(credit=credit, needed=round(needed, 3),
                ok=credit is not None and float(credit) >= needed)


def launch(yes_spend: bool):
    if not yes_spend:
        raise SystemExit("launch rents an instance; pass --yes-spend after the user approves the budget")
    kit = verify_kit()
    check_secrets()
    caffeinate = None
    if sys.platform == "darwin":
        caffeinate = subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())])
    before = balance()
    check = balance_check(before)
    if not check["ok"]:
        raise RuntimeError(f"credit {check['credit']} below worst case + margin {check['needed']}")
    machine, offers, per_machine = search()
    if not offers:
        raise RuntimeError(f"no qualifying offer on machines {MACHINE_IDS} at <= ${MAX_HOURLY_USD}/h")
    log("launch", source=kit["source"], balance_before=before, machine_id=machine,
        search=per_machine, offer=offers[0]["id"], dph_total=offers[0]["dph_total"],
        cpu_cores_effective=offers[0].get("cpu_cores_effective"))
    manifest = create(offers[0])
    outcome = "launch aborted before the run was started on the pod"
    try:
        target = None
        next_log, stopped_polls = 0.0, 0
        while time.time() < manifest["created_at_epoch"] + RUNNING_WAIT_SECONDS:
            try:
                row = owned(manifest)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                log("wait_poll_error", error=str(error)[:300])
                time.sleep(10)
                continue
            if row is None:
                raise RuntimeError("instance disappeared before running")
            save(ROOT / "last-instance.json", public(row))
            if time.time() >= next_log:
                log("waiting_for_ssh", seconds=round(time.time() - manifest["created_at_epoch"]),
                    actual_status=row.get("actual_status"), cur_state=row.get("cur_state"),
                    intended_status=row.get("intended_status"), status_msg=row.get("status_msg"))
                next_log = time.time() + WAIT_LOG_SECONDS
            stopped_polls = stopped_polls + 1 if row.get("intended_status") == "stopped" else 0
            if stopped_polls >= STOPPED_POLLS_ABORT:
                raise RuntimeError(f"instance {row.get('actual_status')}/{row.get('intended_status')}"
                                   " before becoming reachable")
            if row.get("actual_status") == "running" and (row.get("ports") or row.get("ssh_host")):
                target = ssh_target(row)
                if ssh(target, "true", timeout=40).returncode == 0:
                    break
            time.sleep(10)
        else:
            raise RuntimeError(f"instance not reachable by ssh within {RUNNING_WAIT_SECONDS} s")
        log("reachable", host_port=list(target), seconds=time.time() - manifest["created_at_epoch"])
        probe = ssh(target, "P=/venv/main/bin/python; [ -x $P ] || P=python3; "
                            "$P -c \"e=open('/proc/1/environ','rb').read().split(b'\\0');"
                            "print('present' if any(x.startswith(b'CONTAINER_API_KEY=') for x in e)"
                            " and any(x.startswith(b'VAST_CONTAINERLABEL=') for x in e) else 'absent')\"; "
                            "command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync) >/dev/null 2>&1; "
                            "command -v rsync >/dev/null && echo rsync-ok", timeout=300)
        if "present" not in probe.stdout:
            raise RuntimeError("container API key or label absent; remote self-stop impossible")
        if "rsync-ok" not in probe.stdout:
            raise RuntimeError("rsync unavailable on the pod")
        generated = render_remote(manifest, ROOT / "generated")
        log("rendered", **generated)
        if ssh(target, "mkdir -p /workspace/kit /workspace/results").returncode != 0:
            raise RuntimeError("remote mkdir failed")
        log("upload_started", mb=round(kit["payload_total_bytes"] / 1e6, 1))
        upload(target)
        log("upload_done", seconds=time.time() - manifest["created_at_epoch"])
        if time.time() > manifest["latest_start_epoch"]:
            raise RuntimeError("too late to start: evaluations could not finish before the deadline")
        save(REMOTE_STARTED, {"instance_id": manifest["instance_id"], "time": time.time()})
        outcome = "run started on the pod; launch failed before the sync loop"
        started = ssh(target, "cat > /etc/supervisor/conf.d/gzeval.conf <<'EOF'\n"
                              "[program:gzeval]\ncommand=/bin/bash /workspace/kit/remote_start.sh\n"
                              "directory=/workspace/kit\nautostart=false\nautorestart=false\n"
                              "startsecs=0\nstopwaitsecs=120\nstopasgroup=true\nkillasgroup=true\n"
                              "stdout_logfile=/workspace/results/start.log\nredirect_stderr=true\n"
                              "EOF\nsupervisorctl reread; supervisorctl update; "
                              "supervisorctl start gzeval; sleep 3; supervisorctl status gzeval")
        log("remote_start_output", stdout=started.stdout[-800:], stderr=started.stderr[-300:],
            returncode=started.returncode)
        if "RUNNING" not in started.stdout and "EXITED" not in started.stdout:
            raise RuntimeError("remote start failed: " + started.stdout[-300:])
        log("remote_started", seconds=time.time() - manifest["created_at_epoch"])
        outcome = "run started on the pod; launch failed during the sync loop"
        done, seen = False, set()
        while time.time() < manifest["sync_stop_epoch"]:
            time.sleep(min(SYNC_INTERVAL_SECONDS, max(1, manifest["sync_stop_epoch"] - time.time())))
            try:
                rsync(target)
                done = ssh(target, "test -f /workspace/results/DONE && echo yes || echo no",
                           timeout=40).stdout.strip() == "yes"
                out = ROOT / "download" / "results" / "out"
                now_done = {p.stem for p in out.glob("u*.time") if "wall_seconds" in p.read_text()} \
                    if out.exists() else set()
                if now_done - seen:
                    seen = now_done
                    summarize_local()     # partial results.md as each point lands
            except Exception as error:
                log("sync_error", error=str(error)[:300])
            if done:
                break
        log("workload_finished" if done else "sync_deadline", done=done)
        if not done:
            try:
                ssh(target, "pkill -TERM -f '[e]val_pod\\.py'; for i in $(seq 60); do "
                            "test -f /workspace/results/DONE && break; sleep 5; done; echo waited",
                    timeout=400)
            except Exception as error:
                log("stop_error", error=str(error)[:300])
        for _ in range(2):
            try:
                rsync(target, final=True)
            except Exception as error:
                log("sync_error", error=str(error)[:300])
        verify_download()
        summarize_local()
        outcome = "run finished (download verified)" if verified_before() else \
            "run finished (download NOT verified)"
    except BaseException as error:
        outcome = f"{outcome}: {type(error).__name__}: {str(error)[:200]}"
        raise
    finally:
        confirmed = False
        for attempt in range(10):
            try:
                confirmed = teardown(manifest, outcome if attempt == 0 else f"retry: {outcome}")
                break
            except Exception as error:
                log("destroy_retry", attempt=attempt, error=str(error)[:300])
                time.sleep(10)
        ended = time.time()
        rate = manifest["offer_dph_total"]
        try:
            rate = float(json.loads((ROOT / "last-instance.json").read_text()).get("dph_total") or rate)
        except Exception:
            pass
        elapsed = ended - manifest["created_at_epoch"]
        summary = dict(instance_id=manifest["instance_id"], machine_id=manifest["machine_id"],
                       offer_id=manifest["offer_id"], created_epoch=manifest["created_at_epoch"],
                       absence_confirmed=confirmed, ended_epoch=ended, elapsed_seconds=elapsed,
                       hourly_usd=rate, estimated_compute_disk_usd=elapsed / 3600 * rate,
                       within_hard_cap=elapsed <= HARD_CAP_SECONDS, balance_before=before,
                       balance_after=balance(), status_after=status_json(), outcome=outcome,
                       cost_scope="elapsed x quoted hourly; traffic charges not included")
        save(ROOT / "teardown-summary.json", summary)
        if confirmed is True:
            MANIFEST.rename(ROOT / f"instance-{manifest['instance_id']}.closed.json")
        log("teardown", **{k: v for k, v in summary.items() if k != "status_after"})
        if caffeinate is not None:
            caffeinate.terminate()


def recover(yes_spend: bool):
    """Restart a pod-stopped instance, download and verify its results, destroy it.
    Bounded: 15 min to become reachable, else stopped again."""
    if not yes_spend:
        raise SystemExit("recover restarts an instance; pass --yes-spend")
    check_secrets()
    manifest = json.loads(MANIFEST.read_text())
    row = owned(manifest)
    if row is None:
        raise SystemExit("instance already absent; nothing to recover")
    caffeinate = None
    if sys.platform == "darwin":
        caffeinate = subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())])
    began = time.time()
    try:
        if stopped(row):
            result = api("PUT", f"/api/v0/instances/{manifest['instance_id']}/", {"state": "running"})
            log("recover_start_requested", result=result)
        target = None
        while time.time() < began + 900:
            row = owned(manifest)
            if row is None:
                raise RuntimeError("instance disappeared during recovery")
            if row.get("actual_status") == "running" and (row.get("ports") or row.get("ssh_host")):
                target = ssh_target(row)
                if ssh(target, "true", timeout=40).returncode == 0:
                    break
            time.sleep(15)
        else:
            raise RuntimeError("instance not reachable within 15 min (GPU may be rented by "
                               "someone else); destroy by hand or retry later")
        for _ in range(3):
            rsync(target, final=True)
            if verify_download().get("verified"):
                break
        summarize_local()
    finally:
        if verified_before():
            destroy(manifest, "recovered")
            MANIFEST.rename(ROOT / f"instance-{manifest['instance_id']}.closed.json")
        else:
            row = owned(manifest)
            if row is not None and not stopped(row):
                api("PUT", f"/api/v0/instances/{manifest['instance_id']}/", {"state": "stopped"})
                log("recover_stopped_again", elapsed=time.time() - began)
        if caffeinate is not None:
            caffeinate.terminate()


def status_json():
    if MANIFEST.exists():
        row = owned(json.loads(MANIFEST.read_text()))
        return public(row) if row else {"owned": None, "instances": len(instances())}
    return {"instances": len(instances()), "owned": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=["freeze", "refreeze", "dry-run", "launch", "status",
                                           "sync", "destroy", "guard", "recover", "summarize"])
    parser.add_argument("--offline", action="store_true", help="dry-run without read-only API calls")
    parser.add_argument("--yes-spend", action="store_true")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "refreeze":
        refreeze()
    elif args.action == "dry-run":
        dry_run(args.offline)
    elif args.action == "launch":
        launch(args.yes_spend)
    elif args.action == "recover":
        recover(args.yes_spend)
    elif args.action == "guard":
        guard()
    elif args.action == "summarize":
        summarize_local()
        print((ROOT / "results.md").read_text() if (ROOT / "results.md").exists() else "no results.md")
    elif args.action == "destroy":
        manifest = json.loads(MANIFEST.read_text())
        if destroy(manifest, "manual") is True:
            MANIFEST.rename(ROOT / f"instance-{manifest['instance_id']}.closed.json")
            log("manifest_closed", instance_id=manifest["instance_id"])
    elif args.action == "sync":
        row = owned(json.loads(MANIFEST.read_text()))
        if row is None:
            raise SystemExit("no owned instance")
        rsync(ssh_target(row), final=True)
        verify_download()
        summarize_local()
    else:
        print(json.dumps(status_json(), indent=2))


if __name__ == "__main__":
    main()
