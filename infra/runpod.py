"""Bounded RunPod test-pod lifecycle, using only the Python standard library.

Credentials are read from RUNPOD_API_KEY or an explicitly selected --env-file.
Output and manifests contain an allowlist of operational fields, never provider
environment variables or credentials. Creation is deliberately never retried.

The guard must remain running; it is not a provider-enforced spending limit.
Use an independent pod-side deadline as well, and retrieve artifacts before
deleting: termination destroys both container and pod-volume data.

Contracts: https://docs.runpod.io/api-reference/pods/POST/pods
https://docs.runpod.io/api-reference-v2/catalog/list-gpu-types
https://docs.runpod.io/pods/pricing
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid

REST = "https://rest.runpod.io/v1"
CATALOG = "https://api.runpod.io/v2"
GRAPHQL = "https://api.runpod.io/graphql"
IMAGE = "runpod/pytorch:1.2.0-cu1281-torch291-ubuntu2204"
OWNER = "GuanZero/infra/runpod.py/v1"
MAX_POLL_SECONDS = 30.0


class RunPodError(RuntimeError):
    """A diagnostic deliberately excluding HTTP bodies and credential values."""


class ProviderError(RunPodError):
    def __init__(self, status: int | None):
        self.status = status
        super().__init__(f"RunPod HTTP {status}" if status else "RunPod request failed")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the Authorization header to an unverified destination.
        raise ProviderError(code)


def positive(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise RunPodError("expected a finite positive number") from None
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise RunPodError("expected a finite positive number")
    return number


def load_key(env_file: Path | None = None) -> str:
    if env_file is None:
        key = os.environ.get("RUNPOD_API_KEY", "").strip()
    else:
        # Do not source a shell file or read unrelated keys into the environment.
        matches = []
        for line in env_file.read_text(encoding="utf-8").splitlines():
            match = re.fullmatch(r"\s*(?:export\s+)?RUNPOD_API_KEY\s*=\s*(.*?)\s*", line)
            if match:
                try:
                    words = shlex.split(match.group(1), comments=True)
                except ValueError:
                    raise RunPodError("invalid RUNPOD_API_KEY assignment") from None
                if len(words) != 1:
                    raise RunPodError("invalid RUNPOD_API_KEY assignment")
                matches.append(words[0])
        if len(matches) != 1:
            raise RunPodError("explicit env file must contain one RUNPOD_API_KEY assignment")
        key = matches[0]
    if not key or any(character.isspace() for character in key):
        raise RunPodError("RUNPOD_API_KEY missing or invalid")
    return key


class Client:
    def __init__(self, key: str, *, opener=None, timeout: float = 20.0):
        self._key = key
        self._opener = opener or build_opener(NoRedirect())
        self.timeout = min(positive(timeout), MAX_POLL_SECONDS)

    def request(self, method: str, url: str, body: dict | None = None):
        if not any(url == base or url.startswith(base + "/")
                   for base in (REST, CATALOG, GRAPHQL)):
            raise RunPodError("unapproved RunPod API destination")
        if "api_key=" in url or self._key in url:
            raise RunPodError("credentials must not appear in URLs")
        request = Request(url, method=method,
                          data=json.dumps(body).encode() if body is not None else None,
                          headers={"Authorization": "Bearer " + self._key,
                                   "Content-Type": "application/json",
                                   "User-Agent": "GuanZero/0.1"})
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                data = response.read()
        except HTTPError as exc:
            raise ProviderError(exc.code) from None
        except (URLError, OSError, TimeoutError):
            raise ProviderError(None) from None
        try:
            return json.loads(data) if data else None
        except (ValueError, UnicodeError):
            raise RunPodError("RunPod returned invalid JSON") from None

    def account(self) -> dict:
        response = self.request("POST", GRAPHQL, {"query": "query { myself { id "
                                "clientBalance currentSpendPerHr spendLimit isAutoPayEnabled } }"})
        if not isinstance(response, dict) or response.get("errors"):
            raise RunPodError("RunPod account query failed")
        raw = response.get("data", {}).get("myself")
        if not isinstance(raw, dict) or not raw.get("id"):
            raise RunPodError("RunPod account identity unavailable")
        return {key: raw.get(key) for key in
                ("id", "clientBalance", "currentSpendPerHr", "spendLimit", "isAutoPayEnabled")}

    def pods(self) -> list[dict]:
        response = self.request("GET", REST + "/pods")
        if not isinstance(response, list):
            raise RunPodError("RunPod pod-list schema mismatch")
        return response

    def pod(self, pod_id: str) -> dict | None:
        validate_id(pod_id)
        try:
            response = self.request("GET", REST + "/pods/" + pod_id)
        except ProviderError as exc:
            if exc.status == 404:
                return None
            raise
        if not isinstance(response, dict):
            raise RunPodError("RunPod pod schema mismatch")
        return response

    def inventory(self, cloud: str = "SECURE", min_cuda: str = "12.8") -> list[dict]:
        params = urlencode({"include": "AVAILABILITY", "product": "POD", "count": 1,
                            "cloud": cloud, "minCudaVersion": min_cuda})
        response = self.request("GET", CATALOG + "/catalog/gpus?" + params)
        if not isinstance(response, dict) or not isinstance(response.get("gpus"), list):
            raise RunPodError("RunPod GPU catalog schema mismatch")
        return [safe_gpu(gpu) for gpu in response["gpus"]]


def validate_id(pod_id: str) -> None:
    if not isinstance(pod_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{5,100}", pod_id):
        raise RunPodError("invalid pod identity")


def safe_gpu(raw: dict) -> dict:
    result = {key: raw.get(key) for key in ("id", "name", "memory", "availability")}
    result["price"] = {key: raw.get("price", {}).get(key) for key in ("secure", "community")}
    result["dataCenters"] = [{key: item.get(key) for key in ("id", "availability")}
                             for item in raw.get("dataCenters", [])]
    return result


def safe_pod(raw: dict) -> dict:
    result = {key: raw.get(key) for key in ("id", "name", "desiredStatus", "costPerHr",
                                           "gpuCount", "imageName", "vcpuCount", "memoryInGb")}
    machine = raw.get("machine") or {}
    result["machine"] = {key: machine.get(key) for key in ("gpuTypeId", "dataCenterId")}
    address = raw.get("publicIp")
    port = (raw.get("portMappings") or {}).get("22")
    try:
        if address and ipaddress.ip_address(address).is_global and 0 < int(port) < 65536:
            result["ssh"] = {"host": address, "port": int(port), "user": "root"}
    except (ValueError, TypeError):
        pass
    return result


def write_manifest(path: Path, manifest: dict, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            os.chmod(temporary, 0o600)
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("owner") != OWNER or data.get("version") != 1:
        raise RunPodError("manifest is not owned by this helper")
    validate_id(data.get("pod_id"))
    if not data.get("account_id") or not re.fullmatch(r"guanzero-test-[a-f0-9]{32}", data.get("name", "")):
        raise RunPodError("manifest identity is incomplete")
    if data["pod_id"] in data.get("preexisting_pod_ids", []):
        raise RunPodError("refusing to operate on a preexisting pod")
    created = positive(data.get("created_at_epoch"))
    deadline = positive(data.get("deadline_epoch"))
    if deadline <= created or deadline - created > positive(data.get("max_hours")) * 3600 + 1:
        raise RunPodError("manifest deadline exceeds runtime cap")
    return data


def verify_owned(manifest: dict, pod: dict) -> None:
    if (pod.get("id") != manifest["pod_id"] or pod.get("name") != manifest["name"]
            or pod.get("consumerUserId") != manifest["account_id"]):
        raise RunPodError("provider pod identity does not match the owned manifest")


def status(client: Client, path: Path) -> dict:
    manifest = read_manifest(path)
    pod = client.pod(manifest["pod_id"])
    if pod is None:
        return {"id": manifest["pod_id"], "confirmed_gone": True}
    verify_owned(manifest, pod)
    return safe_pod(pod)


def teardown(client: Client, path: Path, *, timeout: float = 120,
             poll_seconds: float = 5, clock: Callable = time.time,
             sleep: Callable = time.sleep) -> dict:
    """Retry idempotent deletion only after matching all live ownership fields."""
    manifest = read_manifest(path)
    deadline = clock() + positive(timeout)
    poll_seconds = min(positive(poll_seconds), MAX_POLL_SECONDS)
    while clock() < deadline:
        try:
            pod = client.pod(manifest["pod_id"])
            if pod is None or pod.get("desiredStatus") == "TERMINATED":
                if pod is not None:
                    verify_owned(manifest, pod)
                manifest.update(state="terminated", confirmed_gone_at_epoch=clock())
                write_manifest(path, manifest)
                return {"id": manifest["pod_id"], "confirmed_gone": True}
            verify_owned(manifest, pod)
            try:
                client.request("DELETE", REST + "/pods/" + manifest["pod_id"])
            except ProviderError as exc:
                if exc.status not in (404, 429, 500, 502, 503, 504, None):
                    raise
        except ProviderError as exc:
            if exc.status not in (429, 500, 502, 503, 504, None):
                raise
        sleep(max(0, min(poll_seconds, deadline - clock())))
    raise RunPodError("pod termination not confirmed; rerun delete with the same manifest")


def create(client: Client, path: Path, *, gpu_id: str, public_key: str,
           max_hourly_usd: float, budget_usd: float, max_hours: float,
           cloud: str = "SECURE", image: str = IMAGE, disk_gb: int = 30,
           volume_gb: int = 10, min_cpus: int = 4, min_ram_gb: int = 16,
           cuda_versions: tuple[str, ...] = ("12.8", "12.9", "13.0"),
           clock: Callable = time.time) -> dict:
    cap, budget, hours = map(positive, (max_hourly_usd, budget_usd, max_hours))
    if path.exists():
        raise RunPodError("manifest already exists; inspect it before another create")
    if cloud not in ("SECURE", "COMMUNITY"):
        raise RunPodError("unsupported cloud")
    if (disk_gb < 1 or volume_gb < 0 or min_cpus < 1 or min_ram_gb < 1
            or not cuda_versions or not all(re.fullmatch(r"\d+\.\d+", v) for v in cuda_versions)):
        raise RunPodError("invalid disk, compute, or CUDA requirements")
    if not re.fullmatch(r"(?:ssh-ed25519|ssh-rsa|ecdsa-sha2-\S+) [A-Za-z0-9+/=]+(?: [^\r\n]*)?", public_key.strip()):
        raise RunPodError("expected an SSH public key, not a private key or fingerprint")
    account = client.account()
    balance = positive(account.get("clientBalance"))
    if balance < cap or budget > balance:
        raise RunPodError("available balance is below the requested test allowance")
    baseline = client.pods()
    before = [pod["id"] for pod in baseline]
    quotes = client.inventory(cloud, min(cuda_versions, key=lambda v: tuple(map(int, v.split('.')))))
    quote_row = next((row for row in quotes if row["id"] == gpu_id), None)
    if quote_row is None or quote_row["availability"] not in ("LOW", "MEDIUM", "HIGH"):
        raise RunPodError("selected GPU has no verified current availability")
    gpu_rate = positive(quote_row["price"][cloud.lower()])
    # Conservative 28-day month; both disk kinds cost $0.10/GB/month while running.
    storage_rate = (disk_gb + volume_gb) * 0.10 / (28 * 24)
    if gpu_rate + storage_rate > cap:
        raise RunPodError("live GPU quote plus storage exceeds the hourly cap")
    runtime = min(hours * 3600, budget / cap * 3600)
    if runtime <= 120:
        raise RunPodError("allowance must exceed two minutes for startup and cleanup")
    started = clock()
    manifest = {"version": 1, "owner": OWNER, "account_id": account["id"],
                "name": "guanzero-test-" + uuid.uuid4().hex, "state": "creating",
                "created_at_epoch": started,
                "created_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
                "deadline_epoch": started + runtime, "max_hours": hours,
                "budget_usd": budget, "max_hourly_usd": cap,
                "quoted_gpu_hourly_usd": gpu_rate, "storage_hourly_usd": storage_rate,
                "gpu_id": gpu_id, "image": image, "cloud": cloud,
                "preexisting_pod_ids": before,
                "billing_note": "estimate; guard requires a running process and provider access"}
    write_manifest(path, manifest, exclusive=True)
    payload = {"name": manifest["name"], "imageName": image,
               "cloudType": cloud, "computeType": "GPU", "gpuTypeIds": [gpu_id],
               "gpuCount": 1, "interruptible": False, "locked": False,
               "containerDiskInGb": disk_gb, "volumeInGb": volume_gb,
               "volumeMountPath": "/workspace", "minVCPUPerGPU": min_cpus,
               "minRAMPerGPU": min_ram_gb, "supportPublicIp": True,
               "ports": ["22/tcp"], "allowedCudaVersions": list(cuda_versions),
               "env": {"PUBLIC_KEY": public_key.strip(), "SSH_PUBLIC_KEY": public_key.strip()}}
    try:
        pod = client.request("POST", REST + "/pods", payload)
    except RunPodError as exc:
        manifest["state"] = "creation_uncertain"
        write_manifest(path, manifest)
        # An ambiguous POST must be reconciled by its unique name, never retried.
        status_note = f" (HTTP {exc.status})" if isinstance(exc, ProviderError) and exc.status else ""
        raise RunPodError("creation not confirmed" + status_note
                          + "; inspect pods for manifest name before retrying") from None
    if not isinstance(pod, dict):
        raise RunPodError("create response invalid; reconcile unique manifest name")
    validate_id(pod.get("id"))
    if pod["id"] in before:
        raise RunPodError("create returned a preexisting pod; no delete was attempted")
    manifest.update(pod_id=pod["id"], state="created")
    write_manifest(path, manifest)
    # Persist the new identity before validation/cleanup so failures are recoverable.
    verify_owned(manifest, pod)
    try:
        actual = positive(pod.get("costPerHr"))
        if actual + storage_rate > cap:
            raise RunPodError("actual allocated rate exceeds hourly cap")
    except RunPodError:
        teardown(client, path)
        raise RunPodError("allocated rate failed cap check; pod termination confirmed") from None
    manifest.update(actual_gpu_hourly_usd=actual,
                    estimated_total_hourly_usd=actual + storage_rate)
    write_manifest(path, manifest)
    return {"manifest": str(path.resolve()), "deadline_epoch": manifest["deadline_epoch"],
            "pod": safe_pod(pod), "estimated_total_hourly_usd": actual + storage_rate}


def guard(client: Client, path: Path, *, poll_seconds: float = 15,
          clock: Callable = time.time, sleep: Callable = time.sleep) -> dict:
    manifest = read_manifest(path)
    if status(client, path).get("confirmed_gone"):
        return {"id": manifest["pod_id"], "confirmed_gone": True}
    poll_seconds = min(positive(poll_seconds), MAX_POLL_SECONDS)
    # Provider requests and confirmation also consume the same paid allowance.
    stop_at = manifest["deadline_epoch"] - 120
    while clock() < stop_at:
        current = read_manifest(path)
        if current.get("state") == "terminated":
            if status(client, path).get("confirmed_gone"):
                return {"id": manifest["pod_id"], "confirmed_gone": True}
        sleep(min(poll_seconds, stop_at - clock()))
    return teardown(client, path, timeout=max(1, manifest["deadline_epoch"] - clock()),
                    clock=clock, sleep=sleep)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path,
                        help="read only RUNPOD_API_KEY from this explicitly selected file")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--cloud", choices=["SECURE", "COMMUNITY"], default="SECURE")
    create_cli = commands.add_parser("create")
    create_cli.add_argument("--manifest", required=True, type=Path)
    create_cli.add_argument("--gpu-id", required=True)
    create_cli.add_argument("--ssh-public-key", required=True, type=Path)
    create_cli.add_argument("--max-hourly-usd", required=True, type=float)
    create_cli.add_argument("--budget-usd", required=True, type=float)
    create_cli.add_argument("--max-hours", required=True, type=float)
    create_cli.add_argument("--cloud", choices=["SECURE", "COMMUNITY"], default="SECURE")
    create_cli.add_argument("--image", default=IMAGE)
    create_cli.add_argument("--disk-gb", type=int, default=30)
    create_cli.add_argument("--volume-gb", type=int, default=10)
    create_cli.add_argument("--min-cpus", type=int, default=4)
    create_cli.add_argument("--min-ram-gb", type=int, default=16)
    for name in ("status", "delete", "guard"):
        action = commands.add_parser(name)
        action.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        client = Client(load_key(args.env_file))
        if args.command == "inspect":
            output = {"account": client.account(), "pods": [safe_pod(p) for p in client.pods()],
                      "gpus": client.inventory(args.cloud)}
        elif args.command == "create":
            output = create(client, args.manifest, gpu_id=args.gpu_id,
                            public_key=args.ssh_public_key.read_text(),
                            max_hourly_usd=args.max_hourly_usd, budget_usd=args.budget_usd,
                            max_hours=args.max_hours, cloud=args.cloud, image=args.image,
                            disk_gb=args.disk_gb, volume_gb=args.volume_gb,
                            min_cpus=args.min_cpus, min_ram_gb=args.min_ram_gb)
        elif args.command == "status":
            output = status(client, args.manifest)
        elif args.command == "delete":
            output = teardown(client, args.manifest)
        else:
            print(json.dumps({"event": "guard_started", "manifest": str(args.manifest)}), flush=True)
            output = guard(client, args.manifest)
        print(json.dumps(output, sort_keys=True), flush=True)
        return 0
    except (RunPodError, OSError, ValueError) as exc:
        # OS and JSON exceptions can contain file content; only our curated errors are shown.
        message = str(exc) if isinstance(exc, RunPodError) else "local configuration or manifest error"
        print(json.dumps({"error": message}), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
