"""Pod-side self-stop at a fixed deadline, or DONE_GRACE seconds after the run
wrote /workspace/results/DONE (backup to the local destroy guard). It can only
STOP the instance (state=stopped: GPU billing ends, the disk and its storage
billing remain); destroying needs the account key, which never leaves the Mac.

Generated per instance by lifecycle.py (INSTANCE/DEADLINE substituted). Uses the
container-scoped CONTAINER_API_KEY from the environment or /proc/1/environ;
never prints it.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

INSTANCE = __INSTANCE__
DEADLINE = __DEADLINE__
DONE_GRACE = __DONE_GRACE__
DONE_FILE = "/workspace/results/DONE"


def container_env(name):
    value = os.environ.get(name)
    if value:
        return value
    try:
        for item in open("/proc/1/environ", "rb").read().split(b"\0"):
            if item.startswith(name.encode() + b"="):
                return item.split(b"=", 1)[1].decode()
    except OSError:
        pass
    return None


def state(value):
    label = container_env("VAST_CONTAINERLABEL") or ""
    if label != f"C.{INSTANCE}":
        raise RuntimeError("instance identity mismatch")
    key = container_env("CONTAINER_API_KEY")
    if not key:
        raise RuntimeError("CONTAINER_API_KEY unavailable")
    request = urllib.request.Request(
        f"https://console.vast.ai/api/v0/instances/{INSTANCE}/",
        data=json.dumps({"state": value}).encode(), method="PUT",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.load(response)
    if data.get("success") is not True:
        raise RuntimeError("provider did not confirm state change")
    print(json.dumps({"epoch": time.time(), "state": value, "success": True}), flush=True)


def verify():
    """Read-only: label and key check plus one GET. Never PUT state=running,
    which restarted the running container on Sept 28 2026."""
    label = container_env("VAST_CONTAINERLABEL") or ""
    if label != f"C.{INSTANCE}":
        raise RuntimeError("instance identity mismatch")
    key = container_env("CONTAINER_API_KEY")
    if not key:
        raise RuntimeError("CONTAINER_API_KEY unavailable")
    request = urllib.request.Request(
        f"https://console.vast.ai/api/v0/instances/{INSTANCE}/", method="GET",
        headers={"Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            print(json.dumps({"epoch": time.time(), "verify": "get", "http": response.status}), flush=True)
    except urllib.error.HTTPError as error:
        print(json.dumps({"epoch": time.time(), "verify": "get", "http": error.code}), flush=True)


if sys.argv[1:] == ["verify"]:
    verify()
else:
    print(json.dumps({"event": "guard_started", "deadline": DEADLINE, "pid": os.getpid()}), flush=True)
    while time.time() < DEADLINE:
        try:
            if time.time() - os.path.getmtime(DONE_FILE) > DONE_GRACE:
                print(json.dumps({"event": "done_grace_elapsed", "epoch": time.time()}), flush=True)
                break
        except OSError:
            pass
        time.sleep(max(0.0, min(30.0, DEADLINE - time.time())))
    for attempt in range(20):
        try:
            state("stopped")
            break
        except Exception as error:
            print(json.dumps({"event": "stop_retry", "attempt": attempt,
                              "type": type(error).__name__}), flush=True)
            time.sleep(10)
