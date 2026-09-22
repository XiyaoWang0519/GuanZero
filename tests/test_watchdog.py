"""Exercise real subprocess shutdown, deadline accounting and safe hook opt-in."""
from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

from infra.watchdog import nonnegative, positive
from infra import watchdog

ROOT = Path(__file__).resolve().parents[1]


def command(tmp_path: Path, child: str, *options: str) -> list[str]:
    return [sys.executable, "-m", "infra.watchdog", "--budget-usd", "1",
            "--hourly-rate-usd", "1", "--max-hours", "1",
            "--grace-seconds", "0.2", "--poll-seconds", "0.01",
            "--log-file", str(tmp_path / "watchdog.jsonl"),
            *options, "--", sys.executable, "-c", child]


def events(tmp_path: Path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "watchdog.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("invalid", ["0", "-1", "nan", "inf", "-inf"])
def test_watchdog_requires_finite_positive_limits(invalid):
    with pytest.raises(argparse.ArgumentTypeError):
        positive(invalid)


def test_nonnegative_allows_zero_but_not_nan():
    assert nonnegative("0") == 0
    with pytest.raises(argparse.ArgumentTypeError):
        nonnegative("nan")


def test_watchdog_preserves_training_exit_code(tmp_path):
    result = subprocess.run(command(tmp_path, "raise SystemExit(7)"),
                            cwd=ROOT, capture_output=True, text=True, timeout=5)
    assert result.returncode == 7, result.stderr
    assert events(tmp_path)[-1]["reason"] == "failed"


def test_budget_exhaustion_refuses_to_start(tmp_path):
    marker = tmp_path / "must-not-exist"
    child = f"from pathlib import Path; Path({str(marker)!r}).touch()"
    result = subprocess.run(command(tmp_path, child, "--spent-usd", "1"),
                            cwd=ROOT, capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert "already exhausted" in result.stderr
    assert not marker.exists()


@pytest.mark.parametrize("limit,reason", [
    (["--max-hours", str(1.0 / 3600)], "walltime"),
    (["--budget-usd", str(1.0 / 3600)], "budget"),
])
def test_deadline_requests_checkpoint_before_limit(tmp_path, limit, reason):
    saved = tmp_path / "saved"
    child = f"""
import signal, time
from pathlib import Path
def save(signum, frame):
    Path({str(saved)!r}).write_text('checkpoint')
    raise SystemExit(0)
signal.signal(signal.SIGTERM, save)
while True:
    time.sleep(0.01)
"""
    result = subprocess.run(command(tmp_path, child, *limit), cwd=ROOT,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 124, (result.stdout, result.stderr)
    assert saved.read_text() == "checkpoint"
    data = events(tmp_path)
    assert data[-1]["reason"] == reason
    assert data[-1]["estimated_spend_usd"] < 1.1 / 3600
    assert not any(row["event"] == "killing" for row in data)


def test_watchdog_kills_unresponsive_training(tmp_path):
    child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
    result = subprocess.run(command(tmp_path, child, "--max-hours", str(0.8 / 3600)),
                            cwd=ROOT, capture_output=True, text=True, timeout=5)
    assert result.returncode == 124, result.stderr
    assert any(row["event"] == "killing" for row in events(tmp_path))


def test_watchdog_forwards_external_sigterm(tmp_path):
    ready, saved = tmp_path / "ready", tmp_path / "saved"
    child = f"""
import signal, time
from pathlib import Path
def save(signum, frame):
    Path({str(saved)!r}).write_text(str(signum))
    raise SystemExit(0)
signal.signal(signal.SIGTERM, save)
Path({str(ready)!r}).touch()
while True:
    time.sleep(0.01)
"""
    process = subprocess.Popen(command(tmp_path, child), cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        process.terminate()
        stdout, stderr = process.communicate(timeout=3)
        assert process.returncode == 128 + signal.SIGTERM, (stdout, stderr)
        assert saved.read_text() == str(signal.SIGTERM)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_teardown_requires_explicit_opt_in_and_runs_after_child(tmp_path):
    marker = tmp_path / "teardown"
    hook = json.dumps([sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"])
    args = command(tmp_path, "pass", "--teardown-command-json", hook)
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert not marker.exists()
    result = subprocess.run(command(tmp_path, "pass", "--teardown-command-json", hook,
                                    "--allow-teardown"),
                            cwd=ROOT, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert marker.exists()
    assert events(tmp_path)[-2]["event"] == "teardown_complete"


def test_teardown_failure_is_visible(tmp_path):
    result = subprocess.run(command(tmp_path, "pass", "--teardown-command-json",
                                    json.dumps([sys.executable, "-c", "raise SystemExit(3)"]),
                                    "--allow-teardown"), cwd=ROOT,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 1
    assert any(row["event"] == "teardown_failed" for row in events(tmp_path))


def test_hung_teardown_is_bounded(tmp_path):
    result = subprocess.run(command(tmp_path, "pass", "--teardown-command-json",
                                    json.dumps([sys.executable, "-c", "import time; time.sleep(60)"]),
                                    "--allow-teardown", "--hook-timeout-seconds", "0.1"),
                            cwd=ROOT, capture_output=True, text=True, timeout=5)
    assert result.returncode == 1
    assert any(row["event"] == "teardown_failed" and row["error"] == "timeout"
               for row in events(tmp_path))


def test_sync_failure_stops_training_and_still_tears_down(tmp_path):
    marker = tmp_path / "teardown"
    hook = json.dumps([sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"])
    result = subprocess.run(command(tmp_path, "import time; time.sleep(60)",
                                    "--sync-source", str(tmp_path / "missing-source"),
                                    "--sync-dest", str(tmp_path / "destination"),
                                    "--sync-interval-seconds", "0.05",
                                    "--teardown-command-json", hook, "--allow-teardown"),
                            cwd=ROOT, capture_output=True, text=True, timeout=5)
    assert result.returncode == 1
    assert marker.exists()
    assert events(tmp_path)[-1]["reason"] == "sync_failed"
    assert any(row["event"] == "final_sync_failed" for row in events(tmp_path))


def test_sync_copies_atomically_written_checkpoint_and_ignores_temps(tmp_path):
    if not shutil.which("rsync"):
        pytest.skip("rsync is required for checkpoint sync")
    source, destination = tmp_path / "source with spaces", tmp_path / "durable copy"
    source.mkdir()
    (source / "latest.pt").write_bytes(b"saved checkpoint")
    (source / "partial.pt.tmp").write_bytes(b"incomplete")
    (source / ".latest.pt.uncommitted").write_bytes(b"incomplete atomic save")
    result = subprocess.run([str(ROOT / "infra/sync.sh"), str(source), str(destination)],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert (destination / "latest.pt").read_bytes() == b"saved checkpoint"
    assert not (destination / "partial.pt.tmp").exists()
    assert not (destination / ".latest.pt.uncommitted").exists()


def teardown_marker(tmp_path: Path) -> tuple[Path, str]:
    marker = tmp_path / "teardown"
    hook = json.dumps([sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"])
    return marker, hook


def test_full_log_disk_after_successful_sync_cannot_skip_teardown(tmp_path, monkeypatch):
    if not shutil.which("rsync"):
        pytest.skip("rsync is required for checkpoint sync")
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    (source / "latest.pt").write_bytes(b"durable checkpoint")
    marker, hook = teardown_marker(tmp_path)
    real_open = Path.open
    log_writes = 0

    def full_disk(path, *args, **kwargs):
        nonlocal log_writes
        if path == tmp_path / "watchdog.jsonl":
            log_writes += 1
            if log_writes >= 2:
                raise OSError(errno.ENOSPC, "No space left on device")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", full_disk)
    result = watchdog.main(command(tmp_path, "pass", "--sync-source", str(source),
                                    "--sync-dest", str(destination),
                                    "--teardown-command-json", hook, "--allow-teardown")[3:])
    assert result == 1  # Telemetry loss is still reported as a failed run.
    assert marker.exists()
    assert (destination / "latest.pt").read_bytes() == b"durable checkpoint"


@pytest.mark.parametrize("failure", [BrokenPipeError(errno.EPIPE, "closed pipe"),
                                     ValueError("I/O operation on closed file")])
def test_unavailable_stdout_cannot_skip_teardown(tmp_path, monkeypatch, failure):
    marker, hook = teardown_marker(tmp_path)

    class BrokenOutput:
        def write(self, value):
            raise failure

        def flush(self):
            pass

    monkeypatch.setattr(sys, "stdout", BrokenOutput())
    result = watchdog.main(command(tmp_path, "pass", "--teardown-command-json", hook,
                                    "--allow-teardown")[3:])
    assert result == 1
    assert marker.exists()
    assert any(row["event"] == "teardown_complete" for row in events(tmp_path))


def test_unexpected_final_sync_error_cannot_skip_teardown(tmp_path, monkeypatch):
    marker, hook = teardown_marker(tmp_path)
    real_popen = subprocess.Popen

    def broken_sync(args, *positional, **kwargs):
        if str(args[0]).endswith("/sync.sh"):
            raise RuntimeError("injected unexpected sync failure")
        return real_popen(args, *positional, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", broken_sync)
    result = watchdog.main(command(tmp_path, "pass", "--sync-source", str(tmp_path),
                                    "--sync-dest", str(tmp_path / "destination"),
                                    "--teardown-command-json", hook, "--allow-teardown")[3:])
    assert result == 1
    assert marker.exists()
    assert any(row["event"] == "final_sync_failed" for row in events(tmp_path))


def test_sigterm_during_spawn_still_owns_and_reaps_child(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    children = []

    def signal_before_assignment(args, *positional, **kwargs):
        process = real_popen(args, *positional, **kwargs)
        children.append(process)
        # Reproduce delivery after creation but before Popen returns to the
        # watchdog's `child = ...` assignment. Its handler must not raise.
        os.kill(os.getpid(), signal.SIGTERM)
        return process

    monkeypatch.setattr(subprocess, "Popen", signal_before_assignment)
    try:
        result = watchdog.main(command(tmp_path, "import time; time.sleep(60)")[3:])
        assert result == 128 + signal.SIGTERM
        assert len(children) == 1
        assert children[0].poll() is not None
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
