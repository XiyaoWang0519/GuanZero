"""Supervise a training command using explicit time and estimated compute limits.

No shell interprets commands. Instance deletion is an optional, operator-supplied
argv hook, enabled separately with --allow-teardown. A process watchdog cannot
bound an actual cloud bill if the provider keeps charging after the process exits.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Sequence


def positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def nonnegative(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def argv_json(value: str) -> list[str]:
    try:
        args = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("expected a JSON argv array") from exc
    if not isinstance(args, list) or not args or not all(
        isinstance(arg, str) and arg and "\x00" not in arg for arg in args
    ):
        raise argparse.ArgumentTypeError("expected a nonempty array of nonempty strings")
    return args


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--budget-usd", required=True, type=positive)
    result.add_argument("--hourly-rate-usd", required=True, type=positive)
    result.add_argument("--spent-usd", default=0.0, type=nonnegative,
                        help="already spent against this budget, including earlier runs")
    result.add_argument("--max-hours", required=True, type=positive)
    result.add_argument("--grace-seconds", default=30.0, type=positive)
    result.add_argument("--poll-seconds", default=1.0, type=positive)
    result.add_argument("--log-file", type=Path)
    result.add_argument("--sync-source", type=Path)
    result.add_argument("--sync-dest")
    result.add_argument("--sync-interval-seconds", default=60.0, type=positive)
    result.add_argument("--hook-timeout-seconds", default=30.0, type=positive)
    result.add_argument("--teardown-command-json", type=argv_json)
    result.add_argument("--allow-teardown", action="store_true")
    result.add_argument("command", nargs=argparse.REMAINDER)
    return result


def signal_group(process: subprocess.Popen, signum: int) -> None:
    """The child starts a new session, so signals never hit the caller's group."""
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def main(argv: Sequence[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        cli.error("a training command is required after --")
    if args.spent_usd >= args.budget_usd:
        cli.error("budget is already exhausted; refusing to start")
    if bool(args.sync_source) != bool(args.sync_dest):
        cli.error("--sync-source and --sync-dest must be supplied together")
    if bool(args.teardown_command_json) != args.allow_teardown:
        cli.error("teardown requires both --teardown-command-json and --allow-teardown")
    budget_seconds = (args.budget_usd - args.spent_usd) / args.hourly_rate_usd * 3600
    runtime_seconds = min(args.max_hours * 3600, budget_seconds)
    if not math.isfinite(runtime_seconds):
        cli.error("computed runtime must be finite")
    # Save, final sync and teardown time all consume the same estimated allowance.
    hook_reserve = args.hook_timeout_seconds * (
        int(bool(args.sync_dest)) + int(bool(args.teardown_command_json))
    )
    reserve = args.grace_seconds + hook_reserve
    if runtime_seconds <= reserve:
        cli.error("remaining allowance must exceed graceful shutdown and hook timeouts")
    started = time.monotonic()
    deadline = started + runtime_seconds
    stop_at = deadline - reserve
    received_signal = 0
    telemetry_failed = False
    stdout_enabled = True
    log_enabled = args.log_file is not None

    def report_output_failure(channel: str, exc: Exception) -> None:
        nonlocal telemetry_failed
        telemetry_failed = True
        # A full disk or disconnected tmux/pipe must never bypass cleanup or
        # opted-in provider teardown. Even the fallback diagnostic is best effort.
        try:
            print(f"watchdog: {channel} output unavailable: {exc}", file=sys.stderr, flush=True)
        except (OSError, ValueError):
            pass

    def emit(event: str, **fields: object) -> None:
        nonlocal stdout_enabled, log_enabled
        elapsed = time.monotonic() - started
        item = dict(event=event, elapsed_seconds=elapsed,
                    estimated_spend_usd=args.spent_usd + elapsed / 3600 * args.hourly_rate_usd,
                    **fields)
        line = json.dumps(item, sort_keys=True)
        if stdout_enabled:
            try:
                print(line, flush=True)
            except (OSError, ValueError) as exc:
                stdout_enabled = False
                report_output_failure("stdout", exc)
        if log_enabled:
            try:
                args.log_file.parent.mkdir(parents=True, exist_ok=True)
                with args.log_file.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
            except (OSError, ValueError) as exc:
                log_enabled = False
                report_output_failure("log file", exc)

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal received_signal
        received_signal = signum

    def run_hook(command_args: list[str], name: str, timeout: float) -> bool:
        if timeout <= 0:
            emit(name + "_failed", error="no time remaining")
            return False
        try:
            # Hooks can contain SSH/provider subprocesses. Bound their whole group.
            hook = subprocess.Popen(command_args, start_new_session=True)
            try:
                code = hook.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                signal_group(hook, signal.SIGKILL)
                hook.wait()
                emit(name + "_failed", error="timeout")
                return False
        except OSError as exc:
            emit(name + "_failed", error=str(exc))
            return False
        emit(name + ("_complete" if code == 0 else "_failed"), returncode=code)
        return code == 0

    sync_command = None
    if args.sync_dest:
        sync_command = [str(Path(__file__).with_name("sync.sh")),
                        str(args.sync_source.resolve()), args.sync_dest]

    handlers = {sig: signal.signal(sig, request_stop)
                for sig in (signal.SIGINT, signal.SIGTERM)}
    child = None
    returncode = 1
    reason = "failed"
    try:
        emit("start", runtime_seconds=runtime_seconds,
             training_seconds=runtime_seconds - reserve,
             teardown_enabled=args.allow_teardown,
             billing_note="estimate only; cloud billing requires provider teardown")
        # request_stop only records the signal. A SIGINT/SIGTERM during Popen
        # therefore cannot unwind before the new process is assigned to child;
        # the loop will immediately stop that now-owned process group.
        child = subprocess.Popen(command, start_new_session=True)
        next_sync = started + args.sync_interval_seconds
        while child.poll() is None:
            now = time.monotonic()
            if received_signal or now >= stop_at:
                reason = (signal.Signals(received_signal).name if received_signal else
                          "budget" if budget_seconds <= args.max_hours * 3600 else "walltime")
                emit("stopping", reason=reason)
                signal_group(child, received_signal or signal.SIGTERM)
                # Never extend the hard boundary after a late wakeup or signal.
                timeout = max(0.0, min(args.grace_seconds, deadline - hook_reserve - now))
                try:
                    child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    emit("killing", reason="grace_expired")
                    signal_group(child, signal.SIGKILL)
                    child.wait()
                returncode = 128 + received_signal if received_signal else 124
                break
            if sync_command and now >= next_sync:
                timeout = min(args.hook_timeout_seconds, stop_at - now)
                if not run_hook(sync_command, "sync", timeout):
                    # Losing the durable checkpoint destination stops expensive work.
                    reason = "sync_failed"
                    signal_group(child, signal.SIGTERM)
                    try:
                        child.wait(timeout=max(0.0, min(args.grace_seconds,
                                                       deadline - hook_reserve - time.monotonic())))
                    except subprocess.TimeoutExpired:
                        signal_group(child, signal.SIGKILL)
                        child.wait()
                    returncode = 1
                    break
                next_sync = time.monotonic() + args.sync_interval_seconds
            time.sleep(max(0.0, min(args.poll_seconds, stop_at - time.monotonic())))
        else:
            code = child.returncode
            returncode = code if code >= 0 else 128 - code
            reason = "completed" if code == 0 else "failed"
    except OSError as exc:
        emit("launch_failed", error=str(exc))
    finally:
        sync_ok = True
        teardown_ok = True
        try:
            try:
                if child is not None:
                    # Also removes orphaned worker processes after a parent exits.
                    signal_group(child, signal.SIGKILL)
                    child.wait()
            except Exception as exc:
                returncode = 1
                emit("child_cleanup_failed", error=str(exc))
            try:
                if sync_command:
                    teardown_reserve = args.hook_timeout_seconds if args.allow_teardown else 0.0
                    sync_ok = run_hook(sync_command, "final_sync", min(
                        args.hook_timeout_seconds, deadline - time.monotonic() - teardown_reserve))
            except Exception as exc:
                sync_ok = False
                emit("final_sync_failed", error=str(exc))
        finally:
            # Keep provider teardown structurally independent of both cleanup
            # and reporting: no failure above may leave an opted-in node billing.
            try:
                if args.allow_teardown:
                    teardown_ok = run_hook(args.teardown_command_json, "teardown", min(
                        args.hook_timeout_seconds, deadline - time.monotonic()))
            except Exception as exc:
                teardown_ok = False
                emit("teardown_failed", error=str(exc))
            finally:
                if not sync_ok or not teardown_ok or telemetry_failed:
                    returncode = 1
                try:
                    emit("stopped", reason=reason, returncode=returncode)
                finally:
                    for sig, handler in handlers.items():
                        signal.signal(sig, handler)
    if telemetry_failed:
        returncode = 1
    return returncode


if __name__ == "__main__":
    sys.exit(main())
