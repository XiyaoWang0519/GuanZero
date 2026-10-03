"""Seat agents that speak the OpenGuanDan message protocol.

An agent gets every message the table sends to its seat through ``handle``
and returns an ``actIndex`` for ``act`` messages (``None`` otherwise).
``BotProcess`` runs a competition bot out of process; when the bot crashes,
times out or answers out of range, the seat plays a uniformly random legal
index for that decision and the failure is counted, never hidden.
"""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import random
import select
import shutil
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]

# Directory of each bot inside ``BOTS_ROOT`` (DanLM's ``baselines/``): the one
# that holds its ``state.py``.
BOT_DIRS = {
    "1st-lalala": "1st-lalala",
    "2nd-egg-pancake": "2nd-egg-pancake",
    "2nd-flush-bomb": "2nd-flush-bomb",
    "2nd-no-ai": "2nd-no-ai",
    "3rd-chick-squad": "3rd-chick-squad/幺鸡小分队",
    "3rd-ez": "3rd-ez",
    "3rd-hulalala": "3rd-hulalala/new_version",
    "fin-caiji": "fin-caiji/caiji",
    "fin-egg-expert": "fin-egg-expert/吃蛋能手",
    "fin-go-go-go": "fin-go-go-go/冲冲冲",
    "fin-guanglan-iot": "fin-guanglan-iot/光蓝物联",
    "fin-honest-seu": "fin-honest-seu/东大老实人",
    "fin-njupt-guandan-ai": "fin-njupt-guandan-ai",
    "fin-xishang-tech": "fin-xishang-tech/锡商科技",
    "part-369": "part-369/369",
    "part-youre-right": "part-youre-right/你说的都对",
}


def bots_root() -> Path:
    root = Path(os.environ.get("BOTS_ROOT", "").strip() or ".work/external/DanLM/baselines")
    if not root.is_dir():
        raise FileNotFoundError(f"competition bots not found at {root}; set BOTS_ROOT to the "
                                "baselines/ directory of a DanLM checkout")
    return root


class RandomAgent:
    name = "random"

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.stats: Counter = Counter()

    def handle(self, msg: dict) -> int | None:
        if msg.get("type") != "act":
            return None
        return self.rng.randrange(int(msg["indexRange"]) + 1)

    def close(self) -> None:
        pass


class BotProcess:
    """One competition bot in a subprocess, with a private working copy."""

    def __init__(self, name: str, rng: random.Random, timeout: float = 10.0) -> None:
        self.name, self.rng, self.timeout = name, rng, timeout
        self.stats: Counter = Counter()
        self.errors: list[str] = []
        self.work = Path(tempfile.mkdtemp(prefix=f"bot-{name}-"))
        shutil.copytree(bots_root() / BOT_DIRS[name], self.work, dirs_exist_ok=True)
        env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONIOENCODING="utf-8",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "eval.ogd_ladder.bot_host", name, str(self.work)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
            env=env, cwd=str(REPO))

    def _ask(self, msg: dict) -> dict | None:
        if self.proc.poll() is not None:
            return None
        self.proc.stdin.write(json.dumps({"msg": msg}, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        ready, _, _ = select.select([self.proc.stdout], [], [], self.timeout)
        if not ready:
            return None
        line = self.proc.stdout.readline()
        return json.loads(line) if line else None

    def handle(self, msg: dict) -> int | None:
        reply = self._ask(msg)
        if msg.get("type") != "act":
            if reply is not None and reply.get("err"):
                self._fail("notify_error", reply["err"])
            return None
        hi = int(msg["indexRange"])
        self.stats["decisions"] += 1
        if reply is None:
            self._fail("dead_or_timeout", "")
            if self.proc.poll() is None:
                self.close()      # a timed-out bot is out of sync; stop asking it
        elif reply.get("err"):
            self._fail("act_error", reply["err"])
        elif reply.get("act") is None or not 0 <= int(reply["act"]) <= hi:
            self._fail("act_out_of_range", f"{reply.get('act')} not in 0..{hi}")
        else:
            return int(reply["act"])
        return self.rng.randrange(hi + 1)

    def _fail(self, kind: str, detail: str) -> None:
        self.stats[kind] += 1
        if len(self.errors) < 5:
            self.errors.append(f"{kind}: {detail}")

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self.proc.kill()
        shutil.rmtree(self.work, ignore_errors=True)
