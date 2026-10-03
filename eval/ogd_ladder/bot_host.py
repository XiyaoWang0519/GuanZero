"""Runs one competition bot in its own process: JSON lines in, JSON lines out.

Usage (spawned by ``eval.ogd_ladder.bots.BotProcess``)::

    python -m eval.ogd_ladder.bot_host <bot-name> <work-dir>

``work-dir`` is a private copy of the bot's source directory; it becomes the
working directory and the head of ``sys.path``, so each bot finds its own
``state``/``action``/``utils`` modules and writes its scratch files (some
keep state in ``data1.txt``) where no other bot sees them. Each request line
is ``{"msg": <server message>}``; each reply line is ``{"act": <index or
null>, "err": <text or null>}``. The bots print freely, so file descriptor 1
is pointed at the log before any bot code runs and the protocol uses a
duplicate of the original stdout.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import traceback


def _old_action(action):
    """OpenGuanDan names the four-joker bomb ``FourKings``; the 2020 server
    sent it as ``["Bomb", "JOKER", cards]``, which is what the bots parse."""
    if isinstance(action, list) and action and action[0] == "FourKings":
        return ["Bomb", "JOKER", action[2]]
    return action


def to_2020(msg: dict) -> dict:
    """The competition-era form of an OpenGuanDan message."""
    msg = dict(msg)
    if "actionList" in msg:
        msg["actionList"] = [_old_action(a) for a in msg["actionList"]]
    for key in ("curAction", "greaterAction"):
        if key in msg:
            msg[key] = _old_action(msg[key])
    if isinstance(msg.get("publicInfo"), list):
        msg["publicInfo"] = [dict(p, playArea=_old_action(p.get("playArea"))) for p in msg["publicInfo"]]
    return msg


def main() -> int:
    name, work_dir = sys.argv[1], sys.argv[2]
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    log = open(os.path.join(work_dir, "_bot.log"), "w", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout = sys.stderr = log
    os.chdir(work_dir)
    sys.path.insert(0, work_dir)
    glue = importlib.import_module(f"eval.ogd_ladder.glue.{name.replace('-', '_')}")
    bot = glue.Glue()
    for line in sys.stdin:
        msg = to_2020(json.loads(line)["msg"])
        act, err = None, None
        try:
            act = bot.handle(msg)
            if act is not None:
                act = int(act)
        except Exception:  # noqa: BLE001 - a bot crash is data, reported upstream
            err = traceback.format_exc(limit=4)[-800:]
            print(err, flush=True)
        proto.write(json.dumps({"act": act, "err": err}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
