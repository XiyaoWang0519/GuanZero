"""Pack the Botzone bot into one zip for upload as a Python 3 bot.

The archive holds a ``gzbot`` package (the NumPy-only modules of this
directory), a ``__main__.py`` that runs it, and optionally the actor weights
at the root (``gz_actor.npz``). Without ``--embed`` the bot looks for
``data/gz_actor.npz`` in Botzone's user storage instead. Usage::

    python -m eval.botzone.pack .work/botzone/u15094.npz .work/botzone/dist/gz_bot.zip --embed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

MODULES = ("protocol.py", "pyengine.py", "mirror.py", "numpy_actor.py", "bot.py")
MAIN = '''import sys
from gzbot.bot import main

main(sys.argv[1:])
'''


def pack(weights: str | Path | None, out: str | Path, embed: bool) -> dict:
    here = Path(__file__).resolve().parent
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("__main__.py", MAIN)
        z.writestr("gzbot/__init__.py", '"""GuanZero Botzone bot (packed)."""\n')
        for name in MODULES:
            z.write(here / name, f"gzbot/{name}")
        if embed:
            if weights is None:
                raise ValueError("--embed needs weights")
            z.write(weights, "gz_actor.npz")
    return {"out": str(out), "bytes": out.stat().st_size, "embedded": bool(embed)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("weights", nargs="?")
    parser.add_argument("out")
    parser.add_argument("--embed", action="store_true", help="put the weights inside the zip")
    args = parser.parse_args(argv)
    print(json.dumps(pack(args.weights, args.out, args.embed), indent=2))


if __name__ == "__main__":
    main()
