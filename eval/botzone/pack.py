"""Pack the Botzone bot into one zip for upload as a Python 3 bot.

The archive holds a ``gzbot`` package (the NumPy-only modules of this
directory), a ``__main__.py`` that runs it, and optionally the actor weights
at the root (``gz_actor.npz``). Without ``--embed`` the bot looks for
``data/gz_actor.npz`` in Botzone's user storage instead. Usage::

    python -m eval.botzone.pack .work/botzone/u15094.npz .work/botzone/dist/gz_bot.zip --embed
"""
from __future__ import annotations

import argparse
import base64
import zlib
import json
from pathlib import Path
import zipfile

MODULES = ("protocol.py", "pyengine.py", "mirror.py", "numpy_actor.py", "bot.py", "search.py")
MAIN = '''import sys
from gzbot.bot import main

main(sys.argv[1:])
'''


def pack(weights: str | Path | None, out: str | Path, embed: bool, search: bool = False, weights_name: str | None = None) -> dict:
    here = Path(__file__).resolve().parent
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        extra = (["--search"] if search else [])
        if weights_name:
            if Path(weights_name).name != weights_name:
                raise ValueError("weights_name must be a filename")
            extra += ["--weights", "data/" + weights_name]
        z.writestr("__main__.py", MAIN.replace("main(sys.argv[1:])",
                      "main(sys.argv[1:] + %r)" % extra))
        z.writestr("gzbot/__init__.py", '"""GuanZero Botzone bot (packed)."""\n')
        for name in MODULES:
            z.write(here / name, f"gzbot/{name}")
        if embed:
            if weights is None:
                raise ValueError("--embed needs weights")
            z.write(weights, "gz_actor.npz")
    return {"out": str(out), "bytes": out.stat().st_size, "embedded": bool(embed)}


def pack_single(out: str | Path, weights_name: str, search: bool = False) -> dict:
    """Plain Python alternative for the site's online source editor."""
    if Path(weights_name).name != weights_name:
        raise ValueError("weights_name must be a filename")
    here = Path(__file__).resolve().parent
    lines = [
        '# GuanZero Botzone bot, Python 3.6 / NumPy.',
        'import os',
        'for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):',
        '    os.environ[name] = "1"',
        'import base64, zlib, sys, types',
        'package = types.ModuleType("gzbot")',
        'package.__path__ = []',
        'sys.modules["gzbot"] = package',
    ]
    for name in ("protocol", "pyengine", "numpy_actor", "mirror", "search", "bot"):
        payload = base64.b64encode(zlib.compress((here / (name + ".py")).read_bytes(), 9)).decode()
        lines.extend([
            'module = types.ModuleType("gzbot.%s")' % name,
            'module.__file__ = __file__',
            'module.__package__ = "gzbot"',
            'sys.modules[module.__name__] = module',
            'setattr(package, %r, module)' % name,
            'exec(compile(zlib.decompress(base64.b64decode(%r)), module.__name__, "exec"), module.__dict__)' % payload,
        ])
    args = (["--search"] if search else []) + ["--weights", "data/" + weights_name]
    lines.append('package.bot.main(%r)' % args)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"out": str(out), "bytes": out.stat().st_size, "embedded": False}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("weights", nargs="?")
    parser.add_argument("out")
    parser.add_argument("--embed", action="store_true", help="put the weights inside the zip")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--weights-name", help="versioned model in Botzone user storage")
    parser.add_argument("--single-file", action="store_true")
    args = parser.parse_args(argv)
    if args.single_file:
        if args.embed or not args.weights_name:
            parser.error("--single-file needs --weights-name and cannot embed weights")
        result = pack_single(args.out, args.weights_name, args.search)
    else:
        result = pack(args.weights, args.out, args.embed, args.search, args.weights_name)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
