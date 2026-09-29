"""Build game-review.html from a copy of .work/demo/viewer.template.html and the recorded games."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
TEMPLATE = HERE.parent / "demo" / "viewer.template.html"

games = json.load(open(HERE / "viewA.games.json")) + json.load(open(HERE / "viewB.games.json"))
games.sort(key=lambda g: g["match_id"])
for g in games:
    for r in g["rounds"]:
        r.pop("seat_return", None)
doc = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rules": "house",
       "note": "GuanZero game review 2026-09-29; greedy play; source export b8c0c54",
       "games": games}
(HERE / "games.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
payload = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
tpl = TEMPLATE.read_text(encoding="utf-8")
(HERE / "viewer.template.html").write_text(tpl, encoding="utf-8")   # the copy used
assert tpl.count("/*GAMES_JSON*/null") == 1
html = tpl.replace("/*GAMES_JSON*/null", payload)
html = html.replace("<title>掼蛋牌局回放</title>", "<title>GuanZero 牌局回放 · Transformer u2623</title>")
html = html.replace("均为模型之间的真实对局，按引擎规则逐步回放",
                    "均为模型之间的真实对局（贪心出牌），按引擎规则逐步回放；队名标明哪一方是 Transformer 及其检查点")
out = HERE / "game-review.html"
out.write_text(html, encoding="utf-8")

# browser-free check: extract the embedded literal and parse it
text = out.read_text(encoding="utf-8")
m = re.search(r"const DATA = (.*?);\nconst POS", text, re.S)
data = json.loads(m.group(1).replace("<\\/", "</"))
assert data == json.loads(json.dumps(doc, ensure_ascii=False))
print("embedded JSON parses:", len(data["games"]), "games,",
      sum(len(g["rounds"]) for g in data["games"]), "rounds,", f"{len(text.encode()) / 1e6:.2f} MB")
