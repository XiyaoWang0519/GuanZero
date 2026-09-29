"""List candidate illustrative moments from the viewer matches' decision records."""
import json
import sys

from review_lib import cards_str

BOMBS = ("Bomb", "StraightFlush", "JokerBomb")
SEAT = ["南", "东", "北", "西"]
decs = [json.loads(l) for f in ("viewA", "viewB") for l in open(f + ".decisions.jsonl")]
games = {g["match_id"]: g for f in ("viewA", "viewB") for g in json.load(open(f + ".games.json"))}


def show(d):
    g = games[int(d["match"][1:])]
    rnd = g["rounds"][d["round"]]
    steps = rnd["steps"][max(0, d["step"] - 6): d["step"]]
    lvl = d["level"]
    ctx = " | ".join(f"{s['step_no']}{SEAT[s['seat']]}:{'过' if s['pass'] else s['type_zh'] + ' ' + ''.join(c['r'] for c in s['cards'])}"
                     for s in [dict(x, step_no=i + max(1, d['step'] - 5)) for i, x in enumerate(steps)])
    print(f"--- {d['match']} 第{d['round'] + 1}回合(打{rnd['level_rank']}) 第{d['step']}手 {SEAT[d['seat']]}({d['label']}) left={d['left']} top={d['top_rel']}")
    print("   hand:", cards_str(d["hand"], lvl))
    print("   ctx :", ctx)
    print("   play:", "过" if d["pass"] else d["type"] + " " + cards_str(d["cards"], lvl), " n_legal", d["n_legal"])
    if d["probs"]:
        print("   top :", "; ".join(f"{a} {p:.2f}" for a, p in d["probs"]))


kind = sys.argv[1]
for d in decs:
    if d["forced"] or d["label"] == "b11":
        continue
    left = d["left"]; s = d["seat"]
    opp = [left[x] for x in range(4) if x % 2 != s % 2 and left[x] > 0]
    mo = min(opp) if opp else 99
    if kind == "bomb_partner" and d["type"] in BOMBS and d["top_rel"] == "partner":
        show(d)
    elif kind == "pass_danger" and d["pass"] and d["top_rel"] == "opponent" and d["has_nonpass"] and mo <= 2:
        show(d)
    elif kind == "bomb_stop" and d["type"] in BOMBS and mo <= 3:
        show(d)
    elif kind == "lowconf" and d["probs"] and d["probs"][0][1] < 0.4 and len(d["hand"]) <= 12:
        show(d)
