"""Print the nightly evaluation summary of one .work/nightly-eval/NAME directory."""
import json, sys, glob, re
from pathlib import Path
W = Path(sys.argv[1])
e = json.load(open(W / "endpoint-vs-baseline.json")); d = json.load(open(W / "endpoint-vs-danlm.json"))
b = json.load(open(W / "endpoint-vs-b11.json"))["reports"]["b11-main"]["duplicates"]
print(f"endpoint vs previous endpoint: {e['mean_net_levels_per_round']:+.3f} {[round(x,3) for x in e['bootstrap_95_ci']]} ({e['deals']} deals)")
print(f"endpoint vs DanLM:             {d['levels_per_round']:+.3f} {[round(x,3) for x in d['levels_per_round_ci95']]}  win {d['round_win_rate']*100:.1f}%")
print(f"endpoint vs B11 (sanity):      {b['mean_net_levels_per_round']:+.3f} (256 deals)")
rows = sorted((int(re.search(r"u(\d+)", f).group(1)), json.load(open(f))) for f in glob.glob(str(W / "curve" / "u*.json")))
if rows:
    print(f"curve vs previous endpoint ({rows[0][1]['deals']} deals per point):")
    for u, r in rows:
        print(f"  u{u}: {r['mean_net_levels_per_round']:+.3f} {[round(x,3) for x in r['bootstrap_95_ci']]}")
