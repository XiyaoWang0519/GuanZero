import sys, torch
from pathlib import Path
from eval.history_frozen import evaluate
torch.set_num_threads(4)
freeze, ckpt, out = map(Path, sys.argv[1:4])
r = evaluate(freeze, ckpt, out)
for name, rep in r["reports"].items():
    d = rep["duplicates"]
    print(name, d["mean_net_levels_per_round"], d.get("bootstrap_95_ci"), "win", rep["full_matches"]["win_rate"], flush=True)
