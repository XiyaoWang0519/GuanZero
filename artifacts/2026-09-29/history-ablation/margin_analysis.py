"""Argmax flips by FULL's own confidence margin (top-1 minus top-2 probability)."""
import numpy as np
from analyze_probe import load

for kind in ("dup", "match"):
    recs, _ = load(kind)
    recs = [r for r in recs if r["n_legal"] >= 2]
    for v in ("none", "round", "swap"):
        rs = [r for r in recs if r["tv"][v] is not None and (v == "none" or r["round"] > 0)]
        if not rs:
            continue
        bins = {"<0.05": [], "0.05-0.2": [], "0.2-0.5": [], ">=0.5": []}
        for r in rs:
            p = sorted((c["full"] for c in r["cands"]), reverse=True)
            m = p[0] - (p[1] if len(p) > 1 else 0.0)
            k = "<0.05" if m < 0.05 else "0.05-0.2" if m < 0.2 else "0.2-0.5" if m < 0.5 else ">=0.5"
            bins[k].append(r["arg"][v] != r["arg"]["full"])
        flips = sum(sum(b) for b in bins.values())
        print(kind, v, "total flips", flips, " | ".join(
            f"margin {k}: n={len(b)} flips={sum(b)} ({np.mean(b) if b else 0:.2%})" for k, b in bins.items()))
