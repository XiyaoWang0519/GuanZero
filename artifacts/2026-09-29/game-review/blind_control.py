"""Control: the same deals with the Transformer's history withheld (observe() is a no-op).

If choices differ from the history-fed run, a recorder that failed to feed history
would have produced different (blind) games; this shows the check above is not vacuous.
"""
import json

import torch

from eval.duplicate import generate_deals
from eval.policies import load_policy
import review_lib as R

torch.set_num_threads(2)
ROOT = "/Users/xiyaowang/Developer/Projects/GuanZero/"
cand = load_policy(ROOT + ".work/longrun-batched-eval-2026-09-29/ckpt/u2623.pt", "cpu")
R.instrument_history(cand)
b11 = load_policy(ROOT + ".work/runpod-longrun-2026-09-25/payload/artifacts/b11-main.pt", "cpu", margin=0.0)
deals = generate_deals(10, seed=777)
res = []
for i, deal in enumerate(deals):
    fed = []
    R.play_deal_recorded(deal, (cand, b11, cand, b11), ["x"] * 4, [], seed=i, trace=fed)
    blind = []
    real_observe = cand.observe
    cand.observe = lambda event: None
    R.play_deal_recorded(deal, (cand, b11, cand, b11), ["x"] * 4, [], seed=i, trace=blind)
    cand.observe = real_observe
    first = next((k for k, (a, b) in enumerate(zip(fed, blind)) if a[:4] != b[:4]), None)
    res.append({"deal": i, "fed_decisions": len(fed), "blind_decisions": len(blind),
                "first_divergence": first})
    print(res[-1], flush=True)
json.dump(res, open("blind_control.json", "w"), indent=1)
