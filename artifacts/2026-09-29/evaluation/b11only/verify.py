"""Compare the B11-only u2201 run with the original two-baseline u2201 run.

Writes b11only/verify.json and b11only/VERDICT (PASS or FAIL). Compares the whole
b11-main report: all 256 per-deal pair scores, all duplicate leg results, every
summary field, and all 64 match pairs.
"""
import json
from pathlib import Path

W = Path("/Users/xiyaowang/Developer/Projects/GuanZero/.work/longrun-batched-eval-2026-09-29")
old = json.loads((W / "out/u2201.json").read_text())
new = json.loads((W / "out-b11/u2201.json").read_text())
a, b = old["reports"]["b11-main"], new["reports"]["b11-main"]
ps_a, ps_b = a["duplicates"]["pair_scores"], b["duplicates"]["pair_scores"]
pairs_a, pairs_b = a["full_matches"]["pairs"], b["full_matches"]["pairs"]
res = dict(
    candidate_sha_equal=old["candidate_sha256"] == new["candidate_sha256"],
    source_sha_equal=old["evaluation_source_sha256"] == new["evaluation_source_sha256"],
    evaluation_source_sha256=new["evaluation_source_sha256"],
    freeze_sha256_original=old["freeze_sha256"], freeze_sha256_b11_only=new["freeze_sha256"],
    baselines_in_new=list(new["reports"]),
    n_pair_scores=(len(ps_a), len(ps_b)),
    pair_scores_differing=sum(x != y for x, y in zip(ps_a, ps_b)),
    n_match_pairs=(len(pairs_a), len(pairs_b)),
    match_pairs_differing=sum(x != y for x, y in zip(pairs_a, pairs_b)),
    duplicates_block_identical=a["duplicates"] == b["duplicates"],
    full_matches_block_identical=a["full_matches"] == b["full_matches"],
    whole_b11_report_identical=a == b,
)
ok = (res["candidate_sha_equal"] and res["source_sha_equal"] and len(ps_a) == len(ps_b) == 256
      and len(pairs_a) == len(pairs_b) == 64 and res["whole_b11_report_identical"])
res["verdict"] = "PASS" if ok else "FAIL"
(W / "b11only/verify.json").write_text(json.dumps(res, indent=2) + "\n")
(W / "b11only/VERDICT").write_text(res["verdict"] + "\n")
print(json.dumps(res, indent=2))
