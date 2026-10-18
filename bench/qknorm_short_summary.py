"""Verify and summarize the capped PPO A/B; reports no playing-strength inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value) -> bool:
    if isinstance(value,float):return math.isfinite(value)
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,list):return all(finite(v) for v in value)
    return True


def summarize(root: Path) -> dict:
    arms={}
    receipts={}
    for arm in ("control","norm16"):
        folder=root/"runs"/arm
        receipt=json.loads((folder/"experiment.json").read_text())
        complete=json.loads((folder/"complete.json").read_text())
        lines=[json.loads(x) for x in (folder/"diagnostic.jsonl").read_text().splitlines()]
        assert complete["completed"] and complete["updates"]==120 and len(lines)==120
        assert [r["index"] for r in lines]==list(range(1,121))
        assert all(r["all_parameters_optimizer_finite"] for r in lines)
        assert finite(lines)
        assert sha(folder/"endpoint.pt")==complete["endpoint_sha256"]
        assert [r["update"] for r in lines]==list(range(94489,94609))
        assert sum(r["step_decisions"] for r in lines)==complete["actual_decisions"]
        clips=[x for r in lines for x in r["clipping"]]
        assert all(c["after"]<=10.0001 for c in clips)
        assert all(len(r["clipping"])==r["minibatches"] for r in lines)
        points={}
        for point in (0,30,60,120):
            p=json.loads((folder/f"probe-{point:03d}.json").read_text())
            assert finite(p) and p["index"]==point
            views={}
            for name,data in p["datasets"].items():
                intervention=data["interventions"]
                mean=intervention["variants"]["calibration_mean"]
                zero=intervention["variants"]["zero"]
                views[name]=dict(mean_tv=mean["tv"]["mean"],mean_kl=mean["kl"]["mean"],
                    mean_argmax_flip=mean["argmax_flip_fraction"],zero_tv=zero["tv"]["mean"],
                    normalized_entropy=[h["normalized_entropy"]["p50"] for h in data["heads"]],
                    query_cosine=[h["query_cosine"]["p50"] for h in data["heads"]])
            fresh=[v for n,v in views.items() if n.startswith("fresh-")]
            points[str(point)]=dict(datasets=views,spectrum=p["spectrum"],
                fresh_mean_tv=statistics.mean(v["mean_tv"] for v in fresh),
                fresh_mean_kl=statistics.mean(v["mean_kl"] for v in fresh),
                fresh_mean_argmax_flip=statistics.mean(v["mean_argmax_flip"] for v in fresh),
                fresh_zero_tv=statistics.mean(v["zero_tv"] for v in fresh))
        config=receipt["config"]
        assert config["aux_warmup_updates"]==0
        totals=[r["policy_loss"]-config["entropy"]*r["entropy"]+
                config["response_coef"]*r["response_loss"]+config["aux_coef"]*r["aux_loss"]
                for r in lines]
        arms[arm]=dict(completed_updates=120,actual_decisions=complete["actual_decisions"],
            optimizer_steps=sum(r["minibatches"] for r in lines),
            actor_gradient_norm_mean=statistics.mean(c["before"] for c in clips),
            actor_gradient_norm_max=max(c["before"] for c in clips),
            actor_gradient_norm_post_max=max(c["after"] for c in clips),
            gradient_clip_fraction=statistics.mean(c["clipped"] for c in clips),
            entropy_mean=statistics.mean(r["entropy"] for r in lines),
            ppo_approx_kl_mean=statistics.mean(r["approx_kl"] for r in lines),
            ppo_approx_kl_max=max(r["approx_kl"] for r in lines),
            ppo_clip_fraction_mean=statistics.mean(r["clip_fraction"] for r in lines),
            full_actor_loss_reconstructed_mean=statistics.mean(totals),
            full_actor_loss_reconstructed_max=max(totals),
            relative_net_q_update_median=statistics.median(v for r in lines for v in r["direction"]["q"]["relative_net_tangent_update"]),
            relative_net_k_update_median=statistics.median(v for r in lines for v in r["direction"]["k"]["relative_net_tangent_update"]),
            effective_scale_max=max(v for r in lines for v in r["effective_scale"]),
            peak_allocated_bytes=max(r["cuda_peak_allocated_bytes"] for r in lines),
            peak_reserved_bytes=max(r["cuda_peak_reserved_bytes"] for r in lines),
            allocation_retries_final=lines[-1]["cuda_allocation_retries"],
            end_norms=lines[-1]["direction"],points=points,
            endpoint_sha256=complete["endpoint_sha256"],
            diagnostics_sha256=sha(folder/"diagnostic.jsonl"))
        receipts[arm]=receipt
    for key in ("parent_sha256","source","initial_population","rng","config","world_size","gpu","wrapper_sha256"):
        assert receipts["control"][key]==receipts["norm16"][key],key
    assert receipts["control"]["migration"] is None
    assert receipts["norm16"]["migration"]["initial_population_unchanged"]
    change={a:arms[a]["points"]["120"]["fresh_mean_tv"]-arms[a]["points"]["0"]["fresh_mean_tv"] for a in arms}
    return dict(completed=True,arms=arms,parent_sha256=receipts["control"]["parent_sha256"],
        matching_initial_population_rng_and_config=True,
        source=receipts["control"]["source"],wrapper_sha256=receipts["control"]["wrapper_sha256"],
        fresh_mean_tv_change=change,descriptive_difference_in_changes=change["norm16"]-change["control"],
        scope="one paired continuation, one rank, 120 updates/arm; no strength or multi-seed inference")


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=summarize(args.root)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    for arm,r in result["arms"].items():
        print(arm,"updates",r["completed_updates"],"decisions",r["actual_decisions"],
              "max gradient",r["actor_gradient_norm_max"],"sensitivity",r["points"]["0"]["fresh_mean_tv"],
              "->",r["points"]["120"]["fresh_mean_tv"])


if __name__=="__main__":main()
