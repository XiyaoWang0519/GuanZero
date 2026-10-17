"""CPU-only checkpoint mechanism audit; never constructs or steps an optimizer.

Run with the matching qknorm branch first on PYTHONPATH (plus python:oracle).
Inputs are trusted local checkpoints/fixtures, loaded with weights_only=False.
All interventions act on detached reader outputs, with the public prefix intact.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import gd
from train.history_model import (DecisionInputs, HistoryActor, HistoryPolicyConfig,
                                MatchGroup, StreamBatch, segment_log_softmax)
from train.history_aux import outcome_loss
from train.history_rollout import HistoryCollector, MatchEventStore, SequenceRolloutBuffer


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(x: torch.Tensor) -> dict:
    x = x.detach().double().flatten()
    if not len(x):
        return {"n": 0}
    assert torch.isfinite(x).all(), "nonfinite diagnostic"
    return dict(n=len(x), mean=float(x.mean()), min=float(x.min()),
                p50=float(x.median()), p95=float(x.quantile(.95)), max=float(x.max()))


def load_actor(path: Path) -> tuple[HistoryActor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actor = HistoryActor(HistoryPolicyConfig(**payload["model_config"]))
    actor.load_state_dict(payload["model"], strict=True)
    actor.eval()
    actor.causal_sdpa = True
    actor.batched_match_attention = True
    return actor, payload


def unpack(raw: dict) -> DecisionInputs:
    raw = dict(raw)
    raw["streams"] = StreamBatch(**raw["streams"])
    raw["match_groups"] = [MatchGroup(**g) for g in raw["match_groups"]]
    return DecisionInputs(**raw)


def capture(actor: HistoryActor, inputs: DecisionInputs) -> dict:
    calls, reader = [], []
    original = F.scaled_dot_product_attention

    def spy(q, k, v, *args, **kwargs):
        result = original(q, k, v, *args, **kwargs)
        if kwargs.get("attn_mask") is not None:
            calls.append((q.detach(), k.detach(), v.detach(), kwargs["attn_mask"], result.detach()))
        return result

    hook = actor.out_proj.register_forward_hook(lambda m, i, o: reader.append(o.detach()))
    F.scaled_dot_product_attention = spy
    try:
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            state = actor.decision_states(None, inputs)
    finally:
        F.scaled_dot_product_attention = original
        hook.remove()
    groups = inputs.match_groups
    assert groups and len(calls) == len(groups) == len(reader)
    attended = torch.zeros_like(state)
    heads = actor.config.heads
    rows = {h: {k: [] for k in ("entropy", "normalized_entropy", "pmax", "gap", "bos_mass",
                                "current_mass", "previous_mass", "q")} for h in range(heads)}
    reference_error = []
    for (q, k, v, mask, out), group, read in zip(calls, groups, reader):
        attended[group.rows] = read
        # Float64 reconstruction diagnoses logits, not CUDA backward behavior.
        score = q.double() @ k.double().transpose(-1, -2) / q.shape[-1] ** .5
        score = score.masked_fill(~mask, float("-inf"))
        valid = score[group.local_match, :, group.match_rank]
        prob = valid.softmax(-1)
        reference = (prob.unsqueeze(-2) @ v.double()[group.local_match]).squeeze(-2)
        actual = out[group.local_match, :, group.match_rank]
        reference_error.append((reference - actual.double()).abs().max())
        ent = -(prob * prob.clamp_min(1e-300).log()).sum(-1)
        count = mask[group.local_match, 0, group.match_rank].sum(-1)
        top = valid.topk(2, dim=-1).values
        # Tokens occupy positions 1..prefix; position 0 is BOS.
        s = inputs.streams
        positions = torch.arange(1, valid.shape[-1])
        r = s.rounds[inputs.match_index[group.rows], :valid.shape[-1] - 1]
        prefix = inputs.prefix[group.rows]
        now = s.rounds[inputs.match_index[group.rows], (prefix - 1).clamp_min(0)]
        visible = positions[None] <= prefix[:, None]
        current = visible & (r == now[:, None])
        previous = visible & (r < now[:, None])
        assert torch.all((current | previous) == visible)
        for h in range(heads):
            values = dict(entropy=ent[:, h],
                          normalized_entropy=ent[:, h] / count.double().log().clamp_min(1e-30),
                          pmax=prob[:, h].max(-1).values,
                          gap=(top[:, h, 0] - top[:, h, 1])[count > 1],
                          bos_mass=prob[:, h, 0],
                          current_mass=(prob[:, h, 1:] * current).sum(-1),
                          previous_mass=(prob[:, h, 1:] * previous).sum(-1),
                          q=q[group.local_match, h, group.match_rank])
            assert torch.allclose(values["bos_mass"] + values["current_mass"] +
                                  values["previous_mass"], torch.ones_like(ent[:, h]), atol=1e-10)
            for key, value in values.items():
                rows[h][key].append(value)
    report = []
    for h in range(heads):
        merged = {k: torch.cat(v) for k, v in rows[h].items()}
        query = F.normalize(merged.pop("q").double(), dim=-1)
        cosine = query @ query.T
        cosine = cosine[~torch.eye(len(query), dtype=torch.bool)]
        report.append({**{k: stats(v) for k, v in merged.items()}, "query_cosine": stats(cosine)})
    return dict(state=state, reader=attended, heads=report,
                reference_max_abs=float(torch.stack(reference_error).max()))


def centered_rms(x: torch.Tensor) -> torch.Tensor:
    x = x.double()
    return (x - x.mean(-1, keepdim=True)).square().mean(-1).sqrt()


def policy(actor: HistoryActor, state: torch.Tensor, inputs: DecisionInputs) -> torch.Tensor:
    logits = actor.candidate_logits(state, inputs.cand, inputs.offsets, rows=inputs.rows)
    return segment_log_softmax(logits, inputs.rows, inputs.decisions).double()


def interventions(actor: HistoryActor, inputs: DecisionInputs, cap: dict,
                  calibration_mean: torch.Tensor, test_rows: torch.Tensor) -> dict:
    with torch.no_grad():
        private = actor.private(inputs.obs.float()) + actor.seat(inputs.seat)
        reader, baseline = cap["reader"], cap["state"]
        state0 = actor.attention_norm(private + reader)
        rebuilt = actor.output_norm(state0 + actor.feed_forward(state0))
        assert torch.allclose(rebuilt, baseline, atol=2e-6, rtol=1e-6)
        logp = policy(actor, baseline, inputs)
        result = dict(private_centered_rms=stats(centered_rms(private)[test_rows]),
                      history_centered_rms=stats(centered_rms(reader)[test_rows]),
                      history_variation_centered_rms=stats(centered_rms(reader-calibration_mean)[test_rows]),
                      variants={})
        for name, replacement in (("identity", reader), ("calibration_mean", calibration_mean.expand_as(reader)),
                                  ("zero", torch.zeros_like(reader))):
            fusion = actor.attention_norm(private + replacement)
            state = actor.output_norm(fusion + actor.feed_forward(fusion))
            other = policy(actor, state, inputs)
            row_ids = inputs.rows
            kl = torch.zeros(inputs.decisions, dtype=torch.float64).index_add_(0, row_ids,
                                    logp.exp() * (logp - other))
            tv = torch.zeros_like(kl).index_add_(0, row_ids, .5 * (logp.exp() - other.exp()).abs())
            flips = []
            for i in test_rows.tolist():
                lo, hi = inputs.offsets[i:i+2].tolist()
                flips.append(logp[lo:hi].argmax() != other[lo:hi].argmax())
            result["variants"][name] = dict(kl=stats(kl[test_rows]), tv=stats(tv[test_rows]),
                argmax_flip_fraction=float(torch.stack(flips).float().mean()),
                fusion_delta_rms=stats((fusion.double()-state0.double()).square().mean(-1).sqrt()[test_rows]),
                state_delta_rms=stats((state.double()-baseline.double()).square().mean(-1).sqrt()[test_rows]))
        assert result["variants"]["identity"]["tv"]["max"] < 1e-8
        return result


def spectrum(actor: HistoryActor) -> dict:
    report = {}
    for name, weight in (("q", actor.q_proj.weight), ("k", actor.kv_proj.weight[:actor.config.width])):
        singular = torch.linalg.svdvals(weight.detach().double())
        report[name] = dict(frobenius=float(singular.norm()), sigma1=float(singular[0]),
                            first_energy_fraction=float(singular[0].square()/singular.square().sum()))
    if actor.config.qk_norm:
        report["effective_scale"] = (actor.qk_temperature().detach() *
                                     (actor.config.width / actor.config.heads) ** .5).tolist()
    return report


def fresh_batch(actor: HistoryActor, seed: int, output: Path) -> tuple[DecisionInputs, dict]:
    """Frozen self-play inference only, sampled evenly across eight envs/time."""
    env = gd.VecEnv(num_envs=8, num_threads=1, seed=seed, log_public_actions=True, log_env_limit=8)
    store, buffer = MatchEventStore(), SequenceRolloutBuffer()
    collector = HistoryCollector(env, actor, store, buffer, torch.Generator().manual_seed(seed+1),
                                 "cpu", kv_cache=True)
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        collector.collect(192)
    data = buffer.compact()
    picked = []
    for e in range(8):
        candidates = np.flatnonzero(data["env"] == e)
        picked.extend(candidates[np.linspace(0, len(candidates)-1, min(24,len(candidates)), dtype=int)])
    picked = np.sort(np.asarray(picked, dtype=np.int64))
    batch = buffer.training_batch(picked, store, "cpu", fields=(), length_groups=2, width=actor.config.width)
    raw = asdict(batch.inputs)
    torch.save(raw, output)
    # Different environments supply calibration and held-out evaluation rows.
    calibration = torch.tensor(np.flatnonzero(data["env"][picked] < 2))
    test = torch.tensor(np.flatnonzero(data["env"][picked] >= 2))
    meta = dict(seed=seed, collected_rows=buffer.n_rows, sampled_rows=len(picked),
                matches=len(set(zip(data["env"][picked].tolist(), data["match"][picked].tolist()))),
                seats=sorted(set(data["seat"][picked].tolist())),
                rounds=sorted(set(data["round"][picked].tolist())),
                prefix=stats(batch.inputs.prefix), calibration_rows=calibration.tolist(), test_rows=test.tolist(),
                sha256=sha(output), path=str(output))
    return batch.inputs, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, help="label=path")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261016)
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {k: Path(v) for k,v in (s.split("=",1) for s in args.checkpoint)}
    fx = torch.load(args.fixture, map_location="cpu", weights_only=False)
    fixture = unpack(fx["batch"]["inputs"])
    batches = {"original_fixture": (fixture, dict(sha256=sha(args.fixture), rows=fixture.decisions))}
    audit = dict(torch=torch.__version__, device="cpu", backend="math", checkpoints={},
                 source=str(Path(inspect.getfile(HistoryActor)).resolve()),
                 source_sha256=sha(Path(inspect.getfile(HistoryActor))), datasets={}, results={})
    for idx, (label,path) in enumerate(paths.items()):
        actor, payload = load_actor(path)
        audit["checkpoints"][label] = dict(path=str(path), sha256=sha(path), progress=payload["progress"],
            model_config=payload["model_config"], lineage=payload["lineage"], spectrum=spectrum(actor),
            config_changes=payload["config_changes"], source=payload["run_identity"]["source"])
        if label != "start":
            print("collect", label, flush=True)
            batches[f"fresh_{label}"] = fresh_batch(actor, 2026101600+idx, args.output/f"fresh-{label}.pt")
    audit["datasets"] = {k:v[1] for k,v in batches.items()}
    for label,path in paths.items():
        actor, payload = load_actor(path)
        before = {k:v.clone() for k,v in actor.state_dict().items()}
        caps = {}
        for name,(inputs,meta) in batches.items():
            print("probe", label, name, flush=True)
            caps[name] = capture(actor,inputs)
        # Per-checkpoint mean calibrated only on fresh control envs 0/1.
        calibration = torch.tensor(batches["fresh_control"][1]["calibration_rows"])
        mean = caps["fresh_control"]["reader"][calibration].mean(0, keepdim=True)
        results = {}
        for name,(inputs,meta) in batches.items():
            cap = caps[name]
            test = torch.tensor(meta["test_rows"]) if "test_rows" in meta else torch.arange(inputs.decisions)
            results[name] = dict(heads=cap["heads"], reference_max_abs=cap["reference_max_abs"],
                interventions=interventions(actor,inputs,cap,mean,test))
        # Same historical outcome target for longitudinal gradient comparison only.
        actor.zero_grad(set_to_none=True)
        with sdpa_kernel(SDPBackend.MATH):
            state = actor.decision_states(None,fixture)
            loss = outcome_loss(actor.outcome_value(state),fx["batch"]["fields"]["outcome_target"])["outcome_loss"]
            loss.backward()
        width, heads = actor.config.width, actor.config.heads
        results["fixture_outcome_gradient"] = dict(loss=float(loss.detach()),
            q_per_head=actor.q_proj.weight.grad.double().reshape(heads,-1).norm(dim=1).tolist(),
            k_per_head=actor.kv_proj.weight.grad[:width].double().reshape(heads,-1).norm(dim=1).tolist())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())
        assert all(torch.equal(v,before[k]) for k,v in actor.state_dict().items())
        audit["results"][label] = results
        (args.output/"mechanism.json").write_text(json.dumps(audit,indent=2,allow_nan=False)+"\n")
    print("complete", args.output/"mechanism.json", flush=True)


if __name__ == "__main__":
    main()
