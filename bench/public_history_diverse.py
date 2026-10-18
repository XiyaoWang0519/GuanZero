"""Independent old-Transformer selection and frozen public-history replication.

Keeps previous measured scripts unchanged. 47241..47244 are compatibility
labels for latent evaluation slots here, NOT the previous native snapshots.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time

import gd
import numpy as np
import torch

import bench.qknorm_match_ablation as matches
from bench.public_history_probe import PublicCollector
from train.history_model import load_history_checkpoint, PLAY_PHASE
from train.history_population import snapshot_weights, weights_digest
from train.history_rollout import MatchEventStore, SequenceRolloutBuffer


def load_actor(path):
    actor, critic, payload = load_history_checkpoint(path, "cpu")
    actor.eval().requires_grad_(False)
    del critic
    return actor, dict(lineage=payload["lineage"], update=payload["progress"]["updates"],
        config=payload["model_config"], checkpoint_sha=matches.sha(path),
        acting_sha=weights_digest(snapshot_weights(actor)), path=str(path))


def question_bank(actors, questions, draws, deal_seed, policy_seed):
    """Only actual public answers leave the actor side. All actors get identical deals."""
    assert draws % 2 == 0
    identities = sorted(actors)
    labels = np.zeros((questions,len(actors),draws),np.int8)
    counts = np.zeros((questions,len(actors),2,3),np.int32)
    seeds = np.random.default_rng(deal_seed).integers(1,2**63-1,size=questions)
    for q,seed in enumerate(seeds):
        n = len(actors)*draws
        env = gd.VecEnv(num_envs=n,num_threads=1,seed=71,
                        log_public_actions=True,log_env_limit=n)
        env.reset(match_seeds=[int(seed)]*n)
        sampling = np.random.default_rng([policy_seed,q]).integers(1,2**63-1,size=draws)
        collector = PublicCollector(env,actors[identities[0]],MatchEventStore(),
            SequenceRolloutBuffer(),device="cpu",kv_cache=True,
            seat_policy=lambda e,m:[identities[e//draws]]*4,
            resolve_policy=actors.__getitem__,seeds=np.tile(sampling,len(actors)).tolist())
        collector.started = True
        with torch.inference_mode():
            collector.collect(1)
        events = env.drain_public_actions()
        assert len(events) == n and len({int(e.seat) for e in events}) == 1
        for event in events:
            assert int(event.round_index)==0 and int(event.phase)==PLAY_PHASE and event.step==0
            encoded = np.asarray(event.encoded_action)
            assert not encoded[146:].any()
            kind = int(encoded[108:121].argmax())
            assert kind != 0
            label = 0 if kind==1 else 1 if kind in (8,9,10) else 2
            identity,draw = divmod(int(event.env_id),draws)
            labels[q,identity,draw] = label
            counts[q,identity,draw//(draws//2),label] += 1
        if q % 32 == 31:
            print(json.dumps(dict(questions=q+1,of=questions)),flush=True)
    return labels,counts


def select_quartet(labels):
    """Maximize minimum pairwise categorical TV; tie break by mean pairwise TV."""
    n = labels.shape[1]
    probabilities = np.stack([(labels==c).mean(2) for c in range(3)],axis=-1)
    distance = np.zeros((n,n))
    for i,j in itertools.combinations(range(n),2):
        distance[i,j] = distance[j,i] = .5*np.abs(probabilities[:,i]-probabilities[:,j]).sum(1).mean()
    def score(indices):
        pairs = [distance[i,j] for i,j in itertools.combinations(indices,2)]
        return min(pairs),float(np.mean(pairs))
    choices = list(itertools.combinations(range(n),4))
    best = max(choices,key=score)
    return best,distance,score(best)


def screening(args):
    candidates = json.loads((args.output/"candidates.json").read_text())
    actors,records = {},[]
    for i,candidate in enumerate(candidates):
        actors[i+1],record = load_actor(candidate["path"])
        assert record["checkpoint_sha"] == candidate["sha256"]
        records.append(record)
    labels,counts = question_bank(actors,64,32,16102691,16102692)
    selected,distance,score = select_quartet(labels)
    np.savez_compressed(args.output/"screen-public.npz",labels=labels,counts=counts,pairwise_tv=distance)
    result = dict(selection_questions=64,selection_seed=16102691,
        selection_method="maximin categorical TV, mean-TV tie break; no prediction outcomes used",
        selected_indices=list(selected),min_pairwise_tv=score[0],mean_pairwise_tv=score[1],
        candidates=records,
        selected={47241+j:dict(records[i],evaluation_slot=j,
                    label_note="evaluation-only alias; not native snapshot identity")
                  for j,i in enumerate(selected)})
    (args.output/"selected.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!='candidates'},indent=2),flush=True)


def load_selected(root):
    manifest = json.loads((root/"selected.json").read_text())
    actors = {}
    for label,record in manifest["selected"].items():
        actor,actual = load_actor(record["path"])
        assert actual["checkpoint_sha"]==record["checkpoint_sha"]
        assert actual["acting_sha"]==record["acting_sha"]
        actors[int(label)] = actor
    return actors,manifest["selected"]


def collect(args):
    opponents,metadata = load_selected(args.output)
    raw = args.output/"raw"
    if args.questions:
        folder = raw/"questions"
        folder.mkdir(parents=True,exist_ok=False)
        labels,counts = question_bank(opponents,256,64,16102677,16102678)
        np.savez_compressed(folder/"public_answers.npz",labels=labels,counts=counts)
        (folder/"complete.json").write_text(json.dumps(dict(complete=True,questions=256,
            public_answers=65536,opponents=metadata,selection_seed_independent=True))+"\n")
        return
    folder = raw/f"wave-{args.wave:02d}"
    folder.mkdir(parents=True,exist_ok=False)
    seeds = np.random.default_rng(16102671).integers(1,2**63-1,size=(16,16))
    schedule = np.repeat(seeds[args.wave],2).tolist()
    original = matches.EvalCollector
    matches.EvalCollector = PublicCollector
    try:
        for arm in ("control","norm16"):
            path = args.base/f"qknorm-short-ppo-2026-10-16/remote/runs/{arm}/endpoint.pt"
            actor,record = load_actor(path)
            result = matches.run_wave(actor,opponents,schedule,"cpu","full",20000)
            PublicCollector.last.tap.save(folder/f"{arm}-public.npz")
            (folder/f"{arm}-matches.json").write_text(json.dumps(result,indent=2)+"\n")
            print(json.dumps(dict(arm=arm,matches=32,seconds=result["seconds"],
                public_events=len(PublicCollector.last.tap.meta),endpoint_sha=record["checkpoint_sha"])),flush=True)
    finally:
        matches.EvalCollector = original
    (folder/"complete.json").write_text(json.dumps(dict(complete=True,opponents=metadata,
        source_sha=matches.sha(__file__),model_sha=matches.sha(__import__("train.history_model",fromlist=["x"]).__file__)))+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--base",type=Path,default=Path(".work"))
    p.add_argument("--screen",action="store_true")
    p.add_argument("--questions",action="store_true")
    p.add_argument("--wave",type=int)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.screen:
        screening(args)
    else:
        assert args.questions or args.wave is not None and 0<=args.wave<16
        collect(args)


if __name__=="__main__":
    main()
