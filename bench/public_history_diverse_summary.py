"""Matched comparisons to the previous close-snapshot public-only diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.public_history_analysis import cluster_weights,confidence


def game_arrays(root):
    records=json.loads((root/"heldout-match-scores.json").read_text())
    keys=sorted((r["seed"],r["team"],r["arm"]) for r in records)
    assert len(keys)==len(set(keys))==512
    seeds=sorted({r["seed"] for r in records})
    assert len(seeds)==128
    grouped=[]
    opponents=[]
    for seed in seeds:
        rows=[r for r in records if r["seed"]==seed]
        assert len(rows)==4
        grouped.append(np.mean([r["scores"] for r in rows],axis=0))
        opponents.append(rows[0]["opponent"])
    return records,keys,seeds,np.asarray(grouped),np.asarray(opponents)


def question_matrix(root,records,seeds):
    with np.load(root/"raw/questions/public_answers.npz",allow_pickle=False) as data:
        counts=data["counts"]
    pi=(counts[:,:,0]+.5)/(32+1.5)
    p0=pi.mean(1)
    freq=counts[:,:,1]/32
    baseline=(p0*p0).sum(1)[:,None]-2*np.einsum("qc,qic->qi",p0,freq)+1
    gains=[]
    for r in records:
        ph=np.einsum("i,qic->qc",np.asarray(r["posterior"]),pi)
        loss=(ph*ph).sum(1)-2*(ph*freq[:,r["identity"]]).sum(1)+1
        gains.append(baseline[:,r["identity"]]-loss)
    gains=np.asarray(gains)
    return np.asarray([gains[[j for j,r in enumerate(records) if r["seed"]==s]].mean(0) for s in seeds])


def diversity(root):
    with np.load(root/"raw/questions/public_answers.npz",allow_pickle=False) as data:
        labels=data["labels"]
    probabilities=np.stack([(labels==c).mean(2) for c in range(3)],axis=-1)
    pairwise=np.asarray([.5*np.abs(probabilities[:,i]-probabilities[:,j]).sum(1)
                         for i in range(4) for j in range(i+1,4)])
    return (np.ptp(labels,axis=1)!=0).mean(1),pairwise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--previous",type=Path,required=True)
    args=p.parse_args()
    old,old_keys,seeds,old_scores,opponents=game_arrays(args.previous)
    new,new_keys,new_seeds,new_scores,new_opponents=game_arrays(args.root)
    assert old_keys==new_keys and seeds==new_seeds and np.array_equal(opponents,new_opponents)
    weights=cluster_weights(opponents,10000,np.random.default_rng(16102680))
    delta=(new_scores[:,0,0]-new_scores[:,1,0])-(old_scores[:,0,0]-old_scores[:,1,0])
    accuracy=np.asarray([np.mean([r["identity"]==r["guessed_identity"] for r in new if r["seed"]==s]) for s in seeds])
    old_accuracy=np.asarray([np.mean([r["identity"]==r["guessed_identity"] for r in old if r["seed"]==s]) for s in seeds])
    old_d,old_pair=diversity(args.previous)
    new_d,new_pair=diversity(args.root)
    qw=np.random.default_rng(16102681).multinomial(256,np.full(256,1/256),size=10000)/256
    q_delta=question_matrix(args.root,new,seeds)-question_matrix(args.previous,old,seeds)
    result=dict(identity_accuracy=confidence(accuracy.mean(),weights@accuracy),
        previous_identity_accuracy=float(old_accuracy.mean()),chance=.25,
        identity_accuracy_change=confidence((accuracy-old_accuracy).mean(),weights@(accuracy-old_accuracy)),
        real_match_brier_history_gain_change=confidence(delta.mean(),weights@delta),
        fixed_question_brier_history_gain_change=confidence(q_delta.mean(),((weights@q_delta)*qw).sum(1)),
        heldout_question_diversity=dict(previous_disagreement=float(old_d.mean()),
            new_disagreement=float(new_d.mean()),
            disagreement_change=confidence((new_d-old_d).mean(),qw@(new_d-old_d)),
            previous_mean_pairwise_tv=float(old_pair.mean()),new_mean_pairwise_tv=float(new_pair.mean()),
            previous_min_pairwise_mean_tv=float(old_pair.mean(1).min()),
            new_min_pairwise_mean_tv=float(new_pair.mean(1).min())),
        limitations="conditional on empirical reference tables; same coarse categories and known four-model roster")
    (args.root/"comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
