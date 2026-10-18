"""Exploratory direct residual check after the fixed snapshot-mixture primary test.

Fixed pseudo-count 32, no tuning: transfer first-three-round excess behavior
over the public-context prediction directly, without requiring identity inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.public_history_analysis import (Tables, cluster_weights, confidence,
    iter_matches, scores, softmax)


def adjust(p, residual, count):
    q = np.maximum(p+residual/(32+count), 1e-8)
    return q/q.sum()


def baseline(tables, arm, row, current):
    pi, prior = tables.probabilities(arm, row)
    return softmax(np.log(prior)+tables.history_logits(arm, current))@pi


def analyze(root):
    tables = Tables()
    for match in iter_matches(root, range(8)):
        tables.add(match)
    tables.finish()
    prefixes = []
    for match in iter_matches(root, range(8,16)):
        residual = np.zeros((2,4))
        count = np.zeros(2)
        current = np.zeros((2,4))
        old_round = -1
        for row in match["rows"]:
            k, lead, y = row[:3]
            if k >= 3:
                break
            if k != old_round:
                current[:] = 0
                old_round = k
            p = baseline(tables, match["arm"], row, current)
            residual[lead] += np.eye(4)[y]-p
            count[lead] += 1
            current[lead,y] += 1
        prefixes.append(dict(arm=match["arm"], seed=match["game"]["seed"],
                             residual=residual, count=count))
    rng = np.random.default_rng(16102679)
    shuffle = np.empty(len(prefixes), np.int32)
    for arm in (0,1):
        indices = np.asarray([j for j,m in enumerate(prefixes) if m["arm"] == arm])
        while True:
            permutation = rng.permutation(indices)
            if all(prefixes[a]["seed"] != prefixes[b]["seed"] for a,b in zip(indices,permutation)):
                break
        shuffle[indices] = permutation
    records = []
    for j,match in enumerate(iter_matches(root,range(8,16))):
        current = np.zeros((2,4))
        old_round = -1
        accum = np.zeros((3,2))
        count = 0
        for row in match["rows"]:
            k,lead,y = row[:3]
            if k != old_round:
                current[:] = 0
                old_round = k
            if k >= 3:
                p0 = baseline(tables,match["arm"],row,current)
                h,s = prefixes[j],prefixes[shuffle[j]]
                predictions = (p0,adjust(p0,h["residual"][lead],h["count"][lead]),
                               adjust(p0,s["residual"][lead],s["count"][lead]))
                for i,p in enumerate(predictions):
                    accum[i] += scores(p,y)
                count += 1
            current[lead,y] += 1
        records.append(dict(seed=match["game"]["seed"],opponent=match["game"]["opponent"],
                             scores=accum/count))
    seeds = sorted({r["seed"] for r in records})
    opponents = np.asarray([next(r["opponent"] for r in records if r["seed"] == s) for s in seeds])
    grouped = np.asarray([np.mean([r["scores"] for r in records if r["seed"] == s],axis=0) for s in seeds])
    weights = cluster_weights(opponents,10000,np.random.default_rng(16102680))
    gain = grouped[:,0]-grouped[:,1]
    vs_shuffle = grouped[:,2]-grouped[:,1]
    with np.load(root/"questions/public_answers.npz",allow_pickle=False) as data:
        counts = data["counts"]
    pi = (counts[:,:,0]+.5)/(32+1.5)
    p0 = pi.mean(1)
    freq = counts[:,:,1]/32
    base = (p0*p0).sum(1)[:,None]-2*np.einsum("qc,qic->qi",p0,freq)+1
    matrices = [np.zeros((512,256)),np.zeros((512,256))]
    held_labels = json.loads((root.parent/"heldout-match-scores.json").read_text())
    for j,match in enumerate(prefixes):
        identity = held_labels[j]["identity"]
        for dest,history in zip(matrices,(prefixes[j],prefixes[shuffle[j]])):
            residual = history["residual"][1,1:]
            prediction = np.maximum(p0+residual/(32+history["count"][1]),1e-8)
            prediction /= prediction.sum(1,keepdims=True)
            brier = (prediction*prediction).sum(1)-2*(prediction*freq[:,identity]).sum(1)+1
            dest[j] = base[:,identity]-brier
    groups = [np.asarray([j for j,p in enumerate(prefixes) if p["seed"] == s]) for s in seeds]
    matrix,null_matrix = [np.asarray([m[g].mean(0) for g in groups]) for m in matrices]
    qw = np.random.default_rng(16102681).multinomial(256,np.full(256,1/256),size=10000)/256
    return dict(exploratory=True,reason="checks a direct behavioral channel beyond snapshot identity",
        pseudo_count=32,hyperparameter_search=False,new_games=0,
        real_match_brier_improvement=confidence(gain[:,0].mean(),weights@gain[:,0]),
        real_match_logloss_improvement=confidence(gain[:,1].mean(),weights@gain[:,1]),
        real_match_brier_over_shuffle=confidence(vs_shuffle[:,0].mean(),weights@vs_shuffle[:,0]),
        fixed_question_brier_improvement=confidence(matrix.mean(),((weights@matrix)*qw).sum(1)),
        fixed_question_brier_over_shuffle=confidence((matrix-null_matrix).mean(),
                                                     ((weights@(matrix-null_matrix))*qw).sum(1)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    result = analyze(args.root)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
