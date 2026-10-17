"""Frozen public-history stage readout. Closed-form statistical calibration only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import bench.qknorm_match_ablation as matches
from bench.public_history_diverse import load_actor, load_selected
from bench.public_history_probe import PublicCollector
from bench.qknorm_match_summary import ENDPOINTS
from train.history_model import DecisionInputs, StreamBatch

ROOT = Path('.work/qknorm-history-stages-2026-10-16')
POOL = Path('.work/public-history-diverse-2026-10-16')
STAGES = ('raw_profile', 'memory_mean', 'memory_opponent', 'memory_last',
          'reader', 'reader_delta', 'state', 'state_delta')


def public_profile(tokens: np.ndarray, team: int) -> np.ndarray:
    """Public opponent action counts; one-hot type is the published action type."""
    assert tokens.ndim == 2 and tokens.shape[1] == 186
    assert not tokens[:, 150:158].any(), 'private flags must never enter a probe'
    selected = tokens[tokens[:, :4].argmax(1) % 2 != team]
    # Profiles omit tribute actions (type0 can denote a pass only in PLAY).
    kinds = selected[:, 112:125].argmax(1)
    categories = np.where(kinds == 0, 0, np.where(kinds == 1, 1,
                 np.where(np.isin(kinds, [8, 9, 10]), 2, 3)))
    counts = np.bincount(categories, minlength=4).astype(float)
    return (counts + .5)/(counts.sum()+2)


def stage_features(actor: Any, sample: dict) -> dict:
    original = sample['original']
    n = int((original[1] < 3).sum())
    assert n > 0 and np.all(original[1][:n] < 3) and np.all(original[1][n:] == 3)
    arrays = [original, tuple(x[n:].copy() for x in original)]
    streams = StreamBatch.from_arrays(arrays, 'cpu')
    c = len(sample['cand'])
    inputs = DecisionInputs(streams, torch.arange(2), streams.lengths,
        torch.as_tensor(np.repeat(sample['obs'][None], 2, axis=0)),
        torch.full((2,), sample['seat']), torch.as_tensor(np.tile(sample['cand'], (2, 1))),
        torch.arange(3)*c, one_decision_per_stream=True)
    encoded = actor.encode_batch(streams)
    query = actor.private(inputs.obs.float()) + actor.seat(inputs.seat)
    attended = actor._attend_encoded(query, encoded, inputs.match_index, inputs.prefix, True)
    hidden = actor.attention_norm(query+attended)
    state = actor.output_norm(hidden+actor.feed_forward(hidden))
    native = actor.decision_states(encoded, inputs)
    assert torch.equal(native, state), 'manual feature tap changed native state'
    table = actor._log_prob_table(inputs, encoded, c)
    error = float((table[0]-torch.tensor(sample['cached_logp'])).abs().max())
    assert error < 1e-4, ('cache parity', error)
    p = table.double().exp(); p /= p.sum(1, keepdim=True)
    old = encoded[0, 1:n+1]
    opponent = torch.tensor(original[0][:n, :4].argmax(1) % 2 != sample['seat'] % 2)
    assert opponent.any()
    play = original[2][:n] == 3
    features = dict(context=torch.cat((query[0], attended[1])),
        memory_mean=old.mean(0), memory_opponent=old[opponent].mean(0),
        memory_last=old[-1], reader=attended[0], reader_delta=attended[0]-attended[1],
        state=state[0], state_delta=state[0]-state[1])
    out = {k:v.cpu().numpy().astype(np.float64) for k,v in features.items()}
    lengths = np.bincount(original[1][:n], minlength=3).astype(float)
    out['context'] = np.concatenate((out['context'], lengths, [len(original[0])-n]))
    out['raw_profile'] = public_profile(original[0][:n][play], sample['seat'] % 2)
    out.update(full_round_tv=float((p[0]-p[1]).abs().sum()/2), cache_error=error,
               past=n, current=len(original[0])-n)
    assert all(np.isfinite(v).all() for v in out.values())
    return out


class StageCollector(PublicCollector):
    last = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.samples: list[dict] = []
        self.seen: set[int] = set()
        StageCollector.last = self

    def _infer_group(self, group: Any, fields: Any, r: Any, stats: Any, mark: Any) -> Any:
        if group.identity == 0:
            picked = [(j, i) for j,i in enumerate(group.rows)
                      if r.match_id[i] == 0 and r.round_index[i] == 3 and r.counts[i] > 1
                      and int(r.env_id[i]) not in self.seen]
            if picked:
                streams, encoded = self._encode_streams(group, self.actor, fields[1], r.prefix)
                inputs = DecisionInputs(streams, fields[0], fields[1], fields[2], fields[3],
                    fields[4], fields[5], fields[6], one_decision_per_stream=group.one_decision_per_stream)
                table = self.actor._log_prob_table(inputs, encoded, group.max_candidates)
                for j,i in picked:
                    e = int(r.env_id[i]); self.seen.add(e)
                    n = int(r.prefix[i]); a,b = map(int, r.offsets[i:i+2])
                    s = group.streams[int(fields[0][j])]
                    self.samples.append(dict(env=e, seat=int(r.seat[i]),
                        original=tuple(x[:n].copy() for x in s.arrays()),
                        obs=r.obs[i].copy(), cand=r.cand[a:b].copy(),
                        cached_logp=table[j, :b-a].cpu().numpy().copy()))
        return super()._infer_group(group, fields, r, stats, mark)


def collect(wave: int) -> None:
    torch.set_num_threads(1)
    folder = ROOT/f'wave-{wave:02d}'; folder.mkdir(exist_ok=False)
    opponents, opponent_meta = load_selected(POOL)
    table = np.random.default_rng(16102671).integers(1, 2**63-1, size=(16,16))
    seeds = np.repeat(table[wave], 2).tolist()
    original = matches.EvalCollector; matches.EvalCollector = StageCollector
    metadata = {}
    try:
        for arm in ENDPOINTS:
            actor, identity = load_actor(Path(f'.work/qknorm-short-ppo-2026-10-16/remote/runs/{arm}/endpoint.pt'))
            assert identity['checkpoint_sha'] == ENDPOINTS[arm]
            metadata[arm] = identity
            result = matches.run_wave(actor, opponents, seeds, 'cpu', 'full', 20000)
            reference = json.loads((POOL/'raw'/f'wave-{wave:02d}'/f'{arm}-matches.json').read_text())
            assert result['records'] == reference['records'], 'FULL replay changed'
            samples = StageCollector.last.samples
            assert {s['env'] for s in samples} == {r['env'] for r in result['records'] if r['rounds']>3}
            rows = []
            with torch.inference_mode():
                for s in samples:
                    features = stage_features(actor, s)
                    e = s['env']
                    features.update(wave=wave, env=e, seed=seeds[e], team=e%2, opponent=(e//2)%4)
                    rows.append(features)
            assert len(rows) == 32
            rows.sort(key=lambda row: row['env'])
            np.savez_compressed(folder/f'{arm}.npz', **{key:np.asarray([r[key] for r in rows]) for key in rows[0]})
            print(json.dumps(dict(wave=wave, arm=arm, rows=len(rows), full_replay=True)), flush=True)
    finally:
        matches.EvalCollector = original
    (folder/'complete.json').write_text(json.dumps(dict(endpoints=metadata, opponents=opponent_meta,
        runner_sha=matches.sha(__file__), model_sha=matches.sha(__import__('train.history_model', fromlist=['x']).__file__),
        full_replay=True, torch=torch.__version__))+'\n')


def ridge_predict(x: np.ndarray, y: np.ndarray, test: np.ndarray, penalty: float = 1.) -> np.ndarray:
    """One fixed closed-form ridge readout, calibration-only scaling, no optimizer."""
    mean = x.mean(0); scale = x.std(0); scale = np.maximum(scale, .01)
    z = (x-mean)/scale; t = (test-mean)/scale
    center = y.mean(0)
    # Dual solve is smaller than feature-space solve for concatenated stages.
    coef = z.T @ np.linalg.solve(z@z.T + len(z)*penalty*np.eye(len(z)), y-center)
    pred = center+t@coef
    pred = np.maximum(pred, 1e-4)
    return pred/pred.sum(1, keepdims=True)


def seed_permutation(seeds: np.ndarray, seed: int) -> np.ndarray:
    """Permute complete seed blocks, aligning team swaps, across different seeds."""
    unique = np.unique(seeds); rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(unique)
        if np.all(perm != unique): break
    mapping = dict(zip(unique, perm))
    positions = {int(s):np.flatnonzero(seeds == s) for s in unique}
    out = np.empty(len(seeds), np.int64)
    for s in unique:
        out[positions[int(s)]] = positions[int(mapping[s])]
    return out


def brier_probabilities(p: np.ndarray, pi: np.ndarray, freq: np.ndarray, identity: np.ndarray) -> np.ndarray:
    """Same-card questions; identity is used only to select the public scoring labels."""
    predicted = np.einsum('mi,qic->mqc', p, pi)
    target = freq[:, identity].transpose(1,0,2)
    return np.sum(predicted**2, axis=-1)-2*np.sum(predicted*target, axis=-1)+1


def analyze() -> None:
    result = {}; all_predictions = {}
    with np.load(POOL/'raw'/'questions'/'public_answers.npz') as raw:
        pi = (raw['counts'][:,:,0]+.5)/33.5
        freq = raw['counts'][:,:,1]/32
    for arm in ENDPOINTS:
        arrays = []
        for wave in range(16):
            assert json.loads((ROOT/f'wave-{wave:02d}'/'complete.json').read_text())['full_replay']
            with np.load(ROOT/f'wave-{wave:02d}'/f'{arm}.npz') as data:
                arrays.append({key:data[key] for key in data.files})
        d = {key:np.concatenate([a[key] for a in arrays]) for key in arrays[0]}
        train = d['wave']<8; test = ~train
        y = np.eye(4)[d['opponent'][train]]
        cp = seed_permutation(d['seed'][train], 16102721)
        tp = seed_permutation(d['seed'][test], 16102722)
        context = d['context']; predictions = {'context':ridge_predict(context[train], y, context[test])}
        for stage in STAGES:
            feature = d[stage]
            predictions[stage] = ridge_predict(np.concatenate((context[train], feature[train]), 1), y,
                                               np.concatenate((context[test], feature[test]), 1))
            predictions[stage+'_shuffle'] = ridge_predict(np.concatenate((context[train], feature[train][cp]), 1), y,
                                               np.concatenate((context[test], feature[test][tp]), 1))
        # Identity labels are NOT features for held-out predictions.
        ids = d['opponent'][test]; seeds = d['seed'][test]
        unique = np.unique(seeds); assert len(unique)==128
        groups = np.array([np.flatnonzero(seeds==s) for s in unique])
        opponents = ids[groups[:,0]]
        from bench.public_history_analysis import cluster_weights
        weights = cluster_weights(opponents, 10000, np.random.default_rng(16102723))
        qw = np.random.default_rng(16102724).multinomial(256, np.full(256,1/256), size=10000)/256
        question_scores = {key:brier_probabilities(p, pi, freq, ids) for key,p in predictions.items()}
        stages = {}
        for key,p in predictions.items():
            identity_brier = np.sum((p-np.eye(4)[ids])**2, axis=1)
            gain = question_scores['context']-question_scores[key]
            matrix = gain[groups].mean(1)
            draws = np.sum((weights@matrix)*qw, axis=1)
            stats = dict(identity_accuracy=float(np.mean(p.argmax(1)==ids)),identity_brier=float(identity_brier.mean()),
                same_card_brier=float(question_scores[key].mean()),
                gain_over_context=dict(estimate=float(matrix.mean()),ci95=np.quantile(draws,[.025,.975]).tolist()))
            if not key.endswith('_shuffle') and key!='context':
                contrast = (question_scores[key+'_shuffle']-question_scores[key])[groups].mean(1)
                draws = np.sum((weights@contrast)*qw, axis=1)
                stats['gain_over_shuffle'] = dict(estimate=float(contrast.mean()),ci95=np.quantile(draws,[.025,.975]).tolist())
            stages[key] = stats
        result[arm] = dict(calibration=256, heldout=256, test_seed_clusters=128,
            current_query_dim=context.shape[1], max_cache_error=float(d['cache_error'].max()),
            stages=stages, full_round_tv=float(d['full_round_tv'][test].mean()))
        all_predictions[arm] = predictions
        np.savez_compressed(ROOT/f'{arm}-heldout-predictions.npz', **predictions,
                            ids=ids, seeds=seeds, wave=d['wave'][test], team=d['team'][test])
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wave',type=int);parser.add_argument('--analyze',action='store_true')
    args=parser.parse_args()
    if args.analyze: analyze()
    else: collect(args.wave)
