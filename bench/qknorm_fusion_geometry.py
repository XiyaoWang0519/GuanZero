"""Same-state, equal-length history-swap geometry at frozen LayerNorm."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from bench.qknorm_fusion_isolation import ROOT, normalization_parts
from bench.qknorm_history_swap import ROOT as SWAP_ROOT, donor_bank, splice
from bench.public_history_diverse import load_actor
from bench.qknorm_match_summary import ENDPOINTS
from train.history_model import StreamBatch, DecisionInputs


def geometry(x: np.ndarray, delta: np.ndarray, eps: float) -> dict[str,float]:
    x=np.asarray(x,float);delta=np.asarray(delta,float)
    center=x-x.mean();norm=np.linalg.norm(center)
    dmean=np.full_like(delta,delta.mean());dc=delta-dmean
    radial=center*np.dot(dc,center)/(norm*norm)
    tangent=dc-radial;energy=np.dot(delta,delta)
    assert energy>0 and norm>0
    p=normalization_parts(np.stack((x,x+delta)),eps)
    finite=(p['unit'][1]-p['unit'][0])*p['scale'][0,0]
    return dict(mean_fraction=float(np.dot(dmean,dmean)/energy),
        radial_fraction=float(np.dot(radial,radial)/energy),
        tangent_fraction=float(np.dot(tangent,tangent)/energy),
        finite_rescaled_energy_fraction=float(np.dot(finite,finite)/energy),
        linearized_relative_error=float(np.linalg.norm(finite-tangent)/np.sqrt(energy)),
        scale=float(p['scale'][0,0]),reader_delta_rms=float(np.sqrt(energy/len(delta))),
        delta_over_centered_input=float(np.sqrt(energy)/norm))


def measure(actor: Any, sample: dict, donor: tuple) -> dict:
    original=sample['original'];swapped=splice(original,donor,sample['round'],True)
    streams=StreamBatch.from_arrays([original,swapped],'cpu');c=len(sample['cand'])
    inputs=DecisionInputs(streams,torch.arange(2),streams.lengths,
        torch.as_tensor(np.repeat(sample['obs'][None],2,0)),torch.full((2,),sample['seat']),
        torch.as_tensor(np.tile(sample['cand'],(2,1))),torch.arange(3)*c,one_decision_per_stream=True)
    encoded=actor.encode_batch(streams)
    query=actor.private(inputs.obs.float())+actor.seat(inputs.seat)
    assert torch.equal(query[0],query[1])
    reader=actor._attend_encoded(query,encoded,inputs.match_index,inputs.prefix,True)
    mixed=query+reader
    native=actor.attention_norm(mixed)
    table=actor._log_prob_table(inputs,encoded,c).double()
    error=float((table[0]-torch.tensor(sample['cached_logp'])).abs().max());assert error<1e-4
    probabilities=table.exp();probabilities/=probabilities.sum(1,keepdim=True)
    result=geometry(mixed[0].double().numpy(),(reader[1].double()-reader[0].double()).numpy(),actor.attention_norm.eps)
    # Double reconstruction uses unrounded mixed coordinates for stable geometry.
    double_mixed=query.double()+reader.double()
    g=actor.attention_norm.weight.detach().double().numpy();b=actor.attention_norm.bias.detach().double().numpy()
    precise=normalization_parts(double_mixed.numpy(),actor.attention_norm.eps)['unit']*g+b
    result.update(native_ln_max_error=float(np.abs(native.numpy()-precise).max()),cache_error=error,
        policy_tv=float((probabilities[0]-probabilities[1]).abs().sum()/2),
        policy_flip=int(probabilities[0].argmax()!=probabilities[1].argmax()))
    assert result['native_ln_max_error']<1e-5
    return result


def run(wave: int) -> None:
    torch.set_num_threads(1);folder=ROOT/f'wave-{wave:02d}';folder.mkdir(exist_ok=False)
    reference=json.loads((SWAP_ROOT/f'wave-{wave:02d}'/'rows.json').read_text());out=[]
    for arm in ENDPOINTS:
        actor,meta=load_actor(Path(f'.work/qknorm-short-ppo-2026-10-16/remote/runs/{arm}/endpoint.pt'))
        assert meta['checkpoint_sha']==ENDPOINTS[arm]
        samples=torch.load(SWAP_ROOT/f'wave-{wave:02d}'/f'{arm}-samples.pt',weights_only=False)
        bank=donor_bank(arm)
        lookup={(d['source'],d['wave'],d['env'],d['round'],d['window_start']):d for d in bank}
        with torch.inference_mode():
            for row in (r for r in reference if r['arm']==arm):
                d=lookup[(row['donor_source'],row['donor_wave'],row['donor_env'],row['round'],row['donor_window_start'])]
                donor=tuple(x.copy() for x in d['arrays'])
                if row['donor_seat_rotation']:donor[0][:,:4]=np.roll(donor[0][:,:4],1,1)
                measured=measure(actor,samples[row['index']],donor)
                assert abs(measured['policy_tv']-row['fixed_metadata']['tv'])<2e-5
                assert measured['policy_flip']==row['fixed_metadata']['flip']
                measured.update(arm=arm,wave=wave,env=row['env'],seed=row['seed'],round=row['round'])
                out.append(measured)
    (folder/'geometry.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(wave=wave,rows=len(out),all_swap_actions_reproduced=True)),flush=True)


def summarize() -> None:
    rows=[]
    for w in range(8,12):rows+=json.loads((ROOT/f'wave-{w:02d}'/'geometry.json').read_text())
    result={}
    for arm in ENDPOINTS:
        rs=[r for r in rows if r['arm']==arm];stats={}
        for k in ('mean_fraction','radial_fraction','tangent_fraction','finite_rescaled_energy_fraction',
                  'linearized_relative_error','scale','reader_delta_rms','delta_over_centered_input'):
            values=np.array([r[k] for r in rs]);stats[k]=dict(mean=float(values.mean()),median=float(np.median(values)),
                                                          p05=float(np.quantile(values,.05)),p95=float(np.quantile(values,.95)))
        stats.update(rows=len(rs),policy_flips=sum(r['policy_flip'] for r in rs),
            max_cache_error=max(r['cache_error'] for r in rs),max_native_ln_error=max(r['native_ln_max_error'] for r in rs))
        result[arm]=stats
    (ROOT/'geometry-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wave',type=int);p.add_argument('--summary',action='store_true');args=p.parse_args()
    if args.summary:summarize()
    else:run(args.wave)
