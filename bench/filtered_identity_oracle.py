"""Frozen-policy identity oracle under identical and marginalized hidden hands.

No fitting. Neutral u195450 generates public prefixes; pool identity is assigned
at the queried node. Uniform determinizations are a controlled information-set
diagnostic, not a posterior over natural persistent-opponent trajectories.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import multiprocessing as mp
from pathlib import Path
import time

import gd
import numpy as np
import torch

from bench.filtered_history_collect import load_players, ROOT as SOURCE
from bench.history_value_tasks import Listener
from bench.qknorm_match_ablation import sha
from eval.history_events import apply_and_observe
from train.history_inference import BatchedHistoryCache
from train.history_model import DecisionInputs, PublicStream, StreamBatch
from train.history_population import weights_digest

ROOT = Path('.work/filtered-identity-oracle-2026-11-15')
SEED = 20261115071
STAGES = ('early', 'middle', 'late')


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def identity_gain(p: np.ndarray, weights: np.ndarray) -> float:
    """Expected multiclass Brier reduction with the exact conditional oracle."""
    center = weights @ p
    return float(np.sum(weights[:, None] * (p - center) ** 2))


def cross_metrics(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> dict:
    """Independent world halves remove upward bias from plug-in heterogeneity."""
    ca, cb = weights @ a, weights @ b
    da, db = a - ca, b - cb
    cross = float(np.sum(weights[:, None] * da * db))
    va, vb = identity_gain(a, weights), identity_gain(b, weights)
    # Predict with A, score in independent B, and vice versa. Labels are
    # integrated exactly using B's probabilities, not sampled as actions.
    blind = 1 + .5 * (ca @ ca + cb @ cb) - 2 * (ca @ cb)
    known = 1 + .5 * np.sum(weights[:, None] * (a*a + b*b)) - 2*np.sum(weights[:, None]*a*b)
    return dict(cross_gain=cross, plugin_gain=identity_gain((a+b)/2, weights),
                split_noise=.5*(va+vb)-cross,
                heldout_blind_brier=float(blind), heldout_identity_brier=float(known),
                heldout_gain=float(blind-known))


def probability_summary(p: np.ndarray, weights: np.ndarray) -> dict:
    center = weights @ p
    choices = p.argmax(1)
    disagreement = np.sum(weights[:, None]*weights[None, :]*(choices[:, None] != choices[None, :]))
    return dict(brier_gain=identity_gain(p, weights),
                blind_brier=float(1-center@center),
                identity_brier=float(1-np.sum(weights[:, None]*p*p)),
                mean_tv_to_mixture=float(.5*np.sum(weights[:, None]*np.abs(p-center))),
                pair_top1_disagreement=float(disagreement))


def encode_world(engine: gd.Engine, state: gd.MatchState) -> dict:
    seat = int(state.to_move)
    legal = engine.legal_actions(state)
    cand = np.asarray([engine.encode_action(a, state, seat) for a in legal], dtype=np.uint8)
    assert len(cand) and np.all(cand[:, 108:119].sum(1) == 1)
    # Public action identity includes physical cards, family and declaration;
    # legal-set indices and private tribute flags are not public labels.
    keys = [row[:146].tobytes() for row in cand]
    return dict(obs=np.asarray(state.observation(seat), dtype=np.uint8), seat=seat,
                cand=cand, keys=keys, kinds=cand[:, 108:119].argmax(1),
                card_count=np.asarray([len(a.cards) for a in legal], dtype=np.float64))


def inputs_for(sb: StreamBatch, worlds: list[dict]) -> DecisionInputs:
    n = len(worlds)
    offsets = np.r_[0, np.cumsum([len(w['cand']) for w in worlds])]
    return DecisionInputs(sb, torch.zeros(n, dtype=torch.long), sb.lengths.expand(n),
        torch.from_numpy(np.stack([w['obs'] for w in worlds])),
        torch.tensor([w['seat'] for w in worlds]),
        torch.from_numpy(np.concatenate([w['cand'] for w in worlds])),
        torch.from_numpy(offsets), one_decision_per_stream=False)


@torch.inference_mode()
def predict_worlds(actor, sb: StreamBatch, encoded: torch.Tensor,
                   worlds: list[dict], batch_size: int = 32) -> list[np.ndarray]:
    result = []
    for start in range(0, len(worlds), batch_size):
        chunk = worlds[start:start+batch_size]
        inputs = inputs_for(sb, chunk)
        lp = actor.candidate_log_probs(inputs, encoded=encoded).double().numpy()
        for lo, hi in zip(inputs.offsets[:-1].tolist(), inputs.offsets[1:].tolist()):
            p = np.exp(lp[lo:hi])
            p /= p.sum()
            if not np.isfinite(p).all():
                raise ValueError('nonfinite probabilities')
            result.append(p)
    return result


def aggregate(p: np.ndarray, ids: np.ndarray, count: int) -> np.ndarray:
    return np.stack([np.bincount(ids, weights=row, minlength=count) for row in p])


def synthetic_pair(p: np.ndarray, cards: np.ndarray) -> np.ndarray:
    """Inference-only positive control; no model or pool is modified.

    Inject a known 25% preference mixture for few versus many cards, on the
    same legal actions. A second epsilon=0 control is exactly the base policy.
    """
    score = 2*(cards-cards.mean())/max(float(np.std(cards)), 1.)
    preference = []
    for sign in (-1, 1):
        logits = sign*score
        r = np.exp(logits-logits.max()); r /= r.sum()
        preference.append(.75*p + .25*r)
    return np.stack(preference)


def stage_of(total: int) -> str:
    return 'early' if total >= 80 else 'middle' if total > 40 else 'late'


@torch.inference_mode()
def generate_root(actor, index: int, seed: int, target_round: int = 3) -> tuple:
    """One public-selected root per independent source, without disagreement screening."""
    stage = STAGES[index % 3]
    engine = gd.Engine(); engine.auto_pass = False
    for attempt in range(16):
        deal_seed = seed + index*1009 + attempt*10000019
        rng = np.random.default_rng(deal_seed ^ 0x1C62A31)
        state = gd.MatchState(); engine.new_match(state, deal_seed)
        stream = PublicStream(); listener = Listener(stream)
        cache = BatchedHistoryCache(actor)
        for step in range(4096):
            if state.phase == gd.Phase.RoundEnd:
                if int(state.round_index) >= target_round:
                    break
                engine.end_round(state)
                if state.winner >= 0:
                    break
                engine.begin_round(state)
                continue
            seat = int(state.to_move)
            remaining = [len(state.hand(i)) for i in range(4)]
            if (state.phase == gd.Phase.Play and int(state.round_index) == target_round
                    and seat % 2 == 1 and stage_of(sum(remaining)) == stage):
                arrays = tuple(x.copy() for x in stream.arrays())
                assert not arrays[0][:, 150:158].any()
                return state, arrays, dict(index=index, stage=stage, attempt=attempt,
                    deal_seed=deal_seed, step=step, round_index=int(state.round_index),
                    actor_seat=seat, observer=(seat+3)%4, remaining=remaining,
                    public_events=len(arrays[0]), top_is_open=bool(state.top_is_open))
            legal = engine.legal_actions(state)
            if len(legal) == 1:
                action = legal[0]
            elif state.phase != gd.Phase.Play:
                action = legal[int(engine.greedy(state))]
            else:
                sb, memory = cache.encode([(0, 0)], [stream])
                world = encode_world(engine, state)
                p = predict_worlds(actor, sb, memory, [world])[0]
                action = legal[int(rng.choice(len(p), p=p))]
            apply_and_observe(engine, state, action, [listener], forced=len(legal)==1)
        else:
            raise RuntimeError('source match step limit exceeded')
    raise RuntimeError(f'no eligible public root after 16 sources: {index}')


_PLAYERS = None


def initialize() -> None:
    global _PLAYERS
    torch.set_num_threads(1)
    _PLAYERS = load_players(SOURCE)


@torch.inference_mode()
def run_root(job: tuple) -> dict:
    root_path, index, seed, worlds_per_half = job
    folder = Path(root_path)/f'root-{index:03d}'
    if (folder/'result.json').exists():
        return json.loads((folder/'result.json').read_text())
    folder.mkdir(exist_ok=True)
    actor, opponents, identity = _PLAYERS
    models = [actor, *opponents.values()]
    before = [weights_digest(a.state_dict()) for a in models]
    started = time.monotonic()
    state, arrays, meta = generate_root(actor, index, seed)
    (folder/'state.bin').write_bytes(state.serialize())
    np.savez_compressed(folder/'public.npz', tokens=arrays[0], rounds=arrays[1], phases=arrays[2])
    observer = meta['observer']; observation = np.asarray(state.observation(observer))
    original = state.serialize()
    hidden_seeds = np.random.default_rng(seed+index*2029+1_000_000_000).choice(
        2**62, size=2*worlds_per_half, replace=False)
    states = [state] + [state.determinize_uniform(observer, int(s)) for s in hidden_seeds]
    assert all(s.hand(observer) == state.hand(observer) for s in states)
    assert all(np.array_equal(s.observation(observer), observation) for s in states)
    alternate = states[1].determinize_uniform(observer, int(hidden_seeds[1]))
    assert [alternate.hand(i) for i in range(4)] == [states[2].hand(i) for i in range(4)]
    engine = gd.Engine(); worlds = [encode_world(engine, s) for s in states]
    sb = StreamBatch.from_arrays([arrays], 'cpu')
    probs = []
    clone_error = serial_error = 0.
    for z, policy in enumerate(opponents.values()):
        encoded = policy.encode_batch(sb)
        p = predict_worlds(policy, sb, encoded, worlds)
        probs.append(p)
        # Actual duplicate forward (not merely relabeling one saved output),
        # and batch-size comparison quantify the numerical null floor.
        if z == 0:
            clone = copy.deepcopy(policy)
            repeat = predict_worlds(clone, sb, clone.encode_batch(sb), worlds[:32])
            serial = predict_worlds(policy, sb, encoded, worlds[:4], batch_size=1)
            clone_error = max(float(np.abs(a-b).max()) for a,b in zip(repeat, p[:32]))
            serial_error = max(float(.5*np.abs(a-b).sum()) for a,b in zip(serial, p[:4]))
            del clone
    weights = np.asarray(identity['weights'])
    count = len(worlds)
    # Canonical union, shared by every policy and both hidden-world halves.
    key_index = {}
    ids = []
    for w in worlds:
        ids.append(np.asarray([key_index.setdefault(k, len(key_index)) for k in w['keys']]))
    nkeys = len(key_index)
    type_prob = np.zeros((len(weights), count, 11))
    synth_type = np.zeros((2, count, 11))
    # Per-quarter sums retain convergence checks without discarding the pairing.
    blocks = 8
    assert worlds_per_half % 4 == 0
    block_size = worlds_per_half//4
    full_blocks = np.zeros((blocks, len(weights), nkeys))
    synth_blocks = np.zeros((blocks, 2, nkeys))
    full_summaries = []; type_summaries = []
    for j, w in enumerate(worlds):
        p = np.stack([row[j] for row in probs])
        local_keys, local_ids = np.unique(ids[j], return_inverse=True)
        local = aggregate(p, local_ids, len(local_keys))
        t = aggregate(p, w['kinds'], 11)
        assert np.allclose(local.sum(1), 1, rtol=0, atol=1e-12)
        assert np.allclose(t.sum(1), 1, rtol=0, atol=1e-12)
        type_prob[:, j] = t
        full_summaries.append(probability_summary(local, weights))
        type_summaries.append(probability_summary(t, weights))
        synthetic = synthetic_pair(p[0], w['card_count'])
        synth_type[:, j] = aggregate(synthetic, w['kinds'], 11)
        if j:
            block = (j-1)//block_size
            full_blocks[block][:, local_keys] += local/block_size
            synth_local = aggregate(synthetic, local_ids, len(local_keys))
            synth_blocks[block][:, local_keys] += synth_local/block_size
        else:
            native = local
            native_keys = local_keys
    results = {}
    for label, summaries, block_prob in (
        ('full_action', full_summaries, full_blocks),
        ('action_type', type_summaries, type_prob[:, 1:].reshape(len(weights), blocks, block_size, 11).mean(2).transpose(1, 0, 2))):
        hidden = {k:float(np.mean([x[k] for x in summaries[1:]])) for k in summaries[0]}
        convergence = {}
        for quarters in (1, 2, 4):
            convergence[str(quarters*block_size)] = cross_metrics(
                block_prob[:quarters].mean(0), block_prob[4:4+quarters].mean(0), weights)
        results[label] = dict(native=summaries[0], conditional_world_mean=hidden,
            marginal=convergence[str(worlds_per_half)], convergence=convergence)
    synth_type_blocks = synth_type[:, 1:].reshape(2, blocks, block_size, 11).mean(2).transpose(1, 0, 2)
    controls = dict(duplicate_forward_max_abs=clone_error, batch_vs_serial_max_tv=serial_error,
        duplicate_identity_gain=identity_gain(np.repeat(native[:1], len(weights), axis=0), weights),
        positive_full=cross_metrics(synth_blocks[:4].mean(0), synth_blocks[4:].mean(0), np.array([.5,.5])),
        positive_type=cross_metrics(synth_type_blocks[:4].mean(0), synth_type_blocks[4:].mean(0), np.array([.5,.5])))
    after = [weights_digest(a.state_dict()) for a in models]
    assert before == after and state.serialize() == original
    assert all(not p.requires_grad and p.grad is None for a in models for p in a.parameters())
    np.savez_compressed(folder/'probabilities.npz', native=native, native_keys=native_keys,
        public_action_keys=np.frombuffer(b''.join(key_index), dtype=np.uint8).reshape(-1, 146),
        full_blocks=full_blocks, type_prob=type_prob, synthetic_full_blocks=synth_blocks,
        synthetic_type_prob=synth_type, hidden_seeds=hidden_seeds, weights=weights)
    result = dict(meta=meta, worlds_per_half=worlds_per_half, models=len(weights),
        forward_decisions=count*len(weights), legal_candidates=[len(w['cand']) for w in worlds],
        metrics=results, controls=controls, frozen_weights_verified=True,
        observer_information_verified=True, seconds=time.monotonic()-started)
    save_json(folder/'result.json', result)
    return result


def run(args) -> None:
    if args.roots % 3 or args.worlds % 4:
        raise ValueError('roots must be divisible by 3, worlds per half by 4')
    args.root.mkdir(parents=True, exist_ok=True)
    _, _, identity = load_players(SOURCE, indices=[])
    protocol = dict(identity=identity, roots=args.roots, worlds_per_half=args.worlds,
        seed=args.seed, stage_counts={s:args.roots//3 for s in STAGES}, rounds_before_root=3,
        root_selection='one fresh neutral u195450 match per root; first odd-seat Play decision in assigned public hand-count stratum; no policy screening',
        stage_bounds='early >=80 total cards; middle 41..79; late <=40',
        source_policy='u195450 all four seats, softmax sampling, declared greedy tribute; full native public prefix',
        identity_assignment='pool type assigned at queried node; fixed original pool weights',
        hidden_sampler='uniform unseen cards respecting own hand, public holdings and remaining counts; no likelihood weighting of past actions',
        outputs=['canonical full public action', '11 action types'],
        primary='weighted identity Brier gain; hidden marginal estimated by cross-product of two independent world means',
        secondary='cross-heldout Brier forecast, full-state policy TV and top1 disagreement; 1/4,1/2,full convergence',
        confidence='95% stratified independent-root bootstrap, 5000 draws; same roots/worlds paired across policies',
        controls='actual cloned frozen-policy forwards, serial/batch parity, null identity labels, 25% synthetic preference mixture',
        scope='frozen CPU inference only; no fitting, optimizer, PPO, main-job interaction or playing-strength claims',
        workers=args.workers, sources={p:sha(p) for p in (__file__, 'train/history_model.py',
            'train/history_inference.py', 'cpp/src/search.cpp', 'bench/filtered_history_collect.py')})
    path = args.root/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('existing protocol differs; use a fresh isolated artifact directory')
    save_json(path, protocol)
    started = time.monotonic()
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
        jobs = [pool.submit(run_root, (str(args.root), i, args.seed, args.worlds)) for i in range(args.roots)]
        for done, future in enumerate(as_completed(jobs), 1):
            result = future.result()
            print(json.dumps(dict(done=done, total=args.roots, index=result['meta']['index'],
                stage=result['meta']['stage'], root_seconds=round(result['seconds'], 2),
                elapsed=round(time.monotonic()-started, 2))), flush=True)
    save_json(args.root/'complete.json', dict(roots=args.roots, seconds=time.monotonic()-started,
        actor_sha_after=sha('.work/danlm-u195450-2026-11-12/u195450.pt'),
        population_sha_after=sha(SOURCE/'population-source.pt')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--roots', type=int, default=192)
    parser.add_argument('--worlds', type=int, default=128)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--workers', type=int, default=4)
    run(parser.parse_args())
