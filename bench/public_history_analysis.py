"""Public-only count-table predictions, held-out matches, and identical-deal controls.

No gradient fitting. Snapshot labels partition calibration counts and score tests;
they never enter a held-out predictor's public features or history likelihood.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def category(action_type):
    return 0 if action_type == 0 else 1 if action_type == 1 else 2 if action_type in (8, 9, 10) else 3


def bucket(cards):
    return int(np.searchsorted([1, 2, 5, 10, 18], cards, side="left"))


def public_rows(meta, actions, results, team):
    """Reconstruct pre-action context from public events only, including forced passes.

    No forced flag is observed. The context uses counts BEFORE the target action.
    Levels for round k come only from round k-1's public result.
    """
    previous = {int(r[1])+1: r for r in results}
    old_round = -1
    for event, encoded in zip(meta, actions):
        _, k, step, seat, left, phase = map(int, event)
        if k != old_round:
            old_round = k
            remaining = np.full(4, 27)
            seen = np.zeros(54, np.int32)
            top_type = top_key = top_size = passes = 0
            holder = -1
            play_started = False
            if k == 0:
                levels, owner = [0, 0], -1
            else:
                prior = previous[k]
                owner, levels = int(prior[3]), list(map(int, prior[4:6]))
            level = levels[owner] if owner >= 0 else 0
        action_type = int(encoded[108:121].argmax())
        if phase != 3:  # gd.Phase.Play is validated by the collector metadata/tests below
            # Do not inspect exchange-only hand structure. Public exchanges finish
            # with 27 cards per hand; exact exchange cards stay in the raw public log.
            continue
        if not play_started:
            remaining[:] = 27
            play_started = True
        lead = int(top_type == 0)
        coarse = (lead, top_type, bucket(remaining[seat]), bucket(remaining[(seat+2)%4]),
                  bucket(min(remaining[(seat+1)%4], remaining[(seat+3)%4])))
        detailed = coarse + (top_key//3, top_size, (holder-seat)%4 if holder >= 0 else 4,
            min(step//16, 5), level//3, levels[seat%2]//3, levels[1-seat%2]//3,
            0 if k < 3 else 1 if k < 6 else 2 if k < 10 else 3)
        fine = detailed + (top_key, int(remaining[seat]), int(remaining[(seat+2)%4]),
            int(seen[52:].sum()), int(seen[level*4:level*4+4].sum()), level)
        if seat % 2 != team:
            yield k, lead, category(action_type), coarse, detailed, fine
        remaining[seat] = left
        if action_type == 0:
            assert not lead, "public trick reconstruction marked a pass as a lead"
            passes += 1
            if passes >= sum(remaining[s] > 0 for s in range(4) if s != holder):
                holder = -1
                top_type = top_key = top_size = passes = 0
        else:
            seen += encoded[:54].astype(np.int32)+encoded[54:108].astype(np.int32)
            top_type = action_type
            top_key = int(encoded[121:136].argmax())
            top_size = int(encoded[:108].sum())
            holder, passes = seat, 0


def iter_matches(root, waves):
    for wave in waves:
        folder = root/f"wave-{wave:02d}"
        assert json.loads((folder/"complete.json").read_text())["complete"]
        for arm_index, arm in enumerate(("control", "norm16")):
            record = json.loads((folder/f"{arm}-matches.json").read_text())["records"]
            with np.load(folder/f"{arm}-public.npz", allow_pickle=False) as data:
                assert set(data.files) == {"public_meta", "public_actions", "public_round_results"}
                meta, actions, results = (data[name] for name in
                    ("public_meta", "public_actions", "public_round_results"))
                for game in record:
                    e = game["env"]
                    keep = meta[:, 0] == e
                    rows = list(public_rows(meta[keep], actions[keep], results[results[:, 0] == e], game["team"]))
                    prefix = np.zeros((2, 4), np.int32)
                    for k, lead, y, *_ in rows:
                        if k < 3:
                            prefix[lead, y] += 1
                    assert prefix.sum() and any(row[0] >= 3 for row in rows)
                    yield dict(arm=arm_index, wave=wave, game=game, prefix=prefix, rows=rows)


def softmax(logits):
    values = np.exp(logits-logits.max())
    return values/values.sum()


class Tables:
    """Fixed hierarchical count smoothing, 64 pseudo-events at each context level."""
    def __init__(self):
        self.tables = [defaultdict(lambda: np.zeros((4, 4))) for _ in range(4)]
        self.prefix_counts = np.zeros((2, 4, 2, 4))

    def add(self, match):
        arm = match["arm"]
        identity = match["game"]["opponent"]-47241
        self.prefix_counts[arm, identity] += match["prefix"]
        for k, lead, y, coarse, detailed, fine in match["rows"]:
            for table, key in zip(self.tables, ((arm, lead), (arm, *coarse),
                                                  (arm, *detailed), (arm, *fine))):
                table[key][identity, y] += 1

    def finish(self):
        pooled = self.prefix_counts.sum(1, keepdims=True)+.5
        pooled /= pooled.sum(-1, keepdims=True)
        self.log_profile = np.log((self.prefix_counts+64*pooled)/
                                 (self.prefix_counts.sum(-1, keepdims=True)+64))

    def history_logits(self, arm, counts):
        # Fixed tempering by 8 avoids pretending actions within a hand are independent.
        return np.einsum("lc,ilc->i", counts, self.log_profile[arm])/8.

    def probabilities(self, arm, row):
        k, lead, y, coarse, detailed, fine = row
        base = self.tables[0][(arm, lead)]
        pooled = (base.sum(0)+.5)/(base.sum()+2)
        p = (base+64*pooled)/(base.sum(1, keepdims=True)+64)
        prior_counts = self.tables[1].get((arm, *coarse), np.zeros((4, 4))).sum(1)
        identity_prior = (prior_counts+16)/(prior_counts.sum()+64)
        for table, key in zip(self.tables[1:], ((arm, *coarse), (arm, *detailed), (arm, *fine))):
            counts = table.get(key)
            if counts is not None:
                p = (counts+64*p)/(counts.sum(1, keepdims=True)+64)
        return p, identity_prior


def scores(p, y):
    return np.asarray([(p*p).sum()-2*p[y]+1, -np.log(max(p[y], 1e-12))])


def confidence(point, draws):
    return dict(estimate=float(point), ci95=np.quantile(draws, [.025, .975]).tolist())


def cluster_weights(opponents, draws, rng):
    result = np.zeros((draws, len(opponents)))
    for identity in sorted(set(opponents)):
        indices = np.flatnonzero(opponents == identity)
        result[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1/len(indices)), size=draws)/len(opponents)
    return result


def analyze(root):
    tables = Tables()
    calibration_matches = 0
    for match in iter_matches(root, range(8)):
        tables.add(match)
        calibration_matches += 1
    tables.finish()
    held = []
    for match in iter_matches(root, range(8, 16)):
        held.append({k:v for k,v in match.items() if k != "rows"})
    assert len(held) == calibration_matches == 512
    rng = np.random.default_rng(16102679)
    shuffle = np.empty(len(held), np.int32)
    for arm in (0, 1):
        indices = np.asarray([j for j,m in enumerate(held) if m["arm"] == arm])
        while True:
            permutation = rng.permutation(indices)
            if all(held[a]["game"]["seed"] != held[b]["game"]["seed"] for a,b in zip(indices, permutation)):
                break
        shuffle[indices] = permutation
    per_match = []
    for j, match in enumerate(iter_matches(root, range(8, 16))):
        arm = match["arm"]
        past = tables.history_logits(arm, match["prefix"])
        unrelated = tables.history_logits(arm, held[shuffle[j]]["prefix"])
        current = np.zeros((2, 4), np.int32)
        old_round = -1
        accumulated = np.zeros((3, 2))
        count = 0
        lead_count = 0
        for row in match["rows"]:
            k, lead, y = row[:3]
            if k != old_round:
                current[:] = 0
                old_round = k
            if k >= 3:
                pi, prior = tables.probabilities(arm, row)
                logits = np.log(prior)+tables.history_logits(arm, current)
                predictions = (softmax(logits)@pi, softmax(logits+past)@pi,
                               softmax(logits+unrelated)@pi)
                for i,p in enumerate(predictions):
                    accumulated[i] += scores(p, y)
                count += 1
                lead_count += lead
            current[lead, y] += 1
        assert count
        game = match["game"]
        per_match.append(dict(seed=game["seed"], team=game["team"], arm=arm,
            opponent=game["opponent"], scored_events=count, lead_events=lead_count,
            scores=(accumulated/count).tolist(), prefix=match["prefix"].tolist(),
            posterior=softmax(past).tolist(),
            guessed_identity=int(past.argmax()), identity=game["opponent"]-47241))
    groups = sorted({m["seed"] for m in per_match})
    assert len(groups) == 128
    opponents = np.asarray([next(m["opponent"] for m in per_match if m["seed"] == seed) for seed in groups])
    grouped = np.asarray([np.mean([m["scores"] for m in per_match if m["seed"] == seed], axis=0) for seed in groups])
    weights = cluster_weights(opponents, 10000, np.random.default_rng(16102680))
    improvement = grouped[:, 0]-grouped[:, 1]
    vs_shuffle = grouped[:, 2]-grouped[:, 1]
    pooled = grouped.mean(0)
    real = dict(calibration_matches=512, heldout_matches=512, independent_test_seed_clusters=128,
        scored_public_events=sum(m["scored_events"] for m in per_match),
        lead_events=sum(m["lead_events"] for m in per_match),
        mean_scores=dict(baseline=pooled[0].tolist(), history=pooled[1].tolist(), shuffled_history=pooled[2].tolist()),
        brier_improvement=confidence(improvement[:, 0].mean(), weights@improvement[:, 0]),
        logloss_improvement=confidence(improvement[:, 1].mean(), weights@improvement[:, 1]),
        brier_history_over_shuffle=confidence(vs_shuffle[:, 0].mean(), weights@vs_shuffle[:, 0]),
        identity_accuracy=float(np.mean([m["guessed_identity"] == m["identity"] for m in per_match])),
        opponent_count=4)
    with np.load(root/"questions/public_answers.npz", allow_pickle=False) as data:
        counts, labels = data["counts"], data["labels"]
    assert counts.shape == (256, 4, 2, 3) and (counts.sum(-1) == 32).all()
    pi = (counts[:, :, 0]+.5)/(32+1.5)
    p0 = pi.mean(1)
    frequencies = counts[:, :, 1]/32
    baseline = (p0*p0).sum(1)[:, None]-2*np.einsum("qc,qic->qi", p0, frequencies)+1
    gain = np.zeros((len(per_match), 256))
    shuffled_gain = np.zeros_like(gain)
    for j, match in enumerate(per_match):
        identity = match["identity"]
        ph = np.einsum("i,qic->qc", np.asarray(match["posterior"]), pi)
        ps = np.einsum("i,qic->qc", np.asarray(per_match[shuffle[j]]["posterior"]), pi)
        for prediction, output in ((ph, gain), (ps, shuffled_gain)):
            brier = (prediction*prediction).sum(1)-2*(prediction*frequencies[:, identity]).sum(1)+1
            output[j] = baseline[:, identity]-brier
    # Crossed uncertainty: resample historical matches by seed and card questions.
    matrix = np.asarray([gain[[j for j,m in enumerate(per_match) if m["seed"] == seed]].mean(0) for seed in groups])
    null_matrix = np.asarray([shuffled_gain[[j for j,m in enumerate(per_match) if m["seed"] == seed]].mean(0) for seed in groups])
    question_weights = np.random.default_rng(16102681).multinomial(256, np.full(256, 1/256), size=10000)/256
    draws = ((weights@matrix)*question_weights).sum(1)
    null_draws = ((weights@(matrix-null_matrix))*question_weights).sum(1)
    fixed = dict(questions=256, calibration_public_answers=32768, test_public_answers=32768,
        same_question_same_cards_and_public_state=True,
        brier_improvement=confidence(matrix.mean(), draws),
        brier_history_over_shuffle=confidence((matrix-null_matrix).mean(), null_draws),
        paired_opponent_category_disagreement=float(np.mean(np.ptp(labels, axis=1) != 0)),
        limitation="fresh-match first-lead questions test persistent style only, not later-round adaptation")
    return dict(complete=True, real_matches=real, fixed_card_questions=fixed,
        methodology=dict(prefix_rounds=3, test_rounds="fourth onward", calibration_waves="00-07",
            test_waves="08-15", primary="real-match categorical Brier improvement; match-weighted",
            smoothing=64, likelihood_tempering=8, bootstrap_draws=10000,
            private_features_used=False, private_candidates_exported=False,
            forced_flags_used=False, neural_training=False,
            limitation="real-match contexts are public summaries; cannot condition on unobserved actual hands")), per_match


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result, matches = analyze(args.root)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    args.output.with_name("heldout-match-scores.json").write_text(json.dumps(matches)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
