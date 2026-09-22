"""Stage B policy head: Q / temperature logits, frozen top-k pruning, sampling."""
from dataclasses import asdict
from pathlib import Path
import random

import gd
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eval.policies import ModelPolicy, PrunedPolicy, load_policy  # noqa: E402
from eval.probes import build_probes  # noqa: E402
from train.ckpt import rng_state, save_checkpoint  # noqa: E402
from train.model import GuandanModel, ModelConfig  # noqa: E402
from train.policy import (PASS_FEATURE, PolicyConfig, StageBPolicy, prune_candidates,  # noqa: E402
                          sample_segments, segment_log_softmax)

# The untracked artifact lives in the main checkout; a worktree sits below it.
M1_FINAL = next((p / ".work/runpod/artifacts/pilot/final.pt"
                 for p in Path(__file__).resolve().parents
                 if (p / ".work/runpod/artifacts/pilot/final.pt").exists()),
                Path(".work/runpod/artifacts/pilot/final.pt"))

SMALL = ModelConfig(obs_dim=gd.OBS_DIM, act_dim=gd.ACT_DIM, state_width=16, state_layers=1,
                    action_width=16, action_layers=1, fusion_width=16, fusion_layers=1)


def real_decisions(count: int, seed: int = 3):
    """Play-phase decisions from greedy self-play: (obs, cand, phase) triples."""
    engine, rows = gd.Engine(), []
    state = gd.MatchState()
    engine.new_match(state, seed)
    while len(rows) < count:
        if state.phase == gd.Phase.RoundEnd:
            engine.new_match(state, seed := seed + 1)
            continue
        actions = engine.legal_actions(state)
        seat = state.to_move
        rows.append((np.asarray(state.observation(seat), dtype=np.float32),
                     np.stack([engine.encode_action(a, state, seat) for a in actions]),
                     int(state.phase)))
        engine.apply(state, actions[engine.greedy(state)])
    return rows


def batch(rows):
    obs = torch.as_tensor(np.stack([r[0] for r in rows]))
    cand = torch.as_tensor(np.concatenate([r[1] for r in rows]).astype(np.float32))
    sizes = torch.tensor([len(r[1]) for r in rows])
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), torch.cumsum(sizes, 0)))
    return obs, cand, offsets, torch.tensor([r[2] for r in rows])


def reference_log_probs(model, obs, cand, offsets, phase, temperature, top_k):
    """Per decision, a dict {original index: log softmax(Q/t) over the pruned set}."""
    with torch.no_grad():
        q = model.score_candidates(obs, cand, offsets, phase)
    out = []
    for i in range(len(offsets) - 1):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        seg, is_pass = q[lo:hi], cand[lo:hi, PASS_FEATURE] > 0.5
        order = sorted(range(hi - lo), key=lambda j: (-float(seg[j]), j))
        kept = sorted(set(order[:top_k]) | set(torch.nonzero(is_pass).flatten().tolist()))
        logp = torch.log_softmax(seg[kept].double() / temperature, 0)
        out.append(dict(zip(kept, logp.tolist())))
    return out


def test_prune_is_ragged_top_k_plus_pass_and_noop_when_small():
    scores = torch.tensor([5., 1., 3., 2.,   0., 9.,   4., 4., 4.])
    offsets = torch.tensor([0, 4, 6, 9])
    keep_pass = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool)
    keep, pruned = prune_candidates(scores, offsets, 2, keep_pass)
    # Segment 0: top two are 0 and 2, pass (1) is kept although it ranks last.
    # Segment 1 has two candidates: untouched. Segment 2: ties break low index.
    assert keep.tolist() == [0, 1, 2, 4, 5, 6, 7]
    assert pruned.tolist() == [0, 3, 5, 7]
    keep, pruned = prune_candidates(scores, offsets, 4)
    assert keep.tolist() == list(range(9)) and pruned.tolist() == offsets.tolist()
    with pytest.raises(ValueError):
        prune_candidates(scores, offsets, 0)


def test_segment_log_softmax_normalises_and_is_differentiable():
    logits = torch.randn(10, generator=torch.Generator().manual_seed(0), requires_grad=True)
    offsets = torch.tensor([0, 1, 4, 10])
    logp = segment_log_softmax(logits, offsets)
    sums = torch.zeros(3).index_add(0, torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 2, 2]), logp.exp())
    torch.testing.assert_close(sums, torch.ones(3))
    torch.testing.assert_close(logp[4:10], torch.log_softmax(logits[4:10], 0))
    logp.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_sampler_frequencies_match_probabilities_and_replay_under_seed():
    logits = torch.tensor([0.0, 1.0, 2.0, -1.0, 0.5, 0.5])
    offsets = torch.tensor([0, 4, 6])
    logp = segment_log_softmax(logits, offsets)
    draws = 40000
    rep_logp = logp.repeat(draws)
    sizes = torch.tensor([4, 2]).repeat(draws)
    rep_offsets = torch.cat((torch.zeros(1, dtype=torch.long), torch.cumsum(sizes, 0)))
    first = sample_segments(rep_logp, rep_offsets, torch.Generator().manual_seed(7))
    again = sample_segments(rep_logp, rep_offsets, torch.Generator().manual_seed(7))
    assert torch.equal(first, again)
    for seg, (lo, hi) in enumerate(((0, 4), (4, 6))):
        freq = torch.bincount(first[seg::2], minlength=hi - lo).double() / draws
        expected = logp[lo:hi].double().exp()
        # Five binomial standard errors per cell.
        assert (freq - expected).abs().max() < 5 * (0.25 / draws) ** 0.5


def test_untrained_policy_is_softmax_of_reference_q_over_pruned_set():
    torch.manual_seed(0)
    model = GuandanModel(SMALL)
    temperature, top_k = 0.7, 5
    policy = StageBPolicy.from_model(model, PolicyConfig(temperature=temperature, top_k=top_k))
    obs, cand, offsets, phase = batch(real_decisions(60))
    sizes = offsets[1:] - offsets[:-1]
    assert (sizes > top_k).any() and (sizes <= top_k).any()
    expected = reference_log_probs(model, obs, cand, offsets, phase, temperature, top_k)
    step = policy.act(obs, cand, offsets, phase, generator=torch.Generator().manual_seed(1))
    is_pass = cand[:, PASS_FEATURE] > 0.5
    for i, dist in enumerate(expected):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        plo, phi = int(step.pruned_offsets[i]), int(step.pruned_offsets[i + 1])
        kept = (step.keep_index[plo:phi] - lo).tolist()
        assert kept == sorted(dist)
        if is_pass[lo:hi].any():
            assert int(torch.nonzero(is_pass[lo:hi])[0]) in kept
        if hi - lo <= top_k:
            assert kept == list(range(hi - lo))
        logp = segment_log_softmax(step.logits, step.pruned_offsets)[plo:phi].detach()
        np.testing.assert_allclose(logp.detach().double().numpy(), list(dist.values()), atol=1e-5)
        assert float(logp.exp().sum()) == pytest.approx(1.0, abs=1e-5)
        chosen = int(step.choice[i])
        assert chosen in dist
        assert float(step.log_prob[i].detach()) == pytest.approx(dist[chosen], abs=1e-5)
        probs = np.exp(list(dist.values()))
        assert float(step.entropy[i].detach()) == pytest.approx(-(probs * np.log(probs)).sum(), abs=1e-5)
    # Same generator seed, same choices; evaluate() reproduces the log-probs.
    again = policy.act(obs, cand, offsets, phase, generator=torch.Generator().manual_seed(1))
    assert torch.equal(step.choice, again.choice)
    log_prob, entropy = policy.evaluate(obs, cand[step.keep_index], step.pruned_offsets,
                                        phase, step.pruned_choice)
    torch.testing.assert_close(log_prob, step.log_prob)
    torch.testing.assert_close(entropy, step.entropy)
    log_prob.sum().backward()
    assert all(p.grad is None for p in policy.reference.parameters())
    assert any(p.grad is not None for p in policy.net.parameters())


def test_k_at_least_n_is_a_noop_and_training_does_not_move_pruning():
    torch.manual_seed(1)
    model = GuandanModel(SMALL)
    obs, cand, offsets, phase = batch(real_decisions(20))
    big = StageBPolicy.from_model(model, PolicyConfig(top_k=int((offsets[1:] - offsets[:-1]).max())))
    keep, pruned = big.prune(obs, cand, offsets, phase)
    assert keep.tolist() == list(range(len(cand))) and torch.equal(pruned, offsets)
    small = StageBPolicy.from_model(model, PolicyConfig(top_k=3))
    before = small.prune(obs, cand, offsets, phase)
    with torch.no_grad():
        for p in small.net.parameters():
            p.add_(torch.randn_like(p))
    after = small.prune(obs, cand, offsets, phase)
    assert torch.equal(before[0], after[0]) and torch.equal(before[1], after[1])


def test_tribute_phases_go_through_their_own_heads():
    torch.manual_seed(2)
    model = GuandanModel(SMALL)
    policy = StageBPolicy.from_model(model, PolicyConfig(temperature=2.0))
    obs = torch.randn(3, SMALL.obs_dim)
    cand = torch.rand(9, SMALL.act_dim)
    cand[:, PASS_FEATURE] = 0
    offsets = torch.tensor([0, 3, 6, 9])
    phase = torch.tensor([1, 2, 3])
    with torch.no_grad():
        q = model.score_candidates(obs, cand, offsets, phase)
        torch.testing.assert_close(policy.logits(obs, cand, offsets, phase), q / 2.0)


def stage_b_checkpoint(tmp_path, model, config):
    policy = StageBPolicy.from_model(model, config, reference_checkpoint_id=None)
    path = tmp_path / "ppo.pt"
    save_checkpoint(path, policy.checkpoint_payload(
        optimizer=torch.optim.Adam(policy.net.parameters()).state_dict(),
        config={"seed": 5}, progress={"updates": 0}, rng=rng_state(np.random.default_rng(0))))
    return policy, path


def test_load_policy_round_trips_a_stage_b_checkpoint(tmp_path):
    torch.manual_seed(3)
    model = GuandanModel(SMALL)
    policy, path = stage_b_checkpoint(tmp_path, model, PolicyConfig(temperature=0.5, top_k=4))
    with torch.no_grad():
        next(policy.net.parameters()).add_(0.1)  # net diverges from reference
    save_checkpoint(path, policy.checkpoint_payload(optimizer={}, config={"seed": 5},
                                                    progress={}, rng={}))
    loaded = load_policy(str(path))
    assert isinstance(loaded, PrunedPolicy) and loaded.stage == "ppo"
    assert loaded.name.startswith("ppo.pt@") and loaded.name.endswith("/ppo")
    assert loaded.stage_b.config == PolicyConfig(temperature=0.5, top_k=4)
    for a, b in zip(policy.net.state_dict().values(), loaded.stage_b.net.state_dict().values()):
        assert torch.equal(a, b)
    for a, b in zip(model.state_dict().values(), loaded.stage_b.reference.state_dict().values()):
        assert torch.equal(a, b)
    probe = build_probes()[0]
    actions = probe.engine.legal_actions(probe.state)
    choice = loaded.select(probe.engine, probe.state, actions, random.Random(0))
    assert 0 <= choice < len(actions)
    sampled = load_policy(f"sample=2:{path}")
    assert sampled.name.endswith("/ppo/sample=2") and sampled.stage_b.config.temperature == 2
    picks = [sampled.select(probe.engine, probe.state, actions, random.Random(s)) for s in range(8)]
    assert picks == [sampled.select(probe.engine, probe.state, actions, random.Random(s))
                     for s in range(8)]


def test_untrained_stage_b_plays_like_its_dmc_start_and_dmc_specs_unchanged(tmp_path):
    torch.manual_seed(4)
    model = GuandanModel(SMALL)
    dmc_path = tmp_path / "dmc.pt"
    save_checkpoint(dmc_path, {"model_config": asdict(SMALL), "model": model.state_dict(),
                               "optimizer": {}, "config": {}, "progress": {}, "rng": {}})
    dmc = load_policy(str(dmc_path))
    assert type(dmc) is ModelPolicy
    _, ppo_path = stage_b_checkpoint(tmp_path, model, PolicyConfig(top_k=3))
    ppo = load_policy(str(ppo_path))
    for probe in build_probes():
        actions = probe.engine.legal_actions(probe.state)
        assert (ppo.select(probe.engine, probe.state, actions, random.Random(0))
                == dmc.select(probe.engine, probe.state, actions, random.Random(0)))
    with pytest.raises(ValueError, match="sample=<T>"):
        load_policy(f"sample:{dmc_path}")
    assert load_policy(f"sample=1.5:{dmc_path}").name.endswith("/sample=1.5")


@pytest.mark.skipif(not M1_FINAL.exists(), reason="M1 final checkpoint not available")
def test_m1_final_initialisation_is_exactly_m1_q_over_temperature():
    temperature = 0.8
    policy = StageBPolicy.from_dmc_checkpoint(M1_FINAL, PolicyConfig(temperature=temperature))
    assert policy.reference_checkpoint_id is not None
    obs, cand, offsets, phase = batch(real_decisions(40, seed=17))
    with torch.no_grad():
        q = policy.reference.score_candidates(obs, cand, offsets, phase)
        assert torch.equal(policy.logits(obs, cand, offsets, phase), q / temperature)
        step = policy.act(obs, cand, offsets, phase, generator=torch.Generator().manual_seed(0))
    expected = reference_log_probs(policy.reference, obs, cand, offsets, phase, temperature, 32)
    for i, dist in enumerate(expected):
        assert int(step.choice[i]) in dist
        assert float(step.log_prob[i].detach()) == pytest.approx(dist[int(step.choice[i])], abs=1e-4)
