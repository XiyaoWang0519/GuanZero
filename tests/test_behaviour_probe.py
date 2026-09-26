"""Behaviour probe: schema-3 logs, match stitching, causality, privacy, controls."""
from dataclasses import asdict
import json

import numpy as np
import pytest
import torch

from eval import collect_belief as collection
from train import behaviour_probe as probe
from train.buffer import Decision
from train.ckpt import save_checkpoint
from train.logs import CANDIDATE_SCHEMA_VERSION, DRIVER_BOT, DRIVER_POLICY, TOKEN_DIM, save_round
from train.model import GuandanModel, ModelConfig

OBS_DIM = probe.ProbeConfig().obs_dim


def tiny_checkpoint(path):
    torch.manual_seed(5)
    config = ModelConfig(state_width=8, state_layers=1, action_width=8,
                         action_layers=1, fusion_width=8, fusion_layers=1)
    save_checkpoint(path, {"model_config": asdict(config), "model": GuandanModel(config).state_dict(),
        "optimizer": {}, "config": {"seed": 7, "action_mode": "canonical"},
        "progress": {}, "rng": {}})
    return path


@pytest.fixture(scope="module")
def collected(tmp_path_factory):
    """A tiny schema-3 collection with styled opponents, shared by the tests below."""
    root = tmp_path_factory.mktemp("probe")
    output = root / "logs"
    collection.collect_belief(tiny_checkpoint(root / "frozen.pt"), output, rounds=32, num_envs=4,
                              seed=103, max_seconds=180, purpose="architecture_probe",
                              candidates=True, styled=True, style_region="mixed")
    return output


def make_round(decisions: int = 3, candidates: int = 4):
    rng = np.random.default_rng(0)
    tokens, metas, rows = [], [], []
    for i in range(decisions * 2):
        token = np.zeros(TOKEN_DIM, np.uint8)
        token[i % 4] = 1
        token[4 + i] = 1
        token[158 + 5] = 1
        tokens.append(token)
        metas.append((i % 3, int(i % 4 == 3), 3))
    for i in range(decisions):
        prefix = 2 * i
        cand = rng.integers(0, 2, size=(candidates, 154), dtype=np.uint8)
        choice = i % candidates
        cand[choice] = tokens[prefix][4:158]
        rows.append(Decision(obs=np.zeros(OBS_DIM, np.uint8), action=cand[choice].copy(),
                             hidden=np.zeros((3, 54), np.uint8), seat=prefix % 4, phase=3,
                             prefix=prefix, cand=cand,
                             cand_abstract=np.array([(prefix + j) % 3 for j in range(candidates)]),
                             choice=choice))
        rows[-1].cand_abstract[choice] = metas[prefix][0]
    return rows, tokens, metas


def meta(match_id=0, round_index=0, env_id=0):
    return {"match_id": match_id, "round_index": round_index, "env_id": env_id,
            "styles": np.zeros((4, 18), np.float32), "seat_driver": [0, 0, 0, 0],
            "style_region": "none", "styled": False}


def test_schema_three_round_trips_candidates_and_token_metadata(tmp_path):
    decisions, tokens, metas = make_round()
    path = tmp_path / "round.npz"
    save_round(path, decisions, tokens, group="g", meta=meta(), token_metas=metas)
    with np.load(path, allow_pickle=False) as data:
        assert int(data["schema_version"]) == CANDIDATE_SCHEMA_VERSION
        assert {"cand", "cand_offsets", "cand_abstract", "choice", "token_abstract",
                "token_forced", "token_phase"} <= set(data.files)
        np.testing.assert_array_equal(data["cand_offsets"], [0, 4, 8, 12])
        np.testing.assert_array_equal(data["choice"], [0, 1, 2])
        np.testing.assert_array_equal(data["token_forced"], [m[1] for m in metas])
        np.testing.assert_array_equal(data["token_abstract"], [m[0] for m in metas])
        for i, p in enumerate(data["prefix"]):
            chosen = data["cand_offsets"][i] + data["choice"][i]
            np.testing.assert_array_equal(data["tokens"][p, 4:158], data["cand"][chosen])


def test_schema_three_rejects_misaligned_or_missing_candidates(tmp_path):
    decisions, tokens, metas = make_round()
    with pytest.raises(ValueError):
        save_round(tmp_path / "a.npz", decisions, tokens, group="g", token_metas=metas)
    with pytest.raises(ValueError):
        save_round(tmp_path / "b.npz", decisions, tokens, group="g", meta=meta(),
                   token_metas=metas[:-1])
    decisions[1].choice = 3
    with pytest.raises(ValueError):
        save_round(tmp_path / "c.npz", decisions, tokens, group="g", meta=meta(), token_metas=metas)
    decisions[1].cand = None
    with pytest.raises(ValueError):
        save_round(tmp_path / "d.npz", decisions, tokens, group="g", meta=meta(), token_metas=metas)


def test_abstract_feature_table_is_one_hot_per_block_and_matches_the_engine():
    table = probe.abstract_features()
    assert table.shape == (probe.VOCAB, probe.FEATURE_DIM)
    assert np.all(table[:, :11].sum(1) == 1) and np.all(table[:, 11:26].sum(1) == 1)
    assert table[:, 26:33].sum() == 13 * 7 and table[:, 33:47].sum() == 13 * 14
    assert table[:, 47:51].sum() == 40
    import gd
    # Every logged candidate row's type/key/bomb columns must match the table;
    # that check runs inside load_matches, so exercise it on a real engine row.
    env = gd.VecEnv(1, num_threads=1, seed=3, rules=gd.RuleConfig.house(), actions=gd.ActionConfig())
    env.reset()
    batch = env.pending()
    ids = [a.abstract_id for a in env.row_actions(0)]
    cand = np.asarray(batch.cand, dtype=np.float32)[:len(ids)]
    columns = np.concatenate((cand[:, 108:119], cand[:, 121:136], cand[:, 136:143]), axis=1)
    np.testing.assert_array_equal(table[ids][:, :33], columns)


def test_collection_stitches_matches_and_rejects_missing_rounds(collected, tmp_path):
    matches, provenance = probe.load_matches(collected)
    assert provenance["candidates"] and provenance["schema_version"] == CANDIDATE_SCHEMA_VERSION
    assert provenance["styled"]
    drivers = np.concatenate([m.driver for m in matches])
    assert set(drivers) == {DRIVER_POLICY, DRIVER_BOT}
    for m in matches:
        assert np.all(np.diff(m.prefix) > 0) and m.prefix[-1] < len(m.tokens)
        assert m.token_round[0] == 0 and np.all(np.diff(m.token_round) >= 0)
        for i, p in enumerate(m.prefix):
            chosen = m.cand_offsets[i] + m.choice[i]
            np.testing.assert_array_equal(m.tokens[p, 4:158], m.cand[chosen])
            assert m.token_abstract[p] == m.cand_abstract[chosen]
            assert m.token_round[p] == m.decision_round[i]
    multi = max(matches, key=lambda m: int(m.token_round.max()))
    if int(multi.token_round.max()) >= 1:
        broken = tmp_path / "broken"
        broken.mkdir()
        for path in collected.iterdir():
            if path.name.startswith("round-"):
                with np.load(path, allow_pickle=False) as data:
                    if str(data["group"]) == multi.group and int(data["round_index"]) == 0:
                        continue
            (broken / path.name).write_bytes(path.read_bytes())
        with pytest.raises(ValueError, match="missing rounds"):
            probe.load_matches(broken)


def small_config(**overrides):
    base = dict(width=32, layers=2, heads=4, action_width=16, fusion_width=16)
    base.update(overrides)
    return probe.ProbeConfig(**base)


def altered_copy(match, **fields):
    copy = probe.Match(**{**match.__dict__})
    for name, value in fields.items():
        setattr(copy, name, value)
    return copy


def test_query_reads_only_its_visible_prefix_and_nothing_private_enters_the_stream(collected):
    torch.manual_seed(1)
    matches, _ = probe.load_matches(collected)
    match = max(matches, key=lambda m: m.decisions)
    model = probe.ProbeModel(small_config(tower="history", head="vocab", ntp_weight=1.0)).eval()
    with torch.no_grad():
        state, at_prefix = model.tower(probe.Batch([match], "cpu"))
    i = match.decisions // 2
    p = int(match.prefix[i])
    tokens = match.tokens.copy()
    tokens[p:] = (tokens[p:] + 1) % 2
    rounds = match.token_round.copy()
    rounds[p:] += 1
    with torch.no_grad():
        state2, at_prefix2 = model.tower(probe.Batch([altered_copy(match, tokens=tokens,
                                                                   token_round=rounds)], "cpu"))
    earlier = torch.as_tensor(match.prefix <= p)
    assert torch.allclose(state[earlier], state2[earlier], atol=1e-5)
    assert torch.allclose(at_prefix[earlier], at_prefix2[earlier], atol=1e-5)
    assert not torch.allclose(state[~earlier], state2[~earlier], atol=1e-5)
    # Private observations change states but never the stream positions the
    # NTP head reads, so NTP stays a public-only prediction.
    private = altered_copy(match, obs=(match.obs + 1) % 2)
    with torch.no_grad():
        state3, at_prefix3 = model.tower(probe.Batch([private], "cpu"))
    assert torch.allclose(at_prefix, at_prefix3, atol=1e-6)
    assert not torch.allclose(state, state3, atol=1e-5)


def test_window_control_sees_exactly_the_last_k_tokens(collected):
    torch.manual_seed(4)
    matches, _ = probe.load_matches(collected)
    match = max(matches, key=lambda m: m.decisions)
    k = 3
    model = probe.ProbeModel(small_config(tower="history", head="cand", window=k)).eval()
    with torch.no_grad():
        state, at_prefix = model.tower(probe.Batch([match], "cpu"))
    # Changing tokens older than k before a decision must not touch it.
    late = match.prefix >= k + 2
    assert late.any()
    i = int(np.flatnonzero(late)[-1])
    p = int(match.prefix[i])
    tokens = match.tokens.copy()
    tokens[:p - k] = (tokens[:p - k] + 1) % 2
    with torch.no_grad():
        state2, at_prefix2 = model.tower(probe.Batch([altered_copy(match, tokens=tokens)], "cpu"))
    assert torch.allclose(state[i], state2[i], atol=1e-5)
    assert torch.allclose(at_prefix[i], at_prefix2[i], atol=1e-5)
    # Changing a token inside the window does.
    tokens = match.tokens.copy()
    tokens[p - 1] = (tokens[p - 1] + 1) % 2
    with torch.no_grad():
        state3, _ = model.tower(probe.Batch([altered_copy(match, tokens=tokens)], "cpu"))
    assert not torch.allclose(state[i], state3[i], atol=1e-5)
    # A decision with fewer than k tokens before it sees exactly its short prefix.
    first = int(np.flatnonzero(match.prefix < k)[0]) if (match.prefix < k).any() else None
    if first is not None:
        full = probe.ProbeModel(small_config(tower="history", head="cand")).eval()
        full.load_state_dict(model.state_dict())
        with torch.no_grad():
            state_full, _ = full.tower(probe.Batch([match], "cpu"))
        assert torch.allclose(state[first], state_full[first], atol=1e-4)


def test_forced_flag_never_enters_the_stream(collected):
    torch.manual_seed(6)
    matches, _ = probe.load_matches(collected)
    match = max(matches, key=lambda m: int(m.token_forced.sum()))
    assert match.token_forced.sum() > 0
    model = probe.ProbeModel(small_config(tower="history", head="cand", ntp_weight=1.0)).eval()
    flipped = altered_copy(match, token_forced=1 - match.token_forced)
    with torch.no_grad():
        state, at_prefix = model.tower(probe.Batch([match], "cpu"))
        state2, at_prefix2 = model.tower(probe.Batch([flipped], "cpu"))
    assert torch.allclose(state, state2) and torch.allclose(at_prefix, at_prefix2)


def test_batched_streams_match_single_match_encoding(collected):
    torch.manual_seed(2)
    matches, _ = probe.load_matches(collected)
    model = probe.ProbeModel(small_config(tower="history", head="cand")).eval()
    two = sorted(matches, key=lambda m: len(m.tokens))[:2]
    with torch.no_grad():
        together, _ = model.tower(probe.Batch(two, "cpu"))
        alone = torch.cat([model.tower(probe.Batch([m], "cpu"))[0] for m in two])
    assert torch.allclose(together, alone, atol=1e-4)


@pytest.mark.parametrize("head", ("vocab", "vocab_struct"))
def test_vocab_heads_mask_illegal_actions(collected, head):
    torch.manual_seed(3)
    matches, _ = probe.load_matches(collected)
    batch = probe.Batch(matches[:1], "cpu")
    state = torch.randn(batch.decisions, 32)
    log_probs, chosen = probe.make_head(small_config(head=head)).abstract_log_probs(state, batch)
    assert torch.all(torch.isinf(log_probs[~batch.legal]))
    assert torch.all(torch.isfinite(log_probs[batch.legal]))
    assert torch.allclose(log_probs.exp().sum(1), torch.ones(batch.decisions), atol=1e-5)
    assert torch.allclose(chosen, log_probs.gather(1, batch.chosen_abstract[:, None])[:, 0])


def test_candidate_head_merges_variants_into_abstract_probabilities(collected):
    torch.manual_seed(3)
    matches, _ = probe.load_matches(collected)
    batch = probe.Batch(matches[:1], "cpu")
    state = torch.randn(batch.decisions, 32)
    log_probs, chosen = probe.CandidateHead(small_config()).abstract_log_probs(state, batch)
    assert torch.allclose(log_probs.exp().sum(1), torch.ones(batch.decisions), atol=1e-4)
    assert torch.all(log_probs[~batch.legal] < -60)
    assert torch.all(chosen <= log_probs.gather(1, batch.chosen_abstract[:, None])[:, 0] + 1e-5)


def test_config_rejects_controls_without_a_history_tower():
    with pytest.raises(ValueError):
        probe.ProbeConfig(tower="flat", ntp_weight=1.0)
    with pytest.raises(ValueError):
        probe.ProbeConfig(tower="flat", window=4)
    with pytest.raises(ValueError):
        probe.ProbeConfig(head="softmax")


@pytest.mark.parametrize("tower,head,weight,window", (
    ("flat", "cand", 0.0, 0), ("history", "vocab_struct", 0.5, 0), ("history", "vocab", 0.5, 4)))
def test_tiny_runs_complete_with_driver_cells(collected, tmp_path, tower, head, weight, window):
    config = small_config(tower=tower, head=head, ntp_weight=weight, window=window)
    out = tmp_path / f"{tower}-{head}-{window}"
    report = probe.run_probe(collected, out, config, steps=4, eval_every=2, batch_matches=1,
                             min_matches=3, patience=2)
    assert report["status"] == "complete" and report["best"]["step"] in (2, 4)
    assert (out / "best.pt").exists()
    means, counts = report["test"]["means"], report["test"]["counts"]
    assert np.isfinite(means["bc_ce/all"]) and 0 <= means["bc_acc/all"] <= 1
    assert counts["bc_ce/policy"] + counts["bc_ce/bot"] == counts["bc_ce/nontrivial"]
    assert ("ntp_ce/all" in means) == (weight > 0)
    if weight > 0:
        assert counts["ntp_ce/next_pass"] + counts["ntp_ce/next_play"] == counts["ntp_ce/all"]
    assert json.loads((out / "report.json").read_text())["status"] == "complete"
