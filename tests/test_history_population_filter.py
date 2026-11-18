"""Tests for behavior-filtered archive diversity, probes, and state adoption."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from train.history_model import DecisionInputs, HistoryPolicyConfig, fresh_player
from train.history_population import (
    ARCHIVE_FILTERS,
    DEFAULT_BEHAVIOR_PROBE_ROWS,
    DEFAULT_BEHAVIOR_PROBE_SEED,
    HistoryPopulation,
    behavior_fingerprint,
    build_behavior_probe,
    decision_inputs_on_device,
    deserialize_decision_inputs,
    freeze_decision_inputs,
    mean_behavior_tv,
    serialize_decision_inputs,
)

BASE = dict(width=32, layers=1, heads=4, action_width=16, fusion_width=16, critic_width=16, critic_layers=1)


def _prob_vec(dist: list[list[float]]) -> np.ndarray:
    """Flatten per-decision probability distributions into a float32 fingerprint."""
    return np.asarray([p for row in dist for p in row], dtype=np.float32)


def test_constants_and_validation() -> None:
    assert ARCHIVE_FILTERS == ("thin", "strength_behavior", "nearest_pair")
    assert DEFAULT_BEHAVIOR_PROBE_ROWS == 128
    assert DEFAULT_BEHAVIOR_PROBE_SEED == 20261117

    actor, _ = fresh_player(HistoryPolicyConfig(**BASE), 0)
    with pytest.raises(ValueError, match="archive_filter"):
        HistoryPopulation(actor, "l", 1, archive_filter="bad")
    with pytest.raises(ValueError, match="archive_min_tv"):
        HistoryPopulation(actor, "l", 1, archive_min_tv=-0.1)
    with pytest.raises(ValueError, match="archive_min_tv"):
        HistoryPopulation(actor, "l", 1, archive_min_tv=float("nan"))
    with pytest.raises(ValueError, match="archive_max_age_updates"):
        HistoryPopulation(actor, "l", 1, archive_max_age_updates=-1)

    with pytest.raises(ValueError, match="decision count"):
        mean_behavior_tv(np.ones(2, np.float32), np.ones(2, np.float32), 0)
    with pytest.raises(ValueError, match="layout"):
        mean_behavior_tv(np.ones(2, np.float32), np.ones(3, np.float32), 1)


def test_thin_default_unchanged_and_stats() -> None:
    actor, _ = fresh_player(HistoryPolicyConfig(**BASE), 1)
    p1 = HistoryPopulation(actor, "l", 5, recent=2, archive_every=2, archive_size=4)
    p2 = HistoryPopulation(actor, "l", 5, recent=2, archive_every=2, archive_size=4, archive_filter="thin")
    for u in range(2, 24, 2):
        p1.snapshot(u)
        p2.snapshot(u)
        p1.prune({})
        p2.prune({})
    assert p1.archive == p2.archive
    assert p1.assignment(0, 0) == p2.assignment(0, 0)
    assert p1.rng.bit_generator.state == p2.rng.bit_generator.state
    stats = p1.archive_diversity_stats()
    assert stats["archive_filter"] == "thin"
    assert stats["archive_min_pair_tv"] == 0.0
    assert stats["archive_mean_pair_tv"] == 0.0
    assert stats["archive_age_span"] > 0
    assert p1.needs_probe_rebind() is False


def test_nearest_pair_vs_strength_behavior_on_drifting_trajectory() -> None:
    """Long drifting trajectory: strength_behavior collapses to recent; nearest_pair keeps span & diversity."""
    actor, _ = fresh_player(HistoryPolicyConfig(**BASE), 2)
    sb_pool = HistoryPopulation(
        actor,
        "drift",
        7,
        recent=2,
        archive_every=1,
        archive_size=4,
        archive_filter="strength_behavior",
        archive_min_tv=0.05,
    )
    np_pool = HistoryPopulation(
        actor,
        "drift",
        7,
        recent=2,
        archive_every=1,
        archive_size=4,
        archive_filter="nearest_pair",
        archive_min_tv=0.05,
    )

    # Phase 1: three orthogonal early styles (TV=1.0), followed by a slow drift from the 4th
    # orthogonal corner [0, 0, 0, 1] toward [0, 0, 0.5, 0.5] (step TV = 0.0625 >= 0.05).
    # strength_behavior admits every step and evicts all early anchors, collapsing to the last 4;
    # nearest_pair merges within the drifting cluster and preserves all 3 early orthogonal anchors.
    anchors = [
        [[1.0, 0.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0, 0.0]],
    ]
    drift = [
        [[0.0, 0.0, k / 16.0, 1.0 - k / 16.0]] for k in range(0, 9)
    ]
    trajectory = anchors + drift
    for step, dist in enumerate(trajectory, start=1):
        vec = _prob_vec(dist)
        sb_pool.snapshot(step * 10, behavior=vec, behavior_decisions=1)
        np_pool.snapshot(step * 10, behavior=vec, behavior_decisions=1)
        sb_pool.prune({})
        np_pool.prune({})

    sb_updates = [sb_pool.metadata[i]["update"] for i in sb_pool.archive]
    np_updates = [np_pool.metadata[i]["update"] for i in np_pool.archive]
    sb_stats = sb_pool.archive_diversity_stats()
    np_stats = np_pool.archive_diversity_stats()

    # strength_behavior dropped all early anchors and holds only the latest 4 steps.
    assert sorted(sb_updates) == [90, 100, 110, 120]
    assert sb_stats["archive_age_span"] == 30
    # nearest_pair preserved early anchors across the whole trajectory with strictly higher min pairwise TV.
    assert sorted(np_updates) == [10, 20, 30, 120]
    assert np_stats["archive_age_span"] == 110
    assert np_stats["archive_min_pair_tv"] == pytest.approx(0.5)
    assert np_stats["archive_min_pair_tv"] > sb_stats["archive_min_pair_tv"]
    assert set(np_pool.behaviors) == set(np_pool.archive)


def test_nearest_pair_in_place_duplicate_and_closest_pair_merge() -> None:
    actor, _ = fresh_player(HistoryPolicyConfig(**BASE), 3)
    pool = HistoryPopulation(
        actor,
        "np",
        11,
        recent=2,
        archive_every=1,
        archive_size=3,
        archive_filter="nearest_pair",
        archive_min_tv=0.05,
    )
    # Seed 3 distinct clusters: A, B, C where (A, B) are closest (TV=0.10) and C is far (TV=0.80).
    id_a = pool.snapshot(10, behavior=_prob_vec([[1.0, 0.0, 0.0, 0.0]]), behavior_decisions=1)
    id_b = pool.snapshot(20, behavior=_prob_vec([[0.9, 0.1, 0.0, 0.0]]), behavior_decisions=1)
    id_c = pool.snapshot(30, behavior=_prob_vec([[0.0, 0.0, 1.0, 0.0]]), behavior_decisions=1)
    assert pool.archive == [id_a, id_b, id_c]

    # 1. Near-duplicate of C (TV=0.02 < 0.05) with newer update replaces C in-place at slot 2.
    id_c2 = pool.snapshot(40, behavior=_prob_vec([[0.0, 0.0, 0.98, 0.02]]), behavior_decisions=1)
    assert pool.archive == [id_a, id_b, id_c2]
    assert pool.archive_replacements == 1
    assert id_c not in pool.behaviors

    # Near-duplicate with OLDER update is rejected without mutating archive.
    id_old_dup = pool.snapshot(35, behavior=_prob_vec([[0.0, 0.0, 0.99, 0.01]]), behavior_decisions=1)
    assert pool.archive == [id_a, id_b, id_c2]
    assert pool.archive_rejects == 1
    assert id_old_dup not in pool.behaviors

    # 2. Candidate D discovers a brand-new region [0, 0, 0, 1] (far from all existing members).
    # Closest pair in {A, B, C2, D} is (A, B) at TV=0.10 -> older (A) is victim, replaced in-place by D.
    id_d = pool.snapshot(50, behavior=_prob_vec([[0.0, 0.0, 0.0, 1.0]]), behavior_decisions=1)
    assert pool.archive == [id_d, id_b, id_c2]
    assert pool.archive_replacements == 2
    assert id_a not in pool.behaviors

    # 3. Candidate E that is closer to B (TV=0.06 >= 0.05) than any other pair, with older update than B -> rejected.
    id_e = pool.snapshot(15, behavior=_prob_vec([[0.84, 0.16, 0.0, 0.0]]), behavior_decisions=1)
    assert pool.archive == [id_d, id_b, id_c2]
    assert pool.archive_rejects == 2
    assert id_e not in pool.behaviors


def test_archive_max_age_updates_eviction() -> None:
    actor, _ = fresh_player(HistoryPolicyConfig(**BASE), 4)
    pool = HistoryPopulation(
        actor,
        "age",
        1,
        recent=1,
        archive_every=1,
        archive_size=4,
        archive_filter="nearest_pair",
        archive_min_tv=0.05,
        archive_max_age_updates=50,
    )
    id1 = pool.snapshot(10, behavior=_prob_vec([[1.0, 0.0, 0.0]]), behavior_decisions=1)
    id2 = pool.snapshot(40, behavior=_prob_vec([[0.0, 1.0, 0.0]]), behavior_decisions=1)
    assert pool.archive == [id1, id2]
    assert pool.archive_age_evictions == 0

    # Update 60: 60 - 10 = 50 <= 50 (kept).
    id3 = pool.snapshot(60, behavior=_prob_vec([[0.0, 0.0, 1.0]]), behavior_decisions=1)
    assert pool.archive == [id1, id2, id3]
    assert pool.archive_age_evictions == 0

    # Update 61: 61 - 10 = 51 > 50 -> id1 evicted by age cap!
    id4 = pool.snapshot(61, behavior=_prob_vec([[0.5, 0.5, 0.0]]), behavior_decisions=1)
    assert pool.archive == [id2, id3, id4]
    assert pool.archive_age_evictions == 1
    assert id1 not in pool.behaviors

    pool.prune({})
    assert id1 not in pool.models
    assert pool.archive_diversity_stats()["archive_age_evictions"] == 1


def test_build_behavior_probe_determinism_stratification_and_roundtrip() -> None:
    actor, _ = fresh_player(HistoryPolicyConfig(**BASE), 17)
    probe_a, meta_a = build_behavior_probe(actor, rows=36, seed=20261117, num_envs=4, max_steps=512)
    probe_b, meta_b = build_behavior_probe(actor, rows=36, seed=20261117, num_envs=4, max_steps=512)

    assert meta_a == meta_b
    assert meta_a["seed"] == 20261117
    assert meta_a["decisions"] == 36
    assert meta_a["cross_round_decisions"] > 0
    assert 0 < meta_a["cross_round_decisions"] < 36  # both round 0 and round >= 1 represented
    assert meta_a["non_forced_decisions"] > 0
    assert meta_a["matches"] > 1

    assert probe_a.decisions == 36
    assert probe_a.obs_mask_prob is None
    assert probe_a.match_groups is None
    assert probe_a.round_index is not None
    assert int(probe_a.streams.tokens.shape[1]) == int(probe_a.prefix.max().item())
    assert torch.equal(probe_a.obs, probe_b.obs)
    assert torch.equal(probe_a.cand, probe_b.cand)
    assert torch.equal(probe_a.streams.tokens, probe_b.streams.tokens)
    assert torch.equal(probe_a.prefix, probe_b.prefix)
    assert torch.equal(probe_a.round_index, probe_b.round_index)

    # Plain dict serialization roundtrip (no custom classes).
    raw = serialize_decision_inputs(probe_a)
    assert type(raw) is dict and type(raw["streams"]) is dict
    assert isinstance(raw["obs"], torch.Tensor) and raw["obs"].device.type == "cpu"
    restored = deserialize_decision_inputs(raw)
    assert restored.obs_mask_prob is None
    assert torch.equal(restored.obs, probe_a.obs)
    assert torch.equal(restored.cand, probe_a.cand)
    assert torch.equal(restored.streams.tokens, probe_a.streams.tokens)
    assert torch.equal(restored.round_index, probe_a.round_index)

    # Device transfer helper keeps obs_mask_prob=None and match_groups=None.
    moved = decision_inputs_on_device(restored, "cpu")
    assert moved.obs_mask_prob is None and moved.match_groups is None
    fp_a = behavior_fingerprint(actor, probe_a)
    fp_restored = behavior_fingerprint(actor, moved)
    np.testing.assert_allclose(fp_a, fp_restored, atol=1e-6)


def test_probe_respects_snapshot_own_obs_mask_and_sequence_decoder() -> None:
    cfg = HistoryPolicyConfig(
        **BASE,
        private_decoder_layers=2,
        private_query_slots=2,
        private_obs_mode="private_only",
    )
    actor_full_mask_off, _ = fresh_player(cfg, 23)
    with torch.no_grad():
        actor_full_mask_off.private[0].weight.mul_(4.0)
        actor_full_mask_off.fusion[-1].weight.mul_(30.0)
    actor_full_mask_off.obs_mask_prob.fill_(0.0)
    actor_masked = copy.deepcopy(actor_full_mask_off)
    actor_masked.obs_mask_prob.fill_(1.0)

    probe, meta = build_behavior_probe(actor_full_mask_off, rows=24, seed=77, num_envs=4, max_steps=384)
    assert probe.obs_mask_prob is None
    assert meta["decisions"] == 24

    fp_off = behavior_fingerprint(actor_full_mask_off, probe)
    fp_on = behavior_fingerprint(actor_masked, probe)
    # Since probe.obs_mask_prob is None, each snapshot applies its own obs_mask_prob buffer.
    tv = mean_behavior_tv(fp_off, fp_on, probe.decisions)
    assert tv > 1e-4

    # Sequence-decoder actor works end-to-end with HistoryPopulation(nearest_pair) and rebind.
    pool = HistoryPopulation(
        actor_full_mask_off,
        "seq",
        9,
        recent=2,
        archive_every=1,
        archive_size=3,
        archive_filter="nearest_pair",
        archive_min_tv=1e-5,
    )
    pool.set_behavior_probe(probe, meta)
    assert not pool.needs_probe_rebind(expected_decisions=24, expected_seed=77)
    assert pool.needs_probe_rebind(expected_decisions=32, expected_seed=77)
    assert pool.needs_probe_rebind(expected_decisions=24, expected_seed=88)

    pool.snapshot(1, behavior=fp_off, behavior_decisions=probe.decisions)
    pool.actor = actor_masked
    pool.snapshot(2, behavior=fp_on, behavior_decisions=probe.decisions)
    assert len(pool.archive) == 2
    assert pool.archive_diversity_stats()["archive_mean_pair_tv"] > 1e-3


def test_state_dict_legacy_rebind_and_adopt_state_dict() -> None:
    actor, _ = fresh_player(HistoryPolicyConfig(**BASE), 31)
    probe1, meta1 = build_behavior_probe(actor, rows=16, seed=101, num_envs=2, max_steps=256)
    probe2, meta2 = build_behavior_probe(actor, rows=20, seed=202, num_envs=2, max_steps=256)

    pool = HistoryPopulation(
        actor,
        "adopt",
        5,
        recent=2,
        archive_every=1,
        archive_size=4,
        archive_filter="nearest_pair",
        archive_min_tv=0.05,
        archive_max_age_updates=0,
    )
    pool.set_behavior_probe(probe1, meta1)
    id1 = pool.snapshot(10, behavior=_prob_vec([[1.0, 0.0]] * 16), behavior_decisions=16)
    id2 = pool.snapshot(20, behavior=_prob_vec([[0.0, 1.0]] * 16), behavior_decisions=16)
    id3 = pool.snapshot(60, behavior=_prob_vec([[0.5, 0.5]] * 16), behavior_decisions=16)
    assert pool.archive == [id1, id2, id3]

    # Exact state_dict / load_state_dict round-trip preserves probe + metadata + behaviors.
    saved = pool.state_dict()
    reloaded = HistoryPopulation(
        actor,
        "adopt",
        5,
        recent=2,
        archive_every=1,
        archive_size=4,
        archive_filter="nearest_pair",
        archive_min_tv=0.05,
        archive_max_age_updates=0,
    )
    reloaded.load_state_dict(saved)
    assert reloaded.behavior_probe is not None
    assert torch.equal(reloaded.behavior_probe.obs, probe1.obs)
    assert reloaded.behavior_probe_meta == meta1
    assert set(reloaded.behaviors) == {id1, id2, id3}

    # Missing fingerprint when behaviors or probe present must raise.
    bad_state = copy.deepcopy(saved)
    bad_state["behaviors"].pop(id2)
    with pytest.raises(ValueError, match="missing behavior fingerprint"):
        reloaded.load_state_dict(bad_state)

    # Pre-rebind uninitialized state (empty behaviors and None probe, non-int behavior_decisions) loads cleanly.
    uninit = copy.deepcopy(saved)
    uninit["behaviors"] = {}
    uninit["behavior_probe"] = None
    uninit["behavior_decisions"] = {}
    reloaded.load_state_dict(uninit)
    assert reloaded.behavior_decisions == 0
    assert reloaded.needs_probe_rebind() is True

    # Rebind computes real fingerprints for all archive members and updates probe + meta.
    reloaded.rebind_behavior_probe(probe2, frozen_probe=probe2, meta=meta2)
    assert not reloaded.needs_probe_rebind(expected_decisions=20, expected_seed=202)
    assert set(reloaded.behaviors) == {id1, id2, id3}
    assert reloaded.behavior_decisions == 20

    # Atomic failure safety on rebind if an archive id cannot resolve.
    broken = copy.deepcopy(reloaded)
    broken.models.pop(id3)
    with pytest.raises(ValueError, match="unknown Transformer snapshot"):
        broken.rebind_behavior_probe(probe1, frozen_probe=probe1, meta=meta1)
    assert broken.behavior_decisions == 20  # unmutated on failure

    # adopt_state_dict: age-filter + cap size + preserve matching probe vs clear mismatched probe.
    saved2 = reloaded.state_dict()
    adopted_match = HistoryPopulation(
        actor,
        "adopt",
        9,
        recent=2,
        archive_every=1,
        archive_size=2,
        archive_filter="nearest_pair",
        archive_min_tv=0.05,
        archive_max_age_updates=45,
    )
    adopted_match.adopt_state_dict(
        saved2, current_update=60, expected_probe_rows=20, expected_probe_seed=202
    )
    # id1 (update=10 -> age 50 > 45) filtered out; remaining [id2, id3] fit archive_size=2.
    assert adopted_match.archive == [id2, id3]
    assert set(adopted_match.behaviors) == {id2, id3}
    assert adopted_match.behavior_probe is not None
    assert not adopted_match.needs_probe_rebind(expected_decisions=20, expected_seed=202)

    # Mismatched expected_probe_rows or switching to "thin" clears probe & behaviors.
    adopted_mismatch = HistoryPopulation(
        actor,
        "adopt",
        9,
        recent=2,
        archive_every=1,
        archive_size=2,
        archive_filter="nearest_pair",
        archive_min_tv=0.05,
    )
    adopted_mismatch.adopt_state_dict(saved2, expected_probe_rows=99, expected_probe_seed=202)
    assert adopted_mismatch.archive == [id2, id3]  # capped to newest 2
    assert adopted_mismatch.behavior_probe is None
    assert adopted_mismatch.behavior_probe_meta is None
    assert adopted_mismatch.behaviors == {}
    assert adopted_mismatch.behavior_decisions == 0
    assert adopted_mismatch.needs_probe_rebind() is True

    adopted_thin = HistoryPopulation(actor, "adopt", 9, recent=2, archive_every=0, archive_filter="thin")
    adopted_thin.adopt_state_dict(saved2)
    assert adopted_thin.archive == []
    assert adopted_thin.behavior_probe is None
    assert adopted_thin.behaviors == {}

    # Seed check tolerance when behavior_probe_meta lacks "seed".
    no_seed_meta = copy.deepcopy(adopted_match)
    no_seed_meta.set_behavior_probe(probe2, meta={"decisions": 20})
    assert not no_seed_meta.needs_probe_rebind(expected_decisions=20, expected_seed=999)
