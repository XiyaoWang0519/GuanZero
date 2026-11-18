"""Migration from legacy private attention to PrivateCrossDecoder."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from train import history_sequence_migrate
from train.history_model import (
    HistoryPolicyConfig,
    fresh_player,
    load_history_checkpoint,
)
from train.history_population import HistoryPopulation, weights_digest
from train.history_ppo import HistoryPPOConfig, HistoryTrainer

BASE = dict(width=32, layers=1, heads=4)
SAMPLE_KWARGS = dict(envs=3, steps=180, matches=3, minibatch_matches=3, rows_per_match=20)


@pytest.fixture(scope="module")
def plain_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("plain_src")
    cfg = HistoryPPOConfig(
        **BASE,
        num_envs=4,
        steps_per_update=128,
        seed=5,
        updates=2,
        epochs=1,
        minibatch_matches=2,
        snapshot_updates=0,
    )
    trainer = HistoryTrainer(cfg, root / "run")
    trainer.run()
    path = root / "plain.pt"
    trainer.save(path)
    return path


@pytest.fixture(scope="module")
def rich_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("rich_src")
    cfg = HistoryPPOConfig(
        **BASE,
        response_mode="auxiliary",
        aux_heads="next,belief,outcome",
        num_envs=4,
        steps_per_update=128,
        seed=7,
        updates=3,
        epochs=1,
        minibatch_matches=2,
        snapshot_updates=1,
        snapshot_probability=1.0,
        population_recent=2,
        population_archive_every=2,
        population_archive_size=4,
        population_archive_share=0.5,
    )
    trainer = HistoryTrainer(cfg, root / "run")
    trainer.run()
    path = root / "rich.pt"
    trainer.save(path)
    return path


def test_validate_source_rejects_unsupported():
    for overrides in (
        dict(window=8),
        dict(style_input=True),
        dict(private_decoder_layers=2),
        dict(private_fusion="split"),
        dict(private_qk_norm="qk"),
        dict(private_qk_norm="query"),
    ):
        actor, _ = fresh_player(HistoryPolicyConfig(**BASE, **overrides), 0)
        with pytest.raises(ValueError):
            history_sequence_migrate.validate_source(actor)


def test_calibrate_reader_bias_empty_fallback():
    actor, _ = fresh_player(HistoryPolicyConfig(**BASE), 0)
    r_bar, stats = history_sequence_migrate.calibrate_reader_bias(actor, [])
    assert r_bar.shape == (BASE["width"],)
    assert torch.equal(r_bar, torch.zeros(BASE["width"]))
    assert stats["rows"] == 0
    assert stats["fallback_zeros"] is True


def _assert_shared_weights_and_adam_preserved(src_path, out_path, obs_mode: str):
    source, src_critic, src_payload = load_history_checkpoint(src_path, "cpu")
    converted, conv_critic, out_payload = load_history_checkpoint(out_path, "cpu")

    # Shared weights are copied bitwise; legacy attention projections are absent.
    src_state = source.state_dict()
    conv_state = converted.state_dict()
    for key in history_sequence_migrate.LEGACY_ATTENTION_KEYS:
        assert key in src_state
        assert key not in conv_state
    for key, value in src_state.items():
        if key not in history_sequence_migrate.LEGACY_ATTENTION_KEYS:
            assert torch.equal(conv_state[key], value), key

    # Critic weights are preserved bitwise.
    for key, value in src_critic.state_dict().items():
        assert torch.equal(conv_critic.state_dict()[key], value), key

    if obs_mode == "private_only":
        assert converted.obs_mask_prob is not None
    else:
        assert converted.obs_mask_prob is None
        assert "obs_mask_prob" not in conv_state

    # Actor Adam optimizer moments are remapped by parameter name; decoder params omitted.
    src_opt = src_payload["optimizer"]["actor"]
    out_opt = out_payload["optimizer"]["actor"]
    conv_params = list(converted.named_parameters())
    assert out_opt["param_groups"][0]["params"] == list(range(len(conv_params)))

    src_names = [name for name, _ in source.named_parameters()]
    src_idx = {name: i for i, name in enumerate(src_names)}
    for new_idx, (name, _) in enumerate(conv_params):
        if name.startswith("private_decoder."):
            assert new_idx not in out_opt["state"], name
        else:
            old_idx = src_idx[name]
            if old_idx in src_opt["state"]:
                assert new_idx in out_opt["state"], name
                old_entry = src_opt["state"][old_idx]
                new_entry = out_opt["state"][new_idx]
                for field in ("step", "exp_avg", "exp_avg_sq"):
                    assert torch.equal(new_entry[field], old_entry[field]), (name, field)

    # Critic optimizer copied intact.
    src_copt = src_payload["optimizer"]["critic"]
    out_copt = out_payload["optimizer"]["critic"]
    assert out_copt["param_groups"] == src_copt["param_groups"]
    assert set(out_copt["state"]) == set(src_copt["state"])
    for idx in src_copt["state"]:
        for field in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(out_copt["state"][idx][field], src_copt["state"][idx][field])


def _assert_diagnostics_healthy(migration: dict, decoder_layers: int, heads: int):
    assert migration["heldout"]["decisions_seen"] > 0
    assert migration["heldout"]["tv_mean"] < 0.05
    assert migration["heldout"]["tv_median"] < 0.05

    attn = migration["attention"]
    assert attn["layers"] == decoder_layers
    assert attn["heads"] == heads
    assert attn["state_error_max"] < 1e-5
    assert len(attn["one_hot_share"]) == decoder_layers
    assert len(attn["entropy_median"]) == decoder_layers
    for l_idx in range(decoder_layers):
        assert len(attn["one_hot_share"][l_idx]) == heads
        assert all(val == 0.0 for val in attn["one_hot_share"][l_idx])
        assert all(np.isfinite(val) and val > 0.0 for val in attn["entropy_median"][l_idx])

    grads = migration["gradients"]
    assert grads["private_decoder"]["all_finite_positive"] is True
    assert grads["private_decoder"]["zero_params"] == []
    assert grads["private_decoder"]["nonfinite_params"] == []
    assert grads["stream"]["all_finite_positive"] is True
    assert grads["stream"]["zero_params"] == []
    assert grads["stream"]["nonfinite_params"] == []


def test_migrate_plain_checkpoint_and_trainer_update(plain_source, tmp_path):
    out_path = tmp_path / "plain-seq.pt"
    migration = history_sequence_migrate.migrate(
        plain_source,
        out_path,
        decoder_layers=2,
        query_slots=4,
        obs_mode="private_only",
        initial_mask_prob=0.0,
        sample_seed=11,
        heldout_seed=12,
        **SAMPLE_KWARGS,
    )
    _assert_shared_weights_and_adam_preserved(plain_source, out_path, "private_only")
    _assert_diagnostics_healthy(migration, decoder_layers=2, heads=BASE["heads"])

    converted, _, out_payload = load_history_checkpoint(out_path, "cpu")
    assert converted.config.private_decoder_layers == 2
    assert converted.config.private_query_slots == 4
    assert converted.config.private_obs_mode == "private_only"
    assert float(converted.obs_mask_prob) == 0.0
    assert out_payload["sequence_migration"] == migration

    # Continue training with HistoryTrainer(..., init_checkpoint=out_path).
    ppo_cfg = HistoryPPOConfig(
        **BASE,
        private_decoder_layers=2,
        private_query_slots=4,
        private_obs_mode="private_only",
        obs_mask_warmup_updates=2,
        init_checkpoint=str(out_path),
        num_envs=4,
        steps_per_update=128,
        seed=21,
        updates=2,
        epochs=1,
        minibatch_matches=2,
        snapshot_updates=0,
    )

    # Check exact ratio == 1.0 on epoch 0 minibatch 0 before stepping.
    probe_trainer = HistoryTrainer(ppo_cfg, tmp_path / "ppo_plain_probe")
    probe_trainer.collect()
    probe_values = probe_trainer.refresh_values()
    probe_samples = probe_trainer.buffer.finalize(probe_values, ppo_cfg.gamma, ppo_cfg.gae_lambda)
    assert probe_samples > 0
    rows = next(probe_trainer.minibatches())
    mb_stats = probe_trainer.minibatch_loss(rows)
    assert float(mb_stats["ratio_deviation"]) < 1e-6

    trainer = HistoryTrainer(ppo_cfg, tmp_path / "ppo_plain")
    # Remapped Adam moments are intact before the first step; decoder params start lazily.
    pre_state = trainer.actor_optimizer.state_dict()["state"]
    assert len(pre_state) == len(out_payload["optimizer"]["actor"]["state"])

    # Full update succeeds with finite gradients and initializes decoder Adam slots.
    line = trainer.update()
    assert line["minibatches"] > 0
    assert np.isfinite(line["policy_loss"])
    assert np.isfinite(line["actor_grad_norm"]) and line["actor_grad_norm"] > 0
    assert np.isfinite(line["encoder_grad_norm"]) and line["encoder_grad_norm"] > 0
    assert line["obs_mask_prob"] == pytest.approx(0.0)
    post_state = trainer.actor_optimizer.state_dict()["state"]
    assert len(post_state) == len(list(trainer.actor.parameters()))

    line2 = trainer.update()
    assert line2["obs_mask_prob"] == pytest.approx(0.5)
    assert np.isfinite(line2["actor_grad_norm"])


def test_migrate_rich_checkpoint_with_aux_response_and_population(rich_source, tmp_path):
    out_path = tmp_path / "rich-seq.pt"
    report_path = tmp_path / "rich-report.json"

    # Exercise CLI main() as well as full aux_heads + response_mode="auxiliary" + population.
    rc = history_sequence_migrate.main([
        str(rich_source),
        str(out_path),
        "--report",
        str(report_path),
        "--decoder-layers",
        "3",
        "--query-slots",
        "4",
        "--obs-mode",
        "private_only",
        "--residual-scale",
        "0.02",
        "--initial-mask-prob",
        "0.0",
        "--sample-seed",
        "13",
        "--heldout-seed",
        "14",
        "--matches",
        "3",
        "--envs",
        "3",
        "--steps",
        "180",
        "--minibatch-matches",
        "3",
        "--rows-per-match",
        "20",
        "--population",
        "migrate",
    ])
    assert rc == 0
    report = json.loads(report_path.read_text())
    _assert_shared_weights_and_adam_preserved(rich_source, out_path, "private_only")
    _assert_diagnostics_healthy(report, decoder_layers=3, heads=BASE["heads"])

    converted, _, out_payload = load_history_checkpoint(out_path, "cpu")
    assert converted.aux_head_names == ("next", "belief", "outcome")
    assert converted.config.response_mode == "auxiliary"
    assert out_payload["config"]["private_decoder_layers"] == 3
    assert out_payload["config"]["private_query_slots"] == 4
    assert out_payload["config"]["private_obs_mode"] == "private_only"

    # Validate migrated population snapshots.
    pop_state = out_payload["population"]
    assert pop_state["snapshots"]
    aux_names = converted.auxiliary_parameter_names()
    for snap_id, rec in pop_state["snapshots"].items():
        model_dict = rec["model"]
        assert "obs_mask_prob" in model_dict
        assert float(model_dict["obs_mask_prob"]) == 0.0
        assert not (set(model_dict) & aux_names)
        for key in history_sequence_migrate.LEGACY_ATTENTION_KEYS:
            assert key not in model_dict
        for k, v in converted.state_dict().items():
            if k.startswith("private_decoder."):
                assert torch.equal(model_dict[k], v), (snap_id, k)
        assert rec["metadata"]["sha256"] == weights_digest(model_dict)

    pool = HistoryPopulation(
        converted,
        pop_state["lineage"],
        0,
        recent=pop_state["recent"],
        snapshot_probability=pop_state["probability"],
        archive_every=pop_state.get("archive_every", 0),
        archive_size=pop_state.get("archive_size", 16),
        archive_share=pop_state.get("archive_share", 0.5),
        archive_filter=pop_state.get("archive_filter", "thin"),
        archive_min_tv=float(pop_state.get("archive_min_tv", 0.05)),
        archive_max_age_updates=int(pop_state.get("archive_max_age_updates", 0)),
    )
    pool.load_state_dict(pop_state)
    assert len(pool.models) == len(pop_state["snapshots"])

    # Continue training with HistoryTrainer(..., init_checkpoint=out_path) WITHOUT
    # explicitly passing --population-archive-every (tests archive_every compatibility).
    ppo_cfg = HistoryPPOConfig(
        **BASE,
        response_mode="auxiliary",
        aux_heads="next,belief,outcome",
        private_decoder_layers=3,
        private_query_slots=4,
        private_obs_mode="private_only",
        obs_mask_warmup_updates=2,
        init_checkpoint=str(out_path),
        rollout_kv_cache=True,
        rollout_paged_cache=True,
        rollout_page_span=True,
        rollout_batched_attention=True,
        rollout_wide_projection=True,
        learner_batched_attention=True,
        learner_length_groups=2,
        num_envs=4,
        steps_per_update=128,
        seed=31,
        updates=3,
        epochs=1,
        minibatch_matches=2,
    )
    trainer = HistoryTrainer(ppo_cfg, tmp_path / "ppo_rich")
    assert len(trainer.population.models) == len(pop_state["snapshots"])
    for snap_actor in trainer.population.models.values():
        assert float(snap_actor.obs_mask_prob) == 0.0

    # Check exact ratio == 1.0 on epoch 0 minibatch 0 for the rich checkpoint too.
    rich_probe = HistoryTrainer(ppo_cfg, tmp_path / "ppo_rich_probe")
    rich_probe.collect()
    rich_values = rich_probe.refresh_values()
    rich_samples = rich_probe.buffer.finalize(rich_values, ppo_cfg.gamma, ppo_cfg.gae_lambda)
    assert rich_samples > 0
    rich_rows = next(rich_probe.minibatches())
    rich_mb = rich_probe.minibatch_loss(rich_rows)
    assert float(rich_mb["ratio_deviation"]) < 1e-6

    line1 = trainer.update()
    assert line1["minibatches"] > 0
    assert np.isfinite(line1["approx_kl"])
    assert np.isfinite(line1["actor_grad_norm"]) and line1["actor_grad_norm"] > 0
    assert np.isfinite(line1["aux_loss"])
    assert line1["obs_mask_prob"] == pytest.approx(0.0)

    line2 = trainer.update()
    assert line2["obs_mask_prob"] == pytest.approx(0.5)
    assert np.isfinite(line2["actor_grad_norm"])


def test_migrate_population_reset_and_full_obs_mode(rich_source, tmp_path):
    out_path = tmp_path / "reset-full.pt"
    migration = history_sequence_migrate.migrate(
        rich_source,
        out_path,
        decoder_layers=1,
        query_slots=1,
        obs_mode="full",
        population="reset",
        sample_seed=15,
        heldout_seed=16,
        **SAMPLE_KWARGS,
    )
    _assert_shared_weights_and_adam_preserved(rich_source, out_path, "full")
    _assert_diagnostics_healthy(migration, decoder_layers=1, heads=BASE["heads"])

    _, _, out_payload = load_history_checkpoint(out_path, "cpu")
    assert "population" not in out_payload
    assert migration["population"]["mode"] == "reset"
    assert migration["population"]["migrated"] is False

    trainer = HistoryTrainer(
        HistoryPPOConfig(
            **BASE,
            response_mode="auxiliary",
            aux_heads="next,belief,outcome",
            private_decoder_layers=1,
            private_query_slots=1,
            private_obs_mode="full",
            init_checkpoint=str(out_path),
            num_envs=4,
            steps_per_update=128,
            seed=41,
            updates=1,
            epochs=1,
            minibatch_matches=2,
            snapshot_updates=0,
        ),
        tmp_path / "ppo_reset",
    )
    assert len(trainer.population.models) == 0
    line = trainer.update()
    assert np.isfinite(line["policy_loss"]) and np.isfinite(line["actor_grad_norm"])


def test_migrate_u195450_shaped_extra_population_and_config_keys(rich_source, tmp_path):
    """Extra keys in config and population (like u195450.pt) migrate cleanly."""
    from train.history_model import save_history_checkpoint

    _, _, payload = load_history_checkpoint(rich_source, "cpu")
    payload["config"].update({
        "next_coef": 0.1,
        "next_coef_final": 0.0,
        "population_archive_filter": "strength_behavior",
    })
    stale_behaviors = {
        int(sid): np.full(12, 0.25, dtype=np.float32)
        for sid in payload["population"]["snapshots"]
    }
    payload["population"].update({
        "archive_filter": "strength_behavior",
        "archive_min_tv": 0.05,
        "archive_rejects": 0,
        "archive_replacements": 0,
        "behavior_decisions": 12,
        "behaviors": stale_behaviors,
        "behavior_probe": {"stale": True},
        "behavior_probe_meta": {"seed": 99},
    })
    src_copy = tmp_path / "u195450-like.pt"
    save_history_checkpoint(src_copy, payload)

    out_path = tmp_path / "u195450-like-seq.pt"
    migration = history_sequence_migrate.migrate(
        src_copy,
        out_path,
        decoder_layers=2,
        query_slots=4,
        obs_mode="private_only",
        sample_seed=19,
        heldout_seed=20,
        **SAMPLE_KWARGS,
    )
    assert migration["population"]["migrated"] is True
    assert migration["population"]["archive_filter"] == "strength_behavior"
    assert migration["population"]["archive_max_age_updates"] == 0

    converted, _, out_payload = load_history_checkpoint(out_path, "cpu")
    pop_state = out_payload["population"]
    assert pop_state["behaviors"] == {}
    assert pop_state["behavior_decisions"] == 0
    assert pop_state["behavior_probe"] is None
    assert pop_state["behavior_probe_meta"] is None

    pool = HistoryPopulation(
        converted,
        pop_state["lineage"],
        0,
        recent=pop_state["recent"],
        snapshot_probability=pop_state["probability"],
        archive_every=pop_state.get("archive_every", 0),
        archive_size=pop_state.get("archive_size", 16),
        archive_share=pop_state.get("archive_share", 0.5),
        archive_filter=pop_state.get("archive_filter", "thin"),
        archive_min_tv=float(pop_state.get("archive_min_tv", 0.05)),
        archive_max_age_updates=int(pop_state.get("archive_max_age_updates", 0)),
    )
    pool.load_state_dict(pop_state)

    trainer = HistoryTrainer(
        HistoryPPOConfig(
            **BASE,
            response_mode="auxiliary",
            aux_heads="next,belief,outcome",
            private_decoder_layers=2,
            private_query_slots=4,
            private_obs_mode="private_only",
            obs_mask_warmup_updates=1,
            init_checkpoint=str(out_path),
            num_envs=4,
            steps_per_update=128,
            seed=51,
            updates=1,
            epochs=1,
            minibatch_matches=2,
        ),
        tmp_path / "ppo_u195450_like",
    )
    assert len(trainer.population.models) == len(payload["population"]["snapshots"])
    line = trainer.update()
    assert np.isfinite(line["actor_grad_norm"]) and line["obs_mask_prob"] == pytest.approx(0.0)


def test_migrate_cli_archive_filter_nearest_pair_and_max_age(rich_source, tmp_path):
    out_path = tmp_path / "nearest-pair.pt"
    report_path = tmp_path / "nearest-pair-report.json"

    rc = history_sequence_migrate.main([
        str(rich_source),
        str(out_path),
        "--report",
        str(report_path),
        "--decoder-layers",
        "2",
        "--query-slots",
        "4",
        "--obs-mode",
        "private_only",
        "--archive-filter",
        "nearest_pair",
        "--archive-max-age-updates",
        "40000",
        "--sample-seed",
        "23",
        "--heldout-seed",
        "24",
        "--matches",
        "3",
        "--envs",
        "3",
        "--steps",
        "180",
        "--minibatch-matches",
        "3",
        "--rows-per-match",
        "20",
    ])
    assert rc == 0
    report = json.loads(report_path.read_text())
    assert report["population"]["archive_filter"] == "nearest_pair"
    assert report["population"]["archive_max_age_updates"] == 40000

    converted, _, out_payload = load_history_checkpoint(out_path, "cpu")
    pop_state = out_payload["population"]
    assert pop_state["archive_filter"] == "nearest_pair"
    assert pop_state["archive_max_age_updates"] == 40000
    assert pop_state["behaviors"] == {}
    assert pop_state["behavior_decisions"] == 0
    assert pop_state["behavior_probe"] is None
    assert pop_state["behavior_probe_meta"] is None
    assert out_payload["config"]["population_archive_filter"] == "nearest_pair"
    assert out_payload["config"]["population_archive_max_age_updates"] == 40000

    pool = HistoryPopulation(
        converted,
        pop_state["lineage"],
        0,
        recent=pop_state["recent"],
        snapshot_probability=pop_state["probability"],
        archive_every=pop_state.get("archive_every", 0),
        archive_size=pop_state.get("archive_size", 16),
        archive_share=pop_state.get("archive_share", 0.5),
        archive_filter="nearest_pair",
        archive_min_tv=float(pop_state.get("archive_min_tv", 0.05)),
        archive_max_age_updates=40000,
    )
    pool.load_state_dict(pop_state)
    assert pool.archive_filter == "nearest_pair"
    assert pool.archive_max_age_updates == 40000
