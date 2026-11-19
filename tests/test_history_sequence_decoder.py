"""Integration tests for sequence cross-decoder, obs masking, caches, PPO, and eval."""
from dataclasses import asdict, replace
import json

import gd
import numpy as np
import pytest
import torch

from eval.botzone.numpy_actor import NumpyHistoryActor
from eval.history_policy import HistoryPolicy
from eval.policies import load_policy
from train.history_config import RESUME_OVERRIDES
from train.history_decoder import PRIVATE_OBS_KEEP_SLICES, PRIVATE_OBS_MODES
from train.history_inference import BatchedHistoryCache, ForkedHistoryCache
from train.history_model import (
    DecisionInputs,
    HistoryActor,
    HistoryPolicyConfig,
    PublicStream,
    StreamBatch,
    checkpoint_payload,
    config_record,
    extend_actor,
    fresh_player,
    load_history_checkpoint,
    save_history_checkpoint,
)
from train.history_paged_cache import KVPagePool, PagedHistoryCache
from train.history_population import HistoryPopulation
from train.history_ppo import (
    HistoryPPOConfig,
    HistoryTrainer,
    build_parser,
    config_from_args,
    parse_resume_overrides,
)
from train.history_rollout import HistoryCollector, MatchEventStore, SequenceRolloutBuffer
from train.history_snapshot_batch import SnapshotHeads, validate_actor
from train.logs import TOKEN_DIM

BASE = dict(width=32, layers=1, heads=4)
SEQ = dict(
    width=32,
    layers=1,
    heads=4,
    private_decoder_layers=2,
    private_query_slots=4,
    private_obs_mode="private_only",
)


def _append_events(stream: PublicStream, count: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for i in range(count):
        token = np.zeros(TOKEN_DIM, dtype=np.uint8)
        token[(i + seed) % 4] = 1
        token[4 + int(rng.integers(0, 108))] = 1
        token[158 + (27 - (i % 27))] = 1
        stream.append_token(token, i // 12, int(gd.Phase.Play))


def _collect_multiround(actor: HistoryActor, *, seed: int = 7, num_envs: int = 4, steps: int = 360):
    env = gd.VecEnv(
        num_envs=num_envs,
        num_threads=1,
        seed=seed,
        log_public_actions=True,
        log_env_limit=num_envs,
    )
    store, buffer = MatchEventStore(), SequenceRolloutBuffer()
    generator = torch.Generator().manual_seed(seed + 1)
    collector = HistoryCollector(env, actor, store, buffer, generator)
    stats = collector.collect(steps)
    return store, buffer, stats


def _per_decision_inputs(inputs: DecisionInputs) -> DecisionInputs:
    s, n = inputs.streams, inputs.decisions
    return DecisionInputs(
        streams=StreamBatch(
            s.tokens[inputs.match_index],
            s.rounds[inputs.match_index],
            s.phases[inputs.match_index],
            s.lengths[inputs.match_index],
        ),
        match_index=torch.arange(n, dtype=torch.long, device=inputs.obs.device),
        prefix=inputs.prefix,
        obs=inputs.obs,
        seat=inputs.seat,
        cand=inputs.cand,
        offsets=inputs.offsets,
        one_decision_per_stream=True,
        round_index=inputs.round_index,
    )


def test_default_config_record_and_actor_completely_unchanged():
    cfg = HistoryPolicyConfig(**BASE)
    assert cfg.private_decoder_layers == 0
    assert cfg.private_query_slots == 1
    assert cfg.private_obs_mode == "full"
    rec = config_record(cfg)
    assert "private_decoder_layers" not in rec
    assert "private_query_slots" not in rec
    assert "private_obs_mode" not in rec

    actor, critic = fresh_player(cfg, 0)
    assert actor.private_decoder is None
    assert actor.obs_mask_prob is None
    assert actor.q_proj is not None and actor.kv_proj is not None and actor.out_proj is not None
    assert "obs_mask_prob" not in actor.state_dict()
    assert not any(k.startswith("private_decoder.") for k in actor.state_dict())

    payload = checkpoint_payload(actor, critic, lineage="default", seed=0)
    assert "private_decoder_layers" not in payload["model_config"]
    assert "private_query_slots" not in payload["model_config"]
    assert "private_obs_mode" not in payload["model_config"]


def test_config_and_path_validations(tmp_path):
    for bad in (
        dict(private_decoder_layers=-1),
        dict(private_query_slots=0),
        dict(private_obs_mode="bogus"),
        dict(private_query_slots=2, private_decoder_layers=0),
        dict(private_decoder_layers=1, window=8),
        dict(private_decoder_layers=1, private_qk_norm="qk"),
        dict(private_decoder_layers=1, private_fusion="split"),
    ):
        with pytest.raises(ValueError):
            HistoryPolicyConfig(**BASE, **bad)

    for bad_ppo in (
        dict(obs_mask_warmup_updates=-1),
        dict(obs_mask_warmup_updates=2, private_obs_mode="full"),
        dict(private_decoder_layers=1, rollout_kv_cache=True, rollout_private_graphs=True),
        dict(private_obs_mode="private_only", rollout_kv_cache=True, rollout_private_graphs=True),
        dict(private_obs_mode="card_counter", rollout_kv_cache=True, rollout_private_graphs=True),
        dict(init_checkpoint="a.pt", habit_pack="b.pt"),
        dict(init_checkpoint="a.pt", habit_init="b.pt"),
    ):
        with pytest.raises(ValueError):
            HistoryPPOConfig(**BASE, **bad_ppo)
    for ok_ppo in (
        dict(private_decoder_layers=1, batch_snapshot_policies=True),
        dict(private_obs_mode="private_only", batch_snapshot_policies=True),
        dict(private_obs_mode="card_counter", obs_mask_warmup_updates=4, batch_snapshot_policies=True),
        dict(private_decoder_layers=1, batch_snapshot_policies_schedule="1:on"),
        dict(
            private_decoder_layers=2,
            private_query_slots=2,
            private_obs_mode="private_only",
            batch_snapshot_policies=True,
        ),
    ):
        assert HistoryPPOConfig(**BASE, **ok_ppo)

    assert "obs_mask_warmup_updates" in RESUME_OVERRIDES
    assert parse_resume_overrides(["obs_mask_warmup_updates=4"]) == {"obs_mask_warmup_updates": 4}
    for frozen_key in ("private_decoder_layers=1", "private_query_slots=2", "private_obs_mode=full"):
        with pytest.raises(ValueError):
            parse_resume_overrides([frozen_key])

    seq_actor, _ = fresh_player(HistoryPolicyConfig(**SEQ), 3)
    validate_actor(seq_actor)
    assert SnapshotHeads().slot(1, seq_actor) == 0
    with pytest.raises(ValueError):
        NumpyHistoryActor(
            {},
            dict(width=32, heads=4, layers=1, max_rounds=16, private_decoder_layers=1),
        )
    with pytest.raises(ValueError):
        NumpyHistoryActor(
            {},
            dict(width=32, heads=4, layers=1, max_rounds=16, private_obs_mode="private_only"),
        )
    with pytest.raises(ValueError):
        NumpyHistoryActor(
            {},
            dict(width=32, heads=4, layers=1, max_rounds=16, private_obs_mode="card_counter"),
        )
    env = gd.VecEnv(num_envs=1, num_threads=1, seed=1, log_public_actions=True, log_env_limit=1)
    collector = HistoryCollector(
        env,
        seq_actor,
        MatchEventStore(),
        SequenceRolloutBuffer(),
        batch_snapshot_policies=True,
    )
    assert collector.batch_snapshot_policies is True
    with pytest.raises(ValueError):
        HistoryCollector(
            env,
            seq_actor,
            MatchEventStore(),
            SequenceRolloutBuffer(),
            kv_cache=True,
            private_graphs=True,
            device="cuda",
        )
    obs_only_actor, _ = fresh_player(
        HistoryPolicyConfig(**BASE, private_obs_mode="private_only"), 3
    )
    with pytest.raises(ValueError):
        HistoryCollector(
            env,
            obs_only_actor,
            MatchEventStore(),
            SequenceRolloutBuffer(),
            kv_cache=True,
            private_graphs=True,
            device="cuda",
        )

    # CLI parser defaults & flags
    default_cfg = config_from_args(build_parser().parse_args(["--output", str(tmp_path)]))
    assert default_cfg == HistoryPPOConfig()
    parsed = config_from_args(
        build_parser().parse_args(
            [
                "--output",
                str(tmp_path),
                "--private-decoder-layers",
                "2",
                "--private-query-slots",
                "4",
                "--private-obs-mode",
                "private_only",
                "--obs-mask-warmup-updates",
                "3",
                "--init-checkpoint",
                "src.pt",
            ]
        )
    )
    assert parsed.private_decoder_layers == 2
    assert parsed.private_query_slots == 4
    assert parsed.private_obs_mode == "private_only"
    assert parsed.obs_mask_warmup_updates == 3
    assert parsed.init_checkpoint == "src.pt"
    assert parsed.policy_config() == HistoryPolicyConfig(
        width=64,
        layers=2,
        heads=4,
        private_decoder_layers=2,
        private_query_slots=4,
        private_obs_mode="private_only",
    )
    cc_parsed = config_from_args(
        build_parser().parse_args(
            [
                "--output",
                str(tmp_path),
                "--private-obs-mode",
                "card_counter",
                "--obs-mask-warmup-updates",
                "2",
            ]
        )
    )
    assert cc_parsed.private_obs_mode == "card_counter"
    assert cc_parsed.obs_mask_warmup_updates == 2


def test_private_only_obs_mask_isolates_public_obs_and_keeps_critic_privileged():
    cfg = HistoryPolicyConfig(**SEQ)
    actor, critic = fresh_player(cfg, 4)
    actor.eval()
    store, buffer, _ = _collect_multiround(actor, seed=5, num_envs=3, steps=120)
    rows = np.arange(min(len(buffer), 48))
    batch = buffer.training_batch(rows, store, "cpu", fields=("hidden",))
    inputs = batch.inputs

    keep_mask = np.zeros(int(gd.OBS_DIM), dtype=bool)
    for start, end in PRIVATE_OBS_KEEP_SLICES:
        keep_mask[start:end] = True
    assert int(keep_mask.sum()) == 171
    assert int((~keep_mask).sum()) == 1678

    rng = np.random.default_rng(11)
    perturbed_obs = inputs.obs.clone()
    pub_idx = torch.as_tensor(np.flatnonzero(~keep_mask), dtype=torch.long)
    perturbed_obs[:, pub_idx] = torch.as_tensor(
        rng.integers(0, 2, size=(len(rows), len(pub_idx)), dtype=np.uint8)
    )
    perturbed_inputs = replace(inputs, obs=perturbed_obs)

    inputs_actor = replace(inputs, obs_mask_prob=None)
    perturbed_actor = replace(perturbed_inputs, obs_mask_prob=None)
    with torch.no_grad():
        actor.obs_mask_prob.fill_(1.0)
        lp_base = actor.candidate_log_probs(inputs_actor)
        lp_pub_perturbed = actor.candidate_log_probs(perturbed_actor)
        assert torch.equal(lp_base, lp_pub_perturbed)
        assert torch.equal(lp_base, actor.candidate_log_probs(inputs))

        # Private dims still affect the actor.
        priv_idx = torch.as_tensor(np.flatnonzero(keep_mask), dtype=torch.long)
        priv_obs = inputs.obs.clone()
        priv_obs[:, priv_idx] = 1 - priv_obs[:, priv_idx]
        assert not torch.equal(lp_base, actor.candidate_log_probs(replace(inputs_actor, obs=priv_obs)))

        # Critic always reads the full observation, even when actor masks public dims.
        hidden = batch.fields["hidden"]
        v_base = critic(inputs.obs, hidden)
        v_pub_perturbed = critic(perturbed_obs, hidden)
        assert not torch.equal(v_base, v_pub_perturbed)

        # With obs_mask_prob == 0.0, public obs dims affect the actor unless per-row obs_mask_prob overrides it.
        actor.obs_mask_prob.fill_(0.0)
        assert not torch.equal(
            actor.candidate_log_probs(inputs_actor),
            actor.candidate_log_probs(perturbed_actor),
        )
        assert torch.equal(
            actor.candidate_log_probs(inputs),
            actor.candidate_log_probs(perturbed_inputs),
        )


@pytest.mark.parametrize(
    "cfg_kwargs",
    [
        dict(**BASE, private_decoder_layers=1, private_query_slots=1, private_obs_mode="full"),
        dict(**SEQ),
        dict(**SEQ, private_hidden=64, private_layers=3, trunk_hidden=96, fusion_width=80),
    ],
)
def test_dispatch_paths_and_caches_match_over_multiround_matches(cfg_kwargs):
    cfg = HistoryPolicyConfig(**cfg_kwargs)
    actor, _ = fresh_player(cfg, 9)
    actor.train()
    if cfg.private_hidden:
        with torch.no_grad():
            actor.trunk[-1].weight.normal_(std=0.05)
            actor.fusion[-1].weight.mul_(100.0)
    if actor.obs_mask_prob is not None:
        actor.obs_mask_prob.fill_(1.0)

    store, buffer, stats = _collect_multiround(actor, seed=13, num_envs=4, steps=420)
    assert stats.rounds >= 4
    data = buffer.compact()
    assert int(data["round"].max()) >= 1
    rows = np.arange(len(buffer))
    batch = buffer.training_batch(rows, store, "cpu", fields=())
    inputs = batch.inputs

    with torch.no_grad():
        # 1. Plain per-match loop (with explicit round_index)
        actor.batched_match_attention = False
        actor.batched_private_attention = False
        actor.wide_private_projection = False
        lp_plain = actor.candidate_log_probs(inputs)

        # 2. Batched match attention (host-planned rank and device-derived rank)
        actor.batched_match_attention = True
        lp_match_planned = actor.candidate_log_probs(inputs)
        lp_match_device = actor.candidate_log_probs(replace(inputs, match_rank=None, match_slots=0))
        actor.batched_match_attention = False

        # 3. Batched private attention (one_decision_per_stream=True, wide=False & True)
        each = _per_decision_inputs(inputs)
        actor.batched_private_attention = True
        actor.wide_private_projection = False
        lp_priv_indep = actor.candidate_log_probs(each)
        actor.wide_private_projection = True
        lp_priv_wide = actor.candidate_log_probs(each)
        actor.batched_private_attention = False
        actor.wide_private_projection = False

        # 4. Learner length groups = 2
        grouped = buffer.training_batch(rows, store, "cpu", fields=(), length_groups=2, width=cfg.width)
        lp_grouped = actor.candidate_log_probs(grouped.inputs)
        actor.batched_match_attention = True
        lp_grouped_batched = actor.candidate_log_probs(grouped.inputs)
        actor.batched_match_attention = False

    for name, other in (
        ("batched_match_planned", lp_match_planned),
        ("batched_match_device", lp_match_device),
        ("batched_private_indep", lp_priv_indep),
        ("batched_private_wide", lp_priv_wide),
        ("length_groups", lp_grouped),
        ("length_groups_batched", lp_grouped_batched),
    ):
        diff = float((other - lp_plain).abs().max())
        assert diff < 1e-5, f"{name} max |dlogp| = {diff}"

    # Cache parity across uncached vs BatchedHistoryCache vs PagedHistoryCache (page_span=False/True)
    keys = sorted(store.streams)
    streams = [store.streams[k] for k in keys]
    key_map = {k: i for i, k in enumerate(keys)}
    re_match_idx = torch.tensor(
        [key_map[(int(e), int(m))] for e, m in zip(data["env"], data["match"])], dtype=torch.long
    )
    re_inputs = replace(inputs, match_index=re_match_idx)
    with torch.no_grad():
        uncached_lp = actor.candidate_log_probs(
            replace(re_inputs, streams=StreamBatch.from_streams(streams, "cpu"))
        )

        batched_cache = BatchedHistoryCache(actor, chunk_size=32)
        b_meta, b_mem = batched_cache.encode(keys, streams)
        b_lp = actor.candidate_log_probs(replace(re_inputs, streams=b_meta), encoded=b_mem)
        batched_diff = float((b_lp - uncached_lp).abs().max())
        assert batched_diff < 1e-5
        if cfg.private_hidden:
            print(f"ordinary KV max |dlogp|={batched_diff:.9g}")

        modes = [(False, 0), (True, 0)]
        if cfg.private_hidden:
            modes.append((True, cfg.private_decoder_layers))
        for page_span, decoder_layers in modes:
            pool = KVPagePool.for_actor(actor, page_tokens=16, pages=4, decoder_layers=decoder_layers)
            assert pool.track_pos is True
            paged_cache = PagedHistoryCache(actor, pool, chunk_size=32, page_span=page_span)
            p_meta, p_mem = paged_cache.encode(keys, streams)
            p_lp = actor.candidate_log_probs(replace(re_inputs, streams=p_meta), encoded=p_mem)
            assert (p_meta.decoder_kv is not None) == bool(decoder_layers)
            paged_diff = float((p_lp - uncached_lp).abs().max())
            assert paged_diff < 1e-5
            if cfg.private_hidden:
                print(f"page_span={page_span}, cached decoder layers={decoder_layers}, "
                      f"max |dlogp|={paged_diff:.9g}")
            assert not pool.pos_seat[:16].any() and not pool.pos_round[:16].any()


def test_cache_byte_accounting_and_forked_cache():
    legacy_actor, _ = fresh_player(HistoryPolicyConfig(**BASE), 1)
    seq_actor, _ = fresh_player(HistoryPolicyConfig(**SEQ), 2)

    s0, s1 = PublicStream(0), PublicStream(1)
    _append_events(s0, 15, seed=1)
    _append_events(s1, 45, seed=2)

    leg_cache = BatchedHistoryCache(legacy_actor, chunk_size=16)
    leg_meta, _ = leg_cache.encode([(0, 0), (1, 1)], [s0, s1])
    assert leg_meta.pos_seat is None and leg_meta.pos_round is None
    leg_expected_entry = sum(
        t.numel() * t.element_size()
        for e in leg_cache.entries.values()
        for t in (*e.keys, *e.values, e.memory)
    )
    leg_pos_bytes = sum(t.numel() * t.element_size() for t in leg_cache._position_cache[1:])
    assert leg_cache.bytes == leg_expected_entry + leg_pos_bytes

    leg_pool = KVPagePool.for_actor(legacy_actor, page_tokens=16, pages=4)
    assert leg_pool.track_pos is False
    assert leg_pool.page_bytes == 16 * (1 * 2 * 32 + 32) * 4

    seq_cache = BatchedHistoryCache(seq_actor, chunk_size=16)
    seq_meta, _ = seq_cache.encode([(0, 0), (1, 1)], [s0, s1])
    assert seq_meta.pos_seat is not None and seq_meta.pos_round is not None
    extra_bytes = sum(2 * e.capacity for e in seq_cache.entries.values())
    seq_expected_entry = sum(
        t.numel() * t.element_size()
        for e in seq_cache.entries.values()
        for t in (*e.keys, *e.values, e.memory)
    )
    seq_pos_bytes = sum(t.numel() * t.element_size() for t in seq_cache._position_cache[1:])
    assert seq_cache.bytes == seq_expected_entry + seq_pos_bytes + extra_bytes

    seq_pool = KVPagePool.for_actor(seq_actor, page_tokens=16, pages=4)
    assert seq_pool.track_pos is True
    assert seq_pool.page_bytes == leg_pool.page_bytes + 2 * 16
    paged = PagedHistoryCache(seq_actor, seq_pool, chunk_size=16)
    paged.encode([(0, 0), (1, 1)], [s0, s1])
    used_pages = sum(len(e.pages) for e in paged.entries.values())
    paged_pos_bytes = sum(t.numel() * t.element_size() for t in paged._position_cache[1:])
    assert paged.bytes == used_pages * seq_pool.page_bytes + paged_pos_bytes
    with pytest.raises(ValueError):
        PagedHistoryCache(seq_actor, leg_pool)

    # ForkedHistoryCache tracks pos_seat / pos_round cleanly
    root_cache = BatchedHistoryCache(seq_actor, chunk_size=16)
    root_cache.prefill([(0, 0)], [s0])
    b0, b1 = PublicStream(0), PublicStream(0)
    b0.extend_tokens(*s0.arrays())
    b1.extend_tokens(*s0.arrays())
    _append_events(b0, 4, seed=10)
    _append_events(b1, 25, seed=11)
    forked = ForkedHistoryCache(root_cache, (0, 0), [b0, b1])
    f_meta, f_mem = forked.encode([0, 1])
    assert f_meta.pos_seat is not None and f_meta.pos_round is not None
    direct_mem = seq_actor.encode_batch(StreamBatch.from_streams([b0, b1], "cpu"))
    for j, b in enumerate((b0, b1)):
        n = b.prefix + 1
        assert torch.allclose(f_mem[j, :n], direct_mem[j, :n], atol=2e-5)
        assert not f_mem[j, n:].any()
        assert not f_meta.pos_seat[j, n:].any() and not f_meta.pos_round[j, n:].any()


@pytest.mark.parametrize("prob", [0.0, 0.5, 1.0])
def test_ppo_ratio_one_across_obs_mask_probs_with_fast_flags(tmp_path, prob):
    cfg = HistoryPPOConfig(
        **SEQ,
        aux_heads="next,belief,outcome",
        response_mode="auxiliary",
        obs_mask_warmup_updates=(2 if prob < 1.0 else 0),
        rollout_kv_cache=True,
        rollout_paged_cache=True,
        rollout_page_span=True,
        rollout_batched_attention=True,
        rollout_wide_projection=True,
        learner_batched_attention=True,
        learner_length_groups=2,
        num_envs=4,
        steps_per_update=80,
        minibatch_matches=2,
        epochs=1,
        seed=5,
    )
    trainer = HistoryTrainer(cfg, tmp_path / f"run-{prob}")
    if prob == 0.5:
        trainer.progress["updates"] = 1
        assert trainer.apply_obs_mask_schedule() == pytest.approx(0.5)
    else:
        assert float(trainer.actor.obs_mask_prob.item()) == pytest.approx(prob)
    trainer.collect()
    values = trainer.refresh_values()
    assert trainer.buffer.finalize(values, cfg.gamma, cfg.gae_lambda) > 0
    rows = next(iter(trainer.minibatches()))
    terms = trainer.minibatch_loss(rows)
    assert float(terms["ratio_deviation"]) < 1e-5


def test_carried_rows_preserve_per_row_obs_mask_prob_and_ratio_one(tmp_path):
    """Incomplete-round rows collected at prob=0.0 carry over into updates at
    prob=0.5 and prob=1.0; epoch-0 minibatch_loss keeps ratio == 1.0 (< 1e-5)
    on the mixed batch while default full-mode leaves buffer chunks untouched."""
    full_trainer = HistoryTrainer(
        HistoryPPOConfig(**BASE, num_envs=4, steps_per_update=20, seed=5),
        tmp_path / "full",
    )
    full_trainer.collect()
    assert "obs_mask_prob" not in full_trainer.buffer.compact()

    cfg = HistoryPPOConfig(
        **SEQ,
        aux_heads="next,belief,outcome",
        response_mode="auxiliary",
        obs_mask_warmup_updates=2,
        rollout_kv_cache=True,
        rollout_paged_cache=True,
        rollout_page_span=True,
        rollout_batched_attention=True,
        rollout_wide_projection=True,
        learner_batched_attention=True,
        learner_length_groups=2,
        num_envs=4,
        steps_per_update=12,
        minibatch_matches=4,
        epochs=1,
        seed=9,
    )
    trainer = HistoryTrainer(cfg, tmp_path / "carried")

    # Update 0 (prob=0.0): collect partial round -> carry over via next_iteration()
    assert trainer.apply_obs_mask_schedule() == pytest.approx(0.0)
    trainer.collector.collect(12, version=0)
    assert len(trainer.buffer) > 0
    assert len(trainer.buffer.completed_rows()) == 0
    cited = trainer.buffer.next_iteration()
    trainer.store.prune(cited)
    assert len(trainer.buffer) > 0
    assert np.all(trainer.buffer.compact()["obs_mask_prob"] == 0.0)

    # Update 1 (prob=0.5): collect more partial-round rows into same open trajectories
    trainer.progress["updates"] = 1
    assert trainer.apply_obs_mask_schedule() == pytest.approx(0.5)
    trainer.collector.collect(12, version=1)
    cited = trainer.buffer.next_iteration()
    trainer.store.prune(cited)

    # Update 2 (prob=1.0): finish the rounds so a single minibatch mixes 0.0, 0.5, 1.0
    trainer.progress["updates"] = 2
    assert trainer.apply_obs_mask_schedule() == pytest.approx(1.0)
    trainer.collector.collect(120, version=2)
    samples = trainer.buffer.finalize(trainer.refresh_values(), cfg.gamma, cfg.gae_lambda)
    assert samples > 0
    sample_probs = trainer.buffer.compact()["obs_mask_prob"][trainer.buffer.samples]
    assert {0.0, 0.5, 1.0} <= set(np.unique(sample_probs).tolist())

    saw_mixed = False
    for rows in trainer.minibatches():
        probs = set(np.unique(trainer.buffer.compact()["obs_mask_prob"][rows]).tolist())
        if {0.0, 0.5, 1.0} <= probs:
            saw_mixed = True
        terms = trainer.minibatch_loss(rows)
        assert float(terms["ratio_deviation"]) < 1e-5
    assert saw_mixed


def test_warmup_schedule_init_checkpoint_resume_aux_and_eval(tmp_path):
    # 1. Source checkpoint with aux heads + response + archived population
    src_cfg = HistoryPPOConfig(
        **SEQ,
        aux_heads="next,belief,outcome",
        response_mode="auxiliary",
        snapshot_updates=1,
        population_archive_every=1,
        num_envs=4,
        steps_per_update=90,
        minibatch_matches=2,
        epochs=1,
        updates=2,
        seed=3,
    )
    src_trainer = HistoryTrainer(src_cfg, tmp_path / "src")
    src_trainer.run()
    assert src_trainer.population.models
    assert src_trainer.population.archive
    src_ckpt = tmp_path / "src" / "latest.pt"

    # Simulate a migrated checkpoint whose actor optimizer omits private_decoder.* states
    # and whose obs_mask_prob starts at 0.0 (snapshots also at 0.0).
    init_payload = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    init_payload["model"]["obs_mask_prob"].fill_(0.0)
    decoder_indices = {
        idx
        for idx, (name, _) in enumerate(src_trainer.actor.named_parameters())
        if name.startswith("private_decoder.")
    }
    assert decoder_indices
    for idx in decoder_indices:
        init_payload["optimizer"]["actor"]["state"].pop(idx, None)
    for snap_rec in init_payload["population"]["snapshots"].values():
        snap_rec["model"]["obs_mask_prob"].fill_(0.0)
        from train.history_population import weights_digest

        snap_rec["metadata"]["sha256"] = weights_digest(snap_rec["model"])
    init_ckpt = tmp_path / "init.pt"
    save_history_checkpoint(init_ckpt, init_payload)

    # Reject combining resume with init_checkpoint
    with pytest.raises(ValueError, match="init-checkpoint"):
        HistoryTrainer(
            HistoryPPOConfig(**SEQ, init_checkpoint=str(init_ckpt)),
            tmp_path / "bad",
            resume=src_ckpt,
        )

    # 2. New lineage via init_checkpoint with default archive_every=0 (adopts init population)
    cont_cfg = HistoryPPOConfig(
        **SEQ,
        aux_heads="next,belief,outcome",
        response_mode="auxiliary",
        obs_mask_warmup_updates=2,
        init_checkpoint=str(init_ckpt),
        lr=1e-4,
        critic_lr=2e-4,
        snapshot_updates=1,
        num_envs=4,
        steps_per_update=90,
        minibatch_matches=2,
        epochs=1,
        updates=3,
        seed=11,
    )
    trainer = HistoryTrainer(cont_cfg, tmp_path / "cont")
    assert trainer.lineage != init_payload["lineage"]
    assert trainer.progress["updates"] == 0
    assert trainer.progress["obs_mask_start_update"] == 0
    assert trainer.progress["obs_mask_initial_prob"] == pytest.approx(0.0)
    assert float(trainer.actor.obs_mask_prob.item()) == pytest.approx(0.0)
    assert trainer.actor_optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert trainer.critic_optimizer.param_groups[0]["lr"] == pytest.approx(2e-4)
    # Non-decoder optimizer state survived; decoder indices omitted for lazy init
    opt_state = trainer.actor_optimizer.state_dict()["state"]
    assert not (set(opt_state) & decoder_indices)
    assert len(opt_state) > 0
    # Population snapshots preserved with new lineage and obs_mask_prob == 0.0
    assert len(trainer.population.models) > 0
    for snap_model in trainer.population.models.values():
        assert float(snap_model.obs_mask_prob.item()) == pytest.approx(0.0)

    manifest = json.loads((tmp_path / "cont" / "manifest.json").read_text())
    assert manifest["init_checkpoint"]["lineage"] == init_payload["lineage"]

    # Run update 1 (prob=0.0) -> save -> resume -> run updates 2 & 3 (prob=0.5, 1.0)
    line1 = trainer.update()
    assert line1["update_samples"] > 0
    assert line1["obs_mask_prob"] == pytest.approx(0.0)
    assert np.isfinite(line1["actor_grad_norm"]) and np.isfinite(line1["aux_loss"])
    mid_ckpt = trainer.save(tmp_path / "cont" / "mid.pt")

    resumed = HistoryTrainer(
        HistoryPPOConfig(updates=3),
        tmp_path / "resumed",
        resume=mid_ckpt,
    )
    assert resumed.lineage == trainer.lineage
    assert resumed.init_checkpoint == trainer.init_checkpoint
    assert float(resumed.actor.obs_mask_prob.item()) == pytest.approx(0.5)
    line2 = resumed.update()
    line3 = resumed.update()
    assert line2["obs_mask_prob"] == pytest.approx(0.5)
    assert line3["obs_mask_prob"] == pytest.approx(1.0)
    for line in (line1, line2, line3):
        assert np.isfinite(line["approx_kl"]) and np.isfinite(line["actor_grad_norm"])
    final_ckpt = resumed.save()

    # Check metrics.jsonl recorded obs_mask_prob on private_only, omitted on full
    rows_jsonl = [
        json.loads(x) for x in (tmp_path / "resumed" / "metrics.jsonl").read_text().splitlines()
    ]
    assert [r["obs_mask_prob"] for r in rows_jsonl] == pytest.approx([0.5, 1.0])
    full_trainer = HistoryTrainer(
        HistoryPPOConfig(**BASE, num_envs=4, steps_per_update=40, minibatch_matches=2, epochs=1),
        tmp_path / "full",
    )
    full_line = full_trainer.update()
    assert "obs_mask_prob" not in full_line

    # 3. Evaluate via load_policy / HistoryPolicy and check extend_actor
    policy = load_policy(str(final_ckpt))
    assert isinstance(policy, HistoryPolicy)
    engine = gd.Engine(gd.RuleConfig.house(), gd.ActionConfig())
    state = gd.MatchState()
    engine.new_match(state, 19)
    policy.start_match(0)
    actions = engine.legal_actions(state)
    inputs = policy.decision_inputs(engine, state, actions)
    assert inputs.round_index is not None and int(inputs.round_index[0]) == int(state.round_index)
    choice = policy.select(engine, state, actions, __import__("random").Random(0))
    assert 0 <= choice < len(actions)

    # batch_inputs with and without explicit round_index
    batch_in = policy.batch_inputs(
        [policy.stream],
        np.array([0], dtype=np.int64),
        np.array([int(state.to_move)], dtype=np.int64),
        np.asarray(state.observation(int(state.to_move)), dtype=np.float32)[None, :],
        np.stack([engine.encode_action(a, state, int(state.to_move)) for a in actions]),
        np.array([0, len(actions)], dtype=np.int64),
        round_index=np.array([int(state.round_index)], dtype=np.int64),
    )
    assert batch_in.round_index is not None
    assert policy.act(batch_in).shape == (1,)

    # extend_actor keeps obs_mask_prob and private_decoder intact
    base_seq, _ = fresh_player(HistoryPolicyConfig(**SEQ), 4)
    base_seq.obs_mask_prob.fill_(0.25)
    ext_seq = extend_actor(base_seq, replace(base_seq.config, aux_heads="next,belief,outcome"))
    assert float(ext_seq.obs_mask_prob.item()) == pytest.approx(0.25)

    # resume-set obs_mask_warmup_updates updates the schedule smoothly from current prob
    resumed_override = HistoryTrainer(
        HistoryPPOConfig(updates=5),
        tmp_path / "resumed-override",
        resume=mid_ckpt,
        resume_overrides=parse_resume_overrides(["obs_mask_warmup_updates=4"]),
    )
    assert resumed_override.progress["obs_mask_start_update"] == 1
    assert resumed_override.progress["obs_mask_initial_prob"] == pytest.approx(0.0)
    line_ov = resumed_override.update()
    assert line_ov["obs_mask_prob"] == pytest.approx(0.0)
    line_ov2 = resumed_override.update()
    assert line_ov2["obs_mask_prob"] == pytest.approx(0.25)


@pytest.mark.parametrize("trunk", [{}, dict(private_hidden=64, private_layers=3, trunk_hidden=96)],
                         ids=["sequence", "large-trunk"])
def test_card_counter_ppo_warmup_resume_and_carried_ratio_one(tmp_path, trunk):
    """card_counter mode supports obs_mask warmup, --resume, carried-row ratio==1.0,
    and merged snapshot batching + paged cache."""
    cc_seq = dict(**BASE, private_decoder_layers=2, private_query_slots=2, private_obs_mode="card_counter")
    carried_cfg = HistoryPPOConfig(
        **cc_seq,
        **trunk,
        aux_heads="next,belief,outcome",
        response_mode="auxiliary",
        obs_mask_warmup_updates=2,
        rollout_kv_cache=True,
        rollout_paged_cache=True,
        rollout_page_span=True,
        rollout_batched_attention=True,
        rollout_wide_projection=True,
        learner_batched_attention=True,
        learner_length_groups=2,
        rollout_decoder_kv_cache=bool(trunk),
        num_envs=4,
        steps_per_update=12,
        minibatch_matches=4,
        epochs=1,
        seed=9,
    )
    carried = HistoryTrainer(carried_cfg, tmp_path / "carried")
    assert carried.progress["obs_mask_start_update"] == 0
    assert carried.progress["obs_mask_initial_prob"] == pytest.approx(0.0)

    assert carried.apply_obs_mask_schedule() == pytest.approx(0.0)
    carried.collector.collect(12, version=0)
    assert len(carried.buffer.completed_rows()) == 0
    carried.store.prune(carried.buffer.next_iteration())

    carried.progress["updates"] = 1
    assert carried.apply_obs_mask_schedule() == pytest.approx(0.5)
    carried.collector.collect(12, version=1)
    carried.store.prune(carried.buffer.next_iteration())

    carried.progress["updates"] = 2
    assert carried.apply_obs_mask_schedule() == pytest.approx(1.0)
    carried.collector.collect(120, version=2)
    samples = carried.buffer.finalize(carried.refresh_values(), carried_cfg.gamma, carried_cfg.gae_lambda)
    assert samples > 0
    saw_mixed = False
    max_ratio_deviation = 0.0
    for rows in carried.minibatches():
        probs = set(np.unique(carried.buffer.compact()["obs_mask_prob"][rows]).tolist())
        if {0.0, 0.5, 1.0} <= probs:
            saw_mixed = True
        terms = carried.minibatch_loss(rows)
        ratio_deviation = float(terms["ratio_deviation"])
        max_ratio_deviation = max(max_ratio_deviation, ratio_deviation)
        assert ratio_deviation < 1e-5
    assert saw_mixed
    if trunk:
        print(f"carried card_counter 0/0.5/1 max |ratio - 1|={max_ratio_deviation:.9g}")

    run_cfg = HistoryPPOConfig(
        **cc_seq,
        **trunk,
        obs_mask_warmup_updates=2,
        snapshot_updates=1,
        batch_snapshot_policies=True,
        batch_snapshot_encoder=True,
        rollout_kv_cache=True,
        rollout_paged_cache=True,
        rollout_page_span=True,
        num_envs=4,
        steps_per_update=90,
        minibatch_matches=2,
        epochs=1,
        updates=3,
        seed=15,
    )
    trainer = HistoryTrainer(run_cfg, tmp_path / "run")
    line1 = trainer.update()
    assert line1["obs_mask_prob"] == pytest.approx(0.0)
    assert np.isfinite(line1["approx_kl"]) and np.isfinite(line1["actor_grad_norm"])
    ckpt = trainer.save()

    resumed = HistoryTrainer(HistoryPPOConfig(updates=3), tmp_path / "resumed", resume=ckpt)
    assert resumed.config.private_obs_mode == "card_counter"
    assert float(resumed.actor.obs_mask_prob.item()) == pytest.approx(0.5)
    line2 = resumed.update()
    line3 = resumed.update()
    assert line2["obs_mask_prob"] == pytest.approx(0.5)
    assert line3["obs_mask_prob"] == pytest.approx(1.0)
    for line in (line2, line3):
        assert np.isfinite(line["approx_kl"]) and np.isfinite(line["actor_grad_norm"])
    rows_jsonl = [
        json.loads(x) for x in (tmp_path / "resumed" / "metrics.jsonl").read_text().splitlines()
    ]
    assert [r["obs_mask_prob"] for r in rows_jsonl] == pytest.approx([0.5, 1.0])
