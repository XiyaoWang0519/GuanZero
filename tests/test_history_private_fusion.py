"""Config-gated private-query/reader fusion (``private_fusion``).

The default ("sum") must leave the architecture record, the parameters and
the outputs of the current lineage bitwise untouched. The "split" arm adds
only a trailing ``reader_norm`` LayerNorm, reuses ``attention_norm`` as the
q_proj input norm and fuses the reader output under its own gain, fixing the
query-dominated fusion of docs/reports/qknorm-fusion-isolation-2026-10-16.md
and private-attention-audit-checkpoints-2026-10-18.md.
"""
import json

import gd
import numpy as np
import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from eval.botzone.numpy_actor import NumpyHistoryActor
from eval.private_attention_audit import batch_inputs, capture, collect_sample, encode
from train import history_fusion_migrate
from train.history_config import RESUME_OVERRIDES
from train.history_model import (DecisionInputs, HistoryActor, HistoryPolicyConfig,
                                 MatchGroup, StreamBatch, checkpoint_payload,
                                 config_record, fresh_player, load_history_checkpoint,
                                 save_history_checkpoint)
from train.history_ppo import (HistoryPPOConfig, HistoryTrainer, build_parser,
                               config_from_args, habit_init_player, parse_resume_overrides)
from train.history_rollout import HistoryCollector, MatchEventStore, SequenceRolloutBuffer
from train.history_snapshot_batch import validate_actor

BASE = dict(width=32, layers=1, heads=4)
ARCH = dict(**BASE, response_mode="auxiliary")
QK_ARMS = ("none", "qk")


@pytest.fixture(scope="module")
def sample():
    actor, _ = fresh_player(HistoryPolicyConfig(**BASE), 3)
    return collect_sample(actor, envs=3, steps=160, seed=5, matches=3, minibatch_matches=3,
                          rows_per_match=20)


def old_fusion(actor, query, attended):
    """The pre-refactor sum fusion, written out inline."""
    state = actor.attention_norm(query + attended)
    return actor.output_norm(state + actor.feed_forward(state))


def manual_fusion(actor, query, attended):
    """The split formula, written out inline."""
    reading = actor.attention_norm(query)
    state = reading + actor.reader_norm(attended)
    return actor.output_norm(state + actor.feed_forward(state))


def grouped(inputs):
    """A copy of ``inputs`` with matches split into two length groups."""
    matches = torch.unique(inputs.match_index)
    groups = []
    for picked in (matches[::2], matches[1::2]):
        if not len(picked):
            continue
        rows = torch.isin(inputs.match_index, picked).nonzero(as_tuple=True)[0]
        local = torch.searchsorted(picked, inputs.match_index[rows])
        groups.append(MatchGroup(matches=picked, rows=rows, local_match=local,
                                 tokens=int(inputs.streams.tokens.shape[1])))
    clone = DecisionInputs(streams=inputs.streams, match_index=inputs.match_index,
                           prefix=inputs.prefix, obs=inputs.obs, seat=inputs.seat,
                           cand=inputs.cand, offsets=inputs.offsets,
                           match_groups=groups)
    return clone


def per_decision(inputs):
    """One stream per decision, for the batched-private-attention path."""
    s, n = inputs.streams, inputs.decisions
    return DecisionInputs(streams=StreamBatch(s.tokens[inputs.match_index],
                                              s.rounds[inputs.match_index],
                                              s.phases[inputs.match_index],
                                              s.lengths[inputs.match_index]),
                          match_index=torch.arange(n), prefix=inputs.prefix,
                          obs=inputs.obs, seat=inputs.seat, cand=inputs.cand,
                          offsets=inputs.offsets, one_decision_per_stream=True)


def test_default_record_and_parameters_unchanged():
    config = HistoryPolicyConfig(**BASE)
    assert config.private_fusion == "sum"
    assert "private_fusion" not in config_record(config)
    actor, _critic = fresh_player(config, 0)
    assert actor.reader_norm is None
    assert not any(key.startswith("reader_norm.") for key in actor.state_dict())
    payload = checkpoint_payload(actor, _critic, lineage="x", seed=0)
    assert "private_fusion" not in payload["model_config"]
    with pytest.raises(ValueError, match="private_fusion"):
        HistoryPolicyConfig(**BASE, private_fusion="rms")


def test_sum_outputs_bitwise_on_every_path(sample):
    """The default arm is the old computation, written out inline."""
    actor, _ = fresh_player(HistoryPolicyConfig(**BASE), 2)
    inputs = batch_inputs(sample[0])
    with torch.no_grad():
        query = actor.private(inputs.obs.float()) + actor.seat(inputs.seat)
        encoded = actor.encode_batch(inputs.streams)
        keys, values = actor.project_keys_values(encoded)
        # plain (per-match loop over encoded streams)
        attended = actor._attend_encoded(query, encoded, inputs.match_index, inputs.prefix,
                                         False)
        want = old_fusion(actor, query, attended)
        assert torch.equal(actor.decision_states(None, inputs), want)
        assert torch.equal(actor.decision_states(encoded, inputs), want)
        # batched match attention (several decisions per match, one padded call)
        actor.batched_match_attention = True
        attended = actor._attend_by_match(query, keys, values, inputs.match_index,
                                          inputs.prefix)
        want = old_fusion(actor, query, attended)
        assert torch.equal(actor.decision_states(None, inputs), want)
        actor.batched_match_attention = False
        # length-grouped learner encode
        grouped_inputs = grouped(inputs)
        attended = actor._attend_length_groups(query, grouped_inputs)
        want = old_fusion(actor, query, attended)
        assert torch.equal(actor.decision_states(None, grouped_inputs), want)
        # batched private attention (one decision per stream)
        each = per_decision(inputs)
        each_keys, each_values = actor.project_keys_values(actor.encode_batch(each.streams))
        allowed = torch.arange(each.streams.tokens.shape[1] + 1)[None] <= each.prefix[:, None]
        actor.batched_private_attention = True
        attended = actor._attend_independent(query, each_keys, each_values, allowed)
        want = old_fusion(actor, query, attended)
        assert torch.equal(actor.decision_states(None, each), want)
        actor.batched_private_attention = False
        # windowed control
        windowed, _ = fresh_player(HistoryPolicyConfig(**BASE, window=8), 2)
        query = windowed.private(inputs.obs.float()) + windowed.seat(inputs.seat)
        stream, lengths = windowed._windowed(inputs)
        keys, values = windowed.project_keys_values(stream)
        allowed = torch.arange(stream.shape[1])[None] <= lengths[:, None]
        attended = windowed._attend(query, keys, values, allowed)
        want = old_fusion(windowed, query, attended)
        assert torch.equal(windowed.decision_states(None, inputs), want)


def test_split_adds_only_trailing_reader_norm():
    base_actor, base_critic = fresh_player(HistoryPolicyConfig(**BASE), 11)
    actor, critic = fresh_player(HistoryPolicyConfig(**BASE, private_fusion="split"), 11)
    added = set(actor.state_dict()) - set(base_actor.state_dict())
    assert added == {"reader_norm.weight", "reader_norm.bias"}
    for key, value in base_actor.state_dict().items():
        assert torch.equal(actor.state_dict()[key], value), key
    for (name, a), (_, b) in zip(base_critic.state_dict().items(), critic.state_dict().items()):
        assert torch.equal(a, b), name
    names = [name for name, _ in actor.named_parameters()]
    assert names[-2:] == ["reader_norm.weight", "reader_norm.bias"]
    assert config_record(actor.config)["private_fusion"] == "split"
    # qk stacks on top; both norms stay trailing relative to the base params.
    qk, _ = fresh_player(HistoryPolicyConfig(**BASE, private_fusion="split",
                                             private_qk_norm="qk"), 11)
    names = [name for name, _ in qk.named_parameters()]
    assert names[-6:] == ["q_norm.weight", "q_norm.bias", "k_norm.weight", "k_norm.bias",
                          "reader_norm.weight", "reader_norm.bias"]
    for key, value in base_actor.state_dict().items():
        assert torch.equal(qk.state_dict()[key], value), key


@pytest.mark.parametrize("qk", QK_ARMS)
def test_split_decision_states_equal_the_manual_formula(qk, sample):
    actor, _ = fresh_player(HistoryPolicyConfig(**BASE, private_fusion="split",
                                                private_qk_norm=qk), 4)
    inputs = batch_inputs(sample[0])
    with torch.no_grad():
        query = actor.private_query(inputs)
        reading = actor.attention_norm(query)
        # plain path
        encoded = actor.encode_batch(inputs.streams)
        keys, values = actor.project_keys_values(encoded)
        attended = actor._attend_encoded(reading, encoded, inputs.match_index,
                                         inputs.prefix, False)
        want = manual_fusion(actor, query, attended)
        assert torch.equal(actor.decision_states(None, inputs), want)
        assert torch.equal(actor.decision_states(encoded, inputs), want)
        # batched match attention
        actor.batched_match_attention = True
        attended = actor._attend_by_match(reading, keys, values, inputs.match_index,
                                          inputs.prefix)
        want = manual_fusion(actor, query, attended)
        assert torch.equal(actor.decision_states(None, inputs), want)
        actor.batched_match_attention = False
        # length groups
        grouped_inputs = grouped(inputs)
        attended = actor._attend_length_groups(reading, grouped_inputs)
        want = manual_fusion(actor, query, attended)
        assert torch.equal(actor.decision_states(None, grouped_inputs), want)
        # batched private attention
        each = per_decision(inputs)
        each_keys, each_values = actor.project_keys_values(actor.encode_batch(each.streams))
        allowed = torch.arange(each.streams.tokens.shape[1] + 1)[None] <= each.prefix[:, None]
        actor.batched_private_attention = True
        attended = actor._attend_independent(reading, each_keys, each_values, allowed)
        want = manual_fusion(actor, query, attended)
        assert torch.equal(actor.decision_states(None, each), want)
        actor.batched_private_attention = False
        # the audit's explicit recomputation agrees (it uses the same methods)
        assert capture(actor, inputs)["state_error"] < 1e-5


@pytest.mark.parametrize("qk", QK_ARMS)
def test_split_windowed_and_different_outputs(qk, sample):
    actor, _ = fresh_player(HistoryPolicyConfig(**BASE, window=8, private_fusion="split",
                                                private_qk_norm=qk), 1)
    inputs = batch_inputs(sample[0])
    with torch.no_grad():
        query = actor.private_query(inputs)
        reading = actor.attention_norm(query)
        stream, lengths = actor._windowed(inputs)
        keys, values = actor.project_keys_values(stream)
        allowed = torch.arange(stream.shape[1])[None] <= lengths[:, None]
        attended = actor._attend(reading, keys, values, allowed)
        want = manual_fusion(actor, query, attended)
        assert torch.equal(actor.decision_states(None, inputs), want)
        # split changes the outputs of a sum actor sharing the weights
        twin = HistoryActor(HistoryPolicyConfig(**BASE, window=8, private_qk_norm=qk))
        twin.load_state_dict({k: v for k, v in actor.state_dict().items()
                              if not k.startswith("reader_norm.")})
        assert not torch.equal(twin.decision_states(None, inputs), want)


def test_split_kv_cache_rollout_matches_recompute():
    """The collector's cached inference goes through the same split fusion."""
    actor, _ = fresh_player(HistoryPolicyConfig(**BASE, private_fusion="split",
                                                private_qk_norm="qk"), 9)
    env = gd.VecEnv(num_envs=2, num_threads=1, seed=4, log_public_actions=True,
                    log_env_limit=2)
    store, buffer = MatchEventStore(), SequenceRolloutBuffer()
    HistoryCollector(env, actor, store, buffer, torch.Generator().manual_seed(1),
                     kv_cache=True).collect(60)
    rows = np.arange(len(buffer))
    inputs, chosen = buffer.decision_inputs(rows, store, "cpu")
    with torch.no_grad():
        recomputed = actor.candidate_log_probs(inputs)[chosen].numpy()
    np.testing.assert_allclose(recomputed, buffer.compact()["logp"], rtol=1e-4, atol=1e-5)


def test_refused_by_paths_that_bypass_the_helpers():
    actor = HistoryActor(HistoryPolicyConfig(**BASE, private_fusion="split"))
    with pytest.raises(ValueError, match="private fusion"):
        validate_actor(actor)
    validate_actor(HistoryActor(HistoryPolicyConfig(**BASE)))
    with pytest.raises(ValueError, match="private fusion"):
        NumpyHistoryActor({}, dict(width=32, heads=4, layers=1, max_rounds=16,
                                   private_fusion="split"))
    with pytest.raises(ValueError, match="private fusion"):
        HistoryPPOConfig(private_fusion="split", batch_snapshot_policies=True)
    with pytest.raises(ValueError, match="private fusion"):
        HistoryPPOConfig(private_fusion="split", batch_snapshot_policies_schedule="2:on")
    # split already normalizes the q_proj input; the "query" arm is redundant
    with pytest.raises(ValueError, match="q_proj input"):
        HistoryPolicyConfig(**BASE, private_fusion="split", private_qk_norm="query")


def test_cli_config_and_checkpoint_round_trip(tmp_path):
    assert config_from_args(build_parser().parse_args(["--output", "x"])).private_fusion == "sum"
    args = build_parser().parse_args(
        ["--output", "x", "--private-fusion", "split", "--private-qk-norm", "qk"])
    config = config_from_args(args)
    assert config.private_fusion == "split"
    assert config.policy_config().private_fusion == "split"
    assert config.policy_config().private_qk_norm == "qk"
    actor, critic = fresh_player(HistoryPolicyConfig(**BASE, private_fusion="split",
                                                     private_qk_norm="qk"), 8)
    with torch.no_grad():
        for p in actor.parameters():
            p.add_(0.01)
    path = tmp_path / "x.pt"
    save_history_checkpoint(path, checkpoint_payload(actor, critic, lineage="ab", seed=8))
    loaded, _, payload = load_history_checkpoint(path)
    assert payload["model_config"]["private_fusion"] == "split"
    assert payload["model_config"]["private_qk_norm"] == "qk"
    assert loaded.config.private_fusion == "split"
    for key, value in actor.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_split_qk_gradients_reach_the_reader_and_norms(sample):
    actor, _ = fresh_player(HistoryPolicyConfig(**BASE, private_fusion="split",
                                                private_qk_norm="qk"), 6)
    batch = sample[0]
    inputs = batch_inputs(batch)
    with sdpa_kernel(SDPBackend.MATH):
        loss = actor.candidate_log_probs(inputs)[batch["chosen"]].sum()
    loss.backward()
    params = dict(actor.named_parameters())
    for name in ("q_proj.weight", "kv_proj.weight", "reader_norm.weight",
                 "reader_norm.bias", "q_norm.weight", "k_norm.weight"):
        grad = params[name].grad
        assert grad is not None and float(grad.abs().max()) > 0, name


def test_trainer_split_update_resume_and_no_resume_change(tmp_path):
    config = HistoryPPOConfig(width=32, layers=1, heads=4, num_envs=3, steps_per_update=120,
                              seed=3, updates=1, epochs=1, minibatch_matches=2,
                              private_fusion="split", private_qk_norm="qk")
    trainer = HistoryTrainer(config, tmp_path / "split")
    assert trainer.actor.reader_norm is not None
    trainer.update()
    assert all(torch.isfinite(p).all() for p in trainer.actor.parameters())
    trainer.save()
    resumed = HistoryTrainer(config, tmp_path / "split",
                             resume=str(tmp_path / "split" / "latest.pt"))
    assert resumed.actor.config.private_fusion == "split"
    assert resumed.actor.config.private_qk_norm == "qk"
    for key, value in trainer.actor.state_dict().items():
        assert torch.equal(resumed.actor.state_dict()[key], value)
    # private_fusion is not a resume override: the flag cannot change a lineage.
    assert "private_fusion" not in RESUME_OVERRIDES
    with pytest.raises(ValueError, match="resume-set"):
        parse_resume_overrides(["private_fusion=sum"])


@pytest.fixture(scope="module")
def legacy_checkpoint(tmp_path_factory):
    """A tiny trained "sum" checkpoint: habit-init compatible, with Adam state."""
    path = tmp_path_factory.mktemp("legacy") / "src.pt"
    trainer = HistoryTrainer(HistoryPPOConfig(**ARCH, num_envs=4, steps_per_update=60, seed=5,
                                              updates=1, epochs=1, minibatch_matches=2),
                             path.parent / "run")
    trainer.update()
    trainer.save(path)
    return path


def test_fusion_migration_end_to_end(legacy_checkpoint, tmp_path):
    source, _, payload = load_history_checkpoint(legacy_checkpoint)
    source = source.eval()
    source.causal_sdpa = True
    out = tmp_path / "converted.pt"
    migration = history_fusion_migrate.migrate(
        legacy_checkpoint, out, sample_seed=5, heldout_seed=6, matches=2,
        reader_gain="calibrated", envs=3, steps=160, minibatch_matches=2,
        rows_per_match=20)
    converted, _, new_payload = load_history_checkpoint(out)
    assert converted.config.private_fusion == "split"
    assert converted.config.private_qk_norm == "qk"
    assert new_payload["fusion_migration"] == migration
    # every source parameter is copied exactly; norms start as prescribed
    for key, value in source.state_dict().items():
        assert torch.equal(converted.state_dict()[key], value), key
    gain = migration["reader_gain"]
    torch.testing.assert_close(converted.reader_norm.weight,
                               gain * converted.attention_norm.weight)
    assert torch.equal(converted.reader_norm.bias, torch.zeros(32))
    assert torch.equal(converted.q_norm.weight, torch.ones(8))
    assert torch.equal(converted.k_norm.weight, torch.ones(8))
    # c equals the calibration formula recomputed on the same sample
    batches = collect_sample(source, envs=3, steps=160, seed=5, matches=2,
                             minibatch_matches=2, rows_per_match=20)
    eps = source.attention_norm.eps
    c_rows = []
    for batch in batches:
        inputs = batch_inputs(batch)
        with torch.no_grad():
            query = source.private_query(inputs)
            encoded = encode(source, inputs.streams)
            attended = source._attend_encoded(query, encoded, inputs.match_index,
                                              inputs.prefix, False)
            c_row = ((attended - attended.mean(-1, keepdim=True)).square().mean(-1).sqrt()
                     / ((query + attended).var(-1, unbiased=False) + eps).sqrt())
            c_rows.append(c_row[inputs.prefix >= 1])
    assert gain == pytest.approx(float(torch.cat(c_rows).median()), rel=1e-6)
    assert migration["calibration"]["rows"] == int(torch.cat(c_rows).numel())
    assert migration["heldout"]["converted"]["tv_mean"] >= 0
    # the CLI writes the same record as the report JSON
    cli_out, report_path = tmp_path / "cli.pt", tmp_path / "report.json"
    history_fusion_migrate.main(
        [str(legacy_checkpoint), str(cli_out), "--report", str(report_path),
         "--reader-gain", "0.5", "--envs", "3", "--steps", "160", "--matches", "2",
         "--minibatch-matches", "2", "--rows-per-match", "20",
         "--heldout-seed", "6"])
    record = json.loads(report_path.read_text())
    assert record["fusion_migration"]["reader_gain"] == 0.5
    cli_actor, _, _ = load_history_checkpoint(cli_out)
    torch.testing.assert_close(cli_actor.reader_norm.weight,
                               0.5 * cli_actor.attention_norm.weight)
    # habit_init_player accepts the converted checkpoint under the split arm
    habit = HistoryPPOConfig(**ARCH, num_envs=4, steps_per_update=60, seed=7, updates=1,
                             epochs=1, minibatch_matches=2, snapshot_updates=0,
                             habit_pack=str(legacy_checkpoint), habit_init=str(out),
                             habit_view="full", habit_axis="pass", habit_strength=4.0,
                             private_fusion="split", private_qk_norm="qk")
    actor, critic, _ = habit_init_player(habit)
    assert actor.reader_norm is not None
    # a habit trainer on the converted init keeps source Adam moments and gets
    # fresh slots for the trailing norm parameters; one update runs on CPU
    trainer = HistoryTrainer(habit, tmp_path / "habit")
    saved = payload["optimizer"]["actor"]["state"]
    assert len(trainer.actor_optimizer.state_dict()["state"]) == len(saved)
    line = trainer.update()
    assert line["update_samples"] > 0 and np.isfinite(line["policy_loss"])
    assert len(trainer.actor_optimizer.state_dict()["state"]) == len(
        trainer.actor.state_dict())


def test_fusion_migration_refuses_non_plain_sources(tmp_path):
    for overrides in (dict(private_qk_norm="qk"), dict(style_input=True),
                      dict(aux_heads="outcome")):
        actor, _ = fresh_player(HistoryPolicyConfig(**BASE, **overrides), 0)
        with pytest.raises(ValueError):
            history_fusion_migrate.validate_source(actor)


PRODUCTION = dict(rollout_kv_cache=True, causal_sdpa=True, rollout_batched_attention=True,
                  learner_batched_attention=True, learner_length_groups=4)


def test_split_habit_collected_log_probs_match_recompute(legacy_checkpoint, tmp_path):
    """CPU version of the pod's CUDA gate for the split arm: the collector's
    production flags (KV cache, batched attention, length groups) agree with
    the learner's recompute through the split fusion."""
    source, _, payload = load_history_checkpoint(legacy_checkpoint)
    split_path = tmp_path / "split-init.pt"
    save_history_checkpoint(split_path,
                            history_fusion_migrate.converted_payload(source, payload, 0.5))
    config = HistoryPPOConfig(**ARCH, num_envs=4, steps_per_update=60, seed=7, updates=1,
                              epochs=1, minibatch_matches=2, snapshot_updates=0,
                              habit_pack=str(legacy_checkpoint), habit_init=str(split_path),
                              habit_view="full", habit_axis="pass", habit_strength=4.0,
                              private_fusion="split", private_qk_norm="qk", **PRODUCTION)
    trainer = HistoryTrainer(config, tmp_path)
    trainer.collect()
    rows = np.arange(len(trainer.buffer))
    assert len(rows) > 0
    log_prob, _ = trainer.recompute_log_probs(rows)
    np.testing.assert_allclose(log_prob.detach().numpy(),
                               trainer.buffer.compact()["logp"][rows], atol=2e-5)
