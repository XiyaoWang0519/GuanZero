"""CPU acceptance for the wider private decision path."""
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from bench.batched_rollout import SharedPrefixCache
from eval.batched import EvalConfig, play_duplicate_batch
from eval.botzone.numpy_actor import NumpyHistoryActor
from eval.duplicate import generate_deals, play_duplicate_teams
from eval.history_policy import load_history_policy
from eval.private_attention_audit import capture
from train.history_cache import PublicHistoryCache
from train.history_config import build_parser, config_from_args, parse_resume_overrides
from train.history_cuda_graphs import PrivateDecisionGraphs
from train.history_fusion_migrate import validate_source as validate_fusion_source
from train.history_model import (HistoryPolicyConfig, checkpoint_payload, config_record,
                                 count_parameters, fresh_player, save_history_checkpoint)
from train.history_ppo import HistoryPPOConfig, HistoryTrainer
from train.history_sequence_migrate import validate_source as validate_sequence_source


TRUNK = dict(private_hidden=512, private_layers=3, trunk_hidden=1024)
LARGE = dict(width=128, layers=4, heads=8, private_decoder_layers=3, private_query_slots=4,
             action_width=128, fusion_width=1024, critic_width=512, critic_layers=3, **TRUNK)
SMALL = dict(width=32, layers=1, heads=4, private_decoder_layers=2, private_query_slots=2,
             private_hidden=80, private_layers=3, trunk_hidden=112,
             action_width=24, fusion_width=96, critic_width=64, critic_layers=3)
HEADS = dict(response_mode="auxiliary", aux_heads="next,belief,outcome")
DECISION_MODULES = {"private", "query_proj", "trunk_norm", "trunk", "state_norm"}


def test_large_widths_cli_checkpoint_record_and_parameters(tmp_path):
    cli = [part for field, value in LARGE.items()
           for part in ("--" + field.replace("_", "-"), str(value))]
    config = config_from_args(build_parser().parse_args(["--output", str(tmp_path), *cli]))
    policy = config.policy_config()
    assert policy == HistoryPolicyConfig(**LARGE)
    assert policy.state_width == 512
    assert all(config_record(policy)[key] == value for key, value in LARGE.items())
    assert not (TRUNK.keys() & config_record(HistoryPolicyConfig()).keys())
    for field in (*TRUNK, "action_width", "fusion_width", "critic_width", "critic_layers"):
        with pytest.raises(ValueError, match="--resume-set"):
            parse_resume_overrides([f"{field}=256"])

    actor, critic = fresh_player(policy, 17)
    sizes = {name: count_parameters(module) for name, module in actor.named_children()}
    assert sum(sizes[name] for name in ("public", "round_embedding", "phase_embedding",
                                        "stream", "stream_norm")) + actor.bos.numel() == 819_968
    assert sizes["private"] + sizes["query_proj"] == 1_538_176
    assert sum(sizes[name] for name in ("seat", "private_decoder", "attention_norm",
                                        "feed_forward", "output_norm")) == 984_952
    assert sizes["trunk_norm"] + sizes["trunk"] + sizes["state_norm"] == 1_183_488
    assert sizes["action_tower"] == 36_352
    assert sizes["fusion"] == 1_707_009
    assert count_parameters(actor) == 6_269_945
    assert count_parameters(critic) == 1_555_969
    actor, critic = fresh_player(replace(policy, **HEADS), 17)
    assert count_parameters(actor) == 6_578_150
    assert count_parameters(critic) == 1_555_969
    explicit, _ = fresh_player(replace(policy, response_mode="explicit"), 17)
    assert explicit.response_head[0][0].in_features == explicit.response_bridge.out_features == 640


@pytest.mark.parametrize("field,value", [("private_decoder_layers", 0), ("window", 1),
                                         ("style_input", True), ("private_qk_norm", "query"),
                                         ("private_fusion", "split"), ("trunk_hidden", 0),
                                         ("private_layers", 0), ("private_hidden", -1)])
def test_invalid_large_combinations(field, value):
    with pytest.raises(ValueError, match="private_hidden"):
        HistoryPolicyConfig(**(SMALL | {"private_query_slots": 1, field: value}))


@pytest.mark.parametrize("use", [PrivateDecisionGraphs, validate_fusion_source,
                                 validate_sequence_source, PublicHistoryCache,
                                 lambda actor: capture(actor, None),
                                 lambda actor: SharedPrefixCache(actor, []),
                                 lambda actor: NumpyHistoryActor({}, config_record(actor.config))])
def test_unsupported_large_paths_reject(use):
    actor, _ = fresh_player(HistoryPolicyConfig(**SMALL), 17)
    with pytest.raises(ValueError, match="private_hidden|single-layer reader|requires HistoryBelief"):
        use(actor)


def test_large_initialization_and_complete_actor_loss(tmp_path):
    cfg = HistoryPPOConfig(**LARGE, **HEADS, num_envs=3, steps_per_update=90, seed=17,
                           torch_threads=1, minibatch_matches=3, snapshot_updates=0,
                           learner_batched_attention=True, learner_length_groups=2)
    trainer = HistoryTrainer(cfg, tmp_path)
    actor = trainer.actor
    trainer.collect()
    assert trainer.buffer.finalize(trainer.refresh_values(), cfg.gamma, cfg.gae_lambda) > 0
    rows = trainer.buffer.samples
    inputs = trainer.buffer.training_batch(rows, trainer.store, "cpu", fields=()).inputs
    with torch.no_grad():
        expected = actor.state_norm(actor.private(inputs.obs.float()))
        torch.testing.assert_close(actor.decision_states(None, inputs), expected, rtol=0, atol=0)
        logp = actor.candidate_log_probs(inputs)
        deviation = float((logp + inputs.counts[inputs.rows].float().log()).abs().max())
    assert deviation < 0.05
    assert not torch.count_nonzero(actor.trunk[-1].weight)
    assert not torch.count_nonzero(actor.trunk[-1].bias)
    assert not torch.count_nonzero(actor.fusion[-1].bias)
    print(f"initial uniform log-probability deviation: {deviation:.9g}")

    decision_params = {name: p for name, p in actor.named_parameters()
                       if name.split(".")[0] in DECISION_MODULES | {"private_decoder"}}
    for step in (0, 1):
        trainer.actor_optimizer.zero_grad(set_to_none=True)
        terms = trainer.minibatch_loss(rows)
        for loss in ("policy_loss", "entropy", "response_loss", "next_loss", "belief_loss", "outcome_loss"):
            assert torch.isfinite(terms[loss])
        terms["policy_total"].backward()
        for name, p in actor.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert not torch.count_nonzero(actor.response_bridge.weight.grad)
        for name, p in decision_params.items():
            gated = name.startswith(("query_proj.", "trunk_norm.", "trunk.0.", "private_decoder."))
            assert bool(torch.count_nonzero(p.grad)) == (step == 1 or not gated), name
        if step == 0:
            trainer.actor_optimizer.step()
    print(f"gradient tensors checked after one step: {len(decision_params)}")


def test_large_duplicate_scalar_and_batched_match_each_leg(tmp_path):
    # Keep the same seeds and deals across both entries and leave the freshly
    # initialized fusion scale intact. A mismatch must identify its deal/leg.
    torch.set_num_threads(1)
    paths = []
    for seed in (41, 42, 43, 44):
        actor, critic = fresh_player(HistoryPolicyConfig(**LARGE, **HEADS), seed)
        path = tmp_path / f"init-{seed}.pt"
        save_history_checkpoint(path, checkpoint_payload(actor, critic, lineage=f"fresh-{seed}", seed=seed))
        paths.append(path)
    policies = [load_history_policy(path, "cpu") for path in paths]
    assert len({p.checkpoint_id for p in policies}) == 4
    assert all(not p.sample for p in policies)
    team, opponents = tuple(policies[:2]), tuple(policies[2:])
    seed = 20261118
    deals = generate_deals(32, seed)
    scalar = [play_duplicate_teams(deal, team, opponents, seed) for deal in deals]
    batched = play_duplicate_batch(deals, team, opponents, seed,
                                   EvalConfig(batch_size=16, kv_cache=True))
    assert len(scalar) == len(batched) == 32
    for index, (expected, actual) in enumerate(zip(scalar, batched)):
        assert expected.first == actual.first, f"deal {index}, first leg (seed={seed})"
        assert expected.swapped == actual.swapped, f"deal {index}, swapped leg (seed={seed})"


def _large_ddp_rank(rank, world, port, argv):
    from torch import distributed as dist

    from bench.training_revision import fingerprint
    from train.history_ddp import HistoryDDPTrainer, _rank_main

    check_synchronized = HistoryDDPTrainer.check_synchronized

    def check_parameter_bytes(trainer):
        check_synchronized(trainer)
        parameters = tuple(p.detach() for p in trainer.parameters_all())
        assert all(torch.isfinite(p).all() for p in parameters)
        digests = [None] * world
        dist.all_gather_object(digests, fingerprint(parameters))
        assert len(set(digests)) == 1

    HistoryDDPTrainer.check_synchronized = check_parameter_bytes
    _rank_main(rank, world, port, argv)


def test_large_two_rank_cpu_ddp(tmp_path, monkeypatch):
    from train import history_ddp

    monkeypatch.setattr(history_ddp, "_rank_main", _large_ddp_rank)

    cli = [part for field, value in (LARGE | HEADS).items()
           for part in ("--" + field.replace("_", "-"), str(value))]
    assert history_ddp.main(["--output", str(tmp_path), "--world-size", "2", "--device", "cpu",
                             "--num-envs", "3", "--steps-per-update", "90", "--updates", "2",
                             "--epochs", "1", "--minibatch-matches", "2", "--torch-threads", "1",
                             "--seed", "5", "--ddp-global-minibatch", *cli]) == 0
    for root in (tmp_path, tmp_path / "rank-1"):
        lines = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
        assert [line["update"] for line in lines] == [1, 2]
        assert all(line["global_update_samples"] > 0 and line["minibatches"] > 0 for line in lines)
    payload = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    assert payload["model_config"] == config_record(HistoryPolicyConfig(**LARGE, **HEADS))
    assert payload["optimizer"]["actor"]["state"] and payload["optimizer"]["critic"]["state"]
    for key in ("model", "critic"):
        assert all(torch.isfinite(value).all() for value in payload[key].values())
