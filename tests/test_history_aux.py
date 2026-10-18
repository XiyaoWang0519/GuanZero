"""Auxiliary heads (train/history_aux.py): targets, insertion invariance, resume."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from train.history_aux import (BOMB, NEXT_LOGITS, NEXT_SLICES, ONE_HOT_FIELDS, MULTI_HOT_FIELDS,
                               belief_loss, next_token_loss, parse_aux_heads, round_outcomes)
from train.history_model import (HistoryPolicyConfig, config_record, extend_actor, fresh_player,
                                 load_history_checkpoint)
from train.history_ppo import HistoryPPOConfig, HistoryTrainer, parse_resume_overrides
from train.logs import TOKEN_DIM


def small_config(**overrides) -> HistoryPPOConfig:
    values = dict(width=32, layers=1, heads=4, num_envs=4, steps_per_update=90, seed=3,
                  updates=1, epochs=1, minibatch_matches=2)
    values.update(overrides)
    return HistoryPPOConfig(**values)


def perfect_logits(tokens: torch.Tensor, scale: float = 30.0) -> torch.Tensor:
    """Logits that put all their mass on each field's target."""
    flat = tokens.reshape(-1, TOKEN_DIM).float()
    logits = torch.full((len(flat), NEXT_LOGITS), -scale)
    for name, field in ONE_HOT_FIELDS:
        logits[:, NEXT_SLICES[name]] = (flat[:, field] * 2 - 1) * scale
    for name, field in MULTI_HOT_FIELDS:
        logits[:, NEXT_SLICES[name]] = (flat[:, field] * 2 - 1) * scale
    bomb = flat[:, BOMB]
    part = torch.full((len(flat), bomb.shape[1] + 1), -scale)
    part[:, :-1] = (bomb * 2 - 1) * scale
    part[:, -1] = torch.where(bomb.sum(-1) > 0, -scale, scale)
    logits[:, NEXT_SLICES["bomb"]] = part
    return logits.reshape(*tokens.shape[:-1], NEXT_LOGITS)


def test_config_canonical_form_and_record():
    assert parse_aux_heads("outcome, next") == ("next", "outcome")
    assert HistoryPolicyConfig(aux_heads="outcome,next").aux_heads == "next,outcome"
    assert "aux_heads" not in config_record(HistoryPolicyConfig())
    assert config_record(HistoryPolicyConfig(aux_heads="belief"))["aux_heads"] == "belief"
    with pytest.raises(ValueError):
        HistoryPolicyConfig(aux_heads="hidden")
    with pytest.raises(ValueError):
        HistoryPolicyConfig(aux_heads="next", window=8)
    with pytest.raises(ValueError):
        HistoryPPOConfig(aux_heads="next", aux_coef=0.0)
    overrides = parse_resume_overrides(["aux_heads=next,belief", "aux_coef=0.05",
                                        "aux_warmup_updates=10"])
    assert overrides == {"aux_heads": "next,belief", "aux_coef": 0.05, "aux_warmup_updates": 10}


def test_heads_leave_base_actor_critic_and_policy_unchanged(tmp_path):
    plain = HistoryTrainer(small_config(), tmp_path / "plain")
    plain.collect()
    rows = plain.buffer.completed_rows()
    inputs, _ = plain.buffer.decision_inputs(rows, plain.store, "cpu")
    base_actor, base_critic = fresh_player(HistoryPolicyConfig(width=32, layers=1, heads=4), 3)
    with_heads, critic = fresh_player(
        HistoryPolicyConfig(width=32, layers=1, heads=4, aux_heads="next,belief,outcome"), 3)
    for name, value in base_actor.state_dict().items():
        assert torch.equal(with_heads.state_dict()[name], value), name
    for name, value in base_critic.state_dict().items():
        assert torch.equal(critic.state_dict()[name], value), name
    added = with_heads.auxiliary_parameter_names()
    assert added == set(with_heads.state_dict()) - set(base_actor.state_dict()) and len(added) == 12
    with torch.no_grad():
        assert torch.equal(base_actor.candidate_log_probs(inputs), with_heads.candidate_log_probs(inputs))
    # extend_actor: same guarantee from a trained actor, heads never removed.
    extended = extend_actor(base_actor, with_heads.config)
    with torch.no_grad():
        assert torch.equal(base_actor.candidate_log_probs(inputs), extended.candidate_log_probs(inputs))
    with pytest.raises(ValueError):
        extend_actor(with_heads, base_actor.config)
    with pytest.raises(ValueError):
        extend_actor(base_actor, replace(with_heads.config, width=64))


def test_next_token_loss_alignment_and_padding():
    torch.manual_seed(0)
    tokens = torch.zeros(2, 5, TOKEN_DIM, dtype=torch.uint8)
    for b in range(2):
        for s in range(5):
            tokens[b, s, (b + s) % 4] = 1                     # seat
            tokens[b, s, 4 + (s * 7) % 54] = 1                # a card
            tokens[b, s, 112 + (s + b) % 13] = 1              # type
            tokens[b, s, 125 + s % 15] = 1                    # key
            if (s + b) % 13 == 8:
                tokens[b, s, 140 + s % 7] = 1                 # bomb size
            tokens[b, s, 147 + s % 3] = 1                     # wilds
            tokens[b, s, 158 + (20 - s)] = 1                  # cards left
    valid = torch.tensor([[True] * 5, [True, True, True, False, False]])
    exact = next_token_loss(perfect_logits(tokens), tokens, valid)
    assert exact["next_loss"].item() < 1e-6
    assert exact["next_type_accuracy"].item() == 1.0 and exact["next_cards_exact"].item() == 1.0
    # Position s must predict token s: logits built from the shifted stream miss.
    shifted = next_token_loss(perfect_logits(tokens.roll(1, dims=1)), tokens, valid)
    assert shifted["next_loss"].item() > 10 and shifted["next_type_accuracy"].item() < 0.5
    # Padding positions never count: corrupting them leaves the loss alone.
    corrupted = tokens.clone()
    corrupted[1, 3:] = 1
    masked = next_token_loss(perfect_logits(tokens), corrupted, valid)
    assert torch.allclose(masked["next_loss"], exact["next_loss"])
    none = next_token_loss(torch.randn(2, 5, NEXT_LOGITS), tokens, torch.zeros_like(valid))
    assert none["next_loss"].item() == 0.0


def test_belief_loss_values():
    hidden = torch.zeros(3, 3, 54)
    hidden[0, 0, :10] = 1          # seat +1 holds ten singles
    hidden[0, 1, 10:15] = 2        # seat +2 holds five pairs
    hidden[1, 2, :27] = 1
    hidden[2] = 0                  # no unseen card at all
    uniform = belief_loss(torch.zeros(3, 54, 3), hidden)
    assert abs(uniform["belief_loss"].item() - np.log(3)) < 1e-6
    # The hand-size guess: row 0 has 10 copies at +1 and 10 at +2, row 1 all at +3.
    assert abs(uniform["belief_baseline"].item() - (20 * np.log(2)) / 47) < 1e-6
    logits = torch.full((3, 54, 3), -30.0)
    logits[0, :10, 0] = logits[0, 10:15, 1] = logits[1, :27, 2] = 30.0
    assert belief_loss(logits, hidden)["belief_loss"].item() < 1e-6
    flat = belief_loss(torch.zeros(3, 54, 3), hidden.reshape(3, -1))
    assert torch.equal(flat["belief_loss"], uniform["belief_loss"])


def test_round_outcomes_and_trainer_with_heads(tmp_path):
    trainer = HistoryTrainer(small_config(aux_heads="next,belief,outcome", aux_coef=0.1,
                                          learner_length_groups=2, updates=2), tmp_path)
    before = {n: p.detach().clone() for n, p in trainer.actor.named_parameters()}
    line = trainer.update()
    rows = trainer.buffer.completed_rows() if len(trainer.buffer) else None
    for key in ("aux_loss", "next_loss", "belief_loss", "belief_baseline", "outcome_loss",
                "outcome_baseline", "next_type_accuracy", "next_seat_accuracy", "next_cards_exact"):
        assert key in line and np.isfinite(line[key]), key
    assert line["aux_coefficient"] == pytest.approx(0.1) and line["aux_loss"] > 0
    assert line["next_loss"] > 0 and line["belief_loss"] > 0 and line["outcome_loss"] > 0
    changed = [n for n, p in trainer.actor.named_parameters() if not torch.equal(before[n], p)]
    assert set(trainer.actor.auxiliary_parameter_names()) <= set(changed)
    assert any(n.startswith("stream.") for n in changed)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["auxiliary_heads"]["heads"] == ["next", "belief", "outcome"]
    assert manifest["auxiliary_heads"]["start_update"] == 0
    # Outcome labels are the trajectory's return; open rounds are refused.
    trainer.collect()
    buffer = trainer.buffer
    complete = buffer.completed_rows()
    outcomes = round_outcomes(buffer, complete)
    data = buffer.compact()
    expected = np.asarray([buffer.trajectories[int(t)].reward for t in data["traj"][complete]])
    assert np.array_equal(outcomes, expected) and set(np.unique(outcomes)) <= {-3, -2, -1, 1, 2, 3}
    open_rows = np.setdiff1d(np.arange(len(buffer)), complete)
    if open_rows.size:
        with pytest.raises(ValueError):
            round_outcomes(buffer, open_rows[:1])


def test_length_groups_give_the_same_auxiliary_loss(tmp_path):
    trainer = HistoryTrainer(small_config(aux_heads="next,belief,outcome", num_envs=6,
                                          steps_per_update=150), tmp_path)
    trainer.collect()
    values = trainer.refresh_values()
    trainer.buffer.finalize(values, 1.0, 0.95)
    rows = trainer.buffer.samples
    assert len(rows) > 8
    full = trainer.minibatch_loss(rows)
    trainer.config = replace(trainer.config, learner_length_groups=2)
    grouped = trainer.minibatch_loss(rows)
    for key in ("next_loss", "belief_loss", "outcome_loss", "aux_loss"):
        assert torch.allclose(full[key], grouped[key], rtol=1e-4, atol=1e-5), key
    assert torch.allclose(full["policy_loss"], grouped["policy_loss"], rtol=1e-4, atol=1e-6)


def test_warmup_coefficient(tmp_path):
    trainer = HistoryTrainer(small_config(aux_heads="belief", aux_coef=0.2, aux_warmup_updates=4),
                             tmp_path)
    ramp = []
    for _ in range(5):
        ramp.append(trainer.aux_coefficient())
        trainer.progress["updates"] += 1
    assert ramp == pytest.approx([0.05, 0.1, 0.15, 0.2, 0.2])


def test_resume_adds_heads_without_changing_the_policy(tmp_path):
    output = tmp_path / "plain"
    trainer = HistoryTrainer(small_config(updates=2, snapshot_updates=1), output)
    trainer.run()
    assert trainer.population.models, "a snapshot must exist to exercise the population load"
    checkpoint = output / "latest.pt"
    saved_actor, _, payload = load_history_checkpoint(checkpoint)
    assert "aux_heads" not in payload["model_config"]
    # Cannot remove heads; can add them.
    resumed = HistoryTrainer(small_config(updates=3), tmp_path / "extended", resume=checkpoint,
                             resume_overrides={"aux_heads": "next,belief,outcome",
                                               "aux_coef": 0.05, "aux_warmup_updates": 3})
    assert resumed.actor.aux_head_names == ("next", "belief", "outcome")
    assert resumed.config.aux_coef == 0.05 and resumed.progress["aux_start_update"] == 2
    assert resumed.config_changes[-1]["changes"]["aux_heads"] == ["", "next,belief,outcome"]
    for name, value in saved_actor.state_dict().items():
        assert torch.equal(resumed.actor.state_dict()[name], value), name
    resumed.collect()
    rows = resumed.buffer.completed_rows()
    inputs, _ = resumed.buffer.decision_inputs(rows, resumed.store, "cpu")
    with torch.no_grad():
        assert torch.equal(saved_actor.candidate_log_probs(inputs),
                           resumed.actor.candidate_log_probs(inputs))
    # Adam moments of the base parameters survive; the heads start fresh.
    state = resumed.actor_optimizer.state_dict()["state"]
    base_count = len(list(saved_actor.parameters()))
    assert len(state) == base_count and all(i in state for i in range(base_count))
    saved_moment = payload["optimizer"]["actor"]["state"][0]["exp_avg"]
    assert torch.equal(state[0]["exp_avg"], saved_moment)
    assert len(resumed.population.models) == len(payload["population"]["snapshots"])
    line = resumed.update()
    assert np.isfinite(line["aux_loss"]) and line["aux_coefficient"] == pytest.approx(0.05 / 3)
    resumed.save()
    reloaded, _, record = load_history_checkpoint(tmp_path / "extended" / "latest.pt")
    assert record["model_config"]["aux_heads"] == "next,belief,outcome"
    assert reloaded.aux_head_names == ("next", "belief", "outcome")
    again = HistoryTrainer(small_config(updates=4), tmp_path / "again",
                           resume=tmp_path / "extended" / "latest.pt")
    assert again.actor.aux_head_names == ("next", "belief", "outcome")
    assert again.progress["aux_start_update"] == 2
    with pytest.raises(ValueError):
        HistoryTrainer(small_config(updates=4), tmp_path / "removed",
                       resume=tmp_path / "extended" / "latest.pt",
                       resume_overrides={"aux_heads": "next"})
