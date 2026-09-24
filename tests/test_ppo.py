"""Stage B PPO learner (STAGE_B_TODO B5): config, rollout, update, resume, play."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import gd
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eval.arena import evaluate_matches  # noqa: E402
from eval.policies import GreedyPolicy, PrunedPolicy, load_policy  # noqa: E402
from train.ckpt import save_checkpoint  # noqa: E402
from train.model import GuandanModel, ModelConfig, select_actions  # noqa: E402
from train.opponents import FrozenModelOpponent, GreedyOpponent, OpponentRows  # noqa: E402
from train.policy import segment_log_softmax  # noqa: E402
from train.ppo import (PPOConfig, PPOTrainer, assign_advantage_filter, load_config,
                       policy_terms)  # noqa: E402
from train.rollout_buffer import RolloutBuffer, RolloutBufferConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
M1_FINAL = next((p / ".work/runpod/artifacts/pilot/final.pt"
                 for p in Path(__file__).resolve().parents
                 if (p / ".work/runpod/artifacts/pilot/final.pt").exists()), None)
SMALL = ModelConfig(obs_dim=gd.OBS_DIM, act_dim=gd.ACT_DIM, state_width=16, state_layers=1,
                    action_width=16, action_layers=1, fusion_width=16, fusion_layers=1)


@pytest.fixture(scope="module")
def init_checkpoint(tmp_path_factory) -> Path:
    """A tiny Stage A (DMC) checkpoint standing in for the M1 final."""
    torch.manual_seed(5)
    model = GuandanModel(SMALL)
    path = tmp_path_factory.mktemp("init") / "init.pt"
    save_checkpoint(path, {"model_config": asdict(SMALL), "model": model.state_dict(),
                           "optimizer": {}, "config": {"action_mode": "canonical", "seed": 1},
                           "progress": {"updates": 0}, "rng": {}})
    return path


def tiny(init: Path, **overrides) -> PPOConfig:
    base = dict(init_checkpoint=str(init), critic_init="", critic_width=16,
                critic_layers=1, opponent="greedy", num_envs=4, num_threads=1, torch_threads=1,
                rollout_steps=48, epochs=2, minibatch_size=64, top_k=8, temperature=1.0, max_updates=100,
                max_seconds=600, checkpoint_seconds=60, snapshot_updates=100, tensorboard=False)
    base.update(overrides)
    return PPOConfig(**base)


class RecordingOpponent(GreedyOpponent):
    """Greedy play that checks the OpponentSource call order as it goes."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.bound = False
        self.live: dict[int, int | None] = {}   # env -> match id seen since its start, None = new
        self.ended: set[int] = set()
        self.rows: list[tuple[int, int]] = []

    def bind(self, env, learner_team: np.ndarray) -> None:
        assert not self.bound and not self.calls
        self.bound = True
        self.learner_team = np.asarray(learner_team).copy()
        self.calls.append(("bind",))

    def on_match_start(self, env_ids: np.ndarray) -> None:
        assert self.bound
        for env in np.asarray(env_ids).tolist():
            self.live[env] = None
            self.ended.discard(env)
        self.calls.append(("start", tuple(np.asarray(env_ids).tolist())))

    def on_match_end(self, env_ids: np.ndarray, learner_won: np.ndarray) -> None:
        assert len(env_ids) == len(learner_won)
        for env in np.asarray(env_ids).tolist():
            assert env in self.live and env not in self.ended
            self.ended.add(env)
        self.calls.append(("end", tuple(np.asarray(env_ids).tolist())))

    def act(self, rows: OpponentRows) -> np.ndarray:
        for env, match, seat in zip(rows.env_id.tolist(), rows.match_id.tolist(), rows.seat.tolist()):
            # A row belongs to a started, not yet ended match, and a match id
            # changes only across an on_match_start for that environment.
            assert env in self.live and env not in self.ended
            if self.live[env] is None:
                self.live[env] = match
            assert self.live[env] == match
            assert seat % 2 != self.learner_team[env]
            self.rows.append((env, seat))
        self.calls.append(("act", rows.rows))
        return super().act(rows)


# ------------------------------------------------------------------ config
def test_config_validation_and_shipped_configs(init_checkpoint):
    PPOConfig().validate()
    tiny(init_checkpoint).validate()
    for name in ("ppo-smoke.json", "ppo.json"):
        load_config(ROOT / "train/configs" / name).validate()
    bad = [dict(clip=0.0), dict(clip=1.5), dict(gamma=0.0), dict(gae_lambda=1.5),
           dict(policy_lr=-1e-5), dict(critic_lr=float("nan")), dict(entropy_coef=-0.1),
           dict(kl_coef=float("inf")), dict(kl_anneal_updates=0), dict(epochs=0),
           dict(minibatch_size=0), dict(num_envs=1), dict(temperature=0.0), dict(top_k=0),
           dict(checkpoint_seconds=900), dict(tribute_policy="random"),
           dict(opponent="styled"), dict(opponent="frozen:"), dict(action_mode="abstract"),
           dict(init_checkpoint=""), dict(buffer_steps=-1)]
    bad += [dict(advantage_filter_quantile=-0.1), dict(advantage_filter_quantile=1.1),
            dict(advantage_filter_quantile=float("nan")),
            dict(advantage_filter_min_magnitude=-0.1)]
    for override in bad:
        with pytest.raises(ValueError):
            tiny(init_checkpoint, **override).validate()
    # Zero learning rates and coefficients are legal: they are the identity test.
    tiny(init_checkpoint, policy_lr=0.0, critic_lr=0.0, kl_coef=0.0, entropy_coef=0.0).validate()
    with pytest.raises(TypeError):
        PPOConfig(unknown_knob=1)


# ----------------------------------------------------------------- rollout
def test_rollout_stores_learner_rows_only_and_drives_the_contract(init_checkpoint, tmp_path):
    opponent = RecordingOpponent()
    trainer = PPOTrainer(tiny(init_checkpoint, rollout_steps=1500), tmp_path, opponent=opponent)
    learner_seen = []
    original = trainer.buffer.add_batch

    def spy(**kwargs):
        learner_seen.extend(zip(kwargs["env_id"].tolist(), kwargs["seat"].tolist()))
        return original(**kwargs)
    trainer.buffer.add_batch = spy
    trainer.collect()
    buf = trainer.buffer
    n = buf.n_steps
    assert n > 0 and n == len(learner_seen) == trainer.progress["learner_decisions"]
    team_of_step = buf.traj_team[buf.traj[:n]]
    env_of_step = buf.traj_env[buf.traj[:n]]
    assert np.all(buf.seat[:n] % 2 == team_of_step)
    assert np.all(team_of_step == trainer.learner_team[env_of_step])
    assert np.all(buf.phase[:n] == int(gd.Phase.Play))  # heuristic tribute stores nothing
    # Opponent rows never enter the buffer, and every opponent row went to the source.
    assert all(seat % 2 != trainer.learner_team[env] for env, seat in opponent.rows)
    assert all(seat % 2 == trainer.learner_team[env] for env, seat in learner_seen)
    assert len(opponent.rows) + len(learner_seen) <= trainer.progress["decisions"]
    # Both teams are learner somewhere; stored candidate sets respect top_k + pass.
    assert set(trainer.learner_team.tolist()) == {0, 1}
    assert buf.cand_count[:n].max() <= trainer.config.top_k + 1
    # Call order: bind, every env started, then per step ends before starts before act.
    assert opponent.calls[0] == ("bind",)
    assert opponent.calls[1] == ("start", tuple(range(trainer.config.num_envs)))
    assert any(call[0] == "end" for call in opponent.calls), "no match ended; lengthen the rollout"
    for before, after in zip(opponent.calls, opponent.calls[1:]):
        assert (before[0], after[0]) not in (("start", "end"), ("start", "start"))
    assert trainer.progress["matches"] == sum(len(c[1]) for c in opponent.calls if c[0] == "end")


def test_default_frozen_opponent_is_the_reference_and_completes_matches(init_checkpoint, tmp_path):
    trainer = PPOTrainer(tiny(init_checkpoint, opponent="frozen", rollout_steps=1500), tmp_path)
    assert isinstance(trainer.opponent, FrozenModelOpponent)
    assert trainer.opponent.model is trainer.policy.reference
    for name in ("bind", "on_match_start", "act", "on_match_end"):
        assert callable(getattr(trainer.opponent, name))
    trainer.collect()
    assert trainer.progress["matches"] > 0


def test_frozen_model_opponent_plays_argmax_q_and_heuristic_tribute():
    torch.manual_seed(2)
    model = GuandanModel(SMALL)
    opponent = FrozenModelOpponent(model)
    env = gd.VecEnv(8, num_threads=1, seed=4)
    env.reset()
    checked_play = checked_tribute = 0
    for _ in range(400):
        batch = env.pending()
        env.drain_finished_rounds()
        offsets = np.asarray(batch.offsets, np.int64)
        rows = OpponentRows(obs=np.asarray(batch.obs), cand=np.asarray(batch.cand),
                            offsets=np.asarray(batch.offsets), env_id=np.asarray(batch.env_id),
                            seat=np.asarray(batch.seat), phase=np.asarray(batch.phase),
                            match_id=np.asarray(batch.match_id),
                            greedy_choice=np.asarray(batch.greedy_choice),
                            styled_choice=np.asarray(batch.styled_choice))
        choices = opponent.act(rows)
        with torch.no_grad():
            q = model.score_candidates(torch.tensor(rows.obs), torch.tensor(rows.cand),
                                       torch.tensor(offsets), torch.tensor(rows.phase))
        best = select_actions(q, torch.as_tensor(offsets)).numpy()
        play = rows.phase == int(gd.Phase.Play)
        assert np.array_equal(choices[play], best[play])
        assert np.array_equal(choices[~play], rows.greedy_choice[~play])
        checked_play += int(play.sum())
        checked_tribute += int((~play).sum())
        env.step(choices)
    assert checked_play and checked_tribute


# ------------------------------------------------------------------ update
def synthetic_filter_buffer(advantages: list[float], pass_steps: tuple[int, ...] = ()) -> RolloutBuffer:
    n = len(advantages)
    buffer = RolloutBuffer(RolloutBufferConfig(num_envs=2, obs_dim=gd.OBS_DIM,
                                              act_dim=gd.ACT_DIM, max_steps=n,
                                              max_candidates=n, max_trajectories=2))
    buffer.n_samples = n
    buffer.samples[:n] = np.arange(n)
    buffer.advantage[:n] = advantages
    buffer.cand_start[:n] = np.arange(n)
    buffer.cand_count[:n] = 1
    buffer.cand[list(pass_steps), 108] = 1
    return buffer


def test_advantage_filter_uses_update_wide_raw_magnitudes_and_keeps_pass():
    a = synthetic_filter_buffer([0.1, 0.4], pass_steps=(0,))
    b = synthetic_filter_buffer([-0.5, 0.0])
    stats = assign_advantage_filter([a, b], 0.5, 0.3)
    assert stats["policy_filter_threshold"] == pytest.approx(0.3)
    assert a.policy_keep[:2].tolist() == [True, True]
    assert b.policy_keep[:2].tolist() == [True, False]
    assert stats["policy_filter_kept_fraction"] == 0.75
    assert stats["policy_filter_pass_kept"] == 1
    assert stats["policy_filter_positive_kept"] == 2
    assert stats["policy_filter_negative_kept"] == 1
    assert assign_advantage_filter([a, b], 0.0, 0.0)["policy_filter_kept_fraction"] == 1
    assert b.policy_keep[:2].all()  # default retains even zero-advantage rows


def test_advantage_filter_all_removed_keeps_one_defined_zero_loss_row():
    buffer = synthetic_filter_buffer([0.0, 0.0, 0.0])
    stats = assign_advantage_filter([buffer], 0.9, 1.0)
    assert buffer.policy_keep[:3].tolist() == [True, False, False]
    assert stats["policy_filter_kept_fraction"] == pytest.approx(1 / 3)
    assert stats["policy_filter_positive_kept"] == stats["policy_filter_negative_kept"] == 0


def test_filtered_update_preserves_other_losses_and_resumes(init_checkpoint, tmp_path):
    config = tiny(init_checkpoint, advantage_filter_quantile=0.8,
                  advantage_filter_min_magnitude=0.05, rollout_steps=160,
                  max_updates=1)
    trainer = PPOTrainer(config, tmp_path)
    trainer.collect()
    trainer.refresh_values()
    assert trainer.buffer.finalize(config.gamma, config.gae_lambda) > 0
    assign_advantage_filter([trainer.buffer], config.advantage_filter_quantile,
                            config.advantage_filter_min_magnitude)
    mb = next(trainer.buffer.minibatches(10**6, np.random.default_rng(0)))
    assert mb["policy_keep"].any() and not mb["policy_keep"].all()
    filtered = trainer.minibatch_loss(mb, 0.1)
    trainer.config = replace(config, advantage_filter_quantile=0.0)
    unfiltered = trainer.minibatch_loss(mb, 0.1)
    assert not torch.allclose(filtered["policy_loss"], unfiltered["policy_loss"])
    for name in ("value_loss", "hidden_loss", "finish_loss", "kl_ref", "entropy"):
        assert torch.equal(filtered[name], unfiltered[name]), name
    trainer.close()

    run = tmp_path / "run"
    PPOTrainer(config, run).run()
    record = metrics(run / "metrics.jsonl")[0]
    assert 0 < record["policy_filter_kept_fraction"] < 1
    assert record["policy_filter_pass_kept"] >= 0
    saved = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    assert saved["config"]["advantage_filter_quantile"] == 0.8
    resumed = PPOTrainer(replace(config, max_updates=2), run, resume=run / "latest.pt")
    assert resumed.progress["resumes"] == 1
    resumed.run()
    assert metrics(run / "metrics.jsonl")[-1]["updates"] == 2
    with pytest.raises(ValueError, match="resume config differs at advantage_filter_quantile"):
        PPOTrainer(replace(config, advantage_filter_quantile=0.5),
                   run, resume=run / "latest.pt")


def snapshot(module) -> dict:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def test_zero_lr_update_is_identity_and_policy_is_reference_softmax(init_checkpoint, tmp_path):
    config = tiny(init_checkpoint, policy_lr=0.0, critic_lr=0.0, kl_coef=0.0, entropy_coef=0.0,
                  rollout_steps=160)
    trainer = PPOTrainer(config, tmp_path)
    before = snapshot(trainer.net)
    stats = trainer.update()
    assert stats["update_samples"] > 0 and trainer.progress["optimizer_steps"] > 0
    after = trainer.net.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    # Nothing moved, so every ratio in every epoch is 1.
    assert stats["clip_fraction"] == 0.0 and abs(stats["approx_kl"]) < 1e-9
    assert stats["ratio_deviation"] < 1e-5 and abs(stats["kl_ref"]) < 1e-6
    # The policy is exactly softmax(Q_M1 / t) on the stored pruned sets.
    trainer.collect()
    buf = trainer.buffer
    trainer.refresh_values()
    assert buf.finalize() > 0
    mb = next(buf.minibatches(10**6, np.random.default_rng(0)))
    with torch.no_grad():
        log_prob, _, log_probs = policy_terms(trainer.policy, mb["obs"], mb["cand"], mb["offsets"],
                                              mb["phase"], mb["chosen"], trainer.phase_code)
        q = trainer.policy.reference.score_candidates(mb["obs"], mb["cand"], mb["offsets"],
                                                      mb["phase"])
    expected = segment_log_softmax(q / config.temperature, mb["offsets"])
    assert torch.allclose(log_probs, expected, atol=1e-6)
    assert torch.allclose(log_prob, mb["logp"], atol=1e-5)


@pytest.mark.skipif(M1_FINAL is None, reason="M1 final checkpoint not available")
def test_zero_lr_step_from_the_m1_final_keeps_softmax_q_over_t(tmp_path):
    config = replace(tiny(M1_FINAL, policy_lr=0.0, critic_lr=0.0, kl_coef=0.0, entropy_coef=0.0,
                          rollout_steps=120, top_k=32, torch_threads=2),
                     temperature=0.5)
    trainer = PPOTrainer(config, tmp_path)
    before = snapshot(trainer.net)
    stats = trainer.update()
    assert stats["update_samples"] > 0
    assert all(torch.equal(before[k], v) for k, v in trainer.net.state_dict().items())
    assert stats["ratio_deviation"] < 1e-4 and abs(stats["kl_ref"]) < 1e-6
    trainer.collect()
    trainer.refresh_values()
    trainer.buffer.finalize()
    mb = next(trainer.buffer.minibatches(10**6, np.random.default_rng(0)))
    with torch.no_grad():
        _, _, log_probs = policy_terms(trainer.policy, mb["obs"], mb["cand"], mb["offsets"],
                                       mb["phase"], mb["chosen"], trainer.phase_code)
        q = trainer.policy.reference.score_candidates(mb["obs"], mb["cand"], mb["offsets"], mb["phase"])
    assert torch.allclose(log_probs, segment_log_softmax(q / 0.5, mb["offsets"]), atol=1e-5)


def test_ratio_is_one_on_the_first_epoch(init_checkpoint, tmp_path):
    # One minibatch per epoch: the whole first epoch runs before any step.
    trainer = PPOTrainer(tiny(init_checkpoint, policy_lr=1e-2, minibatch_size=10**6,
                              rollout_steps=160), tmp_path)
    stats = trainer.update()
    assert stats["update_samples"] > 0
    assert stats["first_ratio_max_deviation"] < 1e-5
    # A large step afterwards does move the policy away from the behaviour.
    assert stats["ratio_deviation"] > 1e-3


# ------------------------------------------------------------- checkpoints
def metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


LOSSES = ("policy_loss", "value_loss", "entropy", "kl_ref", "approx_kl", "hidden_loss",
          "finish_loss", "update_samples", "clip_fraction", "explained_variance")


def test_checkpoint_resume_reproduces_the_next_update(init_checkpoint, tmp_path):
    config = tiny(init_checkpoint, policy_lr=1e-3, critic_lr=1e-3, rollout_steps=96, max_updates=1)
    first, twin = tmp_path / "first", tmp_path / "twin"
    PPOTrainer(config, first).run()
    PPOTrainer(config, twin).run()
    a, b = metrics(first / "metrics.jsonl")[0], metrics(twin / "metrics.jsonl")[0]
    assert {k: a[k] for k in LOSSES} == {k: b[k] for k in LOSSES}, "fresh runs are not seeded"
    saved = torch.load(first / "latest.pt", map_location="cpu", weights_only=False)
    assert saved["stage"] == "ppo" and saved["progress"]["updates"] == 1
    assert {"critic", "critic_optimizer", "critic_model_config", "sampler", "reference_model"} <= saved.keys()
    records = []
    for name in ("resume-a", "resume-b"):
        run = tmp_path / name
        run.mkdir()
        shutil.copy(first / "latest.pt", run / "latest.pt")
        trainer = PPOTrainer(replace(config, max_updates=2), run, resume=run / "latest.pt")
        assert trainer.progress["updates"] == 1 and trainer.progress["resumes"] == 1
        assert all(torch.equal(v, saved["model"][k]) for k, v in trainer.net.state_dict().items())
        assert all(torch.equal(v, saved["critic"][k]) for k, v in trainer.critic.state_dict().items())
        assert (trainer.policy_optimizer.state_dict()["state"][0]["step"]
                == saved["optimizer"]["state"][0]["step"])
        trainer.run()
        records.append(metrics(run / "metrics.jsonl")[-1])
    assert records[0]["updates"] == 2
    assert {k: records[0][k] for k in LOSSES} == {k: records[1][k] for k in LOSSES}
    with pytest.raises(ValueError, match="resume config differs"):
        PPOTrainer(replace(config, clip=0.3), tmp_path / "resume-a",
                   resume=tmp_path / "resume-a" / "latest.pt")
    with pytest.raises(ValueError, match="use --resume"):
        PPOTrainer(config, first)


def test_saved_checkpoint_loads_through_load_policy_and_plays(init_checkpoint, tmp_path):
    trainer = PPOTrainer(tiny(init_checkpoint, max_updates=1, rollout_steps=96), tmp_path)
    trainer.run()
    assert (tmp_path / "checkpoints" / "step-000000000.pt").exists()
    policy = load_policy(str(tmp_path / "latest.pt"))
    assert isinstance(policy, PrunedPolicy) and policy.stage == "ppo"
    assert policy.base_checkpoint_id == trainer.policy.reference_checkpoint_id
    assert policy.heuristic_tribute
    result = evaluate_matches(policy, GreedyPolicy(), count=1, seed=3)
    assert result["matches"] == 1 and result["rounds"] >= 1
