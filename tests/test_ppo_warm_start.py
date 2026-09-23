"""PPO warm start from a Stage B checkpoint (STAGE_B_TODO B8)."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import gd
import pytest

torch = pytest.importorskip("torch")

from eval.policies import model_digest  # noqa: E402
from train.ckpt import load_checkpoint, save_checkpoint  # noqa: E402
from train.model import GuandanModel, ModelConfig  # noqa: E402
from train.policy import policy_from_payload  # noqa: E402
from train.ppo import PPOConfig, PPOTrainer, load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLAY = int(gd.Phase.Play)
SMALL = ModelConfig(obs_dim=gd.OBS_DIM, act_dim=gd.ACT_DIM, state_width=16, state_layers=1,
                    action_width=16, action_layers=1, fusion_width=16, fusion_layers=1)


def stage_a(path: Path, seed: int) -> Path:
    torch.manual_seed(seed)
    save_checkpoint(path, {"model_config": asdict(SMALL), "model": GuandanModel(SMALL).state_dict(),
                           "optimizer": {}, "config": {"action_mode": "canonical", "seed": 1},
                           "progress": {"updates": 0}, "rng": {}})
    return path


def tiny(init: Path, **overrides) -> PPOConfig:
    base = dict(init_checkpoint=str(init), critic_init="", critic_width=16, critic_layers=1,
                num_envs=8, num_threads=1, torch_threads=1, rollout_steps=160, epochs=1,
                minibatch_size=256, top_k=8, temperature=1.0, policy_lr=1e-2, critic_lr=1e-2,
                max_updates=1, max_seconds=600, checkpoint_seconds=60, snapshot_updates=100,
                tensorboard=False)
    base.update(overrides)
    return PPOConfig(**base)


@pytest.fixture(scope="module")
def setup(tmp_path_factory) -> dict[str, Path]:
    """An M1 stand-in and a one-update PPO run from it whose net has moved."""
    root = tmp_path_factory.mktemp("warm")
    init = stage_a(root / "init.pt", 5)
    run = root / "source"
    PPOTrainer(tiny(init, seed=3), run).run()
    return {"init": init, "source": run / "latest.pt", "root": root}


def batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    counts = [3, 1, 5, 2]
    offsets = torch.tensor([0] + list(torch.tensor(counts).cumsum(0)), dtype=torch.long)
    obs = (torch.rand(len(counts), gd.OBS_DIM, generator=g) < 0.1).float()
    cand = (torch.rand(int(offsets[-1]), gd.ACT_DIM, generator=g) < 0.1).float()
    phase = torch.full((len(counts),), PLAY, dtype=torch.long)
    return obs, cand, offsets, phase


def test_warm_start_copies_policy_critic_and_optimizers(setup, tmp_path):
    source = load_checkpoint(setup["source"], "cpu")
    trainer = PPOTrainer(tiny(setup["init"], warm_start=str(setup["source"])), tmp_path)
    src_policy = policy_from_payload(source)
    obs, cand, offsets, phase = batch()
    with torch.no_grad():
        mine = trainer.policy.logits(obs, cand, offsets, phase, PLAY)
        theirs = src_policy.logits(obs, cand, offsets, phase, PLAY)
        ref = trainer.policy.reference.score_candidates(obs, cand, offsets, phase)
        src_ref = src_policy.reference.score_candidates(obs, cand, offsets, phase)
    assert torch.equal(mine, theirs)
    assert torch.equal(ref, src_ref) and not torch.allclose(mine, ref / 1.0)   # the net moved
    x = torch.rand(6, trainer.critic.config.input_dim)
    from train.critic import Critic, CriticConfig
    critic = Critic(CriticConfig(**source["critic_model_config"]))
    critic.load_state_dict(source["critic"])
    with torch.no_grad():
        assert torch.equal(trainer.critic.eval()(x), critic.eval()(x))
    # The frozen reference is still the init (M1) checkpoint.
    init = load_checkpoint(setup["init"], "cpu")
    assert trainer.policy.reference_checkpoint_id == model_digest(init["model"])
    assert all(torch.equal(v, init["model"][k])
               for k, v in trainer.policy.reference.state_dict().items())
    assert trainer.warm_start_digest == model_digest(source["model"])
    # Fresh counters, loaded Adam states with this config's learning rates.
    assert trainer.progress["updates"] == 0
    assert (trainer.policy_optimizer.state_dict()["state"][0]["step"]
            == source["optimizer"]["state"][0]["step"])
    assert (trainer.critic_optimizer.state_dict()["state"][0]["step"]
            == source["critic_optimizer"]["state"][0]["step"])
    assert trainer.policy_optimizer.param_groups[0]["lr"] == 1e-2
    runtime = json.loads((tmp_path / "runtime-000.json").read_text())
    assert runtime["warm_start_source"] == str(setup["source"])
    assert runtime["warm_start_digest"] == trainer.warm_start_digest
    fresh = PPOTrainer(tiny(setup["init"], warm_start=str(setup["source"]),
                            warm_start_optimizers=False), tmp_path / "fresh-opt")
    assert fresh.policy_optimizer.state_dict()["state"] == {}


def test_warm_start_validation(setup, tmp_path):
    other = stage_a(tmp_path / "other.pt", 99)
    with pytest.raises(ValueError, match="different frozen reference"):
        PPOTrainer(tiny(other, warm_start=str(setup["source"])), tmp_path / "a")
    with pytest.raises(ValueError, match="critic_init"):
        tiny(setup["init"], warm_start=str(setup["source"]), critic_init="x.pt").validate()
    with pytest.raises(ValueError, match="Stage B"):
        PPOTrainer(tiny(setup["init"], warm_start=str(setup["init"])), tmp_path / "b")
    with pytest.raises(ValueError, match="critic config"):
        PPOTrainer(tiny(setup["init"], warm_start=str(setup["source"]), critic_width=32),
                   tmp_path / "c")
    with pytest.raises(ValueError, match="temperature/top_k"):
        PPOTrainer(tiny(setup["init"], warm_start=str(setup["source"]), temperature=0.5),
                   tmp_path / "d")
    with pytest.warns(UserWarning, match="temperature"):
        trainer = PPOTrainer(tiny(setup["init"], warm_start=str(setup["source"]), temperature=0.5,
                                  warm_start_policy_override=True), tmp_path / "e")
    assert trainer.policy.config.temperature == 0.5


def test_resume_of_a_warm_started_run(setup, tmp_path):
    config = tiny(setup["init"], warm_start=str(setup["source"]), rollout_steps=96)
    run = tmp_path / "run"
    PPOTrainer(config, run).run()
    saved = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    digest = model_digest(load_checkpoint(setup["source"], "cpu")["model"])
    assert saved["warm_start_source"] == str(setup["source"])
    assert saved["warm_start_digest"] == digest
    assert saved["progress"]["updates"] == 1
    trainer = PPOTrainer(replace(config, max_updates=2), run, resume=run / "latest.pt")
    assert trainer.progress["resumes"] == 1 and trainer.warm_start_digest == digest
    assert all(torch.equal(v, saved["model"][k]) for k, v in trainer.net.state_dict().items())
    trainer.run()
    again = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    assert again["progress"]["updates"] == 2 and again["warm_start_digest"] == digest
    assert json.loads((run / "runtime-001.json").read_text())["warm_start_digest"] == digest
    with pytest.raises(ValueError, match="resume config differs at warm_start"):
        PPOTrainer(replace(config, warm_start=""), run, resume=run / "latest.pt")


def test_league_opponent_with_warm_start_smoke(setup, tmp_path):
    init = setup["init"]
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({
        "config": {"seed": 3, "uniform_mix": 1.0, "snapshot_every": 1, "max_snapshots": 4,
                   "snapshot_temperatures": [None]},
        "entries": [{"spec": str(init), "name": "m1"}, {"spec": "greedy"},
                    {"spec": "styled:bomb-happy"}]}))
    config = tiny(init, warm_start=str(setup["source"]), opponent=f"league:{pool}",
                  rollout_steps=200, max_updates=2)
    run = tmp_path / "run"
    progress = PPOTrainer(config, run).run()
    assert progress["updates"] == 2
    lines = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(lines) == 2 and all(line["update_samples"] > 0 for line in lines)
    saved = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    assert saved["league_state"]["snapshots"] and saved["warm_start_source"] == str(setup["source"])


def test_b8_configs():
    for name, opponent in (("b8-league.json", "league:train/configs/league-b8.json"),
                           ("b8-frozen.json", "frozen")):
        config = load_config(ROOT / "train/configs" / name)
        config.validate()
        assert config.opponent == opponent
        assert (config.init_checkpoint, config.warm_start) == ("artifacts/final.pt",
                                                               "artifacts/ppo-a.pt")
        assert config.critic_init == "" and config.kl_coef == 0.0
        assert (config.policy_lr, config.temperature, config.max_seconds) == (1e-5, 0.02, 21600)
    a, b = (asdict(load_config(ROOT / "train/configs" / n)) for n in ("b8-league.json",
                                                                    "b8-frozen.json"))
    assert {k for k in a if a[k] != b[k]} == {"opponent"}
    pool = json.loads((ROOT / "train/configs/league-b8.json").read_text())
    assert pool["config"]["snapshot_temperatures"] == [None, 0.02]
    assert all(not e["spec"].startswith(".work") for e in pool["entries"])
