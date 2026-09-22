"""Stage B perfect-information critic: input layout, leakage, loader, fit (B2)."""
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import shutil

import gd
import numpy as np
import pytest
import torch

import eval.collect_critic as collector
from eval.policies import load_policy
from train import critic
from train.ckpt import save_checkpoint
from train.model import GuandanModel, ModelConfig
from train.policy import StageBPolicy
from train.rollout_buffer import HIDDEN_DIM, RolloutBuffer, RolloutBufferConfig

ROOT = Path(__file__).resolve().parents[1]
REAL_DATA = next((base / ".work/critic-m1" for base in (ROOT, *ROOT.parents)
                  if (base / ".work/critic-m1/manifest.json").is_file()), None)


def tiny_checkpoint(path: Path) -> Path:
    torch.manual_seed(5)
    config = ModelConfig(state_width=8, state_layers=1, action_width=8,
                         action_layers=1, fusion_width=8, fusion_layers=1)
    model = GuandanModel(config)
    save_checkpoint(path, {"model_config": asdict(config), "model": model.state_dict(),
        "optimizer": {}, "config": {"seed": 7, "action_mode": "canonical"},
        "progress": {"updates": 0}, "rng": {}})
    return path


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("critic-fit")
    checkpoint = tiny_checkpoint(root / "frozen.pt")
    output = root / "data"
    collector.collect_critic(checkpoint, output, rounds=40, num_envs=6, seed=517,
                             max_seconds=120, shard_rounds=4,
                             split_fractions=(0.5, 0.25, 0.25))
    return {"root": root, "data": output, "checkpoint": checkpoint}


def fill_buffer(obs, hidden):
    """Store rows in a RolloutBuffer with one dummy candidate each."""
    n = len(obs)
    buf = RolloutBuffer(RolloutBufferConfig(num_envs=n, obs_dim=obs.shape[1], act_dim=gd.ACT_DIM,
                                            max_steps=n, max_candidates=n, max_trajectories=2 * n))
    stored = buf.add_batch(
        learner=np.ones(n, bool), env_id=np.arange(n),
        match_id=np.zeros(n, np.int64), round_index=np.zeros(n, np.int64),
        seat=np.zeros(n, np.int64), phase=np.full(n, critic.PLAY),
        obs=obs, hidden_counts=hidden, cand=np.zeros((n, gd.ACT_DIM), np.uint8),
        offsets=np.arange(n + 1), chosen=np.zeros(n, np.int64), logp=np.zeros(n, np.float32))
    assert stored == n
    return buf


# ------------------------------------------------------------ input layout
def test_input_matches_rollout_buffer_on_real_env_batches():
    env = gd.VecEnv(num_envs=8, num_threads=1, seed=23)
    env.reset()
    for _ in range(40):  # past the deal, into mid-round play
        env.step(np.array(env.pending().greedy_choice, dtype=np.int32, copy=True))
    batch = env.pending()
    obs = np.array(batch.obs, dtype=np.uint8, copy=True)
    hidden = np.array(batch.hidden_counts, dtype=np.uint8, copy=True)
    assert hidden.reshape(len(obs), -1).any()
    buf = fill_buffer(obs, hidden)
    steps = np.arange(len(obs))
    expected = buf.critic_input(steps)
    mine = critic.critic_input(obs, hidden)
    assert mine.dtype == expected.dtype == np.uint8
    assert mine.tobytes() == expected.tobytes()
    # The dataset path (bit-packed observation) gives the same float tensor.
    packed = critic.to_tensor(np.packbits(obs, axis=1), hidden.reshape(len(obs), -1),
                              zero_hidden=False)
    assert torch.equal(packed, buf.gather(steps)["critic_obs"])


@pytest.mark.skipif(REAL_DATA is None, reason="B1 dataset .work/critic-m1 not present")
def test_input_matches_rollout_buffer_on_collected_shard():
    path = critic.split_paths(REAL_DATA, "test")[0]
    rows = critic.read_shard(path)
    take = slice(0, 512)
    obs = np.unpackbits(rows["obs_bits"][take], axis=1, count=gd.OBS_DIM)
    hidden = rows["hidden"][take]
    buf = fill_buffer(obs, hidden.reshape(-1, 3, 54))
    steps = np.arange(len(obs))
    assert critic.critic_input(obs, hidden).tobytes() == buf.critic_input(steps).tobytes()
    assert torch.equal(critic.to_tensor(rows["obs_bits"][take], hidden, zero_hidden=False),
                       buf.gather(steps)["critic_obs"])


def test_public_mode_zeroes_only_the_hidden_columns():
    rng = np.random.default_rng(0)
    obs = rng.integers(0, 2, (5, gd.OBS_DIM), dtype=np.uint8)
    hidden = rng.integers(1, 3, (5, 3, 54), dtype=np.uint8)
    public = critic.critic_input(obs, hidden, zero_hidden=True)
    assert public.shape == (5, gd.OBS_DIM + HIDDEN_DIM)
    assert np.array_equal(public[:, :gd.OBS_DIM], obs) and not public[:, gd.OBS_DIM:].any()


# ----------------------------------------------------------------- leakage
def test_hidden_counts_never_reach_the_policy():
    config = critic.CriticConfig()
    net = GuandanModel()
    # The policy's only per-decision state input is the actor observation.
    assert net.state_tower[0].in_features == gd.OBS_DIM == config.obs_dim
    assert critic.Critic(config).body[0].in_features == gd.OBS_DIM + HIDDEN_DIM
    for function in (GuandanModel.forward, GuandanModel.score_candidates,
                     StageBPolicy.act, StageBPolicy.evaluate, StageBPolicy.logits,
                     StageBPolicy.prune):
        names = set(inspect.signature(function).parameters)
        assert not {n for n in names if "hidden" in n or "critic" in n}, function
    # policy_view of a critic input is exactly the observation.
    rng = np.random.default_rng(1)
    obs = rng.integers(0, 2, (4, gd.OBS_DIM), dtype=np.uint8)
    hidden = rng.integers(0, 3, (4, 3, 54), dtype=np.uint8)
    assert np.array_equal(critic.policy_view(critic.critic_input(obs, hidden)), obs)
    # The critic shares no tensor with the policy network.
    policy_ptrs = {p.data_ptr() for p in net.parameters()}
    assert not policy_ptrs & {p.data_ptr() for p in critic.Critic(config).parameters()}


def test_policy_output_is_independent_of_hidden_counts_in_the_buffer():
    torch.manual_seed(3)
    small = ModelConfig(state_width=16, state_layers=1, action_width=8, action_layers=1,
                        fusion_width=8, fusion_layers=1)
    policy = StageBPolicy.from_model(GuandanModel(small))
    rng = np.random.default_rng(2)
    obs = rng.integers(0, 2, (6, gd.OBS_DIM), dtype=np.uint8)
    outputs = []
    for fill in (0, 2):
        buf = fill_buffer(obs, np.full((6, 3, 54), fill, np.uint8))
        batch = buf.gather(np.arange(6))
        assert bool((batch["critic_obs"][:, gd.OBS_DIM:] == fill).all())
        with torch.no_grad():
            outputs.append(policy.logits(batch["obs"], batch["cand"], batch["offsets"],
                                         batch["phase"]))
    assert torch.equal(outputs[0], outputs[1])


def test_critic_checkpoint_is_not_loadable_as_a_policy(tmp_path):
    model = critic.Critic(critic.CriticConfig(width=8, layers=1))
    path = tmp_path / "critic.pt"
    optimizer = torch.optim.Adam(model.parameters())
    save_checkpoint(path, critic.checkpoint_payload(
        model, optimizer, critic.FitConfig(width=8, layers=1), {"step": 0},
        np.random.default_rng(0)))
    with pytest.raises(ValueError, match="stage"):
        load_policy(str(path))


# ------------------------------------------------------------------ loader
def test_loader_respects_split_boundaries(dataset, tmp_path):
    data = dataset["data"]
    keys = {}
    for split in collector.SPLITS:
        paths = critic.split_paths(data, split)
        assert paths and all(p.parent.name == split for p in paths)
        rows = critic.load_rows(paths)
        keys[split] = set(rows["match"].tolist())
        manifest_matches = {m for s in json.loads(
            (data / "manifest.json").read_text())["shards"] if s["path"].startswith(split)
            for m in s["matches"]}
        assert len(keys[split]) == len(manifest_matches)
    assert not keys["train"] & keys["val"]
    assert not keys["train"] & keys["test"]
    assert not keys["val"] & keys["test"]
    # Streamed training batches only ever contain train rows.
    train = critic.load_rows(critic.split_paths(data, "train"))
    train_rows = {row.tobytes() + h.tobytes()
                  for row, h in zip(train["obs_bits"], train["hidden"])}
    other = critic.load_rows(critic.split_paths(data, "test"))
    assert not train_rows & {r.tobytes() + h.tobytes()
                             for r, h in zip(other["obs_bits"], other["hidden"])}
    stream = critic.stream_batches(critic.split_paths(data, "train"), 32,
                                   np.random.default_rng(0), mix_shards=2)
    for _ in range(3 * len(train["target"]) // 32):
        bits, hidden, _ = next(stream)
        assert all(r.tobytes() + h.tobytes() in train_rows for r, h in zip(bits, hidden))
    stream.close()
    # A shard in the wrong split directory is refused.
    wrong = tmp_path / "val" / "shard-00000.npz"
    wrong.parent.mkdir()
    shutil.copy(critic.split_paths(data, "train")[0], wrong)
    with pytest.raises(ValueError, match="split"):
        critic.read_shard(wrong)


def test_load_rows_cap_keeps_whole_rounds(dataset):
    paths = critic.split_paths(dataset["data"], "train")
    full = critic.load_rows(paths)
    capped = critic.load_rows(paths, max_decisions=100)
    assert 100 <= len(capped["target"]) < len(full["target"])
    last = capped["round"][-1]
    assert (full["round"] == last).sum() == (capped["round"] == last).sum()


# --------------------------------------------------------- fit and report
def test_tiny_fit_reduces_loss_and_checkpoint_round_trips(dataset):
    out = dataset["root"] / "fit"
    config = critic.FitConfig(data=str(dataset["data"]), output=str(out), width=32, layers=1,
                              batch_size=64, lr=3e-3, max_steps=120, eval_every=40,
                              patience=100, val_decisions=10**6, mix_shards=2, seed=11,
                              threads=1)
    val = critic.load_rows(critic.split_paths(dataset["data"], "val"))
    torch.manual_seed(config.seed)
    untrained = critic.Critic(critic.CriticConfig(width=32, layers=1))
    before = float(np.mean((critic.predict(untrained, val["obs_bits"], val["hidden"],
                                           zero_hidden=False) - val["target"]) ** 2))
    summary = critic.fit(config, log=lambda _: None)
    assert summary["steps"] == 120 and summary["stop_reason"] == "max_steps"
    assert summary["best"]["val_mse"] < before
    assert summary["history"][-1]["train_mse"] < summary["history"][0]["train_mse"]
    model, payload = critic.load_critic(out / "best.pt")
    assert payload["stage"] == "critic" and payload["hidden_mode"] == "perfect"
    assert payload["progress"]["step"] == summary["best"]["step"]
    pred = critic.predict(model, val["obs_bits"], val["hidden"], zero_hidden=False)
    assert np.isclose(np.mean((pred - val["target"]) ** 2), summary["best"]["val_mse"],
                      rtol=1e-5)
    again = critic.Critic(model.config)
    again.load_state_dict(payload["model"])
    for a, b in zip(model.state_dict().values(), again.state_dict().values()):
        assert torch.equal(a, b)
    with pytest.raises(ValueError, match="not a critic"):
        critic.load_critic(dataset["checkpoint"])

    result = critic.report({"perfect": out / "best.pt"}, dataset["data"], resamples=200,
                           threads=1)
    assert set(result["play"]) == {"overall", *collector.STAGES}
    overall = result["play"]["overall"]
    assert overall["decisions"] == result["play_decisions"]
    low, high = overall["ci95_rounds"]["perfect-m1_max_q"]
    point = overall["mse"]["perfect-m1_max_q"]
    assert low <= point <= high
    assert np.isclose(point, overall["mse"]["perfect"] - overall["mse"]["m1_max_q"])
    assert result["tribute"]["decisions"] == result["decisions"] - result["play_decisions"]


def test_bootstrap_resamples_whole_clusters():
    # Two clusters with constant errors 0 and 2: any interval stays within [0, 2]
    # and a single cluster gives a degenerate interval.
    errors = {"e": np.array([0.0, 0.0, 2.0, 2.0])}
    ci = critic.bootstrap(errors, np.ones(4), cluster=np.array([0, 0, 1, 1]),
                          resamples=300, seed=0)
    assert 0.0 <= ci["e"][0] <= ci["e"][1] <= 2.0
    one = critic.bootstrap(errors, np.ones(4), cluster=np.zeros(4), resamples=50, seed=0)
    assert one["e"] == (1.0, 1.0)
