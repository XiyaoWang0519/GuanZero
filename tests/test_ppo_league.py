"""PPO against the B7 league (STAGE_B_TODO B8 readiness): in-process and actors."""
from dataclasses import asdict
import json
from pathlib import Path

import gd
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eval.league_smoke import drive  # noqa: E402
from train.ckpt import load_checkpoint, save_checkpoint  # noqa: E402
from train.league import League, LeagueConfig  # noqa: E402
from train.model import GuandanModel, ModelConfig  # noqa: E402
from train.ppo import PPOConfig, PPOTrainer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SMALL = ModelConfig(obs_dim=gd.OBS_DIM, act_dim=gd.ACT_DIM, state_width=16, state_layers=1,
                    action_width=16, action_layers=1, fusion_width=16, fusion_layers=1)


@pytest.fixture(scope="module")
def init_checkpoint(tmp_path_factory) -> Path:
    torch.manual_seed(5)
    path = tmp_path_factory.mktemp("init") / "init.pt"
    save_checkpoint(path, {"model_config": asdict(SMALL), "model": GuandanModel(SMALL).state_dict(),
                           "optimizer": {}, "config": {"action_mode": "canonical", "seed": 1},
                           "progress": {"updates": 0}, "rng": {}})
    return path


def pool(directory: Path, init: Path, entries=None, **config) -> Path:
    """A pool with the init checkpoint (argmax: fused with the reference, and
    sampled: a league-owned model), greedy, a fixed style and sampled styles."""
    entries = entries or [{"spec": str(init), "name": "m1"},
                          {"spec": f"sample=0.5:{init}", "name": "m1/T=0.5"},
                          {"spec": "greedy"}, {"spec": "styled:bomb-happy"},
                          {"spec": "sampled-style"}]
    path = directory / "pool.json"
    path.write_text(json.dumps({"config": {"seed": 3, "uniform_mix": 1.0, "snapshot_every": 1,
                                           "max_snapshots": 8,
                                           "snapshot_temperatures": [None, 0.5], **config},
                                "entries": entries}))
    return path


def tiny(init: Path, pool_path: Path, **overrides) -> PPOConfig:
    base = dict(init_checkpoint=str(init), critic_init="", critic_width=16, critic_layers=1,
                opponent=f"league:{pool_path}", num_envs=8, num_threads=1, torch_threads=1,
                rollout_steps=400, epochs=1, minibatch_size=256, top_k=8, temperature=1.0,
                max_updates=100, max_seconds=600, checkpoint_seconds=60, snapshot_updates=100,
                tensorboard=False)
    base.update(overrides)
    return PPOConfig(**base)


def metrics(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]


def games(league: League) -> int:
    return sum(e.games for e in league.entries)


def test_league_opponent_config_validation(init_checkpoint, tmp_path):
    with pytest.raises(ValueError, match="opponent"):
        tiny(init_checkpoint, tmp_path / "x", opponent="league:").validate()
    with pytest.raises(ValueError, match="league_snapshot_every"):
        tiny(init_checkpoint, tmp_path / "x", league_snapshot_every=-1).validate()
    for name in ("league-b8.json",):
        data = json.loads((ROOT / "train/configs" / name).read_text())
        assert {e["spec"] for e in data["entries"]} >= {
            "greedy", "sampled-style", "styled:bomb-happy", "styled:bomb-shy",
            "styled:high-lead", "styled:low-lead", "artifacts/final.pt", "artifacts/dmc-b6.pt"}
        LeagueConfig.from_dict(data["config"])
    PPOConfig(**json.loads((ROOT / "train/configs/ppo-league-smoke.json").read_text())).validate()
    config = json.loads((ROOT / "train/configs/ppo-league.json").read_text())
    PPOConfig(**config).validate()
    assert config["opponent"] == "league:train/configs/league-b8.json"
    assert config["policy_lr"] == 1e-5 and config["temperature"] == 0.02


def test_in_process_league_end_to_end(init_checkpoint, tmp_path):
    run = tmp_path / "run"
    trainer = PPOTrainer(tiny(init_checkpoint, pool(tmp_path, init_checkpoint), max_updates=3),
                         run)
    league = trainer.league
    assert trainer.opponent is league and league.external == {str(init_checkpoint)}
    trainer.run()
    rows = metrics(run)
    assert [r["updates"] for r in rows] == [1, 2, 3]
    assert all(np.isfinite(r["policy_loss"]) for r in rows)
    # A snapshot per update, two entries each (argmax and T=0.5), immutable files.
    assert [r["league_snapshot"] for r in rows] == [
        str(run / "league" / f"update-{u:09d}.pt") for u in (1, 2, 3)]
    assert len(league.snapshots) == 3 and len(league.entries) == 5 + 6
    for path, _ in league.snapshots:
        assert load_checkpoint(path)["stage"] == "ppo"
    assert rows[-1]["league/pool_size"] == 11.0
    # Every finished match was credited to exactly one entry.
    assert trainer.progress["matches"] > 0 and games(league) == trainer.progress["matches"]
    assert rows[-1]["league/greedy/games"] == league.entries[2].games
    # The fused M1 entry never loads a second copy of the reference.
    assert str(init_checkpoint) not in league.cache
    saved = load_checkpoint(run / "latest.pt")
    assert saved["league_state"] == league.state_dict()


def test_fused_m1_entry_plays_exactly_like_the_league_model(init_checkpoint, tmp_path):
    """Requirement: pruning uses the frozen M1 and the league's M1 entry,
    served by the same forward, picks what the league's own model picks."""
    entries = [{"spec": str(init_checkpoint), "name": "m1", "weight": 50.0}, {"spec": "greedy"}]
    path = pool(tmp_path, init_checkpoint, entries, uniform_mix=0.0)
    runs = []
    for fused in (True, False):
        trainer = PPOTrainer(tiny(init_checkpoint, path, rollout_steps=600), tmp_path / str(fused))
        if not fused:
            trainer.league.external = frozenset()
            trainer.league_fused = False
        trainer.collect()
        buffer = trainer.buffer
        runs.append({"chosen": buffer.chosen[:buffer.n_steps].copy(),
                     "logp": buffer.logp[:buffer.n_steps].copy(),
                     "obs": buffer.obs[:buffer.n_steps].copy(),
                     "cache": set(trainer.league.cache), "progress": dict(trainer.progress),
                     "assigned": [e.name for e in trainer.league.assigned]})
    fused, plain = runs
    assert "m1" in fused["assigned"]
    assert fused["cache"] == set() and plain["cache"] == {str(init_checkpoint)}
    assert fused["progress"] == plain["progress"]
    for key in ("chosen", "obs"):
        assert np.array_equal(fused[key], plain[key])
    np.testing.assert_allclose(fused["logp"], plain["logp"], rtol=0, atol=1e-6)


@pytest.mark.parametrize("kind", ["league", "frozen"])
def test_shared_reference_opponents_play_exactly_like_their_own_models(init_checkpoint,
                                                                       tmp_path, kind):
    """Stage B opponents whose reference is the learner's (league snapshots at
    argmax and sampled, a frozen: checkpoint) are scored by the rollout's
    shared reference forward: same choices, draws and league state as when
    each model runs on its own."""
    first = PPOTrainer(tiny(init_checkpoint, pool(tmp_path, init_checkpoint), max_updates=1),
                       tmp_path / "first")
    first.run()
    snapshot = str(first.league.snapshots[0][0])
    first.close()
    entries = [{"spec": snapshot, "name": "snap"},
               {"spec": f"sample=0.5:{snapshot}", "name": "snap/T"},
               {"spec": str(init_checkpoint), "name": "m1"}, {"spec": "greedy"}]
    path = pool(tmp_path, init_checkpoint, entries, uniform_mix=1.0)
    opponent = f"league:{path}" if kind == "league" else f"frozen:{snapshot}"
    runs = []
    for shared in (True, False):
        trainer = PPOTrainer(tiny(init_checkpoint, path, rollout_steps=600, opponent=opponent),
                             tmp_path / str(shared))
        assert trainer.shared_opponent
        trainer.shared_opponent = shared
        deferred = []
        fast_act = trainer._fast_act

        def spy(*args, **kwargs):
            deferred.extend(group.size for _, group in (args[9] if len(args) > 9 else ()))
            return fast_act(*args, **kwargs)

        trainer._fast_act = spy
        trainer.collect()
        buffer = trainer.buffer
        runs.append({"chosen": buffer.chosen[:buffer.n_steps].copy(),
                     "obs": buffer.obs[:buffer.n_steps].copy(),
                     "logp": buffer.logp[:buffer.n_steps].copy(),
                     "ref_logp": buffer.ref_logp[:buffer.n_steps].copy(),
                     "progress": dict(trainer.progress), "deferred": sum(deferred),
                     "league": trainer.league.state_dict() if trainer.league else None,
                     "calls": trainer.league.forward_calls if trainer.league else None})
        trainer.close()
    fused, plain = runs
    assert fused["deferred"] > 0 and plain["deferred"] == 0
    for key in ("progress", "league", "calls"):
        assert fused[key] == plain[key], key
    for key in ("chosen", "obs"):
        assert np.array_equal(fused[key], plain[key]), key
    for key in ("logp", "ref_logp"):
        np.testing.assert_allclose(fused[key], plain[key], rtol=0, atol=1e-6)


def test_only_models_on_the_learners_reference_join_its_forward():
    from types import SimpleNamespace

    from train.ppo import RolloutCollector

    device = torch.device("cpu")
    collector = RolloutCollector()
    collector.device = device
    collector.policy = SimpleNamespace(reference_checkpoint_id="m1")

    def model(reference="m1", heuristic_tribute=True, dev=device, stage_b=True):
        pruned = SimpleNamespace(reference_checkpoint_id=reference) if stage_b else None
        return SimpleNamespace(pruned=pruned, heuristic_tribute=heuristic_tribute, device=dev)

    assert collector._shares_reference(model())
    assert not collector._shares_reference(model(reference="b6"))
    assert not collector._shares_reference(model(reference=None))
    assert not collector._shares_reference(model(heuristic_tribute=False))
    assert not collector._shares_reference(model(dev=torch.device("meta")))
    assert not collector._shares_reference(model(stage_b=False))
    collector.policy = SimpleNamespace(reference_checkpoint_id=None)
    assert not collector._shares_reference(model(reference=None))


def test_actor_league_end_to_end_and_snapshots_reach_actors(init_checkpoint, tmp_path):
    run = tmp_path / "run"
    trainer = PPOTrainer(tiny(init_checkpoint, pool(tmp_path, init_checkpoint), actor_processes=2,
                              max_updates=10), run)
    try:
        master = trainer.league
        assert trainer.opponent is None and master.env is None
        first = trainer.update()
        assert first["league_snapshot"].endswith("update-000000001.pt")
        names = {e.name for e in master.entries}
        assert {"snapshot@1", "snapshot@1/T=0.5"} <= names
        # Not in the actors yet: they adopt the pool at the next boundary.
        for state in trainer.actors.request("league_state"):
            assert not state["snapshots"]
        second = trainer.update()
        for state in trainer.actors.request("league_state"):
            assert [p for p, _ in state["snapshots"]] == [p for p, _ in master.snapshots][:1]
            assert {"snapshot@1", "snapshot@1/T=0.5"} <= {e["name"] for e in state["entries"]}
        assert trainer.progress["matches"] > 0 and games(master) == trainer.progress["matches"]
        assert second["league/pool_size"] == 9.0
        assert second["league/actor_active_models_max"] <= master.config.max_active_models
        assert np.isfinite(second["policy_loss"])
    finally:
        trainer.close()


def test_aggregated_weights_equal_the_single_process_update():
    """Results from two actors, applied in actor order by the learner, give the
    table one process gets from the same results in that order; one actor's
    results reproduce that actor's own in-process updates exactly."""
    specs = ["greedy", "styled:bomb-happy", "styled:low-lead", "sampled-style"]
    config = LeagueConfig(seed=1, uniform_mix=0.3, ema_alpha=0.2)
    team = np.arange(8) % 2
    actors, results = [], []
    for seed in (11, 12):
        league = League(specs, LeagueConfig(**{**asdict(config), "seed": seed}))
        league.record_results = True
        env = gd.VecEnv(8, num_threads=1, seed=seed)
        env.reset()
        drive(env, league, team, lambda batch, rows: np.asarray(batch.greedy_choice)[rows],
              matches=12)
        results.append(league.pop_results())
        actors.append(league)
    assert all(results)
    # One actor: the learner's table equals the actor's in-process table.
    solo = League(specs, config)
    solo.apply_results(results[0])
    assert [(e.beat_ema, e.games, e.learner_wins) for e in solo.entries] == \
        [(e.beat_ema, e.games, e.learner_wins) for e in actors[0].entries]
    # Two actors: the same sequential EMA as one process given these results.
    master = League(specs, config)
    for part in results:
        master.apply_results(part)
    expected = {name: [config.initial_beat_rate, 0, 0] for name in specs}
    for name, won in results[0] + results[1]:
        row = expected[name]
        row[0] += config.ema_alpha * ((0.0 if won else 1.0) - row[0])
        row[1] += 1
        row[2] += int(won)
    for entry in master.entries:
        assert entry.beat_ema == pytest.approx(expected[entry.name][0], rel=0, abs=1e-15)
        assert [entry.games, entry.learner_wins] == expected[entry.name][1:]
    # An actor adopting the table samples from exactly the learner's weights.
    actors[1].load_state_dict(master.state_dict(include_rng=False))
    np.testing.assert_array_equal(actors[1].weights(), master.weights())
    # Results of an entry that left the pool are dropped, not misattributed.
    assert master.apply_results([("snapshot@9", True)]) == 0


def test_league_state_round_trip_keeps_running_entries():
    league = League(["greedy", "styled:bomb-happy"], LeagueConfig(seed=4, uniform_mix=1.0),
                    loader=lambda spec, config, seed: None)
    league.add_snapshot("/nonexistent/a.pt", name="a")
    state = league.state_dict()
    copy = League(["greedy", "styled:bomb-happy"], LeagueConfig(seed=99))
    kept = copy.entries[1]
    copy.load_state_dict(state)
    assert copy.state_dict() == state and copy.entries[1] is kept
    assert [e.name for e in copy.snapshots[0][1]] == ["a"]
    assert copy.rng.integers(0, 2**32) == league.rng.integers(0, 2**32)


def test_in_process_resume_restores_the_league(init_checkpoint, tmp_path):
    run = tmp_path / "run"
    path = pool(tmp_path, init_checkpoint)
    first = PPOTrainer(tiny(init_checkpoint, path, max_updates=2), run)
    first.run()
    saved = first.league.state_dict()
    resumed = PPOTrainer(tiny(init_checkpoint, path, max_updates=3), run,
                         resume=run / "latest.pt")
    assert resumed.league.state_dict() == saved     # EMA, tallies, snapshots and RNG
    resumed.run()
    assert [p for p, _ in resumed.league.snapshots] == [
        str(run / "league" / f"update-{u:09d}.pt") for u in (1, 2, 3)]
    assert [r["updates"] for r in metrics(run)] == [1, 2, 3]


def test_actor_resume_restores_the_league_and_actor_samplers(init_checkpoint, tmp_path):
    run = tmp_path / "run"
    path = pool(tmp_path, init_checkpoint)
    first = PPOTrainer(tiny(init_checkpoint, path, actor_processes=2, max_updates=2,
                            rollout_steps=300), run)
    first.run()
    payload = load_checkpoint(run / "latest.pt")
    assert payload["league_state"] == first.league.state_dict()
    assert len(payload["league_actor_rng"]) == 2
    resumed = PPOTrainer(tiny(init_checkpoint, path, actor_processes=2, max_updates=3,
                              rollout_steps=300), run, resume=run / "latest.pt")
    try:
        assert resumed.league.state_dict(include_rng=False) == \
            {k: v for k, v in payload["league_state"].items() if k != "rng"}
        assert resumed.actors.request("league_rng") == payload["league_actor_rng"]
        resumed.update()
        for state in resumed.actors.request("league_state"):
            assert len(state["snapshots"]) == 2   # restored pool reached the actors
    finally:
        resumed.close()
