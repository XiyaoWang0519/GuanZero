"""Stage B opponent league (STAGE_B_TODO B7): `train.league.League`."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import gd
import numpy as np
import pytest

from eval.league_smoke import drive, opponent_rows
from train.league import League, LeagueConfig, load_pool
from train.styles import StyleSpace, neutral

PLAY = int(gd.Phase.Play)


class StubModel:
    """Network stand-in: plays candidate 0 and counts its batched calls."""

    heuristic_tribute = True

    def __init__(self, spec: str, log: list) -> None:
        self.spec, self.log = spec, log

    def choose(self, obs, cand, offsets, phase):
        assert obs.shape[0] == len(offsets) - 1 == len(phase)
        assert cand.shape[0] == offsets[-1]
        assert (np.asarray(phase) == PLAY).all()  # heuristic tribute stays off the net
        self.log.append((self.spec, len(phase)))
        return np.zeros(len(phase), np.int32)


def stub_loader(log: list):
    loaded = []

    def load(spec, config, seed):
        loaded.append(spec)
        return StubModel(spec, log)
    load.loaded = loaded
    return load


class RecordingEnv:
    """Proxy that keeps every `set_styles` array, for style-row assertions."""

    def __init__(self, env) -> None:
        self.env, self.calls = env, []
        self.current = None

    def set_styles(self, styles):
        self.current = np.array(styles, copy=True)
        self.calls.append(self.current)
        self.env.set_styles(styles)

    def __getattr__(self, name):
        return getattr(self.env, name)


def greedy_learner(batch, index):
    return np.asarray(batch.greedy_choice)[index].astype(np.int32)


def test_one_opponent_and_style_per_match_on_real_env():
    num_envs = 8
    raw = gd.VecEnv(num_envs=num_envs, num_threads=2, seed=5)
    raw.reset()
    env = RecordingEnv(raw)
    league = League(["greedy", "styled:bomb-happy", "styled:high-lead", "sampled-style"],
                    LeagueConfig(seed=3, uniform_mix=1.0))
    team = np.array([0, 1] * (num_envs // 2))
    seen: dict[tuple[int, int], tuple] = {}
    styled_rows = [0]
    neutral_row = neutral()

    def check(batch, learner, choice):
        styles = env.current
        for r in range(batch.rows):
            e, m = int(batch.env_id[r]), int(batch.match_id[r])
            opp = 1 - team[e]
            entry = league.assigned[e]
            key = (entry.name, styles[e, opp].tobytes(), styles[e, opp + 2].tobytes())
            assert seen.setdefault((e, m), key) == key       # fixed for the whole match
            np.testing.assert_array_equal(styles[e, opp], styles[e, opp + 2])
            np.testing.assert_array_equal(styles[e, team[e]], neutral_row)
            np.testing.assert_array_equal(styles[e, team[e] + 2], neutral_row)
            if not learner[r] and entry.kind in ("styled", "sampled_style"):
                assert choice[r] == batch.styled_choice[r]
                styled_rows[0] += 1
            if not learner[r] and entry.kind == "greedy":
                assert choice[r] == batch.greedy_choice[r]

    done = drive(env, league, team, greedy_learner, matches=40, on_step=check)
    assert done >= 40
    matches_per_env = {}
    for e, _ in seen:
        matches_per_env[e] = matches_per_env.get(e, 0) + 1
    assert min(matches_per_env.values()) >= 2   # every env crossed a match boundary
    assert styled_rows[0] > 0
    assert len({k[0] for k in seen.values()}) == 4
    for styles in env.calls:  # learner rows never left neutral in any call
        for e in range(num_envs):
            np.testing.assert_array_equal(styles[e, team[e]], neutral_row)
            np.testing.assert_array_equal(styles[e, team[e] + 2], neutral_row)


def test_act_rejects_learner_rows_and_handles_empty_batches():
    env = gd.VecEnv(num_envs=4, num_threads=1, seed=2)
    env.reset()
    league = League(["greedy", "styled:bomb-shy"], LeagueConfig(seed=0))
    team = np.array([0, 1, 0, 1])
    league.bind(env, team)
    league.on_match_start(np.arange(4))
    batch = env.pending()
    seat, env_id = np.asarray(batch.seat), np.asarray(batch.env_id)
    learner = np.flatnonzero(seat % 2 == team[env_id])
    opponent = np.flatnonzero(seat % 2 != team[env_id])
    assert len(learner) and len(opponent)
    with pytest.raises(ValueError, match="learner-seat"):
        league.act(opponent_rows(batch, learner))
    out = league.act(opponent_rows(batch, opponent))
    assert out.shape == (len(opponent),) and out.dtype == np.int32
    empty = league.act(opponent_rows(batch, np.zeros(0, np.int64)))
    assert empty.shape == (0,)


class NullEnv:
    def set_styles(self, styles):
        assert styles.dtype == np.float32 and styles.flags.c_contiguous


def test_active_model_cap_never_exceeded_under_adversarial_sampling():
    log: list = []
    specs = [f"net{i}.pt" for i in range(12)] + ["greedy", "styled:low-lead"]
    config = LeagueConfig(seed=9, max_active_models=4, model_cache_size=5, uniform_mix=0.0)
    league = League(specs, config, loader=stub_loader(log))
    num_envs = 32
    league.bind(NullEnv(), np.zeros(num_envs, np.int64))
    rng = np.random.default_rng(1)
    open_matches = np.zeros(num_envs, bool)
    for _ in range(3000):
        # Adversary: every inactive network looks maximally dangerous.
        for entry in league.entries:
            entry.beat_ema = 1.0 if entry.kind == "network" and entry.spec not in league.active \
                else 0.0
        envs = rng.choice(num_envs, size=int(rng.integers(1, 6)), replace=False)
        ending = envs[open_matches[envs]]
        if len(ending):
            league.on_match_end(ending, rng.random(len(ending)) < 0.5)
        league.on_match_start(envs)
        open_matches[envs] = True
        networks = {e.spec for e in league.assigned if e is not None and e.kind == "network"}
        assert networks == set(league.active)
        assert len(networks) <= config.max_active_models
        counts = {s: sum(e is not None and e.spec == s for e in league.assigned)
                  for s in networks}
        assert counts == league.active
        # LRU keeps every active model and respects the bound.
        for spec in networks:
            league._model(spec)
        assert set(league.active) <= set(league.cache)
        assert len(league.cache) <= config.model_cache_size
    assert len(league.active) == config.max_active_models  # the adversary did push


def test_weights_bounded_floored_and_deterministic():
    config = LeagueConfig(seed=4, ema_alpha=0.1, weight_floor=0.05, uniform_mix=0.0)
    specs = ["greedy", "styled:bomb-happy", "styled:bomb-shy", "sampled-style"]

    def run(seed):
        league = League(specs, LeagueConfig(**{**asdict(config), "seed": seed}))
        league.bind(NullEnv(), np.array([0, 1] * 8))
        league.on_match_start(np.arange(16))
        rng = np.random.default_rng(0)
        history = []
        for _ in range(400):
            env = rng.integers(16, size=3)
            env = np.unique(env)
            # bomb-happy always beats the learner, everyone else always loses.
            won = np.array([league.assigned[e].name != "styled:bomb-happy" for e in env])
            before = np.array([e.beat_ema for e in league.entries])
            league.on_match_end(env, won)
            after = np.array([e.beat_ema for e in league.entries])
            assert (np.abs(after - before) <= len(env) * config.ema_alpha + 1e-12).all()
            assert ((after >= 0) & (after <= 1)).all()
            weights = league.weights()
            assert abs(weights.sum() - 1) < 1e-9
            assert (weights > 0).all()
            league.on_match_start(env)
            history.append(([e.name for e in league.assigned], weights.copy()))
        return league, history

    league, first = run(4)
    _, again = run(4)
    _, other = run(5)
    assert [h[0] for h in first] == [h[0] for h in again]
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(first, again))
    assert [h[0] for h in first] != [h[0] for h in other]
    weights = league.weights()
    names = [e.name for e in league.entries]
    hard = names.index("styled:bomb-happy")
    assert weights[hard] == weights.max()
    # Losing entries sit at the floor: their priority is weight_floor, never 0.
    priorities = np.maximum([e.beat_ema for e in league.entries], config.weight_floor)
    np.testing.assert_allclose(weights, priorities / priorities.sum())
    assert weights.min() >= config.weight_floor / len(names)
    stats = league.stats()
    assert stats["league/styled:bomb-happy/learner_win_rate"] == 0.0
    assert stats["league/active_models"] == 0.0


def test_uniform_mix_blends_with_uniform():
    league = League(["greedy", "styled:bomb-happy"], LeagueConfig(uniform_mix=0.5,
                                                                   weight_floor=0.05))
    league.entries[0].beat_ema, league.entries[1].beat_ema = 0.0, 1.0
    np.testing.assert_allclose(league.weights(),
                               0.5 * np.array([0.05, 1.0]) / 1.05 + 0.25)


def test_snapshot_eviction_keeps_fixed_entries_and_running_matches():
    log: list = []
    config = LeagueConfig(seed=1, max_snapshots=2, snapshot_every=50,
                          snapshot_temperatures=(None, 1.0), uniform_mix=1.0)
    league = League(["greedy", "styled:bomb-happy"], config, loader=stub_loader(log))
    assert not league.should_snapshot(0) and league.should_snapshot(100)
    assert not league.should_snapshot(101)
    league.add_snapshot("run/snap-0050.pt")
    league.bind(NullEnv(), np.zeros(64, np.int64))
    league.on_match_start(np.arange(64))
    playing_old = [e for e in range(64) if league.assigned[e].snapshot]
    assert playing_old
    league.add_snapshot("run/snap-0100.pt")
    league.add_snapshot("run/snap-0150.pt")
    names = [e.name for e in league.entries]
    assert names[:2] == ["greedy", "styled:bomb-happy"]
    assert names[2:] == ["snapshot:snap-0100.pt", "snapshot:snap-0100.pt/T=1",
                         "snapshot:snap-0150.pt", "snapshot:snap-0150.pt/T=1"]
    assert [e.spec for e in league.entries[3::2]] == ["sample=1:run/snap-0100.pt",
                                                      "sample=1:run/snap-0150.pt"]
    # The evicted snapshot stays assigned until those matches end, then is released.
    assert any("snap-0050" in spec for spec in league.active)
    league.on_match_end(np.arange(64), np.ones(64, bool))
    league.on_match_start(np.arange(64))
    assert not any("snap-0050" in spec for spec in league.active)
    assert all("snap-0050" not in e.spec for e in league.assigned)
    for _ in range(5):
        league.add_snapshot(f"run/snap-{_}.pt")
        assert sum(e.snapshot for e in league.entries) == 4
        assert [e.name for e in league.entries[:2]] == ["greedy", "styled:bomb-happy"]


def test_heldout_style_region_is_never_sampled():
    league = League(["sampled-style", "greedy"], LeagueConfig(seed=11, uniform_mix=1.0))
    team = np.array([0, 1] * 32)
    league.bind(NullEnv(), team)
    space = StyleSpace.default()
    sampled = 0
    for _ in range(60):
        league.on_match_start(np.arange(64))
        league.on_match_end(np.arange(64), np.zeros(64, bool))
        for e in range(64):
            opp = 1 - team[e]
            if league.assigned[e].kind == "sampled_style":
                style = league.styles[e, opp]
                assert space.in_bounds(style) and not space.in_heldout(style)
                np.testing.assert_array_equal(style, league.styles[e, opp + 2])
                sampled += 1
            else:
                np.testing.assert_array_equal(league.styles[e, opp], neutral())
    assert sampled > 1000


def test_network_entries_batch_one_forward_per_model_per_step():
    log: list = []
    loader = stub_loader(log)
    league = League(["a.pt", "b.pt", "sample=2:c.pt", "greedy"],
                    LeagueConfig(seed=2, uniform_mix=1.0), loader=loader)
    num_envs = 24
    env = gd.VecEnv(num_envs=num_envs, num_threads=2, seed=8)
    env.reset()
    team = np.arange(num_envs) % 2
    steps_with_nets = [0]

    def check(batch, learner, choice):
        calls = log[:]
        log.clear()
        expected: dict[str, int] = {}
        for r in np.flatnonzero(~learner):
            entry = league.assigned[int(batch.env_id[r])]
            if entry.kind == "network" and int(batch.phase[r]) == PLAY:
                expected[entry.spec] = expected.get(entry.spec, 0) + 1
                assert choice[r] == 0
        assert sorted(calls) == sorted(expected.items())  # one call per model, all its rows
        steps_with_nets[0] += len(expected) > 1

    drive(env, league, team, greedy_learner, steps=400, on_step=check)
    assert steps_with_nets[0] > 0
    assert sorted(set(loader.loaded)) == sorted(loader.loaded)  # cached, loaded once each
    assert league.forward_calls > 0


def test_torch_model_batched_choice_matches_per_decision_policy(tmp_path):
    torch = pytest.importorskip("torch")

    from eval.policies import load_policy
    from train.ckpt import rng_state, save_checkpoint
    from train.league import TorchModel
    from train.model import GuandanModel, ModelConfig
    from train.policy import PolicyConfig, StageBPolicy

    small = ModelConfig(obs_dim=gd.OBS_DIM, act_dim=gd.ACT_DIM, state_width=16,
                        state_layers=1, action_width=16, action_layers=1,
                        fusion_width=16, fusion_layers=1)
    torch.manual_seed(0)
    model = GuandanModel(small)
    dmc = tmp_path / "dmc.pt"
    save_checkpoint(dmc, {"model_config": asdict(small), "model": model.state_dict(),
                          "optimizer": {}, "config": {}, "progress": {}, "rng": {}})
    policy = StageBPolicy.from_model(model, PolicyConfig(top_k=4))
    with torch.no_grad():
        next(policy.net.parameters()).add_(0.2)
    ppo = tmp_path / "ppo.pt"
    save_checkpoint(ppo, policy.checkpoint_payload(
        optimizer={}, config={}, progress={}, rng=rng_state(np.random.default_rng(0))))
    env = gd.VecEnv(num_envs=16, num_threads=1, seed=3)
    env.reset()
    for _ in range(30):
        batch = env.pending()
        env.step(np.array(batch.greedy_choice, np.int32, copy=True))
        env.drain_finished_rounds()
    batch = env.pending()
    play = np.flatnonzero(np.asarray(batch.phase) == PLAY)
    rows = opponent_rows(batch, play)
    for spec in (str(dmc), str(ppo)):
        batched = TorchModel(spec, "cpu", seed=0).choose(
            rows.obs, rows.cand, rows.offsets.astype(np.int64), rows.phase)
        single = load_policy(spec)
        for i, r in enumerate(play):
            o, c = rows.offsets[i], rows.offsets[i + 1]
            obs = torch.as_tensor(rows.obs[i:i + 1])
            cand = torch.as_tensor(rows.cand[o:c])
            off = torch.tensor([0, c - o])
            phase = torch.tensor([PLAY])
            with torch.inference_mode():
                if hasattr(single, "stage_b"):
                    keep, pruned = single.stage_b.prune(obs, cand, off, phase)
                    logits = single.stage_b.logits(obs, cand[keep], pruned, phase)
                    want = int(keep[logits.argmax()])
                else:
                    want = int(single.model.score_candidates(obs, cand, off, phase).argmax())
            assert batched[i] == want


def test_pool_file_round_trip(tmp_path):
    path = tmp_path / "pool.json"
    path.write_text(json.dumps({
        "config": {"seed": 3, "max_active_models": 2, "model_cache_size": 3,
                   "snapshot_temperatures": [None, 0.5]},
        "entries": [{"spec": "greedy"}, {"spec": "styled:bomb-shy", "name": "shy"},
                    {"spec": "sampled-style", "weight": 2.0}, {"spec": "x.pt"}]}))
    config, entries = load_pool(path)
    assert config.max_active_models == 2 and config.snapshot_temperatures == (None, 0.5)
    league = League.from_file(path, loader=stub_loader([]), uniform_mix=0.25)
    assert league.config.uniform_mix == 0.25 and league.config.seed == 3
    assert [e.name for e in league.entries] == ["greedy", "shy", "sampled-style", "x.pt"]
    assert [e.kind for e in league.entries] == ["greedy", "styled", "sampled_style", "network"]
    assert league.entries[2].weight == 2.0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"entries": [{"spec": "greedy", "temp": 1}]}))
    with pytest.raises(ValueError, match="unknown pool entry keys"):
        load_pool(bad)
    bad.write_text(json.dumps({"config": {"nope": 1}, "entries": []}))
    with pytest.raises(ValueError, match="unknown LeagueConfig keys"):
        load_pool(bad)
    with pytest.raises(ValueError, match="duplicate"):
        League(["greedy", "greedy"])
    with pytest.raises(ValueError, match="random"):
        League(["random"])
    with pytest.raises(ValueError, match="model_cache_size"):
        LeagueConfig(max_active_models=4, model_cache_size=2)
    assert Path(path).exists()


def test_match_results_attribute_fifo_whatever_the_call_order():
    league = League(["greedy", "styled:bomb-happy"], LeagueConfig(seed=0, uniform_mix=1.0))
    league.bind(NullEnv(), np.array([0]))
    league.on_match_start(np.array([0]))
    first = league.assigned[0]
    league.on_match_start(np.array([0]))      # next match opened before the end report
    league.on_match_end(np.array([0]), np.array([False]))
    assert first.games == 1 and first.learner_wins == 0
    league.on_match_start(np.array([0]))      # again early: two open, still legal
    with pytest.raises(RuntimeError, match="without ending"):
        league.on_match_start(np.array([0]))
