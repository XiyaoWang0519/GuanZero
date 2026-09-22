"""Style space, schema 2 round logs, styled collection and the coverage report."""
from dataclasses import asdict
import json

import gd
import numpy as np
import pytest
import torch

import eval.collect_belief as collection
import eval.style_coverage as coverage
from train import styles as style_lib
from train.belief_experiment import load_dataset, split_rounds
from train.buffer import Decision
from train.ckpt import save_checkpoint
from train.logs import DRIVER_BOT, DRIVER_POLICY, TOKEN_DIM, save_round
from train.model import GuandanModel, ModelConfig
from train.tribute_data import engine_source_digest

needs_styled_bot = pytest.mark.skipif(
    not style_lib.styled_bot_available(),
    reason="the task-1 styled-bot binding (gd.STYLE_DIM) is not built")


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(5)
    config = ModelConfig(state_width=8, state_layers=1, action_width=8,
                         action_layers=1, fusion_width=8, fusion_layers=1)
    model = GuandanModel(config)
    path = tmp_path / "frozen.pt"
    save_checkpoint(path, {"model_config": asdict(config), "model": model.state_dict(),
        "optimizer": {}, "config": {"seed": 7, "action_mode": "canonical"},
        "progress": {}, "rng": {}})
    return path


# --- style space ----------------------------------------------------------

def test_slot_layout_is_generic_over_the_style_dimension():
    for dim in (style_lib.FALLBACK_STYLE_DIM, style_lib.FALLBACK_STYLE_DIM + 3):
        mapping = style_lib.slot_map(dim)
        assert len(mapping) == dim and sorted(mapping.values()) == list(range(dim))
        assert mapping["bomb_threshold"] == 0
        assert mapping["temperature"] == dim - 1
        assert mapping["follow_aggression"] == dim - 4
    with pytest.raises(ValueError):
        style_lib.slot_map(len(style_lib.TAIL_SLOT_NAMES) + 1)


def test_region_sampling_stays_in_its_box_and_regions_are_disjoint():
    space = style_lib.StyleSpace.default()
    rng = np.random.default_rng(3)
    train = [space.sample(rng, "train") for _ in range(300)]
    heldout = [space.sample(rng, "heldout") for _ in range(300)]
    assert all(space.in_bounds(v) for v in train + heldout)
    assert all(v.dtype == np.float32 and v.shape == (space.dim,) for v in train + heldout)
    assert not any(space.in_heldout(v) for v in train)
    assert all(space.in_heldout(v) for v in heldout)
    assert {style_lib.region_of(space, v) for v in train} == {"train"}
    assert {style_lib.region_of(space, v) for v in heldout} == {"heldout"}
    # Each held-out slot really is confined to its sub-range.
    stack = np.stack(heldout)
    for name, low, high in space.heldout:
        column = stack[:, space.index(name)]
        assert column.min() >= low and column.max() <= high


def test_mixed_region_draws_from_both_regions():
    space = style_lib.StyleSpace.default()
    rng = np.random.default_rng(11)
    labels = [style_lib.region_of(space, space.sample(rng, "mixed", 0.5)) for _ in range(400)]
    assert set(labels) == {"train", "heldout"}
    assert 0.3 < labels.count("heldout") / len(labels) < 0.7


def test_style_space_rejects_impossible_heldout_rules():
    with pytest.raises(ValueError):
        style_lib.StyleSpace.default(heldout={"bomb_threshold": (0.0, 1.0)})
    with pytest.raises(ValueError):
        style_lib.StyleSpace.default(heldout={"no_such_slot": (0.1, 0.2)})
    with pytest.raises(ValueError):
        style_lib.StyleSpace.default(heldout={})


def test_neutral_and_unknown_styles_have_the_declared_shape():
    dim = style_lib.style_dim()
    assert style_lib.neutral(dim).shape == (dim,)
    assert np.isfinite(style_lib.neutral(dim)).all()
    assert np.isnan(style_lib.unknown_style(dim)).all()
    assert style_lib.region_of(style_lib.StyleSpace.default(),
                               style_lib.unknown_style(dim)) == "unknown"


def test_match_style_rng_is_deterministic_per_match():
    first = style_lib.match_style_rng(5, 1, 2).random(4)
    assert np.array_equal(first, style_lib.match_style_rng(5, 1, 2).random(4))
    assert not np.array_equal(first, style_lib.match_style_rng(5, 1, 3).random(4))
    assert not np.array_equal(first, style_lib.match_style_rng(5, 2, 2).random(4))


# --- behaviour histograms -------------------------------------------------

class FakeAction:
    def __init__(self, kind, key=0):
        self.type, self.key = kind, key


class FakeEvent:
    def __init__(self, seat, kind, key=0, step=0, env_id=0, match_id=0, round_index=0):
        self.seat, self.action, self.step = seat, FakeAction(kind, key), step
        self.env_id, self.match_id, self.round_index = env_id, match_id, round_index
        self.phase = int(gd.Phase.Play)


def test_behaviour_histogram_counts_leads_bombs_and_types():
    events = [FakeEvent(0, "Single", key=3, step=0), FakeEvent(1, "Pass", step=1),
              FakeEvent(2, "Pass", step=2), FakeEvent(3, "Pass", step=3),
              FakeEvent(0, "Bomb", key=(4, 14), step=4), FakeEvent(1, "Pass", step=5),
              FakeEvent(2, "Pass", step=6), FakeEvent(3, "Pass", step=7),
              FakeEvent(0, "Pair", key=9, step=8)]
    histogram = style_lib.behaviour_histogram(events)
    seat0 = histogram["0"]
    assert seat0["plays"] == 3 and seat0["leads"] == 3 and seat0["bombs"] == 1
    assert seat0["type_frequency"] == {"Single": 1, "Bomb": 1, "Pair": 1}
    assert seat0["bomb_fraction"] == pytest.approx(1 / 3)
    assert seat0["mean_bomb_step"] == 4
    # Lead rank uses the bomb key's power component, not its size.
    assert seat0["mean_lead_rank"] == pytest.approx((3 + 14 + 9) / 3)
    assert histogram["1"]["passes"] == 2 and histogram["1"]["plays"] == 0
    merged = style_lib.merge_behaviour([histogram])
    assert merged["plays"] == 3 and merged["leads"] == 3 and merged["passes"] == 6


def test_behaviour_histogram_ignores_non_play_phases():
    event = FakeEvent(0, "Single")
    event.phase = int(gd.Phase.Play) + 1
    assert style_lib.behaviour_histogram([event]) == {}


# --- schema 2 round logs --------------------------------------------------

def make_round(seats=(0, 1, 2, 3)):
    tokens = np.zeros((len(seats), TOKEN_DIM), dtype=np.uint8)
    for index, seat in enumerate(seats):
        tokens[index, seat] = 1
    decisions = [Decision(np.zeros(1849, np.uint8), np.zeros(154, np.uint8),
                          np.zeros((3, 54), np.uint8), seat=seat, phase=3, prefix=index)
                 for index, seat in enumerate(seats)]
    return decisions, list(tokens)


def test_schema_two_round_trips_every_new_field(tmp_path):
    dim = style_lib.style_dim()
    decisions, tokens = make_round()
    styles = np.arange(4 * dim, dtype=np.float32).reshape(4, dim)
    meta = {"match_id": 12, "round_index": 3, "env_id": 2, "styles": styles,
            "seat_driver": [DRIVER_BOT, DRIVER_POLICY, DRIVER_BOT, DRIVER_POLICY],
            "style_region": "heldout", "styled": True}
    path = tmp_path / "round.npz"
    save_round(path, decisions, tokens, group="g", meta=meta)
    with np.load(path, allow_pickle=False) as data:
        assert int(data["schema_version"]) == 2
        assert set(data.files) == {"schema_version", "group", "obs", "hidden", "seat",
                                   "prefix", "tokens", "driver", "styles", "seat_driver",
                                   "match_id", "round_index", "env_id", "style_region", "styled"}
        assert int(data["match_id"]) == 12 and int(data["round_index"]) == 3
        assert int(data["env_id"]) == 2 and str(data["style_region"]) == "heldout"
        assert bool(data["styled"]) is True
        np.testing.assert_array_equal(data["styles"], styles)
        np.testing.assert_array_equal(data["seat_driver"], meta["seat_driver"])
        # The per-decision driver is exactly the seat's driver: a policy row can
        # never be bot-driven and a bot row can never be policy-driven.
        np.testing.assert_array_equal(data["driver"], [DRIVER_BOT, DRIVER_POLICY,
                                                       DRIVER_BOT, DRIVER_POLICY])
        for seat, driver in zip(data["seat"], data["driver"]):
            assert driver == meta["seat_driver"][seat]


def test_schema_one_is_still_written_without_meta(tmp_path):
    decisions, tokens = make_round()
    path = tmp_path / "round.npz"
    save_round(path, decisions, tokens, group="g")
    with np.load(path, allow_pickle=False) as data:
        assert int(data["schema_version"]) == 1
        assert set(data.files) == {"schema_version", "group", "obs", "hidden",
                                   "seat", "prefix", "tokens"}


def test_schema_two_rejects_malformed_style_metadata(tmp_path):
    decisions, tokens = make_round()
    meta = {"match_id": 0, "round_index": 0, "env_id": 0,
            "styles": np.zeros((3, style_lib.style_dim()), np.float32),
            "seat_driver": [0, 0, 0, 0], "style_region": "train", "styled": True}
    with pytest.raises(ValueError, match="one style vector and driver per seat"):
        save_round(tmp_path / "bad.npz", decisions, tokens, group="g", meta=meta)


# --- collection driver ----------------------------------------------------

def test_style_assignment_splits_seats_by_team_and_is_deterministic():
    space = style_lib.StyleSpace.default()
    first = collection.StyleAssignment(space, 1, 0, 77, "train", 0.5, "random", True, True)
    again = collection.StyleAssignment(space, 1, 0, 77, "train", 0.5, "random", True, True)
    np.testing.assert_array_equal(first.styles, again.styles)
    assert first.seat_driver == again.seat_driver
    policy = [s for s in range(4) if first.seat_driver[s] == DRIVER_POLICY]
    bots = [s for s in range(4) if first.seat_driver[s] == DRIVER_BOT]
    assert sorted(policy) == sorted([first.policy_team, first.policy_team + 2])
    assert len(policy) == len(bots) == 2 and not set(policy) & set(bots)
    for seat in policy:
        np.testing.assert_array_equal(first.styles[seat], style_lib.neutral(space.dim))
    later = collection.StyleAssignment(space, 1, 1, 77, "train", 0.5, "random", True, True)
    assert not np.array_equal(first.styles[bots[0]], later.styles[bots[0]])


def test_fixed_policy_team_is_honoured():
    space = style_lib.StyleSpace.default()
    for team in ("0", "1"):
        entry = collection.StyleAssignment(space, 0, 0, 1, "train", 0.5, team, True, True)
        assert entry.policy_team == int(team)
        assert [s for s in range(4) if entry.seat_driver[s] == DRIVER_POLICY] == \
            sorted([int(team), int(team) + 2])


def test_heldout_assignment_is_labelled_and_inside_the_region():
    space = style_lib.StyleSpace.default()
    entry = collection.StyleAssignment(space, 0, 0, 2, "heldout", 0.5, "random", True, True)
    assert entry.region == "heldout"
    for seat in range(4):
        if entry.seat_driver[seat] == DRIVER_BOT:
            assert space.in_heldout(entry.styles[seat])


def test_unavailable_styled_bot_records_unknown_styles():
    space = style_lib.StyleSpace.default()
    entry = collection.StyleAssignment(space, 0, 0, 2, "train", 0.5, "random", True, False)
    assert entry.region == "unknown"
    for seat in range(4):
        if entry.seat_driver[seat] == DRIVER_BOT:
            assert np.isnan(entry.styles[seat]).all()


def test_styled_collection_records_styles_drivers_and_indices(checkpoint, tmp_path):
    output = tmp_path / "styled"
    with pytest.warns(RuntimeWarning) if not style_lib.styled_bot_available() \
            else _no_warning():
        report = collection.collect_belief(
            checkpoint, output, rounds=6, num_envs=2, seed=1103, max_seconds=90,
            purpose="architecture_probe", styled=True, style_region="mixed", style_seed=41)
    assert report["status"] == "complete"
    assert report["styled"] is True
    assert report["schema_version"] == 2
    assert report["style_region"] == "mixed" and report["style_seed"] == 41
    assert report["styled_bot_unavailable"] is not style_lib.styled_bot_available()
    assert report["style_space"]["dim"] == style_lib.style_dim()
    assert report["style_space"]["heldout_rule"]
    assert report["behaviour_total"]["plays"] > 0
    matches = json.loads((output / "matches.json").read_text())
    assert matches and len(matches) == report["matches_sampled"]
    seen = {(m["env_id"], m["match_id"]) for m in matches}
    assert len(seen) == len(matches)  # styles sampled exactly once per match
    dim = style_lib.style_dim()
    for match in matches:
        assert len(match["styles"]) == 4 and all(len(v) == dim for v in match["styles"])
        assert sorted(match["seat_driver"]) == [DRIVER_POLICY, DRIVER_POLICY,
                                                DRIVER_BOT, DRIVER_BOT]
    rounds = sorted(output.glob("round-*.npz"))
    assert len(rounds) == 6
    for path in rounds:
        with np.load(path, allow_pickle=False) as data:
            assert int(data["schema_version"]) == 2
            assert 0 <= int(data["env_id"]) < 2 and int(data["round_index"]) >= 0
            match = next(m for m in matches if m["env_id"] == int(data["env_id"])
                         and m["match_id"] == int(data["match_id"]))
            np.testing.assert_array_equal(data["seat_driver"], match["seat_driver"])
            drivers = data["driver"]
            seats = data["seat"]
            assert set(np.unique(drivers)) <= {DRIVER_POLICY, DRIVER_BOT}
            for seat, driver in zip(seats, drivers):
                assert driver == match["seat_driver"][seat]
            policy_rows = drivers == DRIVER_POLICY
            bot_rows = drivers == DRIVER_BOT
            assert not (policy_rows & bot_rows).any()
            assert policy_rows.any() and bot_rows.any()


class _no_warning:
    """Context manager asserting the fallback warning is not raised."""

    def __enter__(self):
        import warnings
        self._ctx = warnings.catch_warnings(record=True)
        self._log = self._ctx.__enter__()
        import warnings as w
        w.simplefilter("always")
        return self

    def __exit__(self, *exc):
        messages = [str(r.message) for r in self._log
                    if issubclass(r.category, RuntimeWarning) and "STYLE_DIM" in str(r.message)]
        self._ctx.__exit__(*exc)
        assert not messages, messages
        return False


def test_styles_resample_exactly_once_per_match_per_seat(checkpoint, tmp_path, monkeypatch):
    made: list[tuple[int, int]] = []
    original = collection.StyleAssignment

    class Counting(original):
        def __init__(self, space, env_id, match_id, *args, **kwargs):
            made.append((env_id, match_id))
            super().__init__(space, env_id, match_id, *args, **kwargs)

    monkeypatch.setattr(collection, "StyleAssignment", Counting)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        collection.collect_belief(checkpoint, tmp_path / "once", rounds=8, num_envs=2,
                                  seed=1104, max_seconds=90, purpose="architecture_probe",
                                  styled=True, style_seed=9)
    assert made and len(made) == len(set(made))


def test_unstyled_collection_is_all_policy_driven(checkpoint, tmp_path):
    output = tmp_path / "plain"
    report = collection.collect_belief(checkpoint, output, rounds=4, num_envs=2, seed=1105,
                                       max_seconds=90, purpose="architecture_probe")
    assert report["styled"] is False and report["styled_bot_unavailable"] is False
    files = list(output.glob("round-*.npz"))
    assert files
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            # Without --styled the checkpoint drives all four seats, exactly as
            # in the published self-play belief collections.
            np.testing.assert_array_equal(data["seat_driver"], [DRIVER_POLICY] * 4)
            assert set(np.unique(data["driver"])) == {DRIVER_POLICY}
            assert str(data["style_region"]) == "none"
            assert bool(data["styled"]) is False


def test_collection_rejects_bad_style_options(checkpoint, tmp_path):
    with pytest.raises(ValueError, match="style region"):
        collection.collect_belief(checkpoint, tmp_path / "a", rounds=2, style_region="nope")
    with pytest.raises(ValueError, match="policy team"):
        collection.collect_belief(checkpoint, tmp_path / "b", rounds=2, policy_team="both")


@needs_styled_bot
def test_bot_rows_take_the_styled_choice(checkpoint, tmp_path, monkeypatch):
    """With the binding present, bot seats must follow styled_choice."""
    seen = {"bot_rows": 0, "matched": 0}
    vec_env = collection.gd.VecEnv

    class Recording:
        def __init__(self, *args, **kwargs):
            self.env = vec_env(*args, **kwargs)
            self.last = None

        def pending(self):
            self.last = self.env.pending()
            return self.last

        def step(self, choices):
            batch = self.last
            styled = np.asarray(batch.styled_choice)
            for row in range(len(styled)):
                if int(batch.phase[row]) == int(gd.Phase.Play):
                    seen["bot_rows"] += 1
                    seen["matched"] += int(choices[row] == styled[row])
            return self.env.step(choices)

        def __getattr__(self, name):
            return getattr(self.env, name)

    monkeypatch.setattr(collection.gd, "VecEnv", Recording)
    collection.collect_belief(checkpoint, tmp_path / "styled", rounds=4, num_envs=2,
                              seed=1106, max_seconds=90, purpose="architecture_probe",
                              styled=True, style_seed=13)
    assert seen["bot_rows"] > 0 and seen["matched"] > 0


# --- coverage report ------------------------------------------------------

def synthetic_collection(path, regions=("train", "heldout")):
    """A tiny collection directory with real, finite style vectors."""
    space = style_lib.StyleSpace.default()
    rng = np.random.default_rng(4)
    path.mkdir(parents=True)
    matches = []
    for index, region in enumerate(regions * 4):
        styles = np.stack([style_lib.neutral(space.dim) if seat % 2 else
                           space.sample(rng, region) for seat in range(4)])
        behaviour = {str(seat): {"plays": 10 + index, "passes": 5, "bombs": 1 + index % 3,
                                 "leads": 4, "type_frequency": {"Single": 8, "Bomb": 2},
                                 "lead_type_frequency": {"Single": 4},
                                 "bomb_timing": {"<4": 1}, "lead_rank": {"4-8": 4},
                                 "bomb_fraction": 0.1, "mean_bomb_step": 6.0,
                                 "mean_lead_rank": 7.0, "pass_fraction": 0.3}
                     for seat in (0, 2)}
        matches.append({"env_id": index % 2, "match_id": index // 2,
                        "policy_team": 1, "seat_driver": [DRIVER_BOT, DRIVER_POLICY,
                                                          DRIVER_BOT, DRIVER_POLICY],
                        "region": region, "styles": [[float(x) for x in row] for row in styles],
                        "behaviour": behaviour, "rounds": 2})
    (path / "matches.json").write_text(json.dumps(matches))
    (path / "provenance.json").write_text(json.dumps({
        "status": "complete", "styled": True, "styled_bot_unavailable": False,
        "style_region": "mixed", "style_seed": 4, "collected_rounds": 16,
        "collected_decisions": 160, "style_space": space.describe(),
        "behaviour_total": style_lib.merge_behaviour(
            [m["behaviour"] for m in matches])}))
    return space, matches


def test_coverage_report_summarises_slots_behaviour_and_disjointness(tmp_path):
    space, matches = synthetic_collection(tmp_path / "data")
    report = coverage.write_report(tmp_path / "data")
    assert report["matches_sampled"] == len(matches)
    assert report["bot_seat_styles"] == 2 * len(matches)
    assert set(report["slot_histograms"]) == set(space.names)
    for entry in report["slot_histograms"].values():
        assert entry["samples"] == 2 * len(matches)
        assert sum(entry["counts"]) == entry["samples"]
        assert entry["min"] >= min(space.lower) and entry["max"] <= max(space.upper)
    assert report["region_check"]["disjoint"] is True
    assert report["region_check"]["seat_styles_by_region"] == {"train": 8, "heldout": 8}
    assert report["region_check"]["unknown_styles"] == 0
    bins = report["behaviour_by_slot_bin"]["bomb_threshold"]
    assert bins and all(entry["plays"] > 0 for entry in bins.values())
    text = (tmp_path / "data" / "coverage.md").read_text()
    assert "# Style coverage report" in text and "Disjoint: **True**" in text
    assert "bomb_threshold" in text
    json.loads((tmp_path / "data" / "coverage.json").read_text())


def test_coverage_report_flags_a_style_outside_its_declared_region(tmp_path):
    space, matches = synthetic_collection(tmp_path / "data", regions=("train",))
    matches[0]["styles"][0] = [float(x) for x in space.sample(np.random.default_rng(1), "heldout")]
    (tmp_path / "data" / "matches.json").write_text(json.dumps(matches))
    report = coverage.build_report(tmp_path / "data")
    assert report["region_check"]["disjoint"] is False
    assert report["region_check"]["violations"][0]["actual"] == "heldout"


def test_coverage_report_runs_on_a_tiny_real_collection(checkpoint, tmp_path):
    output = tmp_path / "tiny"
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        collection.collect_belief(checkpoint, output, rounds=4, num_envs=2, seed=1107,
                                  max_seconds=90, purpose="architecture_probe",
                                  styled=True, style_seed=17)
    report = coverage.write_report(output)
    assert report["status"] == "complete"
    assert report["behaviour_total"]["plays"] > 0
    assert (output / "coverage.md").exists() and (output / "coverage.json").exists()
    if not style_lib.styled_bot_available():
        assert report["styled_bot_unavailable"] is True
        assert report["region_check"]["unknown_styles"] == report["bot_seat_styles"] * 0 + \
            report["region_check"]["seat_styles_by_region"].get("unknown", 0)


def test_coverage_cli_prints_a_summary(tmp_path, capsys):
    synthetic_collection(tmp_path / "data")
    coverage.main(["--collection", str(tmp_path / "data"),
                   "--output", str(tmp_path / "report")])
    printed = json.loads(capsys.readouterr().out)
    assert printed["disjoint"] is True and printed["matches_sampled"] == 8
    assert (tmp_path / "report" / "coverage.md").exists()


# --- splits ---------------------------------------------------------------

def styled_dataset(path, heldout_groups=4):
    """A loadable dataset whose matches carry style regions."""
    path.mkdir()
    space = style_lib.StyleSpace.default()
    rng = np.random.default_rng(2)
    total = 20
    for index in range(total):
        hidden = np.zeros((3, 54), dtype=np.uint8)
        hidden[index % 3, index % 54] = 1
        obs = np.zeros(1849, dtype=np.uint8)
        obs[108 + index % 54] = 1
        for rel in range(3):
            obs[648 + 28 * rel + int(rel == index % 3)] = 1
        tokens = np.zeros((3, TOKEN_DIM), dtype=np.uint8)
        tokens[np.arange(3), np.arange(3)] = 1
        decisions = [Decision(obs, np.zeros(154), hidden, seat=i, phase=3, prefix=i)
                     for i in range(3)]
        region = "heldout" if index < heldout_groups else "train"
        styles = np.stack([space.sample(rng, region) for _ in range(4)])
        save_round(path / f"round-{index:08d}.npz", decisions, list(tokens),
                   f"match-{index}",
                   meta={"match_id": index, "round_index": 0, "env_id": index % 2,
                         "styles": styles, "seat_driver": [1, 0, 1, 0],
                         "style_region": region, "styled": True})
    (path / "provenance.json").write_text(json.dumps({
        "status": "complete", "purpose": "architecture_probe", "learner_updates": 0,
        "engine_source_sha256": engine_source_digest(), "action_mode": "canonical",
        "tribute_policy": "heuristic", "sampling_margin": 0, "collected_rounds": total,
        "seed": 901, "training_seed": 7, "stage": "dmc", "play_mode": "fp32_argmax",
        "checkpoint_id": "a" * 64, "requested_rounds": total,
        "collected_decisions": 3 * total, "match_groups": total}))


def test_schema_two_dataset_loads_and_keeps_its_style_region(tmp_path):
    styled_dataset(tmp_path / "data")
    rounds, provenance, digest = load_dataset(tmp_path / "data")
    assert len(rounds) == 20 and len(digest) == 64
    assert {r["schema_version"] for r in rounds} == {2}
    assert {r["style_region"] for r in rounds} == {"train", "heldout"}
    assert all(r["driver"].shape == r["seat"].shape for r in rounds)


def test_style_region_split_restricts_test_matches_to_heldout(tmp_path):
    styled_dataset(tmp_path / "data", heldout_groups=6)
    rounds, _, _ = load_dataset(tmp_path / "data")
    split = split_rounds(rounds, 123, heldout_styles=True)
    assert {r["style_region"] for r in split["test"]} == {"heldout"}
    assert "heldout" not in {r["style_region"] for r in split["train"]}
    sets = [{r["group"] for r in split[k]} for k in ("train", "validation", "test")]
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i + 1:])
    assert split["test"] and split["train"] and split["validation"]
    # The default whole-match split is unchanged.
    plain = split_rounds(rounds, 123)
    assert [len(plain[k]) for k in ("train", "validation", "test")] == [14, 3, 3]


def test_style_region_split_needs_enough_heldout_matches(tmp_path):
    styled_dataset(tmp_path / "data", heldout_groups=1)
    rounds, _, _ = load_dataset(tmp_path / "data")
    with pytest.raises(ValueError, match="held-out-style matches"):
        split_rounds(rounds, 123, heldout_styles=True)
