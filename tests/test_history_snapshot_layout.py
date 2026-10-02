"""Exact host layout checks for merged frozen-policy rows and sampler draws."""
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from train.history_snapshot_batch import LAYOUT_FIELDS, merged_layout


def fixture(seed, unique, input_dtype):
    rng = np.random.default_rng(seed)
    identities = rng.integers(0, 8, size=41)
    counts = rng.integers(1, 12, size=len(identities), dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    env_id = np.arange(len(identities), dtype=np.int64)
    if not unique:
        env_id //= 4
    match_id = env_id % 3
    prefix = rng.integers(0, 1500, size=len(identities), dtype=np.int64)
    obs = rng.integers(0, 2, size=(len(identities), 17)).astype(input_dtype)
    seat = rng.integers(0, 4, size=len(identities), dtype=np.int64)
    cand = rng.integers(0, 2, size=(offsets[-1], 9)).astype(input_dtype)
    groups = []
    for identity in np.unique(identities)[1:]:
        rows = np.flatnonzero(identities == identity)
        keys = list(zip(env_id[rows].tolist(), match_id[rows].tolist()))
        ordered = list(dict.fromkeys(keys))
        groups.append(SimpleNamespace(
            rows=rows, keys=ordered,
            streams=[SimpleNamespace(prefix=int(rng.integers(1, 1600))) for _ in ordered],
            max_candidates=int(counts[rows].max()),
            one_decision_per_stream=len(ordered) == len(keys)))
    # Exercise slot gaps and non-monotonic slot assignment after population
    # pruning, along with skipped learner rows in the original pending batch.
    slots = rng.choice(len(groups) + 5, len(groups), replace=False).tolist()
    args = (groups, slots, prefix, obs, seat, cand, offsets, counts, env_id, match_id)
    return args


def scalar_reference(groups, slots, prefix, obs, seat, cand, offsets, counts,
                     env_id, match_id):
    """Independent placement from each row/candidate's identity and local index."""
    S, M = max(slots) + 1, max(len(g.rows) for g in groups)
    Cm = max(sum(int(counts[r]) for r in g.rows) for g in groups)
    K = max(g.max_candidates for g in groups)
    out = {name: [] for name in LAYOUT_FIELDS}
    out["seat_pad"] = np.zeros(S * M, np.int64)
    out["cand_state_pad"] = np.zeros(S * Cm, np.int64)
    rows, group_rows, group_streams, draws = [], [], [], []
    stream_base = draw_base = 0
    for group, slot in zip(groups, slots):
        begin = len(rows)
        candidate = 0
        for local_row, source_row in enumerate(group.rows):
            row = len(rows)
            rows.append(source_row)
            padded = slot * M + local_row
            key = (int(env_id[source_row]), int(match_id[source_row]))
            out["obs"].append(obs[source_row])
            out["seat"].append(seat[source_row])
            out["prefix"].append(prefix[source_row])
            out["row_pad"].append(padded)
            out["seat_pad"][padded] = slot * 4 + seat[source_row]
            out["stream"].append(stream_base + group.keys.index(key))
            out["uniform_base"].append(draw_base + local_row * group.max_candidates)
            out["uniform_last"].append(group.max_candidates - 1)
            for local_cand in range(counts[source_row]):
                padded_cand = slot * Cm + candidate
                out["cand"].append(cand[offsets[source_row] + local_cand])
                out["cand_pad"].append(padded_cand)
                out["cand_state_pad"][padded_cand] = padded
                out["cand_row"].append(row)
                out["cand_table"].append(row * K + local_cand)
                candidate += 1
        group_rows.append((begin, len(rows)))
        group_streams.append((stream_base, stream_base + len(group.keys)))
        draws.append((draw_base, draw_base + len(group.rows) * group.max_candidates))
        stream_base = group_streams[-1][1]
        draw_base = draws[-1][1]
    return dict(
        rows=np.asarray(rows, np.int64), group_rows=group_rows,
        group_streams=group_streams,
        arrays=tuple(np.asarray(out[name], np.uint8 if name in {"obs", "cand"} else np.int64)
                     for name in LAYOUT_FIELDS),
        draws=draws, slots=S, rows_per_slot=M, cands_per_slot=Cm, table_width=K,
        length=max(s.prefix for g in groups for s in g.streams) + 1,
        streams=stream_base,
        identity_streams=np.array_equal(out["stream"], np.arange(len(rows))))


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("unique", [True, False])
@pytest.mark.parametrize("input_dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("single_group", [True, False])
def test_merged_layout_matches_scalar_placement(seed, unique, input_dtype, single_group):
    args = fixture(seed, unique, input_dtype)
    if single_group:
        args = (args[0][:1], args[1][:1], *args[2:])
    layout = merged_layout(*args)
    expected = scalar_reference(*args)
    for field in fields(layout):
        actual, reference = getattr(layout, field.name), expected[field.name]
        if field.name == "arrays":
            for name, array, wanted in zip(LAYOUT_FIELDS, actual, reference):
                np.testing.assert_array_equal(array, wanted, err_msg=name)
                assert array.dtype == wanted.dtype
                assert array.flags.c_contiguous
        elif field.name == "rows":
            np.testing.assert_array_equal(actual, reference)
        else:
            assert actual == reference, field.name
