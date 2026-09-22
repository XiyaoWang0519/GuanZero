"""Continuous opponent styles: slot layout, sampling regions and behaviour histograms.

Styles are continuous parameters sampled once per match per seat, never a
fixed list of bots (M2_TODO "Revised next steps"). The slot layout is owned by
the C++ styled bot and mirrored here through :func:`slot_map`, which is the one
place to adjust if `gd.StyleParams` changes:

    slot 0                  bomb_threshold, in [0, 1]
    slots 1 .. T            per-play-type preference log-weights, neutral 0
    slot T+1                follow_aggression, in [0, 1]
    slot T+2                lead_high_bias, in [-1, 1]
    slot T+3                partner_weight, in [0, 1]
    slot T+4                temperature, >= 0

When the styled-bot binding is absent this module still defines the layout and
samples vectors, so that collections degrade to the greedy bot with recorded
NaN styles instead of failing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import gd
import numpy as np

# Non-pass play types, in the order used for the per-type preference weights.
PLAY_TYPE_NAMES: tuple[str, ...] = (
    "Single", "Pair", "Triple", "FullHouse", "Straight", "Tube", "Plate",
    "Bomb", "StraightFlush", "JokerBomb",
)
TAIL_SLOT_NAMES: tuple[str, ...] = (
    "follow_aggression", "lead_high_bias", "partner_weight", "temperature",
)
HEAD_SLOT_NAME = "bomb_threshold"
# Used only while gd.STYLE_DIM is unavailable; the real value wins when present.
FALLBACK_STYLE_DIM = 1 + len(PLAY_TYPE_NAMES) + len(TAIL_SLOT_NAMES)

DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    HEAD_SLOT_NAME: (0.0, 1.0),
    "type_weight": (-1.5, 1.5),
    "follow_aggression": (0.0, 1.0),
    "lead_high_bias": (-1.0, 1.0),
    "partner_weight": (0.0, 1.0),
    "temperature": (0.0, 1.5),
}
# A hypercube corner of the style box, reserved for held-out test matches.
DEFAULT_HELDOUT: dict[str, tuple[float, float]] = {
    HEAD_SLOT_NAME: (0.75, 1.0),
    "follow_aggression": (0.75, 1.0),
}
REGIONS: tuple[str, ...] = ("train", "heldout", "mixed")


def styled_bot_available() -> bool:
    """True when the task-1 styled-bot binding is compiled into `gd`."""
    return hasattr(gd, "STYLE_DIM")


def style_dim() -> int:
    """Style-vector length, from the binding when it exists."""
    dim = int(getattr(gd, "STYLE_DIM", FALLBACK_STYLE_DIM))
    if dim < 1 + 1 + len(TAIL_SLOT_NAMES):
        raise ValueError(f"implausible style dimension {dim}")
    return dim


def slot_map(dim: int) -> dict[str, int]:
    """Slot name to index. The only place the named layout is hardcoded."""
    types = dim - 1 - len(TAIL_SLOT_NAMES)
    if types < 1:
        raise ValueError(f"style dimension {dim} leaves no play-type slots")
    names = PLAY_TYPE_NAMES if types == len(PLAY_TYPE_NAMES) else \
        tuple(f"type{i}" for i in range(types))
    mapping = {HEAD_SLOT_NAME: 0}
    for i, name in enumerate(names):
        mapping[f"prefer_{name}"] = 1 + i
    for i, name in enumerate(TAIL_SLOT_NAMES):
        mapping[name] = 1 + types + i
    return mapping


def slot_names(dim: int) -> tuple[str, ...]:
    """Slot names in index order."""
    mapping = slot_map(dim)
    return tuple(sorted(mapping, key=mapping.__getitem__))


def _range_for(name: str) -> tuple[float, float]:
    return DEFAULT_RANGES.get(name, DEFAULT_RANGES["type_weight"])


@dataclass(frozen=True)
class StyleSpace:
    """A style box plus the held-out sub-box reserved for test matches."""

    dim: int
    names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    heldout: tuple[tuple[str, float, float], ...]

    @classmethod
    def default(cls, dim: int | None = None,
                heldout: dict[str, tuple[float, float]] | None = None) -> "StyleSpace":
        dim = style_dim() if dim is None else int(dim)
        names = slot_names(dim)
        lower, upper = zip(*(_range_for(n) for n in names))
        rule = DEFAULT_HELDOUT if heldout is None else heldout
        box = []
        for name, (low, high) in sorted(rule.items()):
            if name not in names:
                raise ValueError(f"held-out slot {name!r} is not part of the style space")
            index = names.index(name)
            if not lower[index] <= low < high <= upper[index]:
                raise ValueError(f"held-out range for {name!r} is not inside the style box")
            if low <= lower[index] and high >= upper[index]:
                raise ValueError(f"held-out range for {name!r} covers the whole slot")
            box.append((name, float(low), float(high)))
        if not box:
            raise ValueError("the held-out region must constrain at least one slot")
        return cls(dim=dim, names=names, lower=tuple(map(float, lower)),
                   upper=tuple(map(float, upper)), heldout=tuple(box))

    def index(self, name: str) -> int:
        return self.names.index(name)

    def in_bounds(self, vector: Sequence[float]) -> bool:
        v = np.asarray(vector, dtype=np.float64)
        if v.shape != (self.dim,) or not np.isfinite(v).all():
            return False
        return bool((v >= np.asarray(self.lower)).all() and (v <= np.asarray(self.upper)).all())

    def in_heldout(self, vector: Sequence[float]) -> bool:
        """True only inside the full held-out sub-box, so regions are disjoint."""
        v = np.asarray(vector, dtype=np.float64)
        if v.shape != (self.dim,) or not np.isfinite(v).all():
            return False
        return all(low <= v[self.index(name)] <= high for name, low, high in self.heldout)

    def sample(self, rng: np.random.Generator, region: str = "train",
               heldout_fraction: float = 0.5) -> np.ndarray:
        return sample_style(rng, self, region, heldout_fraction)

    def describe(self) -> dict:
        """JSON-safe description recorded in collection provenance."""
        return {
            "dim": self.dim, "slots": list(self.names),
            "bounds": {n: [lo, hi] for n, lo, hi in zip(self.names, self.lower, self.upper)},
            "heldout_rule": {n: [lo, hi] for n, lo, hi in self.heldout},
            "split_rule": ("a style is held out when every listed slot lies inside its "
                           "listed range; training styles are rejection-sampled outside "
                           "that box, so the two regions are disjoint by construction"),
            "styled_bot_available": styled_bot_available(),
        }


def neutral(dim: int | None = None) -> np.ndarray:
    """The style that reproduces the plain greedy bot."""
    dim = style_dim() if dim is None else int(dim)
    params = getattr(gd, "StyleParams", None)
    if params is not None and hasattr(params, "neutral"):
        value = params.neutral()
        array = getattr(value, "v", None)
        if array is None:
            array = value.to_array()
        vector = np.asarray(array, dtype=np.float32).reshape(-1)
        if vector.shape != (dim,):
            raise ValueError("gd.StyleParams.neutral() disagrees with gd.STYLE_DIM")
        return vector
    return np.zeros(dim, dtype=np.float32)


def unknown_style(dim: int | None = None) -> np.ndarray:
    """All-NaN placeholder recorded when the styled bot is unavailable."""
    dim = style_dim() if dim is None else int(dim)
    return np.full(dim, np.nan, dtype=np.float32)


def sample_style(rng: np.random.Generator, space: StyleSpace, region: str = "train",
                 heldout_fraction: float = 0.5) -> np.ndarray:
    """Sample one style vector from `region` of `space`."""
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}")
    if not 0.0 <= heldout_fraction <= 1.0:
        raise ValueError("heldout_fraction must lie in [0, 1]")
    if region == "mixed":
        region = "heldout" if rng.random() < heldout_fraction else "train"
    low = np.asarray(space.lower, dtype=np.float64)
    high = np.asarray(space.upper, dtype=np.float64)
    if region == "heldout":
        vector = rng.uniform(low, high)
        for name, lo, hi in space.heldout:
            vector[space.index(name)] = rng.uniform(lo, hi)
        return vector.astype(np.float32)
    for _ in range(1000):
        vector = rng.uniform(low, high)
        if not space.in_heldout(vector):
            return vector.astype(np.float32)
    raise RuntimeError("the held-out region leaves no room for training styles")


def region_of(space: StyleSpace, vector: Sequence[float]) -> str:
    """Label a concrete style vector, or 'unknown' for a NaN placeholder."""
    v = np.asarray(vector, dtype=np.float64)
    if v.shape != (space.dim,) or not np.isfinite(v).all():
        return "unknown"
    return "heldout" if space.in_heldout(v) else "train"


# --- behaviour histograms -------------------------------------------------

BOMB_STEP_EDGES: tuple[int, ...] = (4, 10, 20, 40)
LEAD_KEY_EDGES: tuple[int, ...] = (4, 8, 12, 15)


def _bucket(value: float, edges: Sequence[float]) -> str:
    index = int(np.digitize([value], edges)[0])
    if index == 0:
        return f"<{edges[0]}"
    if index == len(edges):
        return f">={edges[-1]}"
    return f"{edges[index - 1]}-{edges[index]}"


def _action_key(action: object) -> int:
    key = getattr(action, "key", 0)
    # A Bomb key is the tuple (size, power); rank it by its power component.
    return int(key[1]) if isinstance(key, tuple) else int(key)


class BehaviourAccumulator:
    """Per-seat behaviour counts built from the public action-event stream.

    A play is a lead when it opens the round or follows three consecutive
    passes; the engine logs automatically skipped passes, so control always
    returns after exactly three of them.
    """

    def __init__(self) -> None:
        self._seats: dict[int, dict] = {}
        self._round: tuple | None = None
        self._passes = 0
        self._played = False

    def _seat(self, seat: int) -> dict:
        return self._seats.setdefault(seat, {
            "plays": 0, "passes": 0, "bombs": 0, "leads": 0,
            "type_frequency": {}, "lead_type_frequency": {},
            "bomb_timing": {}, "lead_rank": {},
            "bomb_step_total": 0, "lead_rank_total": 0,
        })

    def add_event(self, event: object) -> None:
        """Feed one `gd.PublicActionEvent`; play-phase events only."""
        if int(getattr(event, "phase", int(gd.Phase.Play))) != int(gd.Phase.Play):
            return
        key = (int(event.env_id), int(event.match_id), int(event.round_index))
        if key != self._round:
            self._round, self._passes, self._played = key, 0, False
        seat = self._seat(int(event.seat))
        action = event.action
        kind = str(action.type)
        if kind == "Pass":
            seat["passes"] += 1
            self._passes += 1
            return
        lead = not self._played or self._passes >= 3
        self._passes, self._played = 0, True
        seat["plays"] += 1
        seat["type_frequency"][kind] = seat["type_frequency"].get(kind, 0) + 1
        if kind in ("Bomb", "StraightFlush", "JokerBomb"):
            step = int(getattr(event, "step", 0))
            seat["bombs"] += 1
            seat["bomb_step_total"] += step
            bucket = _bucket(step, BOMB_STEP_EDGES)
            seat["bomb_timing"][bucket] = seat["bomb_timing"].get(bucket, 0) + 1
        if lead:
            rank = _action_key(action)
            seat["leads"] += 1
            seat["lead_rank_total"] += rank
            seat["lead_type_frequency"][kind] = seat["lead_type_frequency"].get(kind, 0) + 1
            bucket = _bucket(rank, LEAD_KEY_EDGES)
            seat["lead_rank"][bucket] = seat["lead_rank"].get(bucket, 0) + 1

    def as_dict(self) -> dict[str, dict]:
        """JSON-safe per-seat summary with the derived rates."""
        out: dict[str, dict] = {}
        for seat in sorted(self._seats):
            counts = dict(self._seats[seat])
            plays, bombs, leads = counts["plays"], counts["bombs"], counts["leads"]
            counts["bomb_fraction"] = bombs / plays if plays else None
            counts["mean_bomb_step"] = counts.pop("bomb_step_total") / bombs if bombs else None
            counts["mean_lead_rank"] = counts.pop("lead_rank_total") / leads if leads else None
            counts["pass_fraction"] = (counts["passes"] / (plays + counts["passes"])
                                       if plays + counts["passes"] else None)
            out[str(seat)] = counts
        return out


def behaviour_histogram(events: Iterable[object]) -> dict[str, dict]:
    """Per-seat behaviour histogram for any sequence of public action events."""
    accumulator = BehaviourAccumulator()
    for event in events:
        accumulator.add_event(event)
    return accumulator.as_dict()


def merge_behaviour(parts: Iterable[dict[str, dict]]) -> dict:
    """Sum behaviour counts over seats and matches into one histogram."""
    total: dict[str, float] = {"plays": 0, "passes": 0, "bombs": 0, "leads": 0,
                              "bomb_step_total": 0.0, "lead_rank_total": 0.0}
    histograms = {"type_frequency": {}, "lead_type_frequency": {},
                  "bomb_timing": {}, "lead_rank": {}}
    for part in parts:
        for counts in part.values():
            for name in ("plays", "passes", "bombs", "leads"):
                total[name] += int(counts.get(name, 0))
            bombs, leads = int(counts.get("bombs", 0)), int(counts.get("leads", 0))
            if counts.get("mean_bomb_step") is not None:
                total["bomb_step_total"] += float(counts["mean_bomb_step"]) * bombs
            if counts.get("mean_lead_rank") is not None:
                total["lead_rank_total"] += float(counts["mean_lead_rank"]) * leads
            for name, target in histograms.items():
                for bucket, value in (counts.get(name) or {}).items():
                    target[bucket] = target.get(bucket, 0) + int(value)
    plays, bombs, leads = total["plays"], total["bombs"], total["leads"]
    return {
        "plays": int(plays), "passes": int(total["passes"]),
        "bombs": int(bombs), "leads": int(leads),
        "bomb_fraction": bombs / plays if plays else None,
        "mean_bomb_step": total["bomb_step_total"] / bombs if bombs else None,
        "mean_lead_rank": total["lead_rank_total"] / leads if leads else None,
        **{name: dict(sorted(values.items())) for name, values in histograms.items()},
    }


def match_style_rng(style_seed: int, env_id: int, match_id: int) -> np.random.Generator:
    """Deterministic per-match generator, so styles resample exactly once."""
    return np.random.default_rng([int(style_seed), int(env_id), int(match_id)])


def iter_style_slots(space: StyleSpace, vectors: Sequence[Sequence[float]]) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (slot name, column) for a stack of style vectors."""
    stack = np.asarray(vectors, dtype=np.float64).reshape(-1, space.dim)
    for index, name in enumerate(space.names):
        yield name, stack[:, index]
