"""Own-lineage Transformer snapshots, sampled once per match.

Recent snapshots are eligible for new matches. With an archive
(``archive_every`` > 0), every snapshot taken at a multiple of ``archive_every``
updates is also kept; when more than ``archive_size`` are kept, every second
one is dropped (newest kept), so the archive stays spread over the whole run.
A snapshot seat draws from the archive with probability ``archive_share``.
Without an archive the sampler and its random stream are unchanged. Other
snapshots stay resident until their last pinned match ends. The checkpoint embeds all resident weights;
resume restarts environments and drops assignments, but restores the pool/RNG.

``archive_filter="strength_behavior"`` replaces the every-second thin with a
strength+behavior gate: admit by mean total-variation distance on a fixed probe
fingerprint, replace a near-duplicate only when the candidate is newer (stronger
on an ascending lineage), and when full drop the oldest member. Default
``thin`` keeps the historical thinning bit-identical.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
from typing import Any

import numpy as np
import torch

from train.history_model import DecisionInputs, HistoryActor

ARCHIVE_FILTERS = ("thin", "strength_behavior")


def snapshot_weights(model: HistoryActor) -> dict[str, torch.Tensor]:
    """A snapshot's acting weights: the state dict without auxiliary heads, which
    snapshots never use. Digests and saved snapshots are over this view, so a
    lineage keeps its snapshot identities when heads are added."""
    skip = model.auxiliary_parameter_names()
    return {k: v for k, v in model.state_dict().items() if k not in skip}


def weights_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def behavior_fingerprint(actor: HistoryActor, inputs: DecisionInputs) -> np.ndarray:
    """Concatenated candidate probabilities on a fixed probe (float32)."""
    logp = actor.candidate_log_probs(inputs)
    return logp.exp().detach().cpu().numpy().astype(np.float32, copy=False)


def mean_behavior_tv(a: np.ndarray, b: np.ndarray, decisions: int) -> float:
    """Mean total variation across the probe decisions (probs sum to 1 each)."""
    if decisions < 1:
        raise ValueError("behavior TV needs a positive decision count")
    if a.shape != b.shape:
        raise ValueError("behavior vectors must share the probe layout")
    return float(0.5 * np.abs(a.astype(np.float64) - b.astype(np.float64)).sum() / decisions)


class HistoryPopulation:
    def __init__(self, actor: HistoryActor, lineage: str, seed: int,
                 recent: int = 4, snapshot_probability: float = 0.5,
                 archive_every: int = 0, archive_size: int = 16,
                 archive_share: float = 0.5, archive_filter: str = "thin",
                 archive_min_tv: float = 0.05) -> None:
        if recent < 1 or not 0 <= snapshot_probability <= 1:
            raise ValueError("positive pool size and probability in [0, 1] required")
        if archive_every < 0 or archive_size < 1 or not 0 <= archive_share <= 1:
            raise ValueError("archive cadence >= 0, size >= 1, share in [0, 1] required")
        if archive_filter not in ARCHIVE_FILTERS:
            raise ValueError(f"archive_filter must be one of {ARCHIVE_FILTERS}")
        if not 0 <= archive_min_tv <= 1:
            raise ValueError("archive_min_tv must be in [0, 1]")
        self.actor, self.lineage = actor, lineage
        self.recent, self.probability = recent, snapshot_probability
        self.archive_every, self.archive_size = archive_every, archive_size
        self.archive_share = archive_share
        self.archive_filter = archive_filter
        self.archive_min_tv = float(archive_min_tv)
        self.archive: list[int] = []
        self.behaviors: dict[int, np.ndarray] = {}
        self.behavior_decisions: int = 0
        self.rng = np.random.default_rng(seed)
        self.models: dict[int, HistoryActor] = {}
        self.metadata: dict[int, dict[str, Any]] = {}
        self.next_id = 1
        self.seat_matches: Counter = Counter()
        self.decisions: Counter = Counter()
        self.archive_rejects: int = 0
        self.archive_replacements: int = 0

    def snapshot(self, update: int, behavior: np.ndarray | None = None,
                 behavior_decisions: int | None = None) -> int:
        identity = self.next_id
        self.next_id += 1
        model = copy.deepcopy(self.actor).requires_grad_(False).train()
        self.models[identity] = model
        self.metadata[identity] = dict(identity=identity, lineage=self.lineage,
                                       update=update, sha256=weights_digest(snapshot_weights(model)))
        if self.archive_every and update % self.archive_every == 0:
            if self.archive_filter == "thin":
                self.archive.append(identity)
                if len(self.archive) > self.archive_size:
                    self.archive = self.archive[::-1][::2][::-1]
            else:
                self._admit_filtered(identity, behavior, behavior_decisions)
        return identity

    def _admit_filtered(self, identity: int, behavior: np.ndarray | None,
                        behavior_decisions: int | None) -> None:
        if behavior is None or behavior_decisions is None:
            raise ValueError("strength_behavior archive filter requires a behavior fingerprint")
        vector = np.asarray(behavior, dtype=np.float32).reshape(-1)
        decisions = int(behavior_decisions)
        if decisions < 1:
            raise ValueError("behavior_decisions must be positive")
        if self.behavior_decisions and self.behavior_decisions != decisions:
            raise ValueError("behavior probe decision count changed across snapshots")
        if self.behaviors:
            expected = next(iter(self.behaviors.values())).shape
            if vector.shape != expected:
                raise ValueError("behavior vector layout does not match the probe")
        self.behavior_decisions = decisions
        self.behaviors[identity] = vector
        if not self.archive:
            self.archive.append(identity)
            return
        nearest, nearest_tv = None, float("inf")
        for member in self.archive:
            tv = mean_behavior_tv(vector, self.behaviors[member], decisions)
            if tv < nearest_tv:
                nearest, nearest_tv = member, tv
        if nearest_tv < self.archive_min_tv:
            # Near-duplicate: keep the newer (stronger) of the pair.
            if self.metadata[identity]["update"] > self.metadata[nearest]["update"]:
                self.archive[self.archive.index(nearest)] = identity
                self.archive_replacements += 1
            else:
                self.archive_rejects += 1
            return
        if len(self.archive) < self.archive_size:
            self.archive.append(identity)
            return
        # Full diverse archive: drop the oldest (weakest on ascending lineage).
        oldest = min(self.archive, key=lambda i: self.metadata[i]["update"])
        self.archive[self.archive.index(oldest)] = identity
        self.archive_replacements += 1

    def config(self) -> dict:
        return dict(recent=self.recent, probability=self.probability,
                    archive_every=self.archive_every, archive_size=self.archive_size,
                    archive_share=self.archive_share, archive_filter=self.archive_filter,
                    archive_min_tv=self.archive_min_tv)

    def assignment(self, env: int, match: int) -> list[int]:
        # One guaranteed learner seat. The other seats independently sample a
        # recent snapshot, permitting both historical partners and opponents.
        seats = np.zeros(4, np.int64)
        eligible = sorted(self.models)[-self.recent:]
        if eligible:
            anchor = int(self.rng.integers(4))
            for seat in range(4):
                if seat != anchor and self.rng.random() < self.probability:
                    pool = eligible
                    if self.archive and self.rng.random() < self.archive_share:
                        pool = self.archive
                    seats[seat] = self.rng.choice(pool)
        self.seat_matches.update(seats.tolist())
        return seats.tolist()

    def resolve(self, identity: int) -> HistoryActor:
        if identity == 0:
            return self.actor
        if identity not in self.models:
            raise ValueError(f"unknown Transformer snapshot {identity}")
        return self.models[identity]

    def prune(self, assignments) -> None:
        pinned = {int(i) for seats in assignments.values() for i in seats}
        keep = pinned | set(sorted(self.models)[-self.recent:]) | set(self.archive)
        for identity in list(self.models):
            if identity not in keep:
                del self.models[identity]
                del self.metadata[identity]
                self.behaviors.pop(identity, None)

    def state_dict(self) -> dict:
        return dict(lineage=self.lineage, recent=self.recent, probability=self.probability,
                    archive_every=self.archive_every, archive_size=self.archive_size,
                    archive_share=self.archive_share, archive_filter=self.archive_filter,
                    archive_min_tv=self.archive_min_tv, archive=list(self.archive),
                    behavior_decisions=self.behavior_decisions,
                    behaviors={i: v.copy() for i, v in self.behaviors.items()},
                    archive_rejects=self.archive_rejects,
                    archive_replacements=self.archive_replacements,
                    next_id=self.next_id, rng=self.rng.bit_generator.state,
                    seat_matches=dict(self.seat_matches), decisions=dict(self.decisions),
                    snapshots={i: dict(metadata=self.metadata[i],
                                       model={k: v.detach().cpu().clone()
                                              for k, v in snapshot_weights(model).items()})
                               for i, model in self.models.items()})

    def load_state_dict(self, state: dict) -> None:
        saved = dict(recent=state["recent"], probability=state["probability"],
                     archive_every=state.get("archive_every", 0),
                     archive_size=state.get("archive_size", 16),
                     archive_share=state.get("archive_share", 0.5),
                     archive_filter=state.get("archive_filter", "thin"),
                     archive_min_tv=float(state.get("archive_min_tv", 0.05)))
        if state["lineage"] != self.lineage or saved != self.config():
            raise ValueError("population lineage/config mismatch")
        self.models.clear()
        self.metadata.clear()
        self.behaviors.clear()
        for identity, record in state["snapshots"].items():
            meta = record["metadata"]
            if (meta["identity"] != identity or meta["lineage"] != self.lineage
                    or weights_digest(record["model"]) != meta["sha256"]):
                raise ValueError("population snapshot identity/digest mismatch")
            model = copy.deepcopy(self.actor).requires_grad_(False).train()
            # A snapshot saved before auxiliary heads were added lacks their
            # weights; snapshots only act, so those heads keep the copied values.
            missing, unexpected = model.load_state_dict(record["model"], strict=False)
            if unexpected or not set(missing) <= model.auxiliary_parameter_names():
                raise ValueError(f"population snapshot weights do not fit: missing {missing}, "
                                 f"extra {unexpected}")
            self.models[identity], self.metadata[identity] = model, dict(meta)
        self.archive = [int(i) for i in state.get("archive", [])]
        if not set(self.archive) <= set(self.models):
            raise ValueError("archived snapshot missing from the checkpoint")
        self.behavior_decisions = int(state.get("behavior_decisions", 0))
        for identity, vector in state.get("behaviors", {}).items():
            self.behaviors[int(identity)] = np.asarray(vector, dtype=np.float32)
        if self.archive_filter == "strength_behavior":
            missing = [i for i in self.archive if i not in self.behaviors]
            if missing:
                raise ValueError("archived snapshot missing behavior fingerprint")
        self.archive_rejects = int(state.get("archive_rejects", 0))
        self.archive_replacements = int(state.get("archive_replacements", 0))
        self.next_id = state["next_id"]
        if self.next_id <= max(self.models, default=0):
            raise ValueError("population identity counter went backwards")
        self.rng.bit_generator.state = state["rng"]
        self.seat_matches = Counter(state["seat_matches"])
        self.decisions = Counter(state["decisions"])

    def metrics(self) -> dict:
        # The full per-identity counters grow with every snapshot ever taken
        # (740 KB per line at u53810); they stay in state_dict. Per-update
        # metrics carry the learner, resident snapshots and the totals.
        resident = {0, *self.models}
        return dict(resident_snapshots=len(self.models),
                    eligible_snapshots=sorted(self.models)[-self.recent:],
                    archive=[self.metadata[i]["update"] for i in self.archive],
                    archive_filter=self.archive_filter,
                    archive_min_tv=self.archive_min_tv,
                    archive_rejects=self.archive_rejects,
                    archive_replacements=self.archive_replacements,
                    seat_matches={i: self.seat_matches[i] for i in sorted(resident)},
                    decisions={i: self.decisions[i] for i in sorted(resident)},
                    snapshot_seat_matches=sum(self.seat_matches.values()) - self.seat_matches[0],
                    snapshot_decisions=sum(self.decisions.values()) - self.decisions[0])
