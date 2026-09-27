"""Own-lineage Transformer snapshots, sampled once per match.

Recent snapshots are eligible for new matches. With an archive
(``archive_every`` > 0), every snapshot taken at a multiple of ``archive_every``
updates is also kept; when more than ``archive_size`` are kept, every second
one is dropped (newest kept), so the archive stays spread over the whole run.
A snapshot seat draws from the archive with probability ``archive_share``.
Without an archive the sampler and its random stream are unchanged. Other
snapshots stay resident until their last pinned match ends. The checkpoint embeds all resident weights;
resume restarts environments and drops assignments, but restores the pool/RNG.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
from typing import Any

import numpy as np
import torch

from train.history_model import HistoryActor


def weights_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class HistoryPopulation:
    def __init__(self, actor: HistoryActor, lineage: str, seed: int,
                 recent: int = 4, snapshot_probability: float = 0.5,
                 archive_every: int = 0, archive_size: int = 16,
                 archive_share: float = 0.5) -> None:
        if recent < 1 or not 0 <= snapshot_probability <= 1:
            raise ValueError("positive pool size and probability in [0, 1] required")
        if archive_every < 0 or archive_size < 1 or not 0 <= archive_share <= 1:
            raise ValueError("archive cadence >= 0, size >= 1, share in [0, 1] required")
        self.actor, self.lineage = actor, lineage
        self.recent, self.probability = recent, snapshot_probability
        self.archive_every, self.archive_size = archive_every, archive_size
        self.archive_share = archive_share
        self.archive: list[int] = []
        self.rng = np.random.default_rng(seed)
        self.models: dict[int, HistoryActor] = {}
        self.metadata: dict[int, dict[str, Any]] = {}
        self.next_id = 1
        self.seat_matches: Counter = Counter()
        self.decisions: Counter = Counter()

    def snapshot(self, update: int) -> int:
        identity = self.next_id
        self.next_id += 1
        model = copy.deepcopy(self.actor).requires_grad_(False).train()
        self.models[identity] = model
        self.metadata[identity] = dict(identity=identity, lineage=self.lineage,
                                       update=update, sha256=weights_digest(model.state_dict()))
        if self.archive_every and update % self.archive_every == 0:
            self.archive.append(identity)
            if len(self.archive) > self.archive_size:
                self.archive = self.archive[::-1][::2][::-1]
        return identity

    def config(self) -> dict:
        return dict(recent=self.recent, probability=self.probability,
                    archive_every=self.archive_every, archive_size=self.archive_size,
                    archive_share=self.archive_share)

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

    def state_dict(self) -> dict:
        return dict(lineage=self.lineage, recent=self.recent, probability=self.probability,
                    archive_every=self.archive_every, archive_size=self.archive_size,
                    archive_share=self.archive_share, archive=list(self.archive),
                    next_id=self.next_id, rng=self.rng.bit_generator.state,
                    seat_matches=dict(self.seat_matches), decisions=dict(self.decisions),
                    snapshots={i: dict(metadata=self.metadata[i],
                                       model={k: v.detach().cpu().clone()
                                              for k, v in model.state_dict().items()})
                               for i, model in self.models.items()})

    def load_state_dict(self, state: dict) -> None:
        saved = dict(recent=state["recent"], probability=state["probability"],
                     archive_every=state.get("archive_every", 0),
                     archive_size=state.get("archive_size", 16),
                     archive_share=state.get("archive_share", 0.5))
        if state["lineage"] != self.lineage or saved != self.config():
            raise ValueError("population lineage/config mismatch")
        self.models.clear()
        self.metadata.clear()
        for identity, record in state["snapshots"].items():
            meta = record["metadata"]
            if (meta["identity"] != identity or meta["lineage"] != self.lineage
                    or weights_digest(record["model"]) != meta["sha256"]):
                raise ValueError("population snapshot identity/digest mismatch")
            model = copy.deepcopy(self.actor).requires_grad_(False).train()
            model.load_state_dict(record["model"])
            self.models[identity], self.metadata[identity] = model, dict(meta)
        self.archive = [int(i) for i in state.get("archive", [])]
        if not set(self.archive) <= set(self.models):
            raise ValueError("archived snapshot missing from the checkpoint")
        self.next_id = state["next_id"]
        if self.next_id <= max(self.models, default=0):
            raise ValueError("population identity counter went backwards")
        self.rng.bit_generator.state = state["rng"]
        self.seat_matches = Counter(state["seat_matches"])
        self.decisions = Counter(state["decisions"])

    def metrics(self) -> dict:
        return dict(resident_snapshots=len(self.models),
                    eligible_snapshots=sorted(self.models)[-self.recent:],
                    archive=[self.metadata[i]["update"] for i in self.archive],
                    seat_matches=dict(self.seat_matches), decisions=dict(self.decisions))
