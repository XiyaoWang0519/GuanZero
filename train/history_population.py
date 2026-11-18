"""Own-lineage Transformer snapshots, sampled once per match.

Recent snapshots are eligible for new matches. With an archive
(``archive_every`` > 0), snapshots taken at a multiple of ``archive_every``
updates are considered for the historical archive:
- ``archive_filter="thin"`` (default) keeps the historical thinning bit-identical:
  every second snapshot is dropped (newest kept) when exceeding ``archive_size``.
- ``archive_filter="strength_behavior"`` admits by mean total-variation distance on a
  fixed behavior probe, replaces near-duplicates with newer snapshots, and drops
  the oldest member when full.
- ``archive_filter="nearest_pair"`` preserves long-horizon behavioral diversity by
  merging/evicting the older member of the closest behavioral pair when full,
  optionally bounded by ``archive_max_age_updates``.

A snapshot seat draws from the archive with probability ``archive_share``.
Without an archive the sampler and its random stream are unchanged. Other
snapshots stay resident until their last pinned match ends. The checkpoint embeds
all resident weights; resume restarts environments and drops assignments, but
restores the pool/RNG.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
from typing import Any

import gd
import numpy as np
import torch

from train.history_model import DecisionInputs, HistoryActor, StreamBatch

ARCHIVE_FILTERS = ("thin", "strength_behavior", "nearest_pair")
DEFAULT_BEHAVIOR_PROBE_ROWS = 128
DEFAULT_BEHAVIOR_PROBE_SEED = 20261117


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


def _clone_tensor_cpu(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().to("cpu").contiguous().clone()


def freeze_decision_inputs(inputs: DecisionInputs) -> DecisionInputs:
    """Clone ``DecisionInputs`` to CPU, trim streams to ``max_prefix``, and clear mask overrides."""
    max_prefix = int(inputs.prefix.max().item()) if inputs.decisions else 0
    s = inputs.streams
    streams = StreamBatch(
        tokens=_clone_tensor_cpu(s.tokens[:, :max_prefix]),
        rounds=_clone_tensor_cpu(s.rounds[:, :max_prefix]),
        phases=_clone_tensor_cpu(s.phases[:, :max_prefix]),
        lengths=_clone_tensor_cpu(s.lengths.clamp(max=max_prefix)),
        pos_seat=(_clone_tensor_cpu(s.pos_seat[:, : max_prefix + 1])
                  if s.pos_seat is not None else None),
        pos_round=(_clone_tensor_cpu(s.pos_round[:, : max_prefix + 1])
                   if s.pos_round is not None else None),
    )
    return DecisionInputs(
        streams=streams,
        match_index=_clone_tensor_cpu(inputs.match_index),
        prefix=_clone_tensor_cpu(inputs.prefix),
        obs=_clone_tensor_cpu(inputs.obs),
        seat=_clone_tensor_cpu(inputs.seat),
        cand=_clone_tensor_cpu(inputs.cand),
        offsets=_clone_tensor_cpu(inputs.offsets),
        candidate_rows=_clone_tensor_cpu(inputs.candidate_rows),
        one_decision_per_stream=bool(inputs.one_decision_per_stream),
        match_groups=None,
        match_rank=_clone_tensor_cpu(inputs.match_rank),
        match_slots=int(inputs.match_slots),
        style=_clone_tensor_cpu(inputs.style),
        round_index=_clone_tensor_cpu(inputs.round_index),
        obs_mask_prob=None,
    )


def decision_inputs_on_device(inputs: DecisionInputs, device) -> DecisionInputs:
    """Move ``DecisionInputs`` onto ``device`` (non-blocking on CUDA) without mask overrides."""
    target = torch.device(device)
    non_blocking = target.type == "cuda"

    def move(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.to(target, non_blocking=non_blocking)

    s = inputs.streams
    streams = StreamBatch(
        tokens=move(s.tokens),
        rounds=move(s.rounds),
        phases=move(s.phases),
        lengths=move(s.lengths),
        pos_seat=move(s.pos_seat),
        pos_round=move(s.pos_round),
    )
    return DecisionInputs(
        streams=streams,
        match_index=move(inputs.match_index),
        prefix=move(inputs.prefix),
        obs=move(inputs.obs),
        seat=move(inputs.seat),
        cand=move(inputs.cand),
        offsets=move(inputs.offsets),
        candidate_rows=move(inputs.candidate_rows),
        one_decision_per_stream=bool(inputs.one_decision_per_stream),
        match_groups=None,
        match_rank=move(inputs.match_rank),
        match_slots=int(inputs.match_slots),
        style=move(inputs.style),
        round_index=move(inputs.round_index),
        obs_mask_prob=None,
    )


def serialize_decision_inputs(inputs: DecisionInputs) -> dict[str, Any]:
    """Serialize ``DecisionInputs`` into a plain dict of CPU tensors and Python primitives."""
    s = inputs.streams
    return {
        "streams": {
            "tokens": _clone_tensor_cpu(s.tokens),
            "rounds": _clone_tensor_cpu(s.rounds),
            "phases": _clone_tensor_cpu(s.phases),
            "lengths": _clone_tensor_cpu(s.lengths),
            "pos_seat": _clone_tensor_cpu(s.pos_seat),
            "pos_round": _clone_tensor_cpu(s.pos_round),
        },
        "match_index": _clone_tensor_cpu(inputs.match_index),
        "prefix": _clone_tensor_cpu(inputs.prefix),
        "obs": _clone_tensor_cpu(inputs.obs),
        "seat": _clone_tensor_cpu(inputs.seat),
        "cand": _clone_tensor_cpu(inputs.cand),
        "offsets": _clone_tensor_cpu(inputs.offsets),
        "candidate_rows": _clone_tensor_cpu(inputs.candidate_rows),
        "one_decision_per_stream": bool(inputs.one_decision_per_stream),
        "match_rank": _clone_tensor_cpu(inputs.match_rank),
        "match_slots": int(inputs.match_slots),
        "style": _clone_tensor_cpu(inputs.style),
        "round_index": _clone_tensor_cpu(inputs.round_index),
    }


def deserialize_decision_inputs(payload: dict[str, Any]) -> DecisionInputs:
    """Reconstruct ``DecisionInputs`` from a plain serialized dictionary."""
    sd = payload["streams"]
    streams = StreamBatch(
        tokens=_clone_tensor_cpu(sd["tokens"]),
        rounds=_clone_tensor_cpu(sd["rounds"]),
        phases=_clone_tensor_cpu(sd["phases"]),
        lengths=_clone_tensor_cpu(sd["lengths"]),
        pos_seat=_clone_tensor_cpu(sd.get("pos_seat")),
        pos_round=_clone_tensor_cpu(sd.get("pos_round")),
    )
    return DecisionInputs(
        streams=streams,
        match_index=_clone_tensor_cpu(payload["match_index"]),
        prefix=_clone_tensor_cpu(payload["prefix"]),
        obs=_clone_tensor_cpu(payload["obs"]),
        seat=_clone_tensor_cpu(payload["seat"]),
        cand=_clone_tensor_cpu(payload["cand"]),
        offsets=_clone_tensor_cpu(payload["offsets"]),
        candidate_rows=_clone_tensor_cpu(payload.get("candidate_rows")),
        one_decision_per_stream=bool(payload.get("one_decision_per_stream", False)),
        match_groups=None,
        match_rank=_clone_tensor_cpu(payload.get("match_rank")),
        match_slots=int(payload.get("match_slots", 0)),
        style=_clone_tensor_cpu(payload.get("style")),
        round_index=_clone_tensor_cpu(payload.get("round_index")),
        obs_mask_prob=None,
    )


def _interleave_stratum(
    indices: np.ndarray,
    env_arr: np.ndarray,
    match_arr: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    """Shuffle indices within a stratum and interleave distinct ``(env, match)`` keys."""
    if len(indices) == 0:
        return []
    by_match: dict[tuple[int, int], list[int]] = {}
    for idx in indices.tolist():
        key = (int(env_arr[idx]), int(match_arr[idx]))
        by_match.setdefault(key, []).append(idx)
    keys = list(by_match)
    key_order = rng.permutation(len(keys))
    queues: list[list[int]] = []
    for k_idx in key_order.tolist():
        bucket = by_match[keys[k_idx]]
        perm = rng.permutation(len(bucket))
        # Pop from tail in O(1), so reverse the shuffled order.
        queues.append([bucket[i] for i in perm.tolist()][::-1])
    interleaved: list[int] = []
    while queues:
        next_queues: list[list[int]] = []
        for q in queues:
            interleaved.append(q.pop())
            if q:
                next_queues.append(q)
        queues = next_queues
    return interleaved


def build_behavior_probe(
    actor: HistoryActor,
    rows: int = DEFAULT_BEHAVIOR_PROBE_ROWS,
    *,
    seed: int = DEFAULT_BEHAVIOR_PROBE_SEED,
    num_envs: int = 8,
    max_steps: int = 1024,
    device: str | torch.device = "cpu",
) -> tuple[DecisionInputs, dict[str, Any]]:
    """Build a deterministic, stratified self-play probe for behavior fingerprinting."""
    from train.history_rollout import HistoryCollector, MatchEventStore, SequenceRolloutBuffer

    target_rows = int(rows)
    if target_rows < 1:
        raise ValueError("behavior probe requires at least 1 row")
    probe_device = torch.device(device)
    n_envs = max(2, min(int(num_envs), max(2, target_rows // 4)))
    env = gd.VecEnv(
        num_envs=n_envs,
        num_threads=1,
        seed=int(seed),
        log_public_actions=True,
        log_env_limit=n_envs,
    )
    store = MatchEventStore()
    buffer = SequenceRolloutBuffer(obs_dim=actor.config.obs_dim, act_dim=actor.config.act_dim)
    gen = torch.Generator(device=probe_device).manual_seed(int(seed) + 101)
    probe_actor = copy.deepcopy(actor).to(probe_device).requires_grad_(False).train()
    collector_kwargs: dict[str, Any] = dict(
        device=probe_device,
        kv_cache=not bool(actor.config.window),
    )
    if actor.config.style_input:
        collector_kwargs["row_style"] = lambda _e, _m: 1
    collector = HistoryCollector(env, probe_actor, store, buffer, gen, **collector_kwargs)

    target_pool = max(target_rows * 4, 64)
    steps_done = 0
    step_limit = max(1, int(max_steps))
    while True:
        chunk_steps = min(64, step_limit - steps_done) if steps_done < step_limit else 64
        collector.collect(chunk_steps)
        steps_done += chunk_steps
        n_buf = len(buffer)
        if n_buf == 0:
            continue
        if steps_done >= step_limit:
            break
        has_cross_round = any(bool(np.any(c["round"] >= 1)) for c in buffer.chunks)
        if n_buf >= target_pool and has_cross_round:
            break

    data = buffer.compact()
    rounds_arr = data["round"]
    counts_arr = data["cand_count"]
    env_arr = data["env"]
    match_arr = data["match"]
    own_cards = data["obs"][:, :108].sum(axis=1, dtype=np.int64)
    is_cross_round = (rounds_arr >= 1).astype(np.int64)
    stage = np.where(own_cards >= 19, 0, np.where(own_cards >= 10, 1, 2))
    stratum_id = is_cross_round * 3 + stage
    non_forced = counts_arr > 1

    sel_rng = np.random.default_rng(int(seed) + 202)
    preferred = [
        _interleave_stratum(np.flatnonzero((stratum_id == s) & non_forced), env_arr, match_arr, sel_rng)
        for s in range(6)
    ]
    picked: list[int] = []
    # Pop from head via reversed queues for O(1) round-robin across strata.
    pref_queues = [q[::-1] for q in preferred if q]
    while len(picked) < target_rows and pref_queues:
        next_queues: list[list[int]] = []
        for q in pref_queues:
            picked.append(q.pop())
            if len(picked) == target_rows:
                break
            if q:
                next_queues.append(q)
        pref_queues = next_queues

    if len(picked) < target_rows:
        forced = [
            _interleave_stratum(
                np.flatnonzero((stratum_id == s) & (~non_forced)), env_arr, match_arr, sel_rng
            )
            for s in range(6)
        ]
        forced_queues = [q[::-1] for q in forced if q]
        while len(picked) < target_rows and forced_queues:
            next_queues = []
            for q in forced_queues:
                picked.append(q.pop())
                if len(picked) == target_rows:
                    break
                if q:
                    next_queues.append(q)
            forced_queues = next_queues

    picked_arr = np.asarray(picked, dtype=np.int64)
    if len(picked_arr) < target_rows:
        picked_arr = np.resize(picked_arr, target_rows)
    picked_arr = np.sort(picked_arr[:target_rows])

    inputs, _ = buffer.decision_inputs(picked_arr, store, "cpu")
    frozen = freeze_decision_inputs(inputs)
    cross_round_decisions = (
        int((frozen.round_index >= 1).sum().item())
        if frozen.round_index is not None
        else 0
    )
    meta = {
        "seed": int(seed),
        "decisions": int(frozen.decisions),
        "matches": int(len(frozen.streams.lengths)),
        "cross_round_decisions": cross_round_decisions,
        "non_forced_decisions": int((frozen.counts > 1).sum().item()),
    }
    return frozen, meta


@torch.no_grad()
def behavior_fingerprint(actor: HistoryActor, inputs: DecisionInputs) -> np.ndarray:
    """Concatenated candidate probabilities on a fixed probe (float32)."""
    actor_device = next(actor.parameters()).device
    batch = decision_inputs_on_device(inputs, actor_device) if inputs.obs.device != actor_device else inputs
    logp = actor.candidate_log_probs(batch)
    return logp.exp().detach().cpu().numpy().astype(np.float32, copy=False)


def mean_behavior_tv(a: np.ndarray, b: np.ndarray, decisions: int) -> float:
    """Mean total variation across the probe decisions (probs sum to 1 each)."""
    if decisions < 1:
        raise ValueError("behavior TV needs a positive decision count")
    if a.shape != b.shape:
        raise ValueError("behavior vectors must share the probe layout")
    return float(0.5 * np.abs(a.astype(np.float64) - b.astype(np.float64)).sum() / decisions)


class HistoryPopulation:
    def __init__(
        self,
        actor: HistoryActor,
        lineage: str,
        seed: int,
        recent: int = 4,
        snapshot_probability: float = 0.5,
        archive_every: int = 0,
        archive_size: int = 16,
        archive_share: float = 0.5,
        archive_filter: str = "thin",
        archive_min_tv: float = 0.05,
        archive_max_age_updates: int = 0,
    ) -> None:
        if recent < 1 or not 0 <= snapshot_probability <= 1:
            raise ValueError("positive pool size and probability in [0, 1] required")
        if archive_every < 0 or archive_size < 1 or not 0 <= archive_share <= 1:
            raise ValueError("archive cadence >= 0, size >= 1, share in [0, 1] required")
        if archive_filter not in ARCHIVE_FILTERS:
            raise ValueError(f"archive_filter must be one of {ARCHIVE_FILTERS}")
        if not np.isfinite(archive_min_tv) or not 0 <= archive_min_tv <= 1:
            raise ValueError("archive_min_tv must be finite and in [0, 1]")
        if archive_max_age_updates < 0:
            raise ValueError("archive_max_age_updates must be non-negative")
        self.actor, self.lineage = actor, lineage
        self.recent, self.probability = recent, snapshot_probability
        self.archive_every, self.archive_size = archive_every, archive_size
        self.archive_share = archive_share
        self.archive_filter = archive_filter
        self.archive_min_tv = float(archive_min_tv)
        self.archive_max_age_updates = int(archive_max_age_updates)
        self.archive: list[int] = []
        self.behaviors: dict[int, np.ndarray] = {}
        self.behavior_decisions: int = 0
        self.behavior_probe: DecisionInputs | None = None
        self.behavior_probe_meta: dict[str, Any] | None = None
        self.rng = np.random.default_rng(seed)
        self.models: dict[int, HistoryActor] = {}
        self.metadata: dict[int, dict[str, Any]] = {}
        self.next_id = 1
        self.seat_matches: Counter = Counter()
        self.decisions: Counter = Counter()
        self.archive_rejects: int = 0
        self.archive_replacements: int = 0
        self.archive_age_evictions: int = 0

    def snapshot(
        self,
        update: int,
        behavior: np.ndarray | None = None,
        behavior_decisions: int | None = None,
    ) -> int:
        identity = self.next_id
        self.next_id += 1
        model = copy.deepcopy(self.actor).requires_grad_(False).train()
        self.models[identity] = model
        self.metadata[identity] = dict(
            identity=identity,
            lineage=self.lineage,
            update=update,
            sha256=weights_digest(snapshot_weights(model)),
        )
        if self.archive_every and update % self.archive_every == 0:
            if self.archive_filter == "thin":
                self.archive.append(identity)
                if len(self.archive) > self.archive_size:
                    self.archive = self.archive[::-1][::2][::-1]
            else:
                self._admit_filtered(identity, behavior, behavior_decisions)
        return identity

    def _admit_filtered(
        self,
        identity: int,
        behavior: np.ndarray | None,
        behavior_decisions: int | None,
    ) -> None:
        if behavior is None or behavior_decisions is None:
            raise ValueError(f"{self.archive_filter} archive filter requires a behavior fingerprint")
        vector = np.asarray(behavior, dtype=np.float32).reshape(-1)
        decisions = int(behavior_decisions)
        if decisions < 1:
            raise ValueError("behavior_decisions must be positive")
        missing = [i for i in self.archive if i not in self.behaviors]
        if missing:
            raise ValueError("archived snapshot missing behavior fingerprint")
        if self.behaviors:
            if self.behavior_decisions and self.behavior_decisions != decisions:
                raise ValueError("behavior probe decision count changed across snapshots")
            expected = next(iter(self.behaviors.values())).shape
            if vector.shape != expected:
                raise ValueError("behavior vector layout does not match the probe")

        self.behavior_decisions = decisions
        self.behaviors[identity] = vector
        u_c = int(self.metadata[identity]["update"])

        if self.archive_max_age_updates > 0 and self.archive:
            kept: list[int] = []
            for member in self.archive:
                u_m = int(self.metadata[member]["update"])
                if u_c - u_m > self.archive_max_age_updates:
                    self.archive_age_evictions += 1
                    self.behaviors.pop(member, None)
                else:
                    kept.append(member)
            self.archive = kept

        if not self.archive:
            self.archive.append(identity)
            return

        best_key = None
        nearest = -1
        nearest_tv = float("inf")
        for member in self.archive:
            tv = mean_behavior_tv(vector, self.behaviors[member], decisions)
            key = (tv, int(self.metadata[member]["update"]), member)
            if best_key is None or key < best_key:
                best_key = key
                nearest = member
                nearest_tv = tv

        if nearest_tv < self.archive_min_tv:
            u_nearest = int(self.metadata[nearest]["update"])
            if (u_c, identity) > (u_nearest, nearest):
                self.archive[self.archive.index(nearest)] = identity
                self.behaviors.pop(nearest, None)
                self.archive_replacements += 1
            else:
                self.behaviors.pop(identity, None)
                self.archive_rejects += 1
            return

        if len(self.archive) < self.archive_size:
            self.archive.append(identity)
            return

        if self.archive_filter == "strength_behavior":
            oldest = min(self.archive, key=lambda i: (int(self.metadata[i]["update"]), i))
            u_oldest = int(self.metadata[oldest]["update"])
            if (u_c, identity) > (u_oldest, oldest):
                self.archive[self.archive.index(oldest)] = identity
                self.behaviors.pop(oldest, None)
                self.archive_replacements += 1
            else:
                self.behaviors.pop(identity, None)
                self.archive_rejects += 1
            return

        # "nearest_pair": drop the older member of the closest pair in candidates.
        candidates = [*self.archive, identity]
        while len(candidates) > self.archive_size:
            best_pair_key = None
            best_x = best_y = -1
            for a in range(len(candidates)):
                x = candidates[a]
                u_x = (int(self.metadata[x]["update"]), x)
                bx = self.behaviors[x]
                for b in range(a + 1, len(candidates)):
                    y = candidates[b]
                    u_y = (int(self.metadata[y]["update"]), y)
                    tv = mean_behavior_tv(bx, self.behaviors[y], decisions)
                    pair_key = (tv, min(u_x, u_y), max(u_x, u_y))
                    if best_pair_key is None or pair_key < best_pair_key:
                        best_pair_key = pair_key
                        best_x, best_y = x, y
            u_bx = (int(self.metadata[best_x]["update"]), best_x)
            u_by = (int(self.metadata[best_y]["update"]), best_y)
            victim = best_x if u_bx < u_by else best_y
            if victim == identity:
                candidates.remove(identity)
                self.behaviors.pop(identity, None)
                self.archive_rejects += 1
            else:
                if identity in candidates and len(candidates) == self.archive_size + 1:
                    candidates.remove(identity)
                    idx = candidates.index(victim)
                    candidates[idx] = identity
                else:
                    candidates.remove(victim)
                self.behaviors.pop(victim, None)
                self.archive_replacements += 1
        self.archive = candidates

    def set_behavior_probe(
        self,
        frozen_probe: DecisionInputs,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if frozen_probe.decisions < 1:
            raise ValueError("behavior probe needs a positive decision count")
        self.behavior_probe = freeze_decision_inputs(frozen_probe)
        self.behavior_probe_meta = copy.deepcopy(meta) if meta is not None else None

    def rebind_behavior_probe(
        self,
        inputs: DecisionInputs,
        *,
        frozen_probe: DecisionInputs | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Recompute archive fingerprints on a new frozen probe atomically."""
        if self.archive_filter not in ("strength_behavior", "nearest_pair"):
            raise ValueError("rebind_behavior_probe requires strength_behavior or nearest_pair")
        decisions = int(inputs.decisions)
        if decisions < 1:
            raise ValueError("behavior probe needs a positive decision count")
        rebound = {
            identity: behavior_fingerprint(self.resolve(identity), inputs)
            for identity in self.archive
        }
        self.behaviors = rebound
        self.behavior_decisions = decisions
        self.behavior_probe = (
            freeze_decision_inputs(frozen_probe)
            if frozen_probe is not None
            else freeze_decision_inputs(inputs)
        )
        self.behavior_probe_meta = copy.deepcopy(meta) if meta is not None else None

    def needs_probe_rebind(
        self,
        expected_decisions: int | None = None,
        expected_seed: int | None = None,
    ) -> bool:
        if self.archive_filter == "thin":
            return False
        if self.behavior_probe is None:
            return True
        if expected_decisions is not None and int(self.behavior_probe.decisions) != int(expected_decisions):
            return True
        if (
            expected_seed is not None
            and self.behavior_probe_meta is not None
            and "seed" in self.behavior_probe_meta
            and int(self.behavior_probe_meta["seed"]) != int(expected_seed)
        ):
            return True
        if any(i not in self.behaviors for i in self.archive):
            return True
        if self.archive and self.behavior_decisions != int(self.behavior_probe.decisions):
            return True
        return False

    def config(self) -> dict:
        return dict(
            recent=self.recent,
            probability=self.probability,
            archive_every=self.archive_every,
            archive_size=self.archive_size,
            archive_share=self.archive_share,
            archive_filter=self.archive_filter,
            archive_min_tv=self.archive_min_tv,
            archive_max_age_updates=self.archive_max_age_updates,
        )

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
        archive_set = set(self.archive)
        keep = pinned | set(sorted(self.models)[-self.recent:]) | archive_set
        for identity in list(self.models):
            if identity not in keep:
                del self.models[identity]
                del self.metadata[identity]
        for identity in list(self.behaviors):
            if identity not in archive_set and identity not in keep:
                del self.behaviors[identity]

    def state_dict(self) -> dict:
        return dict(
            lineage=self.lineage,
            recent=self.recent,
            probability=self.probability,
            archive_every=self.archive_every,
            archive_size=self.archive_size,
            archive_share=self.archive_share,
            archive_filter=self.archive_filter,
            archive_min_tv=self.archive_min_tv,
            archive_max_age_updates=self.archive_max_age_updates,
            archive=list(self.archive),
            behavior_decisions=self.behavior_decisions,
            behaviors={i: v.copy() for i, v in self.behaviors.items()},
            behavior_probe=(
                serialize_decision_inputs(self.behavior_probe)
                if self.behavior_probe is not None
                else None
            ),
            behavior_probe_meta=(
                copy.deepcopy(self.behavior_probe_meta)
                if self.behavior_probe_meta is not None
                else None
            ),
            archive_rejects=self.archive_rejects,
            archive_replacements=self.archive_replacements,
            archive_age_evictions=self.archive_age_evictions,
            next_id=self.next_id,
            rng=self.rng.bit_generator.state,
            seat_matches=dict(self.seat_matches),
            decisions=dict(self.decisions),
            snapshots={
                i: dict(
                    metadata=self.metadata[i],
                    model={
                        k: v.detach().cpu().clone()
                        for k, v in snapshot_weights(model).items()
                    },
                )
                for i, model in self.models.items()
            },
        )

    def _load_snapshots(self, state: dict) -> tuple[dict[int, HistoryActor], dict[int, dict[str, Any]]]:
        models: dict[int, HistoryActor] = {}
        metadata: dict[int, dict[str, Any]] = {}
        for identity, record in state["snapshots"].items():
            meta = record["metadata"]
            if (
                meta["identity"] != identity
                or meta["lineage"] != self.lineage
                or weights_digest(record["model"]) != meta["sha256"]
            ):
                raise ValueError("population snapshot identity/digest mismatch")
            model = copy.deepcopy(self.actor).requires_grad_(False).train()
            # A snapshot saved before auxiliary heads were added lacks their
            # weights; snapshots only act, so those heads keep the copied values.
            missing, unexpected = model.load_state_dict(record["model"], strict=False)
            if unexpected or not set(missing) <= model.auxiliary_parameter_names():
                raise ValueError(
                    f"population snapshot weights do not fit: missing {missing}, extra {unexpected}"
                )
            models[identity] = model
            metadata[identity] = dict(meta)
        return models, metadata

    def load_state_dict(self, state: dict) -> None:
        saved = dict(
            recent=state["recent"],
            probability=state["probability"],
            archive_every=state.get("archive_every", 0),
            archive_size=state.get("archive_size", 16),
            archive_share=state.get("archive_share", 0.5),
            archive_filter=state.get("archive_filter", "thin"),
            archive_min_tv=float(state.get("archive_min_tv", 0.05)),
            archive_max_age_updates=int(state.get("archive_max_age_updates", 0)),
        )
        if state["lineage"] != self.lineage or saved != self.config():
            raise ValueError("population lineage/config mismatch")
        models, metadata = self._load_snapshots(state)
        archive = [int(i) for i in state.get("archive", [])]
        if not set(archive) <= set(models):
            raise ValueError("archived snapshot missing from the checkpoint")
        next_id = state["next_id"]
        if next_id <= max(models, default=0):
            raise ValueError("population identity counter went backwards")

        raw_bd = state.get("behavior_decisions", 0)
        bd = (
            int(raw_bd)
            if isinstance(raw_bd, (int, np.integer)) and not isinstance(raw_bd, bool)
            else 0
        )
        behaviors = {
            int(identity): np.asarray(vector, dtype=np.float32)
            for identity, vector in state.get("behaviors", {}).items()
        }
        raw_probe = state.get("behavior_probe")
        probe = deserialize_decision_inputs(raw_probe) if raw_probe is not None else None
        raw_meta = state.get("behavior_probe_meta")
        probe_meta = copy.deepcopy(raw_meta) if raw_meta is not None else None

        if self.archive_filter in ("strength_behavior", "nearest_pair") and (
            behaviors or probe is not None
        ):
            missing = [i for i in archive if i not in behaviors]
            if missing:
                raise ValueError("archived snapshot missing behavior fingerprint")

        self.models = models
        self.metadata = metadata
        self.archive = archive
        self.behavior_decisions = bd
        self.behaviors = behaviors
        self.behavior_probe = probe
        self.behavior_probe_meta = probe_meta
        self.archive_rejects = int(state.get("archive_rejects", 0))
        self.archive_replacements = int(state.get("archive_replacements", 0))
        self.archive_age_evictions = int(state.get("archive_age_evictions", 0))
        self.next_id = next_id
        self.rng.bit_generator.state = state["rng"]
        self.seat_matches = Counter(state["seat_matches"])
        self.decisions = Counter(state["decisions"])

    def adopt_state_dict(
        self,
        state: dict,
        *,
        current_update: int | None = None,
        expected_probe_rows: int | None = None,
        expected_probe_seed: int | None = None,
    ) -> None:
        """Load a checkpoint population under ``self.config()``, adapting archive and probe."""
        if state["lineage"] != self.lineage:
            raise ValueError("population lineage mismatch")
        models, metadata = self._load_snapshots(state)
        raw_archive = [int(i) for i in state.get("archive", [])]
        if not set(raw_archive) <= set(models):
            raise ValueError("archived snapshot missing from the checkpoint")
        next_id = state["next_id"]
        if next_id <= max(models, default=0):
            raise ValueError("population identity counter went backwards")

        if self.archive_every == 0:
            adapted_archive: list[int] = []
        else:
            adapted_archive = list(raw_archive)
            if self.archive_max_age_updates > 0:
                ref_update = (
                    int(current_update)
                    if current_update is not None
                    else max((int(m["update"]) for m in metadata.values()), default=0)
                )
                adapted_archive = [
                    i
                    for i in adapted_archive
                    if ref_update - int(metadata[i]["update"]) <= self.archive_max_age_updates
                ]
            if len(adapted_archive) > self.archive_size:
                keep_ids = set(
                    sorted(adapted_archive, key=lambda i: (int(metadata[i]["update"]), i))[
                        -self.archive_size :
                    ]
                )
                adapted_archive = [i for i in adapted_archive if i in keep_ids]

        raw_probe = state.get("behavior_probe")
        raw_meta = state.get("behavior_probe_meta")
        raw_behaviors = {
            int(identity): np.asarray(vector, dtype=np.float32)
            for identity, vector in state.get("behaviors", {}).items()
        }
        raw_bd = state.get("behavior_decisions", 0)
        bd = (
            int(raw_bd)
            if isinstance(raw_bd, (int, np.integer)) and not isinstance(raw_bd, bool)
            else 0
        )
        probe = deserialize_decision_inputs(raw_probe) if raw_probe is not None else None
        rows_ok = expected_probe_rows is None or (
            probe is not None and int(probe.decisions) == int(expected_probe_rows)
        )
        seed_ok = (
            expected_probe_seed is None
            or raw_meta is None
            or "seed" not in raw_meta
            or int(raw_meta["seed"]) == int(expected_probe_seed)
        )
        can_preserve = (
            self.archive_filter != "thin"
            and probe is not None
            and rows_ok
            and seed_ok
            and all(i in raw_behaviors for i in adapted_archive)
            and (not adapted_archive or bd == int(probe.decisions))
        )

        self.models = models
        self.metadata = metadata
        self.archive = adapted_archive
        self.next_id = next_id
        self.rng.bit_generator.state = state["rng"]
        self.seat_matches = Counter(state["seat_matches"])
        self.decisions = Counter(state["decisions"])
        self.archive_rejects = int(state.get("archive_rejects", 0))
        self.archive_replacements = int(state.get("archive_replacements", 0))
        self.archive_age_evictions = int(state.get("archive_age_evictions", 0))
        if can_preserve:
            self.behavior_probe = probe
            self.behavior_probe_meta = copy.deepcopy(raw_meta) if raw_meta is not None else None
            self.behaviors = {i: raw_behaviors[i] for i in adapted_archive}
            self.behavior_decisions = int(probe.decisions) if not bd else bd
        else:
            self.behaviors = {}
            self.behavior_decisions = 0
            self.behavior_probe = None
            self.behavior_probe_meta = None

    def archive_diversity_stats(self) -> dict[str, float | int | str]:
        updates = [int(self.metadata[i]["update"]) for i in self.archive]
        age_span = (max(updates) - min(updates)) if len(updates) >= 2 else 0
        min_pair_tv = 0.0
        mean_pair_tv = 0.0
        if (
            self.archive_filter != "thin"
            and len(self.archive) >= 2
            and self.behavior_decisions >= 1
            and all(i in self.behaviors for i in self.archive)
        ):
            pair_tvs: list[float] = []
            for a in range(len(self.archive)):
                ba = self.behaviors[self.archive[a]]
                for b in range(a + 1, len(self.archive)):
                    bb = self.behaviors[self.archive[b]]
                    pair_tvs.append(mean_behavior_tv(ba, bb, self.behavior_decisions))
            min_pair_tv = float(min(pair_tvs))
            mean_pair_tv = float(sum(pair_tvs) / len(pair_tvs))
        return {
            "archive_filter": self.archive_filter,
            "archive_min_tv": self.archive_min_tv,
            "archive_max_age_updates": self.archive_max_age_updates,
            "archive_rejects": self.archive_rejects,
            "archive_replacements": self.archive_replacements,
            "archive_age_evictions": self.archive_age_evictions,
            "archive_min_pair_tv": min_pair_tv,
            "archive_mean_pair_tv": mean_pair_tv,
            "archive_age_span": age_span,
        }

    def metrics(self) -> dict:
        # The full per-identity counters grow with every snapshot ever taken
        # (740 KB per line at u53810); they stay in state_dict. Per-update
        # metrics carry the learner, resident snapshots and the totals.
        resident = {0, *self.models}
        return {
            "resident_snapshots": len(self.models),
            "eligible_snapshots": sorted(self.models)[-self.recent:],
            "archive": [self.metadata[i]["update"] for i in self.archive],
            "seat_matches": {i: self.seat_matches[i] for i in sorted(resident)},
            "decisions": {i: self.decisions[i] for i in sorted(resident)},
            "snapshot_seat_matches": sum(self.seat_matches.values()) - self.seat_matches[0],
            "snapshot_decisions": sum(self.decisions.values()) - self.decisions[0],
            **self.archive_diversity_stats(),
        }
