"""Cold-start PPO for the history Transformer player (STAGE_C T2, DESIGN v0.6 8).

Training boundary (DESIGN 1.1): ``fresh_player`` is the only initialization
path; ``--resume`` accepts a ``history_ppo`` checkpoint of this lineage and
nothing else. There is no teacher, no old MLP, no candidate pruning and no
KL toward anything; the only old-policy comparison is PPO's ratio against
the collecting actor's stored log-probabilities.

Reward and credit assignment replicate ``train/ppo.py``: the reward of an
``(env, team, round)`` trajectory is ``RoundResult.seat_return[team]`` (the
engine's per-round team level return, +3/+2/+1 by finish order with the
opposite sign for the other team, level-A handling included), written on the
team's last stored row of that round with ``done``; every other reward is
zero; GAE uses ``gamma`` 1.0 and ``gae_lambda`` 0.95 with a terminal boundary
at round end and no bootstrap into the next round; values come from the
current critic at ``learn`` time; advantages are normalized per minibatch;
only completed trajectories train and unfinished rounds carry over.

At every epoch and minibatch the actor's log-probabilities are recomputed
from the raw public tokens of the minibatch's matches (``StreamBatch`` of
whole matches, per-row ``prefix``), so gradients flow through the stream
encoder. Minibatches are groups of whole matches, never independent rows.

Usage::

    python -m train.history_ppo --output DIR --updates N --num-envs E \\
        --steps-per-update S --seed K [--width --layers --heads --window --lr \\
        --clip --entropy --epochs --minibatch-matches --resume CKPT]
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass, fields
import json
import math
import os
import signal
from pathlib import Path
import sys
import time
from typing import Any, Iterator, Sequence
import uuid

import gd
import numpy as np
import torch
from torch.nn import functional as F
from infra.history_artifacts import engine_compatible, engine_digest, source_identity, sha256

from train.ckpt import restore_rng, rng_state
from train.history_model import (extend_actor, STAGE, TOKEN_SCHEMA_VERSION,
                                 HistoryActor, HistoryCritic, HistoryPolicyConfig,
                                 checkpoint_payload, config_record, count_parameters,
                                 fresh_player, load_history_checkpoint, save_history_checkpoint)
from train.history_rollout import (CollectStats, HistoryCollector, MatchEventStore,
                                   SequenceRolloutBuffer, TrainingBatch, _PhaseTimer)
from train.history_population import HistoryPopulation
from train.history_response import RESPONSE_SCHEMA, opponent_response_labels
from train.history_aux import (AUX_HEADS, belief_loss, next_token_loss, next_token_loss_compact,
                               outcome_loss, parse_aux_heads, round_outcomes)
from train.history_transfers import runtime_settings
from train.logs import TOKEN_DIM

REWARD_SEMANTICS = {
    "reward": "RoundResult.seat_return[team], team = seat % 2, on the team's last stored "
              "row of the (env, team, round) trajectory; zero elsewhere",
    "terminal": "done at round end; no bootstrap across rounds (value past the last "
                "row is zero)",
    "values": "current HistoryCritic(obs, hidden_counts) at learn time",
    "trajectories": "completed rounds only; unfinished rounds carry over to the next update",
    "advantages": "GAE(gamma, gae_lambda), normalized per minibatch",
    "tribute": "engine greedy heuristic for Tribute/BackTribute rows; those rows are "
               "never PPO rows but their public exchange events are in the stream",
}


@dataclass
class HistoryPPOConfig:
    # architecture (HistoryPolicyConfig)
    width: int = 64
    layers: int = 2
    heads: int = 4
    window: int = 0
    max_rounds: int = 16
    response_mode: str = "none"
    response_coef: float = 0.1
    # Auxiliary heads on the shared encoder (train/history_aux.py): a subset of
    # "next", "belief", "outcome". Their summed loss enters the actor loss with
    # weight aux_coef, ramped linearly over aux_warmup_updates updates from the
    # update the heads were inserted (0: full weight at once). A resume may add
    # heads to a trained lineage (``--resume-set aux_heads=...``): the policy is
    # unchanged at insertion, only the new heads' weights are fresh.
    aux_heads: str = ""
    aux_coef: float = 0.1
    aux_warmup_updates: int = 0
    # rollout
    num_envs: int = 16
    num_threads: int = 1
    steps_per_update: int = 64
    seed: int = 0
    causal_sdpa: bool = False           # opt-in until same-device CUDA A/B acceptance
    rollout_kv_cache: bool = False      # public-only, invalidated across learner updates
    rollout_batched_attention: bool = False  # opt-in; target-device bitwise acceptance required
    rollout_wide_projection: bool = False  # with batched attention: one q/out GEMM; FP32, not bitwise
    rollout_private_graphs: bool = False  # bounded CUDA inference graphs; sampling stays eager
    learner_batched_attention: bool = False  # one padded attention call per minibatch; FP32, not bitwise
    # Learner encode in up to N length groups, each stream only as far as its rows
    # read (history_model.length_groups): same loss, far less padding; FP32, not bitwise.
    learner_length_groups: int = 0
    # Auxiliary response head on the executed candidates only (the only rows its
    # loss reads); the policy still scores every candidate. FP32, not bitwise.
    learner_chosen_response: bool = False
    # With length groups and the "next" head: run the head and a fused loss on
    # the valid positions only (one log-softmax, one BCE, one column sum) instead
    # of per field over every padded position. Same loss; FP32, not bitwise.
    learner_compact_next: bool = False
    rollout_graph_budget_mb: int = 512
    rollout_graph_policy_budget_mb: int = 128  # per-policy cap inside the total budget
    rollout_triton_cache: bool = False  # lossless KV update/packing; requires Triton
    rollout_triton_min_batch: int = 1
    # One shared page pool for every identity's public KV cache
    # (train/history_paged_cache.py): a fixed number of operations per encode
    # call instead of several per cached match. Bitwise on CPU; replaces Triton.
    rollout_paged_cache: bool = False
    # Paged cache: pad attention keys and the decision memory to the next 64-token
    # page instead of the next power of two. Tier 2 (FP32 reduction shapes change).
    rollout_page_span: bool = False
    # Rebuild the learner's public KV cache for every current match at the start
    # of each collection, all streams in one pass per prefill chunk, instead of
    # stream by stream as learner rows first read them. Tier 2 (learner seats'
    # FP32 reduction order: the same tokens are encoded in different batches).
    rollout_prefill_learner_cache: bool = False
    # Paged cache: run each append's key/value gather and attention, the
    # learner's private attention over its memory and (with the merged snapshot
    # encoder) the merged snapshot call's attention over its memory, in up to N
    # end-sorted span groups, each padded only to its own longest stream
    # (train/history_paged_cache.span_plan). 1 is off. Tier 2 (FP32 shapes change).
    rollout_span_groups: int = 1
    # Paged cache, batched rollout attention: store the learner's private-attention
    # keys/values (kv_proj of its memory) per token when encoded, instead of
    # projecting the whole gathered memory at every step. Tier 2 (GEMM row count).
    rollout_private_kv_pages: bool = False
    # With the paged cache and the learner cache prefill: prefill streams rebuilt
    # from BOS in one causal pass per length group (no pool gather, no mask)
    # instead of one masked append per 128-token chunk. Tier 2 (learner seats).
    rollout_causal_prefill: bool = False
    # Paged cache: run every cache append's attention (and, with private K/V
    # pages, the learner's private attention) in the Triton paged-attention kernel,
    # reading keys/values in place from the pool pages up to each stream's end
    # (train/history_paged_attention.py). CUDA only: elsewhere a no-op (bitwise
    # the default path). Tier 2 (FP32 softmax order); needs the GPU gate first.
    rollout_paged_attention_kernel: bool = False
    profile_collection: bool = False   # synchronized phase timings; diagnostic only
    profile_learn: bool = False        # the same for the PPO update's phases
    profile_collection_warmup: int = 0  # unprofiled updates of this process before profiling
    # All snapshot identities' play rows of a vector step in one merged actor call
    # (train/history_snapshot_batch.py): same distribution, FP32 reduction-order
    # noise on snapshot seats only; learner rows, sampling and generator unchanged.
    batch_snapshot_policies: bool = False
    # Diagnostic A/B inside one process: "N:on,M:off,..." blocks of this process's
    # updates; the last block's arm persists. Empty: batch_snapshot_policies throughout.
    batch_snapshot_policies_schedule: str = ""
    # With merged snapshot inference and the paged cache: those identities' public
    # streams are encoded in one pass per step over stacked encoder weights
    # (train/history_paged_cache.merged_encode); tier 2, snapshot seats only.
    batch_snapshot_encoder: bool = False
    # With merged snapshot inference: each stacked snapshot head stores its
    # query/key and value/output projections pre-multiplied, so each attention
    # side is one batched linear layer instead of four kernels
    # (train/history_snapshot_batch._head_values). Tier 2, snapshot seats only.
    rollout_merged_head_prefold: bool = False
    # Release the CUDA caching allocator's free blocks (torch.cuda.empty_cache)
    # after collect and after learn. Allocator timing only; no numeric change.
    # None follows the update's merged-snapshot arm: with the arm on, no snapshot
    # private graphs are captured, so nothing else ever trims and each process's
    # reserved memory only ratchets upward.
    rollout_trim_cuda_cache: bool | None = None
    # With the trim on: skip the after-collect / after-learn trims while the
    # cached free bytes (reserved - allocated) are at most this many MiB, so
    # the next phase reuses the cache instead of re-allocating it. 0 trims
    # always. Allocator timing only; the graphs_released trim never skips.
    rollout_trim_slack_mb: int = 0
    rollout_device: str | None = None  # None shares the learner device
    # learner
    lr: float = 3e-4
    critic_lr: float | None = None       # None: same as lr
    # Linear learning-rate decay: from update lr_decay_start (absolute, so restarts
    # land on the same schedule) over lr_decay_updates updates, both rates scale
    # to lr_final / lr and then stay there. lr_decay_updates 0: constant rates.
    lr_final: float = 0.0
    lr_decay_start: int = 0
    lr_decay_updates: int = 0
    clip: float = 0.2
    entropy: float = 0.01
    value_coef: float = 1.0
    epochs: int = 2
    minibatch_matches: int = 4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    grad_clip: float = 10.0
    normalize_advantages: bool = True
    # exploration floor, learner seats only: pi_b = (1 - eps) softmax(logits / T) + eps / n
    rollout_temperature: float = 1.0
    rollout_epsilon: float = 0.0
    behaviour_weight_cap: float = 1.0
    # lifecycle
    updates: int = 1
    checkpoint_updates: int = 1
    torch_threads: int = 0               # 0 leaves torch's default
    snapshot_updates: int = 2           # 0 disables for isolated throughput sweeps
    # historical opponent archive; 0 keeps only the recent snapshots
    population_archive_every: int = 0
    population_archive_size: int = 16
    population_archive_share: float = 0.5
    population_recent: int = 4
    snapshot_probability: float = 0.5
    # data parallel (train/history_ddp.py): normalize advantages and weight the
    # loss over the union of the ranks' minibatches, as one minibatch would be
    ddp_global_minibatch: bool = False
    # Gradient all-reduce transport of the data-parallel ranks: "gloo" (TCP ring)
    # or "shm" (same-node /dev/shm mapping, train/history_shm_reduce.py) that adds
    # in gloo's ring order, so it equals gloo bitwise at any world size; a start-up
    # self-check falls back to gloo on any mismatch. Opt-in for operational reasons
    # (/dev/shm size, spin barriers), not numerics.
    ddp_reduce: str = "gloo"
    # torch.optim.Adam(fused=True) for both optimizers: one fused kernel per step
    # with device-resident step counters; FP32 arithmetic differs from the
    # foreach path (tier 2). Checkpoint optimizer state loads either way.
    adam_fused: bool = False
    # Planted-habit diagnostic (train/history_habit.py; approved exceptions, never
    # the main lineage). habit_pack: frozen base checkpoint of the styled opponent
    # pack (empty: off). habit_init: checkpoint the learner continues from.
    habit_pack: str = ""
    habit_axis: str = "lead_single"
    habit_strength: float = 1.0
    habit_view: str = "full"            # full | round | oracle
    habit_init: str = ""

    def __post_init__(self) -> None:
        if self.rollout_graph_budget_mb < 1 or self.rollout_graph_policy_budget_mb < 1:
            raise ValueError('rollout graph memory budget must be positive')
        if self.rollout_triton_min_batch < 1:
            raise ValueError('rollout Triton minimum batch must be positive')
        if self.profile_collection_warmup < 0:
            raise ValueError('profile_collection_warmup must not be negative')
        if self.rollout_trim_slack_mb < 0:
            raise ValueError('rollout_trim_slack_mb must not be negative')
        parse_arm_schedule(self.batch_snapshot_policies_schedule)
        if (self.batch_snapshot_policies or self.batch_snapshot_policies_schedule) and (
                self.window or self.response_mode == "explicit"):
            raise ValueError("batch_snapshot_policies requires full history and no explicit "
                             "response bridge")
        if (self.rollout_private_graphs or self.rollout_triton_cache
                or self.rollout_paged_cache) and not self.rollout_kv_cache:
            raise ValueError('KV cache layouts and CUDA rollout optimizations require rollout_kv_cache')
        if self.rollout_paged_cache and self.rollout_triton_cache:
            raise ValueError('rollout_paged_cache replaces rollout_triton_cache; choose one')
        if self.rollout_page_span and (not self.rollout_paged_cache
                                       or self.rollout_private_graphs):
            raise ValueError('rollout_page_span requires rollout_paged_cache and no private graphs')
        if not 1 <= self.rollout_span_groups <= 4 or (self.rollout_span_groups > 1 and (
                not self.rollout_paged_cache or self.rollout_private_graphs)):
            raise ValueError('rollout_span_groups must be 1-4; more than one requires '
                             'rollout_paged_cache and no private graphs')
        if self.rollout_private_kv_pages and (not self.rollout_paged_cache
                                              or not self.rollout_batched_attention
                                              or self.rollout_private_graphs):
            raise ValueError('rollout_private_kv_pages requires rollout_paged_cache, '
                             'rollout_batched_attention and no private graphs')
        if self.rollout_causal_prefill and not (self.rollout_paged_cache
                                                and self.rollout_prefill_learner_cache):
            raise ValueError('rollout_causal_prefill requires rollout_paged_cache and '
                             'rollout_prefill_learner_cache')
        if self.rollout_paged_attention_kernel and (not self.rollout_paged_cache
                                                    or self.rollout_private_graphs):
            raise ValueError('rollout_paged_attention_kernel requires rollout_paged_cache and '
                             'no private graphs')
        if self.rollout_prefill_learner_cache and (not self.rollout_kv_cache
                                                   or self.habit_view == "round"):
            raise ValueError('rollout_prefill_learner_cache requires the match-keyed '
                             'rollout_kv_cache')
        if self.batch_snapshot_encoder and not (
                self.rollout_paged_cache
                and (self.batch_snapshot_policies or self.batch_snapshot_policies_schedule)):
            raise ValueError('batch_snapshot_encoder requires rollout_paged_cache and merged '
                             'snapshot policies')
        if self.rollout_merged_head_prefold and not (self.batch_snapshot_policies
                                                     or self.batch_snapshot_policies_schedule):
            raise ValueError('rollout_merged_head_prefold requires merged snapshot policies')
        if self.rollout_wide_projection and not self.rollout_batched_attention:
            raise ValueError('rollout_wide_projection requires rollout_batched_attention')
        if self.ddp_reduce not in ("gloo", "shm"):
            raise ValueError("ddp_reduce must be gloo or shm")
        if self.rollout_device not in (None, 'cpu', 'cuda'):
            raise ValueError('rollout_device must be cpu, cuda, or None')
        if self.learner_length_groups < 0 or (self.learner_length_groups and self.window):
            raise ValueError("learner_length_groups must not be negative and needs full history")
        if min(self.num_envs, self.steps_per_update, self.epochs, self.minibatch_matches,
               self.checkpoint_updates) <= 0:
            raise ValueError("environment, step, epoch, minibatch and checkpoint counts "
                             "must be positive")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma must be in (0, 1] and gae_lambda in [0, 1]")
        if self.lr <= 0 or self.clip <= 0 or self.entropy < 0:
            raise ValueError("lr and clip must be positive; entropy must not be negative")
        if self.lr_decay_updates < 0 or self.lr_decay_start < 0 or (
                self.lr_decay_updates and not 0 < self.lr_final <= self.lr):
            raise ValueError("lr decay needs non-negative counts and 0 < lr_final <= lr")
        if (not 0 < self.rollout_temperature < math.inf or not 0 <= self.rollout_epsilon <= 1
                or not 0 < self.behaviour_weight_cap < math.inf):
            raise ValueError("rollout_temperature and behaviour_weight_cap must be positive "
                             "and finite; rollout_epsilon must be in [0, 1]")
        if (self.snapshot_updates < 0 or self.population_recent < 1
                or not 0 <= self.snapshot_probability <= 1
                or self.population_archive_every < 0 or self.population_archive_size < 1
                or not 0 <= self.population_archive_share <= 1):
            raise ValueError("invalid population schedule")
        if self.rollout_kv_cache and self.window:
            raise ValueError("rollout KV cache requires full history")
        if self.response_mode not in ("none", "auxiliary", "explicit"):
            raise ValueError("unknown response_mode")
        if not math.isfinite(self.response_coef) or self.response_coef < 0:
            raise ValueError("response_coef must be finite and nonnegative")
        if self.response_mode != "none" and self.response_coef == 0:
            raise ValueError("prediction arms require a positive response_coef")
        if self.learner_chosen_response and self.response_mode != "auxiliary":
            raise ValueError("learner_chosen_response needs the auxiliary response mode")
        if self.learner_compact_next and not self.learner_length_groups:
            raise ValueError("learner_compact_next needs learner_length_groups")
        parse_aux_heads(self.aux_heads)
        if not math.isfinite(self.aux_coef) or self.aux_coef < 0 or self.aux_warmup_updates < 0:
            raise ValueError("aux_coef must be finite and nonnegative; aux_warmup_updates >= 0")
        if self.aux_heads and self.aux_coef == 0:
            raise ValueError("auxiliary heads require a positive aux_coef")
        from train.history_habit import HABIT_VIEWS, STYLE_AXES
        if self.habit_pack:
            if self.habit_axis not in STYLE_AXES or self.habit_view not in HABIT_VIEWS:
                raise ValueError("unknown habit axis or view")
            if not (math.isfinite(self.habit_strength) and self.habit_strength >= 0):
                raise ValueError("habit_strength must be finite and nonnegative")
            if (self.snapshot_updates or self.batch_snapshot_policies
                    or self.batch_snapshot_policies_schedule or self.window):
                raise ValueError("the habit pack replaces snapshots: snapshot_updates 0, no "
                                 "merged snapshot inference, full history")
            if self.habit_view == "oracle" and self.rollout_private_graphs:
                raise ValueError("private graphs do not capture the oracle style input")
        elif self.habit_init or self.habit_view != "full":
            raise ValueError("habit_init and habit_view need habit_pack")

    def policy_config(self) -> HistoryPolicyConfig:
        return HistoryPolicyConfig(width=self.width, layers=self.layers, heads=self.heads,
                                   window=self.window, max_rounds=self.max_rounds,
                                   response_mode=self.response_mode,
                                   aux_heads=self.aux_heads,
                                   style_input=bool(self.habit_pack)
                                   and self.habit_view == "oracle")

    @property
    def habit_round(self) -> bool:
        return bool(self.habit_pack) and self.habit_view == "round"

    @classmethod
    def from_payload(cls, config: dict[str, Any], **overrides: Any) -> "HistoryPPOConfig":
        names = {f.name for f in fields(cls)}
        values = {k: v for k, v in config.items() if k in names}
        values.update(overrides)
        return cls(**values)


def habit_init_player(config: HistoryPPOConfig
                      ) -> tuple[HistoryActor, HistoryCritic, dict[str, Any]]:
    """Planted-habit exception: continue from a trained history checkpoint.

    Same architecture required; a style-input (ORACLE) actor adds only its
    zero-initialised style embedding, so its initial policy equals the source.
    """
    base_actor, base_critic, payload = load_history_checkpoint(config.habit_init, "cpu")
    policy = config.policy_config()
    if config_record(base_actor.config) != config_record(
            HistoryPolicyConfig(**{**asdict(policy), "style_input": False})):
        raise ValueError("habit_init architecture differs from the configured one")
    actor, critic = HistoryActor(policy), HistoryCritic(policy)
    missing, unexpected = actor.load_state_dict(base_actor.state_dict(), strict=False)
    if unexpected or set(missing) != ({"style_embedding.weight"} if policy.style_input else set()):
        raise ValueError(f"habit_init weights do not fit: missing {missing}, extra {unexpected}")
    critic.load_state_dict(base_critic.state_dict())
    return actor, critic, payload


def load_extended_optimizers(trainer: "HistoryTrainer", payload: dict[str, Any], *,
                             keep_lr: bool) -> None:
    """Adam moments of a checkpoint whose actor had fewer parameters: the extra
    ones (a style embedding, auxiliary heads) are registered last and start
    fresh. ``keep_lr`` keeps the configured learning rates instead of the saved."""
    for name, optimizer in (("actor", trainer.actor_optimizer),
                            ("critic", trainer.critic_optimizer)):
        saved = copy.deepcopy(payload["optimizer"][name])
        count = len(list(optimizer.param_groups[0]["params"]))
        known = saved["param_groups"][0]["params"]
        if len(saved["param_groups"]) != 1 or len(known) > count or (
                name == "critic" and len(known) != count):
            raise ValueError("saved optimizer state does not fit the extended actor")
        saved["param_groups"][0]["params"] = list(range(count))
        lr = optimizer.param_groups[0]["lr"]
        optimizer.load_state_dict(saved)
        if keep_lr:
            optimizer.param_groups[0]["lr"] = lr


def load_habit_optimizers(trainer: "HistoryTrainer", payload: dict[str, Any]) -> dict[str, Any]:
    """Adam moments of the source checkpoint (new style embedding starts fresh);
    learning rates stay the configured ones. Returns the init record."""
    load_extended_optimizers(trainer, payload, keep_lr=True)
    return dict(path=str(trainer.config.habit_init),
                sha256=sha256(Path(trainer.config.habit_init)),
                lineage=payload.get("lineage"), update=payload["progress"]["updates"],
                decisions=payload["progress"]["decisions"])


def parse_arm_schedule(schedule: str) -> list[tuple[int, bool]]:
    """``"N:on,M:off"`` to ``[(N, True), (M, False)]``; empty gives ``[]``."""
    blocks = []
    for item in [part.strip() for part in schedule.split(",") if part.strip()]:
        count, sep, arm = item.partition(":")
        if not sep or arm not in ("on", "off") or not count.isdigit() or int(count) < 1:
            raise ValueError("batch_snapshot_policies_schedule takes N:on|off blocks, N >= 1")
        blocks.append((int(count), arm == "on"))
    return blocks


# Config fields a resume may change (``--resume-set``): the layout of the batch over
# processes and threads, checkpoint cadence and graph budgets (the per-update batch
# stays num_envs x world size), the diagnostic collection profile (timing only),
# merged snapshot inference and encoding (tier 2: frozen snapshot seats' float-order noise only),
# the allocator cache trim and its slack gate (allocator timing only), the public KV cache storage
# (Triton copies: same attention inputs; paged pool: bitwise on CPU, but SDPA reads
# strided K/V views, so tier 2 on CUDA for learner seats too until the CUDA gate says
# otherwise; private graphs replay the same kernels; page-padded spans, span
# groups, private K/V pages, the paged attention kernel and the batched (or causal)
# learner cache prefill are tier 2), the
# learn-phase profile (timing only),
# the learner's length-grouped
# encode, chosen-only auxiliary response head and compact next-token loss (tier 2:
# same loss, learner float-order noise), the data-parallel gradient transport
# (bitwise with gloo, self-checked), fused Adam (tier 2: FP32 order), plus snapshot_updates,
# which changes dynamics.
RESUME_OVERRIDES = frozenset({"num_envs", "num_threads", "minibatch_matches", "torch_threads",
                              "checkpoint_updates", "ddp_global_minibatch", "ddp_reduce",
                              "adam_fused",
                              "rollout_graph_budget_mb", "rollout_graph_policy_budget_mb",
                              "rollout_triton_min_batch", "snapshot_updates",
                              "profile_collection", "profile_collection_warmup",
                              "batch_snapshot_policies", "batch_snapshot_policies_schedule",
                              "rollout_trim_cuda_cache", "rollout_trim_slack_mb",
                              "rollout_paged_cache",
                              "rollout_triton_cache", "batch_snapshot_encoder",
                              "learner_length_groups", "learner_chosen_response",
                              "learner_compact_next",
                              "rollout_page_span", "rollout_prefill_learner_cache",
                              "rollout_span_groups", "rollout_private_kv_pages",
                              "rollout_causal_prefill", "rollout_paged_attention_kernel",
                              "rollout_merged_head_prefold",
                              "profile_learn",
                              "aux_heads", "aux_coef", "aux_warmup_updates",
                              "rollout_private_graphs",
                              "lr_final", "lr_decay_start", "lr_decay_updates"})


def parse_resume_overrides(items: list[str] | None) -> dict[str, Any]:
    """``KEY=VALUE`` strings to typed HistoryPPOConfig values."""
    types = {f.name: f.type for f in fields(HistoryPPOConfig)}
    values: dict[str, Any] = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        key = key.strip().replace("-", "_")
        if not sep or key not in RESUME_OVERRIDES:
            raise ValueError(f"--resume-set takes KEY=VALUE with KEY in {sorted(RESUME_OVERRIDES)}")
        if "None" in str(types[key]) and "bool" in str(types[key]) and raw.lower() in (
                "auto", "none"):
            values[key] = None
        elif "bool" in str(types[key]):
            if raw.lower() not in ("true", "false", "1", "0"):
                raise ValueError(f"{key} needs true/false")
            values[key] = raw.lower() in ("true", "1")
        elif str(types[key]) == "str":
            values[key] = raw
        elif str(types[key]) == "float":
            values[key] = float(raw)
        else:
            values[key] = int(raw)
    return values


def _joined(parts: list[torch.Tensor]) -> torch.Tensor:
    """``torch.cat(parts)``; a single part as it is (the same values, one copy fewer)."""
    return parts[0] if len(parts) == 1 else torch.cat(parts)


def segment_entropy(log_probs: torch.Tensor, rows: torch.Tensor, count: int) -> torch.Tensor:
    """Entropy of each decision's full candidate distribution, ``[count]``."""
    terms = -(log_probs.exp() * log_probs)
    return torch.zeros(count, device=log_probs.device, dtype=log_probs.dtype).index_add_(
        0, rows, terms)


class HistoryTrainer:
    """Collect, learn, checkpoint. ``update()`` is one PPO iteration."""

    STAT_KEYS = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
                 "ratio_deviation", "actor_grad_norm", "encoder_grad_norm", "critic_grad_norm",
                 "response_loss", "response_accuracy", "response_event_fraction",
                 "behaviour_weight_mean", "behaviour_weight_cap_fraction",
                 "aux_loss", "aux_coefficient", "next_loss", "next_type_accuracy",
                 "next_seat_accuracy", "next_cards_exact", "belief_loss", "belief_baseline",
                 "outcome_loss", "outcome_baseline")
    # (mean, std) of the whole data-parallel minibatch's advantages, else None
    advantage_moments: tuple[float, float] | None = None
    # Profile mode (``profile_learn``): synchronized wall time per learn phase.
    learn_mark = staticmethod(lambda name: None)

    def __init__(self, config: HistoryPPOConfig, output: str | Path, device: str = "cpu",
                 resume: str | Path | None = None, allow_source_change: bool = False,
                 resume_overrides: dict[str, Any] | None = None) -> None:
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if config.torch_threads:
            torch.set_num_threads(config.torch_threads)
        payload: dict[str, Any] | None = None
        extended = False
        if resume is not None:
            actor, critic, payload = load_history_checkpoint(resume, self.device)
            saved_config = HistoryPPOConfig.from_payload(payload["config"], updates=config.updates)
            overrides = dict(resume_overrides or {})
            unknown = set(overrides) - RESUME_OVERRIDES
            if unknown:
                raise ValueError(f"resume cannot change {sorted(unknown)}")
            config = HistoryPPOConfig.from_payload(payload["config"], updates=config.updates,
                                                   **overrides)
            self.lineage = str(payload["lineage"])
            if actor.config != config.policy_config():
                # Only auxiliary heads may be added on resume (extend_actor checks).
                actor = extend_actor(actor, config.policy_config())
                # The critic's architecture does not depend on the heads; it keeps
                # its weights and takes the shared config the checkpoint requires.
                critic.config = actor.config
                extended = True
        elif config.habit_init:
            actor, critic, init_payload = habit_init_player(config)
            self.lineage = f"habit-{config.habit_view}-{config.seed}-{uuid.uuid4().hex[:8]}"
        else:
            actor, critic = fresh_player(config.policy_config(), config.seed)
            self.lineage = f"{STAGE}-{config.seed}-{uuid.uuid4().hex[:8]}"
        self.config = config
        self.rollout_device = torch.device(config.rollout_device or self.device)
        if self.rollout_device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA rollout requested but CUDA is unavailable')
        self.run_identity = dict(engine_digest=engine_digest(), source=source_identity(),
                                 token_schema=TOKEN_SCHEMA_VERSION)
        # A lineage may continue under newer trainer source only on request, and
        # only with the same engine and token schema; the change is recorded.
        self.source_changes = list(payload.get("source_changes", [])) if payload else []
        if payload is not None and payload.get("run_identity") != self.run_identity:
            saved = payload.get("run_identity") or {}
            if not (allow_source_change
                    and engine_compatible(saved)
                    and saved.get("token_schema") == self.run_identity["token_schema"]):
                raise ValueError("resume source/engine/token identity mismatch")
            self.source_changes.append(dict(
                at_update=int(payload["progress"]["updates"]),
                previous_engine_digest=saved.get("engine_digest"),
                engine_digest=self.run_identity["engine_digest"],
                previous_source_sha256=saved.get("source", {}).get("source_sha256"),
                previous_revision=saved.get("source", {}).get("revision"),
                source_sha256=self.run_identity["source"]["source_sha256"],
                revision=self.run_identity["source"]["revision"]))
        # Resume may change how the batch is laid out over processes (and, on
        # request, the snapshot cadence); every change is recorded in the lineage.
        self.config_changes = list(payload.get("config_changes", [])) if payload else []
        if payload is not None:
            changed = {k: [getattr(saved_config, k), getattr(config, k)]
                       for k in sorted(RESUME_OVERRIDES)
                       if getattr(saved_config, k) != getattr(config, k)}
            if changed:
                self.config_changes.append(dict(at_update=int(payload["progress"]["updates"]),
                                                changes=changed))
        # Dropout is zero, so train mode is the same policy as eval mode; staying
        # in train mode keeps the encoder on one kernel path for both the
        # behaviour log-probabilities and the learner's recomputation.
        self.actor = actor.to(self.device).train()
        self.actor.causal_sdpa = config.causal_sdpa
        self.actor.batched_private_attention = config.rollout_batched_attention
        self.actor.wide_private_projection = config.rollout_wide_projection
        self.actor.batched_match_attention = config.learner_batched_attention
        self.critic = critic.to(self.device).train()
        critic_lr = config.critic_lr if config.critic_lr is not None else config.lr
        fused = True if config.adam_fused else None
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr, fused=fused)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr,
                                                 fused=fused)
        self.rng = np.random.default_rng(config.seed)
        self.generator = torch.Generator(device=self.rollout_device)
        self.generator.manual_seed(config.seed)
        self.progress: dict[str, Any] = {"updates": 0, "decisions": 0, "rounds": 0,
                                         "matches": 0, "samples": 0, "learner_rows": 0,
                                         "elapsed_seconds": 0.0}
        if config.habit_pack:
            from train.history_habit import HabitPack
            self.population = HabitPack(self.actor, self.lineage, config.seed + 17,
                                        config.habit_pack, config.habit_axis,
                                        config.habit_strength, self.rollout_device)
        else:
            self.population = HistoryPopulation(self.actor, self.lineage, config.seed + 17,
                                                config.population_recent,
                                                config.snapshot_probability,
                                                config.population_archive_every,
                                                config.population_archive_size,
                                                config.population_archive_share)
        self.habit_init = None
        if payload is None and config.habit_init:
            self.habit_init = load_habit_optimizers(self, init_payload)
        elif payload is not None:
            self.habit_init = payload.get("habit_init")
        if payload is not None:
            if extended:
                load_extended_optimizers(self, payload, keep_lr=False)
            else:
                self.actor_optimizer.load_state_dict(payload["optimizer"]["actor"])
                self.critic_optimizer.load_state_dict(payload["optimizer"]["critic"])
            self.progress.update(payload["progress"])
            if extended or (config.aux_heads and "aux_start_update" not in self.progress):
                self.progress["aux_start_update"] = int(self.progress["updates"])
            rng = payload["rng"]
            if rng.get("sampler_device", "cpu") != self.rollout_device.type:
                raise ValueError("resume requires the same sampler device type")
            if "population" in payload:
                self.population.load_state_dict(payload["population"])
                self.population.prune({})  # resumed environments discard old assignments
            elif config.snapshot_updates:
                raise ValueError("population-enabled resume requires saved population state")
            restore_rng(rng, self.rng)
            self.generator.set_state(rng["sampler"].cpu())
        elif config.aux_heads:
            self.progress["aux_start_update"] = 0
        self.apply_adam_policy()
        self.session_first_update = int(self.progress["updates"])
        env_offset = self.rank_streams()
        self.env = gd.VecEnv(num_envs=config.num_envs, num_threads=config.num_threads,
                             seed=config.seed + 1000 * self.progress["updates"] + env_offset,
                             log_public_actions=True, log_env_limit=config.num_envs)
        if config.habit_round:
            from train.history_habit import RoundEventStore
            self.store = RoundEventStore()
        else:
            self.store = MatchEventStore()
        self.buffer = SequenceRolloutBuffer()
        self.rollout_actor = (self.actor if self.rollout_device == self.device else
                              copy.deepcopy(self.actor).to(self.rollout_device).requires_grad_(False))
        self.rollout_snapshots: dict[int, Any] = {}
        self.collector = self.make_collector()
        self.prior_elapsed = float(self.progress["elapsed_seconds"])
        self.started = time.monotonic()
        self.metrics_path = self.output / "metrics.jsonl"
        self.stop_requested = False
        self.resume_count = int(payload.get("resume_count", 0)) + 1 if payload else 0
        self.population_event(dict(event="resume" if payload else "start",
                                   resume_count=self.resume_count,
                                   discarded_partial_trajectories=bool(payload),
                                   source_change=bool(payload) and bool(self.source_changes)
                                   and self.source_changes[-1]["at_update"] == self.progress["updates"],
                                   cache="ephemeral; rebuilt from raw public histories" if config.rollout_kv_cache
                                         else "none; raw public histories restart with environments"))
        self.write_manifest()

    def rank_streams(self) -> int:
        """Data-parallel hook (train/history_ddp.py), called once by the
        constructor after any restore and before the environments are built or
        anything is written: share the lineage and reseed this rank's streams.
        Returns the environment seed offset; a single process is rank 0."""
        return 0

    def parameters_stepped(self) -> list[torch.Tensor]:
        """Every parameter the two optimizers update."""
        return [p for optimizer in (self.actor_optimizer, self.critic_optimizer)
                for group in optimizer.param_groups for p in group["params"]]

    def apply_adam_policy(self) -> None:
        """Make both optimizers follow ``config.adam_fused``, whatever state they loaded.

        ``Optimizer.load_state_dict`` adopts the saved param-group settings, so a
        checkpoint written under the other setting would silently switch the
        implementation; this restores the configured one (both directions load).
        Adam's step counters are exact integers in float32 and move without a
        value change: to the parameter's device for fused (as fused Adam keeps
        them), to the CPU otherwise, where Adam creates them. A checkpoint loaded
        with ``map_location=cuda`` would otherwise leave them on the GPU, and the
        foreach path reads every counter with ``.item()`` at every step.
        """
        fused = bool(self.config.adam_fused)
        for optimizer in (self.actor_optimizer, self.critic_optimizer):
            for group in optimizer.param_groups:
                group["fused"] = True if fused else None
                if fused:
                    group["foreach"] = None
                on_device = fused or bool(group.get("capturable"))
                for p in group["params"]:
                    state = optimizer.state.get(p)
                    if state and torch.is_tensor(state.get("step")):
                        state["step"] = state["step"].to(p.device if on_device else "cpu")

    def make_collector(self) -> HistoryCollector:
        """The rollout collector over this trainer's env, store and buffer."""
        config = self.config
        return HistoryCollector(self.env, self.rollout_actor, self.store, self.buffer,
                                self.generator, self.rollout_device,
                                seat_policy=self.population.assignment,
                                resolve_policy=self.resolve_rollout_policy,
                                assignment_log=self.population_event,
                                kv_cache=config.rollout_kv_cache,
                                private_graphs=config.rollout_private_graphs,
                                private_graph_budget_mb=config.rollout_graph_budget_mb,
                                private_graph_policy_budget_mb=config.rollout_graph_policy_budget_mb,
                                triton_cache=config.rollout_triton_cache,
                                triton_min_batch=config.rollout_triton_min_batch,
                                profile=self.collection_profiled(),
                                temperature=config.rollout_temperature,
                                epsilon=config.rollout_epsilon,
                                batch_snapshot_policies=self.snapshot_batching(),
                                paged_cache=config.rollout_paged_cache,
                                batch_snapshot_encoder=config.batch_snapshot_encoder,
                                page_span=config.rollout_page_span,
                                round_streams=config.habit_round,
                                row_style=(self.population.style
                                           if self.actor.config.style_input else None),
                                prefill_learner_cache=config.rollout_prefill_learner_cache,
                                span_groups=config.rollout_span_groups,
                                private_kv_pages=config.rollout_private_kv_pages,
                                causal_prefill=config.rollout_causal_prefill,
                                paged_attention_kernel=config.rollout_paged_attention_kernel,
                                merged_head_prefold=config.rollout_merged_head_prefold)

    def snapshot_batching(self) -> bool:
        """Merged snapshot inference for this update (the schedule's block, if any)."""
        blocks = parse_arm_schedule(self.config.batch_snapshot_policies_schedule)
        if not blocks:
            return self.config.batch_snapshot_policies
        done = int(self.progress["updates"]) - self.session_first_update
        for count, arm in blocks:
            if done < count:
                return arm
            done -= count
        return blocks[-1][1]

    def cuda_cache_trimmed(self) -> bool:
        """Trim the allocator cache in this update (explicit, or the merged arm)."""
        if self.config.rollout_trim_cuda_cache is None:
            return self.snapshot_batching()
        return bool(self.config.rollout_trim_cuda_cache)

    def trim_cuda_cache(self, point: str, trims: dict[str, dict], slack_mb: int = 0) -> None:
        """``torch.cuda.empty_cache()`` with reserved/allocated bytes before and
        after and the seconds it took, recorded under ``trims[point]``. Only free
        cached blocks are returned to the driver; live tensors, KV caches and
        captured graphs are untouched, so nothing computed changes. A no-op
        (zero bytes) without CUDA. With ``slack_mb`` > 0 the trim is skipped
        while the cached free bytes it could release are at most that many MiB:
        reserved - allocated, less the free fragments of partly used segments
        (``inactive_split_bytes``), which empty_cache cannot return. The record
        then has ``skipped=True`` and after = before readings."""
        devices = [d for d in (self.rollout_device, self.device) if d.type == "cuda"]
        begin = time.perf_counter()
        record = dict(reserved_before=0, allocated_before=0, reserved_after=0,
                      allocated_after=0)
        if devices:
            device = devices[0]
            torch.cuda.synchronize(device)
            record.update(reserved_before=torch.cuda.memory_reserved(device),
                          allocated_before=torch.cuda.memory_allocated(device))
            releasable = 0
            if slack_mb > 0:
                split = torch.cuda.memory_stats(device).get("inactive_split_bytes.all.current", 0)
                releasable = max(record["reserved_before"] - record["allocated_before"]
                                 - int(split), 0)
                record["releasable_before"] = releasable
            if slack_mb > 0 and releasable <= slack_mb << 20:
                record.update(skipped=True, reserved_after=record["reserved_before"],
                              allocated_after=record["allocated_before"])
            else:
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                record.update(reserved_after=torch.cuda.memory_reserved(device),
                              allocated_after=torch.cuda.memory_allocated(device))
        record["seconds"] = time.perf_counter() - begin
        trims[point] = record

    @contextmanager
    def torch_profile(self, phase: str):
        """Diagnostic, off by default: with ``GUANZERO_TORCH_PROFILE_DIR`` set, the
        updates of this process listed in ``GUANZERO_TORCH_PROFILE_UPDATES`` (comma
        separated, 0 = the first update after start or resume) run ``phase`` under
        ``torch.profiler`` and write per-operator and per-kernel totals to
        ``<dir>/rank-<r>-update-<n>-<phase>.json``. Nothing trained or sampled
        changes; that update's timings include the profiler's overhead."""
        directory = os.environ.get(TORCH_PROFILE_ENV)
        index = int(self.progress["updates"]) - self.session_first_update
        wanted = {int(item) for item in
                  os.environ.get(TORCH_PROFILE_UPDATES_ENV, "").split(",") if item.strip()}
        if not directory or index not in wanted:
            yield
            return
        from torch.profiler import ProfilerActivity, profile
        cuda = "cuda" in (self.device.type, self.rollout_device.type)
        path = Path(directory) / (f"rank-{getattr(self, 'rank', 0)}-update-{index}-{phase}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(phase=phase, update=int(self.progress["updates"]), process_update=index,
                      cuda=cuda, torch=torch.__version__)

        def failed(stage: str, error: Exception) -> None:
            # A diagnostic must not end the run: record why there is no profile.
            path.write_text(json.dumps(dict(record, events=[], error=(
                f"{stage}: {type(error).__name__}: {error}")[:500]), indent=1) + "\n")

        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
        try:
            if cuda:
                torch.cuda.synchronize()
            profiler = profile(activities=activities)
            profiler.__enter__()
        except Exception as error:
            failed("start", error)
            yield
            return
        begin = time.perf_counter()
        try:
            yield
        finally:
            try:
                if cuda:
                    torch.cuda.synchronize()
                seconds = time.perf_counter() - begin
                profiler.__exit__(None, None, None)
                pick = lambda event, *names: next((getattr(event, name) for name in names
                                                   if hasattr(event, name)), 0)
                events = [dict(name=event.key, count=event.count,
                               device=str(getattr(event, "device_type", "")).rsplit(".", 1)[-1],
                               self_cpu_us=event.self_cpu_time_total,
                               cpu_us=event.cpu_time_total,
                               self_device_us=pick(event, "self_device_time_total",
                                                   "self_cuda_time_total"),
                               device_us=pick(event, "device_time_total", "cuda_time_total"))
                          for event in profiler.key_averages()]
                events.sort(key=lambda row: -(row["self_device_us"] or row["self_cpu_us"]))
                path.write_text(json.dumps(dict(
                    record, wall_seconds=seconds,
                    self_device_seconds=sum(row["self_device_us"] for row in events) / 1e6,
                    self_cpu_seconds=sum(row["self_cpu_us"] for row in events) / 1e6,
                    events=events), indent=1) + "\n")
            except Exception as error:
                failed("finish", error)

    def collection_profiled(self) -> bool:
        """Profile this update's collection: on, after this process's warmup updates."""
        return (self.config.profile_collection and int(self.progress["updates"])
                - self.session_first_update >= self.config.profile_collection_warmup)

    def population_event(self, event: dict) -> None:
        with (self.output / "population.jsonl").open("a") as stream:
            stream.write(json.dumps({"lineage": self.lineage,
                                     "rollout_session": getattr(self, "resume_count", 0),
                                     **event}) + "\n")

    # -- manifest ---------------------------------------------------------------

    def write_manifest(self) -> None:
        manifest = {
            "stage": STAGE, "lineage": self.lineage, "init": "random", "teacher": None,
            "config": asdict(self.config), "model_config": config_record(self.actor.config),
            "habit": self.habit_record(),
            "engine_digest": self.run_identity["engine_digest"],
            "source": self.run_identity["source"],
            "source_changes": self.source_changes,
            "config_changes": self.config_changes,
            "engine_binary_sha256": sha256(Path(gd._gd_core.__file__)),
            "token_schema": {"version": TOKEN_SCHEMA_VERSION, "dim": int(TOKEN_DIM),
                             "forced_bit": False, "private_tribute_flags": False},
            "reward": REWARD_SEMANTICS,
            "response_prediction": {
                "schema": RESPONSE_SCHEMA, "mode": self.config.response_mode,
                "coefficient": self.config.response_coef,
                "target_source": "executed actions and future public events in own-lineage self-play",
                "policy_input": "detached predicted probabilities only in explicit mode",
                "target_seat_or_future_events_in_actor_input": False,
            },
            "auxiliary_heads": {
                "heads": list(self.actor.aux_head_names), "coefficient": self.config.aux_coef,
                "warmup_updates": self.config.aux_warmup_updates,
                "start_update": self.progress.get("aux_start_update"),
                "targets": "next public token at every stream position; hidden-hand counts "
                           "(critic storage); the acting team's round return",
                "policy_input": "none: heads read the encoder/decision state only",
            },
            "candidates": "full canonical set in engine order; every candidate selectable",
            "seats": "match-pinned current/own-lineage snapshot seats; learner rows only",
            "population": {"snapshot_updates": self.config.snapshot_updates,
                           "recent": self.config.population_recent,
                           "snapshot_probability": self.config.snapshot_probability,
                           "archive_every": self.config.population_archive_every,
                           "archive_size": self.config.population_archive_size,
                           "archive_share": self.config.population_archive_share,
                           "guaranteed_learner_seats": 1},
            "inference": {**runtime_settings(self.rollout_device),
                          "causal_sdpa": self.rollout_actor.causal_sdpa,
                          "rollout_kv_cache": self.collector.kv_cache,
                          "rollout_batched_attention": self.rollout_actor.batched_private_attention,
                          "rollout_wide_projection": self.rollout_actor.wide_private_projection,
                          "rollout_private_graphs": self.collector.private_graphs,
                          "learner_batched_attention": self.actor.batched_match_attention,
                          "learner_length_groups": self.config.learner_length_groups,
                          "learner_chosen_response": self.config.learner_chosen_response,
                          "learner_compact_next": self.config.learner_compact_next,
                          **self.reduce_record(),
                          "adam_fused": self.config.adam_fused,
                          "rollout_graph_budget_mb": self.config.rollout_graph_budget_mb,
                          "rollout_graph_policy_budget_mb": self.config.rollout_graph_policy_budget_mb,
                          "rollout_triton_cache": self.collector.triton_cache,
                          "rollout_triton_min_batch": self.collector.triton_min_batch,
                          "rollout_paged_cache": self.collector.paged_cache,
                          "batch_snapshot_encoder": self.collector.batch_snapshot_encoder,
                          "rollout_page_span": self.collector.page_span,
                          "rollout_prefill_learner_cache":
                              self.collector.prefill_learner_cache,
                          "rollout_span_groups": self.collector.span_groups,
                          "rollout_private_kv_pages": self.collector.private_kv_pages,
                          "rollout_causal_prefill": self.collector.causal_prefill,
                          "rollout_paged_attention_kernel":
                              self.collector.paged_attention_kernel,
                          "rollout_paged_attention_kernel_active":
                              self.collector.paged_attention_kernel_active(),
                          "rollout_merged_head_prefold": self.collector.merged_head_prefold,
                          "batch_snapshot_policies": self.config.batch_snapshot_policies,
                          "batch_snapshot_policies_schedule":
                              self.config.batch_snapshot_policies_schedule,
                          "rollout_trim_cuda_cache": self.config.rollout_trim_cuda_cache,
                          "rollout_trim_slack_mb": self.config.rollout_trim_slack_mb,
                          "cuda_cache_trim": "torch.cuda.empty_cache() after collect and after "
                                             "learn (None: with the merged-snapshot arm), "
                                             "skipped while reserved - allocated <= "
                                             "rollout_trim_slack_mb MiB (0: never skipped); "
                                             "allocator timing only",
                          "snapshot_batching": "merged head over stacked snapshot weights; "
                                               "per-identity public KV cache; snapshot seats "
                                               "bypass private graphs",
                          "reuse_cache_lengths": self.collector.reuse_cache_lengths,
                          "cache_boundary": "public-only; separate policy/env/match; learner invalidated before learn"},
            "resume": "restore optimizer/RNG/population; discard partial rounds and assignments; "
                      "restart environments and public histories; no persistent cache",
            "parameters": {"actor": count_parameters(self.actor),
                           "critic": count_parameters(self.critic)},
            "torch": torch.__version__, "device": str(self.device),
            "rollout_device": str(self.rollout_device),
            "weight_transfer": "before each collection; included in collection timing",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def reduce_record(self) -> dict[str, Any]:
        """Manifest fields of the gradient all-reduce (requested transport here;
        data-parallel training adds the active one)."""
        return {"ddp_reduce": self.config.ddp_reduce}

    # -- one iteration ----------------------------------------------------------

    def resolve_rollout_policy(self, identity: int):
        if self.rollout_device == self.device:
            return self.population.resolve(identity)
        if identity == 0:
            return self.rollout_actor
        if identity not in self.rollout_snapshots:
            self.rollout_snapshots[identity] = copy.deepcopy(
                self.population.resolve(identity)).to(self.rollout_device)
        return self.rollout_snapshots[identity]

    def collect(self):
        if self.rollout_actor is not self.actor:
            # Keep the learner and sampling weights identical. load_state_dict
            # changes parameter versions, invalidating the public KV cache.
            self.rollout_actor.load_state_dict({k: v.detach().to(self.rollout_device)
                                               for k, v in self.actor.state_dict().items()})
            for identity in list(self.rollout_snapshots):
                if identity not in self.population.models:
                    del self.rollout_snapshots[identity]
        self.collector.policy_decisions.clear()
        with self.collector.frozen_weights():
            return self.collector.collect(self.config.steps_per_update,
                                          version=self.progress["updates"])

    @torch.no_grad()
    def refresh_values(self, chunk: int = 4096) -> np.ndarray:
        """Current critic value of every stored row (completed or not).

        Evaluated in the same ``chunk``-row calls as always; the outputs are
        joined on the device and downloaded with one copy (one sync instead of
        one per chunk), the same float32 values bit for bit."""
        data = self.buffer.compact()
        n = len(self.buffer)
        if n == 0:
            return np.zeros(0, np.float32)
        parts = []
        for begin in range(0, n, chunk):
            obs = torch.as_tensor(data["obs"][begin:begin + chunk], device=self.device)
            hidden = torch.as_tensor(data["hidden"][begin:begin + chunk], device=self.device)
            parts.append(self.critic(obs, hidden).float())
        values = parts[0] if len(parts) == 1 else torch.cat(parts)
        return values.cpu().numpy()

    def recompute_log_probs(self, rows: np.ndarray, streams=None
                            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Current-actor log-probability of each stored row's chosen candidate
        and the entropy over its full candidate set, from raw tokens."""
        if self.config.habit_pack:
            batch = self.buffer.training_batch(rows, self.store, self.device, streams, fields=(),
                                               extra=self.habit_extra(rows),
                                               round_streams=self.config.habit_round)
            inputs, chosen = batch.inputs, batch.chosen
            inputs.style = batch.fields.get("style")
        else:
            inputs, chosen = self.buffer.decision_inputs(rows, self.store, self.device, streams)
        log_probs = self.actor.candidate_log_probs(inputs)
        return log_probs[chosen], segment_entropy(log_probs, inputs.rows, inputs.decisions)

    def habit_extra(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        """Per-row opponents' style for a style-input (ORACLE) learner."""
        if not self.actor.config.style_input:
            return {}
        data = self.buffer.compact()
        return {"style": np.asarray([self.population.style(e, m) for e, m in
                                     zip(data["env"][rows].tolist(), data["match"][rows].tolist())],
                                    np.int64)}

    def minibatches(self) -> Iterator[np.ndarray]:
        return self.buffer.minibatches(self.config.minibatch_matches, self.rng)

    def prepare_minibatch(self, rows: np.ndarray) -> TrainingBatch:
        """The minibatch's host collation and upload: response labels, outcome
        targets and style extras, then ``training_batch``'s packed copy. Depends
        only on ``rows`` and the buffer/store, which ``learn`` never changes, so
        ``learn`` prepares the next minibatch while this one's backward runs."""
        cfg = self.config
        extra = ({"response_target": opponent_response_labels(
            self.buffer, self.store, rows, round_streams=cfg.habit_round)}
                 if cfg.response_mode != "none" else {})
        if "outcome" in self.actor.aux_head_names:
            extra["outcome_target"] = round_outcomes(self.buffer, rows)
        extra.update(self.habit_extra(rows))
        # Actor inputs, stored row data and response targets: one packed upload.
        return self.buffer.training_batch(rows, self.store, self.device, extra=extra or None,
                                          length_groups=cfg.learner_length_groups,
                                          width=cfg.width, round_streams=cfg.habit_round)

    def minibatch_loss(self, rows: np.ndarray, batch: TrainingBatch | None = None
                       ) -> dict[str, torch.Tensor]:
        """Loss terms of ``rows``; ``batch`` is their ``prepare_minibatch`` (made
        here when not given)."""
        cfg = self.config
        predict = cfg.response_mode != "none"
        if batch is None:
            batch = self.prepare_minibatch(rows)
            self.learn_mark("batch_and_upload")
        inputs, chosen, row = batch.inputs, batch.chosen, batch.fields
        inputs.style = row.get("style")
        # Nothing below creates a device tensor from Python data (torch.tensor,
        # new_tensor, as_tensor) or reads one back (.item(), float()): on CUDA
        # either is a stream synchronization that stalls this minibatch's kernel
        # queue (tests/test_history_learner_ops.py audits it).
        response_stats = {}
        aux_stats = {}
        # Per-row 0/1 diagnostics, reduced together at the end (see below).
        indicators: list[torch.Tensor] = []
        heads = self.actor.aux_head_names
        candidate_rows = inputs.rows                # host-planned by training_batch
        if not predict and not heads:
            all_log_probs = self.actor.candidate_log_probs(inputs)
        else:
            from train.history_model import segment_log_softmax
            encoded_streams: list = []
            state = self.actor.decision_states(None, inputs, encoded_streams if heads else None)
            chosen_only = cfg.learner_chosen_response
            logits, response = self.actor.candidate_outputs(
                state, inputs.cand, inputs.offsets, predict=predict,
                response_rows=chosen if (predict and chosen_only) else None, rows=candidate_rows)
            all_log_probs = segment_log_softmax(logits, candidate_rows, inputs.decisions)
            if predict:
                targets = row["response_target"]
                # Outcomes exist ONLY for executed actions.
                prediction = response if chosen_only else response[chosen]
                response_stats = {"response_loss": F.cross_entropy(prediction, targets)}
                indicators += [prediction.argmax(-1) == targets, targets != 0]
            if heads:
                aux_stats = self.auxiliary_losses(state, inputs, row, encoded_streams)
        log_prob = all_log_probs[chosen]
        entropy = segment_entropy(all_log_probs, candidate_rows, inputs.decisions).mean()
        old, behaviour = row["logp"], row["behaviour_logp"]
        advantage, returns = row["advantage"], row["returns"]
        log_ratio = log_prob - old
        ratio = log_ratio.exp()
        if self.advantage_moments is not None:   # data parallel: the cross-rank minibatch
            advantage = (advantage - self.advantage_moments[0]) / (self.advantage_moments[1] + 1e-8)
        elif cfg.normalize_advantages and len(advantage) > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        # Exploration floor: rows were drawn from pi_b, so each advantage carries
        # the truncated per-action weight min(cap, pi_old / pi_b). It is applied
        # AFTER normalization, so the minibatch statistics stay those of the raw
        # GAE advantages and the weight only rescales rows. With temperature 1
        # and epsilon 0, logp == behaviour_logp bitwise, the weight is exactly
        # 1.0 and this multiplication leaves the loss unchanged bit for bit.
        raw_weight = (old - behaviour).exp()
        weight = raw_weight.clamp(max=cfg.behaviour_weight_cap)
        advantage = advantage * weight
        clipped = torch.minimum(ratio * advantage,
                                ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advantage)
        surrogate = -clipped.mean()
        policy_total = surrogate - cfg.entropy * entropy
        if response_stats:
            policy_total = policy_total + cfg.response_coef * response_stats["response_loss"]
        if aux_stats:
            policy_total = policy_total + aux_stats["aux_coefficient"] * aux_stats["aux_loss"]
        value_loss = F.mse_loss(self.critic(inputs.obs, row["hidden"]), returns)
        with torch.no_grad():
            excess = ratio - 1
            deviation = excess.abs()
            approx_kl = (excess - log_ratio).mean()
            ratio_deviation = deviation.max()
            # Means of 0/1 indicators over the same rows: their float sums are
            # exact integers in any order, so one stacked mean gives every rate
            # bit for bit as separate means would.
            rates = torch.stack([deviation > cfg.clip, raw_weight > cfg.behaviour_weight_cap,
                                 *indicators], 1).float().mean(0).unbind()
        clip_fraction, weight_capped = rates[:2]
        if predict:
            response_stats.update(response_accuracy=rates[2], response_event_fraction=rates[3])
        self.learn_mark("forward")
        return {"policy_total": policy_total, "value_total": cfg.value_coef * value_loss,
                "policy_loss": surrogate, "value_loss": value_loss, "entropy": entropy,
                "approx_kl": approx_kl, "clip_fraction": clip_fraction,
                "ratio_deviation": ratio_deviation, "behaviour_weight_mean": weight.mean(),
                "behaviour_weight_cap_fraction": weight_capped, **response_stats, **aux_stats}

    def aux_coefficient(self) -> float:
        """This update's auxiliary-loss weight: ``aux_coef`` ramped linearly over
        ``aux_warmup_updates`` updates from the heads' insertion."""
        cfg = self.config
        if not cfg.aux_warmup_updates:
            return cfg.aux_coef
        since = int(self.progress["updates"]) - int(self.progress.get("aux_start_update", 0)) + 1
        return cfg.aux_coef * min(1.0, since / cfg.aux_warmup_updates)

    def auxiliary_losses(self, state: torch.Tensor, inputs, row: dict[str, torch.Tensor],
                         encoded_streams: list) -> dict[str, torch.Tensor]:
        """The configured heads' losses and diagnostics (train/history_aux.py);
        ``aux_loss`` is their sum and ``aux_coefficient`` this update's weight."""
        heads = self.actor.aux_head_names
        stats: dict[str, torch.Tensor] = {}
        total = state.new_zeros(())
        if "next" in heads:
            if not encoded_streams:
                raise RuntimeError("the next-token head needs the encoded streams")
            if self.config.learner_compact_next and inputs.next_weight is not None:
                stats.update(self.compact_next_token_loss(inputs, encoded_streams))
            else:
                stats.update(next_token_loss(*self.next_token_inputs(inputs, encoded_streams)))
            total = total + stats["next_loss"]
        if "belief" in heads:
            stats.update(belief_loss(self.actor.belief_logits(state), row["hidden"]))
            total = total + stats["belief_loss"]
        if "outcome" in heads:
            stats.update(outcome_loss(self.actor.outcome_value(state), row["outcome_target"]))
            total = total + stats["outcome_loss"]
        stats["aux_loss"] = total
        # A fill with a scalar argument, not ``new_tensor``: a tensor built from
        # Python data is a pageable host-to-device copy, which on CUDA waits for
        # every kernel queued so far (the encoders and heads of this minibatch)
        # before the PPO tail and the backward can be queued. Same FP32 value.
        stats["aux_coefficient"] = torch.full((), self.aux_coefficient(), dtype=state.dtype,
                                              device=state.device)
        return stats

    def next_token_inputs(self, inputs, encoded_streams: list
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(logits, tokens, valid)`` of the next-token head over the encoded
        positions, groups concatenated in plan order. Positions up to each
        match's longest cited prefix count (the tokens some decision of the
        minibatch read), so the loss is the same set of positions whether or not
        the learner encodes in length groups; position s has seen s tokens and
        predicts token s, positions at or past a stream's length are padding.
        With length groups the mask comes planned from ``training_batch``."""
        s = inputs.streams
        planned = inputs.next_weight is not None and encoded_streams[0][1] is not None
        logits, tokens, valid = [], [], []
        for encoded, group in encoded_streams:
            if group is None:
                part, lengths = s.tokens, s.lengths
                cited = torch.zeros_like(lengths).scatter_reduce(
                    0, inputs.match_index, inputs.prefix, reduce="amax")
            else:
                n = group.tokens
                part = s.tokens[group.matches, :n]
                if not planned:
                    lengths = s.lengths[group.matches].clamp(max=n)
                    cited = torch.zeros_like(lengths).scatter_reduce(
                        0, group.local_match, inputs.prefix[group.rows], reduce="amax")
            count = part.shape[1]
            if group is None or not planned:
                limit = torch.minimum(lengths, cited)
                valid.append((torch.arange(count, device=part.device)[None]
                              < limit[:, None]).reshape(-1))
            head = self.actor.next_logits(encoded[:, :count])
            logits.append(head.reshape(-1, head.shape[-1]))
            tokens.append(part.reshape(-1, part.shape[-1]))
        return _joined(logits), _joined(tokens), inputs.next_weight if planned else _joined(valid)

    def compact_next_token_loss(self, inputs, encoded_streams: list) -> dict[str, torch.Tensor]:
        """``learner_compact_next`` (tier 2): the next-token head and its fused
        loss (``next_token_loss_compact``) on the valid positions only, gathered
        once over all length groups. The positions are the planned mask's
        nonzeros at a host-known count, so finding them needs no host read."""
        s = inputs.streams
        hidden, tokens = [], []
        for encoded, group in encoded_streams:
            n = group.tokens
            hidden.append(encoded[:, :n].reshape(-1, encoded.shape[-1]))
            tokens.append(s.tokens[group.matches, :n].reshape(-1, s.tokens.shape[-1]))
        index = torch.nonzero_static(inputs.next_weight, size=inputs.next_positions)[:, 0]
        logits = self.actor.next_logits(_joined(hidden).index_select(0, index))
        return next_token_loss_compact(logits, _joined(tokens).index_select(0, index))

    def learn_summary(self) -> tuple[dict[str, Any], int]:
        """Values, GAE and the update's data statistics; returns ``(stats, samples)``.

        ``stats`` holds every learning key (``None`` until ``learn`` fills it).
        """
        cfg = self.config
        self.collector.invalidate_learner_cache()
        values = self.refresh_values()
        samples = self.buffer.finalize(values, cfg.gamma, cfg.gae_lambda)
        stats: dict[str, Any] = {"update_samples": samples, "minibatches": 0}
        stats.update({key: None for key in self.STAT_KEYS})
        stats.update({"mean_reward": None, "mean_abs_reward": None, "mean_return": None,
                      "explained_variance": None})
        if samples == 0:
            return stats, samples
        rows = self.buffer.samples
        returns = self.buffer.returns[rows]
        variance = float(np.var(returns))
        stats["explained_variance"] = (1.0 - float(np.var(returns - values[rows])) / variance
                                       if variance > 0 else 0.0)
        stats["mean_return"] = float(returns.mean())
        prefixes = self.buffer.compact()["prefix"][rows]
        stats["learn_mean_prefix"] = float(prefixes.mean())
        stats["learn_max_prefix"] = int(prefixes.max())
        terminal = self.buffer.done[rows]
        rewards = self.buffer.reward[rows][terminal]
        # All four seats are the learner, so the signed mean is zero-sum noise;
        # the magnitude is the levels at stake per finished trajectory.
        stats["mean_reward"] = float(rewards.mean()) if rewards.size else None
        stats["mean_abs_reward"] = float(np.abs(rewards).mean()) if rewards.size else None
        return stats, samples

    # Hooks that data-parallel training overrides (train/history_ddp.py).

    def epoch_batches(self, samples: int) -> Iterator[np.ndarray | None]:
        """One epoch's minibatch rows; ``None`` is a step without local data."""
        return self.minibatches()

    def backward_minibatch(self, rows: np.ndarray | None, batch: TrainingBatch | None = None
                           ) -> dict[str, torch.Tensor] | None:
        """Loss terms of ``rows`` (prepared as ``batch`` if given) with their
        gradients accumulated.

        One backward call over policy + value: the actor and critic graphs are
        disjoint and the sum's backward passes the seed gradient to each
        unchanged, so every gradient is bitwise the one of two separate calls.
        """
        terms = self.minibatch_loss(rows, batch)
        (terms["policy_total"] + terms["value_total"]).backward()
        self.learn_mark("backward")
        return terms

    def reduce_gradients(self, real: bool) -> int:
        """Number of minibatches in the gradient about to be applied (0 skips the step)."""
        return 1

    def collective(self) -> bool:
        """Whether learning runs collectives, so every step happens on every rank."""
        return False

    def clip_actor_gradients(self, parameters: list[torch.nn.Parameter],
                             encoder_index: list[int]
                             ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``clip_grad_norm_(parameters, grad_clip)`` and the squared gradient norms
        of the encoder tensors (``parameters[i]`` for ``i`` in ``encoder_index``, in
        that order) from ONE per-tensor norm pass instead of two.

        Bitwise the same as the separate calls: ``clip_grad_norm_`` is
        ``_foreach_norm`` over the present gradients, ``vector_norm`` of their
        stack and ``clip_grads_with_norm_`` (coefficient ``clamp(max_norm /
        (total + 1e-6), max=1)``, one foreach multiply), and a tensor's foreach
        norm does not depend on the other tensors in the call. Returns the total
        norm (a CPU zero without gradients, as ``clip_grad_norm_``) and the
        encoder's squared norms (None without encoder gradients).
        """
        position: dict[int, int] = {}
        grads = []
        for index, p in enumerate(parameters):
            if p.grad is not None:
                position[index] = len(grads)
                grads.append(p.grad)
        if not grads:
            return torch.tensor(0.0), None
        norms = torch._foreach_norm(grads, 2.0)
        total = torch.linalg.vector_norm(torch.stack(norms), 2.0)
        torch.nn.utils.clip_grads_with_norm_(parameters, self.config.grad_clip, total)
        encoder = [norms[position[i]] for i in encoder_index if i in position]
        return total, (torch.stack(encoder) ** 2 if encoder else None)

    def stats_to_host(self, values: list[torch.Tensor],
                      encoder: torch.Tensor | None) -> list[float]:
        """``[*values[:2], *encoder, *values[2:]]`` as host floats in one transfer.

        The values are FP32 scalars on the learner device already; only a CPU
        zero (a module without gradients) or another dtype is converted, as the
        per-value ``.to(device, float32)`` did. Stacking copies, so the floats
        are the same."""
        kind = self.device.type
        ready = [v if v.dtype == torch.float32 and v.device.type == kind
                 else v.to(self.device, torch.float32) for v in values]
        with torch.no_grad():
            parts = [torch.stack(ready[:2])]
            if encoder is not None:
                parts.append(encoder)
            if len(ready) > 2:
                parts.append(torch.stack(ready[2:]))
            return torch.cat(parts).tolist()

    def learn(self) -> dict[str, Any]:
        cfg = self.config
        phases = CollectStats()
        mark = self.learn_mark = _PhaseTimer(phases, self.device, cfg.profile_learn)
        stats, samples = self.learn_summary()
        stats["learn_phase_seconds"] = phases.phase_seconds
        mark("values_and_gae")
        if samples == 0 and not self.collective():
            return stats
        # Loss terms and gradient norms stay on the device until one transfer per
        # minibatch, which also carries the finiteness check before the step.
        # Each value is the same FP32 number the per-term float() reads gave, and
        # the totals add them in the same order in double precision.
        keys = [key for key in self.STAT_KEYS
                if key not in ("actor_grad_norm", "encoder_grad_norm", "critic_grad_norm")]
        totals = {key: 0.0 for key in self.STAT_KEYS}
        count = steps = 0
        actor_parameters = list(self.actor.parameters())
        critic_parameters = list(self.critic.parameters())
        actor_index = {id(p): index for index, p in enumerate(actor_parameters)}
        encoder_index = [actor_index[id(p)] for p in self.actor.stream.parameters()]
        for _ in range(cfg.epochs):
            # The epoch's minibatch rows up front (the same permutation draw).
            batches = list(self.epoch_batches(samples))
            ready = None
            for index, rows in enumerate(batches):
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                mark("minibatch_setup")
                terms = self.backward_minibatch(rows, ready)
                # Collate and upload the next minibatch now, while this backward
                # still runs on the device, instead of after the optimizer step:
                # the same batch from the same rows, only earlier on the host.
                following = batches[index + 1] if index + 1 < len(batches) else None
                ready = None if following is None else self.prepare_minibatch(following)
                if following is not None:
                    mark("batch_and_upload")
                reduced = self.reduce_gradients(terms is not None)
                mark("gradient_reduce")
                if reduced == 0:
                    continue
                # The encoder's per-tensor norms (squared and summed as before)
                # come from the actor clip's own per-tensor norm pass.
                actor, encoder = self.clip_actor_gradients(actor_parameters, encoder_index)
                critic = torch.nn.utils.clip_grad_norm_(critic_parameters, cfg.grad_clip)
                present = [key for key in keys if terms is not None and key in terms]
                host = self.stats_to_host([actor, critic, *(terms[key] for key in present)],
                                          encoder)
                encoders = 0 if encoder is None else len(encoder)
                actor, critic = host[0], host[1]
                if not (math.isfinite(actor) and math.isfinite(critic)):
                    raise FloatingPointError("non-finite gradient; stopping before the step")
                mark("norms_clip_and_stats")
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                if self.config.adam_fused:
                    # The fused kernel updates parameters without bumping their
                    # autograd version counters; caches and graphs that detect
                    # weight changes by version (history_inference._signature)
                    # must still see this step.
                    torch.autograd.graph.increment_version(self.parameters_stepped())
                mark("optimizer")
                steps += 1
                if terms is None:
                    continue
                encoder_total = 0.0
                for value in host[2:2 + encoders]:
                    encoder_total += value
                for key, value in zip(present, host[2 + encoders:]):
                    totals[key] += value
                totals["actor_grad_norm"] += actor
                totals["encoder_grad_norm"] += math.sqrt(encoder_total)
                totals["critic_grad_norm"] += critic
                count += 1
        # Every rank took the same optimizer steps; snapshot decisions use this count.
        stats["minibatches"] = steps
        if self.collective():
            stats["local_minibatches"] = count
        if count:
            for key in self.STAT_KEYS:
                stats[key] = totals[key] / count
        self.progress["samples"] += samples
        return stats

    def lr_scale(self, update: int) -> float:
        """Factor on the configured rates at ``update`` (linear decay, then flat)."""
        config = self.config
        if config.lr_decay_updates <= 0:
            return 1.0
        frac = min(max((update - config.lr_decay_start) / config.lr_decay_updates, 0.0), 1.0)
        return 1.0 + (config.lr_final / config.lr - 1.0) * frac

    def apply_lr_schedule(self) -> float:
        """Set both optimizers' rates for the coming learn; returns the factor."""
        scale = self.lr_scale(int(self.progress["updates"]))
        if self.config.lr_decay_updates > 0:
            critic_lr = (self.config.critic_lr if self.config.critic_lr is not None
                         else self.config.lr)
            for group in self.actor_optimizer.param_groups:
                group["lr"] = self.config.lr * scale
            for group in self.critic_optimizer.param_groups:
                group["lr"] = critic_lr * scale
        return scale

    def update(self) -> dict[str, Any]:
        """Collect, learn, log one metrics line and advance the counters."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        # Profiling only adds synchronized timings and counters; it never changes
        # what is collected, so switching it between updates is safe.
        self.collector.profile = self.collection_profiled()
        # Merged snapshot inference changes only how frozen snapshot seats are
        # evaluated (tier 2), so it may switch between updates as well.
        self.collector.batch_snapshot_policies = self.snapshot_batching()
        # Allocator cache trim (allocator timing only). Points: after collect,
        # so learn (the phase with the highest device readings) starts from the
        # live set rather than on top of collection's freed variable-shape
        # blocks; after learn, so collection does not sit on the learner's
        # activation cache. Switching the merged arm on releases the snapshot
        # private graphs; their pools only return to the driver on a trim, which
        # is therefore never gated. rollout_trim_slack_mb gates the other two.
        trim = self.cuda_cache_trimmed()
        slack_mb = self.config.rollout_trim_slack_mb
        trims: dict[str, dict] = {}
        if trim and self.collector.batch_snapshot_policies:
            if self.collector.release_snapshot_graphs():
                self.trim_cuda_cache("graphs_released", trims)
        t0 = time.perf_counter()
        with self.torch_profile("collect"):
            collected = self.collect()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t1 = time.perf_counter()
        collection_cache = self.collector.cache_metrics()
        if trim:
            self.trim_cuda_cache("after_collect", trims, slack_mb)
        t1_learn = time.perf_counter()
        lr_scale = self.apply_lr_schedule()
        with self.torch_profile("learn"):
            stats = self.learn()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t2 = time.perf_counter()
        if trim:
            self.trim_cuda_cache("after_learn", trims, slack_mb)
        cited = self.buffer.next_iteration()
        pruned = self.store.prune(cited)
        self.progress["updates"] += 1
        self.population.decisions.update(self.collector.policy_decisions)
        if (stats["minibatches"] and self.config.snapshot_updates
                and self.progress["updates"] % self.config.snapshot_updates == 0):
            identity = self.population.snapshot(self.progress["updates"])
            self.population_event(dict(event="snapshot", **self.population.metadata[identity]))
        self.population.prune(self.collector.assignments)
        self.progress["decisions"] += collected.decisions
        self.progress["learner_rows"] += collected.learner_rows
        self.progress["rounds"] += collected.rounds
        self.progress["matches"] += collected.matches
        self.progress["elapsed_seconds"] = (self.prior_elapsed + time.monotonic()
                                            - self.started)
        # Each metric source read once (all host-side reads, no device work).
        cache = self.collector.cache_metrics()
        memory_stats = torch.cuda.memory_stats(self.device) if self.device.type == "cuda" else {}
        line = {
            "update": self.progress["updates"], "decisions": self.progress["decisions"],
            "rounds": self.progress["rounds"], "matches": self.progress["matches"],
            "step_decisions": collected.decisions, "step_learner_rows": collected.learner_rows,
            "step_rounds": collected.rounds, "step_matches": collected.matches,
            "round_gain": collected.gain / collected.rounds if collected.rounds else None,
            "lr_scale": lr_scale,
            "decisions_per_sec": collected.decisions / max(t1 - t0, 1e-9),
            "collect_seconds": t1 - t0, "learn_seconds": t2 - t1_learn,
            "learn_decisions_per_sec": stats["update_samples"] / max(t2 - t1_learn, 1e-9),
            "learn_exposures_per_sec": stats["update_samples"] * self.config.epochs / max(t2 - t1_learn, 1e-9),
            "learner_collect_decisions_per_sec": collected.learner_rows / max(t1 - t0, 1e-9),
            "mean_prefix": collected.mean_prefix, "max_prefix": collected.prefix_max,
            "store_matches": len(self.store), "store_tokens": self.store.tokens,
            "pruned_matches": pruned, "carried_rows": len(self.buffer),
            "elapsed_seconds": self.progress["elapsed_seconds"],
            "population": self.population.metrics(),
            "policy_version_min": int(self.buffer.compact()["version"].min()) if len(self.buffer) else None,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0,
            "cache_bytes": cache["bytes"],
            "cache": cache,
            "collection_cache": collection_cache,
            "private_graphs": self.collector.graph_metrics(),
            "cache_transfers": self.collector.transfer_metrics(),
            "collection_phase_seconds": collected.phase_seconds,
            "collection_profile_synchronized": self.collector.profile,
            "collection_group_phase_seconds": collected.group_phase_seconds,
            "collection_policy_call_rows": {group: {str(size): count for size, count
                                                    in sorted(sizes.items())}
                                            for group, sizes in collected.policy_call_rows.items()},
            "collection_policy_batches": collected.policy_batches,
            "collection_steps": collected.steps,
            "rollout_batch_snapshot_policies": self.collector.batch_snapshot_policies,
            "rollout_trim_cuda_cache": trim,
            "cuda_trim": trims,
            "cuda_trim_seconds": sum(record["seconds"] for record in trims.values()),
            "snapshot_heads": self.collector.snapshot_head_metrics(),
            "rollout_epsilon_pick_fraction": (collected.epsilon_picks / collected.learner_rows
                                              if collected.learner_rows else None),
            "rollout_behaviour_entropy": (collected.behaviour_entropy_sum / collected.learner_rows
                                          if collected.learner_rows else None),
            "cuda_allocated_bytes": torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0,
            "cuda_reserved_bytes": torch.cuda.memory_reserved(self.device) if self.device.type == "cuda" else 0,
            "cuda_inactive_split_peak_bytes": memory_stats.get("inactive_split_bytes.all.peak", 0),
            "cuda_allocation_retries": memory_stats.get("num_alloc_retries", 0),
        }
        line.update(stats)
        self.extend_metrics(line)
        with self.metrics_path.open("a") as stream:
            stream.write(json.dumps(line) + "\n")
        return line

    def habit_record(self) -> dict[str, Any] | None:
        """Planted-habit diagnostic identity: pack, view and initial checkpoint."""
        if not self.config.habit_pack:
            return None
        return dict(diagnostic="planted-habit; never the main lineage",
                    view=self.config.habit_view, pack=self.population.pack,
                    init=self.habit_init,
                    learner_seats="one team (two seats) per match; both other seats the pack",
                    style_input=self.actor.config.style_input)

    def extend_metrics(self, line: dict[str, Any]) -> None:
        """Hook for subclasses to add fields to the metrics line before it is written."""

    # -- checkpoints ------------------------------------------------------------

    def payload(self) -> dict[str, Any]:
        self.progress["elapsed_seconds"] = (self.prior_elapsed + time.monotonic()
                                            - self.started)
        rng = rng_state(self.rng)
        rng["sampler"] = self.generator.get_state()
        rng["sampler_device"] = self.rollout_device.type
        payload = checkpoint_payload(
            self.actor, self.critic, lineage=self.lineage, seed=self.config.seed,
            optimizer={"actor": self.actor_optimizer.state_dict(),
                       "critic": self.critic_optimizer.state_dict()},
            config=asdict(self.config), progress=dict(self.progress), rng=rng)
        if self.config.habit_pack:
            payload.update(habit=self.habit_record(), habit_init=self.habit_init)
        payload.update(population=self.population.state_dict(), run_identity=self.run_identity,
                       resume_count=self.resume_count, source_changes=self.source_changes,
                       config_changes=self.config_changes)
        return payload

    def save(self, path: str | Path | None = None, copies: Sequence[str | Path] = ()) -> Path:
        """Write the checkpoint to ``path`` (default ``latest.pt``) and the same
        bytes, from one payload and one serialization, to each of ``copies``."""
        path = Path(path) if path is not None else self.output / "latest.pt"
        save_history_checkpoint(path, self.payload(), copies)
        return path

    def run(self) -> None:
        while self.progress["updates"] < self.config.updates and not self.stop_requested:
            line = self.update()
            print(json.dumps({k: line[k] for k in ("update", "decisions", "rounds",
                                                    "update_samples", "policy_loss",
                                                    "value_loss", "entropy",
                                                    "decisions_per_sec", "mean_prefix")}),
                  flush=True)
            if self.progress["updates"] % self.config.checkpoint_updates == 0:
                self.save(copies=[self.output / f"update-{self.progress['updates']:06d}.pt"])
        self.save()


# ---- diagnostics ----------------------------------------------------------------

CPROFILE_ENV = "GUANZERO_CPROFILE_DIR"
TORCH_PROFILE_ENV = "GUANZERO_TORCH_PROFILE_DIR"
TORCH_PROFILE_UPDATES_ENV = "GUANZERO_TORCH_PROFILE_UPDATES"
CPROFILE_BUILTINS_ENV = "GUANZERO_CPROFILE_BUILTINS"


def run_profiled(trainer: "HistoryTrainer", rank: int = 0) -> None:
    """``trainer.run()``; with ``GUANZERO_CPROFILE_DIR`` set, under cProfile.

    Diagnostic only and off by default: the profile of this process's main
    thread is written to ``<dir>/rank-<rank>.prof`` (``pstats`` format) when
    ``run`` returns, including after a SIGTERM/SIGINT stop request, or raises.
    It changes nothing that is trained or sampled, but the profiler's per-call
    overhead makes the run's timings unusable for throughput comparisons.

    C functions are not entries by default: their time counts in the calling
    Python function's own time. Under Python 3.12 recording them
    (``GUANZERO_CPROFILE_BUILTINS=1``) drops every enclosing frame of some
    torch C calls (``torch.save``'s zip writer, calls inside collection) from
    the call tree, so ``run``/``update``/``collect`` would be missing.
    """
    directory = os.environ.get(CPROFILE_ENV)
    if not directory:
        trainer.run()
        return
    import cProfile
    path = Path(directory) / f"rank-{rank}.prof"
    path.parent.mkdir(parents=True, exist_ok=True)
    profiler = cProfile.Profile(builtins=os.environ.get(CPROFILE_BUILTINS_ENV) == "1")
    profiler.enable()
    try:
        trainer.run()
    finally:
        profiler.disable()
        profiler.dump_stats(str(path))


# ---- CLI ------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--steps-per-update", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=16)
    parser.add_argument("--response-mode", choices=("none", "auxiliary", "explicit"), default="none")
    parser.add_argument("--response-coef", type=float, default=0.1)
    parser.add_argument("--aux-heads", default="",
                        help="auxiliary heads on the shared encoder, a comma-separated subset "
                             f"of {','.join(AUX_HEADS)} (train/history_aux.py)")
    parser.add_argument("--aux-coef", type=float, default=0.1,
                        help="weight of the summed auxiliary losses in the actor loss")
    parser.add_argument("--aux-warmup-updates", type=int, default=0,
                        help="ramp the auxiliary weight linearly over this many updates "
                             "from the heads' insertion (0: full weight at once)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch-matches", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-epsilon", type=float, default=0.0)
    parser.add_argument("--behaviour-weight-cap", type=float, default=1.0)
    parser.add_argument("--checkpoint-updates", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--snapshot-updates", type=int, default=2)
    parser.add_argument("--population-recent", type=int, default=4)
    parser.add_argument("--snapshot-probability", type=float, default=0.5)
    parser.add_argument("--population-archive-every", type=int, default=0)
    parser.add_argument("--population-archive-size", type=int, default=16)
    parser.add_argument("--population-archive-share", type=float, default=0.5)
    parser.add_argument("--habit-pack", default="",
                        help="planted-habit diagnostic: frozen base checkpoint of the styled "
                             "opponent pack (replaces snapshots; never the main lineage)")
    parser.add_argument("--habit-axis", default="lead_single",
                        choices=["pass", "bomb", "lead_single"])
    parser.add_argument("--habit-strength", type=float, default=1.0)
    parser.add_argument("--habit-view", default="full", choices=["full", "round", "oracle"])
    parser.add_argument("--habit-init", default="",
                        help="planted-habit diagnostic: checkpoint the learner continues from")
    parser.add_argument("--causal-sdpa", action="store_true")
    parser.add_argument("--rollout-kv-cache", action="store_true")
    parser.add_argument("--rollout-batched-attention", action="store_true")
    parser.add_argument("--rollout-wide-projection", action="store_true",
                        help="with --rollout-batched-attention: one q/out projection over all "
                             "rows (FP32; reduction order differs, not bitwise)")
    parser.add_argument("--rollout-private-graphs", action="store_true")
    parser.add_argument("--rollout-page-span", action="store_true",
                        help="with --rollout-paged-cache: pad cache attention and decision "
                             "memory to 64-token pages, not powers of two (FP32; not bitwise)")
    parser.add_argument("--learner-batched-attention", action="store_true",
                        help="learner: all matches of a minibatch in one padded attention "
                             "call (FP32; reduction order differs, not bitwise)")
    parser.add_argument("--learner-length-groups", type=int, default=0,
                        help="learner: encode each minibatch in up to N length groups, each "
                             "stream only as far as its rows read (FP32; not bitwise)")
    parser.add_argument("--learner-chosen-response", action="store_true",
                        help="learner, auxiliary response mode: run the response head on "
                             "executed candidates only, the rows its loss reads (FP32; not bitwise)")
    parser.add_argument("--learner-compact-next", action="store_true",
                        help="learner, with --learner-length-groups and the next-token head: "
                             "head and fused loss on valid positions only (FP32; not bitwise)")
    parser.add_argument("--adam-fused", action="store_true",
                        help="fused Adam kernels for both optimizers (FP32; not bitwise)")
    parser.add_argument("--rollout-graph-budget-mb", type=int, default=512)
    parser.add_argument("--rollout-graph-policy-budget-mb", type=int, default=128,
                        help="per-policy private-graph cap within the total budget")
    parser.add_argument("--rollout-triton-cache", action="store_true")
    parser.add_argument("--rollout-triton-min-batch", type=int, default=1)
    parser.add_argument("--rollout-paged-cache", action="store_true",
                        help="one shared page pool for the public KV cache: fixed operations "
                             "per encode call (bitwise on CPU; replaces --rollout-triton-cache)")
    parser.add_argument("--batch-snapshot-policies", action="store_true",
                        help="all snapshot identities' rows of a vector step in one merged "
                             "actor call (FP32; snapshot seats' reduction order differs)")
    parser.add_argument("--batch-snapshot-encoder", action="store_true",
                        help="with --batch-snapshot-policies and --rollout-paged-cache: encode "
                             "all snapshot identities' public streams in one pass per step "
                             "(FP32; snapshot seats' reduction order differs)")
    parser.add_argument("--rollout-merged-head-prefold", action="store_true",
                        help="with --batch-snapshot-policies: pre-multiply each snapshot "
                             "head's query/key and value/output projections (FP32; snapshot "
                             "seats' reduction order differs)")
    parser.add_argument("--batch-snapshot-policies-schedule", default="",
                        help="diagnostic A/B: N:on|off blocks of this process's updates")
    parser.add_argument("--rollout-trim-cuda-cache", type=optional_bool, default=None,
                        metavar="auto|true|false",
                        help="torch.cuda.empty_cache() after collect and after learn "
                             "(allocator timing only); auto follows the merged-snapshot arm")
    parser.add_argument("--rollout-trim-slack-mb", type=int, default=0,
                        help="with the trim on: skip the after-collect/after-learn trim while "
                             "reserved - allocated CUDA memory is at most this many MiB "
                             "(0: always trim; allocator timing only)")
    parser.add_argument("--rollout-prefill-learner-cache", action="store_true",
                        help="rebuild the learner's public KV cache for all current matches "
                             "at the start of each collection (FP32; learner seats' "
                             "reduction order differs)")
    parser.add_argument("--rollout-span-groups", type=int, default=1,
                        help="with --rollout-paged-cache: attend in up to N end-sorted span "
                             "groups per cache append, learner decision step and merged "
                             "snapshot step, each padded to its own longest stream (FP32; "
                             "not bitwise); 1 is off")
    parser.add_argument("--rollout-private-kv-pages", action="store_true",
                        help="with --rollout-paged-cache and --rollout-batched-attention: store "
                             "the learner's private-attention keys/values per token when "
                             "encoded instead of projecting its memory every step (FP32; "
                             "not bitwise)")
    parser.add_argument("--rollout-causal-prefill", action="store_true",
                        help="with --rollout-paged-cache and --rollout-prefill-learner-cache: "
                             "prefill rebuilt learner streams in one causal pass per length "
                             "group (FP32; learner seats' reduction order differs)")
    parser.add_argument("--rollout-paged-attention-kernel", action="store_true",
                        help="with --rollout-paged-cache: cache appends (and, with "
                             "--rollout-private-kv-pages, learner private attention) attend "
                             "in the Triton paged kernel, reading K/V in place; CUDA only, "
                             "a no-op elsewhere (FP32; not bitwise)")
    parser.add_argument("--profile-collection", action="store_true")
    parser.add_argument("--profile-learn", action="store_true",
                        help="synchronized wall time per learn phase (learn_phase_seconds)")
    parser.add_argument("--profile-collection-warmup", type=int, default=0,
                        help="with --profile-collection: unprofiled updates of this process "
                             "before profiling starts (resident snapshots and histories "
                             "need ~25 updates to reach steady state after a resume)")
    parser.add_argument("--rollout-device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--allow-source-change", action="store_true",
                        help="resume under different trainer source (same engine and token schema)")
    parser.add_argument("--resume-set", action="append", default=[], metavar="KEY=VALUE",
                        help="with --resume: change a batch-layout field (recorded); "
                             "see RESUME_OVERRIDES")
    return parser


def optional_bool(raw: str) -> bool | None:
    """``auto``/``none`` -> None, ``true``/``1`` -> True, ``false``/``0`` -> False."""
    value = raw.lower()
    if value in ("auto", "none"):
        return None
    if value not in ("true", "false", "1", "0"):
        raise argparse.ArgumentTypeError("expected auto, true or false")
    return value in ("true", "1")


def config_from_args(args: argparse.Namespace) -> HistoryPPOConfig:
    names = {f.name for f in fields(HistoryPPOConfig)}
    return HistoryPPOConfig(**{k: v for k, v in vars(args).items() if k in names})


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    trainer = HistoryTrainer(config, args.output, device=args.device, resume=args.resume,
                             allow_source_change=args.allow_source_change,
                             resume_overrides=parse_resume_overrides(args.resume_set))
    def stop(signum, frame):
        trainer.stop_requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    run_profiled(trainer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
