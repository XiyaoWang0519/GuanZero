"""Run configuration and CLI for cold-start history PPO.

Configuration validation and argument parsing use only the standard library.
The policy architecture is materialized lazily when a trainer requests it.
Auxiliary-head and habit names match ``train.history_aux`` and
``train.history_habit`` so this module does not import the training runtime.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import math
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from train.history_model import HistoryPolicyConfig


# Same sets as train.history_aux.AUX_HEADS, train.history_habit and
# train.history_model.PRIVATE_QK_NORMS.
_AUX_HEADS = ("next", "belief", "outcome")
_STYLE_AXES = ("pass", "bomb", "lead_single")
_HABIT_VIEWS = ("full", "round", "oracle")
_PRIVATE_QK_NORMS = ("none", "query", "qk")


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


def _validate_aux_heads(spec: str) -> None:
    """Reject unknown auxiliary heads. Order is canonicalized when the policy is built."""
    names = [part.strip() for part in str(spec).split(",") if part.strip()]
    unknown = set(names) - set(_AUX_HEADS)
    if unknown:
        raise ValueError(f"unknown auxiliary heads {sorted(unknown)}; choose from {_AUX_HEADS}")


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
    # Private cross-attention Q/K normalization (HistoryPolicyConfig.private_qk_norm):
    # a fresh-lineage A/B arm only; a resume cannot change it.
    private_qk_norm: str = "none"
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
    # Release the CUDA caching allocator's free blocks (torch.cuda.empty_cache)
    # after collect and after learn. Allocator timing only; no numeric change.
    # None follows the update's merged-snapshot arm: with the arm on, no snapshot
    # private graphs are captured, so nothing else ever trims and each process's
    # reserved memory only ratchets upward.
    rollout_trim_cuda_cache: bool | None = None
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
    # thin: historical every-second archive shrink. strength_behavior: admit by
    # probe TV distance and prefer newer snapshots (Branch A mature league).
    population_archive_filter: str = "thin"
    population_archive_min_tv: float = 0.05
    population_behavior_probe_rows: int = 32
    population_recent: int = 4
    snapshot_probability: float = 0.5
    # data parallel (train/history_ddp.py): normalize advantages and weight the
    # loss over the union of the ranks' minibatches, as one minibatch would be
    ddp_global_minibatch: bool = False
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
        parse_arm_schedule(self.batch_snapshot_policies_schedule)
        if (self.batch_snapshot_policies or self.batch_snapshot_policies_schedule) and (
                self.window or self.response_mode == "explicit"):
            raise ValueError("batch_snapshot_policies requires full history and no explicit "
                             "response bridge")
        if (self.batch_snapshot_policies or self.batch_snapshot_policies_schedule) and (
                self.private_qk_norm != "none"):
            raise ValueError("batch_snapshot_policies does not implement private Q/K norms")
        if (self.rollout_private_graphs or self.rollout_triton_cache
                or self.rollout_paged_cache) and not self.rollout_kv_cache:
            raise ValueError('KV cache layouts and CUDA rollout optimizations require rollout_kv_cache')
        if self.rollout_paged_cache and self.rollout_triton_cache:
            raise ValueError('rollout_paged_cache replaces rollout_triton_cache; choose one')
        if self.rollout_page_span and (not self.rollout_paged_cache
                                       or self.rollout_private_graphs):
            raise ValueError('rollout_page_span requires rollout_paged_cache and no private graphs')
        if self.rollout_prefill_learner_cache and (not self.rollout_kv_cache
                                                   or self.habit_view == "round"):
            raise ValueError('rollout_prefill_learner_cache requires the match-keyed '
                             'rollout_kv_cache')
        if self.batch_snapshot_encoder and not (
                self.rollout_paged_cache
                and (self.batch_snapshot_policies or self.batch_snapshot_policies_schedule)):
            raise ValueError('batch_snapshot_encoder requires rollout_paged_cache and merged '
                             'snapshot policies')
        if self.rollout_wide_projection and not self.rollout_batched_attention:
            raise ValueError('rollout_wide_projection requires rollout_batched_attention')
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
                or not 0 <= self.population_archive_share <= 1
                or self.population_archive_filter not in ("thin", "strength_behavior")
                or not 0 <= self.population_archive_min_tv <= 1
                or self.population_behavior_probe_rows < 1):
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
        _validate_aux_heads(self.aux_heads)
        if not math.isfinite(self.aux_coef) or self.aux_coef < 0 or self.aux_warmup_updates < 0:
            raise ValueError("aux_coef must be finite and nonnegative; aux_warmup_updates >= 0")
        if self.aux_heads and self.aux_coef == 0:
            raise ValueError("auxiliary heads require a positive aux_coef")
        if self.habit_pack:
            if self.habit_axis not in _STYLE_AXES or self.habit_view not in _HABIT_VIEWS:
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
        from train.history_model import HistoryPolicyConfig
        return HistoryPolicyConfig(width=self.width, layers=self.layers, heads=self.heads,
                                   window=self.window, max_rounds=self.max_rounds,
                                   response_mode=self.response_mode,
                                   aux_heads=self.aux_heads,
                                   private_qk_norm=self.private_qk_norm,
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
# the allocator cache trim (allocator timing only), the public KV cache storage
# (Triton copies: same attention inputs; paged pool: bitwise on CPU, but SDPA reads
# strided K/V views, so tier 2 on CUDA for learner seats too until the CUDA gate says
# otherwise; private graphs replay the same kernels; page-padded spans and the
# batched learner cache prefill are tier 2), the learn-phase profile (timing only),
# the learner's length-grouped
# encode and chosen-only auxiliary response head (tier 2: same loss, learner
# float-order noise), plus snapshot_updates,
# which changes dynamics.
RESUME_OVERRIDES = frozenset({"num_envs", "num_threads", "minibatch_matches", "torch_threads",
                              "checkpoint_updates", "ddp_global_minibatch",
                              "rollout_graph_budget_mb", "rollout_graph_policy_budget_mb",
                              "rollout_triton_min_batch", "snapshot_updates",
                              "profile_collection", "profile_collection_warmup",
                              "batch_snapshot_policies", "batch_snapshot_policies_schedule",
                              "rollout_trim_cuda_cache", "rollout_paged_cache",
                              "rollout_triton_cache", "batch_snapshot_encoder",
                              "learner_length_groups", "learner_chosen_response",
                              "rollout_page_span", "rollout_prefill_learner_cache",
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


# ---- CLI ------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cold-start PPO for the history Transformer player "
                    "(STAGE_C T2, DESIGN v0.6 8).")
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
                             f"of {','.join(_AUX_HEADS)} (train/history_aux.py)")
    parser.add_argument("--aux-coef", type=float, default=0.1,
                        help="weight of the summed auxiliary losses in the actor loss")
    parser.add_argument("--aux-warmup-updates", type=int, default=0,
                        help="ramp the auxiliary weight linearly over this many updates "
                             "from the heads' insertion (0: full weight at once)")
    parser.add_argument("--private-qk-norm", choices=_PRIVATE_QK_NORMS, default="none",
                        help="private cross-attention normalization: LayerNorm the q_proj "
                             "input (query) or q and k per head (qk); fresh lineages only")
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
    parser.add_argument("--batch-snapshot-policies-schedule", default="",
                        help="diagnostic A/B: N:on|off blocks of this process's updates")
    parser.add_argument("--rollout-trim-cuda-cache", type=optional_bool, default=None,
                        metavar="auto|true|false",
                        help="torch.cuda.empty_cache() after collect and after learn "
                             "(allocator timing only); auto follows the merged-snapshot arm")
    parser.add_argument("--rollout-prefill-learner-cache", action="store_true",
                        help="rebuild the learner's public KV cache for all current matches "
                             "at the start of each collection (FP32; learner seats' "
                             "reduction order differs)")
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
