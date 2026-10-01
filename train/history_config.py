"""Run configuration and CLI for cold-start history PPO.

Configuration validation and argument parsing use only the standard library.
The policy architecture is materialized lazily when a trainer requests it.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import math
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from train.history_model import HistoryPolicyConfig


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
    profile_collection: bool = False   # synchronized phase timings; diagnostic only
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
        if (self.rollout_private_graphs or self.rollout_triton_cache
                or self.rollout_paged_cache) and not self.rollout_kv_cache:
            raise ValueError('KV cache layouts and CUDA rollout optimizations require rollout_kv_cache')
        if self.rollout_paged_cache and self.rollout_triton_cache:
            raise ValueError('rollout_paged_cache replaces rollout_triton_cache; choose one')
        if self.rollout_page_span and (not self.rollout_paged_cache
                                       or self.rollout_private_graphs):
            raise ValueError('rollout_page_span requires rollout_paged_cache and no private graphs')
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

    def policy_config(self) -> HistoryPolicyConfig:
        from train.history_model import HistoryPolicyConfig
        return HistoryPolicyConfig(width=self.width, layers=self.layers, heads=self.heads,
                                   window=self.window, max_rounds=self.max_rounds,
                                   response_mode=self.response_mode)

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
# otherwise; private graphs replay the same kernels; page-padded spans are tier 2),
# the learner's length-grouped
# encode (tier 2: same loss, learner float-order noise), plus snapshot_updates,
# which changes dynamics.
RESUME_OVERRIDES = frozenset({"num_envs", "num_threads", "minibatch_matches", "torch_threads",
                              "checkpoint_updates", "ddp_global_minibatch",
                              "rollout_graph_budget_mb", "rollout_graph_policy_budget_mb",
                              "rollout_triton_min_batch", "snapshot_updates",
                              "profile_collection", "profile_collection_warmup",
                              "batch_snapshot_policies", "batch_snapshot_policies_schedule",
                              "rollout_trim_cuda_cache", "rollout_paged_cache",
                              "rollout_triton_cache", "batch_snapshot_encoder",
                              "learner_length_groups", "rollout_page_span",
                              "rollout_private_graphs"})


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
    parser.add_argument("--profile-collection", action="store_true")
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

