"""Stage B policy: softmax over fusion-head logits on a frozen-M1-pruned set.

DESIGN.md 8.4 item 1. The trainable network is an ordinary `GuandanModel`
initialized from the Stage A (M1) checkpoint; its phase-head scalar divided by
`PolicyConfig.temperature` is the policy logit, so at initialization the policy
is exactly softmax(M1 Q / temperature). Candidates are first pruned to the top
`top_k` by a frozen copy of the Stage A network, and pass is always kept when it
is legal. Tribute and back-tribute decisions use their own phase heads through
the same path; their candidate sets are far below `top_k`, so pruning never
touches them.

Everything is ragged: a batch is a flat candidate tensor plus `offsets`, the
same layout as `GuandanModel.score_candidates` and `select_actions`.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .model import GuandanModel, ModelConfig, select_actions

STAGE = "ppo"
# Action encoding [108, 121) is the type one-hot starting at Type::Pass
# (cpp/include/gd/encoder.h, kActType), so this column marks a pass candidate.
PASS_FEATURE = 108


@dataclass(frozen=True)
class PolicyConfig:
    temperature: float = 1.0
    top_k: int = 32
    chunk_size: int = 32768

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if self.top_k <= 0 or self.chunk_size <= 0:
            raise ValueError("top_k and chunk_size must be positive")


@dataclass
class PolicyStep:
    """One batched decision. Indices are local to each decision's segment."""
    choice: Tensor          # [n] chosen index into the ORIGINAL candidate list
    pruned_choice: Tensor   # [n] chosen index into the pruned candidate list
    log_prob: Tensor        # [n] log-probability of the choice under the policy
    entropy: Tensor         # [n] entropy of the pruned-set distribution
    keep_index: Tensor      # [m] flat indices into `cand` of kept candidates
    pruned_offsets: Tensor  # [n + 1] offsets of the kept candidates
    logits: Tensor          # [m] policy logits of kept candidates
    # [m] log softmax(Q_ref / temperature) over the pruned set, the KL target
    # the learner needs; cached here so the learner never re-runs the reference.
    ref_log_probs: Tensor | None = None
    # Scalar bool tensor, all reference scores finite; set when act() was asked
    # not to synchronize for the check (the caller then checks it later).
    finite: Tensor | None = None


def segment_rows(offsets: Tensor, total: int) -> Tensor:
    n = len(offsets) - 1
    return torch.repeat_interleave(torch.arange(n, device=offsets.device),
                                   offsets[1:] - offsets[:-1], output_size=total)


def pass_mask(cand: Tensor) -> Tensor:
    return cand[:, PASS_FEATURE] > 0.5


def prune_candidates(ref_scores: Tensor, offsets: Tensor, top_k: int,
                     keep: Tensor | None = None, *,
                     check_finite: bool = True) -> tuple[Tensor, Tensor]:
    """Keep the `top_k` best reference scores per segment, plus every `keep`.

    Ties break toward the lower original index. Returns the kept flat indices in
    ascending order (so segments stay contiguous and in original order) and the
    offsets of the pruned ragged batch. A segment with at most `top_k`
    candidates is kept whole. `check_finite=False` skips the (device-syncing)
    finiteness check; the caller must then check the scores itself.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if check_finite and not torch.isfinite(ref_scores).all():
        raise ValueError("reference scores must be finite")
    n, total = len(offsets) - 1, len(ref_scores)
    rows = segment_rows(offsets, total)
    order = torch.argsort(ref_scores, descending=True, stable=True)
    order = order[torch.argsort(rows[order], stable=True)]
    rank = torch.empty_like(order)
    rank[order] = torch.arange(total, device=order.device) - offsets[:-1][rows[order]]
    kept = rank < top_k
    if keep is not None:
        kept |= keep
    keep_index = torch.nonzero(kept).flatten()
    # A segment sum rather than bincount, which synchronises CUDA to size its output.
    counts = offsets.new_zeros(n).index_add_(0, rows, kept.to(offsets.dtype))
    pruned_offsets = torch.cat((offsets.new_zeros(1), torch.cumsum(counts, 0)))
    return keep_index, pruned_offsets


def segment_log_softmax(logits: Tensor, offsets: Tensor) -> Tensor:
    """Log-softmax within each ragged segment; differentiable in `logits`."""
    n = len(offsets) - 1
    rows = segment_rows(offsets, len(logits))
    maxima = logits.new_full((n,), -torch.inf).scatter_reduce(
        0, rows, logits.detach(), reduce="amax", include_self=True)
    shifted = logits - maxima[rows]
    sums = shifted.new_zeros(n).index_add(0, rows, shifted.exp())
    return shifted - sums.log()[rows]


def segment_entropy(log_probs: Tensor, offsets: Tensor) -> Tensor:
    rows = segment_rows(offsets, len(log_probs))
    return -log_probs.new_zeros(len(offsets) - 1).index_add(
        0, rows, log_probs.exp() * log_probs)


def sample_segments(log_probs: Tensor, offsets: Tensor,
                    generator: torch.Generator | None = None, *,
                    validate: bool = True) -> Tensor:
    """Inverse-CDF draw per segment, one uniform per decision, in float64.
    `validate=False` skips the device-syncing empty-segment check for callers
    whose segments are nonempty by construction."""
    sizes = offsets[1:] - offsets[:-1]
    if validate and (sizes <= 0).any():
        raise ValueError("every decision needs at least one candidate")
    n = len(sizes)
    rows = segment_rows(offsets, len(log_probs))
    device = generator.device if generator is not None else log_probs.device
    uniform = torch.rand(n, generator=generator, device=device,
                         dtype=torch.float64).to(log_probs.device)
    cumulative = torch.cumsum(log_probs.detach().double().exp(), 0)
    start = torch.cat((cumulative.new_zeros(1), cumulative))[offsets[:-1]]
    within = cumulative - start[rows]
    # Normalise by the segment total so rounding never leaves mass uncovered.
    total = within[offsets[1:] - 1]
    below = (within <= uniform[rows] * total[rows]).long()
    count = below.new_zeros(n).index_add(0, rows, below)
    return torch.minimum(count, sizes - 1)


def gather_segments(values: Tensor, offsets: Tensor, local: Tensor) -> Tensor:
    return values[offsets[:-1] + local]


class StageBPolicy:
    """Trainable policy network plus the frozen Stage A reference for pruning.

    Only `net` is trainable; build the optimizer from `policy.net.parameters()`.
    The reference is never registered for gradients and is always scored under
    `torch.no_grad()`.
    """

    def __init__(self, net: GuandanModel, reference: GuandanModel,
                 config: PolicyConfig = PolicyConfig(),
                 reference_checkpoint_id: str | None = None) -> None:
        if net.config != reference.config:
            raise ValueError("policy and reference networks must share a ModelConfig")
        self.net = net
        self.reference = reference.eval().requires_grad_(False)
        self.config = config
        self.reference_checkpoint_id = reference_checkpoint_id

    @classmethod
    def from_model(cls, model: GuandanModel, config: PolicyConfig = PolicyConfig(),
                   reference_checkpoint_id: str | None = None) -> "StageBPolicy":
        """Start a policy from a Stage A network; both copies are independent."""
        return cls(copy.deepcopy(model), copy.deepcopy(model), config,
                   reference_checkpoint_id)

    @classmethod
    def from_dmc_checkpoint(cls, path: str | Path, config: PolicyConfig = PolicyConfig(),
                            device: str | torch.device = "cpu") -> "StageBPolicy":
        from eval.policies import model_digest
        from .ckpt import load_checkpoint

        payload = load_checkpoint(path, device=device)
        if payload.get("stage", "dmc") not in ("dmc", "a2"):
            raise ValueError("Stage B starts from a Stage A or A2 checkpoint")
        model = GuandanModel(ModelConfig(**payload["model_config"])).to(device)
        model.load_state_dict(payload["model"])
        return cls.from_model(model, config, model_digest(payload["model"]))

    def to(self, device: str | torch.device) -> "StageBPolicy":
        self.net.to(device)
        self.reference.to(device)
        return self

    def logits(self, obs: Tensor, cand: Tensor, offsets: Tensor, phase: Tensor,
               phase_code: int | None = None, state: Tensor | None = None) -> Tensor:
        """Policy logits for every candidate: phase-head scalar / temperature.
        `state` optionally passes a precomputed `net.state_tower(obs)`."""
        scores = self.net.score_candidates(obs, cand, offsets, phase,
                                           chunk_size=self.config.chunk_size,
                                           phase_code=phase_code, state=state)
        return scores / self.config.temperature

    def prune(self, obs: Tensor, cand: Tensor, offsets: Tensor, phase: Tensor,
              phase_code: int | None = None) -> tuple[Tensor, Tensor]:
        """Frozen-reference top-k plus pass. Returns (keep_index, pruned_offsets)."""
        with torch.no_grad():
            ref_scores = self.reference.score_candidates(
                obs, cand, offsets, phase, chunk_size=self.config.chunk_size,
                phase_code=phase_code)
        return prune_candidates(ref_scores, offsets, self.config.top_k, pass_mask(cand))

    def act(self, obs: Tensor, cand: Tensor, offsets: Tensor, phase: Tensor, *,
            generator: torch.Generator | None = None, greedy: bool = False,
            phase_code: int | None = None, ref_scores: Tensor | None = None,
            sync_checks: bool = True) -> PolicyStep:
        """Prune, then sample (or take the argmax) over the pruned set.

        `ref_scores`, when given, are the frozen reference's scores of `cand`
        (the caller already ran the reference, e.g. fused with an opponent
        that plays the same network); otherwise the reference runs here.
        `sync_checks=False` defers the finiteness check to `PolicyStep.finite`
        and skips checks that hold by construction, so the whole call issues
        no device synchronization except the pruning's `nonzero`.
        """
        if ref_scores is None:
            with torch.no_grad():
                ref_scores = self.reference.score_candidates(
                    obs, cand, offsets, phase, chunk_size=self.config.chunk_size,
                    phase_code=phase_code)
        keep_index, pruned_offsets = prune_candidates(
            ref_scores, offsets, self.config.top_k, pass_mask(cand), check_finite=sync_checks)
        pruned_cand = cand[keep_index]
        logits = self.logits(obs, pruned_cand, pruned_offsets, phase, phase_code)
        log_probs = segment_log_softmax(logits, pruned_offsets)
        if greedy:
            local = select_actions(logits.detach(), pruned_offsets)
        else:
            local = sample_segments(log_probs, pruned_offsets, generator, validate=sync_checks)
        flat = pruned_offsets[:-1] + local
        ref_log_probs = segment_log_softmax(ref_scores[keep_index] / self.config.temperature,
                                            pruned_offsets)
        return PolicyStep(choice=keep_index[flat] - offsets[:-1], pruned_choice=local,
                          log_prob=log_probs[flat],
                          entropy=segment_entropy(log_probs, pruned_offsets),
                          keep_index=keep_index, pruned_offsets=pruned_offsets,
                          logits=logits, ref_log_probs=ref_log_probs,
                          finite=None if sync_checks else torch.isfinite(ref_scores).all())

    def choose(self, obs: Tensor, cand: Tensor, offsets: Tensor, phase: Tensor, *,
               generator: torch.Generator | None = None, greedy: bool = False,
               phase_code: int | None = None,
               ref_scores: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """`act`'s choice alone, for opponents: the same candidate index per
        row from the same draws, without log-probabilities, entropy or
        reference log-probabilities, and without device synchronization
        except the pruning's `nonzero`. Returns (choice, finite), `finite`
        being whether the reference scores were all finite; the caller checks
        it when it copies the choice back."""
        if ref_scores is None:
            with torch.no_grad():
                ref_scores = self.reference.score_candidates(
                    obs, cand, offsets, phase, chunk_size=self.config.chunk_size,
                    phase_code=phase_code)
        keep_index, pruned_offsets = prune_candidates(
            ref_scores, offsets, self.config.top_k, pass_mask(cand), check_finite=False)
        logits = self.logits(obs, cand[keep_index], pruned_offsets, phase, phase_code)
        if greedy:
            local = select_actions(logits, pruned_offsets)
        else:
            local = sample_segments(segment_log_softmax(logits, pruned_offsets), pruned_offsets,
                                    generator, validate=False)
        return (keep_index[pruned_offsets[:-1] + local] - offsets[:-1],
                torch.isfinite(ref_scores).all())

    def evaluate(self, obs: Tensor, pruned_cand: Tensor, pruned_offsets: Tensor,
                 phase: Tensor, pruned_choice: Tensor,
                 phase_code: int | None = None) -> tuple[Tensor, Tensor]:
        """Differentiable (log_prob, entropy) of stored choices, for the learner."""
        log_probs = segment_log_softmax(
            self.logits(obs, pruned_cand, pruned_offsets, phase, phase_code), pruned_offsets)
        return (gather_segments(log_probs, pruned_offsets, pruned_choice),
                segment_entropy(log_probs, pruned_offsets))

    def checkpoint_payload(self, *, optimizer: dict[str, Any], config: dict[str, Any],
                           progress: dict[str, Any], rng: dict[str, Any],
                           tribute_policy: str = "heuristic") -> dict[str, Any]:
        """Fields for `train.ckpt.save_checkpoint`; `eval.policies.load_policy`
        recognises the `stage` marker and restores the frozen reference too."""
        if tribute_policy not in ("heuristic", "learned"):
            raise ValueError("tribute_policy must be heuristic or learned")
        return {"stage": STAGE, "tribute_policy": tribute_policy,
                "model_config": asdict(self.net.config), "model": self.net.state_dict(),
                "reference_model": self.reference.state_dict(),
                "reference_checkpoint_id": self.reference_checkpoint_id,
                "policy_config": asdict(self.config),
                "optimizer": optimizer, "config": config, "progress": progress, "rng": rng}


def policy_from_payload(payload: dict[str, Any],
                        device: str | torch.device = "cpu") -> StageBPolicy:
    """Rebuild a `StageBPolicy` from a loaded Stage B checkpoint payload."""
    from eval.policies import model_digest

    if payload.get("stage") != STAGE:
        raise ValueError("not a Stage B policy checkpoint")
    model_config = ModelConfig(**payload["model_config"])
    net, reference = GuandanModel(model_config), GuandanModel(model_config)
    net.load_state_dict(payload["model"])
    reference.load_state_dict(payload["reference_model"])
    reference_id = payload.get("reference_checkpoint_id")
    if reference_id is not None and model_digest(payload["reference_model"]) != reference_id:
        raise ValueError("reference weights do not match reference_checkpoint_id")
    return StageBPolicy(net, reference, PolicyConfig(**payload["policy_config"]),
                        reference_id).to(device)
