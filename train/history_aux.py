"""Auxiliary prediction heads on the history actor's shared encoder (Oct 2026).

Three supervised targets, all free labels of own-lineage self-play, none of
which enters the actor's inputs or scoring path (``aux_heads`` in
``HistoryPolicyConfig``; a head's weights change nothing until its loss is on):

* ``next``: the next public token at EVERY encoded stream position, all seats
  and phases. Position ``s`` of ``encode_stream`` has seen the first ``s``
  tokens and predicts token ``s``, so the labels are the stream itself. The
  target is factorised over the token's fields (one cross-entropy per one-hot
  field, a mean binary cross-entropy per multi-hot field) instead of a flat
  action vocabulary, so the head stays a width x 179 linear map.
* ``belief``: who holds which unseen cards, from the decision state. The
  rollout buffer stores the three hidden hands' per-card counts (relative
  seats +1, +2, +3) for the critic; the head predicts, per card, the seat each
  unseen copy sits in. That per-card seat distribution is exactly what a
  constrained hidden-hand sampler for test-time search needs.
* ``outcome``: the acting team's return of the current round, from the
  decision state (regression; the dense value-like signal DanLM's trunk gets).

Information boundary: labels are only read by the losses below, which the
learner computes from stored rows and public streams; ``HistoryActor`` never
receives a label. Future public events supply ``next`` labels, hidden counts
and round results supply the other two.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from train.logs import TOKEN_DIM

AUX_HEADS = ("next", "belief", "outcome")

# Public token layout (train/logs.py public_token, cpp/include/gd/encoder.h).
SEAT = slice(0, 4)
CARDS1 = slice(4, 58)
CARDS2 = slice(58, 112)
TYPE = slice(112, 125)
KEY = slice(125, 140)
BOMB = slice(140, 147)          # one-hot for bombs, all zero otherwise
WILDS = slice(147, 150)
TRIBUTE_FLAGS = slice(150, 158)  # always zero in the public token
CARDS_LEFT = slice(158, 186)
assert CARDS_LEFT.stop == TOKEN_DIM

# One-hot fields (cross-entropy) and multi-hot fields (mean binary cross-entropy).
ONE_HOT_FIELDS = (("seat", SEAT), ("type", TYPE), ("key", KEY), ("wilds", WILDS),
                  ("cards_left", CARDS_LEFT))
MULTI_HOT_FIELDS = (("cards1", CARDS1), ("cards2", CARDS2))
BOMB_CLASSES = BOMB.stop - BOMB.start + 1      # sizes 4..10 plus "not a bomb"
NEXT_LOGITS = (sum(s.stop - s.start for _, s in ONE_HOT_FIELDS)
               + sum(s.stop - s.start for _, s in MULTI_HOT_FIELDS) + BOMB_CLASSES)
NEXT_FIELDS = len(ONE_HOT_FIELDS) + len(MULTI_HOT_FIELDS) + 1   # the bomb field
HIDDEN_SEATS = 3
NUM_CARDS = 54


def parse_aux_heads(spec: str) -> tuple[str, ...]:
    """``"next,belief"`` -> ``("next", "belief")`` in canonical order; "" -> ()."""
    names = [part.strip() for part in str(spec).split(",") if part.strip()]
    unknown = set(names) - set(AUX_HEADS)
    if unknown:
        raise ValueError(f"unknown auxiliary heads {sorted(unknown)}; choose from {AUX_HEADS}")
    return tuple(name for name in AUX_HEADS if name in names)


def canonical_aux_heads(spec: str) -> str:
    return ",".join(parse_aux_heads(spec))


def _next_slices() -> dict[str, slice]:
    """Logit slices of the next-token head, in layout order."""
    slices, start = {}, 0
    for name, field in (*ONE_HOT_FIELDS, *MULTI_HOT_FIELDS):
        width = field.stop - field.start
        slices[name] = slice(start, start + width)
        start += width
    slices["bomb"] = slice(start, start + BOMB_CLASSES)
    assert start + BOMB_CLASSES == NEXT_LOGITS
    return slices


NEXT_SLICES = _next_slices()


def next_token_loss(logits: Tensor, tokens: Tensor, valid: Tensor) -> dict[str, Tensor]:
    """Factorised next-token loss over positions where ``valid`` is true.

    ``logits`` ``[..., NEXT_LOGITS]`` predict ``tokens`` ``[..., TOKEN_DIM]`` at
    the same index (the caller aligns position ``s`` with token ``s``);
    ``valid`` ``[...]`` masks padding. Returns the field losses' mean
    (``next_loss``; the mean over the eight fields keeps this head at the same
    order of magnitude as the other heads, so ``aux_coef`` weighs them evenly),
    each field's own mean, and type/seat accuracies plus the fraction of
    positions whose 108 card bits are all right (``next_cards_exact``). Means
    are over valid positions; an all-padding batch gives zeros.
    """
    logits = logits.reshape(-1, NEXT_LOGITS).float()
    tokens = tokens.reshape(-1, TOKEN_DIM)
    weight = valid.reshape(-1).to(logits.dtype)
    total = weight.sum().clamp_min(1.0)

    def masked_mean(values: Tensor) -> Tensor:
        return (values * weight).sum() / total

    stats: dict[str, Tensor] = {}
    loss = logits.new_zeros(())
    for name, field in ONE_HOT_FIELDS:
        target = tokens[:, field].argmax(-1)
        part = logits[:, NEXT_SLICES[name]]
        stats[f"next_{name}_loss"] = masked_mean(F.cross_entropy(part, target, reduction="none"))
        loss = loss + stats[f"next_{name}_loss"]
        if name in ("type", "seat"):
            stats[f"next_{name}_accuracy"] = masked_mean((part.argmax(-1) == target).to(logits.dtype))
    cards_right = torch.ones_like(weight, dtype=torch.bool)
    for name, field in MULTI_HOT_FIELDS:
        target = tokens[:, field].to(logits.dtype)
        part = logits[:, NEXT_SLICES[name]]
        stats[f"next_{name}_loss"] = masked_mean(
            F.binary_cross_entropy_with_logits(part, target, reduction="none").mean(-1))
        loss = loss + stats[f"next_{name}_loss"]
        cards_right &= ((part > 0) == (target > 0.5)).all(-1)
    stats["next_cards_exact"] = masked_mean(cards_right.to(logits.dtype))
    bomb = tokens[:, BOMB]
    bomb_target = torch.where(bomb.sum(-1) > 0, bomb.argmax(-1),
                              torch.full_like(bomb[:, 0], BOMB_CLASSES - 1, dtype=torch.long))
    stats["next_bomb_loss"] = masked_mean(
        F.cross_entropy(logits[:, NEXT_SLICES["bomb"]], bomb_target, reduction="none"))
    loss = loss + stats["next_bomb_loss"]
    stats["next_loss"] = loss / NEXT_FIELDS
    return stats


def belief_loss(logits: Tensor, hidden: Tensor) -> dict[str, Tensor]:
    """Per-card seat assignment likelihood of the hidden counts.

    ``logits`` ``[n, 54, 3]``: for each card, one logit per relative seat
    (+1, +2, +3). ``hidden`` ``[n, 3, 54]`` (or ``[n, 162]``): the stored
    counts of that seat's unseen copies of that card. The loss is the mean
    negative log-probability per unseen copy of the seat that actually holds
    it; ``belief_baseline`` is the same quantity for the hand-size proportional
    guess (what uniform determinization implies), so the gap is the head's
    information in nats per card.
    """
    counts = hidden.reshape(len(hidden), HIDDEN_SEATS, NUM_CARDS).to(logits.dtype).transpose(1, 2)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    total = counts.sum().clamp_min(1.0)
    loss = -(counts * log_probs).sum() / total
    with torch.no_grad():
        per_seat = counts.sum(1)                                   # [n, 3]
        share = per_seat / per_seat.sum(-1, keepdim=True).clamp_min(1.0)
        baseline = -(counts * share.clamp_min(1e-12).log()[:, None, :]).sum() / total
    return {"belief_loss": loss, "belief_baseline": baseline}


def outcome_loss(prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
    """Squared error of the predicted round return; ``outcome_baseline`` is the
    minibatch's target variance (the error of predicting the mean)."""
    prediction = prediction.reshape(-1).float()
    target = target.reshape(-1).to(prediction.dtype)
    loss = F.mse_loss(prediction, target)
    with torch.no_grad():
        baseline = target.var(unbiased=False) if len(target) > 1 else target.new_zeros(())
    return {"outcome_loss": loss, "outcome_baseline": baseline}


def round_outcomes(buffer, rows: np.ndarray) -> np.ndarray:
    """The acting team's round return for each stored row (its trajectory's
    reward); rows of unfinished rounds are refused."""
    rows = np.asarray(rows, np.int64)
    traj = buffer.compact()["traj"][rows]
    records = buffer.trajectories
    complete = np.fromiter((t.complete for t in records), bool, len(records))
    if not complete[traj].all():
        raise ValueError("round outcomes require a completed round")
    rewards = np.fromiter((t.reward for t in records), np.float32, len(records))
    return rewards[traj]
