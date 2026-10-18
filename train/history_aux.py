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

# The six categorical fields: the five one-hot fields, then the bomb size.
# Card bits are one contiguous block in the token and in the logits.
_CATEGORICAL = (*(name for name, _ in ONE_HOT_FIELDS), "bomb")
_SEAT_FIELD, _TYPE_FIELD = _CATEGORICAL.index("seat"), _CATEGORICAL.index("type")
CARD_TOKENS = slice(CARDS1.start, CARDS2.stop)
CARD_LOGITS = slice(NEXT_SLICES["cards1"].start, NEXT_SLICES["cards2"].stop)
assert CARDS1.stop == CARDS2.start and NEXT_SLICES["cards1"].stop == NEXT_SLICES["cards2"].start
# Compact loss: every categorical field's logits padded to a common width with
# an appended -inf column (index NEXT_LOGITS), so one log-softmax covers all six.
_PAD = max(NEXT_SLICES[name].stop - NEXT_SLICES[name].start for name in _CATEGORICAL)
_LOGIT_LAYOUT = [list(range(NEXT_SLICES[name].start, NEXT_SLICES[name].stop))
                 + [NEXT_LOGITS] * (_PAD - (NEXT_SLICES[name].stop - NEXT_SLICES[name].start))
                 for name in _CATEGORICAL]
_LOGIT_LAYOUT_CACHE: dict[torch.device, Tensor] = {}


def _logit_layout(device: torch.device) -> Tensor:
    """The compact loss's constant column index, uploaded once per device (not per
    call: a tensor from Python data is a host-to-device copy, a stream sync on CUDA)."""
    if device not in _LOGIT_LAYOUT_CACHE:
        with torch.inference_mode(False):
            _LOGIT_LAYOUT_CACHE[device] = torch.tensor(_LOGIT_LAYOUT, dtype=torch.long,
                                                       device=device).reshape(-1)
    return _LOGIT_LAYOUT_CACHE[device]


def next_token_targets(tokens: Tensor) -> list[Tensor]:
    """``[N]`` int64 targets of the categorical fields, in ``_CATEGORICAL``
    order: each one-hot field's argmax (an all-zero field gives 0), then the
    bomb class (``BOMB_CLASSES - 1`` for "not a bomb"). The bomb target fills
    the argmax where no size bit is set: the same integers as a ``where``."""
    targets = [tokens[:, field].argmax(-1) for _, field in ONE_HOT_FIELDS]
    bomb = tokens[:, BOMB]
    targets.append(bomb.argmax(-1).masked_fill_(bomb.amax(-1) == 0, BOMB_CLASSES - 1))
    return targets


def _next_token_rates(logits: Tensor, tokens: Tensor, targets: list[Tensor],
                      weight: Tensor | None, total) -> dict[str, Tensor]:
    """Seat/type accuracy and ``next_cards_exact``. The indicators are 0/1, so
    their weighted sums are exact integers in any order: one stacked column sum
    gives each rate bit for bit as three separate masked means would."""
    with torch.no_grad():
        cards_right = ((logits[:, CARD_LOGITS] > 0) == (tokens[:, CARD_TOKENS] > 0)).all(-1)
        right = torch.stack((logits[:, NEXT_SLICES["seat"]].argmax(-1) == targets[_SEAT_FIELD],
                             logits[:, NEXT_SLICES["type"]].argmax(-1) == targets[_TYPE_FIELD],
                             cards_right), 1).to(logits.dtype)
        if weight is not None:
            right = right * weight[:, None]
        rates = right.sum(0) / total
    seat, kind, cards = rates.unbind()
    return {"next_seat_accuracy": seat, "next_type_accuracy": kind, "next_cards_exact": cards}


def next_token_loss(logits: Tensor, tokens: Tensor, valid: Tensor) -> dict[str, Tensor]:
    """Factorised next-token loss over positions where ``valid`` is true.

    ``logits`` ``[..., NEXT_LOGITS]`` predict ``tokens`` ``[..., TOKEN_DIM]`` at
    the same index (the caller aligns position ``s`` with token ``s``);
    ``valid`` ``[...]`` masks padding (bool, or the same mask as 0/1 floats).
    Returns the field losses' mean (``next_loss``; the mean over the eight
    fields keeps this head at the same order of magnitude as the other heads,
    so ``aux_coef`` weighs them evenly), each field's own mean, and type/seat
    accuracies plus the fraction of positions whose 108 card bits are all
    right (``next_cards_exact``). Means are over valid positions; an
    all-padding batch gives zeros.

    Each field's loss is its own cross-entropy (one-hot fields, bomb size) or
    mean binary cross-entropy (card fields), added in field order; the accuracy
    statistics come from one integer-exact stacked sum, the same values as
    separate masked means.
    """
    logits = logits.reshape(-1, NEXT_LOGITS).float()
    tokens = tokens.reshape(-1, TOKEN_DIM)
    weight = valid.reshape(-1).to(logits.dtype)
    total = weight.sum().clamp_min(1.0)

    def masked_mean(values: Tensor) -> Tensor:
        return (values * weight).sum() / total

    targets = next_token_targets(tokens)
    stats: dict[str, Tensor] = {}
    loss = logits.new_zeros(())
    for (name, _), target in zip(ONE_HOT_FIELDS, targets):
        part = logits[:, NEXT_SLICES[name]]
        stats[f"next_{name}_loss"] = masked_mean(F.cross_entropy(part, target, reduction="none"))
        loss = loss + stats[f"next_{name}_loss"]
    cards = tokens[:, CARD_TOKENS].to(logits.dtype)
    for name, field in MULTI_HOT_FIELDS:
        target = cards[:, field.start - CARD_TOKENS.start:field.stop - CARD_TOKENS.start]
        part = logits[:, NEXT_SLICES[name]]
        stats[f"next_{name}_loss"] = masked_mean(
            F.binary_cross_entropy_with_logits(part, target, reduction="none").mean(-1))
        loss = loss + stats[f"next_{name}_loss"]
    stats["next_bomb_loss"] = masked_mean(
        F.cross_entropy(logits[:, NEXT_SLICES["bomb"]], targets[-1], reduction="none"))
    loss = loss + stats["next_bomb_loss"]
    stats["next_loss"] = loss / NEXT_FIELDS
    stats.update(_next_token_rates(logits, tokens, targets, weight, total))
    return stats


def next_token_loss_compact(logits: Tensor, tokens: Tensor) -> dict[str, Tensor]:
    """``next_token_loss`` over rows that are all valid positions, fused
    (``learner_compact_next``, tier 2): one log-softmax over the six
    categorical fields padded to a common width with ``-inf``, one binary
    cross-entropy over the 108 card bits and one column sum for the eight
    field means. Same statistics, FP32 reduction order differs."""
    logits = logits.reshape(-1, NEXT_LOGITS).float()
    tokens = tokens.reshape(-1, TOKEN_DIM)
    count = len(logits)
    targets = next_token_targets(tokens)
    wide = torch.cat((logits, logits.new_full((count, 1), float("-inf"))), 1)
    log_probs = wide.index_select(1, _logit_layout(logits.device)).view(
        count, len(_CATEGORICAL), _PAD).log_softmax(-1)
    nll = -log_probs.gather(-1, torch.stack(targets, 1)[..., None])[..., 0]  # [N, 6]
    cards = tokens[:, CARD_TOKENS].to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits[:, CARD_LOGITS], cards, reduction="none")
    bce = bce.view(count, len(MULTI_HOT_FIELDS), NUM_CARDS).mean(-1)        # [N, 2]
    total = max(count, 1)
    per_field = torch.cat((nll, bce), 1).sum(0) / total
    names = (*_CATEGORICAL, *(name for name, _ in MULTI_HOT_FIELDS))
    stats = {f"next_{name}_loss": value for name, value in zip(names, per_field.unbind())}
    stats["next_loss"] = per_field.sum() / NEXT_FIELDS
    stats.update(_next_token_rates(logits, tokens, targets, None, total))
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
