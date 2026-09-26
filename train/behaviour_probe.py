"""Offline behaviour probe: history Transformer versus flat tower on logged play.

Step 2 of ``docs/reports/transformer-inventory-2026-09-25.md``. On a schema-3
collection (``eval/collect_belief.py --candidates``) it trains a policy state
tower by behaviour cloning (BC) of the logged actors' choices, with or without
the public history stream and with or without the next-public-event (NTP)
auxiliary loss, and compares policy heads: candidate scoring, the shape of the
current ``GuandanModel`` head; a fixed vocabulary over abstract actions
(``gd.NUM_ABSTRACT``) under a legal mask; and a structured vocabulary whose
entries are embedded from their type, key, bomb size, pair slot and suit.

The public stream of a match is one sequence: every play, pass, forced pass
and tribute of every round, in order, tagged with its round index. A decision
at prefix ``p`` may read only the first ``p`` tokens. No private information
enters the stream; the acting seat's observation enters as one query. The
engine's forced-pass flag is engine-private legality information (it says the
seat could not beat the trick), so it never enters a token; it is stored only
as metadata (STAGE_C_TODO.md, fixed experiment boundary).

Two facts shape what the numbers can mean:

* A frozen argmax policy is a deterministic function of the observation and
  the candidate set, both of which the flat tower sees. On policy-driven rows
  BC therefore cannot reward history; those rows are a ceiling check. Rows
  driven by styled bots depend on a per-match style vector that is not in
  the observation, so they are the cell where history can pay. Every metric
  is reported per driver for that reason.
* NTP predicts the actor's next non-forced play from public information only
  (the stream position at the decision's prefix). Its control is the same
  model restricted to the last ``window`` tokens, which is roughly what the
  flat observation already carries; full history against a short window is
  the direct test of whether the ordered long history matters.

This is a pipeline check and a head choice, not a strength claim: DESIGN.md
7.4 requires an equal-compute RL comparison for that.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import gd
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from train.logs import CANDIDATE_SCHEMA_VERSION, DRIVER_BOT, DRIVER_POLICY, TOKEN_DIM
from train.model import mlp

VOCAB = int(gd.NUM_ABSTRACT)
PLAY_PHASE = int(gd.Phase.Play)
ACT_DIM = int(gd.ACT_DIM)
# Observation bit that says the acting seat is leading the trick
# (cpp/include/gd/encoder.h kObsTrickLeading).
LEADING_BIT = 967
PASS_ID = 0
HEADS = ("cand", "vocab", "vocab_struct")


# ---- abstract-action vocabulary --------------------------------------------

# Layout of the 393 abstract ids, cpp/include/gd/action.h (RULES.md 11.1).
ABSTRACT_LAYOUT = (
    ("pass", 0, 1), ("single", 1, 15), ("pair", 16, 15), ("triple", 31, 13),
    ("full_house", 44, 13 * 14), ("straight", 226, 10), ("tube", 236, 12), ("plate", 248, 13),
    ("bomb", 261, 13 * 7), ("straight_flush", 352, 10 * 4), ("joker_bomb", 392, 1))
# Column offsets of the action encoding (cpp/include/gd/encoder.h).
ACT_TYPE, ACT_KEY, ACT_BOMB_SIZE = 108, 121, 136
# Type::Pass .. Type::JokerBomb in the encoder's one-hot.
ENGINE_TYPE_INDEX = {"pass": 0, "single": 1, "pair": 2, "triple": 3, "full_house": 4,
                     "straight": 5, "tube": 6, "plate": 7, "bomb": 8, "straight_flush": 9,
                     "joker_bomb": 10}
FEATURE_DIM = 11 + 15 + 7 + 14 + 4


def abstract_features() -> np.ndarray:
    """[VOCAB, FEATURE_DIM] one-hot description of every abstract id.

    Columns: engine type (11), key or window (15), bomb size 4..10 (7), full
    house pair slot (14), straight flush suit (4). The key of a bomb is its
    power and of a straight flush its window, as in the encoder.
    """
    table = np.zeros((VOCAB, FEATURE_DIM), dtype=np.float32)
    for name, start, count in ABSTRACT_LAYOUT:
        for local in range(count):
            row = table[start + local]
            row[ENGINE_TYPE_INDEX[name]] = 1
            if name in ("single", "pair", "triple", "straight", "tube", "plate"):
                row[11 + local] = 1
            elif name == "full_house":
                row[11 + local // 14] = 1
                row[11 + 15 + 7 + local % 14] = 1
            elif name == "bomb":
                row[11 + local // 7] = 1
                row[11 + 15 + local % 7] = 1
            elif name == "straight_flush":
                row[11 + local // 4] = 1
                row[11 + 15 + 7 + 14 + local % 4] = 1
            else:
                row[11] = 1   # pass and joker bomb carry key 0 in the encoder
    return table


# ---- data -----------------------------------------------------------------

@dataclass
class Match:
    """One match's public stream and the logged decisions of every logged seat."""
    group: str
    tokens: np.ndarray          # [T, TOKEN_DIM] uint8
    token_round: np.ndarray     # [T] int64
    token_phase: np.ndarray     # [T] int64
    token_forced: np.ndarray    # [T] int64
    token_abstract: np.ndarray  # [T] int64, -1 outside the play vocabulary
    obs: np.ndarray             # [n, obs_dim] uint8
    seat: np.ndarray            # [n] int64
    driver: np.ndarray          # [n] int64, DRIVER_POLICY or DRIVER_BOT
    prefix: np.ndarray          # [n] int64, tokens visible to the decision
    decision_round: np.ndarray  # [n] int64
    cand: np.ndarray            # [sum_k, ACT_DIM] uint8
    cand_offsets: np.ndarray    # [n + 1] int64
    cand_abstract: np.ndarray   # [sum_k] int64
    choice: np.ndarray          # [n] int64, index into the decision's candidates

    @property
    def decisions(self) -> int:
        return len(self.obs)


def load_matches(directory: Path, *, max_rounds: int | None = None) -> tuple[list[Match], dict]:
    """Load a schema-3 collection and stitch its rounds into whole matches.

    Every match keeps its rounds in order from round 0; a match whose earlier
    rounds are missing is rejected, because its stream would silently start
    mid-way. A quota-truncated tail is fine: every prefix is still exact.
    """
    provenance = json.loads((directory / "provenance.json").read_text())
    if provenance.get("status") != "complete" or not provenance.get("candidates"):
        raise ValueError("a complete collection with candidate sets is required")
    if provenance.get("purpose") != "architecture_probe" or provenance.get("learner_updates") != 0:
        raise ValueError("behaviour probes need a frozen-policy architecture_probe collection")
    paths = sorted(directory.glob("round-*.npz"))
    if max_rounds is not None:
        paths = paths[:max_rounds]
    rounds: dict[tuple[int, int], list[dict]] = {}
    features = abstract_features()
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if int(data["schema_version"]) != CANDIDATE_SCHEMA_VERSION:
                raise ValueError(f"schema {CANDIDATE_SCHEMA_VERSION} round expected: {path}")
            item = {key: data[key].copy() for key in (
                "obs", "seat", "driver", "prefix", "tokens", "cand", "cand_offsets",
                "cand_abstract", "choice", "token_abstract", "token_forced", "token_phase")}
            item["group"] = str(data["group"])
            item["round_index"] = int(data["round_index"])
            key = (int(data["env_id"]), int(data["match_id"]))
        validate_round(item, path, features)
        rounds.setdefault(key, []).append(item)
    matches = []
    for key in sorted(rounds):
        parts = sorted(rounds[key], key=lambda r: r["round_index"])
        indices = [r["round_index"] for r in parts]
        if indices != list(range(len(parts))):
            raise ValueError(f"match {key} is missing rounds: {indices}")
        matches.append(stitch(parts))
    if not matches:
        raise ValueError("no rounds found")
    return matches, provenance


def validate_round(item: dict, path: Path, features: np.ndarray | None = None) -> None:
    tokens, prefix, offsets = item["tokens"], item["prefix"], item["cand_offsets"]
    if tokens.shape[1:] != (TOKEN_DIM,) or item["cand"].shape[1:] != (ACT_DIM,):
        raise ValueError(f"unexpected token or candidate width: {path}")
    if len(prefix) and (np.any(np.diff(prefix) <= 0) or prefix[0] < 0 or prefix[-1] >= len(tokens)):
        raise ValueError(f"decision prefixes must be increasing and inside the round: {path}")
    if len(offsets) != len(prefix) + 1 or offsets[-1] != len(item["cand"]):
        raise ValueError(f"candidate offsets do not cover the decisions: {path}")
    if not np.all(np.isin(item["driver"], (DRIVER_POLICY, DRIVER_BOT))):
        raise ValueError(f"unknown decision driver: {path}")
    for i, p in enumerate(prefix):
        chosen = offsets[i] + item["choice"][i]
        if not offsets[i] <= chosen < offsets[i + 1]:
            raise ValueError(f"choice outside the candidate set: {path}")
        if not np.array_equal(tokens[p, 4:4 + ACT_DIM], item["cand"][chosen]):
            raise ValueError(f"the token at a decision's prefix is not its chosen action: {path}")
        if item["token_abstract"][p] != item["cand_abstract"][chosen]:
            raise ValueError(f"token and candidate abstract ids disagree: {path}")
    ids = item["cand_abstract"]
    if np.any((ids < 0) | (ids >= VOCAB)):
        raise ValueError(f"candidate abstract id outside the play vocabulary: {path}")
    # The vocabulary table must agree with the encoder's own type, key and
    # bomb-size columns on every logged candidate.
    features = abstract_features() if features is None else features
    cand = item["cand"].astype(np.float32)
    columns = np.concatenate((cand[:, ACT_TYPE:ACT_TYPE + 11], cand[:, ACT_KEY:ACT_KEY + 15],
                              cand[:, ACT_BOMB_SIZE:ACT_BOMB_SIZE + 7]), axis=1)
    if not np.array_equal(features[ids][:, :11 + 15 + 7], columns):
        raise ValueError(f"abstract id layout disagrees with the action encoding: {path}")


def stitch(parts: list[dict]) -> Match:
    tokens, rounds_of_token = [], []
    offsets_base, cand_base = 0, 0
    prefix, decision_round, cand_offsets = [], [], [np.zeros(1, dtype=np.int64)]
    for r in parts:
        prefix.append(r["prefix"] + offsets_base)
        decision_round.append(np.full(len(r["prefix"]), r["round_index"], dtype=np.int64))
        cand_offsets.append(r["cand_offsets"][1:] + cand_base)
        tokens.append(r["tokens"])
        rounds_of_token.append(np.full(len(r["tokens"]), r["round_index"], dtype=np.int64))
        offsets_base += len(r["tokens"])
        cand_base += len(r["cand"])
    cat = lambda key, dtype=np.int64: np.concatenate([r[key] for r in parts]).astype(dtype)
    return Match(
        group=parts[0]["group"], tokens=np.concatenate(tokens).astype(np.uint8),
        token_round=np.concatenate(rounds_of_token), token_phase=cat("token_phase"),
        token_forced=cat("token_forced"), token_abstract=cat("token_abstract"),
        obs=cat("obs", np.uint8), seat=cat("seat"), driver=cat("driver"),
        prefix=np.concatenate(prefix), decision_round=np.concatenate(decision_round),
        cand=cat("cand", np.uint8), cand_offsets=np.concatenate(cand_offsets),
        cand_abstract=cat("cand_abstract"), choice=cat("choice"))


def split_matches(matches: list[Match], split_seed: int, *, min_matches: int = 10
                  ) -> dict[str, list[Match]]:
    """Whole-match split by seeded SHA-256: 15% test, 15% validation, rest train."""
    if len(matches) < min_matches:
        raise ValueError(f"the probe needs at least {min_matches} matches, got {len(matches)}")
    ordered = sorted(matches, key=lambda m: hashlib.sha256(
        f"{split_seed}:{m.group}".encode()).digest())
    size = max(1, int(len(ordered) * .15))
    splits = {"test": ordered[:size], "validation": ordered[size:2 * size],
              "train": ordered[2 * size:]}
    if any(not part for part in splits.values()):
        raise ValueError("every split needs at least one match; collect more matches")
    return splits


# ---- models ---------------------------------------------------------------

@dataclass(frozen=True)
class ProbeConfig:
    tower: str = "history"          # history | flat
    head: str = "cand"              # cand | vocab | vocab_struct
    ntp_weight: float = 0.0
    window: int = 0                 # 0 = whole match; k = only the last k tokens
    width: int = 128
    layers: int = 4
    heads: int = 4
    max_rounds: int = 16
    action_width: int = 128
    fusion_width: int = 128
    obs_dim: int = int(gd.OBS_DIM)

    def __post_init__(self) -> None:
        if self.tower not in ("history", "flat") or self.head not in HEADS:
            raise ValueError(f"tower must be history or flat; head must be one of {HEADS}")
        if self.ntp_weight < 0 or (self.tower == "flat" and self.ntp_weight > 0):
            raise ValueError("the NTP loss needs the history tower and a non-negative weight")
        if self.window < 0 or (self.tower == "flat" and self.window):
            raise ValueError("window applies to the history tower and must not be negative")
        if min(self.width, self.layers, self.heads, self.max_rounds,
               self.action_width, self.fusion_width) <= 0 or self.width % self.heads:
            raise ValueError("positive sizes required; width must be divisible by heads")


class Batch:
    """Device tensors for a list of matches, streams padded to the longest."""

    def __init__(self, matches: list[Match], device: str) -> None:
        self.matches = matches
        self.lengths = torch.tensor([len(m.tokens) for m in matches], device=device)
        longest = int(self.lengths.max())
        count = len(matches)

        def padded(key: str, fill: int, dtype: torch.dtype) -> Tensor:
            out = torch.full((count, longest), fill, dtype=dtype)
            for b, m in enumerate(matches):
                out[b, :len(m.tokens)] = torch.as_tensor(getattr(m, key))
            return out.to(device)

        tokens = torch.zeros((count, longest, TOKEN_DIM), dtype=torch.uint8)
        for b, m in enumerate(matches):
            tokens[b, :len(m.tokens)] = torch.as_tensor(m.tokens)
        self.tokens = tokens.to(device)
        self.token_round = padded("token_round", 0, torch.long)
        self.token_phase = padded("token_phase", PLAY_PHASE, torch.long)
        self.token_forced = padded("token_forced", 0, torch.long)

        stack = lambda key: torch.as_tensor(
            np.concatenate([getattr(m, key) for m in matches])).to(device)
        self.obs, self.seat, self.driver, self.prefix = (
            stack("obs"), stack("seat"), stack("driver"), stack("prefix"))
        self.match_index = torch.as_tensor(np.concatenate(
            [np.full(m.decisions, b) for b, m in enumerate(matches)])).to(device)
        self.cand, self.cand_abstract, self.choice = (
            stack("cand"), stack("cand_abstract"), stack("choice"))
        counts = np.concatenate([np.diff(m.cand_offsets) for m in matches])
        self.offsets = torch.as_tensor(np.concatenate(([0], np.cumsum(counts)))).to(device)
        self.counts = torch.as_tensor(counts).to(device)
        self.rows = torch.repeat_interleave(torch.arange(self.decisions, device=device), self.counts)
        self.chosen = self.offsets[:-1] + self.choice
        self.chosen_abstract = self.cand_abstract[self.chosen]
        self.leading = self.obs[:, LEADING_BIT].bool()
        self.legal = torch.zeros(self.decisions, VOCAB, device=device, dtype=torch.bool)
        self.legal[self.rows, self.cand_abstract] = True

    @property
    def decisions(self) -> int:
        return len(self.obs)


def sinusoidal(length: int, width: int, device, dtype) -> Tensor:
    positions = torch.arange(length, device=device, dtype=dtype)[:, None]
    frequencies = torch.exp(torch.arange(0, width, 2, device=device, dtype=dtype)
                            * (-math.log(10000.0) / width))
    table = torch.zeros(length, width, device=device, dtype=dtype)
    table[:, 0::2] = torch.sin(positions * frequencies)
    table[:, 1::2] = torch.cos(positions * frequencies)
    return table


class HistoryTower(nn.Module):
    """Public stream Transformer plus a private query, DESIGN.md 7.2 shape.

    ``encode`` sees only public tokens, so one stream encoding serves every
    seat of a match. The query of a decision may attend only to positions
    ``<= prefix`` (BOS plus the tokens before the decision). With
    ``window = k`` every decision instead gets its own stream of the last
    ``k`` tokens before its prefix, so no layer stack can see further back.
    """

    def __init__(self, config: ProbeConfig) -> None:
        super().__init__()
        self.config = config
        width = config.width
        self.public = nn.Linear(TOKEN_DIM, width)
        self.round_embedding = nn.Embedding(config.max_rounds, width)
        self.phase_embedding = nn.Embedding(4, width)
        self.bos = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, config.heads, width * 4, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.stream = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.stream_norm = nn.LayerNorm(width)
        self.private = mlp(config.obs_dim, width, 2)
        self.seat = nn.Embedding(4, width)
        self.q_proj = nn.Linear(width, width)
        self.kv_proj = nn.Linear(width, 2 * width)
        self.out_proj = nn.Linear(width, width)
        self.attention_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(nn.Linear(width, width * 4), nn.ReLU(),
                                          nn.Linear(width * 4, width))
        self.output_norm = nn.LayerNorm(width)
        self.ntp_head = nn.Linear(width, VOCAB)

    def encode_tokens(self, tokens: Tensor, rounds: Tensor, phases: Tensor,
                      lengths: Tensor) -> tuple[Tensor, Tensor]:
        """Causal encoding of padded public streams; returns (stream, padding)."""
        inputs = tokens.float()
        embedded = (self.public(inputs)
                    + self.round_embedding(rounds.clamp(max=self.config.max_rounds - 1))
                    + self.phase_embedding(phases.clamp(0, 3)))
        stream = torch.cat((self.bos.expand(len(inputs), -1, -1), embedded), dim=1)
        length = stream.shape[1]
        stream = stream + sinusoidal(length, self.config.width, stream.device, stream.dtype)
        padding = torch.arange(length, device=stream.device)[None] > lengths[:, None]
        causal = torch.ones(length, length, device=stream.device, dtype=torch.bool).triu(1)
        encoded = self.stream(stream, mask=causal, src_key_padding_mask=padding)
        return self.stream_norm(encoded), padding

    def encode(self, batch: Batch) -> tuple[Tensor, Tensor]:
        return self.encode_tokens(batch.tokens, batch.token_round, batch.token_phase,
                                  batch.lengths)

    def windowed(self, batch: Batch) -> tuple[Tensor, Tensor]:
        """Per-decision streams of the last ``window`` tokens; returns (stream, lengths)."""
        k = self.config.window
        device = batch.obs.device
        lengths = batch.prefix.clamp(max=k)
        start = batch.prefix - lengths
        positions = start[:, None] + torch.arange(k, device=device)[None]      # [n, k]
        valid = positions < batch.prefix[:, None]
        positions = positions.clamp(min=0, max=batch.tokens.shape[1] - 1)
        rows = batch.match_index[:, None].expand(-1, k)
        def gather(source: Tensor, fill: Tensor) -> Tensor:
            picked = source[rows, positions]
            return torch.where(valid if picked.ndim == 2 else valid[..., None], picked, fill)
        tokens = gather(batch.tokens, torch.zeros((), dtype=torch.uint8, device=device))
        stream, _ = self.encode_tokens(
            tokens, gather(batch.token_round, torch.zeros((), dtype=torch.long, device=device)),
            gather(batch.token_phase, torch.full((), PLAY_PHASE, dtype=torch.long, device=device)),
            lengths)
        return stream, lengths

    def ntp_logits(self, stream: Tensor) -> Tensor:
        return self.ntp_head(stream)

    def private_query(self, batch: Batch) -> Tensor:
        return self.private(batch.obs.float()) + self.seat(batch.seat)

    def attend(self, query: Tensor, keys: Tensor, values: Tensor, allowed: Tensor) -> Tensor:
        """query [n, w]; keys/values [1, S, w] shared or [n, S, w] per row; allowed [n, S]."""
        width, heads = self.config.width, self.config.heads
        depth = width // heads
        q = self.q_proj(query).view(-1, heads, depth)                                # [n, h, d]
        k = keys.view(keys.shape[0], keys.shape[1], heads, depth)
        v = values.view(values.shape[0], values.shape[1], heads, depth)
        if keys.shape[0] == 1:
            out = F.scaled_dot_product_attention(
                q.transpose(0, 1)[None], k[0].transpose(0, 1)[None], v[0].transpose(0, 1)[None],
                attn_mask=allowed[None, None])[0].transpose(0, 1)                   # [n, h, d]
        else:
            out = F.scaled_dot_product_attention(
                q[:, :, None], k.transpose(1, 2), v.transpose(1, 2),
                attn_mask=allowed[:, None, None])[:, :, 0]                          # [n, h, d]
        return self.out_proj(out.reshape(-1, width))

    def finish(self, query: Tensor, attended: Tensor) -> Tensor:
        state = self.attention_norm(query + attended)
        return self.output_norm(state + self.feed_forward(state))

    def forward(self, batch: Batch) -> tuple[Tensor, Tensor]:
        """Returns (state [n, w], the stream position at every decision's prefix [n, w])."""
        query = self.private_query(batch)
        if self.config.window:
            stream, lengths = self.windowed(batch)
            keys, values = self.kv_proj(stream).chunk(2, dim=-1)
            allowed = torch.arange(stream.shape[1], device=stream.device)[None] <= lengths[:, None]
            attended = self.attend(query, keys, values, allowed)
            at_prefix = stream[torch.arange(batch.decisions, device=stream.device), lengths]
            return self.finish(query, attended), at_prefix
        stream, _ = self.encode(batch)
        keys, values = self.kv_proj(stream).chunk(2, dim=-1)
        positions = torch.arange(stream.shape[1], device=stream.device)
        attended = torch.empty_like(query)
        for b in range(stream.shape[0]):
            rows = (batch.match_index == b).nonzero(as_tuple=True)[0]
            if not len(rows):
                continue
            allowed = positions[None] <= batch.prefix[rows][:, None]
            attended[rows] = self.attend(query[rows], keys[b:b + 1], values[b:b + 1], allowed)
        at_prefix = stream[batch.match_index, batch.prefix]
        return self.finish(query, attended), at_prefix


class FlatTower(nn.Module):
    """The v1 shape: an MLP over the flat observation, no history."""

    def __init__(self, config: ProbeConfig) -> None:
        super().__init__()
        self.config = config
        self.tower = mlp(config.obs_dim, config.width, config.layers)
        self.seat = nn.Embedding(4, config.width)

    def forward(self, batch: Batch) -> tuple[Tensor, Tensor | None]:
        return self.tower(batch.obs.float()) + self.seat(batch.seat), None


class CandidateHead(nn.Module):
    """Score every concrete candidate from the state and its action encoding."""

    def __init__(self, config: ProbeConfig) -> None:
        super().__init__()
        self.action_tower = mlp(ACT_DIM, config.action_width, 2)
        self.fusion = nn.Sequential(mlp(config.width + config.action_width, config.fusion_width, 2),
                                    nn.Linear(config.fusion_width, 1))

    def abstract_log_probs(self, state: Tensor, batch: Batch) -> tuple[Tensor, Tensor]:
        """Returns (log-prob per abstract id [n, VOCAB], log-prob of the chosen candidate)."""
        scores = self.fusion(torch.cat((state[batch.rows], self.action_tower(batch.cand.float())),
                                       dim=-1)).squeeze(-1)
        log_probs = segment_log_softmax(scores, batch.rows, len(state))
        probs = torch.zeros(len(state), VOCAB, device=state.device, dtype=log_probs.dtype)
        probs.index_put_((batch.rows, batch.cand_abstract), log_probs.exp(), accumulate=True)
        return probs.clamp_min(1e-30).log(), log_probs[batch.chosen]


class VocabHead(nn.Module):
    """One logit per abstract action, illegal entries masked out.

    ``structured`` embeds every id from its type, key, bomb size, pair slot
    and suit features, so entries share parameters; otherwise each id is an
    independent row of a linear layer.
    """

    def __init__(self, config: ProbeConfig, structured: bool) -> None:
        super().__init__()
        self.structured = structured
        if structured:
            self.register_buffer("features", torch.as_tensor(abstract_features()))
            self.embed = nn.Sequential(mlp(FEATURE_DIM, config.width, 2),
                                       nn.Linear(config.width, config.width))
        else:
            self.out = nn.Linear(config.width, VOCAB)

    def abstract_log_probs(self, state: Tensor, batch: Batch) -> tuple[Tensor, Tensor]:
        if self.structured:
            logits = state @ self.embed(self.features).T / math.sqrt(state.shape[1])
        else:
            logits = self.out(state)
        log_probs = F.log_softmax(logits.masked_fill(~batch.legal, float("-inf")), dim=-1)
        return log_probs, log_probs.gather(1, batch.chosen_abstract[:, None])[:, 0]


def segment_log_softmax(scores: Tensor, rows: Tensor, count: int) -> Tensor:
    top = torch.full((count,), float("-inf"), device=scores.device, dtype=scores.dtype)
    top = top.scatter_reduce(0, rows, scores, reduce="amax")
    shifted = (scores - top[rows]).exp()
    total = torch.zeros(count, device=scores.device, dtype=scores.dtype).index_add_(0, rows, shifted)
    return scores - top[rows] - total[rows].log()


def make_head(config: ProbeConfig) -> nn.Module:
    if config.head == "cand":
        return CandidateHead(config)
    return VocabHead(config, structured=config.head == "vocab_struct")


class ProbeModel(nn.Module):
    def __init__(self, config: ProbeConfig) -> None:
        super().__init__()
        self.config = config
        self.tower = HistoryTower(config) if config.tower == "history" else FlatTower(config)
        self.head = make_head(config)

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        state, at_prefix = self.tower(batch)
        abstract, chosen = self.head.abstract_log_probs(state, batch)
        bc_loss = -chosen.mean()
        out = {"bc_loss": bc_loss, "abstract_log_probs": abstract, "loss": bc_loss}
        if self.config.ntp_weight > 0 and at_prefix is not None:
            # Public-only prediction of the decision's own action: what the
            # stream alone says this actor will do next.
            logits = self.tower.ntp_logits(at_prefix)
            ntp_loss = F.cross_entropy(logits, batch.chosen_abstract)
            out.update(ntp_loss=ntp_loss, ntp_logits=logits,
                       loss=bc_loss + self.config.ntp_weight * ntp_loss)
        return out


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ---- training and evaluation ----------------------------------------------

class Accumulator:
    def __init__(self) -> None:
        self.sums: dict[str, list[float]] = {}

    def add(self, name: str, total: float, count: int) -> None:
        entry = self.sums.setdefault(name, [0.0, 0])
        entry[0] += float(total)
        entry[1] += int(count)

    def means(self) -> dict[str, float]:
        return {name: (total / count if count else float("nan"))
                for name, (total, count) in self.sums.items()}

    def counts(self) -> dict[str, int]:
        return {name: count for name, (_, count) in self.sums.items()}


def cells(batch: Batch) -> dict[str, Tensor]:
    """Evaluation cells: nontrivial decisions split by driver and by lead/follow."""
    nontrivial = batch.legal.sum(1) > 1
    policy, bot = batch.driver == DRIVER_POLICY, batch.driver == DRIVER_BOT
    return {"all": torch.ones_like(nontrivial), "nontrivial": nontrivial,
            "policy": nontrivial & policy, "bot": nontrivial & bot,
            "lead": nontrivial & batch.leading, "follow": nontrivial & ~batch.leading,
            "policy_lead": nontrivial & policy & batch.leading,
            "policy_follow": nontrivial & policy & ~batch.leading,
            "bot_lead": nontrivial & bot & batch.leading,
            "bot_follow": nontrivial & bot & ~batch.leading}


@torch.no_grad()
def evaluate(model: ProbeModel, matches: list[Match], device: str, batch_matches: int,
             ntp: bool = False) -> dict:
    """Held-out metrics at the abstract-action level, so every head compares."""
    model.eval()
    acc = Accumulator()
    for start in range(0, len(matches), batch_matches):
        batch = Batch(matches[start:start + batch_matches], device)
        out = model(batch)
        log_probs = out["abstract_log_probs"]
        chosen = log_probs.gather(1, batch.chosen_abstract[:, None])[:, 0]
        correct = (log_probs.argmax(1) == batch.chosen_abstract).float()
        masks = cells(batch)
        for name, mask in masks.items():
            acc.add(f"bc_ce/{name}", -chosen[mask].sum(), int(mask.sum()))
            acc.add(f"bc_acc/{name}", correct[mask].sum(), int(mask.sum()))
        if ntp and "ntp_logits" in out:
            logits, labels = out["ntp_logits"], batch.chosen_abstract
            losses = F.cross_entropy(logits, labels, reduction="none")
            hits = (logits.argmax(1) == labels).float()
            masks.update(next_pass=labels == PASS_ID, next_play=labels != PASS_ID)
            for name, mask in masks.items():
                acc.add(f"ntp_ce/{name}", losses[mask].sum(), int(mask.sum()))
                acc.add(f"ntp_acc/{name}", hits[mask].sum(), int(mask.sum()))
    model.train()
    return {"means": acc.means(), "counts": acc.counts()}


def run_probe(data: Path, output: Path, config: ProbeConfig, *, steps: int = 2000,
              batch_matches: int = 4, lr: float = 3e-4, seed: int = 31, split_seed: int = 20260925,
              eval_every: int = 200, patience: int = 5, device: str = "cpu",
              max_rounds: int | None = None, min_matches: int = 10,
              max_seconds: float = float("inf")) -> dict:
    """Train one arm and evaluate it on held-out matches at the best validation step."""
    started = time.monotonic()
    if steps <= 0 or batch_matches <= 0 or eval_every <= 0 or patience <= 0:
        raise ValueError("steps, batch size, evaluation interval and patience must be positive")
    output.mkdir(parents=True, exist_ok=True)
    matches, provenance = load_matches(data, max_rounds=max_rounds)
    splits = split_matches(matches, split_seed, min_matches=min_matches)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ProbeModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ntp = config.ntp_weight > 0
    drivers = {name: {"policy": int(sum((m.driver == DRIVER_POLICY).sum() for m in part)),
                      "bot": int(sum((m.driver == DRIVER_BOT).sum() for m in part))}
               for name, part in splits.items()}
    report = {
        "status": "running", "config": asdict(config), "data": str(data),
        "collection_checkpoint_id": provenance.get("checkpoint_id"),
        "collection_styled": bool(provenance.get("styled")),
        "steps_requested": steps, "batch_matches": batch_matches, "lr": lr, "seed": seed,
        "split_seed": split_seed, "parameters": {
            "total": count_parameters(model), "tower": count_parameters(model.tower),
            "head": count_parameters(model.head)},
        "matches": {name: len(part) for name, part in splits.items()},
        "decisions": drivers,
        "tokens": {name: sum(len(m.tokens) for m in part) for name, part in splits.items()},
        "curve": [], "best": None,
    }
    train = splits["train"]
    order = rng.permutation(len(train))
    cursor = 0
    best_loss, best_step, best_state, stale = float("inf"), 0, None, 0
    try:
        for step in range(1, steps + 1):
            if time.monotonic() - started > max_seconds:
                raise TimeoutError(f"probe exceeded {max_seconds} seconds")
            if cursor + batch_matches > len(order):
                order = rng.permutation(len(train))
                cursor = 0
            picked = [train[i] for i in order[cursor:cursor + batch_matches]]
            cursor += batch_matches
            batch = Batch(picked, device)
            out = model(batch)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            if not torch.isfinite(out["loss"]):
                raise FloatingPointError(f"non-finite loss at step {step}")
            if step % eval_every == 0 or step == steps:
                validation = evaluate(model, splits["validation"], device, batch_matches, ntp)
                point = {"step": step, "train_loss": float(out["loss"].detach()),
                         "train_bc_loss": float(out["bc_loss"].detach()),
                         "validation": validation["means"], "elapsed": time.monotonic() - started}
                if ntp:
                    point["train_ntp_loss"] = float(out["ntp_loss"].detach())
                report["curve"].append(point)
                selected = validation["means"]["bc_ce/all"]
                if ntp:
                    selected = selected + config.ntp_weight * validation["means"]["ntp_ce/all"]
                if selected < best_loss:
                    best_loss, best_step, stale = selected, step, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    torch.save({"config": asdict(config), "model": best_state, "step": step},
                               output / "best.pt")
                else:
                    stale += 1
                    if stale >= patience:
                        report["stop"] = "plateau"
                        break
        else:
            report["stop"] = "step_cap"
        model.load_state_dict(best_state)
        report["best"] = {"step": best_step, "validation_selected": best_loss}
        report["test"] = evaluate(model, splits["test"], device, batch_matches, ntp)
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "incomplete"
        report["error_type"] = type(error).__name__
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tower", default="history", choices=("history", "flat"))
    parser.add_argument("--head", default="cand", choices=HEADS)
    parser.add_argument("--ntp-weight", type=float, default=0.0)
    parser.add_argument("--window", type=int, default=0,
                        help="history tower sees only the last k tokens (0 = whole match)")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-matches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--split-seed", type=int, default=20260925)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--min-matches", type=int, default=10)
    parser.add_argument("--max-seconds", type=float, default=float("inf"))
    args = parser.parse_args(argv)
    config = ProbeConfig(tower=args.tower, head=args.head, ntp_weight=args.ntp_weight,
                         window=args.window, width=args.width, layers=args.layers,
                         heads=args.heads)
    report = run_probe(args.data, args.output, config, steps=args.steps,
                       batch_matches=args.batch_matches, lr=args.lr, seed=args.seed,
                       split_seed=args.split_seed, eval_every=args.eval_every,
                       patience=args.patience, device=args.device, max_rounds=args.max_rounds,
                       min_matches=args.min_matches, max_seconds=args.max_seconds)
    print(json.dumps({"status": report["status"], "best": report["best"],
                      "test": report.get("test", {}).get("means")}))


if __name__ == "__main__":
    main()
