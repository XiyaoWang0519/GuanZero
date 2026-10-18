"""Batched paired-search rollouts: C++ VecEnv stepping, one shared-prefix KV
cache, one forward per step for every branch of many searches.

Design: docs/reports/batched-rollout-design-2026-10-18.md. Drop-in for
``history_habit_round.team_rollout`` when the opponent types are logit tilts
``pi_z ∝ pi exp(z b f)`` of the observer's own network (the habit pack): the
trunk is evaluated once per decision and only the opponent rows get their
type's tilt. Each search's root history is encoded once and shared by all of
its branches; a branch stores only the events of its own rollout.

Needs a ``gd`` build with ``VecEnv.load``, ``stop_at_round_end`` and
``drain_public_arrays`` (this commit's engine).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os

import gd
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from bench.qknorm_match_ablation import MatchSampler, inverse_cdf
from train.history_attention import finish, project
from train.history_habit import style_feature
from train.history_inference import BatchedHistoryCache
from train.history_model import DecisionInputs, PublicStream, StreamBatch, segment_log_softmax, sinusoidal
from train.logs import TOKEN_DIM

ACT_TRIBUTE_FLAGS = 146
POSITIONS = 4096                 # sinusoidal table length; streams stay far below it
CHUNK_ROWS = 1024                # rows per attention chunk (bounds transient memory)


@dataclass
class SearchRequest:
    """One paired search: every type x candidate x world from ``state``.

    ``z`` lists the opponent types (+1/-1 tilts of strength ``b`` on ``axis``);
    a type of 0 means the untilted network. ``candidates`` index the root's
    legal actions in engine order. Values come from the mover's seat.
    """
    state: gd.MatchState
    arrays: tuple
    candidates: np.ndarray
    seeds: np.ndarray
    z: tuple
    axis: str
    b: float


def query_heads(actor, query: Tensor) -> Tensor:
    """``[n, heads, depth]`` private query heads; plain ``q_proj`` on builds
    without the query/QK-norm options."""
    if hasattr(actor, '_query_heads'):
        return actor._query_heads(query)
    cfg = actor.config
    return actor.q_proj(query).view(-1, cfg.heads, cfg.width // cfg.heads)


def key_heads(actor, k: Tensor) -> Tensor:
    return actor._key_heads(k) if hasattr(actor, '_key_heads') else k


def public_tokens(seat: np.ndarray, encoded: np.ndarray, cards_left: np.ndarray) -> np.ndarray:
    """Vectorised ``train.logs.public_token`` for drained events."""
    n = len(seat)
    token = np.zeros((n, TOKEN_DIM), np.uint8)
    token[np.arange(n), seat] = 1
    token[:, 4:158] = encoded
    token[:, 4 + ACT_TRIBUTE_FLAGS:4 + 154] = 0
    token[np.arange(n), 158 + cards_left] = 1
    return token


class SharedPrefixCache:
    """Public KV for branches that extend one of several root streams.

    Root keys/values/memory are stored once per search ([search, heads, R, d]);
    each row keeps only its suffix. Attention over root + suffix is computed
    as one softmax over both score blocks: the same function as attending to
    the concatenated stream, with a different FP reduction order.
    """

    def __init__(self, actor, roots: list[tuple]) -> None:
        self.actor, cfg = actor, actor.config
        self.heads, self.depth, self.width = cfg.heads, cfg.width // cfg.heads, cfg.width
        proto = actor.bos
        self.device, self.dtype = proto.device, proto.dtype
        base = BatchedHistoryCache(actor)
        streams = []
        for tokens, rounds, phases in roots:
            s = PublicStream()
            for t, r, p in zip(tokens, rounds, phases):
                s.append_token(t, int(r), int(p))
            streams.append(s)
        keys = [(0, i) for i in range(len(streams))]
        base.prefill(keys, streams)
        entries = [base.entries[k] for k in keys]
        self.root_len = np.array([e.length for e in entries], np.int64)     # BOS included
        rmax = int(self.root_len.max())
        n = len(entries)
        self.rk = [proto.new_zeros(n, self.heads, rmax, self.depth) for _ in range(cfg.layers)]
        self.rv = [proto.new_zeros(n, self.heads, rmax, self.depth) for _ in range(cfg.layers)]
        self.rm = proto.new_zeros(n, rmax, self.width)
        for i, e in enumerate(entries):
            for layer in range(cfg.layers):
                self.rk[layer][i, :, :e.length] = e.keys[layer][:, :e.length]
                self.rv[layer][i, :, :e.length] = e.values[layer][:, :e.length]
            self.rm[i, :e.length] = e.memory[:e.length]
        self.root_len_t = torch.as_tensor(self.root_len, device=self.device)
        # Decision-time keys/values of the root memory, once per search.
        rkv = actor.kv_proj(self.rm)
        k, v = rkv.chunk(2, dim=-1)
        self.dk = key_heads(actor, k.view(n, rmax, self.heads, self.depth))
        self.dv = v.view(n, rmax, self.heads, self.depth)
        self.positions = sinusoidal(POSITIONS, self.width, self.device, self.dtype)
        self.rows = 0

    def add_rows(self, search: np.ndarray, capacity: int = 32) -> None:
        cfg = self.actor.config
        self.search = np.asarray(search, np.int64)
        self.search_t = torch.as_tensor(self.search, device=self.device)
        self.rows = rows = len(search)
        self.suffix = np.zeros(rows, np.int64)           # encoded suffix positions per row
        self.cap = capacity
        self.sk = [torch.zeros(rows, self.heads, capacity, self.depth, device=self.device, dtype=self.dtype)
                   for _ in range(cfg.layers)]
        self.sv = [torch.zeros_like(self.sk[0]) for _ in range(cfg.layers)]
        self.sm = torch.zeros(rows, capacity, self.width, device=self.device, dtype=self.dtype)
        self.pend_tok = np.zeros((rows, 4, TOKEN_DIM), np.uint8)
        self.pend_round = np.zeros((rows, 4), np.int64)
        self.pend_phase = np.zeros((rows, 4), np.int64)
        self.pending = np.zeros(rows, np.int64)

    def _grow(self, need: int) -> None:
        cap = self.cap
        while cap < need:
            cap *= 2
        if cap == self.cap:
            return
        self.sk = [self._widen4(t, cap) for t in self.sk]
        self.sv = [self._widen4(t, cap) for t in self.sv]
        sm = self.sm.new_zeros(self.rows, cap, self.width)
        sm[:, :self.cap] = self.sm
        self.sm, self.cap = sm, cap

    def _widen4(self, t: Tensor, cap: int) -> Tensor:
        out = t.new_zeros(t.shape[0], t.shape[1], cap, t.shape[3])
        out[:, :, :t.shape[2]] = t
        return out

    def push(self, rows: np.ndarray, tokens: np.ndarray, rounds: np.ndarray, phases: np.ndarray) -> None:
        """Queue events (in order) for later encoding; rows may repeat."""
        if not len(rows):
            return
        order = np.argsort(rows, kind='stable')
        rows, tokens, rounds, phases = rows[order], tokens[order], rounds[order], phases[order]
        first = np.r_[True, rows[1:] != rows[:-1]]
        start = np.flatnonzero(first)
        rank = np.arange(len(rows)) - np.repeat(start, np.diff(np.r_[start, len(rows)]))
        slot = self.pending[rows] + rank
        need = int(slot.max()) + 1
        if need > self.pend_tok.shape[1]:
            grow = max(need, 2 * self.pend_tok.shape[1])
            for name in ('pend_tok', 'pend_round', 'pend_phase'):
                old = getattr(self, name)
                new = np.zeros((old.shape[0], grow) + old.shape[2:], old.dtype)
                new[:, :old.shape[1]] = old
                setattr(self, name, new)
        self.pend_tok[rows, slot] = tokens
        self.pend_round[rows, slot] = rounds
        self.pend_phase[rows, slot] = phases
        np.add.at(self.pending, rows, 1)

    @torch.no_grad()
    def flush(self, rows: np.ndarray) -> None:
        rows = rows[self.pending[rows] > 0]
        rows = rows[np.argsort(self.search[rows], kind='stable')]       # searches contiguous
        for lo in range(0, len(rows), CHUNK_ROWS):
            self._encode(rows[lo:lo + CHUNK_ROWS])

    def _groups(self, rows: np.ndarray) -> list[tuple[int, int, int, int]]:
        """(search, begin, end, root length) for rows sorted by search."""
        s = self.search[rows]
        cut = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
        return [(int(s[a]), int(a), int(b), int(self.root_len[s[a]])) for a, b in zip(cut[:-1], cut[1:])]

    def _encode(self, rows: np.ndarray) -> None:
        actor, cfg = self.actor, self.actor.config
        counts = self.pending[rows]
        w = int(counts.max())
        L = int((self.suffix[rows] + counts).max())
        self._grow(L)
        dev = self.device
        rt = torch.as_tensor(rows, device=dev)
        t = torch.as_tensor(self.pend_tok[rows, :w], device=dev)
        r = torch.as_tensor(self.pend_round[rows, :w], device=dev)
        p = torch.as_tensor(self.pend_phase[rows, :w], device=dev)
        suffix = torch.as_tensor(self.suffix[rows], device=dev)
        rlen = self.root_len_t[self.search_t[rt]]
        steps = torch.arange(w, device=dev)
        pos = rlen[:, None] + suffix[:, None] + steps[None]                  # absolute positions
        state = (actor.public(t.to(self.dtype)) + actor.round_embedding(r.clamp(0, cfg.max_rounds - 1))
                 + actor.phase_embedding(p.clamp(0, 3)))
        state = state + self.positions[pos]
        valid = steps[None] < torch.as_tensor(counts, device=dev)[:, None]
        # suffix slot j is visible to new token i iff j <= suffix + i
        suf_mask = torch.arange(L, device=dev)[None, None] <= (suffix[:, None, None] + steps[None, :, None])
        write_rows = rt[:, None].expand(-1, w)[valid]
        write_pos = (suffix[:, None] + steps[None])[valid]
        groups = self._groups(rows)
        rg = max(g[3] for g in groups)
        scale = 1.0 / math.sqrt(self.depth)
        n = len(rows)
        for layer_index, layer in enumerate(actor.stream.layers):
            q, k, v = project(layer, state)                                  # [B, h, w, d]
            self.sk[layer_index][write_rows, :, write_pos] = k.transpose(1, 2)[valid]
            self.sv[layer_index][write_rows, :, write_pos] = v.transpose(1, 2)[valid]
            sk = self.sk[layer_index][rt, :, :L]
            sv = self.sv[layer_index][rt, :, :L]
            s_suf = (q @ sk.transpose(-1, -2)) * scale
            s_suf = s_suf.masked_fill(~suf_mask[:, None], float('-inf'))
            s_root = q.new_full((n, self.heads, w, rg), float('-inf'))
            # Root blocks as one [heads, rows*steps, R] bmm per search: no
            # per-row copy of the shared root keys (a broadcast matmul would
            # expand them).
            for search, a, b, R in groups:
                qa = q[a:b].transpose(0, 1).reshape(self.heads, (b - a) * w, self.depth)
                sc = torch.bmm(qa, self.rk[layer_index][search, :, :R].transpose(-1, -2)) * scale
                s_root[a:b, :, :, :R] = sc.view(self.heads, b - a, w, R).transpose(0, 1)
            weights = torch.softmax(torch.cat((s_root, s_suf), dim=-1), dim=-1)
            attended = weights[..., rg:] @ sv
            for search, a, b, R in groups:
                wa = weights[a:b, :, :, :R].transpose(0, 1).reshape(self.heads, (b - a) * w, R)
                o = torch.bmm(wa, self.rv[layer_index][search, :, :R])
                attended[a:b] += o.view(self.heads, b - a, w, self.depth).transpose(0, 1)
            state = finish(layer, state, attended)
        if actor.stream.norm is not None:
            state = actor.stream.norm(state)
        state = actor.stream_norm(state)
        self.sm[write_rows, write_pos] = state[valid]
        self.suffix[rows] += counts
        self.pending[rows] = 0

    @torch.no_grad()
    def attend(self, rows: np.ndarray, query: Tensor) -> Tensor:
        """Private decision attention of ``query`` [B, width] over each row's
        whole encoded stream (root + suffix), as ``_attend_encoded``."""
        actor = self.actor
        order = np.argsort(self.search[rows], kind='stable')
        result = torch.empty_like(query)
        scale = 1.0 / math.sqrt(self.depth)
        dev = self.device
        for lo in range(0, len(rows), CHUNK_ROWS):
            pick = order[lo:lo + CHUNK_ROWS]
            rr = rows[pick]
            pick_t = torch.as_tensor(pick, device=dev)
            n = len(rr)
            q = query_heads(actor, query[pick_t])                             # [B, h, d]
            L = int(self.suffix[rr].max())
            mem = self.sm[torch.as_tensor(rr, device=dev), :L]
            k, v = actor.kv_proj(mem).chunk(2, dim=-1)
            k = key_heads(actor, k.view(n, L, self.heads, self.depth))
            v = v.view(n, L, self.heads, self.depth)
            s_suf = torch.einsum('bhd,blhd->bhl', q, k) * scale
            ok = torch.arange(L, device=dev)[None] < torch.as_tensor(self.suffix[rr], device=dev)[:, None]
            s_suf = s_suf.masked_fill(~ok[:, None], float('-inf'))
            groups = self._groups(rr)
            rg = max(g[3] for g in groups)
            s_root = q.new_full((n, self.heads, rg), float('-inf'))
            for search, a, b, R in groups:
                kh = self.dk[search, :R].transpose(0, 1)                     # [h, R, d]
                sc = torch.bmm(q[a:b].transpose(0, 1), kh.transpose(-1, -2)) * scale   # [h, n, R]
                s_root[a:b, :, :R] = sc.transpose(0, 1)
            weights = torch.softmax(torch.cat((s_root, s_suf), dim=-1), dim=-1)
            o = torch.einsum('bhl,blhd->bhd', weights[..., rg:], v)
            for search, a, b, R in groups:
                vh = self.dv[search, :R].transpose(0, 1)                     # [h, R, d]
                o[a:b] += torch.bmm(weights[a:b, :, :R].transpose(0, 1), vh).transpose(0, 1)
            result[pick_t] = actor.out_proj(o.reshape(n, self.width))
        return result


def device_inverse_cdf(table: Tensor, uniform: Tensor) -> Tensor:
    """``qknorm_match_ablation.inverse_cdf`` on the table's own device (FP64 when
    the backend has it, else FP32): same rule, no host copy of the table."""
    dtype = torch.float64 if table.device.type != 'mps' else torch.float32
    probabilities = table.to(dtype).exp()
    probabilities = probabilities / probabilities.sum(1, keepdim=True)
    cdf = probabilities.cumsum(1)
    cdf[:, -1] = 1.
    return (cdf < uniform.to(dtype)[:, None]).sum(1)


def splitmix_uniform(seed: np.ndarray, seat: np.ndarray, counter: np.ndarray) -> np.ndarray:
    """Counter-based U[0,1): splitmix64 of (seed, seat, counter); vectorised."""
    with np.errstate(over='ignore'):
        x = (seed.astype(np.uint64) ^ (seat.astype(np.uint64) << np.uint64(56))
             ^ (counter.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)))
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


@torch.no_grad()
def batched_search(actor, requests: list[SearchRequest], sampler: str = 'fast',
                   num_threads: int = 0, max_steps: int = 512) -> list[np.ndarray]:
    """Q tables ``[type, candidate, world, (team round win, PPO return)]`` per request."""
    if sampler not in ('fast', 'reference'):
        raise ValueError('sampler must be fast or reference')
    device = actor.bos.device
    engine = gd.Engine()
    states, meta = [], []          # meta: (request, type index, candidate index, world index)
    roots = []
    for ri, req in enumerate(requests):
        seat = int(req.state.to_move)
        worlds = [req.state.determinize_uniform(seat, int(s)) for s in req.seeds]
        roots.append(req.arrays)
        for ti in range(len(req.z)):
            for ci in range(len(req.candidates)):
                for wi, w in enumerate(worlds):
                    states.append(w)
                    meta.append((ri, ti, ci, wi))
    meta = np.asarray(meta, np.int64)
    n = len(states)
    req_of = meta[:, 0]
    mover = np.array([int(r.state.to_move) for r in requests], np.int64)[req_of]
    team = mover % 2
    zrow = np.array([requests[r].z[t] for r, t in zip(meta[:, 0], meta[:, 1])], np.float64)
    brow = np.array([requests[r].b for r in meta[:, 0]], np.float64)
    axes = sorted({r.axis for r in requests})
    axis_row = np.array([axes.index(requests[r].axis) for r in meta[:, 0]], np.int64)
    branch_seed = np.array([int(requests[r].seeds[w]) for r, w in zip(meta[:, 0], meta[:, 3])], np.int64)
    cand_choice = np.array([int(requests[r].candidates[c]) for r, c in zip(meta[:, 0], meta[:, 2])], np.int32)

    # VecEnv threads only shard engine stepping/encoding; results do not depend on them.
    threads = num_threads or max(1, min(16, (os.cpu_count() or 1) // 2))
    env = gd.VecEnv(1, threads, 0, log_public_actions=True, stop_at_round_end=True)
    env.load(states)
    cache = SharedPrefixCache(actor, roots)
    cache.add_rows(req_of)
    ref = MatchSampler(branch_seed) if sampler == 'reference' else None
    counter = np.zeros((n, 4), np.int64)
    values = np.full((n, 2), np.nan)

    # Root decision: every branch takes its fixed candidate after an exact
    # check that the sampled world left the mover's view unchanged.
    batch = env.pending()
    if batch.rows != n:
        raise RuntimeError('every loaded world must start at the root decision')
    rows_env = np.asarray(batch.env_id)
    obs = np.asarray(batch.obs)
    cand = np.asarray(batch.cand)
    offsets = np.asarray(batch.offsets)
    for ri, req in enumerate(requests):
        s = int(req.state.to_move)
        legal = engine.legal_actions(req.state)
        want_obs = np.asarray(req.state.observation(s))
        want_cand = np.asarray([engine.encode_action(a, req.state, s) for a in legal], np.float32)
        for r in np.flatnonzero(req_of[rows_env] == ri):
            if not np.array_equal(obs[r], want_obs):
                raise ValueError('private world altered the mover observation')
            if not np.array_equal(cand[offsets[r]:offsets[r + 1]], want_cand):
                raise ValueError('private world altered the legal candidates')
    env.step(np.ascontiguousarray(cand_choice[rows_env], dtype=np.int32))

    for _ in range(max_steps):
        # pending() advances first (lone passes, round ends); its events and
        # the previous step's are drained before any decision reads the cache.
        batch = env.pending()
        e_env, e_seat, e_phase, e_round, e_left, _, e_enc = env.drain_public_arrays()
        if len(e_env):
            cache.push(np.asarray(e_env, np.int64), public_tokens(e_seat, e_enc, e_left),
                       np.asarray(e_round, np.int64), np.asarray(e_phase, np.int64))
        for res in env.drain_finished_rounds():
            i = int(res.env_id)
            values[i] = [int(res.winning_team) == team[i], float(res.seat_return[int(mover[i])])]
        if batch.rows == 0:
            break
        rows_env = np.asarray(batch.env_id).astype(np.int64)
        seats = np.asarray(batch.seat).astype(np.int64)
        offsets = np.asarray(batch.offsets).astype(np.int64)
        counts = np.diff(offsets)
        choice = np.zeros(batch.rows, np.int32)
        multi = np.flatnonzero(counts > 1)
        if len(multi):
            br = rows_env[multi]
            cache.flush(br)
            obs = torch.as_tensor(np.asarray(batch.obs)[multi], device=device)
            cand_all = np.asarray(batch.cand)
            lens = counts[multi]
            idx = np.arange(int(lens.sum())) + np.repeat(offsets[multi] - np.r_[0, np.cumsum(lens)[:-1]], lens)
            c = torch.as_tensor(cand_all[idx], device=device)
            off = np.r_[0, np.cumsum(counts[multi])]
            off_t = torch.as_tensor(off, device=device)
            seat_t = torch.as_tensor(seats[multi], device=device)
            query = actor.private(obs.float()) + actor.seat(seat_t)
            attended = cache.attend(br, query)
            state = actor.attention_norm(query + attended)
            state = actor.output_norm(state + actor.feed_forward(state))
            nrows = torch.repeat_interleave(torch.arange(len(multi), device=device),
                                            torch.as_tensor(counts[multi], device=device))
            logits = actor.candidate_logits(state, c, off_t, rows=nrows)
            inputs = DecisionInputs(StreamBatch(torch.empty(len(multi), 0, TOKEN_DIM, dtype=torch.uint8, device=device),
                                                torch.empty(len(multi), 0, dtype=torch.long, device=device),
                                                torch.empty(len(multi), 0, dtype=torch.long, device=device),
                                                torch.zeros(len(multi), dtype=torch.long, device=device)),
                                    torch.arange(len(multi), device=device), torch.zeros(len(multi), dtype=torch.long, device=device),
                                    obs, seat_t, c, off_t, one_decision_per_stream=True)
            opp = (seats[multi] % 2) != team[br]
            tilt = np.where(opp, zrow[br] * brow[br], 0.0)
            if np.any(tilt != 0):
                bias = torch.zeros(len(c), device=device, dtype=logits.dtype)
                for a, axis in enumerate(axes):
                    sel = opp & (axis_row[br] == a) & (tilt != 0)
                    if sel.any():
                        f = style_feature(inputs, axis)
                        coef = torch.as_tensor(np.where(sel, tilt, 0.0), device=device, dtype=logits.dtype)
                        bias = bias + coef[nrows] * f.to(logits.dtype)
                base = segment_log_softmax(logits, nrows, len(multi))
                tilted = torch.as_tensor(tilt != 0, device=device)[nrows]
                # Untilted rows keep the single softmax of the reference path.
                log_probs = torch.where(tilted, segment_log_softmax(base + bias, nrows, len(multi)), base)
            else:
                log_probs = segment_log_softmax(logits, nrows, len(multi))
            width = int(counts[multi].max())
            table = torch.full((len(multi), width), float('-inf'), device=device, dtype=log_probs.dtype)
            local = torch.arange(len(c), device=device) - off_t[:-1][nrows]
            table[nrows, local] = log_probs
            if ref is not None:
                u = ref.draw(br, seats[multi])
            else:
                u = splitmix_uniform(branch_seed[br], seats[multi], counter[br, seats[multi]])
                counter[br, seats[multi]] += 1
            if ref is not None:
                picked, _ = inverse_cdf(table.cpu(), torch.as_tensor(u))     # exact reference path
            else:
                picked = device_inverse_cdf(table, torch.as_tensor(u, device=device))
            choice[multi] = picked.cpu().numpy().astype(np.int32)
        env.step(choice)
    else:
        raise RuntimeError('unfinished rollout branch; no partial result accepted')
    if not np.isfinite(values).all():
        raise RuntimeError('a branch ended without a round result')
    out = []
    for ri, req in enumerate(requests):
        sel = req_of == ri
        q = np.zeros((len(req.z), len(req.candidates), len(req.seeds), 2))
        q[meta[sel, 1], meta[sel, 2], meta[sel, 3]] = values[sel]
        out.append(q)
    return out
