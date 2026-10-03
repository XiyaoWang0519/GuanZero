"""NumPy forward pass of a ``train.history_model.HistoryActor`` (greedy play).

Botzone's Python sandbox has NumPy but no usable PyTorch, so the bot scores
candidates with this re-implementation of the actor's inference path:
causal public-stream encoder (pre-LN ``nn.TransformerEncoder``), private
query over the whole encoded prefix, candidate tower and fusion head. The
auxiliary heads and the response head never enter the policy's scores at
inference and are not exported. ``eval/botzone/export.py`` writes the
weights; ``tests/test_botzone.py`` checks the logits against the torch actor.

NumPy only, Python 3.6 compatible.
"""
import json
import math
from typing import Dict

import numpy as np

TOKEN_DIM = 4 + 154 + 28
PLAY_PHASE = 3
_EPS = 1e-5


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + _EPS) * weight + bias


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def sinusoidal(length: int, width: int) -> np.ndarray:
    positions = np.arange(length, dtype=np.float32)[:, None]
    frequencies = np.exp(np.arange(0, width, 2, dtype=np.float32)
                         * np.float32(-math.log(10000.0) / width))
    table = np.zeros((length, width), dtype=np.float32)
    table[:, 0::2] = np.sin(positions * frequencies)
    table[:, 1::2] = np.cos(positions * frequencies)
    return table


class NumpyHistoryActor(object):
    """Scores the candidates of one decision from its public stream."""

    def __init__(self, weights: Dict[str, np.ndarray], config: dict) -> None:
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        self.config = dict(config)
        self.width = int(config["width"])
        self.heads = int(config["heads"])
        self.layers = int(config["layers"])
        self.max_rounds = int(config["max_rounds"])
        if int(config.get("window", 0)):
            raise ValueError("only the full-history actor (window 0) is supported")
        if config.get("response_mode", "none") == "explicit":
            raise ValueError("the explicit response bridge is not supported")
        if config.get("style_input"):
            raise ValueError("style-input actors are not supported")

    @staticmethod
    def load(path: object) -> "NumpyHistoryActor":
        with np.load(path, allow_pickle=False) as z:
            config = json.loads(str(z["__config__"]))
            weights = {k: z[k] for k in z.files if not k.startswith("__")}
        return NumpyHistoryActor(weights, config)

    def _linear(self, x: np.ndarray, name: str) -> np.ndarray:
        return x.dot(self.w[name + ".weight"].T) + self.w[name + ".bias"]

    def _mlp(self, x: np.ndarray, name: str, layers: int) -> np.ndarray:
        for i in range(layers):
            x = _relu(self._linear(x, "%s.%d" % (name, 2 * i)))
        return x

    def _self_attention(self, x: np.ndarray, prefix: str) -> np.ndarray:
        """Causal multi-head self-attention, x [T, w]."""
        length, width = x.shape
        depth = width // self.heads
        qkv = x.dot(self.w[prefix + ".in_proj_weight"].T) + self.w[prefix + ".in_proj_bias"]
        q, k, v = (qkv[:, i * width:(i + 1) * width].reshape(length, self.heads, depth)
                   .transpose(1, 0, 2) for i in range(3))
        scores = np.matmul(q, k.transpose(0, 2, 1)) / np.float32(math.sqrt(depth))
        future = np.triu(np.ones((length, length), dtype=bool), 1)
        scores = np.where(future[None], np.float32(-np.inf), scores)
        out = np.matmul(_softmax(scores), v).transpose(1, 0, 2).reshape(length, width)
        return self._linear(out, prefix + ".out_proj")

    def encode_stream(self, tokens: np.ndarray, rounds: np.ndarray, phases: np.ndarray
                      ) -> np.ndarray:
        """``[T + 1, width]``: BOS then one position per public token."""
        w = self.w
        tokens = np.asarray(tokens, dtype=np.float32).reshape(-1, TOKEN_DIM)
        rounds = np.clip(np.asarray(rounds, dtype=np.int64), 0, self.max_rounds - 1)
        phases = np.clip(np.asarray(phases, dtype=np.int64), 0, 3)
        embedded = (self._linear(tokens, "public") + w["round_embedding.weight"][rounds]
                    + w["phase_embedding.weight"][phases])
        x = np.concatenate([w["bos"].reshape(1, self.width), embedded], 0)
        x = x + sinusoidal(len(x), self.width)
        for i in range(self.layers):
            p = "stream.layers.%d" % i
            x = x + self._self_attention(_layer_norm(x, w[p + ".norm1.weight"],
                                                     w[p + ".norm1.bias"]), p + ".self_attn")
            h = _layer_norm(x, w[p + ".norm2.weight"], w[p + ".norm2.bias"])
            x = x + self._linear(_relu(self._linear(h, p + ".linear1")), p + ".linear2")
        return _layer_norm(x, w["stream_norm.weight"], w["stream_norm.bias"])

    def decision_state(self, encoded: np.ndarray, obs: np.ndarray, seat: int) -> np.ndarray:
        """The private query attending to every encoded position, ``[width]``."""
        w = self.w
        width, heads = self.width, self.heads
        depth = width // heads
        query = (self._mlp(np.asarray(obs, dtype=np.float32)[None], "private", 2)
                 + w["seat.weight"][int(seat)][None])
        kv = self._linear(encoded, "kv_proj")
        keys = kv[:, :width].reshape(-1, heads, depth).transpose(1, 0, 2)
        values = kv[:, width:].reshape(-1, heads, depth).transpose(1, 0, 2)
        q = self._linear(query, "q_proj").reshape(heads, 1, depth)
        scores = np.matmul(q, keys.transpose(0, 2, 1)) / np.float32(math.sqrt(depth))
        attended = np.matmul(_softmax(scores), values).reshape(1, width)
        attended = self._linear(attended, "out_proj")
        state = _layer_norm(query + attended, w["attention_norm.weight"], w["attention_norm.bias"])
        hidden = self._linear(_relu(self._linear(state, "feed_forward.0")), "feed_forward.2")
        return _layer_norm(state + hidden, w["output_norm.weight"], w["output_norm.bias"])[0]

    def candidate_logits(self, state: np.ndarray, cand: np.ndarray) -> np.ndarray:
        actions = self._mlp(np.asarray(cand, dtype=np.float32), "action_tower", 2)
        fused = np.concatenate([np.repeat(state[None], len(actions), 0), actions], 1)
        return self._linear(self._mlp(fused, "fusion.0", 2), "fusion.1")[:, 0]

    def logits(self, tokens: np.ndarray, rounds: np.ndarray, phases: np.ndarray,
               obs: np.ndarray, seat: int, cand: np.ndarray) -> np.ndarray:
        """One logit per candidate; the greedy policy takes the argmax."""
        encoded = self.encode_stream(tokens, rounds, phases)
        return self.candidate_logits(self.decision_state(encoded, obs, seat), cand)
