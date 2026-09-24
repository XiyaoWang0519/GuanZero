"""Stage B PPO learner (DESIGN 8.4, STAGE_B_TODO B5).

One iteration ("update") is a rollout of `rollout_steps` vector steps followed
by `epochs` passes of clipped-surrogate PPO over the rounds that completed.

Rollout. A `gd.VecEnv` in which the learner controls both seats of team
`e % 2` in environment `e`. Rows of the other team go to an `OpponentSource`
(`train/opponents.py`), driven in the contract's order after every
`pending()`: drain finished rounds and report ended matches, report started
matches (changed `match_id`), and only then build `OpponentRows` from the
batch. Learner play rows are sampled by `StageBPolicy.act` over the frozen-M1
top-k-plus-pass set and stored in the B3 `RolloutBuffer` with their behaviour
log-probability, the PRUNED candidate list and the pruned index, so the
learner re-evaluates exactly the set that was sampled from. Tribute and
back-tribute rows follow `tribute_policy`: "heuristic" (default, A2 decision)
plays the engine heuristic and stores nothing, "learned" samples them from the
tribute heads and stores them as ordinary trajectory steps.

Learning. The perfect-information critic (B2) is refreshed on every completed
step with the current critic weights, then GAE with discount `gamma` and
`gae_lambda` over whole rounds, the round-end team return as the only reward.
Policy loss: clipped surrogate, minus `entropy_coef` times entropy, plus
`kl_coef` (linearly annealed to zero over `kl_anneal_updates`) times
KL(pi || pi_M1) on the pruned set, where pi_M1 = softmax(Q_M1 / temperature),
plus the DMC auxiliary hidden-hand and finish losses on the policy network.
The critic is trained jointly on MSE to the GAE returns with its own optimizer.

League (B8). `opponent = "league:<pool.json>"` builds a `train.league.League`
from the pool file (network paths resolved like other artifacts, sampler
seeded from the trainer RNG). Every `snapshot_every` updates (pool config, or
`league_snapshot_every`) the policy is written to an immutable
`league/update-<n>.pt` and added with `League.add_snapshot`, and a checkpoint
follows at once. Metrics carry `League.stats()`; checkpoints carry the league
state (entries, EMAs, tallies, snapshots, sampler RNG) and a resume continues
the same pool. League entries that are the frozen M1 itself at argmax are
"external": their play rows go through the rollout's shared reference forward
(the B5b fusion) instead of a second copy of the network; every other network
entry runs its own batched forward inside the league. With actor processes
each actor has its own league and the learner keeps one weight table; see
`train/ppo_actors.py` for the semantics.

Warm start (B8). `warm_start = <ppo checkpoint>` initialises the policy net,
the critic and (by default) both Adam states from an earlier Stage B run,
with fresh counters and environments; the frozen reference is still
`init_checkpoint`, and the source must share it (reference digest). The source
path and its policy weights digest are recorded in the runtime metadata and
every checkpoint, and survive resumes.

Exploiters (B11, DESIGN 8.4 and 9.4). A league run and an exploiter run
cooperate through files, on one host. The league run sets
`league_import_dir`: after every update it imports each new
`exploiter-*.pt` there into its league (`League.import_new`). The exploiter
run plays `opponent = "follow:<league run>/league"`, i.e. always the league
learner's newest snapshot (`FollowOpponent`; until the first one exists, its
own `warm_start`, which should be the league run's). With
`exploiter_publish_dir` set, every `exploiter_cycle_updates` updates, or
earlier once its match win rate over the last `exploiter_window_updates`
updates reaches `exploiter_win_rate` (after `exploiter_min_updates`), it
writes its policy to `exploiter-<n>-u<update>.pt` there, then resets its
policy to the newest league snapshot with a fresh policy optimizer (the
critic and its optimizer carry on) and starts the next cycle. With
`exploiter_pin_target` its opponent stays on one snapshot for the whole cycle
and moves to the one it reset to. Its training
match win rate is the league learner's live exploitability.

Checkpoints use the `dmc.py` layout (`latest.pt`, hard-linked snapshots in
`checkpoints/`, `metrics.jsonl`, `config.json`, `runtime-*.json`) and write a
Stage B payload (`stage="ppo"`) that `eval.policies.load_policy` plays. As in
`dmc.py`, environments and in-progress rounds restart on resume.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import json
import math
import os
import platform
from pathlib import Path
import random
import resource
import signal
import sys
import time
from typing import Any, Iterator, Sequence
import warnings

import numpy as np
import torch
from torch.nn import functional as F

import gd
from train.ckpt import load_checkpoint, restore_rng, rng_state, save_checkpoint
from train.critic import Critic, CriticConfig, load_critic
from train.dmc import schedule, tensor
from train.model import GuandanModel, ModelConfig, select_actions
from train.opponents import (FollowOpponent, FrozenModelOpponent, OpponentRows, OpponentSource,
                             config_opponent)
from train.policy import (PolicyConfig, StageBPolicy, gather_segments, policy_from_payload,
                          segment_entropy, segment_log_softmax, segment_rows)
from train.rollout_buffer import (HIDDEN_DIM, RolloutBuffer, RolloutBufferConfig, _ragged_index,
                                  to_device)

PLAY = int(gd.Phase.Play)
LEAGUE = "league:"
FOLLOW = "follow:"
M1_FINAL = ".work/runpod/artifacts/pilot/final.pt"
CRITIC_FIT = ".work/critic-fit/perfect/best.pt"


@dataclass
class PPOConfig:
    seed: int = 7
    # Initial policy (a Stage A/A2 checkpoint) and the frozen pruning/KL reference.
    init_checkpoint: str = M1_FINAL
    # B2 critic weights; empty string starts a fresh critic of critic_width x critic_layers.
    critic_init: str = CRITIC_FIT
    critic_width: int = 512
    critic_layers: int = 4
    # Warm start (B8): a Stage B ("ppo") checkpoint whose policy net and critic
    # weights initialise this run, with fresh counters and environments. The
    # frozen pruning/KL reference stays init_checkpoint, and the source must
    # have been trained against that same reference (reference_checkpoint_id),
    # with the same model and critic configs, action_mode and tribute policy.
    # critic_init must then be "" (or the warm_start path itself).
    warm_start: str = ""
    # Also load the source's policy and critic Adam states (lr is reset to
    # policy_lr / critic_lr).
    warm_start_optimizers: bool = True
    # The source's temperature and top_k must equal this config's unless this
    # is set; a temperature change is then logged as a warning.
    warm_start_policy_override: bool = False
    # "frozen" (the init checkpoint, argmax Q), "frozen:<path>" (any checkpoint
    # load_policy plays, at argmax; B9 exploiters use it), "greedy", or
    # "league:<pool.json>", the B7 opponent league (train/league.py) with
    # learner snapshots every `snapshot_every` updates of the pool config.
    opponent: str = "frozen"
    # League only: overrides the pool's snapshot_every when positive.
    league_snapshot_every: int = 0
    # League only (B11): import exploiter-*.pt files published here.
    league_import_dir: str = ""
    # Exploiter mode (B11), with opponent "follow:<dir>": publish here and reset.
    exploiter_publish_dir: str = ""
    exploiter_cycle_updates: int = 500
    exploiter_min_updates: int = 100
    exploiter_window_updates: int = 50
    exploiter_win_rate: float = 0.0     # early publish threshold; 0 disables
    # Pin the followed snapshot for a whole cycle (in-process rollout only).
    exploiter_pin_target: bool = False
    tribute_policy: str = "heuristic"
    action_mode: str = "canonical"
    num_envs: int = 1024
    num_threads: int = 8
    torch_threads: int = 4
    rollout_steps: int = 128
    epochs: int = 4
    minibatch_size: int = 4096
    clip: float = 0.2
    policy_lr: float = 1e-5
    critic_lr: float = 1e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    entropy_coef: float = 0.01
    kl_coef: float = 0.1
    kl_anneal_updates: int = 1000
    target_kl: float = 0.0         # stop the epochs early past this approx KL; 0 disables
    value_coef: float = 1.0
    hidden_weight: float = 0.1
    finish_weight: float = 0.1
    grad_clip: float = 10.0
    temperature: float = 0.02      # measured: M1 sampled at 0.02 is within noise of argmax M1 (B5 report)
    top_k: int = 32
    candidate_chunk: int = 32768
    # Rollout buffer capacities; 0 derives them from num_envs and rollout_steps.
    buffer_steps: int = 0
    buffer_candidates: int = 0
    buffer_trajectories: int = 0
    checkpoint_seconds: float = 300
    snapshot_updates: int = 100
    max_updates: int = 100000
    max_seconds: float = 86400
    tensorboard: bool = True
    # Throughput path (B5b): one upload of the batch per step, one frozen-M1
    # forward shared by pruning and the default frozen opponent, no device
    # synchronisation inside a step or a minibatch except the pruning's
    # nonzero and the one copy of results back. False keeps the B5 path
    # (separate opponent call, per-call uploads); both sample the same
    # actions under a seed, see tests/test_ppo_throughput.py.
    fast_rollout: bool = True
    # 0 rolls out in this process. W > 0 steps num_envs / W environments in
    # each of W actor processes (train/ppo_actors.py), each with num_threads
    # engine threads and torch_threads torch threads, synchronously: actors
    # wait while the learner trains, so actors add no policy lag (carried-over
    # rounds still hold steps from the previous weights). Needs a
    # config opponent (frozen, frozen:<path>, greedy, league:<pool.json>);
    # the league's semantics across actors are in train/ppo_actors.py.
    actor_processes: int = 0

    def validate(self) -> None:
        positive = ("num_envs", "num_threads", "torch_threads", "rollout_steps", "epochs",
                    "minibatch_size", "clip", "kl_anneal_updates", "grad_clip", "temperature",
                    "top_k", "candidate_chunk", "checkpoint_seconds", "snapshot_updates",
                    "max_updates", "max_seconds", "critic_width", "critic_layers",
                    "exploiter_cycle_updates", "exploiter_min_updates",
                    "exploiter_window_updates")
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        nonnegative = ("policy_lr", "critic_lr", "entropy_coef", "kl_coef", "target_kl",
                       "value_coef", "hidden_weight", "finish_weight", "buffer_steps",
                       "buffer_candidates", "buffer_trajectories", "actor_processes",
                       "league_snapshot_every", "exploiter_win_rate")
        for name in nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if self.num_envs < 2:
            raise ValueError("num_envs must be at least 2 so both learner teams are covered")
        if self.clip >= 1:
            raise ValueError("clip must be in (0, 1)")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma must be in (0, 1] and gae_lambda in [0, 1]")
        if self.checkpoint_seconds > 600:
            raise ValueError("checkpoint_seconds must be at most 600")
        if self.tribute_policy not in ("heuristic", "learned"):
            raise ValueError("tribute_policy must be heuristic or learned")
        if self.action_mode not in ("canonical", "full"):
            raise ValueError("action_mode must be canonical or full")
        if not (self.opponent in ("frozen", "greedy")
                or (self.opponent.startswith("frozen:") and len(self.opponent) > 7)
                or (self.opponent.startswith(FOLLOW) and len(self.opponent) > len(FOLLOW))
                or (self.opponent.startswith(LEAGUE) and len(self.opponent) > len(LEAGUE))):
            raise ValueError("opponent must be frozen, frozen:<path>, follow:<dir>, greedy "
                             "or league:<pool.json>")
        if self.opponent.startswith(FOLLOW) and not self.warm_start:
            raise ValueError("a follow: opponent plays warm_start until the first snapshot")
        if self.exploiter_publish_dir and not self.opponent.startswith(FOLLOW):
            raise ValueError("exploiter_publish_dir needs a follow:<dir> opponent")
        if self.league_import_dir and not self.opponent.startswith(LEAGUE):
            raise ValueError("league_import_dir needs a league:<pool.json> opponent")
        if self.exploiter_win_rate > 1:
            raise ValueError("exploiter_win_rate must be at most 1")
        if self.exploiter_pin_target and (self.actor_processes or not self.exploiter_publish_dir):
            raise ValueError("exploiter_pin_target needs exploiter mode and in-process rollout")
        if not self.init_checkpoint:
            raise ValueError("init_checkpoint is required")
        if self.warm_start and self.critic_init and self.critic_init != self.warm_start:
            raise ValueError("warm_start supplies the critic: set critic_init to \"\" "
                             "(or to the warm_start path)")
        if self.actor_processes and (self.num_envs % self.actor_processes
                                     or self.num_envs // self.actor_processes < 2):
            raise ValueError("num_envs must split into actor_processes shards of at least 2")

    def buffer_config(self) -> RolloutBufferConfig:
        # About half of all rows are learner rows; a round in progress at the
        # iteration boundary is carried over, so leave room for one per team.
        steps = self.buffer_steps or self.num_envs * (self.rollout_steps + 128)
        return RolloutBufferConfig(
            num_envs=self.num_envs, obs_dim=gd.OBS_DIM, act_dim=gd.ACT_DIM,
            max_steps=steps,
            max_candidates=self.buffer_candidates or steps * (self.top_k + 1),
            max_trajectories=self.buffer_trajectories or self.num_envs * 2 * (self.rollout_steps // 8 + 2),
            carry_over=True)


# Fields that may change on resume: runtime limits and resources only.
MUTABLE_ON_RESUME = {"max_updates", "max_seconds", "checkpoint_seconds", "snapshot_updates",
                     "tensorboard", "torch_threads", "num_threads", "init_checkpoint", "critic_init",
                     "fast_rollout", "actor_processes"}


def resolve_artifact(path: str | Path) -> Path:
    """A relative `.work/...` path also resolves in the main checkout when this
    runs from a git worktree below it."""
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    for parent in Path(__file__).resolve().parents:
        if (parent / path).exists():
            return parent / path
    raise FileNotFoundError(f"{path} not found here or in any parent checkout")


def resolve_policy_spec(spec: str) -> str:
    """A league entry spec with its checkpoint path (if any) made absolute via
    `resolve_artifact`, so actor processes and worktrees load the same file."""
    from train.league import entry_kind

    if entry_kind(spec) != "network":
        return spec
    head, path = "", spec
    if spec.startswith("sample:") or spec.startswith("sample="):
        head, _, path = spec.partition(":")
        head += ":"
    return head + str(resolve_artifact(path))


def league_setup(pool: str | Path, device: torch.device) -> tuple[Any, list[dict[str, Any]]]:
    """LeagueConfig and resolved entry dicts of a pool file, for this device.
    Entry names default to the spec as written in the file."""
    from train.league import load_pool

    league_config, raw = load_pool(resolve_artifact(pool))
    entries = [{"spec": resolve_policy_spec(item["spec"]),
                "name": item.get("name") or item["spec"],
                "weight": float(item.get("weight", 1.0))} for item in raw]
    return replace(league_config, device=str(device)), entries


def segment_sum(values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    rows = segment_rows(offsets, len(values))
    return values.new_zeros(len(offsets) - 1).index_add(0, rows, values)


def policy_terms(policy: StageBPolicy, obs: torch.Tensor, cand: torch.Tensor,
                 offsets: torch.Tensor, phase: torch.Tensor, chosen: torch.Tensor,
                 phase_code: int | None, state: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`StageBPolicy.evaluate` plus the full pruned log-probabilities (for the
    KL term) from one forward pass. Returns (log_prob, entropy, log_probs)."""
    log_probs = segment_log_softmax(policy.logits(obs, cand, offsets, phase, phase_code,
                                                  state=state), offsets)
    return (gather_segments(log_probs, offsets, chosen),
            segment_entropy(log_probs, offsets), log_probs)


def reference_kl(policy: StageBPolicy, log_probs: torch.Tensor, obs: torch.Tensor,
                 cand: torch.Tensor, offsets: torch.Tensor, phase: torch.Tensor,
                 phase_code: int | None) -> torch.Tensor:
    """Per-decision KL(pi || softmax(Q_ref / t)) over the stored pruned set."""
    with torch.no_grad():
        ref = policy.reference.score_candidates(obs, cand, offsets, phase,
                                                chunk_size=policy.config.chunk_size,
                                                phase_code=phase_code)
        ref_log_probs = segment_log_softmax(ref / policy.config.temperature, offsets)
    return segment_sum(log_probs.exp() * (log_probs - ref_log_probs), offsets)


def cached_reference_kl(log_probs: torch.Tensor, ref_log_probs: torch.Tensor,
                        offsets: torch.Tensor) -> torch.Tensor:
    """`reference_kl` from the reference log-probabilities stored at rollout
    time. The reference is frozen and the stored pruned set is exactly what it
    scored, so this equals the recomputed value up to float reassociation."""
    return segment_sum(log_probs.exp() * (log_probs - ref_log_probs), offsets)


def concat_minibatches(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Join `RolloutBuffer.gather` results of several buffers into one batch
    (ragged offsets rebased; `steps` are then per-buffer indices)."""
    if len(parts) == 1:
        return parts[0]
    joined = {key: torch.cat([part[key] for part in parts]) for key in parts[0]
              if key != "offsets"}
    offsets, base = [parts[0]["offsets"][:1]], 0
    for part in parts:
        offsets.append(part["offsets"][1:] + base)
        base += len(part["cand"])
    joined["offsets"] = torch.cat(offsets)
    return joined


def _host_view(array: np.ndarray) -> torch.Tensor:
    """Zero-copy tensor over a read-only engine buffer. Callers only read it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.from_numpy(array)


class Uploader:
    """Per-step host-to-device transfer of the pending batch.

    On CPU the engine buffers are used in place. On CUDA every array is
    narrowed to its wire dtype (the encoder's features are 0/1, exact in
    uint8) while being copied into a reusable pinned buffer, sent
    asynchronously and widened on the device. A pinned buffer is rewritten
    only on the next step, after the step's results were copied back, which
    synchronises the stream, so an in-flight copy is never overwritten.
    Buffers grow geometrically and are then reused: no allocation in steady
    state.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.pinned: dict[str, torch.Tensor] = {}
        # The current host source for each upload. Rollout storage can gather
        # from the already narrowed CUDA copy instead of recopying float32
        # engine features. These views last only until that name is uploaded
        # again (and CPU views only until the engine advances).
        self.host_arrays: dict[str, np.ndarray] = {}

    def __call__(self, name: str, array: np.ndarray, wire: torch.dtype,
                 dtype: torch.dtype) -> torch.Tensor:
        source = _host_view(np.ascontiguousarray(array))
        if self.device.type != "cuda":
            self.host_arrays[name] = array
            return source if source.dtype == dtype else source.to(dtype)
        size = source.numel()
        buffer = self.pinned.get(name)
        if buffer is None or buffer.numel() < size:
            capacity = max(size, 0 if buffer is None else buffer.numel() * 3 // 2, 1)
            buffer = torch.empty(capacity, dtype=wire, pin_memory=True)
            self.pinned[name] = buffer
        staged = buffer[:size].view(source.shape)
        staged.copy_(source)
        self.host_arrays[name] = staged.numpy()
        return staged.to(self.device, non_blocking=True).to(dtype)


def shares_opponent_forward(config: PPOConfig, phase_code: int | None,
                            opponent: OpponentSource | None, fused_opponent: bool) -> bool:
    """Whether the rollout plays the opponent through `act_split`, scoring the
    rows of its models that share the learner's frozen reference in its own
    reference forward (`RolloutCollector._fast_act`). Needs the fast path and
    heuristic tribute, where that forward computes what the model's own
    reference would, up to the float rounding of a differently sized batch
    (bitwise equal on CPU for batches of two or more rows)."""
    return (config.fast_rollout and phase_code == PLAY and not fused_opponent
            and hasattr(opponent, "act_split"))


def to_host(device: torch.device, *values: torch.Tensor) -> list[np.ndarray]:
    """Copy tensors back with a single synchronisation."""
    if device.type != "cuda":
        return [value.numpy() for value in values]
    host = [value.to("cpu", non_blocking=True) for value in values]
    torch.cuda.current_stream(device).synchronize()
    return [value.numpy() for value in host]


class RolloutCollector:
    """The rollout half of the trainer: vector steps into the round buffer.

    Shared by `PPOTrainer` (in-process rollout) and the actor processes of
    `train/ppo_actors.py`. Needs `config`, `device`, `policy`, `env`,
    `learner_team`, `env_match`, `buffer`, `opponent`, `fused_opponent`,
    `league_fused` (the opponent is a `League` whose `external` specs are the
    frozen reference), `shared_opponent` (see `shares_opponent_forward`),
    `upload`, `timers`, `generator`, `progress`, `window`, `phase_code`,
    `bound` and `should_stop()`.
    """

    def _finish_rounds(self, results) -> None:
        ended, won = [], []
        for result in results:
            env = int(result.env_id)
            team = int(self.learner_team[env])
            self.buffer.finish_round(env, int(result.match_id), int(result.round_index),
                                     result.seat_return, result.order)
            self.progress["rounds"] += 1
            self.window["rounds"] += 1
            self.window["learner_return"] += float(result.seat_return[team])
            if result.match_winner >= 0:
                learner_won = int(result.match_winner) == team
                ended.append(env)
                won.append(learner_won)
                self.progress["matches"] += 1
                self.progress["learner_match_wins"] += learner_won
                self.window["matches"] += 1
                self.window["wins"] += learner_won
        if ended:
            self.opponent.on_match_end(np.asarray(ended, np.int32), np.asarray(won, bool))

    def _start_matches(self, env_id: np.ndarray, match_id: np.ndarray) -> None:
        if not self.bound:
            # Contract: every environment starts a match right after bind.
            self.env_match.fill(-1)
            self.env_match[env_id] = match_id
            self.opponent.on_match_start(np.arange(len(self.env_match), dtype=np.int32))
            self.bound = True
            return
        changed = match_id != self.env_match[env_id]
        if changed.any():
            envs = np.unique(env_id[changed])
            self.env_match[env_id[changed]] = match_id[changed]
            self.opponent.on_match_start(envs.astype(np.int32))

    @staticmethod
    def _ragged_rows(batch, rows: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Candidates and rebased offsets of a subset of batch rows."""
        if rows.size == len(offsets) - 1:
            return np.asarray(batch.cand), offsets
        counts = offsets[rows + 1] - offsets[rows]
        cand = np.asarray(batch.cand)[_ragged_index(offsets[rows], counts, int(counts.sum()))]
        local = np.zeros(rows.size + 1, np.int64)
        np.cumsum(counts, out=local[1:])
        return cand, local

    def _opponent_rows(self, batch, rows: np.ndarray, offsets: np.ndarray,
                       env_id: np.ndarray, seat: np.ndarray, phase: np.ndarray,
                       match_id: np.ndarray) -> OpponentRows:
        cand, local = self._ragged_rows(batch, rows, offsets)
        # styled_choice is read only now, after on_match_start (set_styles may
        # have recomputed it for rows already pending).
        return OpponentRows(
            obs=np.asarray(batch.obs)[rows], cand=cand, offsets=local.astype(np.int32),
            env_id=env_id[rows], seat=seat[rows], phase=phase[rows], match_id=match_id[rows],
            greedy_choice=np.asarray(batch.greedy_choice)[rows],
            styled_choice=np.asarray(batch.styled_choice)[rows])

    def _learner_act(self, batch, rows: np.ndarray, offsets: np.ndarray, env_id: np.ndarray,
                     seat: np.ndarray, phase: np.ndarray, match_id: np.ndarray,
                     choices: np.ndarray) -> None:
        cand, local = self._ragged_rows(batch, rows, offsets)
        obs = np.asarray(batch.obs)[rows]
        with torch.inference_mode():
            step = self.policy.act(tensor(obs, self.device, torch.float32),
                                   tensor(cand, self.device, torch.float32),
                                   tensor(local, self.device, torch.long),
                                   tensor(phase[rows], self.device, torch.long),
                                   generator=self.generator, phase_code=self.phase_code)
        keep = step.keep_index.cpu().numpy()
        choices[rows] = step.choice.cpu().numpy()
        self.buffer.add_batch(
            learner=np.ones(rows.size, bool), env_id=env_id[rows], match_id=match_id[rows],
            round_index=np.asarray(batch.round_index)[rows], seat=seat[rows], phase=phase[rows],
            obs=obs, hidden_counts=np.asarray(batch.hidden_counts)[rows],
            cand=cand[keep], offsets=step.pruned_offsets.cpu().numpy(),
            chosen=step.pruned_choice.cpu().numpy(), logp=step.log_prob.float().cpu().numpy(),
            ref_logp=step.ref_log_probs.float().cpu().numpy())
        self.progress["learner_decisions"] += int(rows.size)

    def _shares_reference(self, model) -> bool:
        """An opponent's Stage B model whose pruning reference has the learner's
        frozen reference weights (same digest), on this device, playing play
        rows only: the rollout's shared reference forward can score its rows,
        after which it only prunes and runs its own policy network."""
        pruned = getattr(model, "pruned", None)
        reference = self.policy.reference_checkpoint_id
        return (pruned is not None and reference is not None
                and bool(getattr(model, "heuristic_tribute", False))
                and pruned.reference_checkpoint_id == reference
                and getattr(model, "device", None) == self.device)

    def _fast_act(self, batch, learner_rows: np.ndarray, opponent_rows: np.ndarray,
                  offsets: np.ndarray, env_id: np.ndarray, seat: np.ndarray,
                  phase: np.ndarray, match_id: np.ndarray, choices: np.ndarray,
                  shared: Sequence[tuple[Any, np.ndarray]] = ()) -> None:
        """Learner sampling, the fused frozen opponent's play-row argmax
        (`opponent_rows`) and the play rows of every `shared` (model, rows)
        opponent group (see `_shares_reference`), from ONE reference forward
        over all of those rows.

        Rows are laid out learner first, then the argmax rows, then each shared
        group, so each part is a contiguous slice of the scored candidates and
        nothing on the device needs a data-dependent shape except the pruning.
        Results come back in one copy."""
        rows = np.concatenate((learner_rows, opponent_rows, *(g for _, g in shared)))
        if not rows.size:
            return
        counts = offsets[rows + 1] - offsets[rows]
        local = np.zeros(rows.size + 1, np.int64)
        np.cumsum(counts, out=local[1:])
        src = _ragged_index(offsets[rows], counts, int(local[-1]))
        n_learner = learner_rows.size
        n_fused = n_learner + opponent_rows.size
        c_learner = int(local[n_learner])
        c_fused = int(local[n_fused])
        with self._phase("upload"):
            obs_all = self.upload("obs", batch.obs, torch.uint8, torch.float32)
            cand_all = self.upload("cand", batch.cand, torch.uint8, torch.float32)
            index = self.upload("index", np.concatenate((rows, src)), torch.int64, torch.long)
            local_t = self.upload("offsets", local, torch.int64, torch.long)
            phase_t = self.upload("phase", phase[rows], torch.int64, torch.long)
            obs = obs_all.view(-1, gd.OBS_DIM).index_select(0, index[:rows.size])
            cand = cand_all.view(-1, gd.ACT_DIM).index_select(0, index[rows.size:])
        policy = self.policy
        with torch.inference_mode():
            with self._phase("reference forward"):
                ref = policy.reference.score_candidates(
                    obs, cand, local_t, phase_t, chunk_size=policy.config.chunk_size,
                    phase_code=self.phase_code)
            outputs = [torch.isfinite(ref).all()]
            if n_learner:
                with self._phase("prune + policy forward + sample"):
                    step = policy.act(obs[:n_learner], cand[:c_learner],
                                      local_t[:n_learner + 1], phase_t[:n_learner],
                                      generator=self.generator, phase_code=self.phase_code,
                                      ref_scores=ref[:c_learner], sync_checks=False)
                outputs += [step.choice, step.pruned_choice, step.log_prob.float(),
                            step.keep_index, step.pruned_offsets, step.ref_log_probs.float()]
            if opponent_rows.size:
                with self._phase("opponent argmax"):
                    outputs.append(select_actions(ref[c_learner:c_fused],
                                                  local_t[n_learner:n_fused + 1] - c_learner))
            begin = n_fused
            for model, group in shared:
                end = begin + group.size
                c0, c1 = int(local[begin]), int(local[end])
                with self._phase("shared opponent prune + forward"):
                    choice, _ = model.pruned.choose(
                        obs[begin:end], cand[c0:c1], local_t[begin:end + 1] - c0,
                        phase_t[begin:end], generator=model.generator,
                        greedy=not model.sample, phase_code=self.phase_code,
                        ref_scores=ref[c0:c1])
                outputs.append(choice)
                begin = end
            with self._phase("device to host"):
                host = to_host(self.device, *outputs)
        if not bool(host[0]):
            raise FloatingPointError("non-finite reference scores")
        for (_, group), choice in zip(shared, host[len(host) - len(shared):]):
            choices[group] = choice
        if opponent_rows.size:
            choices[opponent_rows] = host[len(host) - len(shared) - 1]
        if not n_learner:
            return
        choice, pruned_choice, log_prob, keep, pruned_offsets, ref_logp = host[1:7]
        choices[learner_rows] = choice
        with self._phase("buffer add"):
            self.buffer.add_batch(
                learner=np.ones(n_learner, bool), env_id=env_id[learner_rows],
                match_id=match_id[learner_rows],
                round_index=np.asarray(batch.round_index)[learner_rows],
                seat=seat[learner_rows], phase=phase[learner_rows],
                obs=self.upload.host_arrays["obs"], obs_index=learner_rows,
                hidden_counts=np.asarray(batch.hidden_counts)[learner_rows],
                cand=self.upload.host_arrays["cand"], cand_index=src[keep],
                offsets=pruned_offsets, chosen=pruned_choice, logp=log_prob, ref_logp=ref_logp)
        self.progress["learner_decisions"] += n_learner

    @contextmanager
    def _phase(self, name: str):
        """Accumulate wall time per rollout/learner phase when `timers` is set
        (bench only); synchronises CUDA at both ends so time is attributed to
        the phase that spent it. A no-op otherwise."""
        if self.timers is None:
            yield
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.timers[name] = self.timers.get(name, 0.0) + time.perf_counter() - start

    def collect(self) -> None:
        self.net.eval()
        heuristic = self.config.tribute_policy == "heuristic"
        for _ in range(self.config.rollout_steps):
            if self.should_stop():
                break
            with self._phase("env pending (C++)"):
                batch = self.env.pending()
            with self._phase("round bookkeeping"):
                # Contract order: ended matches, then started matches, then rows.
                self._finish_rounds(self.env.drain_finished_rounds())
                n = batch.rows
                if not n:
                    raise RuntimeError("environment produced no pending decisions")
                env_id = np.asarray(batch.env_id, np.int64)
                match_id = np.asarray(batch.match_id, np.int64)
                self._start_matches(env_id, match_id)
                seat = np.asarray(batch.seat, np.int64)
                phase = np.asarray(batch.phase, np.int64)
                offsets = np.asarray(batch.offsets, np.int64)
                learner = seat % 2 == self.learner_team[env_id]
                choices = np.array(batch.greedy_choice, dtype=np.int32, copy=True)
                opponent_rows = np.flatnonzero(~learner)
                acting = learner & (phase == PLAY) if heuristic else learner
                learner_rows = np.flatnonzero(acting)
            if self.fused_opponent:
                # Tribute rows keep greedy_choice, as FrozenModelOpponent.act does.
                opponent_play = opponent_rows[phase[opponent_rows] == PLAY]
                self._fast_act(batch, learner_rows, opponent_play, offsets, env_id, seat,
                               phase, match_id, choices)
            else:
                fused_rows = opponent_rows[:0]
                if self.league_fused and opponent_rows.size:
                    # League matches against the frozen M1 itself: its play
                    # rows join the shared reference forward (argmax, as the
                    # league's own model of that checkpoint would play), its
                    # tribute rows keep greedy_choice (heuristic tribute).
                    external = self.opponent.external_rows(env_id[opponent_rows])
                    if external.any():
                        fused = opponent_rows[external]
                        fused_rows = fused[phase[fused] == PLAY]
                        opponent_rows = opponent_rows[~external]
                shared = []
                if opponent_rows.size:
                    with self._phase("opponent act"):
                        rows_in = self._opponent_rows(batch, opponent_rows, offsets, env_id,
                                                      seat, phase, match_id)
                        if self.shared_opponent:
                            # Groups of models sharing the frozen reference
                            # come back unplayed; _fast_act scores them.
                            picked, deferred = self.opponent.act_split(
                                rows_in, self._shares_reference)
                            shared = [(model, opponent_rows[index]) for model, index in deferred]
                        else:
                            picked = self.opponent.act(rows_in)
                        picked = np.asarray(picked, np.int32)
                    if picked.shape != opponent_rows.shape:
                        raise ValueError("opponent returned the wrong number of choices")
                    choices[opponent_rows] = picked
                if learner_rows.size or fused_rows.size or shared:
                    if self.config.fast_rollout:
                        self._fast_act(batch, learner_rows, fused_rows, offsets, env_id,
                                       seat, phase, match_id, choices, shared)
                    else:
                        with self._phase("learner act (B5 path)"):
                            self._learner_act(batch, learner_rows, offsets, env_id, seat,
                                              phase, match_id, choices)
            counts = offsets[1:] - offsets[:-1]
            if ((choices < 0) | (choices >= counts)).any():
                raise ValueError("a choice lies outside its candidate list")
            with self._phase("env step (C++)"):
                self.env.step(choices)
            self.progress["decisions"] += n



class PPOTrainer(RolloutCollector):
    def __init__(self, config: PPOConfig, run_dir: str | Path, device: str = "cpu",
                 resume: str | Path | None = None,
                 opponent: OpponentSource | None = None) -> None:
        config.validate()
        self.config = config
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("training currently supports cpu or cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if (self.run_dir / "latest.pt").exists() and resume is None:
            raise ValueError("run directory already has a checkpoint; use --resume")
        torch.set_num_threads(config.torch_threads)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        policy_config = PolicyConfig(temperature=config.temperature, top_k=config.top_k,
                                     chunk_size=config.candidate_chunk)
        self.progress: dict[str, Any] = {
            "updates": 0, "optimizer_steps": 0, "decisions": 0, "learner_decisions": 0,
            "rounds": 0, "matches": 0, "learner_match_wins": 0, "samples": 0,
            "elapsed_seconds": 0.0, "resumes": 0, "snapshot_update": 0,
            "exploiter_cycle_start": 0, "exploiter_published": 0,
        }
        payload = None
        if resume:
            payload = load_checkpoint(resume, self.device)
            if payload.get("stage") != "ppo":
                raise ValueError("PPO resume requires a Stage B (ppo) checkpoint")
            defaults = asdict(PPOConfig())   # fields added after the checkpoint was written
            for key, value in asdict(config).items():
                if key not in MUTABLE_ON_RESUME and value != payload["config"].get(key,
                                                                                  defaults[key]):
                    raise ValueError(f"resume config differs at {key}")
            self.policy = policy_from_payload(payload, self.device)
            if self.policy.config != policy_config:
                raise ValueError("resume policy config differs")
            self.critic = Critic(CriticConfig(**payload["critic_model_config"]))
            self.critic.load_state_dict(payload["critic"])
            self.init_source = payload.get("init_source")
            self.critic_source = payload.get("critic_source")
            # A resumed run keeps the warm start it was started from.
            self.warm_start_source = payload.get("warm_start_source")
            self.warm_start_digest = payload.get("warm_start_digest")
        else:
            init_path = resolve_artifact(config.init_checkpoint)
            init = load_checkpoint(init_path, self.device)
            if init.get("stage", "dmc") not in ("dmc", "a2"):
                raise ValueError("Stage B starts from a Stage A or A2 checkpoint")
            if init.get("config", {}).get("action_mode", "canonical") != config.action_mode:
                raise ValueError("init checkpoint action_mode differs from the config")
            if config.tribute_policy == "learned" and init.get("tribute_policy") != "learned":
                raise ValueError("learned tribute starts from an A2 checkpoint's fitted heads")
            from eval.policies import model_digest
            model = GuandanModel(ModelConfig(**init["model_config"]))
            model.load_state_dict(init["model"])
            self.policy = StageBPolicy.from_model(model, policy_config,
                                                  model_digest(init["model"])).to(self.device)
            self.init_source = str(init_path)
            self.warm_start_source = self.warm_start_digest = None
            if config.warm_start:
                warm = self._warm_start(init)
                self.critic_source = self.warm_start_source
                self.critic = Critic(CriticConfig(**warm["critic_model_config"]))
                self.critic.load_state_dict(warm["critic"])
            elif config.critic_init:
                self.critic_source = str(resolve_artifact(config.critic_init))
                self.critic, _ = load_critic(self.critic_source, str(self.device))
            else:
                self.critic_source = None
                self.critic = Critic(CriticConfig(width=config.critic_width,
                                                  layers=config.critic_layers))
        self.net = self.policy.net
        self.critic.to(self.device)
        dims = (self.net.config.obs_dim, self.net.config.act_dim, self.critic.config.input_dim)
        if dims != (gd.OBS_DIM, gd.ACT_DIM, gd.OBS_DIM + HIDDEN_DIM):
            raise ValueError("policy or critic dimensions do not match the engine")
        self.policy_optimizer = torch.optim.Adam(self.net.parameters(), lr=config.policy_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.generator = torch.Generator(device=self.device)
        if payload is not None:
            self.policy_optimizer.load_state_dict(payload["optimizer"])
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
            self.progress.update(payload["progress"])
            restore_rng(payload["rng"], self.rng)
            self.generator.set_state(payload["sampler"].cpu())
            self.progress["resumes"] += 1
        else:
            if config.warm_start and config.warm_start_optimizers:
                self.policy_optimizer.load_state_dict(warm["optimizer"])
                self.critic_optimizer.load_state_dict(warm["critic_optimizer"])
                for optimizer, lr in ((self.policy_optimizer, config.policy_lr),
                                      (self.critic_optimizer, config.critic_lr)):
                    for group in optimizer.param_groups:
                        group["lr"] = lr
            self.generator.manual_seed(int(self.rng.integers(0, 2**63)))
        self.phase_code = PLAY if config.tribute_policy == "heuristic" else None
        metadata = {
            "python": platform.python_version(), "torch": torch.__version__,
            "device": str(self.device), "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "policy_parameters": sum(p.numel() for p in self.net.parameters()),
            "critic_parameters": sum(p.numel() for p in self.critic.parameters()),
            "reference_checkpoint_id": self.policy.reference_checkpoint_id,
            "init_source": self.init_source, "critic_source": self.critic_source,
            "warm_start_source": self.warm_start_source,
            "warm_start_digest": self.warm_start_digest,
            "resume_source": str(resume) if resume else None,
        }
        (self.run_dir / f"runtime-{self.progress['resumes']:03d}.json").write_text(
            json.dumps(metadata, indent=2) + "\n")
        actions = gd.ActionConfig.full() if config.action_mode == "full" else gd.ActionConfig()
        # As in dmc.py, environments restart on resume from a seed drawn from the restored RNG.
        self.actors = None
        self.window = {"matches": 0, "wins": 0, "rounds": 0, "learner_return": 0.0}
        # Exploiter mode: (wins, matches) per update of the current cycle.
        self.exploiter_window: list[tuple[int, int]] = []
        # The authoritative league (weights, snapshots, tallies): the opponent
        # itself in-process, an unbound copy fed by the actors otherwise.
        self.league = None
        self.league_entries: list[dict[str, Any]] = []
        self.league_external: frozenset[str] = frozenset()
        self.league_fused = False
        self.league_snapshot_taken = False
        self.actor_league_info: dict[str, float] = {}
        if config.opponent.startswith(LEAGUE):
            if opponent is not None:
                raise ValueError("pass either an opponent object or a league: config opponent")
            from train.league import League
            self.league_config, self.league_entries = league_setup(
                config.opponent[len(LEAGUE):], self.device)
            if config.league_snapshot_every:
                self.league_config = replace(self.league_config,
                                             snapshot_every=config.league_snapshot_every)
            self.league_external = self._fused_league_specs(self.league_entries)
            self.league_fused = bool(self.league_external)
            league_state = payload.get("league_state") if payload is not None else None
        if config.actor_processes:
            # Environments, opponents and shard buffers live in actor processes
            # (train/ppo_actors.py); the buffers are shared memory read here.
            if opponent is not None:
                raise ValueError("actor_processes builds opponents from config.opponent; "
                                 "an OpponentSource object needs the in-process rollout")
            from train.ppo_actors import ActorPool, shard_sizes
            self.env = self.buffer = self.opponent = None
            self.fused_opponent = False
            actor_rng = None
            if self.league_entries:
                self.league = League(self.league_entries, self.league_config)
                if league_state is not None:
                    self.league.load_state_dict(league_state)
                saved = payload.get("league_actor_rng") if payload is not None else None
                if saved is not None and len(saved) == config.actor_processes:
                    actor_rng = saved
            self.actors = ActorPool(self, shard_sizes(config.num_envs, config.actor_processes),
                                    league_rng=actor_rng)
            self.buffers = self.actors.buffers
        else:
            self.env = gd.VecEnv(config.num_envs, num_threads=config.num_threads,
                                 seed=int(self.rng.integers(0, 2**63)), actions=actions)
            self.env.reset()
            self.learner_team = np.arange(config.num_envs, dtype=np.int64) % 2
            self.env_match = np.full(config.num_envs, -1, np.int64)
            self.buffer = RolloutBuffer(config.buffer_config())
            self.buffers = [self.buffer]
            if self.league_entries:
                # Seeded from the trainer RNG, so PPOConfig.seed fixes the draws.
                self.league = League(self.league_entries, replace(
                    self.league_config, seed=int(self.rng.integers(0, 2**63))))
                self.league.external = self.league_external
                if league_state is not None:
                    self.league.load_state_dict(league_state)
                self.opponent = self.league
            else:
                self.opponent = opponent if opponent is not None else self._default_opponent()
            self.opponent.bind(self.env, self.learner_team)
            # FrozenModelOpponent.act is a pure function of its rows: argmax of
            # its network's play head on play rows, the heuristic otherwise.
            # When that network IS the frozen pruning reference, the fast path
            # scores both teams' rows in one reference forward and takes the
            # same argmax (bind/on_match_start/on_match_end are still called in
            # contract order).
            self.fused_opponent = (config.fast_rollout
                                   and isinstance(self.opponent, FrozenModelOpponent)
                                   and self.opponent.model is self.policy.reference)
        self.shared_opponent = shares_opponent_forward(config, self.phase_code, self.opponent,
                                                       self.fused_opponent)
        self.upload = Uploader(self.device)
        self.timers: dict[str, float] | None = None   # phase timers, see bench/ppo_throughput.py
        self.bound = False
        self.stop_requested = False
        self.started = self.last_save = time.monotonic()
        self.prior_elapsed = self.progress["elapsed_seconds"]
        self.prior_decisions = self.progress["decisions"]
        self.writer = None
        if config.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.run_dir / "tensorboard"))
        (self.run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")

    def _warm_start(self, init: dict[str, Any]) -> dict[str, Any]:
        """Validate `config.warm_start` against the init checkpoint and load its
        policy weights into `self.policy.net` (the reference stays the init
        model). Returns the source payload for the critic and optimizers."""
        from eval.policies import model_digest

        cfg = self.config
        path = resolve_artifact(cfg.warm_start)
        warm = load_checkpoint(path, self.device)
        if warm.get("stage") != "ppo":
            raise ValueError("warm_start must be a Stage B (ppo) checkpoint")
        source = policy_from_payload(warm, "cpu")   # also checks its reference digest
        if source.reference_checkpoint_id != self.policy.reference_checkpoint_id:
            raise ValueError("warm_start was trained against a different frozen reference "
                             "than init_checkpoint")
        if ModelConfig(**warm["model_config"]) != self.policy.net.config:
            raise ValueError("warm_start model config differs from init_checkpoint")
        critic = CriticConfig(**warm["critic_model_config"])
        if (critic.width, critic.layers) != (cfg.critic_width, cfg.critic_layers):
            raise ValueError("warm_start critic config differs from critic_width/critic_layers")
        if warm.get("config", {}).get("action_mode", "canonical") != cfg.action_mode:
            raise ValueError("warm_start action_mode differs from the config")
        if warm.get("tribute_policy", "heuristic") != cfg.tribute_policy:
            raise ValueError("warm_start tribute_policy differs from the config")
        mine = self.policy.config
        if (source.config.temperature, source.config.top_k) != (mine.temperature, mine.top_k):
            if not cfg.warm_start_policy_override:
                raise ValueError("warm_start temperature/top_k differ from the config; set "
                                 "warm_start_policy_override to allow it")
            if source.config.temperature != mine.temperature:
                warnings.warn(f"warm_start was trained at temperature {source.config.temperature}"
                              f", this run samples at {mine.temperature}", stacklevel=2)
        self.policy.net.load_state_dict(warm["model"])
        self.warm_start_source = str(path)
        self.warm_start_digest = model_digest(warm["model"])
        return warm

    def _fused_league_specs(self, entries: list[dict[str, Any]]) -> frozenset[str]:
        """League specs that ARE the frozen pruning reference playing argmax
        (a Stage A checkpoint with the reference's weights digest). Their play
        rows are scored by the rollout's shared reference forward instead of
        a second copy of the same network; see `RolloutCollector.collect`.
        Only on the fast path with heuristic tribute, where that forward is
        exactly what the league's model would compute."""
        from eval.policies import model_digest
        from train.league import entry_kind

        if not (self.config.fast_rollout and self.phase_code == PLAY):
            return frozenset()
        fused = set()
        for item in entries:
            spec = item["spec"]
            if entry_kind(spec) != "network" or spec.startswith("sample"):
                continue
            payload = load_checkpoint(spec, "cpu")
            if (payload.get("stage", "dmc") == "dmc"
                    and payload.get("tribute_policy", "heuristic") == "heuristic"
                    and model_digest(payload["model"]) == self.policy.reference_checkpoint_id):
                fused.add(spec)
        return frozenset(fused)

    def _default_opponent(self) -> OpponentSource:
        spec = self.config.opponent
        if spec.startswith("frozen:"):
            spec = "frozen:" + str(resolve_artifact(spec[len("frozen:"):]))
        # "frozen" is the M1 reference itself, already on the device and never trained.
        return config_opponent(spec, self.policy.reference, str(self.device),
                               self.config.candidate_chunk, fallback=self.follow_fallback(),
                               pinned=self.config.exploiter_pin_target)

    def follow_fallback(self) -> str | None:
        """A follow: opponent's model before the first snapshot: warm_start."""
        if not self.config.opponent.startswith(FOLLOW):
            return None
        return str(resolve_artifact(self.config.warm_start))

    # ----------------------------------------------------------------- learner
    @torch.no_grad()
    def refresh_values(self) -> None:
        """Critic values of every completed step, written into the buffer(s)."""
        self.critic.eval()
        chunk = max(1, self.config.minibatch_size)
        for buffer in self.buffers:
            steps = buffer.pending_value_steps()
            for begin in range(0, steps.size, chunk):
                part = steps[begin:begin + chunk]
                x = to_device(buffer.critic_input(part), self.device, torch.float32)
                buffer.value[part] = self.critic(x).float().cpu().numpy()

    def collect(self) -> None:
        if self.actors is None:
            super().collect()
            return
        with self._phase("actor processes collect"):
            self.actor_stats = self.actors.collect(self, self.config.rollout_steps)

    def minibatches(self) -> Iterator[dict[str, torch.Tensor]]:
        """One epoch of minibatches over every buffer's finalized samples."""
        size = self.config.minibatch_size
        if len(self.buffers) == 1:
            yield from self.buffers[0].minibatches(size, self.rng, self.device)
            return
        counts = [buffer.n_samples for buffer in self.buffers]
        shard = np.repeat(np.arange(len(self.buffers)), counts)
        local = np.concatenate([buffer.samples[:buffer.n_samples] for buffer in self.buffers])
        order = self.rng.permutation(local.size)
        for begin in range(0, order.size, size):
            chosen = order[begin:begin + size]
            parts = [self.buffers[s].gather(local[chosen[shard[chosen] == s]], self.device)
                     for s in np.unique(shard[chosen])]
            yield concat_minibatches(parts)

    def close(self) -> None:
        """Stop the actor processes, if any. Idempotent."""
        if self.actors is not None:
            self.actors.close()

    def kl_coef(self) -> float:
        return schedule(self.config.kl_coef, 0.0, self.progress["updates"],
                        self.config.kl_anneal_updates)

    # Scalar minibatch terms reported as means over the update's minibatches.
    STAT_KEYS = ("policy_loss", "value_loss", "entropy", "kl_ref", "hidden_loss", "finish_loss",
                 "approx_kl", "clip_fraction", "ratio_deviation", "policy_grad_norm",
                 "critic_grad_norm")

    def learn(self) -> dict[str, float]:
        cfg = self.config
        with self._phase("critic values"):
            self.refresh_values()
        with self._phase("GAE"):
            samples = sum(buffer.finalize(cfg.gamma, cfg.gae_lambda) for buffer in self.buffers)
        stats: dict[str, float] = {"update_samples": samples, "kl_coef": self.kl_coef()}
        if samples == 0:
            return stats
        returns = np.concatenate([b.returns[b.samples[:b.n_samples]] for b in self.buffers])
        values = np.concatenate([b.value[b.samples[:b.n_samples]] for b in self.buffers])
        variance = float(np.var(returns))
        stats["explained_variance"] = (1.0 - float(np.var(returns - values)) / variance
                                       if variance > 0 else 0.0)
        stats["mean_return"] = float(returns.mean())
        stats["mean_value"] = float(values.mean())
        kl_coef = stats["kl_coef"]
        # Statistics and finiteness stay on the device and are read once per
        # epoch, so a minibatch never waits for the device. A non-finite loss
        # or gradient raises at the end of its epoch instead of before its
        # optimizer step; the run then stops without saving, as before.
        totals = torch.zeros(len(self.STAT_KEYS), dtype=torch.float64, device=self.device)
        count = 0
        first_ratio_deviation = None
        stopped_epoch = cfg.epochs
        self.net.train()
        self.critic.train()
        for epoch in range(cfg.epochs):
            epoch_kl = torch.zeros((), dtype=torch.float64, device=self.device)
            finite = torch.ones((), dtype=torch.bool, device=self.device)
            epoch_batches = 0
            batches = self.minibatches()
            while True:
                with self._phase("minibatch gather + upload"):
                    mb = next(batches, None)
                if mb is None:
                    break
                with self._phase("learner forward"):
                    terms = self.minibatch_loss(mb, kl_coef)
                with self._phase("learner backward + step"):
                    self.policy_optimizer.zero_grad(set_to_none=True)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    (terms["policy_total"] + cfg.value_coef * terms["value_loss"]).backward()
                    policy_grad = torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                                                 cfg.grad_clip)
                    critic_grad = torch.nn.utils.clip_grad_norm_(self.critic.parameters(),
                                                                 cfg.grad_clip)
                    finite &= (torch.isfinite(terms["policy_total"])
                               & torch.isfinite(terms["value_loss"])
                               & torch.isfinite(policy_grad) & torch.isfinite(critic_grad))
                    self.policy_optimizer.step()
                    self.critic_optimizer.step()
                self.progress["optimizer_steps"] += 1
                terms["policy_grad_norm"] = policy_grad
                terms["critic_grad_norm"] = critic_grad
                totals += torch.stack([terms[key].detach().double() for key in self.STAT_KEYS])
                if first_ratio_deviation is None:
                    first_ratio_deviation = terms["ratio_deviation"].detach()
                count += 1
                epoch_kl += terms["approx_kl"].detach().double()
                epoch_batches += 1
            if not bool(finite):
                raise FloatingPointError("non-finite PPO loss or gradient")
            if cfg.target_kl and float(epoch_kl) / max(1, epoch_batches) > cfg.target_kl:
                stopped_epoch = epoch + 1
                break
        stats.update(zip(self.STAT_KEYS, (totals / count).tolist()))
        stats["first_ratio_max_deviation"] = float(first_ratio_deviation)
        stats["epochs_run"] = stopped_epoch
        self.progress["samples"] += samples
        return stats

    def minibatch_loss(self, mb: dict[str, torch.Tensor], kl_coef: float) -> dict[str, torch.Tensor]:
        cfg = self.config
        # One state-tower pass feeds both the policy logits and the auxiliary
        # heads (B5 ran it twice; the gradient is the same sum either way).
        state = self.net.state_tower(mb["obs"]) if cfg.fast_rollout else None
        log_prob, entropy, log_probs = policy_terms(
            self.policy, mb["obs"], mb["cand"], mb["offsets"], mb["phase"], mb["chosen"],
            self.phase_code, state=state)
        log_ratio = log_prob - mb["logp"]
        ratio = log_ratio.exp()
        advantage = mb["advantage"]
        if cfg.normalize_advantages and len(advantage) > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        surrogate = -torch.minimum(ratio * advantage,
                                   ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advantage).mean()
        if cfg.fast_rollout:
            kl_ref = cached_reference_kl(log_probs, mb["ref_logp"], mb["offsets"]).mean()
        else:
            kl_ref = reference_kl(self.policy, log_probs, mb["obs"], mb["cand"], mb["offsets"],
                                  mb["phase"], self.phase_code).mean()
        if state is None:
            state = self.net.state_tower(mb["obs"])
        aux = self.net.auxiliary(state)
        hidden_target = mb["critic_obs"][:, gd.OBS_DIM:].long().reshape(-1)
        hidden_loss = F.cross_entropy(aux["hidden"].reshape(-1, 3), hidden_target)
        finish_loss = F.cross_entropy(aux["finish"], mb["finish"])
        value_loss = F.mse_loss(self.critic(mb["critic_obs"]), mb["returns"])
        policy_total = (surrogate - cfg.entropy_coef * entropy.mean() + kl_coef * kl_ref
                        + cfg.hidden_weight * hidden_loss + cfg.finish_weight * finish_loss)
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = ((ratio - 1).abs() > cfg.clip).float().mean()
            ratio_deviation = (ratio - 1).abs().max()
        return {"policy_total": policy_total, "policy_loss": surrogate, "value_loss": value_loss,
                "entropy": entropy.mean(), "kl_ref": kl_ref, "hidden_loss": hidden_loss,
                "finish_loss": finish_loss, "approx_kl": approx_kl,
                "clip_fraction": clip_fraction, "ratio_deviation": ratio_deviation}

    # -------------------------------------------------------------- bookkeeping
    def save(self, snapshot: bool = False) -> None:
        self.progress["elapsed_seconds"] = self.prior_elapsed + time.monotonic() - self.started
        if snapshot:
            self.progress["snapshot_update"] = self.progress["updates"]
        payload = self.policy.checkpoint_payload(
            optimizer=self.policy_optimizer.state_dict(), config=asdict(self.config),
            progress=dict(self.progress), rng=rng_state(self.rng),
            tribute_policy=self.config.tribute_policy)
        payload.update({
            "critic_model_config": asdict(self.critic.config), "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "sampler": self.generator.get_state(),
            "init_source": self.init_source, "critic_source": self.critic_source,
            "warm_start_source": self.warm_start_source,
            "warm_start_digest": self.warm_start_digest,
            "league": {"stage": "B", "opponents": (
                [e.name for e in self.league.entries] if self.league is not None
                else [self.config.opponent if self.opponent is None
                      else getattr(self.opponent, "name", type(self.opponent).__name__)])},
        })
        if self.league is not None:
            # Entry EMAs, tallies, snapshot list and sampler RNG; in actor mode
            # also each actor's sampler RNG (actors are idle between updates).
            payload["league_state"] = self.league.state_dict()
            if self.actors is not None:
                payload["league_actor_rng"] = self.actors.request("league_rng")
        self.league_snapshot_taken = False
        save_checkpoint(self.run_dir / "latest.pt", payload)
        if snapshot:
            snapshots = self.run_dir / "checkpoints"
            snapshots.mkdir(exist_ok=True)
            target = snapshots / f"step-{self.progress['updates']:09d}.pt"
            if not target.exists():
                os.link(self.run_dir / "latest.pt", target)
        self.last_save = time.monotonic()

    def should_stop(self) -> bool:
        return (self.stop_requested or self.progress["updates"] >= self.config.max_updates
                or time.monotonic() - self.started >= self.config.max_seconds)

    def maintenance(self) -> None:
        snapshot = self.progress["updates"] - self.progress["snapshot_update"] >= self.config.snapshot_updates
        # A new league snapshot is saved at once, so a resume sees the same pool.
        if (snapshot or self.league_snapshot_taken
                or time.monotonic() - self.last_save >= self.config.checkpoint_seconds):
            self.save(snapshot=snapshot)

    def metric(self, values: dict[str, Any]) -> dict[str, Any]:
        elapsed = time.monotonic() - self.started
        self.progress["elapsed_seconds"] = self.prior_elapsed + elapsed
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = rss if sys.platform == "darwin" else rss * 1024
        window = self.window
        record = {**self.progress, **values, "wall_seconds": elapsed,
                  "decisions_per_second": (self.progress["decisions"] - self.prior_decisions) / max(elapsed, 1e-9),
                  "window_matches": window["matches"], "window_rounds": window["rounds"],
                  "learner_match_win_rate": window["wins"] / window["matches"] if window["matches"] else None,
                  "learner_round_return": window["learner_return"] / window["rounds"] if window["rounds"] else None,
                  "cumulative_learner_match_win_rate": (self.progress["learner_match_wins"] / self.progress["matches"]
                                                        if self.progress["matches"] else None),
                  "process_peak_rss_mib": rss_bytes / 2**20}
        if self.device.type == "cuda":
            record["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated(self.device) / 2**20
        self.window = {"matches": 0, "wins": 0, "rounds": 0, "learner_return": 0.0}
        with (self.run_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if self.writer:
            for key, value in record.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.writer.add_scalar(key, value, self.progress["updates"])
        print(json.dumps(record, allow_nan=False), flush=True)
        return record

    def update(self) -> dict[str, Any]:
        """One PPO iteration: rollout, learn on completed rounds, clear."""
        collect_start = time.monotonic()
        self.collect()
        collect_seconds = time.monotonic() - collect_start
        learn_start = time.monotonic()
        stats = self.learn()
        if self.actors is None:
            self.buffer.next_iteration()    # actors clear their shards before collecting
        else:
            stats.update(self.actor_stats)
        self.progress["updates"] += 1
        if self.league is not None:
            if self.league.should_snapshot(self.progress["updates"]):
                stats["league_snapshot"] = str(self.league_snapshot())
            if self.config.league_import_dir:
                imported = self.league.import_new(self.config.league_import_dir)
                if imported:
                    stats["league_imported"] = ",".join(e.name for e in imported)
                    self.league_snapshot_taken = True   # save at once, as for a snapshot
            stats.update(self.league_stats())
        if self.config.exploiter_publish_dir:
            stats.update(self.exploiter_cycle())
        return self.metric({**stats, "collect_seconds": collect_seconds,
                            "learn_seconds": time.monotonic() - learn_start})

    def league_snapshot(self) -> Path:
        """Write the current policy to an immutable file and add it to the
        league (one entry per snapshot temperature). Never `latest.pt`: league
        models load lazily and must not change under a running match. In actor
        mode the actors pick it up at the next update boundary."""
        updates = self.progress["updates"]
        directory = self.run_dir / "league"
        suffix = ""
        if (directory / f"update-{updates:09d}.pt").exists():
            # An earlier run crashed after writing it; never overwrite a snapshot.
            suffix = f"-r{self.progress['resumes']:02d}"
        path = directory / f"update-{updates:09d}{suffix}.pt"
        save_checkpoint(path, self.policy.checkpoint_payload(
            optimizer={}, config=asdict(self.config), progress=dict(self.progress), rng={},
            tribute_policy=self.config.tribute_policy))
        self.league.add_snapshot(path, name=f"snapshot@{updates}{suffix}")
        self.league_snapshot_taken = True
        return path

    def exploiter_cycle(self) -> dict[str, Any]:
        """Exploiter mode, after each update: publish and reset when the cycle
        is due (see the module docstring). Returns metrics."""
        cfg = self.config
        self.exploiter_window.append((self.window["wins"], self.window["matches"]))
        del self.exploiter_window[:-cfg.exploiter_window_updates]
        wins = sum(w for w, _ in self.exploiter_window)
        matches = sum(m for _, m in self.exploiter_window)
        rate = wins / matches if matches else 0.0
        cycle = self.progress["updates"] - self.progress["exploiter_cycle_start"]
        stats: dict[str, Any] = {"exploiter/cycle_updates": cycle,
                                 "exploiter/window_win_rate": rate,
                                 "exploiter/published": self.progress["exploiter_published"]}
        early = (cfg.exploiter_win_rate > 0 and cycle >= cfg.exploiter_min_updates
                 and matches > 0 and rate >= cfg.exploiter_win_rate)
        if cycle < cfg.exploiter_cycle_updates and not early:
            return stats
        stats["exploiter_publish"] = str(self.exploiter_publish())
        stats["exploiter_reset_to"] = self.exploiter_reset()
        stats["exploiter/published"] = self.progress["exploiter_published"]
        return stats

    def exploiter_publish(self) -> Path:
        """Write the policy as the next exploiter-<n>-u<update>.pt, never
        overwriting (a crash after the write republishes under a new n)."""
        directory = Path(self.config.exploiter_publish_dir)
        directory.mkdir(parents=True, exist_ok=True)
        n = self.progress["exploiter_published"] + 1
        while True:
            path = directory / f"exploiter-{n:04d}-u{self.progress['updates']:09d}.pt"
            if not any(directory.glob(f"exploiter-{n:04d}-*.pt")):
                break
            n += 1
        save_checkpoint(path, self.policy.checkpoint_payload(
            optimizer={}, config=asdict(self.config), progress=dict(self.progress), rng={},
            tribute_policy=self.config.tribute_policy))
        self.progress["exploiter_published"] = n
        return path

    def exploiter_reset(self) -> str:
        """Policy weights from the opponent's current checkpoint (the newest
        league snapshot, or warm_start before the first), a fresh policy
        optimizer, a new cycle. The critic carries on. Saved at once."""
        # A pinned opponent moves to the newest snapshot now, and this cycle
        # exploits it. In actor mode the actors hold the opponents; scan here.
        source = (self.opponent.retarget() if isinstance(self.opponent, FollowOpponent)
                  else FollowOpponent(self.config.opponent[len(FOLLOW):],
                                      self.follow_fallback(), rescan_seconds=0).current())
        payload = load_checkpoint(source, self.device)
        if payload.get("stage") != "ppo":
            raise ValueError("an exploiter resets to a Stage B checkpoint")
        if payload.get("reference_checkpoint_id") != self.policy.reference_checkpoint_id:
            raise ValueError("the followed run uses a different frozen reference")
        self.net.load_state_dict(payload["model"])     # in place: actors share these tensors
        self.policy_optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.policy_lr)
        self.progress["exploiter_cycle_start"] = self.progress["updates"]
        self.exploiter_window.clear()
        self.league_snapshot_taken = True   # save at once, so a resume does not republish
        return source

    def league_stats(self) -> dict[str, float]:
        """League.stats() of the authoritative league, finite values only
        (entries without games have no win rate), plus per-actor model counts."""
        stats = {k: v for k, v in self.league.stats().items() if math.isfinite(v)}
        if self.actors is not None:
            for key in ("active_models", "cached_models", "forward_calls"):
                stats.pop(f"league/{key}", None)
            stats.update(self.actor_league_info)
        return stats

    def run(self) -> dict[str, Any]:
        def stop(_signum, _frame):
            self.stop_requested = True
        handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        completed = False
        try:
            self.save(snapshot=self.progress["updates"] == 0)
            while not self.should_stop():
                self.update()
                self.maintenance()
            completed = True
            return self.progress
        finally:
            try:
                if completed:
                    self.save()
            finally:
                self.close()
                if self.writer:
                    self.writer.close()
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)


def load_config(path: str | Path, **overrides: Any) -> PPOConfig:
    data = json.loads(Path(path).read_text())
    # Keys starting with "_" are comments (e.g. "_note").
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    data.update({k: v for k, v in overrides.items() if v is not None})
    return PPOConfig(**data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="train/configs/ppo-smoke.json")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args(argv)
    config = load_config(args.config, max_updates=args.max_updates, max_seconds=args.max_seconds)
    PPOTrainer(config, args.run_dir, args.device, args.resume).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
