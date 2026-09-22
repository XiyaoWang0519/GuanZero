# Stage B PPO throughput (B5b)

Date: 2026-09-22. Code: `train/ppo.py`, `train/ppo_actors.py` (new),
`train/policy.py`, `train/rollout_buffer.py`, `train/model.py`. Bench:
`bench/ppo_throughput.py`. Tests: `tests/test_ppo_throughput.py`,
`tests/test_ppo_actors.py`.

## Why

On the B6 pod (RTX 5090, 64 vCPU), three trainers ran side by side: DMC and two
PPO arms. Each PPO process (2,048 envs, 8 engine threads, 4 torch threads) made
about 27 k decisions/s on about 2.5 to 4 cores. GPU use was 44 to 73 % across
all three processes, and the load average was about 8 of 64. One Python thread
per process was the limit, not compute.

## Profile before (HEAD `241eb15`, this Mac, CPU)

Setup: an Apple M4 Pro, the M1 final plus the B2 critic, 1,024 envs, rollout
64, 4 epochs, minibatch 4,096, and the frozen M1 as opponent. Timers wrapped
each collaborator, and one update ran as warm-up. The Mac was otherwise idle
(18:55 run).

| Phase | Share of wall |
|---|---|
| learner forward (policy + **reference re-run for the KL term**) | 35.6 % (reference alone 13.7 %) |
| learner backward + Adam | about 40 % (the rest of `learn`) |
| learner act (prune forward + sort, policy forward, sample, 5 D2H copies) | 12.9 % |
| opponent act (a second frozen-M1 forward over the same batch) | 6.5 % |
| minibatch gather (numpy fancy index, then uint8 to float on the host) | 1.9 % |
| env `pending` + `step` (C++) | 3.2 % |
| buffer add, ragged gathers, bookkeeping, GAE | < 2 % |

Throughput was 11.8 k decisions/s. `collect` alone ran at 51 k/s, so the
learner took 77 % of wall time. With a 16-wide network standing in for
everything except model compute (`--tiny`, the overhead that a fast GPU
leaves), throughput was 55 k/s and the learner was still 61 % of wall time.
Each minibatch cost about 15 small device synchronisations: `float()` on 10
stats, 2 `clip_grad_norm_(error_if_nonfinite)`, an `isfinite` check and
`steps.cpu()`. Each rollout step cost about 10: 5 separate `.cpu()` calls,
`isfinite`, `bincount`, the `sizes<=0` check and the opponent's own checks.
Every host-to-device copy of uint8 features was widened to float32 on the host
first. PyTorch converts dtype on the CPU side of a CPU-to-CUDA copy, so this
moved 4x the bytes from pageable memory.

## What changed

1. **The KL reference is cached at rollout.** `PolicyStep.ref_log_probs`, which
   is log softmax(Q_M1/T) over the pruned set, is stored per candidate in the
   buffer (`RolloutBuffer.ref_logp`). The learner no longer re-runs the frozen
   M1 in any epoch.
2. **One frozen-M1 forward per step.** Pruning and the default `frozen`
   opponent share it. `FrozenModelOpponent.act` is a pure argmax of that same
   network, so the opponent's argmax is taken from the shared scores. The
   `OpponentSource` contract is unchanged: `bind`, `on_match_start` and
   `on_match_end` are still called in order. Other opponents, such as a league
   or greedy, still go through `act()`.
3. **One upload and one download per step.** The whole pending batch goes
   through reusable pinned buffers as uint8, is widened on the device, and rows
   are selected there. On CPU it is used in place with no copy. All results come
   back through a single synchronisation (`Uploader`, `to_host`).
4. **Minibatches have no synchronisation.** Features travel as uint8 through
   pinned memory. Stats, the finiteness check and the approx KL accumulate on
   the device and are read once per epoch. `bincount` became a segment sum.
5. **The learner runs the state tower once per minibatch.** Its output feeds
   both the policy logits and the auxiliary heads. B5 ran it twice, and the
   gradient is the same.
6. **Synchronous actor processes, opt-in** with `actor_processes: W`. There are
   W shards, each with its own `VecEnv`, policy copy, opponent and sampler, on
   the trainer's device. Shard buffers live in shared memory and are read in
   place by the learner. Actors wait while the learner trains, so there is
   **no policy lag**. `/dev/shm` is checked up front.
7. `fast_rollout: false` keeps the B5 path for A/B runs. Both
   `fast_rollout` and `actor_processes` can change on resume.

## Semantics

- **In-process fast path: identical under a seed.** With the real M1 final,
  256 envs and 100 steps, the B5 path and the fast path gave bitwise-equal
  stored choices, candidates, behaviour log-probabilities and reference
  log-probabilities (maximum difference 0.0) and equal progress counters. The
  tests check the same equality with a small network for the frozen opponent,
  the greedy opponent and learned tribute. They also check that advantages and
  stats match and that the cached-KL gradient equals the B5 gradient
  (rtol 1e-4). Weights agree to the order of the learning rate after two
  updates, because Adam amplifies float-level gradient noise on near-zero
  entries.
- **Error timing changed.** A non-finite loss or gradient now raises at the end
  of its epoch instead of before its optimizer step. The run still stops
  without saving.
- **Actor mode has no policy lag.** Every step collected in an update matches
  the learner's log-probability within 1e-5
  (`test_actor_updates_have_no_policy_lag`), and actors run the published
  weights (digest test). Two things differ from in-process. First, the random
  streams: shard env seeds and samplers are drawn from the trainer RNG, and
  minibatches are permuted over the union of shards. Second, the env layout
  (learner team = env % 2 within each shard). Rounds still in progress are
  carried over exactly as in-process. Only config opponents are supported, not
  an `OpponentSource` object such as a league: one league sampler across shards
  is future work.

## CPU before and after

The coordinator's cross-play evaluation (6 worker processes) ran in the
background throughout. Load average (1 min) was 7 to 25, falling during the
runs. Settings were at most 6 threads, 1,024 envs, rollout 64, 4 epochs,
minibatch 4,096 and default `PPOConfig`. Runs were interleaved BASE → FAST →
ACT2 in each repetition. Each run was one warm-up update plus 3 measured
updates. BASE is the HEAD code with 3 engine threads and 3 torch threads. FAST
is the B5b in-process path with the same threads. ACT2 is two actors with 1
engine thread and 2 torch threads each.

| Model | BASE median | FAST median | ACT2 median | FAST/BASE (paired, median) |
|---|---|---|---|---|
| M1 size, 5 reps | 4.2 k/s | 6.1 k/s | 5.8 k/s | **1.44x** (1.07 to 2.05) |
| tiny (overhead only), 3 reps, load about 6 | 53.8 k/s | 59.5 k/s | 54.5 k/s | 1.08x |

The two least loaded M1 repetitions gave BASE 7.7 k and 12.5 k against FAST
15.9 k and 15.8 k. The learner median fell from 36.0 s to 23.6 s per 3
updates. A later profiled FAST run at load 4.4 reached 16.1 k/s. Its breakdown
was learner backward 45 %, learner forward 24 %, reference forward 13 %,
policy/prune/sample 7 %, env C++ 5 %, and gather, upload and buffer add 5 %.

On this CPU, actors cannot help. Model compute saturates the cores, actors and
learner share them, and each actor got only 1 engine thread. The gains that
depend on synchronisation and transfer (items 3, 4 and 6) are invisible on CPU,
because synchronisation is free there and copies are memcpy.

## Expected GPU gain (not measured, no rental)

On the pod, the removed work maps like this. The KL re-forward and the
duplicate state tower were about 25 % of learner FLOPs. The learner and
rollout synchronisations were about 15 per minibatch and about 10 per step. The
4x H2D bytes were pageable. The second M1 forward was a separate launch
sequence. I expect **about 1.5 to 2x in-process**. Going further means
spending idle cores: with `actor_processes` 4 to 8 at 8 engine threads each,
rollout time divides by W while the learner is unchanged. At the 27 k/s
baseline, **3 to 4x** looks plausible if the learner is under about 35 % of
update time. The bench measures exactly this.

Command for the next pod, about 5 minutes, running the B5 path, then fast
in-process, then 4 and 8 actors:

```sh
df -h /dev/shm   # actors need 2.7 GiB free here for 2,048 envs (checked at start)
.venv/bin/python bench/ppo_throughput.py --config train/configs/ppo.json \
    --device cuda --compare --actors 0 4 8 --updates 6 --warmup 2 --out bench-ppo.json
# optional per-phase breakdown (adds syncs):
.venv/bin/python bench/ppo_throughput.py --config train/configs/ppo.json --device cuda --profile --updates 3
```

## Open issues

- GPU numbers are still missing, and so is the right W for the pod. The learner
  share on GPU is unknown until the bench runs.
- There is no league (`OpponentSource` object) in actor mode.
- `candidate_chunk` (32,768) splits learner minibatches into several launch
  sequences on GPU for no memory benefit, since autograd keeps every chunk.
  Raising it in the GPU config is a free knob to try.
- Async actors (overlapping rollout with learning, one update of lag) are not
  built. They would need double-buffered shards and a lag-bounded ratio test.
