# Learner staging and reference skip: two exact PPO speedups

Date: September 24, 2026. Code: `train/rollout_buffer.py` (`StagedSamples`),
`train/ppo.py`, `train/ppo_actors.py`, `bench/ppo_throughput.py`. Tests:
`tests/test_ppo_throughput.py` (five new cases). Evidence script and raw
output: `.work/perf-learner-2026-09-24/` (git-ignored copy of the scratchpad
runs described below).

Both changes are on by default and may be switched on resume
(`MUTABLE_ON_RESUME`). Neither changes any stored row, any sampled action,
any loss term or any weight: the requirement was no compromise on the model,
and the evidence is bitwise equality on CPU with the real checkpoints.

## Why the rollout is slow, measured

The rollout of a frozen-opponent arm on this Mac (M4 Pro, 2,048 environments,
10 engine threads, 4 Torch threads, `bench/ppo_throughput.py --profile`,
2 updates) spends its collect time as follows.

| Collect phase | Seconds | Share |
|---|---:|---:|
| Reference forward (M1 scores every candidate for pruning) | 1.85 | 56% |
| Policy forward, prune, sample | 0.97 | 30% |
| Critic values, buffer writes | 0.40 | 12% |
| C++ engine `pending` + `step` | 0.11 | 3% |

The engine is 3% of collection. What the rollout waits for is neural
inference over 2,048 rows × up to 32 candidates per step, and on a GPU the
serial Python step around it (engine, upload, forwards, copy back, buffer
write) leaves the device idle between calls. A CPU-only actor machine would
therefore move the expensive part onto the slowest device; the guidance in
`docs/COMPUTE_GUIDE.md` item 4 should be read with that in mind.

The reference forward is only useful when it prunes. Over 31,478 training
decisions of the M1 policy (256 environments after two warm-up updates):

| Candidates in the row | Rows | Candidate volume |
|---|---:|---:|
| at most 32 (`top_k`, nothing to prune) | 93.2% | 40.5% |
| more than 32 | 6.8% | 59.5% |

## What changed

1. **`PPOConfig.learn_on_device`** (default true). `refresh_values` uploads
   the completed trajectories' uint8 observations, hidden rows, candidates
   and reference log-probabilities once per update (`RolloutBuffer.stage`);
   critic chunks and every minibatch of every epoch then `index_select` on
   the device from host-computed indices, so a minibatch uploads a few
   hundred kilobytes of indices instead of gathering and pinning tens of
   megabytes of features. The critic chunks are the same 8,192-row slices as
   before. `RolloutBuffer.gather` falls back to the host path for rows that
   are not finalized samples (the actor no-lag test reads such rows).
2. **`PPOConfig.skip_unpruned_reference`** (default true). In the fast
   rollout, once the KL coefficient has annealed to zero, the reference
   forward runs only on rows with more than `top_k` candidates (learner rows
   and shared-reference league opponents alike) and its scores are scattered
   into a zero-filled full-length tensor. `prune_candidates` keeps a row of at
   most `top_k` candidates whole, in the engine's order, whatever the scores,
   so the policy forward, the draw and the stored row are unchanged. The
   rows the fused frozen opponent plays are always scored (its argmax needs
   them). Skipped learner rows store `ref_logp = NaN`; the learner keeps NaN
   out of the graph and reports `kl_ref` over the scored rows only. While
   `kl_coef` is positive every row is scored, as before; actors receive the
   flag with each collect command.

`bench/ppo_throughput.py` takes `--learn-on-device 0|1` and
`--skip-unpruned-reference 0|1` for A/B runs on a pod.

## Equivalence evidence (real checkpoints, CPU)

M1 final as init and reference, B2 critic, `temperature 0.02`, `top_k 32`,
256 environments, 64 steps, 4 epochs, minibatch 2,048, seed 7, three to four
updates per run. Every run with a change on was compared with the same
configuration with both changes off: every stored row (observations, hidden
rows, candidates, choices, behaviour log-probabilities, phases, seats,
trajectories, rewards), every policy and critic weight after every update,
and every logged statistic other than the ones below.

| Case | Rows, choices, log-probs, weights | Differences |
|---|---|---|
| Frozen opponent, `kl_coef 0`, `learn_on_device` only | identical | none, every statistic equal |
| Frozen opponent, `kl_coef 0`, skip only, and both | identical | `ref_logp` of scored rows ≤ 5.3e-5; `kl_ref` statistic |
| Frozen opponent, `kl_coef 0.1` annealed to 0 over 2 updates, both | identical | updates 0–1 fully identical; from update 2 as above (≤ 6.9e-5) |
| Two actor processes, `kl_coef 0`, both | identical | as above (≤ 5.2e-5) |
| League arm: B8 league final warm start, B11 pool, a snapshot every update so shared-reference opponents play, both | identical | as above (≤ 7.6e-5); snapshot paths |

The `ref_logp` differences on scored rows are the float rounding of scoring
a differently composed batch (a score difference of about 1e-6 becomes 5e-5
after division by the temperature), the same class as the September 23
fusion change; they enter nothing when `kl_coef` is zero, and when it is not
zero the skip is off and the values are bitwise equal. About 90% of learner
rows were skipped in these runs. On CUDA the same rounding can flip a
near-tie among the candidates ranked around `top_k` in a scored row, as the
fusion report already notes: statistically equivalent, not bitwise.

The new tests reproduce this with the small test network: host and device
gathers equal for any subset, identical statistics and weights with and
without staging (in-process and with two actors), identical rows and weights
with and without the skip, the skip inactive until the KL coefficient is
zero, and identical loss terms and gradients between a partial-reference
minibatch and the same minibatch with zeroed reference values.

## Speed

Measured on this Mac only (CPU device), 2,048 environments, ABAB, 2 updates
per run:

| | Both off | Both on |
|---|---:|---:|
| Collect, 2 updates | 3.45–3.60 s | 3.07–3.12 s |
| of which reference forward | 1.97–2.05 s | 1.51–1.52 s |
| Learn, 2 updates | 11.6–12.2 s | 12.1–12.5 s |

The skip removes about a quarter of the reference forward on CPU (the 6.8%
of rows still scored hold 60% of the candidates, and the smaller batch runs
less efficiently), so collection is about 11% faster here. Learning is
compute-bound on CPU and unchanged, as expected: the staging change targets
the per-minibatch host gather, pinning and upload that the RTX 4090
validation could not see inside its 54 ms per minibatch (16 ms of which was
loss, backward and clipping). Its effect is therefore unmeasured until the
next pod runs the bench with the flags off and on, which `docs/PERF_TODO.md`
now lists first.
