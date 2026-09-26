# Exploration floor for learner seats (2026-09-26)

Code and tests only; no training run was launched and no result is claimed.

## Settings

Three new `HistoryPPOConfig` fields; defaults reproduce the old behaviour exactly.

| Field | Flag | Default | Meaning |
|---|---|---|---|
| `rollout_temperature` | `--rollout-temperature` | 1.0 | T, the softmax temperature for learner-seat sampling |
| `rollout_epsilon` | `--rollout-epsilon` | 0.0 | eps, the mass spread uniformly over legal candidates |
| `behaviour_weight_cap` | `--behaviour-weight-cap` | 1.0 | c, the truncation of the importance weight |

The fields are stored in `asdict(config)`. They therefore reach the checkpoint
`config`, the manifest and `resolved-config.json`, and `--resume` restores
them. Checkpoints written before this change resume with the defaults.

## Behaviour distribution

Over the n legal candidates, with actor logits l:
`pi_b(a) = (1 - eps) * softmax(l / T)(a) + eps / n`.

`HistoryActor.explore` implements this. It draws one Gumbel-argmax sample from
`log_softmax(l / T)`. When eps > 0 it also draws a coin, u < eps, and on
heads replaces the sample with a uniform index in [0, n). Frozen snapshot seats
still call `HistoryActor.act`, unchanged, at T = 1 with no epsilon. Evaluation,
greedy and inference paths also still call `act`.

## Buffer and learner

- `logp` keeps its old meaning: log pi_target(a), the collecting actor's T = 1,
  eps = 0 log-softmax. The PPO ratio pi_theta / pi_old is therefore unchanged.
- The new row field `behaviour_logp` (float32) stores log pi_b(a) as sampled.
  It flows everywhere `logp` does: add_step, compact, the empty buffer and
  carry-over rows. No existing field changes dtype or shape.
- `minibatch_loss` multiplies the advantage by the truncated importance weight
  `w = min(c, exp(logp - behaviour_logp)) = min(c, pi_old(a) / pi_b(a))`. The weight is applied after per-minibatch advantage normalisation. The
  normalisation statistics are those of the raw GAE advantages, and w only
  rescales individual rows. Since w > 0, the product sits inside the clipped
  objective: min(r w A, clip(r) w A) = w min(r A, clip(r) A).
- The value loss is not weighted, and GAE and the critic are not corrected
  for the behaviour policy (this is not V-trace). With small eps the critic
  learns roughly V of pi_b, and that bias is accepted.

## New metrics (per update line; existing names unchanged)

- `behaviour_weight_mean`: the mean of w, averaged over minibatches.
- `behaviour_weight_cap_fraction`: the fraction of rows with a raw ratio > c.
- `rollout_epsilon_pick_fraction`: the fraction of learner rows from the uniform branch.
- `rollout_behaviour_entropy`: the mean entropy of pi_b over each decision's
  legal candidates, measured at collection. The existing `entropy` stays the
  learner-time target entropy.

## Bitwise default guarantee and its tests

With T = 1 and eps = 0, `explore` samples from the same table as `act`, with
the same single Gumbel draw. It returns the same choice, and `behaviour_logp`
has the same values as `logp`. In the learner, `logp - behaviour_logp` is
exactly 0, so w is exactly 1.0 and `advantage * 1.0` is the identity.
`tests/test_history_exploration.py` compares `act` and `explore` with a copy of
the pre-change `act` (identical choices, log-probabilities and generator state
afterwards), a whole collector run with one patched to the pre-change `act`
(identical choice logs and buffers), and `minibatch_loss` with a copy of the
pre-change loss (every term `torch.equal`; actor gradients bitwise equal on
one CPU thread, since multithreaded CPU backward is not reproducible even for
the same loss twice). Other tests cover `behaviour_logp` against an
independent float64 mixture (1e-5), chi-square frequencies on fixed 3- and
4-candidate logits, the weights, cap fraction and weighted surrogate, snapshot
seats never calling `explore`, the CLI, validation, checkpoint and resume
round trip, and buffer dtypes.

## Random-stream caveat

When eps = 0, the generator consumption is identical to before, at any T.
When eps > 0, every learner batch draws two extra `[n]` float64 uniforms after
the Gumbel table: first the coin, then the uniform index. Snapshot-seat
sampling and later learner batches then see a shifted stream: eps > 0 runs are
seed-reproducible but not draw-for-draw comparable with eps = 0 runs.
