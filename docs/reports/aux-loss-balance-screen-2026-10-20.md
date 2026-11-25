# Aux-loss balance screen: next vs PPO on real minibatches (2026-10-20)

Status: **CPU screen complete.** On four Velocity-saved PPO minibatches from the
fault-period main run, weighted next-token gradients on the public encoder
dominate PPO by **25×–119×** (mean **58×**). Encoder cosine(next, PPO) is
negative on **3/4** batches (conflict rate 0.75). Cheap single-knob next-
coefficient ablations (belief/outcome held at `aux_coef=0.1`) restore encoder
alignment with PPO as next is lowered or turned off. **No optimizer step, no
training job, no main-lineage change.** Tooling lands in this PR; a train-time
`next_coef` knob is proposed but not implemented.

## Question and boundary

Campaign parallel screen (2026-10-20): does next dominate and/or conflict
with PPO on real minibatches? Starting point:
[history-training-audit-2026-10-17](history-training-audit-2026-10-17.md)
reported ≈53× next over PPO on one 328-row fixture and omitted belief.

This round:

* Uses **four** rank `post_reduce` fixtures from
  `experiments/stack-gradient-2026-10-13` (same save family as the audit).
* Rank-1 SHA matches the audit fixture:
  `9c3d1018ac52453852536ff5ef497eb1dbcbd553f02e199b30c9570e6c152cd8`.
* Includes **belief** (stored `hidden` labels) and **outcome** beside next and
  PPO.
* Measures public-encoder gradient norms and pairwise cosines; never steps
  weights (parameters and `.grad` verified unchanged).
* Fixtures embed the collecting actor; this is on-policy math for those
  batches, not a reanchored u86800 check and not the buggy efficient-CUDA
  backward of the incident.

## Method

Tool: `bench/aux_loss_balance_screen.py` (math SDPA, CPU, one thread).

For each fixture, unweighted losses `ppo` / `next` / `belief` / `outcome` are
differentiated separately. Dominance uses the configured weight
`aux_coef=0.1` on each aux head (matching training). Pairwise cosines use the
flat concatenated public-encoder parameter gradients. Conflict means cosine
&lt; 0.

Single-knob ablations form
`total = ppo + next_coef·next + 0.1·belief + 0.1·outcome` with
`next_coef ∈ {0.1, 0.05, 0.03, 0.01, 0.0}` and report encoder
`cos(total, ppo)` and the PPO share of `‖g_encoder‖`. Belief, outcome,
response, and PPO hyperparameters are not changed.

## Results

| fixture | decisions | next/PPO (×) | cos(next,PPO) | conflict | cos(total,PPO) @0.1 | @0.03 | @0 |
|---|---:|---:|---:|:---:|---:|---:|---:|
| rank0 | 369 | 119.1 | −0.022 | yes | −0.014 | +0.003 | +0.297 |
| rank1 | 328 | 53.3 | −0.043 | yes | −0.011 | +0.066 | +0.802 |
| rank2 | 406 | 25.6 | −0.029 | yes | +0.016 | +0.120 | +0.833 |
| rank3 | 405 | 35.4 | +0.047 | no | +0.071 | +0.123 | +0.325 |

Means across the four batches:

| quantity | ppo | next | belief | outcome |
|---|---:|---:|---:|---:|
| weighted encoder ‖g‖ | 0.00595 | **0.255** | 0.00063 | 0.00949 |
| dominance vs PPO | 1 | **58.4×** | 0.12× | 1.89× |
| mean cos with PPO | 1 | **−0.012** | −0.171 | +0.155 |

* Next dominates PPO on **4/4** fixtures.
* Next conflicts with PPO on **3/4** (rate 0.75). Rank3 is weakly aligned but
  still next-dominated (PPO is only ~2% of the combined encoder update at
  `next_coef=0.1`).
* Belief is small on the encoder (~0.12× PPO) and often conflicts with PPO
  (rate 0.5); it does not drive the encoder update.
* Outcome is order-1 relative to PPO (mean 1.9×) and is not the subject of
  this single-knob proposal.
* At default weights, mean `cos(total, PPO) ≈ 0.015` and PPO contributes
  ~2.4% of encoder ‖g‖. Turning next off raises mean cosine to **0.56** and
  PPO’s share to ~47%.

Rank1 reproduces the audit’s 53.3× figure bit-for-bit on the same SHA.

## Proposal (train-time, not run here)

Single-knob **next** ablations are warranted. Shared `aux_coef` cannot do
this: lowering it also scales belief and outcome. A dedicated `next_coef`
(default = current `aux_coef`) with belief/outcome left on `aux_coef` is the
minimal knob.

Suggested arms (freeze everything else):

1. `next_coef=0.1` (control / current)
2. `next_coef=0.03` (lower)
3. anneal `0.1 → 0` over ~1k updates
4. `next_coef=0` (off; heads may remain for eval)

Acceptance for a later train job is outside this screen: holdout strength /
history-use metrics, not encoder cosine alone. This report does **not**
authorize GPU training.

## Tooling and checks

* `bench/aux_loss_balance_screen.py` — multi-fixture screen + summary JSON
* `tests/test_aux_loss_balance_screen.py` — cosine/mask/summary helpers
* Related: `tests/test_history_training_signal.py` still passes (7 tests in
  this pair)
* Local evidence (ignored): `.work/aux-loss-balance-2026-10-20/results/summary.json`

## Limits

* Four fault-period rank fixtures, one save family; not a sweep over mature
  healthy checkpoints or post-recovery u86800 on-policy batches.
* Cosine on concatenated encoder parameters is a direction diagnostic, not a
  proof of win-rate harm or of optimal coefficients.
* CPU math SDPA only; not a replay of the historical efficient-CUDA fault.
* No parameter update was performed.
