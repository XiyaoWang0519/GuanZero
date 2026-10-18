# Split private fusion and the pass b=4 habit A/B (October 18, 2026)

Status: **implemented and locally tested; training PLANNED, not launched.** The
six Vast runs need about $21.6 of worst-case credit; the account had $2.19 at
kit dry-run, so launch refuses. No training result exists yet. The main lineage
and its default architecture are unchanged.

## Why

The [fusion isolation](qknorm-fusion-isolation-2026-10-16.md),
[history stages](qknorm-history-stages-2026-10-16.md) and
[checkpoint audit](private-attention-audit-checkpoints-2026-10-18.md) reports point
at two mechanical properties of `HistoryActor.decision_states`:

- The private query `q = private(obs) + seat` enters `q_proj` unnormalized (norm
  ~1,760-2,650 on m7/u86800, dominated by a shared mean vector), and the heads are
  one-hot with exact-zero q/k gradients.
- `attention_norm(q + r)` sums a query of RMS ~31 with a reader output of RMS ~0.4,
  so the reader is small next to q, and the per-row centering is driven by q.

## What was implemented (commit `a2a9ffc`, local branch `fusion-split-habit-2026-10-18`)

`HistoryPolicyConfig.private_fusion` / `--private-fusion sum|split`, default `sum`
(bitwise unchanged, omitted from checkpoint records). With `split`:

```
h     = attention_norm(q)            # q_proj reads the centered, normalized query
r     = out_proj(attend(q_proj(h), k, v))
state = h + reader_norm(r)           # the reader has its own LN and gain
state = output_norm(state + FFN(state))
```

Combined with the existing `private_qk_norm=qk` (per-head q/k LayerNorm) this is the
tested arm `split+qk`. `split+query` is rejected; merged snapshot inference and the
Botzone NumPy actor refuse `split`. `bench/batched_rollout.py` and
`eval/private_attention_audit.py` use the actor's new `private_query` /
`reader_input` / `fuse` methods.

Two consequences of the design:

- Per-row LayerNorm fixes scale and centering. It cannot remove a query direction
  shared across decisions (the audit's `query` arm only lowered min q cosine to
  0.993). Whether that collapse recurs under training is measured, not assumed.
- With identity q/k norms, logits are bounded by sqrt(depth) = 4, so attention
  starts near uniform (below).

`train/history_fusion_migrate.py` converts a `sum`/`none` checkpoint to `split+qk`:
every source parameter copied, q/k norms identity, `reader_norm.weight = c *
attention_norm.weight`, bias 0. c is the median over decisions of
`rms(r - mean r) / sqrt(var(q + r) + eps)` for the legacy actor. At init the reader
therefore enters with the relative magnitude it had inside the legacy LayerNorm,
but under its own learnable gain. Source Adam moments are kept; the six new
parameters are trailing and start with fresh moments.

`eval/history_habit_eval.py` gains `--per-env N` (score match slots 0..N-1 of every
env, with (team, z) a pure function of the slot, so the opponent draw is paired
across checkpoints and short matches are not favoured) and `--flip-style` (ORACLE
reads the wrong z; opponents unchanged). `bench/fusion_habit_mechanism.py` computes
reader share, z readouts against a match-permutation null and opposite-z vs same-z
history-swap sensitivity.

### Conversion of u2623

Source u2623 (`6fd3b94d...`), output `53ca22cb...`; self-play calibration sample
seed 20261018 (1,504 rows), independent held-out sample seed 20261019 (1,692 rows).

| quantity | value |
|---|---|
| reader gain c (median) | 0.1404 (q05 0.090, q25 0.113, q75 0.169, q95 0.204) |
| converted vs u2623 policy TV | mean 0.0073, median 0.0027, p95 0.029, max 0.180 |
| greedy-action changes | 0.74% of 1,627 multi-candidate rows |
| reference: u2623 with reader output zeroed | TV mean 0.030, p95 0.118; greedy 3.6% |
| converted one-hot share (8 heads) | 0 on every head |
| converted median top attention prob | 0.003-0.013 (near uniform) |

The starting-policy shift is about a quarter of the reader-ablation scale. The
converted reader starts as a near-uniform pool over the visible tokens and must
learn its q/k norm gains to sharpen.

### Tests

`tests/test_history_private_fusion.py` (15): bitwise `sum` on all five attention
paths; `split` (with and without qk) equals the explicit formula on every path; KV
cache collection matches learner recompute, including a split habit trainer with the
production flags; rejects, CLI, checkpoint round-trip, resume; nonzero gradients on
q_proj/kv_proj/reader_norm/q_norm/k_norm; converter (exact copy, gain formula
recomputed independently, `habit_init_player` accepts the output, one CPU update).
Also passed: the qk-norm, habit, snapshot-batch, Botzone, private-attention-audit,
batched-rollout, history model/config/PPO/aux/inference/rollout suites. The CUDA
gate `test_cuda_habit_views_start_on_policy[full-split]` has only skipped locally;
it is a required pod gate.

## Planned experiment

Sealed protocol: `.work/fusion-habit-2026-10-18/protocol.md`; kit:
`.work/fusion-habit-2026-10-18/kit/` (source `a2a9ffc`, built by `make_kit.py`;
dry-runs pass except the credit check).

| arm | view | architecture | init |
|---|---|---|---|
| FULL-legacy | full | sum / none | u2623 |
| FULL-split | full | split / qk | converted u2623 |
| ORACLE | full + true z | sum / none | u2623 |

Pack: frozen u2623, axis pass, strength 4. u2623 recipe on one RTX 4090 with 4 DDP
ranks; 611 updates (40,042,496 collected decisions), checkpoints every 76 updates;
seeds 2026101801/02 shared by all arms. Worst case $3.61 per run at a $0.55/h ceiling
(6 h 30 m hard cap), $21.6 for all six; expected roughly half that at the cheapest
current offers.

Primary contrast: mean over seeds of R(FULL-split) - R(FULL-legacy) at the endpoint,
R = mean round `seat_return`, 3,000 slot-paired matches per checkpoint. "Better" or
"worse" needs both seed differences in the same direction and the pooled 95% interval
excluding zero; otherwise unresolved. ORACLE - FULL-legacy, ORACLE correct-vs-flipped
z, FULL-vs-ROUND view and the gain over each arm's start are reference contrasts.

What this test can and cannot show:

- Pass b=4 is identifiable within the current round (stage 0a), so the test is about
  public-history use through the private reader, within-round included, not
  cross-round memory.
- The +8.96pp of the [round-value study](history-habit-round-2026-10-17.md) is an
  oracle-z search planner against a fixed 50/50 blind planner. It is not a ceiling or
  a target for a learned FULL policy.
- Two seeds cannot estimate training variance; intervals are evaluation noise only.
- This is a separate causal experiment, as the
  [habit plan](habit-attention-ab-plan-2026-10-17.md) requires for fusion changes.
  It does not replace stages 1a/1b there, and a positive result would not by itself
  justify adopting the fusion in the main lineage.

## Artifacts (local, `.work/` is ignored)

- `.work/fusion-habit-2026-10-18/u2623-split-qk.pt`, `migration-report.json`
- `.work/fusion-habit-2026-10-18/protocol.md`, `eval_endpoint.sh`, `kit/`
- `.work/fusion-habit-2026-10-18/smoke/` (2-update CPU smokes of all three arms)
