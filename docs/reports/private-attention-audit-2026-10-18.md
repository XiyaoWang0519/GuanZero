# Private cross-attention: multi-batch audit tooling and Q/K-norm prototype (2026-10-18)

Status: **implemented and measured on random-init and synthetic actors only**
on the CPU VM that wrote this page. The trained-checkpoint measurement is
[private attention on m7 and u86800](private-attention-audit-checkpoints-2026-10-18.md).
The numbers below are still the random-init and synthetic-stress results.

The trained checkpoints this audit is meant for (`m7-u73780`, `u86800`, `m9`/latest
`runs/main`) and the 2026-10-14 rank-1 fixture were **not available** on the
CPU VM used here: there is no `.work/` or `runs/` directory, and a filesystem
search for `u86800`, `u73780`, `m7*.pt` and `rank1-fixture` found nothing. No
checkpoint was substituted. CPU only. No GPU rented, no training job launched,
main lineage untouched.

Follows [the 2026-10-15 private-attention diagnosis](private-attention-diagnosis-2026-10-15.md)
(one fixture: 8 matches, 328 decisions), which asked whether its saturation,
BOS landing and zero q/k gradients hold beyond that one minibatch.

## What was built

1. **`eval/private_attention_audit.py`**, a multi-minibatch CPU audit. It collects
   one self-play sample with a chosen sampler actor: whole matches, completed
   rounds only, rows subsampled evenly in time per match, and learner-style
   minibatches of whole matches. It then measures every listed actor on that same
   sample (`--save-sample`/`--load-sample`). Per head and decision:
   - FP64 entropy, top probability, top-1 minus top-2 gap, the gap in FP32 ulps,
     and the top-1 |score|;
   - landing (BOS, last visible token, current round, earlier round) and the
     number of distinct positions per match;
   - mechanism: pairwise cosine of q across decisions, the argmax under the
     minibatch-mean q, the q_proj input's norm and its mean/residual parts,
     and the singular values of q/k;
   - history sensitivity. Decisions in round ≥ 1 are re-scored with their
     earlier rounds' token contents shuffled or swapped for another match's
     tokens (round, phase and length kept), with a whole-stream shuffle as a
     positive control and an identity control.
   - math-SDPA gradient norms per module and per q/k head, for the ratio-1 PPO
     policy loss and the outcome loss;
   - Adam √v per head, including the share of entries below the 1e-20
     subnormal floor (read from the checkpoint's `optimizer['actor']`);
   - dead and rare ReLU units in the private block and in every encoder FFN.

   Everything is stratified by round bucket (0, 1–3, 4–7, 8+) and by tribute
   round, meaning the decision's current round has a public exchange token. On
   every minibatch the explicit recomputation is checked against
   `HistoryActor.decision_states` (max |Δstate| ≤ 1.2e-6 in all runs below). It
   writes `audit.json` and `summary.md`. The encoder uses the existing
   `causal_sdpa` runtime path (no T×T mask; needed for ~2,200-token matches in
   15 GB) and is encoded in chunks of streams. Only the private attention runs
   under math SDPA, as in the 10-15 report.
2. **`HistoryPolicyConfig.private_qk_norm`** (`none` | `query` | `qk`, CLI
   `--private-qk-norm`, default `none`):
   - `query`: a LayerNorm(width) on the q_proj input. The residual
     `attention_norm(query + attended)` still reads the raw query.
   - `qk`: a LayerNorm over the head depth, shared across heads, on q and on k
     after projection, in the style of QK-norm. Logits are bounded by the
     LayerNorm gains.

   All private attention paths (per-match loop, batched match, independent
   rows, length groups, windowed control, KV-cache rollout) go through the new
   `project_query`/`project_keys_values` helpers. With `none` nothing is
   created: `config_record`, state-dict keys, initialisation and outputs are
   unchanged, and old checkpoints load as before. Arms draw no random numbers
   (LayerNorm init), so all other actor and critic parameters match the
   baseline at the same seed. Merged snapshot inference and the Botzone NumPy
   actor refuse the new arms. The arm is not in `RESUME_OVERRIDES`, so it is a
   fresh-lineage A/B arm only.
3. Tests: `tests/test_history_qk_norm.py` (14) and
   `tests/test_private_attention_audit.py` (9). The full Python suite passed:
   1393 passed, 332 skipped (CUDA).

## Measurements (random init and synthetic stress; not trained checkpoints)

Sample: main-lineage architecture (width 128, 4 layers, 8 heads, aux
`next,belief,outcome`), sampler `fresh:1`, 32 envs × 1,200 steps, seed
20261018. That gives 48 matches in 6 minibatches of 8 matches, 48 rows per
match, so 2,304 decisions. Rounds 0–15 (496 / 745 / 641 / 422 decisions in the
round buckets); 1,554 decisions in tribute rounds; streams up to 2,196 tokens.
The same sample is used for every arm and stress level.

**Stress** (`--stress MEAN:QGAIN:KGAIN`) gives fresh weights the trained regime's
numerics. It adds a constant of norm 2,700 to every private query (the 10-15
report measured a query norm of 2.7k dominated by its mean) and scales q_proj
and the k rows of kv_proj. All arms get identical weights. The norm modules are
extra parameters at their identity init.

### Saturation and gradients (all 2,256 decisions that see ≥ 1 token)

| setting | arm | one-hot share per head (min–max) | top-1 \|score\| median (head range) | heads with exact-zero q grad on ≥ 5/6 minibatches | q_proj policy grad median |
|---|---|---|---|---:|---:|
| init | none | 0–0 | 0.11–0.60 | 0 | 5.6e-5 |
| init | query | 0–0 | 0.11–0.58 | 0 | 5.3e-5 |
| init | qk | 0–0 | 0.38–1.69 | 0 | 2.0e-4 |
| stress 2700:50:20 | none | 0.996–1.0 | 4.0e4–2.4e5 | 2 (h0 0.83, h3 1.0) | 1.2e-7 |
| stress 2700:50:20 | query | 0.11–0.82 | 157–976 | 0 | 5.7e-7 |
| stress 2700:50:20 | qk | 0–0 | 0.50–2.75 | 0 | 1.7e-8 |
| stress 2700:200:100 | none | **1.0–1.0** | 8.0e5–4.7e6 | **8** (zero on 83–100%) | 8e-25 |
| stress 2700:200:100 | query | 0.93–0.99 | 3.2e3–2.0e4 | 0 | 3.8e-7 |
| stress 2700:200:100 | qk | 0–0 | 0.50–2.75 | 0 | 4.3e-9 |

At stress 2700:200:100 the baseline reproduces the 10-15 pattern on all 6
minibatches:
- **every head is one-hot in every stratum** (each round bucket, tribute and
  non-tribute rounds);
- the minimum gap is 29 nats;
- q cosine across decisions is 0.99997, and the mean-q argmax is unchanged on
  98.8–100% of rows;
- a median of 2–11 distinct landing positions per 48-decision match
  (depending on the head), i.e. landing is set by the match;
- q/k gradients are exactly zero on most minibatches.

### Adam probe (30 steps, lr 3e-4, fixed sample, stress 2700:200:100)

This is a numerics probe of whether q/k can move, not training.

| arm | q_proj / k_proj relative change | after probe: one-hot share (min–max) | after probe: heads with exact-zero q grad on some minibatch |
|---|---|---|---:|
| none | 1.3e-5 / 2.7e-5 | 0.999–1.0 | 8 (zero share 0.67–1.0) |
| query | 2.9e-5 / 8.4e-5 | 0.945–1.0 | 6 (zero share 0.17–1.0) |
| qk | 4.7e-7 / 2.3e-6 | 0–0 | 0 |

v_proj and out_proj moved about 2.3% in every arm.

### History sensitivity (1,808 decisions in round ≥ 1)

| setting | arm | swap_earlier TV median / max | attention-output relative change (swap) | argmax moved, non-BOS |
|---|---|---|---:|---:|
| init | none | 1.9e-6 / 2.0e-5 | 0.0045 | 0.36 |
| init | qk | 2.3e-6 / 2.5e-5 | 0.0050 | 0.36 |
| stress 200:100 | none | 4.1e-7 / 5.2e-6 | 0.18 | 0.40 |
| stress 200:100 | qk | 1.5e-8 / 2.3e-7 | 0.0052 | 0.41 |

The identity control gives TV ≤ 1.3e-7, and round-0 decisions are unchanged
(TV ≤ 1.5e-8) under every perturbation. Random-init policies are close to
uniform, so these TVs say nothing about trained history use.

Dead ReLU units at init (never active on the sample): private block 4.9%, encoder
layers 0–0.6%; rare (< 1% of rows) 7.2% and 1.8–8.4%. The 10-15 report's
"> 50% of private-block √v at the subnormal floor" is not checkable without
the checkpoint.

## Conclusions

**Measured:**

- The audit runs on 48 matches / 6 minibatches in under a minute per actor
  on 4 CPU cores (3 actors with all perturbations in 1 min 52 s), with ≤ 1.2e-6 agreement with the model's own decision states.
- At random init with the main-lineage architecture, no head is saturated in
  any round or tribute stratum (top probability ≤ 0.003). Landing varies by
  decision, and q/k gradients are nonzero on every minibatch. Saturation in the
  lineage is therefore learned, consistent with the 10-15 finding that m5 was
  already one-hot.
- With trained-like numerics on otherwise random weights, the baseline's
  saturation is global over this sample. All 8 heads are one-hot in every
  round bucket and tribute stratum on all 6 minibatches, and the exact-zero
  q/k gradient appears on most minibatches. Hard saturation is a property of
  the score scale, not of particular inputs.
- `qk` bounds the scores (top-1 |score| ≤ 2.75 at every stress level). It
  removes the exact-zero softmax Jacobian, and it stays soft and nonzero after
  the Adam probe. `query` alone does not. It cuts scores by about 250×, but
  large q/k weights still saturate heads (93–99% one-hot), and the Adam probe
  drifts back toward exact-zero gradients.

**Not answered / limits:**

- **Whether saturation is global for `m7`, `u86800` or the latest checkpoint is
  unmeasured on this VM.** The checkpoints were not available here. The synthetic result
  shows what the audit detects, not what the lineage does. The later Mac run is
  [private attention on m7 and u86800](private-attention-audit-checkpoints-2026-10-18.md).
- No arm makes q decision-dependent. With a dominant constant query component,
  the q cosine stays 0.99997 under `query` and `qk`. With `qk` the attention
  becomes a soft, match-dependent average rather than a decision-dependent
  read. Centring the query across decisions is a different change and was not
  tested.
- Under `qk` the q/k gradients scale like 1/‖W‖ (LayerNorm scale invariance):
  4.3e-9 at stress versus 2.0e-4 at init. Adam rescales them, but values below
  eps=1e-8 are damped (relative q/k change 4.7e-7 in the probe). So `qk` cannot
  repair an already-drifted checkpoint by itself. It is a fresh-lineage arm
  whose value depends on keeping weights moderate from the start.
- The stress scales q/k uniformly (σ1/σ2 ≈ 226/221), while the lineage has
  σ1 ≫ σ2 (234/20). Random-init policies are near uniform. The Adam probe is
  30 steps on a fixed sample. None of this predicts playing strength.

## Next step (requires the checkpoints; no GPU needed)

On a machine with `.work/` checkpoints, run (CPU only, minutes in total):

```sh
PYTHONPATH=python:oracle:. .venv/bin/python -m eval.private_attention_audit \
    --out .work/attn-audit-ckpt --sampler <u86800.pt> --save-sample .work/attn-audit-ckpt/sample.pt \
    <m7-u73780.pt> <u86800.pt> <latest runs/main checkpoint>
```

Use `--sampler <m7>` for a second sample if sampler dependence matters. Any
change to the main lineage's architecture, and any `qk` A/B lineage, needs
separate authorization.

## Artifacts (`.work/attn-audit-2026-10-18/`, not in the repo; copied with this run's artifacts)

| file | content | SHA256 first 12 |
|---|---|---|
| `sample.pt` | 6 minibatches, 2,304 decisions | `3c2516662500` |
| `init/audit.json`, `init/summary.md` | three arms, random init | `eb276defbcd6` |
| `stress-200-100/audit.json`, `summary.md` | stress 2700:200:100, Adam probe 30 steps | `b1a6a262326d` |
| `stress-50-20/audit.json`, `summary.md` | stress 2700:50:20 | `458e182110b2` |

Commands (PyTorch 2.14.1+cpu, 4 cores; `eval/private_attention_audit.py` at
this commit):

```sh
A="-m eval.private_attention_audit fresh:1 fresh:1:query fresh:1:qk"
$PY $A --out init --envs 32 --steps 1200 --matches 48 --minibatch-matches 8 --rows-per-match 48 --save-sample sample.pt
$PY $A --out stress-200-100 --load-sample sample.pt --stress 2700:200:100 --adam-steps 30
$PY $A --out stress-50-20 --load-sample sample.pt --stress 2700:50:20 --no-perturb
```
