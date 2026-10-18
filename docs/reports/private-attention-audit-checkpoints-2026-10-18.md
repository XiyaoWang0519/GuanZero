# Private cross-attention on m7 and u86800 (October 18, 2026)

Status: measured on CPU. No training, no GPU, no change to the default
`private_qk_norm` (`none`). m9 is not on this machine and was not replaced
with another save. A later same-day pass, appended below, compares `none`,
`query` and `qk` on these weights without an optimizer step.

This is the checkpoint run asked for by
[the audit-tooling report](private-attention-audit-2026-10-18.md). It uses
`eval/private_attention_audit.py` unchanged. The 2026-10-15 diagnosis remains
the single-fixture result; this page is 48 self-play matches per sample.

## What was audited

Worktree `GuanZero-worktrees/private-attention-audit` on
`cursor/private-attention-audit-qk-norm-e2ef` (`516474c` plus this report).
The checkout at `Developer/Projects/GuanZero` stayed on
`codex/history-investigation-2026-10-17`. The paired-eval worktree was only
read, for the averaged file.

| actor | path | role | SHA-256 |
|---|---|---|---|
| m7 u73780 | `.work/velocity-milestones/m7-u73780.pt` | required | `fa10bd1ac34c91e173902be28bb22f8d31233de50f534fa89fdb6d092d74fefb` |
| u86800 | `.work/velocity-milestones/u86800.pt` | required | `639ccda15452ef101561e836f684f238cb985bfcf412edd48fb063e383d4c720` |
| avg4 | paired-eval worktree `.work/paired-mac-2026-10-18/avg4-u86800.pt` | extra only | `418a4b60b8817c4aaf64484a5b54773f277b925d308b98aded928d05d8387a1d` |

These are the same m7, u86800 and four-checkpoint average hashes as the
October 18 paired DanLM rerun. The average is u82460, u83948, u85560 and
u86800. It has no Adam state. `velocity-milestones/` has m1–m8 and u86800,
and no m9 file. Later lineage saves exist (through `u99200-6501M.pt`) and
were not used.

Each checkpoint's `model_config` omits `private_qk_norm`. The loader fills
the default `none`, which creates no query or Q/K LayerNorm. The audit sets
`query` or `qk` only on a `fresh:SEED:NORM` spec. This run passed checkpoint
paths, so it does not say what those arms would do to these weights.

## Setup

PyTorch 2.14.0, Python 3.12.12, 8 CPU threads. Encoder runtime is
`causal_sdpa`. Private-attention gradients use math SDPA. Both runs exited 0.

```sh
PYTHONPATH=python:oracle:. python -m eval.private_attention_audit \
    --threads 8 --out .work/attn-audit-ckpt-2026-10-18 \
    --sampler <u86800.pt> --save-sample .work/attn-audit-ckpt-2026-10-18/sample.pt \
    <m7-u73780.pt> <u86800.pt> <avg4-u86800.pt>
```

Defaults: 32 envs, 1,500 steps, seed 20261018, 48 matches, 8 matches per
minibatch, 48 rows per match. A second run used `--sampler <m7-u73780.pt>`
and wrote `.work/attn-audit-ckpt-2026-10-18/m7-sampler/`. No `--stress` and
no `--adam-steps`.

Each sample is 2,304 decisions, 6 minibatches, 48 decisions per match.
2,256 decisions see at least one token. The u86800 sample has 1,589
tribute-round decisions and rounds 0–17. The m7 sample has 1,626
tribute-round decisions and rounds 0–20. Explicit recomputation matches
`decision_states` with max |Δstate| = 0 on every actor. DanLM was not an
opponent; every seat is the sampler.

## Saturation holds

On both samples, m7 and u86800 are one-hot on every head and every decision
that sees a token (top probability > 1 − 1e−6). That is true in round 0,
rounds 1–3, rounds 4–7, rounds 8+, tribute rounds and non-tribute rounds.
No head has a top-1/top-2 gap below one FP32 ulp. Minimum gaps are 28–72
nats on the softest head and 10⁴–10⁷ nats on the others.

| actor | sample | BOS heads (share 1) | other heads | distinct landings / match | q cosine (min head) | query norm (mean part) |
|---|---|---|---|---:|---:|---|
| m7 | u86800 | 0, 1, 3, 4, 5, 6 | h2, h7 | 8 and 4 | 0.999805 | 1,768 (1,730) |
| m7 | m7 | 0, 1, 3, 4, 5, 6 | h2, h7 | 8 and 5 | 0.999804 | 1,759 (1,728) |
| u86800 | u86800 | 0–6 | h7 | 7 | 0.999864 | 2,652 (2,619) |
| u86800 | m7 | 0–6 | h7 | 7.5 | 0.999862 | 2,629 (2,607) |

Mean-q argmax matches the real argmax on at least 0.999 of rows (1.0 on the
BOS heads). q/k singular values are σ1 ≫ σ2: m7 224 / 19.8 and 187 / 25.2;
u86800 234 / 19.6 and 198 / 25.5.

Policy and outcome q/k gradients are exactly 0 on 5 or 6 of the 6
minibatches for every head. The exceptions are the non-BOS heads, and the
largest such q gradient is 4e−20 (m7 head 7 on the u86800 sample). Median
q/k gradient is 0. v_proj and out_proj are not zero (policy medians about
0.01–0.12).

Adam √v below 1e−20, read from the checkpoint, is 0 on every m7 q/k head.
On u86800, q_proj heads 0, 1, 4 and 5 are entirely at that floor (√v about
8e−22), head 3 is 22.5% of entries, head 2 is 0.05%, and heads 6 and 7 are
above it (√v 3e−13 and 1e−5).

## History content barely moves the policy

Decisions in round ≥ 1 (1,880 on each sample) were re-scored with earlier
rounds swapped or shuffled, and with the whole stream shuffled. Identity
and round-0 TV are 0.

| actor | sample | swap TV median / max | greedy changes | attention-output relative change |
|---|---|---|---:|---:|
| m7 | u86800 | 5.6e−4 / 0.067 | 9 | 0.023 |
| m7 | m7 | 5.1e−4 / 0.070 | 7 | 0.021 |
| u86800 | u86800 | 2.9e−4 / 0.079 | 6 | 0.008 |
| u86800 | m7 | 2.7e−4 / 0.038 | 5 | 0.008 |

The non-BOS heads do change landing under those shuffles (argmax-moved
share 0.85–0.99 among non-BOS heads). That does not move the action
distribution: greedy changes are at most 9 of 1,880 decisions.

The average, included only as an extra, matches this picture. On the
u86800 sample, head 7 is one-hot on 2,255 of 2,256 rows (minimum gap 10.8
nats). On the m7 sample every head is one-hot on every row. Swap TV median
is 1.6e−4 and 1.5e−4. There is no Adam √v to read.

## What `private_qk_norm` did in the first command

Nothing to the measured actors. That command passed checkpoint paths, and
the audit attaches `query` or `qk` only to a `fresh:` spec. The default
stayed `none`. The comparison on these same weights is
[below](#norm-arms-on-the-trained-weights).

## Limits

Two self-play samples, not the October 15 fixture and not DanLM deals.
Saturation and the dead q/k Jacobian are properties of these weights on
these states. The small policy TV says the sampled decisions are
insensitive to earlier-round token contents; it is not a playing-strength
result. Dead ReLU shares (private block about 74–77% never active on the
u86800 sample) are for this sample only.

## Norm arms on the trained weights

Same two samples, same audit code, no optimizer step. For `query` and `qk`
the checkpoint is loaded into a second actor whose `private_qk_norm` is that
arm. The new LayerNorm weights are the identity init (scale 1, bias 0);
every saved parameter matches the `none` actor. `HistoryPolicyConfig`'s
default remains `none`. The `none` column reproduces the one-hot shares and
exact-zero q/k gradients above. Math-SDPA policy-loss gradient norms are the
same measurement as the first run. Explicit state agrees with
`decision_states` within 5e−6.

One-hot means top probability above 1 − 1e−6. Ranges cover the eight heads
and both samples.

| actor | arm | one-hot share | median top probability | heads with exact-zero q and k grad | minimum q cosine |
|---|---|---|---|---|---:|
| m7 | none | 1–1 | 1 | 6 of 8 (the other two are zero on 4–5 of 6 minibatches) | 0.99980 |
| m7 | query | 0.84–1 (h7), 0.94–0.96 (h2), else 1 | 1 on every head | h2 and h7 nonzero; the other six stay exact zero | 0.9933 |
| m7 | qk | 0–0 in every stratum | 0.003–0.020 | none (nonzero on all 6 minibatches) | 0.99981 |
| u86800 | none | 1–1 | 1 | 7 of 8 (h7 is zero on 5 of 6 minibatches) | 0.99986 |
| u86800 | query | 0.057–0.060 on h7, else 1 | 0.94–0.95 on h7, else 1 | h7 nonzero; the other seven stay exact zero | 0.9929 |
| u86800 | qk | 0–0 in every stratum | 0.003 on six heads; h2 0.26; h7 0.08–0.09 | none (nonzero on all 6 minibatches) | 0.99986 |

**`qk` stops the one-hot heads.** No head on either checkpoint has a row
above 1 − 1e−6, in any round bucket or tribute stratum. Median top-1
|score| falls from 1e5–1e8 to 2.8–4.0. Six heads become nearly flat (median
top probability about 0.003, entropy about 5.83). They are not all uniform:
u86800 head 2 still has median top probability 0.26 and a 99th percentile of
0.95–0.96 (maximum 0.994), and m7 heads 2 and 7 reach about 0.98 on the tail
while their medians stay near 0.02.

**`query` does not.** It LayerNorms the q_proj input, which cuts that
input's norm from about 1,760–2,650 to 11.3, and it cuts median top-1
|score| by roughly 200–600×. The large q/k weights still saturate six or
seven heads. Only the heads that leave one-hot acquire a q/k gradient.

**q/k gradients become nonzero under `qk`, on every head and every
minibatch.** Policy-loss medians per head are about 2e−9 to 8e−5 for q and
up to 8e−4 for k. The whole q_proj median is 3e−5 to 9e−5. Under `query` the
still-one-hot heads keep an exact-zero q and k gradient. The heads that
soften do not: m7 h2/h7 and u86800 h7 have median q gradients from 1e−5 to
0.013.

**The query direction does not start depending on the decision.** Cosine is
measured on the projected per-head q, after the arm's norm. Under `qk` the
minimum stays 0.99981–0.99986, the same as `none`. Under `query` it falls
only to 0.993. The pre-projection query is still a large constant plus a
smaller residual (mean norm 1,730 of 1,768 on m7; 2,619 of 2,652 on u86800).
A soft head can change its argmax anyway: mean-q agreement on u86800 head 7
drops to 0.57–0.59 under `query`, and the lowest `qk` head is about 0.90.
That is an argmax on a shallow score gap, not a decision-specific query.

This does not train the norm, and it says nothing about playing strength.
`qk` here is an identity-init wrapper around weights that were trained
without it.
