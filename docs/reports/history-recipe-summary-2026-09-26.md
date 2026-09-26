# History Transformer recipe campaign: summary, September 26, 2026 UTC

Goal: move the from-scratch history Transformer's development curve above the
plateau of the batch-size runs (-2.1..-2.4 net levels/round against frozen B11,
entropy below 0.3, no full-match wins) without giving up model capability
(no lower precision, history truncation, smaller legal action set, MLP teacher
or filter) and within a cumulative RunPod cap of $30.

**Development result:** the curve now sits well above that plateau. Two PPO
epochs is a seed-robust improvement: four seeds against six control seeds,
+0.37 [+0.25, +0.49] against B11 over updates 650-900. With current-policy-only
self-play or data-parallel collection added, two lineages reached about
-1.0..-1.3 against B11 and +1.2..+1.5 against the engine greedy heuristic,
and held a +0.5..+0.8 lead over the same-seed control through update 1,500.
**Strength result: none.** Every snapshot still loses on average to both frozen
MLPs. Full-match wins appeared, at 1-2 of 16 in isolated snapshots (several
of them on the recipe arms), which is far from a win-rate claim.
**Cost:** $4.65 estimated over six allocations (account balance fell by $4.46).

## What was found, in order

1. **The old plateau is the greedy heuristic's level** (local CPU). B11 beats the
   engine greedy bot by +2.02 on the same 64 deals, and the batch-size A arm
   scored +0.27 against greedy. A greedy-scale curve was added to every
   evaluation because scores against B11 compress near the -3 floor.
2. **Pure self-play needs many samples** (local CPU). An M1-architecture MLP
   trained by pure self-play DMC was still -0.63 against greedy after 1.85M
   samples, so the Transformer was not less sample efficient at that scale.
   This calibration model was evaluation-only and never touched a Transformer run.
3. **This model trains faster on CPU than on the rented GPU.** The 64-wide model
   is dominated by kernel launches on CUDA (about 730 decisions/s on an RTX 4090
   pod) against about 2,900 on this Mac's CPU and 450-600 per process on EPYC
   pods. All runs trained on pod CPUs (the FP32 reference path covered by the
   CPU/CUDA equivalence tests) with 3-8 arms in parallel. The in-pod CPU control
   matched the CUDA control's curve against both frozen MLPs.
4. **Training-seed noise is about 0.2-0.35 levels/round per window.** Two runs
   of the same configuration differed by that much. Every later run had in-pod
   controls, and conclusions use pooled seeds (`pooled.py`).
5. **Run 1:** entropy 0.03, width 128 / 4 layers, 2 epochs and 16 recent
   snapshots, one variable each. Only 2 epochs helped; 16 snapshots hurt.
6. **Run 2:** added `train/history_ddp.py` (data parallel, gradient all-reduce,
   parameter checksum each update). Six ranks raised throughput 5.3x per lineage,
   but 5x the rows gave only +0.1..+0.2: sample count alone was not the limit.
7. **Run 3** (after one no-machine pod and two HTTP-500 creates): 2 epochs
   replicated on two more seeds; 4 epochs, lr 6e-4 and 1-match minibatches did
   not help. The two-epoch benefit is not simply "more optimizer steps".
8. **Run 4** (2 epochs as the new reference): 2 epochs replicated a fourth time;
   data parallel (+0.30) and self-play only (+0.24) accelerated early learning on
   top of it; GAE lambda 1.0 and critic lr 1e-3 were worse.
9. **Run 5:** resumed the SP, DP and control lineages across pods (identity
   checks passed; required moving ten iCloud conflict copies out of the source
   tree first). The recipe lineages kept their lead but levelled off near -1.1.

## Every allocation and its cost

| Run | Kit | Pod | Hardware | Duration | Est. cost | Outcome |
|---|---|---|---|---:|---:|---|
| 1 | recipe1 | `dcw1netcoxtjc5` | RTX 3090 Secure, EPYC 7H12 | 84.3 min | $0.709 | 5 arms; 2 epochs positive |
| 2 | recipe2 | `nb191swewkrtkh` | RTX 3090 Secure, EPYC 7H12 | 84.5 min | $0.710 | DDP engineering + A/B |
| 3 | recipe3 | `dgvo65fruljrb7` | RTX 5090 Community | 15.2 min | $0.176 | never got a machine; deleted by monitor; balance unchanged |
| — | recipe3b | none | RTX 2000 Ada Secure | — | $0 | two HTTP 500 creates, reconciled, nothing created |
| 3c | recipe3c | `nq3gyo7f48oco7` | RTX PRO 4500 Blackwell Secure, EPYC 7763 | 83.8 min | $1.012 | 8 arms; 2 epochs replicated |
| 4 | recipe4 | `j2irwcd88p1d75` | RTX PRO 4500 Blackwell Secure, EPYC 7713P | 84.6 min | $1.022 | one HTTP 500 first; 6 arms on 2 epochs |
| 5 | recipe5 | `r30ec2896ztpuf` | RTX PRO 4500 Blackwell Secure, EPYC 7713P | 84.3 min | $1.017 | lineage continuation + SPD |
| | | | | **Total** | **$4.646** | balance $32.3145 → $27.8538 (−$4.46) |

Each allocation had its own kit, manifest, approval record ("用户预授权，累计上限
30 美元"), per-run cap ($0.60-$1.45, under 2 hours) and ledger entry in
`.work/runpod-ledger-2026-09-26.json`. Before every create, the account and pod
list were checked (zero pods, $0/hour). After every run, downloads were SHA-256
verified before deletion, the monitor confirmed deletion, and an independent
readback found zero pods and $0/hour. The local guard and `caffeinate` ran for
the whole of every allocation. No readback was ever nonzero, and the $25 stop
line was never approached.

## Which curve is best

Development deals only (the frozen 64 deals, two swapped legs; the final-test
seed was never opened). No best checkpoint was promoted; whole curves are in
each kit's `results/curves.json`.

| Recipe (lineage) | Best window mean vs B11 | vs greedy | Full-match wins |
|---|---:|---:|---|
| Original control, 6 CPU seeds | -1.83..-2.07 (650-900) | +0.4..+0.8 | 1/16 once (run 1 L, update 700) |
| Two epochs, 4 seeds | -1.33..-1.85 (650-900) | +0.75..+1.09 | 1/16 twice |
| Two epochs + self-play only (run 4 SP → run 5 SPc) | **-1.07 (950-1,200)** | +1.33 | 1/16 in 11 snapshots |
| Two epochs + 3-rank DDP (run 4 DP → run 5 DPc) | **-1.05 (950-1,200)** | +1.33 | 1-2/16 in 9 snapshots |
| Two epochs + self-play + 2-rank DDP (SPD) | -1.26 (350-600) | +1.20 | 1/16 in 5 snapshots |

The best single development points are SP -0.98 (update 1,300) and DP -1.00
(update 1,200) against B11. They are single snapshots on 64 deals, reported
only to show the ceiling of the curves.

## Engineering results versus strength results

- **Engineering (verified):** `train/history_ddp.py` (data-parallel trainer with
  resume; tests: one rank bitwise equals the base trainer; two ranks stay
  checksum-identical, average real-minibatch gradients and resume one lineage).
  All history tests pass locally (112 passed, 6 CUDA skips) and on every pod
  (98-101 passed including CUDA). The CPU-on-pod training path, the kit tooling
  with a cumulative ledger, and cross-pod lineage resume also work.
- **Development-curve results (A/B on development deals):** two epochs beats the
  control (4 vs 6 seeds). Self-play only and data parallel help on top of it
  early (single seeds). The rest are listed above.
- **Strength:** no promotion. The best recipes still lose about one level per
  round to B11 and win at most 2 of 16 full matches in any snapshot.

No production default was changed. `HistoryPPOConfig` already defaults to
`epochs=2`; the old kits had overridden it to 1. The KV/SDPA flags stay opt-in.

## Recommended next steps

1. **Adopt the development recipe for future runs:** two epochs, current-policy
   self-play (snapshot probability 0) or data parallel, KV + SDPA, CPU training
   on many-core pods. Keep an in-pod control and at least two seeds for any claim.
2. **The new plateau (-1.0..-1.3) calls for a new axis, not more of the same.**
   Candidates in order of expected value: (a) capacity with the new recipe
   (width 128 was only tested under the old one-epoch recipe and was slow; with
   data parallel it becomes affordable); (b) a looped or deeper decision module
   (T6); (c) the next-event auxiliary objective on own-lineage self-play (T7);
   (d) a KL- or entropy-targeted PPO step instead of a fixed coefficient, since
   the two-epoch benefit and the four-epoch harm suggest step size matters.
3. **Throughput:** per-process speed on EPYC pods (~500 decisions/s) is about a
   sixth of this Mac's. Faster-core hosts, or batching the per-entry KV append,
   would multiply samples per dollar without changing the model.
4. **Evaluation:** keep the greedy yardstick next to B11 until the curves reach
   parity, and add full-match pairs (more than 8 seeds) once wins become regular.
5. The final-test set stays sealed until a recipe and endpoint are predeclared.

## Files

Reports: `history-recipe1-2026-09-26.md` … `history-recipe5-2026-09-26.md`.
Kits: `.work/runpod-history-recipe{1,2,3,3b,3c,4,5}-2026-09-26/`. Tooling and
pooled analyses: `.work/history-recipe-kits/`. Local diagnosis (greedy scale,
DMC calibration, profiles): `.work/history-diagnosis-2026-09-26/`. Ledger:
`.work/runpod-ledger-2026-09-26.json`.
