# Recipe run 2: data-parallel collection versus two in-pod controls, September 26, 2026 UTC

Run 1 left open whether the plateau is simply sample-limited. This run adds an
opt-in data-parallel trainer (`train/history_ddp.py`) and compares it with two
single-process controls on the same pod. **Engineering result: six ranks raise
one lineage's throughput 5.3x with parameters verified identical across ranks.
Development result: 5.0x the learner rows moved the curve only about +0.1 to
+0.2 over the in-pod control, so raw sample count is not the main limit at this
scale.** No full-match wins; no strength claim.

## What was built (engineering, capability-preserving)

`HistoryDDPTrainer` subclasses the existing trainer. Every rank has its own
environments, event store, buffer and seat RNG; all ranks start from the same
`fresh_player(seed)` and one broadcast lineage. Each PPO minibatch step
all-reduces the actor and critic gradients (gloo, CPU), divides by the number
of ranks that had a real minibatch at that step, clips the averaged gradient
and applies the same Adam step. A parameter checksum is compared across ranks
after construction and after every update; snapshots are taken at the same
updates on every rank. Architecture, FP32, full public history, full canonical
candidates, reward/GAE and own-lineage population rules are unchanged.
`world_size = 1` delegates to the base trainer.

Tests (`tests/test_history_ddp.py`): one rank is bitwise identical to
`HistoryTrainer` over two updates (losses, sample counts, every parameter);
two ranks average real-minibatch gradients correctly, keep identical checksums
and snapshot digests, take the same number of optimizer steps, learn, report
global sample counts equal to the sum of ranks, and collect different
trajectories. Local: 111 history tests passed, 6 CUDA skips. Pod gate: 100
passed (CPU and CUDA). The larger effective batch changes training dynamics,
so the development A/B below is required before it counts as evidence.

## Design and identity

| Arm | Configuration | Updates | Learner rows (all ranks) |
|---|---|---:|---:|
| D | control config, 6 ranks x 32 envs (12,288 decisions/update, 16 steps/update of 6x larger minibatches) | 738 | 5,478,067 |
| L2 | control config, single process, seed 2026092603 | 904 | 1,111,271 |
| L3 | control config, single process, seed 2026092604 | 907 | 1,123,804 |

Control config = batch-size kit arm A (32 envs, KV + SDPA, lr 3e-4, entropy
0.01, 1 epoch, 2 matches/minibatch, snapshots every 2 updates, 4 recent,
p = 0.5), trained on CPU. Each arm had a 4,800-second cap; D used 6 x (2 engine
+ 2 torch) threads, controls 2 + 2.

| Identity | Value |
|---|---|
| Source SHA-256 | `16bbd654…0406f` (adds `train/history_ddp.py` and its test; no existing file changed) |
| Archive / manifest SHA-256 | `fb26f5fc…28f3` / `dfa02564…` (full hashes in the kit) |
| Host | AMD EPYC 7H12, cgroup quota 31.1 CPUs, RTX 3090 Secure, EU-CZ-1 |
| Final slim checkpoints | D `fc7dc519…`, L2 `6d555daa…`, L3 `fa3e4d86…` |

Throughput: D averaged about 6.5 s per update for 12,288 decisions; the
controls about 5.3 s for 2,048. Per lineage and wall-clock hour that is
5.3x the decisions of a single process on this host.

## Development curves

Each cell: net levels/round vs B11 / vs greedy / entropy, on the frozen 64 deals.

| Update | D | L2 | L3 | control A |
|---:|---|---|---|---|
| 100 | -2.05 / +0.41 / 0.39 | -2.18 / +0.59 / 0.59 | -2.16 / +0.27 / 0.49 | -2.19 / +0.24 / 0.53 |
| 200 | -1.87 / +0.69 / 0.29 | -2.02 / +0.32 / 0.34 | -2.01 / +0.48 / 0.28 | -2.08 / -0.01 / 0.24 |
| 300 | -1.95 / +0.70 / 0.19 | -1.96 / +0.46 / 0.34 | -1.93 / +0.59 / 0.27 | -2.25 / +0.27 / 0.31 |
| 400 | -1.68 / +0.70 / 0.17 | -1.78 / +0.39 / 0.20 | -2.10 / +0.48 / 0.24 | |
| 600 | -1.82 / +0.83 / 0.20 | -1.88 / +0.56 / 0.23 | -2.12 / +0.62 / 0.15 | |
| 700 | -1.97 / +0.87 / 0.19 | -1.84 / +0.70 / 0.15 | -2.22 / +0.52 / 0.10 | |
| 850 | | -1.71 / +0.86 / 0.16 | -2.12 / +0.47 / 0.13 | |

Pooled windows (paired whole-deal bootstrap, 10,000 resamples):

| Window | Comparison | vs B11 | vs long-run endpoint | vs greedy |
|---|---|---:|---:|---:|
| 100-300 | D - control A | **+0.24 [+0.08, +0.41]** | +0.16 [-0.03, +0.35] | **+0.39 [+0.13, +0.64]** |
| 100-300 | D - L2 | +0.10 [-0.10, +0.29] | **+0.18 [+0.03, +0.34]** | +0.17 [-0.05, +0.40] |
| 100-300 | L3 - L2 (seed) | -0.01 [-0.17, +0.14] | +0.04 [-0.08, +0.16] | +0.10 [-0.14, +0.34] |
| 350-600 | D - L2 | +0.08 [-0.09, +0.26] | +0.17 [-0.01, +0.35] | +0.22 [-0.02, +0.46] |
| 350-600 | L3 - L2 (seed) | -0.14 [-0.31, +0.03] | +0.15 [-0.02, +0.32] | +0.18 [-0.07, +0.42] |
| 650-900 | D - L2 (2 D snapshots) | -0.22 [-0.41, -0.02] | +0.24 [+0.01, +0.48] | +0.25 [+0.03, +0.48] |
| 650-900 | L3 - L2 (seed) | -0.24 [-0.44, -0.05] | -0.03 [-0.19, +0.13] | -0.13 [-0.40, +0.14] |

Full-match wins: zero in every evaluated snapshot of every arm.

## Reading

- **D passes the acceptance rule** against control A (not below over 0-300)
  and is mostly at or above the in-pod control, but the gain is small and
  inconsistent in sign against B11 late. It spends 5x the learner rows for it.
- **The same-configuration seed pair (L3 vs L2) differs by up to 0.24** in a
  window, with intervals excluding zero in one case. Seed variance is as
  large as every effect measured so far except run 1's two-epoch arm.
- **All three CPU controls (run 1 L, L2, L3) sit about +0.1..+0.2 above the
  CUDA control A against B11 over 100-300** (L3: +0.13 [+0.01, +0.26]). This
  may be a device effect or an unlucky control-A draw; comparisons should use
  in-pod controls, which every later arm now has.
- **Interpretation:** D keeps 16 optimizer steps per update with 6x larger
  minibatches; run 1's P doubled the steps with the same minibatches and gained
  more with fewer rows. Optimizer steps (or step size) per collected sample look
  more limiting than sample count. Run 3 tests 4 epochs, 1-match minibatches and
  lr 6e-4 directly. DDP stays opt-in; it is the throughput lever to combine with
  whatever recipe wins.

No default, checkpoint or document promotion follows; no strength claim.

## Budget, artifacts and closure

User pre-authorized RunPod use (cumulative cap $30, stop new rentals at $25).
A first create request for an RTX 5090 Community pod was refused locally for no
availability (no resource created). Pod `nb191swewkrtkh` (RTX 3090 Secure,
$0.504/hour including disk) was created at 07:42:33 UTC and confirmed gone after
5,067 seconds: estimated **$0.710**. All 176 remote files were SHA-256
verified before deletion; independent readback: zero pods, $0/hour, balance
$30.8981. Running total for this goal: **$1.42** (balance delta $1.42).

Kit: `.work/runpod-history-recipe2-2026-09-26/` (`run-manifest.json`,
`approval.json`, `pod.json`, `progress.jsonl`, `verified-artifacts.json`,
`teardown.json`, `independent-provider-readback.json`,
`local/results/arm-{D,L2,L3}/` with every rank's metrics under `arm-D/rank-*`,
`results/curves.json`, `results/analysis.json`, `results/greedy/`).
