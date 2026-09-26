# Recipe run 1: five one-variable arms on CPU, September 26, 2026 UTC

The plateau at -2.1..-2.4 net levels/round against frozen B11 is the level of
the engine's greedy heuristic, not a hard ceiling of the history route. One
RTX 3090 Secure allocation (training on its CPUs) ran five fresh random-start
arms in parallel for 80 minutes each, each changing one variable from the
batch-size kit's A-arm control. **Two PPO epochs per update (arm P) is the only
change with a positive development signal; a 16-snapshot population (arm H) is
worse; entropy 0.03, width 128 / 4 layers and the control itself stay in the
plateau band.** One seed per arm; these are development curves, not strength.

## Local diagnosis before renting (CPU, free)

- **The plateau is greedy level.** On the same 64 development deals, B11 beats
  the engine greedy heuristic by +2.02 net levels/round; the batch-size A arm
  at update 300 scores +0.27 [-0.10, +0.62] against greedy (sampling instead of
  argmax: +0.14) and +2.73 against random. Scores against B11 compress near the
  -3 floor, so a greedy yardstick was added to every curve (duplicate only,
  same deals, `results/greedy/`).
- **The sample count is small for self-play from scratch.** A pure self-play
  MLP DMC run with M1's architecture (512x4) and no greedy mixing, trained
  locally for 53 s, scored -2.09 / -1.07 / -0.33 / -0.47 / -0.63 against greedy
  after 0.10 / 0.20 / 0.40 / 0.79 / 1.85 M samples. M1 itself used 50% greedy
  exploration at the start and 12.8 M learner samples to reach +1.06. The
  history Transformer's +0.27 at 0.38 M learner rows is not less sample efficient
  than the MLP at this scale. That calibration model was evaluation-only and
  never touched a Transformer run.
- **The rented GPU path was host-bound.** The same 32-environment KV/SDPA
  collection runs at about 2,900 decisions/s on this Mac's CPU with two torch
  threads versus about 730 on the RTX 4090 pod. The model is 64 wide; CUDA
  kernel launches and the pod's host dominate. Training on CPU is the FP32
  reference path already covered by the CPU/CUDA equivalence tests.

## Design and identity

| Arm | Change from the control (A arm: 32 envs, KV + SDPA, lr 3e-4, entropy 0.01, 1 epoch, 2 matches/minibatch, snapshots every 2 updates, 4 recent, p = 0.5) | Updates reached | Learner rows |
|---|---|---:|---:|
| L | none (in-pod control replicate; CPU instead of CUDA) | 887 | 1,091,872 |
| E | entropy coefficient 0.03 | 867 | 1,069,914 |
| W | width 128, 4 layers | 343 | 430,982 |
| P | 2 PPO epochs | 713 | 885,031 |
| H | 16 recent eligible snapshots | 825 | 1,020,756 |

All arms: seed 2026092603, FP32 (TF32 off), full public history, full canonical
candidate set, unchanged reward/GAE, own-lineage snapshots only, no MLP on the
pod. Each arm had a 4,800-second cap and ran 2 engine + 2 torch threads.
Snapshots every 10 updates to 300, then every 50 (slim actor/critic files).
Learner rows are summed from each arm's `metrics.jsonl`; the arm receipts'
`learned_rows` field covers only the last 400 updates (health window).

| Identity | Value |
|---|---|
| Source SHA-256 | `e86a9761…3008` (identical to the batch-size and stack runs; no training code changed) |
| Archive / manifest SHA-256 | `fe7171f5…c181` / `2f636e6a…514f9c` |
| Host | AMD EPYC 7H12 (shared 256-thread host), cgroup quota 31.1 CPUs, RTX 3090 |
| Test gate on pod | 98 passed (CPU and CUDA history tests) |
| Final slim checkpoints | L `30b776f6…`, E `827f28b8…`, W `b30979db…`, P `618d1c1a…`, H `97f24875…` |

## Development curves

Net levels/round on the frozen 64 deals, two swapped legs, deterministic argmax;
each cell is "vs B11 / vs greedy". Full data: `results/curves.json`.

| Update | L | E | W | P | H | control A (CUDA, batch kit) |
|---:|---|---|---|---|---|---|
| 0 | -2.99 / -3.00 | -2.99 / -3.00 | -3.00 / -2.95 | -2.99 / -3.00 | -2.99 / -3.00 | -2.99 / -3.00 |
| 50 | -2.34 / +0.16 | -2.25 / +0.05 | -2.62 / -0.77 | -2.24 / +0.30 | -1.97 / +0.35 | -2.22 / +0.30 |
| 100 | -2.12 / +0.19 | -2.14 / +0.23 | -2.23 / +0.18 | -2.06 / +0.43 | -2.38 / +0.22 | -2.19 / +0.24 |
| 200 | -2.04 / +0.35 | -1.94 / +0.55 | -2.02 / +0.61 | -2.01 / +0.24 | -2.02 / +0.45 | -2.08 / -0.01 |
| 300 | -1.85 / +0.55 | -1.95 / +0.74 | -1.99 / +0.33 | -1.97 / +0.45 | -2.05 / +0.20 | -2.25 / +0.27 |
| 400 | -1.99 / +0.24 | -1.81 / +0.74 | | -1.80 / +0.80 | -1.86 / +0.84 | |
| 500 | -2.18 / +0.54 | -1.77 / +0.89 | | -1.71 / +0.58 | -1.88 / +0.53 | |
| 600 | -1.90 / +0.77 | -1.94 / +0.88 | | -1.71 / +0.78 | -1.97 / +0.30 | |
| 700 | -2.12 / +0.50 | -1.87 / +0.65 | | -1.57 / +1.05 | -2.07 / +0.15 | |
| 850 | -1.94 / +0.80 | -1.86 / +0.87 | | | | |

Entropy at updates 100 / 300 / 600 / 850: L 0.54 / 0.25 / 0.21 / 0.16,
E 0.89 / 0.76 / 0.32 / 0.60, P 0.80 / 0.54 / 0.36, H 0.48 / 0.28 / 0.17.

Pooled windows (per-deal pair scores averaged over snapshots; paired whole-deal
bootstrap, 10,000 resamples). Up to update 300 the reference is the original
control A; later the reference is the in-pod control L.

| Window | Comparison | vs B11 | vs long-run endpoint | vs greedy |
|---|---|---:|---:|---:|
| 100-300 | L - control A (same config) | +0.04 [-0.11, +0.19] | -0.03 [-0.17, +0.11] | +0.25 [+0.06, +0.44] |
| 100-300 | E - control A | +0.06 [-0.11, +0.22] | +0.07 [-0.10, +0.23] | +0.23 [-0.01, +0.46] |
| 100-300 | W - control A | +0.01 [-0.16, +0.19] | -0.03 [-0.22, +0.15] | +0.14 [-0.10, +0.37] |
| 100-300 | P - control A | +0.11 [-0.06, +0.28] | -0.02 [-0.18, +0.13] | +0.20 [+0.01, +0.39] |
| 100-300 | H - control A | +0.02 [-0.10, +0.14] | -0.09 [-0.22, +0.04] | -0.04 [-0.27, +0.18] |
| 350-600 | P - L | **+0.21 [+0.03, +0.39]** | +0.16 [-0.04, +0.35] | +0.13 [-0.10, +0.36] |
| 350-600 | E - L | +0.09 [-0.09, +0.28] | +0.17 [+0.00, +0.33] | +0.19 [-0.03, +0.42] |
| 650-850 | P - L (2 P snapshots) | +0.27 [-0.01, +0.56] | **+0.30 [+0.07, +0.56]** | +0.29 [-0.01, +0.58] |
| 650-850 | H - L | -0.18 [-0.39, +0.03] | -0.03 [-0.22, +0.16] | **-0.53 [-0.80, -0.27]** |

Full-match wins: one match out of 16 against B11 at L update 700 (that
snapshot's duplicate score was -2.13); zero in every other snapshot of every arm.

## Reading

- **Training-seed noise is as large as the arm effects.** L and control A have the
  same configuration and seed (L on CPU, A on CUDA; trajectories diverge anyway).
  They agree against both frozen MLPs but differ by +0.25 against greedy over
  100-300. Single-seed differences under about 0.3 are not evidence.
- **Two epochs (P) is the one positive signal and passes the acceptance rule.** It
  is not below the control over 0-300, and beats the in-pod control over 350-600
  against B11 in the same wall clock with fewer learner rows (885k vs 1.09M by the
  end); at matched rows (about 0.87M, both near update 700) P scores -1.57 and L -2.12. Its 700-update point
  (-1.57 vs B11, +1.05 vs greedy) is the highest development point of any run.
  More gradient steps per collected sample look useful; four epochs is untested.
- **A longer population window (H) hurts** late in training against greedy.
- **Entropy 0.03 (E) keeps entropy near 0.5-0.6 but gives no significant gain**;
  width 128 / 4 layers (W) gives no gain at matched updates and costs 2.6x time.
- **The control does not break the plateau with 3x the updates**: L stays at
  about -2.0 vs B11 / +0.5..+0.8 vs greedy from update 300 to 887.
- **Engineering:** the CPU path matched the CUDA control's curve against both
  frozen MLPs (see L - control A). Per-process speed on this shared EPYC host was
  about 400-550 decisions/s, far below this Mac (about 2,900); arms scale by adding
  processes, not threads.

No default, checkpoint or document promotion follows. Nothing here is a
strength claim: full-match wins are 1/16 at a single snapshot.

## Budget, artifacts and closure

User pre-authorized RunPod use (cumulative cap $30, stop new rentals at $25;
approval text in `approval.json`). Pod `dcw1netcoxtjc5` (RTX 3090 Secure,
EU-CZ-1, $0.504/hour including disk) was created at 06:17:00 UTC and
confirmed gone after 5,061 seconds (84.3 minutes): estimated **$0.709**;
the balance moved from $32.3145 to $31.6411 ($0.673). All 249 remote files
were SHA-256 verified before deletion; an independent readback found zero
pods and $0/hour. Running total for this goal: $0.71
(`.work/runpod-ledger-2026-09-26.json`).

Kit: `.work/runpod-history-recipe1-2026-09-26/` (`run-manifest.json`,
`approval.json`, `pod.json`, `progress.jsonl`, `verified-artifacts.json`,
`teardown.json`, `independent-provider-readback.json`, `local/results/arm-*/`,
`results/curves.json`, `results/analysis.json`, `results/greedy/`).
Scripts: `.work/history-recipe-kits/` (`prepare.py`, `arm.py`, `recipe.py`,
`analyze.py`, `greedy_curves.sh`). Diagnosis: `.work/history-diagnosis-2026-09-26/`.
