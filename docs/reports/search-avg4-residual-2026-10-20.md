# Branch B: avg4 residual-shuffle search (2026-10-20)

Status: **measured.** First fixed-budget residual on the frozen avg4 parent
(self-play vs plain avg4). Positive but tiny; not a promote signal for student
training. DanLM Linux residual remains blocked (Mac-only).

## Setup

| Item | Value |
|---|---|
| Checkpoint | `avg4-u86800.pt` (eval-only weight average) |
| SHA256 | `21cc19d91fa299e17d0c1a3af2cf51f186c9cc3a22aef1ab44539880c53c702d` |
| Host | Velocity `max-compute-11`, SLURM job **8261** `gz-search-avg4` |
| Partition | `short` CPU-only (16 cpus, 12 workers) — no Branch A GPU contention |
| Deals / seed | 200 / `20261020` |
| Wall | 419.7 s job; per-deal mean 1.12 s (`plain`) / 6.65 s (`search32`) |
| Source | `~/guanzero/src-campaign-2026-10-20` + portable `search_sweep` provenance |

Arms (one variable = search on vs off):

| Arm | Config |
|---|---|
| `plain` | `unseen_threshold=0` (never triggers) |
| `search32` | 10 s / 32 worlds / min 8 / KV batch 8 / `selection=prior` / `value_margin=0.5` / trigger `unseen` @ 12 |

Opponent is always the plain loaded policy (no search). Duplicate first+swapped.

Artifacts on Velocity: `~/guanzero/runs/search-avg4-residual/out/{summary,provenance,raw}.json*`.

## Results

| Arm | net (lv/rnd) | 95% CI | win rate | overrides / calls |
|---|---:|---|---:|---:|
| plain | 0.000 | [0, 0] | 0.500 | 0 / 14179 |
| search32 | **+0.0175** | [+0.0025, +0.0375] | 0.5025 | 10 / 14187 |

Paired contrast **search32 − plain**: **+0.0175 [+0.0025, +0.0375]** (200 deals).

Sparseness: **196/200 deals tied at Δ=0**; four positive deals (Δ ∈ {0.5, 0.5, 1.0, 1.5}).
Search triggered 1252 times, completed 1251 (1 fallback), exhausted budget 15 times.
Override fraction **0.80%**. Search latency p50/p95/max ≈ 0.33 / 3.7 / 10.0 s;
worlds median/max 32/32 when triggered.

## Interpretation

- CI excludes zero, but the effect is two orders of magnitude below historical
  DanLM search gains (~+0.3 lv/rnd on earlier checkpoints). Those were
  **vs DanLM**, not self-play.
- Against a matched plain avg4 opponent, residual-shuffle search almost never
  overrides the prior (`value_margin=0.5` + `selection=prior`). Completing
  rollouts is not the bottleneck; action change is.
- Do **not** promote to student distillation from this residual alone.

## Follow-ups (same deals / seed / budget; CPU `short`)

| Job | Change | Paired Δ lv/rnd | Overrides | Verdict |
|---:|---|---:|---:|---|
| **8267** | `value_margin=0` (else = residual) | **+0.0175 [+0.0025, +0.0375]** | 10 (same) | **Negative** — margin was not the bottleneck |
| **8269** | `selection=mean` + `value_margin=0` | **+0.030 [−0.040, +0.0975]** | 198 (15.7% of triggers) | **Negative** — more overrides, CI crosses zero |

Artifacts: `~/guanzero/runs/search-avg4-vm0/`, `~/guanzero/runs/search-avg4-mean/`.

`selection=mean` unlocks overrides (198 vs 10) but does not deliver a reliable
self-play gain on 200 deals. Stop further self-play search config churn until
A/next_coef free capacity or a Mac DanLM residual is available
([danlm-linux-velocity-2026-10-20](danlm-linux-velocity-2026-10-20.md)).

## Limitations

Self-play residual understates exploitability of weaker external bots. 200 deals
is enough to reject a large gain, not to rank tiny config tweaks. No GPU search;
CPU path matches the portable Linux `_gd_core` build used on Velocity.
