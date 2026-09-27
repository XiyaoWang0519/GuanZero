# Stage C: Transformer self-play from scratch

Current plan, September 25, 2026. Source: [DESIGN.md](DESIGN.md), especially
sections 1, 7--9. This replaces the old C0--C8 execution order and G6--G12
promotion gates; those labels in dated reports retain their historical meaning.

## Fixed experiment boundary

Random actor/critic/head initialization; direct self-play PPO; current and
historical Transformer players only in training seats. No old MLP weights,
datasets, teachers, opponents, partners, action ranking or reference KL.
Existing MLPs and external agents are evaluation-only. Evaluation trajectories
never train the policy, critic or auxiliary heads. Rule-based exchange choices
remain a declared, identical tribute-only exception in the first experiment.

The standard history Transformer and the looped variant are both trained from
scratch. Public raw history spans the match. Timing features, search and
MLP imitation are not prerequisites. Public-history privacy includes not
exposing an opponent's engine-private forced-pass/legality flag.

## Current implementation boundary

The engine, public event logging, basic Transformer/cache prototypes, old PPO
utilities and evaluation adapters exist. `train/behaviour_probe.py` is an
offline diagnostic, not a cold-start RL trainer. `train/ppo.py` still requires
the old MLP reference and its collector/buffer do not establish sequence RL.
The newly planned tasks below are pending unless an explicit receipt is added.
Existing partial code or tests in another session are not completion evidence.

## Work queue

| ID | Work | Depends on | Acceptance | Status |
|---|---|---|---|---|
| T0 | Consolidate plan and audit old training dependencies | — | Design, tracker, training guide and inventory agree; actor/critic initialization, pool, replay, pruning, KL and evaluation boundaries enumerated | documentation complete; runtime enforcement pending |
| T1 | Random-start standard history actor, independent critic, full canonical candidate head and checkpoint marker | T0 | Fresh initialization loads no old checkpoint; all declared legal candidates selectable; private queries never enter public cache; old evaluator checkpoints still load | implemented; CPU-verified 2026-09-25 in `train/history_model.py` by `tests/test_history_model.py` (68 cases: fresh init never touches disk; every canonical candidate finite and sampleable, no top-k/reference attribute; `encode_stream` has no private argument and private changes leave the public encoding bit-identical; forced flag and tribute bits cannot enter `PublicStream`; checkpoint marker `stage=history_ppo`/`init=random`/`teacher=None` round-trips and refuses dmc/ppo/a2; old MLP evaluator checkpoints still load). Limits: `max_rounds=16` clamps the round embedding; no cache, full-prefix recomputation is the reference. Bounded GPU pilot verified 2026-09-26 (T4) |
| T2 | Public match-event store, sequence rollout buffer and PPO probability recomputation | T1 | Prefix includes opponent/forced-pass/exchange events; no forced-private bit in actor tokens; no future/private leakage; gradients reach history encoder; env/match/round attribution and chunk carry-over correct | implemented; CPU-verified 2026-09-25 in `train/history_rollout.py` and `train/history_ppo.py` by `tests/test_history_rollout.py` (8) and `tests/test_history_ppo.py` (6): prefixes equal an independent VecEnv recount and include tribute, forced-pass and all-seat events; the store never reads `forced`; stored behaviour log-probs equal the learner recompute (2.4e-7), truncation to the exact prefix and other rows' observations change nothing; one update changes every `actor.stream` parameter; mid-match update boundary keeps full prefixes and rewards land only on terminal rows; source-boundary test bans StageBPolicy/prune/top_k/reference/load_policy; interrupted save leaves the last checkpoint intact and resume continues. Smoke: 8 envs x 3 updates completes 48 rounds and learns. Limits: buffer/store not checkpointed, so resume drops in-progress rounds and reseeds deals; the original non-learner-seat rejection was replaced by tested match-pinned frozen Transformer execution in T4; actor kept in train mode (eval-mode parity < 1e-5 on torch 2.14). Bounded GPU pilot verified 2026-09-26 (T4) |
| T3 | History-aware scalar, batched and external evaluation adapters | T1 | Every policy observes the same available public events in order; correct reset between matches; full-prefix reference parity; unsupported policies fail explicitly | implemented; CPU-verified 2026-09-25 in `eval/history_policy.py` plus hooks in `eval/policies.py`, `eval/duplicate.py`, `eval/arena.py`, `eval/batched.py`, `eval/danlm/arena.py` by `tests/test_history_eval.py` (13): scalar streams equal VecEnv streams token for token across rounds and tribute phases (forced passes resolved in Python exactly as `VecEnv::advance`); duplicate legs and DanLM deals reset every round, `play_matches` resets only per match; batched play equals scalar play bit for bit; DanLM lockstep asserts events == applied actions + mirror forced passes; unknown stages and `select` before `start_match` fail explicitly; 264 existing evaluator tests still pass. Limits: `eval/probes.py`, `eval/tribute.py`, `eval/stage_b_baseline.py`, `eval/record_games.py` are unhooked and fail explicitly with a history policy; `MatchState` exposes no step counter to Python, so `verify_stream` checks cards-left consistency only |
| T4 | Direct PPO cold start and Transformer-only population; bounded remote GPU pilot | T2, T3 | Random critic, no MLP reference/filter or training seat; snapshots actually sampled and pinned per match; legal finite actions, optimizer updates, resume/cache rebuild, memory and throughput recorded | bounded pilot complete 2026-09-26 UTC: one RTX 4090, measured 4/8/16 envs then selected 8 (67.2% collection retention from mean prefix 166 to 754); 200 random-start updates, 63,948 learner samples, 1,724 rounds, real save/resume, 100 snapshots and 34,204 historical-policy decisions. 98 remote tests passed / 1 external-adapter skip; all 54 artifacts hash-verified before deletion, independent zero pods / $0 hourly spend. Full-prefix recomputation remains; no KV cache claim. Pilot allocator reservation reached 24.31 GB versus 1.55 GB allocated, so sustained memory headroom requires profiling before scale-up. [Run receipt and limits](reports/history-t4-pilot-2026-09-26.md) |
| T5 | Periodic fixed-MLP baseline comparisons and failure analysis | T3, T4 | Immutable checkpoint/config hashes; development deals separate from final tests; same-deal seat swaps and full matches; mean/interval plus behavior cells versus cumulative GPU-hours | initial three-endpoint development comparison complete in [T4 receipt](reports/history-t4-pilot-2026-09-26.md): frozen B11 / segment-2 endpoint, 64 duplicate deals and 8 full-match pairs each, raw legs retained, final-test material unopened. At update 200, paired net-level improvements from initialization were +0.656 / +0.414, but full-match wins remained 0/16 against each. Batch-size recipe test 2026-09-26 UTC ([receipt](reports/history-batch-2026-09-26.md)): three 300-update arms with every-10-update CPU curves; a 4x larger per-update batch reaches the same −2.1..−2.4 plateau per learner row, entropy collapses below 0.5 in every arm, KV/SDPA vs dense within ±0.15, full-match wins 0/16 throughout. Recipe campaign 2026-09-26 UTC ([summary](reports/history-recipe-summary-2026-09-26.md), runs [1](reports/history-recipe1-2026-09-26.md)–[5](reports/history-recipe5-2026-09-26.md), $4.65 over six CPU-on-pod allocations): the old plateau equals the engine greedy heuristic's level; seed noise is 0.2–0.35 per window, so in-pod controls and pooled seeds are required. Two PPO epochs beats six control seeds with four seeds (+0.37 [+0.25, +0.49] vs B11 over updates 650–900; never below control over 0–300). On top of it, current-policy-only self-play and the new data-parallel trainer (`train/history_ddp.py`, engineering-verified) speed learning; resumed lineages hold a +0.5..+0.8 lead over the control to update 1,500 but level off near −1.0..−1.3 vs B11. Entropy 0.03, width 128/4 layers (old recipe), 4 epochs, lr 6e-4, 1-match minibatches, 16 recent snapshots, GAE λ 1 and critic lr 1e-3 gave no gain or hurt. Full-match wins 1–2/16 in isolated snapshots; every snapshot still loses on average; no strength promotion, final test unopened |
| T6 | Shared-parameter looped decision module and controlled comparison | T4, initial T5 curve | Both arms start randomly with matched inputs/action support/population rules; stored loop count reused by PPO; supported-depth strength/latency curve and comparable-compute standard Transformer control | pending |
| T7 | Next-event objective and history-use ablations | T4, initial T5 curve | Only own-lineage self-play labels; RL-only vs RL+prediction separated from looping; prediction scores not called strength; reduced-history controls preserve legality | shared versus explicit opponent-response connection complete 2026-09-27 UTC ([plan](reports/history-response-plan-2026-09-26.md), [results](reports/history-response-results-2026-09-27.md), $3.84 of $6, three paired seeds): auxiliary prediction B − PPO-only A +0.29 [−0.04, +0.60] vs B11; explicit C − B −0.16 [−0.35, +0.04], not pursued. All endpoints −1.2..−1.8 vs B11, full-match wins 0–2%, entropy collapsed. Next work builds on B; history-use controls pending |
| T8 | Multi-seed confirmation and final held-out test | promising T6/T7 comparison | At least three training seeds for a robust architecture claim; preselected endpoint/config; raw paired records, exclusions, independent test results and full cost | pending |

Implement the standard actor and sequence/evaluation contract first. Reserve
a clean recurrent-module interface so T6 can reuse the same pipeline. A poor
early MLP score is a learning-curve observation, not an automatic rejection
of cold-start training. Investigate correctness and exploration before scaling.

## Experiment and acceptance protocol

- **Pilot:** bounded engineering/learning check, not a strength claim. Verify
  no teacher dependency, no information leakage and a working update/resume.
- **Learning:** freeze MLP evaluator identities/settings before a run and
  report checkpoint strength against cumulative GPU-hours, with uncertainty.
  Current-vs-current self-play win rate does not measure improvement.
- **Recurrence:** test a checkpoint at its trained loop depths, initially
  considering 1/2/4/8, then compare independent standard and looped training
  arms. Select depth using development results. Do not demand strictly
  monotonic gains or claim looping works because it beats only the old MLP.
- **Budget:** primary arms use equal total training wall clock with comparable
  hardware/resource shares. Record actual GPU-hours, decisions and cost, and
  compare inference latency/FLOPs. Equal parameter count alone is insufficient.
- **Evidence:** bootstrap whole paired deals/matches; report full-match win
  rates separately from round results. Training-seed variability is separate
  from a deal confidence interval. Keep negative cells and adapter failures.
- **Final test:** predeclare primary comparison, sample budget, endpoint
  selection, exclusions and regression tolerance before opening test deals.
  Do not extend the test just because an interval overlaps zero.

## Historical work retained as evidence

| Historical item | Record | Role now |
|---|---|---|
| C0 / G6 DanLM calibration | [report](reports/stage-c-danlm.md) | External evaluation infrastructure; no gate on whether history RL may start |
| C1 pruning audit/union pilot | [audit](reports/stage-c-pruning-audit.md), [pilot](reports/candidate-union-pilot-2026-09-24.md) | Explains old MLP action support; no frozen-reference pruning in the new actor |
| C2 and continuation experiments | [continuation](reports/kaggle-campaign-results-2026-09-26.md) | Frozen evaluation candidates and engineering evidence; no warm start |
| C3 search prototype | [prototype](reports/stage-c-search-v0.md), [transfer](reports/stage-c-search-transfer-2026-09-24.md) | Retained separately; search deferred |
| Old belief/memory experiments | [M2 index](M2_TODO.md) | Historical supervised results; no history-policy gate |
| Old C7 distillation proposal | Superseded by T6 | No action-ranking imitation prerequisite or MLP initialization |

No further cloud launch, new budget, model replacement or runtime migration is
implied by updating this tracker. T4's approved bounded run has GPU,
learning/resume/population, artifact and teardown receipts; future rentals
require a new explicit scope and budget.

Post-T4 stack follow-up: [local optimization report](reports/history-stack-2026-09-26.md).
Causal SDPA and batched public KV caching are implemented behind opt-in flags,
with learner-weight invalidation, raw-history rebuild and full regression
coverage. The separately approved [CUDA comparison](reports/history-stack-cuda-2026-09-26.md)
completed: 98 Python checks, 81 C++ cases, 12 repeated timing cases plus two
profiles, 336 total PPO updates. KV improved pooled full-update throughput
1.76x with all-current seats and 1.50x with mixed snapshots; mixed peak
allocator reservation fell from 24.68 to 0.17 GB for this small configuration.
All 83 artifacts were verified before deletion; independent zero pods / $0
hourly spend, estimated cost $0.128. Defaults remain off: longer training
trajectories differ even within a variant, and sustained playing-quality
equivalence has not been established. T4's original archive/results remain
intact; this engineering comparison does not close T5 or promote strength.

Recipe campaign, September 26 UTC: five bounded runs on pod CPUs within the
user's pre-authorized cumulative $30 RunPod cap (spent $4.65, ledger
`.work/runpod-ledger-2026-09-26.json`). Development curves moved from the
−2.1..−2.4 plateau to about −1.0..−1.3 against B11 with two epochs plus
self-play or data parallel; see the [summary](reports/history-recipe-summary-2026-09-26.md).
`train/history_ddp.py` adds opt-in data-parallel training with resume; no
production default changed (`HistoryPPOConfig.epochs` already defaults to 2).
This is development-curve evidence, not a strength claim; T5 stays open.
