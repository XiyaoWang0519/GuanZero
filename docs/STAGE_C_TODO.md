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
| T1 | Random-start standard history actor, independent critic, full canonical candidate head and checkpoint marker | T0 | Fresh initialization loads no old checkpoint; all declared legal candidates selectable; private queries never enter public cache; old evaluator checkpoints still load | implemented; CPU-verified 2026-09-25 in `train/history_model.py` by `tests/test_history_model.py` (68 cases: fresh init never touches disk; every canonical candidate finite and sampleable, no top-k/reference attribute; `encode_stream` has no private argument and private changes leave the public encoding bit-identical; forced flag and tribute bits cannot enter `PublicStream`; checkpoint marker `stage=history_ppo`/`init=random`/`teacher=None` round-trips and refuses dmc/ppo/a2; old MLP evaluator checkpoints still load). Limits: `max_rounds=16` clamps the round embedding; no cache, full-prefix recomputation is the reference. GPU pilot pending (T4) |
| T2 | Public match-event store, sequence rollout buffer and PPO probability recomputation | T1 | Prefix includes opponent/forced-pass/exchange events; no forced-private bit in actor tokens; no future/private leakage; gradients reach history encoder; env/match/round attribution and chunk carry-over correct | implemented; CPU-verified 2026-09-25 in `train/history_rollout.py` and `train/history_ppo.py` by `tests/test_history_rollout.py` (8) and `tests/test_history_ppo.py` (6): prefixes equal an independent VecEnv recount and include tribute, forced-pass and all-seat events; the store never reads `forced`; stored behaviour log-probs equal the learner recompute (2.4e-7), truncation to the exact prefix and other rows' observations change nothing; one update changes every `actor.stream` parameter; mid-match update boundary keeps full prefixes and rewards land only on terminal rows; source-boundary test bans StageBPolicy/prune/top_k/reference/load_policy; interrupted save leaves the last checkpoint intact and resume continues. Smoke: 8 envs x 3 updates completes 48 rounds and learns. Limits: buffer/store not checkpointed, so resume drops in-progress rounds and reseeds deals; population hook records seat identities but raises for any non-learner seat; actor kept in train mode (eval-mode parity < 1e-5 on torch 2.14). GPU pilot pending (T4) |
| T3 | History-aware scalar, batched and external evaluation adapters | T1 | Every policy observes the same available public events in order; correct reset between matches; full-prefix reference parity; unsupported policies fail explicitly | implemented; CPU-verified 2026-09-25 in `eval/history_policy.py` plus hooks in `eval/policies.py`, `eval/duplicate.py`, `eval/arena.py`, `eval/batched.py`, `eval/danlm/arena.py` by `tests/test_history_eval.py` (13): scalar streams equal VecEnv streams token for token across rounds and tribute phases (forced passes resolved in Python exactly as `VecEnv::advance`); duplicate legs and DanLM deals reset every round, `play_matches` resets only per match; batched play equals scalar play bit for bit; DanLM lockstep asserts events == applied actions + mirror forced passes; unknown stages and `select` before `start_match` fail explicitly; 264 existing evaluator tests still pass. Limits: `eval/probes.py`, `eval/tribute.py`, `eval/stage_b_baseline.py`, `eval/record_games.py` are unhooked and fail explicitly with a history policy; `MatchState` exposes no step counter to Python, so `verify_stream` checks cards-left consistency only |
| T4 | Direct PPO cold start and Transformer-only population; bounded remote GPU pilot | T2, T3 | Random critic, no MLP reference/filter or training seat; snapshots actually sampled and pinned per match; legal finite actions, optimizer updates, resume/cache rebuild, memory and throughput recorded | pending. Measured starting point (CPU, width 32, 1 layer, 8 envs, no cache): collection fell from about 2,500 to 580 decisions/s as the mean prefix grew from 144 to 720 tokens, the O(T^2) full-prefix recomputation DESIGN 7.3 anticipates; batched caches or amortised recompute are the first T4 measurement |
| T5 | Periodic fixed-MLP baseline comparisons and failure analysis | T3, T4 | Immutable checkpoint/config hashes; development deals separate from final tests; same-deal seat swaps and full matches; mean/interval plus behavior cells versus cumulative GPU-hours | pending |
| T6 | Shared-parameter looped decision module and controlled comparison | T4, initial T5 curve | Both arms start randomly with matched inputs/action support/population rules; stored loop count reused by PPO; supported-depth strength/latency curve and comparable-compute standard Transformer control | pending |
| T7 | Next-event objective and history-use ablations | T4, initial T5 curve | Only own-lineage self-play labels; RL-only vs RL+prediction separated from looping; prediction scores not called strength; reduced-history controls preserve legality | pending |
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

No cloud launch, new budget, model replacement or runtime migration is implied
by updating this tracker. The next concrete deliverable is T1--T3 implementation
and verified readiness, followed by a separately specified remote pilot.
