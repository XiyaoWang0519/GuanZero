# Stage C implementation tracker: external baseline, learning signal, search

Status: drafted September 23, 2026 from the merged research review
(`DESIGN.md` v0.5, `reports/model-research-2026-09-23.md`). Nothing below is
implemented. Design source: `DESIGN.md` 7.4 (architecture gates), 8.4 items
7 to 11 (Stage B2, learning signal), 8.5 (search), 9.2 (external baseline).
Rules of the house in `CLAUDE.md` still apply, in particular rule 2 (every
rule decision behind `RuleConfig`) and rule 6 (a test vector in RULES.md
before new rule behaviour) for the `botzone` profile.

## Decisions carried in

- **Start point** for every training arm is the B8 league final,
  `.work/runpod-b8/results/runs/league/run/latest.pt`, with the B8 pool
  (`train/configs/league-b8.json`) plus the B9 exploiters, unless B11
  produces a stronger final first. Then B11's main is the start point and
  the tracker says so.
- **Equal-compute rule** as in Stage B: every strength claim compares
  arms given the same wall clock on the same host from the same start.
  Arms sharing a GPU are run in sequence or with fixed resource splits.
- **Revised G3 protocol** is the acceptance test for any change to the
  default: head to head against the control with a 95% duplicate interval
  above zero, and not worse on the held-out suite (greedy, four fixed
  styled bots, held-out sampled styles), paired per deal.
- **Order (owner decision, Sept. 23).** The pruning audit C1(a) is the
  first task: it is a CPU script and its answer changes what every later
  arm is allowed to choose from. C0 runs alongside it. C2 arms follow. C3
  starts on CPU as soon as C0's adapter exists, because it needs no
  training. C7, the looped state tower, is an experimental architecture
  arm the owner asked for; its supervised probe can start now on CPU and
  its RL arm joins the C2 comparison. C4 and C5 are conditional on the
  results of C3 and C0.
- **After C0 and C1(a), Sept. 23 evening.** Both are done. The pruning cap
  is not binding. DanLM is far ahead: B8 −2.07 levels/round, and DanLM's own
  MLP baseline is 1.6 levels/round ahead of us too, so the gap is training
  volume before architecture. Revised order: (1) DanLM is the yardstick for
  every arm from now on, `eval/danlm/arena.py duplicate` on at least 1,000
  deals per checkpoint (about 75 s with 8 workers); (2) C2 at a materially
  larger budget than Stage B's one-to-six hours, with the top-64 and
  soft-pruning arms inside it; (3) C5 opens by G6, with the FableDan recipe,
  after the C2 budget question is answered so that architecture and budget
  are not confounded; (4) C3 search and C7 looped tower as before.
- **Not started until gated:** the history tower (C5) waits for C0; the
  joint sampler (C4) waits for C3; distillation of search waits for the C3
  budget curve.

## Gates for Stage C

1. **G6, external calibration (C0).** Not pass/fail. Reports the B8 league
   final against DanLM under the `botzone` profile: duplicate net levels per
   round with a bootstrap interval over at least 4,000 deals, and at least
   1,000 full matches with seats swapped, plus the replay-diff divergence
   classes and how each was closed. A gap in DanLM's favour opens C5; a win
   with the interval clear of even closes C5 for now.
2. **G7, pruning (C1).** The widened-candidate arm is not worse than the
   top-32 control under the revised G3 protocol. If it is better, the wider
   set becomes the default for later arms.
3. **G8, learning signal (C2).** At least one arm beats the control under
   the revised G3 protocol. Each passing change becomes a default only on
   its own arm's evidence; the combination is its own arm.
4. **G9, search (C3).** The learned-belief search arm beats the blueprint
   on duplicate deals with a 95% interval above zero at a stated per-move
   budget, and the uniform-belief arm is reported beside it so that the
   belief and search contributions are separated. A budget curve at three
   per-move budgets is part of the report.
5. **G10, joint belief (C4).** The autoregressive sampler has higher joint
   log-likelihood of the true hidden hands than the marginal-plus-rejection
   sampler on held-out rounds, a higher valid-sample rate, and the C3 search
   arm improves when it is swapped in.
6. **G11, history tower (C5).** Same as G2 in `STAGE_B_TODO.md`: beats
   the MLP arm at equal wall clock with intervals clear of even, three
   seeds, and the GRU control does not match it.
7. **G12, looped state tower (C7).** Two parts. Probe: on held-out logs
   the looped tower's action-ranking and hidden-hand accuracy improve
   monotonically as the test-time loop count rises from 1 to the training
   maximum, at matched parameter count with the MLP; a tower whose
   accuracy does not move with loop count has not learned to use the
   loops and fails the probe. Strength: the RL arm beats the MLP arm under
   the revised G3 protocol at equal wall clock, with decisions per second
   reported per loop count.

A gate that fails is reported, not tuned around.

## Tasks

Each row is sized for one session. "Depends" names the row that must be
merged first.

| # | Task | Depends | Acceptance | Status |
|---|---|---|---|---|
| C0 | **DanLM external baseline.** (a) `eval/danlm/` adapter: a Python 3.12 virtual environment for DanLM's macOS binary extensions, a driver that plays DanLM in its own engine with our policy on the other seats through a JSON or in-process bridge, and a trace logger in the `ogd_adapter` style. (b) Rules diff list: tribute, anti-tribute, level A, wild usage, action types, double-win tail, written into RULES.md section 13 as parity checks. (c) `botzone` `RuleConfig` profile with test vectors for every difference (RULES.md section 12). (d) `replay_diff` of at least 1,000 logged DanLM matches under `botzone` with every divergence class closed. (e) G6 numbers. DanLM is never vendored, trained on or shipped: non-commercial licence, evaluation only. | none | `reports/stage-c-danlm.md` with the diff list, divergence classes, G6 tables and checkpoint hashes; profile tests in `cpp/tests` and `tests/` | done, **G6: large gap in DanLM's favour.** `eval/danlm/` (bridge, lockstep arena, verified fast observation), 8 tests. Rules parity in `RULES.md` 13.3: no `botzone` field needed, tribute/trick/score agree in every measured round; two legal-set-only classes remain. B8 league final vs DanLM over 3,998 deals: **−2.065 [−2.097, −2.034] levels/round, 11.5% of rounds**; M1 −2.46; DanLM's own MLP reproduction −0.48 (harness reproduces the author's 59.6% calibration at 57.5%). [Report](reports/stage-c-danlm.md). Consequence: C5 opens, but the ladder says training volume first (see Order) |
| C1 | **Candidate pruning audit and widened arms.** (a) `eval/pruning_audit.py`: on 100,000 B8-league decisions, fraction of decisions whose canonical set exceeds 32 by phase and stage, types and bomb sizes pruned, and the divergence rate between the league policy's argmax on the full canonical set and on the pruned set (a divergence indicator only, since logits of never-trained actions are not calibrated). (b) `PPOConfig` fields for `top_k` and a `candidate_union` mode (frozen top-k plus the current policy's top-m plus a uniform slice); the buffer stores the exact sampled set and behaviour log-probs; the learner never changes support inside an update. (c) Three equal-wall-clock arms: top 32, top 64, union. | none for (a); (b),(c) need the B8 start | audit JSON and report; tests that the stored support equals what was sampled from and that pass is always present; G7 in `reports/stage-c-pruning.md` | (a) done: `eval/pruning_audit.py`, 2 tests, one million decisions each on the B8 league final and frozen arm in 36 s on CPU. Pruning is active on 7.65% of play decisions (84.6% of early leads, 0.01% of late follows); the policy's chosen action has M1 rank 0 in 52.5% of decisions, p99 rank 14, rank 31 in 0.03%: no boundary pressure. Full-set argmax outside the set 0.32%, mostly untrained single leads; top 64 would admit 99.89%. [Report](reports/stage-c-pruning-audit.md). Recommendation: no standalone (c) rental; inside C2 run one `top_k = 64` arm and one `candidate_union` arm (frozen top 32 plus a uniform slice from outside, so that no legal action has probability zero; a soft prior instead of a hard cut, the AlphaGo move-37 argument, owner request Sept. 23). (b) pending |
| C2 | **Learning-signal arms (Stage B2).** One `PPOConfig` field each: (a) `critic_kind = q`: critic input (observation, hidden counts, action) through the shared action tower, `advantage_estimator = expected_sarsa` computing the VRPO trace from the Q-critic under the current policy over the stored candidate set; GAE stays as `v`/`gae`. (b) `advantage_filter_quantile`: policy loss only on samples above the quantile, with a minimum magnitude. (c) `policy_ema`: EMA weights used for snapshots and evaluation. (d) `match_value`: `train/match_value.py` fits V_match over (level, team levels, owner, fail counts) from self-play matches; reward becomes the round return plus or times a ΔV_match term, both modes behind fields; checked on the level-A behaviour probes. (e) `scale`: state width 1024, six layers, equal wall clock, throughput reported. Arms from the same start on one host in sequence, plus a combination arm of whatever passes. | B8 start; C1 default if G7 passed | unit tests: expected-SARSA advantage equals return minus Q when λ is 1 and the policy is deterministic; filter keeps pass rows; EMA and match-value round-trip through checkpoints; G8 in `reports/stage-c-signal.md` | pending |
| C3 | **Endgame search prototype.** `train/search.py` or `eval/search.py`: (a) trigger by unseen-card threshold (config); (b) sampler v0: marginal hidden head plus exact constraints (cards left per seat, two-deck multiplicity, known holdings, unseen set) by sequential assignment with rejection, valid-sample rate logged; (c) rollouts through `VecEnv::fork` with the blueprint at every seat, each seat observing only its legal view in the sampled world, value = mean return, optional critic cutoff; (d) correction = KL-regularized tabular update toward the blueprint with a margin, never bare argmax; (e) `load_policy` spec `search:<path>` so the arena, duplicate and crossplay tools use it unchanged. Three arms: none, uniform-legal belief, learned belief; budgets 20, 100, 500 ms per move measured, not assumed. | C0 adapter optional, B8 start | tests: samples satisfy every constraint, no seat reads another seat's true hand during rollouts, incremental equality of fork-and-play against a fresh env, correction reduces to the blueprint at infinite KL weight; G9 in `reports/stage-c-search.md` with the budget curve | pending |
| C4 | **Joint belief sampler.** `train/belief_sampler.py`: autoregressive model over the unseen cards conditioned on the observation (and, as a separate arm, the public history stream from `train/history_cache.py`), trained on self-play labels from the existing collections; metrics are joint log-likelihood of the true hands, valid-sample rate and sample diversity; optional reweighting by the blueprint's likelihood of observed passes. Swapped into C3 as `belief = ar`. | C3 with G9 passed | held-out metrics against the C3 v0 sampler; C3 rerun with the sampler; G10 in `reports/stage-c-belief.md` | pending, conditional |
| C5 | **History tower with behaviour prediction.** Only if G6 shows a gap. FableDan configuration: 4-block, width-128 causal Transformer over public event tokens with RoPE, QK-norm, RMSNorm, SwiGLU; hand and state through the existing MLP; next non-forced public event prediction by seat (NTP) and the existing hidden and finish heads as auxiliaries; batched rollout KV cache from `train/history_cache.py`; whole-round replay with causal masks in the PPO buffer. Arms: MLP control, MLP plus history, MLP plus history plus NTP, GRU plus NTP; three seeds each; equal wall clock. | C0 (G6 gap), C2 defaults | pipeline tests: private information never enters the shared stream, teacher-forced labels never enter the acting context, cache invalidation on weight publish; G11 in `reports/stage-c-history.md` | pending, conditional |
| C6 | **Docs and gate update.** `DESIGN.md` 8.4, 8.5 and 9.2 with what was actually built; `TRAINING.md` with the Stage C commands; `M2_TODO.md` and the milestone table. | C0 to C3 | one report per gate linked from here | pending |
| C8 | **Match memory as raw context (owner decision Sept. 23, replaces the v3 summary design).** Extends C5's token stream across the whole match: previous rounds' plays, passes, tribute cards, round ends and revealed hands stay as raw tokens with a round-index embedding and a seat embedding; `memory_rounds` config field (0 = per-round context as DanLM, `all` = full match) bounds what the query sees; KV cache per environment spans the match and is cleared at match end; environment count set by the cache budget (about 2 KB per token at 4 layers, width 128). Training opponents must carry stable habits for a whole match: the fixed-style bots, sampled styles held per match, and snapshots from different seeds; human records only from consenting players or public bot-platform logs (`DESIGN.md` 1.3 item 4 stands). Evaluation at match level, not belief CE: against fixed-style opponents, net levels in rounds 6 and later minus rounds 1 to 5, compared with the same model given a shuffled or truncated earlier-round history, and DanLM duplicate deals as the strength guard. | C5 | tests: no private token of another seat ever enters the stream; round-index and seat embeddings round-trip; `memory_rounds = 0` reproduces C5 bit for bit; cache spans rounds and clears at match end. Report `reports/stage-c-memory.md` with the adaptation contrast and the strength guard | pending, after C5 |
| C7 | **Looped state tower (experimental, owner request Sept. 23).** `train/looped_model.py`: a state tower of the recurrent-depth kind (`DESIGN.md` 7.5): a small prelude embeds the hand as card tokens, the candidate set as action tokens and the flat context as one token; one shared Transformer block is applied `loops` times with the prelude output re-injected each pass; a coda produces the state embedding and, for the candidate-set variant, per-candidate scores. Training tricks from the recurrent-depth literature behind config fields: loop count sampled per batch from a range, truncated backprop through the last `k` loops, optional loss at every loop. (a) Supervised probe on the existing 100,000-round logs, CPU: targets are the B8 league policy's action ranking (top-1 and pairwise) and the hidden hands; MLP control at matched parameters; evaluate at loop counts 1, 2, 4, 8, 16. (b) RL arm: initialize by distilling the B8 policy into the looped tower, then PPO at equal wall clock against the MLP control from the same start; report decisions per second at each loop count, since collection is the wall-clock bottleneck (`DESIGN.md` 8.6). | (a) none; (b) C2 defaults | tests: loop count 1 reproduces a plain block; output is finite and bounded at the maximum loop count; truncated gradient touches only the last `k` loops; probe and RL reports as G12 in `reports/stage-c-looped.md` | pending |

## Parallel start

Sessions that can begin immediately: C1(a) first, then C0, C7(a), C3(b)
and C3(c) on CPU with the B8 league final. C2 arms need one GPU rental and follow C1(c) so
that the candidate default is settled first. C3's evaluation arms can run
on CPU at the 500 ms budget and on GPU for the curve.

## Not in scope

- v3 match memory. Retest at match level with a shuffled-history control
  after Stage C (`DESIGN.md` 14, item 8).
- Learned tribute heads (Stage A2 decision stands).
- CFR, fictitious play, R-NaD or MuZero-style learned models
  (`DESIGN.md` section 2, Reevaluating policy gradient methods; LAMIR).
- LLM players (`DESIGN.md` 1.3).
- Search during collection for every decision. Distillation only after the
  G9 budget curve, on a sampled fraction, with total cost reported.
