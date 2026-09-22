# Guandan AI: system design

Status: v0.4, empirical progress updated Sept. 22, 2026. v0.2 fixed the house rules (RULES.md section 13). v0.3 added phase heads and brought learned tribute forward to M2. v0.4 adds the Transformer history encoder (v2), the match memory experiment (v3) and the belief probe that gates them. Owner: Irvin. Companion documents: `RULES.md` (rules and acceptance tests), `gd_reference.py` (Python rules oracle).

Implementation update, Sept. 21: the pre-training M1 pipeline is implemented
and verified on CPU and an RTX 5090. See `M1_TODO.md`, `TRAINING.md`,
`reports/M1-preflight.md`, and `reports/M1-runpod.md`. The bounded GPU pilot
completed 34,496 updates and 71.27M decisions, with checkpoints retrieved and
the pod deleted. This does not mark the M1 strength gate complete; internal
learning measurements cannot replace the unavailable external benchmark agents.

M2 update, Sept. 21: Stage A2 collection, frozen-head fitting and paired tribute
evaluation are implemented. A 4,096-position, three-seed experiment did not
improve on heuristic exchanges, so the heuristic remains the baseline. See
`M2_TODO.md` and `reports/M2-A2.md`. The public known-holdings encoder was also
corrected before collection; PPO, the critic and league remain pending.

Source audit correction, resolved Sept. 22: the first 100,000-round styled
collection used a six-update local smoke checkpoint, not the trained M1 final.
That experiment remains historical evidence for its own distribution in
[the original scaled report](reports/M2-belief-scaled.md). A new 100,000-round
collection from the verified M1 final (34,496 updates) and the three-seed
comparison are now complete; the datasets and results are kept separate.

Transformer follow-up: the earlier 4,096-round prototype passed its supervised
history gate (`reports/M2-belief.md`). The corrected 100,000-round comparison
now supports the no_history query tower for Stage B under this tested protocol:
its mean hidden-hand loss is 1.6086% lower than flat, while adding history wins
one seed and loses two. This is supervised evidence, not a playing-strength
result. The separate cross-round memory experiment is also complete: overall
memory benefit wins two seeds and loses one, and all three adaptation intervals
include zero. Its original gate remains `v3_not_yet_justified`, so this
prototype is not promoted into Stage B. Shared-cache inference is implemented,
but no history playing policy has been promoted. See
[the corrected comparison](reports/M2-belief-corrected.md) and
[the memory report](reports/M2-memory.md). All 15 selected checkpoints are
verified and the GPU is released; see [the completed run record](reports/M2-next-run.md).

## 1. Summary

Build a reinforcement learning agent for Guandan (掼蛋) that beats every published baseline and is competitive with experienced human players, trained on one rented GPU node on a personal budget.

The plan in one paragraph: a fast C++ rules engine with a batched Python interface feeds a small network that scores candidate actions. The first version of that network is a plain MLP. The second reads the full play history of the round with a Transformer, which no published Guandan agent does. Deep Monte Carlo self-play produces a strong baseline. PPO with a perfect-information critic and an opponent league takes it further, with optional search in endgames. Progress is measured by an evaluation harness built on duplicate deals and on the OpenGuanDan benchmark agents.

### 1.1 Why this is still worth doing

OpenGuanDan (January 2026) evaluated the three published learning agents against 16 human volunteers. The best one, GS2, won 42.5% of rounds overall and none of them exceeded 50%. The authors place current agents between beginner and intermediate human level. The problem is open.

### 1.2 Success criteria

Match win rates are measured inside the OpenGuanDan simulator over 1,000 matches per opponent, which is the protocol of the benchmark paper.

| Gate | Criterion | Calibration from the OpenGuanDan paper |
|---|---|---|
| M0 | Engine passes every test in RULES.md, replays OpenGuanDan traces without divergence and meets the throughput target. **Met**, see `docs/reports/M0.md`. | Their Java simulator reports about 25 million steps per hour on 10 parallel environments |
| M1 | At least 80% against each of Rule One to Rule Four | DanZero scores 83% to 92% against them |
| M2 | At least 55% against DanZero and at least 50% against SDMC | SDMC beats DanZero 57.6% |
| M3 | At least 55% against GS2 | GS2 beats SDMC 56.7% and DanZero 62.0% |
| Stretch | At least 50% of rounds against intermediate and advanced human players over 100 or more rounds | No published agent has done this |

### 1.3 Non-goals

1. An LLM-based player. LLMs are not competitive here and the papers agree.
2. Multi-node distributed training.
3. Regional rule variants beyond the config switches in RULES.md.
4. Running the agent on public online platforms against people who have not agreed to play a bot.

## 2. Background and what we take from prior work

| Work | Method | What we reuse |
|---|---|---|
| DouZero (arXiv 2106.06135) | Deep Monte Carlo, action features as network input, self-play | The whole base recipe |
| DanZero (arXiv 2210.17087) | DouZero recipe on Guandan. 513-dim state, 54-dim action, MLP. 160 CPUs and 1 GPU for 30 days. | Reward design, feature list and the ablation showing that explicit wild card features help |
| DanZero+ (arXiv 2312.02561) | PPO on top of DMC, with the DMC model proposing candidate actions | Top-k candidate pruning for the policy stage |
| SDMC (Ge et al., 2024) | DMC with an expert policy guiding early self-play, plus soft action sampling | Heuristic opponents early in training. Sampling among near-best actions at play time. |
| GS2 (Ge et al., NeurIPS 2023) | Subgame refinement on top of the SDMC blueprint | Evidence that search on top of a learned policy pays: GS2 beats its own blueprint 56.7% |
| GuanZero (arXiv 2402.13582) | Behavior flags in the input to induce cooperation | Cooperation must be measured directly, not inferred from win rate |
| Tjong (Li et al., 2024) | Transformer Mahjong agent, supervised. Beat CNN, MLP, RNN, ResNet and ViT variants in a tournament. | Attention helps on tile and card structure in a neighboring game. It is supervised and small scale, so only weak evidence. |
| TAO (ICLR 2024) and OMIS (NeurIPS 2024) | Transformers that adapt in context to unseen opponent policies | Precedent for the v3 match memory. Their environments are far smaller than Guandan. |
| PerfectDou (arXiv 2203.16406) | Perfect-information critic, imperfect-information actor | Critic design for the PPO stage |
| OpenGuanDan (arXiv 2602.00676) | Java simulator, JSON per-player API, seven built-in agents | Reference rules, baseline opponents, evaluation protocol |

AlphaZero-style MCTS is not the base method. Its search assumes a fully observed state, while Guandan hides three hands, has legal action sets on the order of 10^4 on an opening hand and runs about 100 decisions per player per match. Search is kept for endgames, where hidden information is small.

## 3. Requirements

**Functional**

1. Simulate complete matches under RULES.md, including tribute, anti-tribute and the level A rules.
2. Enumerate legal actions in full mode and in canonical mode.
3. Accept explicit deals for duplicate evaluation and scripted scenarios.
4. Encode observations and candidate actions into tensors without Python in the hot loop.
5. Train, checkpoint, resume, evaluate and play against humans from one repository.

**Non-functional**

| Requirement | Target | Note |
|---|---|---|
| Engine throughput | At least 100,000 decisions per second per core with random play in canonical mode | Target to validate in M0. The reference simulator totals about 7,000 steps per second. |
| Determinism | Same seed and actions give the same state hash on any machine | Needed for duplicate deals and for debugging |
| Resumability | Training survives a kill at any moment and loses at most 10 minutes | Makes interruptible cloud instances usable |
| Budget | About 500 USD of compute through M2, enforced by a watchdog | See section 10 |
| Operator time | The train and evaluate loop runs unattended under Claude Code | The human reads dashboards and decides at gates |

**Constraints.** One developer working through a coding agent. No local GPU. Single rented node. C++20, Python 3.11 or newer, PyTorch.

## 4. Architecture

```
                 +--------------------------- one process, one node ---------------------------+
                 |                                                                              |
  seeds, deals   |   gd_core (C++20)                      Python                                |
  ------------>  |   +----------------------+   views    +----------------------------+        |
                 |   | VecEnv: N matches    | ---------> | rollout loop               |        |
                 |   |  rules, movegen,     |  obs,      |  batch -> model on GPU     |        |
                 |   |  tribute, features   |  cands     |  pick action per decision  |        |
                 |   |  thread pool         | <--------- |  env.step(choices)         |        |
                 |   +----------------------+  choices   +-------------+--------------+        |
                 |                                                     | finished rounds        |
                 |                                       +-------------v--------------+        |
                 |                                       | replay buffer -> learner   |        |
                 |                                       |  DMC, later PPO + critic   |        |
                 |                                       +-------------+--------------+        |
                 |                                                     | checkpoints            |
                 +-----------------------------------------------------|------------------------+
                                                                       v
                        +----------------------+      +-----------------------------+
                        | league pool          | <--> | evaluator                   |
                        |  past checkpoints    |      |  duplicate deals, Elo,      |
                        |  heuristic bots      |      |  behavior probes,           |
                        +----------------------+      |  OpenGuanDan adapter        |
                                                      +-----------------------------+
```

Components:

| Component | Language | Responsibility |
|---|---|---|
| `gd_core` | C++20 | Cards, rules, move generation, round and match state machine, feature encoder, heuristic bots, vectorized environment |
| `gd` Python package | pybind11 | Zero-copy views of observation and candidate buffers, `DealSpec`, replay |
| `oracle/` | Python | `gd_reference.py`, the slow independent rules oracle. Test-only. |
| `train/` | PyTorch | Model, DMC learner, PPO learner, league, buffer, checkpointing |
| `eval/` | Python | Arena, duplicate deals, Elo, probes, OpenGuanDan adapter and trace tools |
| `play/` | Python | CLI and a minimal web UI for human games |
| `infra/` | Docker, shell | Image, launch scripts, checkpoint sync, watchdog |

### 4.1 Key decisions and trade-offs

| # | Decision | Alternative | Why |
|---|---|---|---|
| D1 | Model-free self-play with action features as input | AlphaZero-style MCTS | See section 2. Proven on DouDizhu and Guandan. |
| D2 | Own C++ engine | Train inside the OpenGuanDan Java simulator | The simulator is socket and JSON based and about two orders of magnitude too slow for single-node training. We still use it for parity tests and final evaluation. |
| D3 | Synchronous vectorized rollout in one process | Actor and learner processes with RPC, as in DouZero and DanZero | No parameter staleness, no serialization, large GPU batches, far less code. Costs some utilization while the learner runs. |
| D4 | Canonical action set for training | Full concrete action set | Cuts candidates from thousands to roughly a hundred or two. Risk: a reduction removes a useful play. Each reduction has its own flag (RULES.md 11.2). |
| D5 | Self-play runs full matches, learning targets are per-round returns | Independent random rounds | Levels, tribute hands and lead positions then follow their true distribution with no sampling tricks |
| D6 | One network shared by all four seats | One network per seat or role | Seats are symmetric in Guandan, unlike the three roles of DouDizhu |
| D7 | Duplicate deals as the primary internal metric | Plain win rate | Card luck dominates single results. Pairing removes most of it. |
| D8 | Python oracle written independently of the engine | Test the engine against itself | A shared misunderstanding of the rules would otherwise pass every test |
| D9 | History enters through a causal Transformer over public action tokens, read by one private query token | Flat cumulative features only, or an LSTM over the last few moves | Who passed on what, and in which order, is the evidence for hidden hands and is absent from flat features. Keeping the stream public lets all four seats share one KV cache per environment. |
| D10 | Cross-round opponent memory as one summary vector per seat per finished round | The raw token history of the whole match in context | Raw match context costs about 75 GB of KV cache across 8,192 environments. Summaries cost almost nothing. |
| D11 | Architecture choices are gated by a cheap supervised belief probe before any RL compute is spent | Decide by full RL ablations only | Hours instead of weeks, and it tests the exact mechanism the Transformer is supposed to improve |

## 5. Rules engine

### 5.1 Data structures

1. `CardId`: 0 to 53 as in RULES.md section 3.
2. `Hand`: two 64-bit masks over card ids, `has1` (at least one copy) and `has2` (two copies). Cached views: count per rank (15 entries) and one 13-bit rank mask per suit for straight flush checks.
3. `Action`: type, key, abstract id (0 to 392), card multiset as a `Hand`, wild count.
4. `RoundState` and `MatchState`: plain structs, trivially copyable, so that search can clone them cheaply later.

### 5.2 Move generation

Two layers.

**Layer 1, abstract feasibility.** For each abstract action compute the deficit: the number of required cards the hand lacks in natural cards. The action is feasible if the deficit is at most the number of wild cards held. When following, only abstract actions that beat the top play are visited, which makes most follow decisions very cheap.

**Layer 2, concretization.** For a feasible abstract action choose actual cards per rank. Full mode enumerates every suit combination and every wild assignment. Canonical mode applies the three reductions of RULES.md 11.2. Straight flushes are generated from the per-suit rank masks.

Forced moves are resolved inside the engine. If pass is the only legal action the environment passes on its own and records it, so the network is never called for it.

### 5.3 API sketch

```cpp
struct EnvConfig { RuleConfig rules; ActionMode mode; int history_len; /* ... */ };

class VecEnv {
 public:
  VecEnv(int num_envs, int num_threads, EnvConfig cfg, uint64_t seed);
  void reset(std::span<const DealSpec> deals = {});   // empty means random deals
  // Advance every env until it needs a decision or its match ends.
  // Fills flat buffers: obs [B, obs_dim], cand [sum_k, act_dim], offsets [B + 1],
  // plus env id, seat, phase (tribute, back-tribute, play) per row.
  DecisionBatch pending();
  void step(std::span<const int32_t> choice_index);   // one index per pending row
  std::span<const RoundResult> drain_finished_rounds();
  // Copy one env into `copies` fresh slots at its current decision point.
  // Used for counterfactual tribute branches and later for search.
  std::vector<int> fork(int env_id, int copies);
};
```

Python sees read-only NumPy views of reusable CPU buffers. The learner copies
features into owned storage and pins transfer tensors when using CUDA; the
engine storage itself is not pinned. Tribute and back-tribute decisions use
the same batch path with their own candidate lists and an in-engine heuristic
choice. Stage A uses that choice. Private hidden-hand labels are separate from
observations, and every decision/result is keyed by environment, match and
round. See `PY_API.md` for the implemented interface.

### 5.4 Testing strategy

| Layer | What | Tool | Volume |
|---|---|---|---|
| Unit | Every vector in RULES.md section 12 | C++ test framework | All |
| Oracle cross-check | `interpret` and `beats` on random multisets of 1 to 10 cards at random levels | pytest plus the oracle | 10^5 cases |
| Completeness | Full mode legal set equals the oracle's brute force on random hands of 12 cards or fewer | pytest plus the oracle | 10^4 cases |
| Property | The ten invariants of RULES.md section 14 under random play | hypothesis and a C++ fuzzer | 10^7 rounds, plus 10^5 under ASan and UBSan |
| Scenario | T-FLOW scripts, tribute cases, match end cases | pytest with `DealSpec` | All |
| Differential | Replay logged OpenGuanDan games (section 9.2) | Trace tools | At least 1,000 matches |
| Performance | Decisions per second, candidates per decision (mean, p50, p99, max), both modes | `bench/` | On every engine change |

## 6. Observation and action encoding

Binary planes use two bits per card id, one for at least one copy and one for two copies. That is 108 values per card set.

| Block | Size | Content |
|---|---|---|
| Own hand | 108 | |
| Unseen cards | 108 | Everything not in own hand and not yet played |
| Cards played by LHO, partner, RHO, self | 4 x 108 | Cumulative for the round |
| Cards left for LHO, partner, RHO | 3 x 28 | One-hot, 0 to 27. Exact, derived from public plays, which is legal under the house declaration rule. |
| Finish status | 4 x 4 | Per seat: active or the finish position |
| Levels | 3 x 13 | Round level, own team level, opposing team level |
| Wild information | 3 + 3 + 12 | Wild cards held, wild cards still unseen and DanZero-style flags for what a held wild card can complete |
| Trick context | about 160 | Top play (action encoding), who holds it (LHO, partner, RHO or none), passes so far, leading flag |
| Last action per other seat | 3 x action encoding | |
| Phase and roles | 3 + 6 | Phase one-hot (play, tribute, back-tribute). Own finishing role last round. Whether the tribute counterpart is the partner or an opponent. |
| Tribute context | 4 x (54 + 8) | Every public exchange of this round: the card plus payer and receiver as relative seats. Zero when there was no tribute. |
| Known holdings | 3 x 54 | Cards publicly known to sit in another seat's hand: tribute and back-tribute cards that seat has not played yet |
| History (v2) | up to 160 tokens per round | One token per play or pass: absolute seat, action encoding, cards left after the action. Public information only. See section 7.2. |
| Match memory (v3) | up to 3 x 24 vectors | One summary vector per other seat per finished round. See section 7.3. |

| Action encoding block | Size |
|---|---|
| Cards | 108 |
| Type one-hot | 13 (the 11 play types plus tribute and back-tribute) |
| Key one-hot | 15 |
| Bomb size one-hot | 7 |
| Wild cards used | 3 |
| Tribute candidate flags | 8 | Only for tribute phases: copies of that rank held, whether giving it breaks a pair, a triple or a bomb, whether it is SF-relevant (RULES.md 11.2), whether it is a 5 or a 10 |

The encoder lives in C++ and is covered by golden tests: fixed states with checked-in expected tensors.

v1 ships without the history block so that the first baseline is a plain MLP comparable to DanZero. v2 and v3 are described in section 7.

## 7. Model

Two towers with late fusion. This interface is fixed across versions. v1, v2 and v3 differ only in how the state embedding is produced, so the action tower, the heads, the losses and the rollout loop never change.

1. **State tower**: produces a state embedding. Runs once per decision.
2. **Action tower**: small MLP over the action encoding. Runs once per candidate.
3. **Fusion heads, one per phase**: play, tribute and back-tribute. Each is an MLP over the concatenation and gives one scalar per candidate. For the play head that scalar is `Q(s, a)` in the DMC stage and the policy logit in the PPO stage. The two tribute heads output the advantage of giving that card (section 8.3). A tribute candidate is a single card, so it reuses the action tower.
4. **Auxiliary heads** from the state embedding: hidden hand prediction for each other seat (per card id, 0, 1 or 2 copies) and final finishing position. Self-play gives the labels for free.

DouZero and DanZero run the entire network once per candidate. With about 100 candidates on a lead, the two-tower layout costs a small fraction of that.

Stage A supports bf16 inference on capable CUDA devices, bounded candidate
chunks, and ragged segment-wise selection. CUDA graph bucketing is a future
performance optimization, to be justified by measurements on the selected GPU.

### 7.1 v1: MLP state tower

The flat features of section 6 go through a 4 x 512 MLP. Action tower 2 x 256, fusion 3 x 256, a few million parameters in total. Purpose: validate the whole pipeline, reach gate M1 and serve as the baseline every later version is measured against.

### 7.2 v2: Transformer over the round history

**Why.** The network never sees hidden hands, so its play quality depends on how well it infers them. The evidence is in the sequence: who passed on which play, who spent a bomb on what, in which order. Cumulative played-card features throw that away. DanZero sees only the last action of each seat. This is the most likely place to pass the published agents on architecture alone.

**Public stream.** One token per play or pass, built from public information only and tagged with the absolute seat. A causal Transformer encodes the stream. Starting size: 4 layers, width 256, 4 heads. Because nothing private enters the stream, its encoding is identical for all four seats, so each environment keeps one KV cache and appends one token per action.

**Private query.** At a decision the acting seat's private and contextual features (own hand, seat, levels, wild card information, tribute context, known holdings) form one query token. It cross-attends to the cached stream and the result, passed through an MLP, is the state embedding. In serving terms the round is prefilled incrementally and each decision decodes one token.

**Training.** One round is one sequence. Query tokens for every decision of the round are interleaved with the public tokens under a block mask: a query sees the public prefix up to its time and itself, and public tokens never see queries. That rule keeps private information out of the shared stream. One forward pass trains every decision of the round, so the replay buffer stores rounds, not independent samples.

**Memory.** Per token the cache holds 2 x 4 layers x 256 x 2 bytes, about 4 KB. At up to 160 tokens per round that is about 0.65 MB per environment and about 5 GB for 8,192 environments in fp16. If that crowds the learner on a 24 GB card, halve the environments.

### 7.3 v3: match memory for opponent habits (experiment)

**Goal.** Adapt within a match to how these particular opponents play: whether they hoard bombs, how they spend wild cards, how they defend when someone is nearly out. No published Guandan agent does this, and it matters most against humans, which is where every published agent is still below 50%.

**Mechanism.** When a round ends, pool the v2 stream outputs at each seat's tokens, together with that seat's remaining cards if the round end revealed them, into one summary vector per seat. The private query of later rounds also attends to these vectors. A match holds a few dozen of them, so the cost is negligible.

**Precondition.** Testing adaptation to individual opponents needs stable differences between their styles. Identical learner copies may share tendencies, but do not provide controlled between-opponent style variation. The league therefore needs stylized opponents whose style stays fixed for a whole match: bomb-happy and bomb-shy variants, older checkpoints at different sampling temperatures, heuristic bots with different parameters. Memory can still receive gradients and learn other match information without this variation, so a prediction gain alone does not identify opponent-habit learning.

**Styled heuristic bots.** The source of those habits is `styled_bot` in `cpp/src/bots.cpp`: one heuristic player driven by a continuous 18-float style vector, `StyleParams`, rather than a fixed list of bots. Slot 0 is a bomb-early threshold that sets the opponent-pressure level at which the bomb class unlocks; slots 1 to 13 are one additive log-weight per play type of the `Type` enum, applied when leading; then a follow-aggressiveness weight, a lead high/low bias, a partner-cooperation weight that decides how close the partner may be to going out before it is overtaken, and a sampling temperature. The bot reduces every candidate to one scalar score and takes the argmax, or a softmax draw at a non-zero temperature; the score unit is one logit, so a type preference of 1 multiplies that play's probability by `e` at temperature 1. `StyleParams::neutral()` reproduces `greedy_bot` exactly, which is what pins the parameterisation to the existing baseline. Styles change nothing in the rules engine, move generation or `RoundState`/`MatchState`: they only reorder candidates the engine already produced. `VecEnv::set_styles` carries a style per environment and seat and reports `styled_choice` beside `greedy_choice`, so a match can hold its styles fixed for its whole length, which is the precondition above. Sampling styles per match, from a distribution that a probe can hold out regions of, is what makes the section 7.4 probe able to measure cross-round opponent modelling at all.

**Status.** Research, not schedule. It proceeds only if the adaptation metric of section 9.1 shows an effect in a small run.

### 7.4 Belief probe: the gate for v2 and v3

Once v1 self-play produces logs, run a supervised experiment with no RL in it: train only the hidden hand head on logged decisions, once with the v1 tower and once with the v2 tower, at matched parameter counts. Compare held-out log loss, broken down by stage of the round and by seat (partner against opponents). It costs a few GPU hours.

1. If v2 predicts hidden hands clearly better, build v2 into the RL pipeline and confirm with an equal-compute duplicate-deal comparison against v1.
2. If the two are close, v2 is demoted and the compute goes to the critic, the league and endgame search, which are the larger levers by the record of prior work: PerfectDou passed DouZero through its critic and GS2 passed SDMC through search.

Measured status, Sept. 22: the corrected styled-opponent probe is complete on
100,000 rounds from the trained M1 final and three training seeds. Mean
held-out-style test loss is 0.3421048375 (flat), 0.3366016650 (no_history) and
0.3366812724 (history). The no_history structure improves flat by 1.6086%;
additional history wins one seed and loses two. The pooled paired difference,
defined as no_history loss minus history loss, is -0.000081 with a 95%
whole-match bootstrap interval [-0.000100, -0.000061], conditional on the
three fitted seeds. The original three-seed history gate remains
`v2_not_yet_justified`. **Stage B uses no_history for this tested protocol.**
This does not establish RL or playing-strength gains, or exclude benefits
from a different scale or training protocol.

Seven of nine fits reached the 50,000-step cap; the two seed-33 query models
stopped for validation plateau at 42,000 and selected step 36,000. Overall
convergence is not established. Excluding the 35 possibly quota-truncated test
matches in a post-hoc sensitivity analysis leaves the conclusion unchanged;
the primary split and gate are unchanged. The earlier smoke-policy result is
retained separately as historical evidence. See
[the corrected report and paired intervals](reports/M2-belief-corrected.md).
The independent v3 memory-versus-masked experiment is complete on the corrected
collection with seeds 41/42/43. Mean test loss is 0.336921326 (memory) versus
0.337168381 (masked). Overall paired benefit wins two seeds and loses one;
the late-benefit interval is positive for seed 41, crosses zero for seed 42
and is negative for seed 43. All three paired adaptation intervals cross
zero. The pooled overall and late benefits are positive conditional on these
fits, but do not remove the optimizer-seed disagreement. Pooled adaptation is
+0.000022, with 95% interval [-0.000130, +0.000173]. Excluding 35 possibly
quota-truncated test matches leaves the conclusion unchanged. Three of six
fits reached the 50,000-step cap, so convergence is not established.

The unchanged v3 gate is `v3_not_yet_justified`. Stage B continues with
no_history; this memory prototype is not promoted. This does not rule out
larger-scale memory models or the value of opponent habits, and no RL or
playing-strength claim follows from this supervised experiment. See
[the memory report](reports/M2-memory.md). Both experiments are complete,
all 15 selected checkpoints and the immutable archive are verified, and pod
deletion is confirmed. Runtime and cost evidence are in
[the completed run record](reports/M2-next-run.md).

The same probe, with opponents of fixed style and the log loss measured round by round within a match, is the first test for v3.

Expectation, stated in advance so that results can be judged against it: v2 is a moderate and fairly certain gain, a few points of win rate at equal compute, not a step change. v3 has the highest ceiling and the highest chance of showing nothing.

## 8. Training

### 8.1 Rollout loop

`N` matches run in lockstep, with `N` between 4,096 and 16,384. Each iteration takes one decision per environment, runs one forward pass and steps all environments in the C++ thread pool. When a round finishes, every stored decision of that round receives its return and moves to the replay buffer. Collection and learning alternate in the same process. With the v1 model the buffer holds independent decisions. With v2 it holds whole rounds as sequences, and the rollout keeps one KV cache per environment that is cleared at every round end (section 7.2).

### 8.2 Stage A: Deep Monte Carlo

1. Target: the round return for the acting seat's team. Plus 3, 2 or 1 for the Banker's team by the partner's position, the negative for the other team. In a level A round, a Banker and Dweller result scores 0 for the team that owns the round, because it cannot pass A that way. This follows DanZero and OpenGuanDan.
2. Loss: mean squared error between `Q(s, a)` and the return, plus weighted auxiliary losses. No bootstrapping, no discount.
3. Exploration: epsilon-greedy over candidates, epsilon annealed from 0.1 to 0.01.
4. Bootstrapping out of random play: early on, a share of opposing seats is driven by the in-engine greedy bot, decaying from 50% to 0. Only network-driven seats produce training samples. This is the SDMC lesson without needing an expert.
5. Play-time policy: sample uniformly among candidates whose Q is within a small margin of the best one, to be less predictable.

Collection and learning share a process. Completed-round samples are used for
one learner phase and then discarded; incomplete rounds retain their recorded
decisions across phases until the return exists. The initial configuration
collects 32 vector steps and performs 64 minibatch updates of 2,048 samples,
with explicit sample/throughput metrics for tuning. No staleness clipping is
applied in Stage A.

### 8.3 Stage A2: tribute and back-tribute heads

Tribute is strategy, not bookkeeping. The payer picks the suit of the card it gives up. The receiver picks any card from 2 to 10 to return, and the right card depends on whether it goes to an opponent or, after a Banker and Dweller win, to the partner. The exchanged cards are public, which the play policy can exploit through the known holdings block.

Why it is not learned from the first step: the value of a returned card is defined by how well the following play uses the hands. While the play policy is weak that signal is noise. The signal is also small. One card at the start of a round moves the expected return by hundredths of a level against a standard deviation near 2.

Stage A therefore uses a heuristic, and Stage A2 learns the heads with a low-variance target:

1. **Heuristic (Stage A).** Tribute: among tied top cards give one whose suit is not SF-relevant. Back-tribute to an opponent: the lowest card that breaks no pair, triple, bomb or straight flush window, avoiding the 5 and the 10 because every straight contains one of them. Back-tribute to the partner: the highest eligible card under the same structural constraints.
2. **Counterfactual branches.** At a sampled fraction of tribute decisions (start at 2% of rounds) the environment is forked once per candidate card. Every branch is played to the end by the current play policy in greedy mode with identical seeds. Holding the deal fixed cancels card luck, the same idea as duplicate deals.
3. **Target and loss.** The target for candidate `k` is its branch return minus the mean over branches. The head regresses it with a Huber loss. Rollouts use the true hidden hands while the head sees only the acting seat's observation, so it learns the expected advantage over what it cannot see.
4. **Freezing.** The shared towers stay frozen while the tribute heads train. The heads inherit the play policy's understanding of hand structure and the play policy cannot be damaged. Stage B unfreezes everything.
5. **Double tribute** has two tribute and two back-tribute decisions. Branch one decision at a time and let the current heads or the heuristic make the others.
6. **Measuring the payoff.** Same play model, learned heads against the heuristic, on duplicate rounds that start with a tribute phase. The difference in levels per round is what tribute strategy is worth. If it is within noise the heuristic stays and the compute goes elsewhere.

Cost: with about ten candidates per back-tribute decision, labelling 2% of rounds adds roughly 20% to rollout compute.

First measured decision (Sept. 21): retain the heuristic. The three fitted
heads scored -0.0815, -0.0645 and -0.0905 net levels per round against it on
1,000 shared fresh duplicate deals per candidate; no candidate had a positive
lower 95% bound. This is evidence for the current v1/data/budget configuration,
not a general claim that tribute learning cannot help. Full evidence and the
observation-semantics correction are recorded in `reports/M2-A2.md`.

### 8.4 Stage B: PPO with a perfect-information critic and a league

1. Policy: softmax over fusion logits, initialized from Stage A as `Q / temperature`. Candidates are pruned to the top `k` by the frozen Stage A network, with `k` around 32, plus pass.
2. Critic: a separate network that sees all four hands and outputs the expected round return. Used only during training.
3. Advantage: GAE with discount 1. Entropy bonus, plus a KL penalty toward the Stage A policy that is annealed away.
4. League: the learner controls both seats of one team. The other team is drawn from a pool: the latest checkpoint, older checkpoints, heuristic bots and, if v3 is pursued, stylized opponents that keep one style for a whole match (section 7.3). Opponents that beat the learner more often are sampled more. Only learner seats produce samples. At most four distinct opponent models are active at a time so that inference still batches well.
5. Auxiliary losses stay on.
6. Tribute and back-tribute decisions are ordinary steps of the trajectory from here on. Their heads start from Stage A2 and train with the same advantage estimates as the play head.

### 8.5 Stage C: endgame search (optional)

When few cards remain unseen, sample hidden hands consistent with public information, weighted by the hidden hand head. For each of the top few candidates, roll the round out with the policy in every seat and average the returns. Override the policy only when the margin is clear. The engine's trivially copyable state makes rollouts cheap.

### 8.6 Measured first-run load

The September 21 RTX 5090 pilot used 4,096 environments, eight engine threads,
four Torch threads and the full 3.08M-parameter v1 model with bf16. In five
minutes it processed 18.77M decisions (about 62,500/s), 324,987 rounds and
9,024 optimizer updates. Peak host RSS was about 4.2 GiB; CUDA peak allocation
was 666 MiB, with a largest candidate batch of 803,808. A typical 32-step
collection phase took 1.8–1.9 seconds versus about 0.15–0.22 seconds for 64
learner updates. Collection, including inference and Python trajectory work,
dominates this configuration. These measurements replace the earlier estimate
that the learner would be the bottleneck. They are from a short supervised
run, not a sustained daily throughput guarantee. See `reports/M1-runpod.md`.

## 9. Evaluation

### 9.1 Internal arena

1. **Duplicate deals.** A deal fixes four hands, the round level and the leader. It is played twice with the two teams swapping seats, and the score is the difference in level gain. With a per-deal standard deviation near 2, 10,000 deals resolve differences of about 0.04 levels per round. Report the mean with a bootstrap confidence interval.
2. **Match play.** Full matches, win rate with a Wilson interval. Needed because tribute and level A play only exist at match level.
3. **Elo** across all checkpoints from duplicate results, refreshed on every new checkpoint.
4. **Metrics per agent pair**: mean level gain per round, Banker rate, double win rate, match win rate.
5. **Belief quality**: held-out log loss of the hidden hand head, by stage of the round and by seat. Tracked for every checkpoint, and the deciding metric of the belief probe (section 7.4).
6. **Adaptation (v3 only)**: against opponents of fixed style, mean level gain in rounds 6 and later minus rounds 1 to 5 of the same matches, compared with the same model with its match memory masked. No difference means nothing was learned.

### 9.2 OpenGuanDan adapter and differential testing

1. `ogd_adapter/client.py` connects our agent to the OpenGuanDan simulator through its per-player JSON API and maps its `actionList` entries to our actions.
2. `ogd_adapter/trace_logger.py` runs four random clients, logs every message and reconstructs the four hands from the `beginning` notifications.
3. `ogd_adapter/replay_diff.py` replays each logged match in our engine. At every decision it compares the legal action set (full mode, normalized to type, key and card multiset). At every round end it compares the finishing order, the level changes and the tribute flow.
4. The replay runs under the `ogd` rule profile. Divergences are grouped by class. Each class either fixes a bug, fills in a value of the `ogd` profile or closes a parity check in RULES.md section 13. Training and human play use the `house` profile.
5. For gates M1 to M3 the agent plays 1,000 matches against each built-in agent.

Settled in M0, and not as hoped. The repository ships no agents and no weights: it contains the rules server, an Electron client and the move generator jar, nothing else. It also carries no LICENSE file, so nothing from it is vendored here and the adapter locates a local clone through `OGD_ROOT`.

The consequence is that the opponents named in the M1, M2 and M3 gates cannot currently be played against. The fallback from the risk table becomes the plan: train a DanZero-style baseline with our own pipeline and use it as the reference point, treating the published win rates as calibration rather than as a ladder. This needs a decision before M2 begins, because it changes what those gates can mean. See `docs/reports/M0.md` section 5.

M0 also replaced the socket route with something cheaper: the repository's `guandan-java/` directory exposes the Java move generator to Python directly, which is what the trace logger and replay diff use.

### 9.3 Behavior probes

Scripted positions with a known good answer, reported as pass rates per checkpoint:

1. Partner holds one card and we lead: do we lead a low single?
2. An opponent holds one card: do we avoid leading singles and cover singles with our highest card?
3. Partner's play is the top play and the opponents have passed: do we pass instead of overtaking?
4. Opponent about to go out: do we spend a bomb?
5. No threat on the table: do we refrain from wasting a bomb or a wild card?
6. Teammate lead: after inheriting the lead, do we play toward our partner's likely finishing order?
7. Tribute: with several top cards of different suits, do we avoid giving the suit that completes our own straight flush?

Cooperation failures hide inside decent win rates, which is why these are tracked separately.

### 9.4 Robustness and humans

1. **Exploiter test.** Freeze a checkpoint, train a fresh agent against it for a fixed budget and report how much it wins. A rising number across checkpoints means the policy is getting more exploitable.
2. **Human play.** Web UI, every game logged and replayable. Only with players who know they are facing a bot. The UI follows the house rule and shows another seat's card count only at ten cards or fewer.

## 10. Infrastructure and cost

1. One Docker image. A fresh node is ready after `git clone`, image pull and checkpoint download.
2. Checkpoints every 10 minutes: model, optimizer, step, RNG state and league metadata, written atomically and synced to object storage or a network volume. Environments are stateless and restart from fresh deals.
3. Launch through `infra/run_train.sh` inside tmux. Metrics go to TensorBoard or Weights and Biases.
4. `infra/watchdog.py` stops and deletes the instance when the budget cap is reached, when Elo has not improved for a set number of hours or when the run finishes. On marketplace providers a stopped instance still bills storage, so the watchdog deletes.
5. Pick the node by vCPU count and price, not by GPU tier. A 4090 or 5090 class card is sufficient.

Price snapshot, to be rechecked before each long run: RunPod listed the RTX 4090 from 0.34 USD per hour on its community tier and 0.69 on its secure tier, with the RTX 5090 at 0.99 (August 2026). Vast.ai 4090 listings ran about 0.35 to 0.50 on demand and 0.29 to 0.31 interruptible (April 2026). One 4090 running all month is therefore about 250 to 500 USD on demand. Interruptible capacity is advertised at 50% to 80% less, which the resumable design is meant to exploit.

| Phase | GPU time | Rough cost |
|---|---|---|
| M0 | None | 0 |
| M1 smoke runs | Hours | Under 10 USD |
| M1 full | Days | Tens of USD |
| M2 and M3 | Weeks | Low hundreds of USD |

## 11. Repository layout

```
guandan-ai/
  CLAUDE.md               build, test and style instructions for the coding agent
  docs/                   DESIGN.md, RULES.md
  cpp/                    gd_core: include/gd/*.h, src/, tests/
  python/gd/              pybind11 package
  oracle/                 gd_reference.py, test_gd_reference.py
  train/                  model.py, dmc.py, ppo.py, league.py, buffer.py, ckpt.py, configs/
  eval/                   arena.py, duplicate.py, elo.py, probes/, ogd_adapter/
  play/                   cli.py, web/
  bench/                  throughput benchmarks
  infra/                  Dockerfile, run_train.sh, sync.sh, watchdog.py
  tests/                  pytest and hypothesis suites
```

## 12. Milestones

Gates, not dates. Each milestone ends with a short written report in `docs/reports/`.

| Milestone | Deliverable | Gate |
|---|---|---|
| M0 | Engine, oracle cross-checks, fuzzing, vectorized env, benchmarks, OpenGuanDan parity, feature encoder | Section 1.2, row M0 |
| M1 | v1 model, DMC training end to end, arena, Elo, behavior probes, belief probe on the self-play logs | Row M1 |
| M2 | v2 Transformer if the belief probe supports it, confirmed at equal compute. Learned tribute heads with their measured payoff. PPO with critic and league. Exploiter test. | Row M2 |
| M3 | Endgame search, human play UI, v3 match memory experiment with stylized league opponents | Row M3 and the stretch goal |

### 12.1 M0 task list

Work top to bottom. Each task is done when its check passes in CI.

| # | Task | Check |
|---|---|---|
| 1 | Scaffold: CMake, pybind11, pytest, CI. Add the oracle and its tests. Write `CLAUDE.md`. | `python3 oracle/test_gd_reference.py` passes in CI |
| 2 | Cards, `Hand`, text parsing and printing | Round trip tests over all 54 ids and random hands |
| 3 | `power`, windows, `interpret`, `beats` in C++ | Every vector in RULES.md section 12. 10^5 random multisets agree with the oracle. |
| 4 | Move generation, full mode | Equal to the oracle on 10^4 random small hands. Sound on random 27-card hands. |
| 5 | Move generation, canonical mode | Subset and best-reading invariants. Candidate count statistics recorded. |
| 6 | Round and match state machine, tribute, level A rules, `DealSpec` | T-FLOW scenarios and tribute vectors |
| 7 | Random and greedy bots, fuzzer | 10^7 rounds with all invariants. 10^5 rounds clean under ASan and UBSan. |
| 8 | `VecEnv` and Python bindings | Python smoke test plays 1,000 matches with random choices |
| 9 | Benchmarks | Report written. Target met or the gap explained. |
| 10 | OpenGuanDan trace logger and replay diff | 1,000 matches replay with no unexplained divergence. `ogd` profile filled in, parity checks in RULES.md closed and the document updated. |
| 11 | Feature encoder v1 | Golden tests |

## 13. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Our rules differ from the benchmark | Evaluation numbers are meaningless | Trace replay in M0 under the `ogd` profile. The `house` profile differs only in rare tribute ties and match-level A rules. |
| A canonical reduction removes plays that matter | Strength ceiling | Per-reduction flags, ablation in M1 against full mode on a small budget |
| Self-play cycles or collapses into a narrow style | Beats itself, loses to others | League, heuristic opponents, exploiter test, external baselines at every gate |
| Agents fail to cooperate | Poor results with a partner | Behavior probes, auxiliary hand prediction, critic that sees the partner's hand |
| Evaluation noise hides regressions | Wrong decisions at gates | Duplicate deals, confidence intervals, fixed evaluation seeds |
| Interruptible instance reclaimed | Lost work | 10 minute checkpoints, stateless environments, automatic resume |
| Cost overrun | Budget | Watchdog cap, smoke runs before long runs, throughput measured first |
| Built-in agent weights unavailable or unusable | No external ladder for M2 and M3 | Check in M0. Fallback: retrain a DanZero-style baseline with our own pipeline as the reference point. |
| Engine slower than the target | Longer and costlier training | Profile move generation first. The load estimate has an order of magnitude of slack. |
| v2 adds complexity without strength | Wasted weeks | Belief probe first, then an equal-compute comparison. The two-tower interface keeps v1 as a drop-in fallback. |
| KV cache or sequence replay slows the rollout loop | Lower throughput than v1 | Measure decisions per second for v2 before any long run. Fewer environments or a narrower encoder if needed. |
| v3 learns nothing because opponents have no habits | Research time with no result | Stylized league opponents, the adaptation metric and a small run before any commitment |

## 14. What to revisit later

1. Training the tribute heads jointly from the first step, if Stage A2 shows that tribute is worth a lot.
2. A match-level value so the agent trades round reward for match win probability near level A.
3. Whether `wild_usage = all` or a richer treatment of suits buys strength.
4. Population diversity beyond a single learner, if the exploiter test shows a persistent weakness.
5. Multi-GPU only if the learner is measurably the bottleneck and the budget allows.
6. Attention over the hand itself, with cards as tokens, if the flat hand features turn out to limit v2.

## 15. References

1. Zha et al., DouZero, ICML 2021. arXiv 2106.06135.
2. Lu et al., DanZero, IEEE CoG 2023. arXiv 2210.17087.
3. Zhao et al., DanZero+. arXiv 2312.02561.
4. Yanggong et al., GuanZero. arXiv 2402.13582.
5. Yang et al., PerfectDou, NeurIPS 2022. arXiv 2203.16406.
6. Li et al., Suphx. arXiv 2003.13590.
7. Ge et al., Efficient subgame refinement for extensive-form games (GS2), NeurIPS 2023.
8. Ge et al., Solving Guandan poker games with deep reinforcement learning (SDMC), Journal of Computer Research and Development, 2024.
9. Li et al., OpenGuanDan. arXiv 2602.00676. Code: https://github.com/GameAI-NJUPT/OpenGuanDan
10. Li et al., Tjong: a transformer-based Mahjong AI, CAAI Transactions on Intelligence Technology, 2024.
11. Jing et al., Towards offline opponent modeling with in-context learning (TAO), ICLR 2024.
12. Jing et al., Opponent modeling with in-context search (OMIS), NeurIPS 2024.
