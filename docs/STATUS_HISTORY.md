# 历史状态记录（2026-11-19 之前）

这是 2026 年 11 月 19 日改写前的 `STATUS.md` 全文，原样保留，按日期倒序记录到当天为止的各项进展。
当前状态见 [STATUS.md](STATUS.md)，术语见 [GLOSSARY.md](GLOSSARY.md)。下面的相对链接仍然有效。

---

# Project orientation for agents

November 19: Velocity job 22095 passed 16 CUDA checks on each L40S, then
stopped at the legacy cold frozen replay. The largest candidate logp delta
was `1.2207e-4` against a `1e-4` limit. Choices and sampler RNG matched.
The eight-rank check, PPO throughput, card memory peak and fallback run were
not reached. Job 22092 separately exited before training with `KeyError: 'R'`
after passing its 10 parity tests. See the
[gate result](reports/large-decision-trunk-plan-2026-11-18.md#11-velocity-门禁结果2026-11-19).

November 18 (implemented, local CPU checks passed): the opt-in wider decision
path keeps the public stream at width 128, adds a 512-wide private encoder and
a residual `640->1024->512` state trunk, and scores candidates through
`640->1024->1024->1`. The target actor has 6,578,150 parameters with the
auxiliary heads and the critic has 1,555,969. Legacy fingerprints, CPU cache
and PPO checks, save/resume, CPU DDP and evaluation smoke checks passed. See
the [implementation record](reports/large-decision-trunk-plan-2026-11-18.md#9-实现记录2026-11-18).
CUDA numerics, throughput, memory and playing strength are unmeasured.
The authorized two-L40S gate is queued as job 22095 in an isolated directory.
Its [submission record](reports/large-decision-trunk-plan-2026-11-18.md#10-velocity-门禁提交记录2026-11-18)
records the benchmark and fallback settings. The queue receipt contains no results.

November 18: Velocity job 22092 replaced 22087 before it ran. The scheduler
released the new two-L40S job to the queue for the 40,000-update
`seq-cc-trick-np-u234993` run. It enables all three rollout options for the
first production update if CUDA replay and two eight-rank PPO updates pass.
If the optimization checks fail, the segment uses its baseline configuration.
The [submission record](reports/velocity-rollout-optimizations-2026-11-18.md#作业-22092-替换-22087)
identifies the source and launch script.

November 18: Three rollout optimizations are available as opt-in training and
resume flags. On frozen 22087 weights, the learner decoder K/V cache and compact
snapshot projections together improved local CPU collection throughput by 19.0%
over four trials per arm. Velocity CPU job 22090 passed the frozen replay and
resume checks. Velocity full PPO throughput and GPU memory for the new
combination remain unmeasured. See the
[implementation and validation record](reports/velocity-rollout-optimizations-2026-11-18.md).

November 18: The experimental `card_counter` mask now retains the current
trick and the remaining-card counter. It retains 545 observation dimensions
and masks 1,304. Velocity job 22087 replaced 22048 before it started and
queues the revised mask in `runs/seq-cc-trick-np-u234993`. The November 17
entry below describes the original submission. See the
[training guide](TRAINING.md) for the current configuration and the
[deployment record](reports/sequence-decoder-nearest-pair-2026-11-17.md)
for the replacement's source identity and validation scope.

November 17: [multi-layer sequence decoder, card-counter masking and nearest-pair archive](reports/sequence-decoder-nearest-pair-2026-11-17.md)
implemented opt-in `PrivateCrossDecoder` (per-head QK-norm, multi-slot queries,
relative seat/round/step biases), `card_counter`/`private_only` observation
masking with PPO-consistent warmup, offline checkpoint migration
(`train.history_sequence_migrate` + `--init-checkpoint`), merged snapshot
inference/encoding support, and `nearest_pair` population archiving with a
persistent stratified multi-round behavior probe. Local suites (87 C++, 1,764+
Python, snapshot revision verification), Velocity CPU preflight/migration job
22039 (`u234993` -> `u234993-seq-cc-np.pt`, heldout TV 0.00262, 0% one-hot
saturation), and L40S CUDA parity/DDP smoke checks (22040, 22049) passed after
allowing Tier-2 1-ULP float tolerance on CUDA strided paged KV; 2xL40S training
job 22048 is queued (`runs/seq-cc-np-u234993`). Full 2-GPU throughput and
playing-strength gains remain unmeasured; paused `runs/main` is unchanged.

November 17: Main training paused on Velocity at the user's request. The
queued continuation 22000 was cancelled before starting, and the main run's
`STOP` marker was set. The durable u234993 checkpoint from job 21968 was
verified by SHA-256. See the [paired DanLM evaluation](reports/main-u234993-danlm-2026-11-17.md).

November 15: [public-history identification and layer diagnostics](reports/history-identification-2026-11-15.md)
completed 8,000 frozen rounds, 48 diagnostic fits, three fit audits and a fresh
160-history conditional-opportunity test. Public policy likelihoods recover
79% of known-identity prediction benefit after three rounds; a reference using
only publicly determined opponent hands recovers 67% without hidden-hand Monte
Carlo. Raw-token and all tested frozen-layer/readout probes fail to generalize,
so no single encoder/attention layer is uniquely implicated. Publicly reconstructed
choice opportunities plus a frozen u195450 reference recover a small 4.56% after
ten rounds on new histories. Some posterior Monte Carlo remains unstable; no
playing-strength or unique reward/capacity cause is established. Main unchanged.

November 15: [frozen filtered-pool identity oracle](reports/filtered-identity-oracle-2026-11-15.md)
completed 192 independent roots and two sets of 128 hidden worlds per root.
Same-hand policies differ: weighted pair top-action disagreement is 7.80%.
After uniform hidden-hand averaging, known identity still improves independent
prediction by Brier 0.000869 for full actions / 0.001415 for 11 action types;
both paired 95% intervals exclude zero. Only 6.2% / 11.9% of same-world
full-information identity benefit remains. This is a neutral-prefix, node-assigned
identity diagnostic, not natural-history identification or playing-strength
evidence. All models remained frozen; no main training changes.

November 14: [u195450 next-opponent prediction: reader structure x capacity](reports/history-next-prediction-2026-11-14.md)
completed 54 offline fits and 320 fresh test matches (957 prediction points).
Matched reader budgets of about 68k/134k parameters did not establish a benefit
from correct history. Original small/large and improved small select zero-history
residuals; improved large scores Brier 0.480992 versus current-only 0.479400,
with paired intervals crossing zero. The identity reference is also inconclusive.
All four fit training data but generalize poorly; this does not establish history
uselessness or rule out whole-model capacity limits. No main training changes.

November 14: [u195450 reader head, query and aggregation controls](reports/history-reader-mechanism-2026-11-14.md)
completed 78 isolated CPU runs and 320 fresh test matches. Removing diagnostic
head death restores normalized-reader gradients but does not fix full-history
retrieval; query variants score about 51–52%. Explicit public seat/round masks
raise attention retrieval to 67.48%. With matched parameters and only match-level
fact supervision, nonlinear-before-mean scores 82.07% versus 64.23% after-mean;
nonlinear-before-max reaches 94.74%, falling to 49.74% on mismatched history.
This supports addressing and aggregation bottlenecks for public fact retrieval,
not improved strategy or playing strength. No main training changes.

November 14: [u195450 public-history fact decoding and reader learning](reports/history-answer-learnability-2026-11-14.md)
reuses the filtered trajectories in isolated CPU diagnostics. Token-wise decoding
recovers previous-round facts from the final encoder at 95.31% balanced accuracy;
native reader/state probes remain near 50%. Frozen-encoder/query reader training
scores 50.05% for native and 51.67% for Q/K-normalized copies, without a reliable
paired gain. Normalization improves gradient availability but a diagnostic-head
optimization issue limits negative learning claims. No main training changes.

November 13: [u195450 / pinned current filtered-pool history diagnostics](reports/filtered-history-diagnosis-2026-11-13.md)
completed 1,280 four-round matches and 82,560 frozen score-round branches on
20 snapshots from u196664. Known identity improves same-deal behavior prediction,
but tested public-history probes do not recover that benefit; independent
decision-return gains remain unresolved and selection estimates are unstable.
A fixed reader vector changes 2/960 heldout greedy actions. One rare CPU
cache/recompute numerical discrepancy qualifies the earlier small-sample parity
result; native cached activations were used. No main training changes.

November 12: [u184140 / u195450 history-use root-cause audit](reports/history-root-cause-2026-11-12.md)
found saturated private Q/K selection and nearly constant history readout on
48 freshly collected match trajectories. Q/K gradients were zero on all six
audited minibatches; diagnostic normalization restored nonzero gradients.
Replacing the readout with one mean vector changed 4/1,953 cross-round greedy
actions. Current-weight CPU cache checks passed. This establishes sampled
mechanisms, not a trained remedy or improved history-based strength; main unchanged.

Documentation review: October 6, 2026. This is an index of recorded status,
not a live-job dashboard or authorization to run experiments. Verify current
code, checkpoint identity and external state for the task at hand.

October 25–26 Velocity operations: [node-local staging and GPU allocation](reports/velocity-nfs-staging-2026-10-25.md).
Jobs on `max-compute-11` run from a `/dev/shm` env built from the internet. A guard
fails a job that would import Python, torch or native libraries from NFS `/home`,
or that sees GPUs outside its SLURM allocation. League and next-coef arms moved
mid-run from two GPUs (own card plus an unscheduled shared card) to one card each;
compare segments across that switch only with annotation. Implemented and measured;
no strength claim.

October 17 synthesis: [complete history-utilization investigation report](reports/history-utilization-investigation-summary-2026-10-17.md)
connects the QK-norm audit, bounded short PPO, full-match and public-only prediction
studies, content swaps, fusion diagnostics, and training/fault evidence. It records
the u77500 resume correction and separates measured findings from unresolved
capacity/training hypotheses. Documentation only; no new experiment or adoption.

October 17 decision-value pilot: [bounded CPU result](reports/history-decision-value-2026-10-17.md)
completed 8 fresh own-hand information sets and 2,880 frozen score-round branches,
averaging compatible hidden deals before action selection and using independent
evaluation deals. Identity-informed Q selection gains −0.00521 PPO round return
(exploratory root-bootstrap 95% interval −0.14844 to +0.10417); the near-uniform
public-history posterior selects the same action as the prior in all 32 cases.
Native greedy FULL/ROUND choices also match, despite nonzero probability changes.
Six selection worlds per action leave substantial Q selection noise; this does
not establish that identity has no decision value. Prioritize a reproducible
positive-value task before PPO adaptation; no native/history4 or norm16 training.
With a fixed calibration mean, history4 is absorbable into out_proj and tests
learnability/optimization rather than increased policy expressivity. Main unchanged.

October 17 expanded Q check: [CPU stability and independent value replication](reports/history-decision-stability-2026-10-17.md)
completed 40,960 rounds on the same eight roots/candidate sets. Independent
64-world replicas agree on identity top actions in 46.88% of contexts and pairwise
rankings in 66.09%, below the descriptive stability gate despite reduced Q noise.
Choices pooled over 128 worlds were sealed before another 128 evaluation worlds:
identity gain +0.03467 (fixed-root paired-world 95% interval −0.02246 to +0.09106),
public posterior +0.01099 (−0.00244 to +0.02466; only one changed action).
Neither establishes positive decision value or stable optimal selection; these
are reused opening roots, not unseen-context confirmation. Source/checkpoint and
choice seals verified, 19 related tests pass. No PPO or main model changes.

October 17 larger fixed-root check: [512/512/1024-world CPU result](reports/history-decision-precision-2026-10-17.md)
completed 327,680 score-round branches with at most four single-threaded local
CPU workers. Identity top1/ranking agreement is 46.88%/78.75%; 84.38% of
contexts have both cross-choice gaps within 0.1 round return. Q noise decreased,
but strict ranking remains below the descriptive gate. After sealing pooled1024
choices, independent evaluation gives identity −0.00076 (fixed-root 95% interval
−0.01227 to +0.01102); public posterior never changes the prior action.
Retain the post-hoc recheck: previous pooled128 identity choices still gain
+0.02356 over their weaker old prior on the new evaluation worlds, but their
absolute return remains below the new type-blind prior. Type-matching controls
remain inconclusive. Thus no established identity benefit beyond the stronger
new prior, not a claim that identity is valueless. All raw returns/seals/source
identities verified, 23 related tests pass. No PPO, extra budget or main change.

October 17 new midgame task search: [opponent-sensitive task screening](reports/history-opponent-sensitive-tasks-2026-10-17.md)
completed 64 generated rounds and 144,000 local CPU score branches within the
preregistered 327,680 ceiling. Public-policy disagreement prefiltered 32 distinct
source-round nodes; crossfit screening selected six, then sealed pooled512-world
choices before independent512-world confirmation. Identity win gain is −0.6755pp
(fixed-root paired-world 97.5% interval [−1.2288, −0.1302]pp); type alignment
against the same action mixture is −0.0529pp with an interval crossing zero.
Identity gate failed, so the separate history evaluation was not run; the sealed
posterior also never changes the prior action. This controlled midgame hand-redraw
and type-assignment task is not natural-match conditional inference. No established
positive-value acceptance task, extra sampling, PPO, GPU or main-model change;
22 related tests and raw-array/source/checkpoint/choice-seal audits pass.

October 17 identity-loss diagnosis: [Q noise versus opponent response](reports/history-identity-diagnosis-2026-10-17.md)
completed a fixed 368,640 new local CPU branches on the same six roots: fresh1024
estimates, pooled2048 choices sealed before fresh2048 evaluation. The legacy512
identity policy still loses −0.3316pp on new worlds; several originally positive
Q gaps reverse. The refined identity policy improves absolute win probability by
+0.5778pp over legacy (descriptive paired 95% interval [+0.3621,+0.7894]pp).
Its gain over its stronger blind prior is +0.1648pp (paired-world 97.5% interval
[−0.0773,+0.4008]pp), while correct type matching versus the same marginal action
mixture gains +0.2991pp ([+0.1490,+0.4491]pp). Thus wrong finite-sample action
choices explain the loss; opponents are not wholly response-equivalent, but a
reliable advantage beyond the blind prior remains unestablished. The conservative
95% candidate-limited oracle upper bound is 3.627pp, so practical identity value
above 2pp is not ruled out. All chunks, reused-data/source/checkpoint hashes,
choice seals and summaries verified; 29 related tests pass. No budget extension,
PPO, historical-input experiment, GPU or main-model change.

October 17 planted-habit oracle value: [one-step value of knowing z](reports/history-habit-value-2026-10-17.md)
gave the true style of the Sept 30 habit pack (u2623 + z*b*f logit tilt) to a
rollout planner on 23 fresh nodes, 759,808 local CPU branches. For the phase-1a
pack (lead_single b=1) the upward-biased plug-in value is +0.02pp round win; even
pass b=4 is +0.27pp, with cross-fitted gains crossing zero for every style.
Opponent types do change relative action values (pass b=4 interaction RMS 3.25pp)
but rarely the best action. The 1a ORACLE null is therefore most simply explained
by near-zero one-step value; whole-round cumulative value is not measured.

October 17 planted-habit whole-round value: [late-round oracle vs type-blind search](reports/history-habit-round-2026-10-17.md)
searched every observer-team decision once some seat holds <= 10 cards, with paired
oracle-z and 50/50-blind planners on the same deals. Over 201 complete deals per
style (prefix 0-200; plan was 800, cut by the $2 Vast budget and local load),
pass b=4 gains +8.96pp round win [+1.99, +15.92], while the phase-1a pack
(lead_single b=1) gives +1.49pp [-3.48, +6.47]. Multi-step targeting is valuable
for a strong habit even though one-step value was +0.27pp; the 1a pack remains
inconclusive. Spent $1.76 on Vast CPU instance 56445523, destroyed. No training.

October 17 habit identifiability (stage 0a of the [habit attention plan](reports/habit-attention-ab-plan-2026-10-17.md)):
[ideal-observer posterior](reports/habit-identifiability-2026-10-17.md) over 64 matches per
style. pass b=4 is identified within the current round (ROUND posterior 0.948 in the
first half of rounds 2+), so FULL vs ROUND cannot test cross-round reading with it.
The preregistered rule selects pass b=1 (ROUND 0.691 / FULL 0.983) and bomb b=2
(0.747 / 0.975); their cross-round decision value (stage 0b) is unmeasured.
No training, local CPU only.

October 18 stage 0b: [cross-round decision value](reports/habit-crossround-value-2026-10-18.md),
700 matches per style (CPU 0-198, RTX 4090 199-699 with the batched engine, $0.50).
Late-round full-history minus current-round-posterior planner: pass b=1 -0.57pp
[-2.71, +1.57], bomb b=2 0.00pp [-1.43, +1.29]. Weak habits need memory but are
worth ~nothing; strong habits are worth targeting but visible within one round, so
stage 1 should not start on single-axis tilt opponents. The batched VecEnv /
shared-prefix rollout engine ([design](reports/batched-rollout-design-2026-10-18.md))
reproduces team_rollout bit for bit on CPU and CUDA; ~1,900 vs ~68 matches/hour.

October 18 Stage 1 design correction: [revised diagnostic plan](reports/habit-attention-ab-plan-2026-10-17.md)
splits a two-arm pass b=4 exploitation check from a conditional-habit screen, then
permits ORACLE/FULL/ROUND only after independent public-information and decision-value
gates. The proposed first habit caps strong endgame-response tilts at one opportunity
per opponent seat per round; it has not been implemented or measured. The historical
+8.96pp compares oracle with a fixed 50/50-blind planner, not ROUND, so equal learned
ORACLE/ROUND strength alone cannot diagnose PPO/fusion failure. Source inspection also
shows the old likelihood diagnostic conditions on the opponent's actual hand/legal set;
its posterior is privileged, not an implemented public-history estimator. Existing 0a/0b
numbers remain historical instrument results. Matched-context oracle input, conditional
rollouts and public posteriors remain implementation gates; no new training or compute.

October 18 fusion arm: [split private fusion and habit A/B plan](reports/fusion-split-habit-2026-10-18.md).
Opt-in `private_fusion=split` (q_proj reads `attention_norm(q)`; `state = LN(q) +
reader_norm(r)`), default `sum` bitwise unchanged; with `private_qk_norm=qk` it is
the tested arm. u2623 converts with a calibrated reader gain (c = 0.140; starting
policy TV mean 0.0073 vs 0.030 for a zeroed reader; no one-hot heads). The protocol
and Vast kit are sealed for FULL-legacy / FULL-split / ORACLE on pass b=4, 2 shared
seeds × 40M decisions each. Pass b=4 is visible within a round, so this tests reader
use, not cross-round memory. Not launched (credit $2.19 vs ~$21.6 worst case); no
training result; main lineage unchanged. October 19: moved to Velocity (1 L40S/run,
dated protocol amendment); gate 8040 + jobs 8041-8046 queued, main job 7973 held
until they finish. October 20: four FULL runs finished cleanly at 611 updates; ORACLE
runs training; CPU-only evaluation on Velocity (amendment 2026-10-20) and the automatic
7973 release are armed in an on-cluster coordinator; no evaluation result yet.

## Baseline and documented readiness

- Main route: randomly initialized public-history Transformer, self-play PPO,
  current/past Transformer population. Existing MLPs and DanLM are evaluation
  baselines. See [DESIGN.md](DESIGN.md) and [agent rules](../AGENTS.md).
- Sequence training, resume/population, history-aware evaluation and cache
  paths have implementation receipts. See [TRAINING.md](TRAINING.md) for
  operations and [TRAINING_EVIDENCE.md](TRAINING_EVIDENCE.md) for limits.
  T0--T8 in [STAGE_C_TODO.md](STAGE_C_TODO.md) are original milestone IDs.
- The [strength summary](reports/strength-summary-2026-10-04.md) includes the
  u20264 curve, October 5 search measurements and the October 6 m1/m2
  milestones (m2 u35588: -0.869 plain, -0.77 with search), and on October 8 m3
  u43276 (-0.665; search gain +0.13 on 800 deals) and m4 u50840 (-0.546), and on October 12 m5 u58528 (-0.497), m6 u66092
  (-0.425) and m7 u73780 (-0.334; -0.224 with search, gain +0.08 [-0.01, +0.17] on 800 deals). It is a dated summary, not
  a statement of today's latest checkpoint. The October 6
  [distillation plan](reports/search-distillation-plan-2026-10-06.md) also
  references m1/u27900; verify its artifacts before selecting that model.
- [Botzone search deployment](reports/botzone-search-2026-10-04.md) records a
  separate platform package and deadline-limited search. Its ladder sample
  and budget cannot be substituted for offline search evidence.
- October 18, 2026 [training throughput refactor](reports/training-throughput-refactor-2026-10-18.md):
  public-event batching, batch KV reservation and shared gradient reductions
  have local CPU component measurements and exact original-revision PPO/resume
  fingerprints. The private CUDA math-SDPA safeguard now covers every private
  route. Full Python/native/oracle suites passed; complete CPU update gains are
  small and variable. Target CUDA acceptance/throughput and production adoption
  remain unrecorded. See the report's committed evidence and limitations.

## Research directions and unresolved work

| Direction | Recorded state / next evidence needed |
|---|---|
| Continued training and auxiliary heads | [Experiment review](reports/experiment-review-2026-10-04.md), [auxiliary heads](reports/aux-heads-2026-10-02.md): distinguish lineage progress from multi-seed causal evidence |
| Evaluation reliability | The October 4 review identifies external-adapter exclusions and paired-record gaps. October 18 ([receipt](reports/paired-eval-checkpoint-average-2026-10-18.md)): per-deal arena records, `eval.paired` and `train.checkpoint_average` implemented and tested on synthetic checkpoints only; no real checkpoint or DanLM measured. The cause of the mini-ladder `their_choice_missing` exclusions is a static hypothesis (wild cards declared below the level rank), not yet checked against data |
| History use and opponent diversity | [History ablation](reports/history-ablation-2026-09-29.md), [diversity proposal](reports/opponent-diversity-plan-2026-10-03.md); earlier-checkpoint findings do not establish current history dependence |
| Planted-habit diagnostic | [Stage 1 revision, October 18](reports/habit-attention-ab-plan-2026-10-17.md): two-arm exploitation check, then sparse conditional-habit/public-information/value gates before three-arm training; design only. [September 30 phase 1a](reports/history-habit-phase1a-2026-09-30.md) remains historical, separate-lineage evidence |
| Cheaper search | [Fused search design](reports/fused-search-design-2026-10-05.md): explicitly design only, no implementation or measured gain claimed |
| Search distillation | [October 6 plan](reports/search-distillation-plan-2026-10-06.md): proposed; distinguishes offline Stage 1 from a Stage 2 design-boundary amendment |
| Learned tribute / back-tribute | Exchanges still use the fixed `tribute_bot` heuristic (DESIGN 1.3). [October 6 brief](reports/tribute-learning-brief-2026-10-06.md): open question for outside advice; paired DanLM data show different back-tribute choices but no measurable loss on tribute deals; the September 21 [A2 attempt](reports/M2-A2.md) was negative |
| Variance reduction / Q-boost | [VRPO design](reports/vrpo-design-2026-10-04.md), [experiment review](reports/experiment-review-2026-10-04.md): proposals/exploratory evidence, not demonstrated main-lineage gains |
| Runtime optimization | [PERF_TODO.md](PERF_TODO.md): implemented paths, measured scopes and remaining verification |

October 13 runtime evidence: [isolated stack A/B](reports/training-stack-ab-2026-10-13.md)
measured page-span complete-PPO throughput +10.28% on 2 L40S / 8 ranks,
with real-rollout parity passing. Production's 4 GPU layout remains unverified;
the main configuration is unchanged. Learner rollout projection failed its
numerical gate, so its performance measurement is diagnostic only.

October 13 startup incident: the next main slice stopped after repeated CUDA
initialization failures, including uncorrectable ECC errors with MPS off.
Current source identity passes. The bounded health check confirmed 3 of 4 allocated GPUs usable; page-span
complete-PPO throughput improved 8.50% on those 3 L40S / 8 ranks, with rollout
parity passing. Four healthy GPUs remain unverified and main training is not
restarted; its configuration remains unchanged. See the dated
[incident and validation record](reports/training-startup-ecc-2026-10-13.md)
for evidence and pending work; this is not a live queue status.

## Contract versus dated proposals

`DESIGN.md` is the September 25 baseline design; later test-time search and
multi-rank engineering have dated receipts. Its deferred-work list describes
that baseline, not a claim that those components are absent today. This
organization pass does not approve a new training signal or rewrite the
research contract. Use the distillation plan's explicit boundary discussion
for that proposed extension; do not infer approval from this index.

## Navigation and maintenance

[Documentation map](README.md) · [All reports](reports/README.md).
Update this page when recorded readiness or direction changes, linking the
owning source and its limits. Keep checkpoint and job claims dated; do not
copy a historical budget or an old session's authorization into a new launch.

October 13 follow-up [projection experiment](reports/training-stack-next-2026-10-13.md):
packed valid-token projection passed CUDA parity, but PPO stopped at update index 97
on a non-finite gradient norm. The completed subset showed no speed gain; this
candidate is not ready for adoption. Main configuration remains unchanged.

October 14 gradient diagnosis: [7220 results](reports/training-gradient-diagnosis-2026-10-13.md)
confirmed finite actor gradients overflowing the FP32 norm on the page-span base arm
(3 healthy GPUs, 8 ranks, 94 completed updates). Rank 1 already had a huge local
backward gradient; outcome-loss replay showed the largest component gradient.
The operator-level cause remains unresolved; this is not a successful repair or
throughput result. Main source/manifest unchanged; no new experiment submitted.

October 14 follow-up: [rank 1 fixture replay](reports/training-fixture-backward-2026-10-14.md)
reproduced the outcome loss on CPU with normal gradients, but device/version differ.
Isolated job 7323 compares CUDA default/math SDPA and repeatability before any repair
or cache/reduction benchmark. Main configuration remains unchanged.

October 14 update: [7323/7324 fixture evidence](reports/training-fixture-backward-2026-10-14.md)
localized erroneous gradient amplification to masked efficient SDPA backward on
PyTorch 2.11.0+cu128 / L40S. A grad-enabled private-attention math fallback passed
fixture finite-gradient and 1e-4 candidate/chosen-logp gates. Job 7325 now tests
120 full PPO updates on the isolated repaired baseline before cache/reduction
optimization; production remains unchanged and no new speedup is claimed.

October 14 stability result (supersedes the pending 7325 status above):
[7325](reports/training-fixture-backward-2026-10-14.md) completed 120 full PPO
updates on every one of 8 ranks / 3 healthy L40S; gradient metrics were finite
and per-update parameter checksums passed. This is bounded stability evidence.
[7366 cache/reduction comparison](reports/training-cache-reduce-2026-10-14.md)
is submitted and pending resources. It gates real-rollout/cache parity and
reduction equivalence before two separate full-PPO A/B tests against the repaired
page-span baseline. No new speedup measured; production remains unchanged.

October 14 final 7366 result (supersedes pending status above):
[cache/reduction evidence](reports/training-cache-reduce-2026-10-14.md) passed
rollout and reduction parity; both stages completed 8 ranks × 120 updates on
3 healthy L40S. Against repaired page-span baselines, cache throughput +1.23%
(paired block 95% interval +0.18% to +2.19%); pinned host reduction −1.02%
(interval −2.68% to +0.68%), no demonstrated gain. 114 files verified and local
analysis matches remote. No combined test or main-lineage adoption; production
source and manifest unchanged. This round is closed, with no new job submitted.

October 14 user-directed follow-up: retain cache optimization only in the
isolated baseline; do not adopt pinned host reduction. [7392 encoding/wait
diagnosis](reports/training-encode-wait-2026-10-14.md) is submitted to the existing
cluster, with passive arrival traces and a separate synchronized diagnostic.
No new speedup claim and no main-lineage changes.

October 14 evening: [7392 diagnosis](reports/training-encode-wait-2026-10-14.md)
completed on 3 healthy L40S; all 8 ranks finished passive 40 / sync 32 updates,
119 files SHA-verified. Synchronized rollout cache/collation averages 0.997s,
learner public forward 0.250s with 22.41% token-slot padding. Passive reduction
arrival spread averages 16.64ms (p95 48.57ms); this does not establish a causal
rank/GPU imbalance or a speedup. Next target is incremental cache internals;
keep cache optimization, reject pinned reduction. Main identity/manifest unchanged;
no new job, diagnostic monitor paused on closeout.

October 14 follow-up: [7452 cache internals/reuse](reports/training-cache-reuse-2026-10-14.md)
is submitted under user authorization. It profiles append/gather/metadata and
then gates exact append-row-index reuse within one encode call before full PPO
ABBA. Local CPU coverage passed; CUDA parity and speed remain unmeasured.
Main unchanged; rejected pinned reduction stays disabled.

October 14 final [7452 result](reports/training-cache-reuse-2026-10-14.md):
CUDA parity passed; all 8 ranks completed 120 PPO updates on 3 healthy L40S.
Index reuse throughput −0.435% (paired 95% interval −1.301% to +0.369%), no
proven gain despite 50.79% candidate gather hits. Do not adopt this candidate;
retain previous index_select. Profile points to append (ordinary 0.626s,
merged 0.408s inclusive) over final memory gather (0.086s), with nested timing
and synchronization caveats. 113 files verified; local analysis matches remote;
main identity/manifest unchanged. No new job; monitor paused on closeout.

October 14 user-directed [append detail diagnosis](reports/training-append-detail-2026-10-14.md):
7457 submitted; retain index_select, reject row-index reuse and pinned reduction.
Event and synchronized probes split event preparation/KV/attention/FFN. No new
candidate yet. Stop if no clear actionable bottleneck; do not keep profiling.
Main unchanged, existing cluster only.

October 15 closeout: [7457 append detail](reports/training-append-detail-2026-10-14.md)
completed both 8×32 stages on 3 healthy L40S; 103 files verified. Attention is
the largest measured leaf (~0.367s/rank/update in event and sync diagnostics),
but no clear removable/reusable work supports a new optimization. Per user stop
condition, stop exploration: no new candidate/job/profile. Keep prior isolated
index_select only; main identity/manifest unchanged. Monitor paused on closeout.

October 15 main-lineage incident and restart: [gradient blow-up report](reports/main-gradient-blowup-2026-10-15.md).
The main run's actor gradient norm exploded during u74484–77584 (masked efficient
SDPA backward on L40S), degrading the auxiliary heads; it caused the u77585 stop.
Main training resumed October 15 (job 7464) with the math-SDPA fix, index_select
cache gathers, page-span and a per-GPU health probe that skips the ECC-faulty card;
source identity rebuilt with `--allow-source-change`. Measured vs DanLM (4,000 deals):
m8 u81344 −0.376 [−0.414, −0.340], u86800 −0.312 [−0.350, −0.277] (m7 −0.334);
the model recovered, no rollback, and u86800 versus m7 is within error. This
supersedes the "production unchanged / not restarted" statements above for the main run.

October 16 bounded diagnostic: [QK-norm three-arm 500M mechanism audit](reports/qknorm-mechanism-2026-10-16.md) verified all arms at u94488 / +503,840,768 decisions with matching checkpoint, evaluation and runtime-source hashes. CPU math probes on the original fixture plus 576 fresh frozen-policy decisions show qk-both avoids extreme one-hot saturation and restores Q/K outcome gradients, while queries remain highly aligned and replacing the reader output with an independently calibrated mean changes policy little. Existing internal strength comparisons remain inconclusive; cross-round adaptation is unmeasured. Next proposed work is an offline reader-plasticity candidate check, not an adopted repair. No new training submitted or existing queue state changed by this audit.

October 16 offline follow-up: [Q/K rescaling and Adam plasticity audit](reports/qknorm-plasticity-2026-10-16.md) passed CPU forward gates for per-head joint weight/bias target norms 16, 8, 4 and 2.309401 on the existing 904 decision rows across three private-attention paths. Intrinsic direction gradients were preserved; disposable one-step outcome-loss counterfactuals favor the conservative norm-16 candidate with gradient-coordinate moment transport (m/a, v/a²), unchanged shared step/lr/eps, over resets. This changes optimizer dynamics and does not establish learning or strength gains; CUDA/full-loss/population migration remain unverified. Six numerical tests passed. No training or migrated checkpoint was created.

October 16 CUDA gate follow-up: [norm-16 complete actor-loss validation](reports/qknorm-cuda-gate-2026-10-16.md) passes CPU prechecks including PPO/entropy/response/next/belief/outcome, native and active clipping, disposable Adam steps, and all 13 stored snapshots. Isolated CUDA job 7936 failed at the GPU health check with uncorrectable ECC before model execution. Health-selecting retry 7937 is pending resources (scheduler estimate 20:29 America/Toronto); CUDA remains unverified. A bounded two-hour collector will preserve outputs and append the actual result. No training or production checkpoint/source mutation.

October 16 CUDA diagnostic completion: [norm-16 full-actor-loss gate](reports/qknorm-cuda-gate-2026-10-16.md) passed on one health-selected L40S in isolated job 7937: forward, nine complete actor-loss/gradient/clipping cases, disposable Adam state checks, all 13 stored snapshots and private CUDA graph capture/replay. Original checkpoint unchanged. No training, migrated checkpoint or main-lineage adoption; DDP/full PPO/strength remain unverified.

October 16 user-authorized next step: [isolated norm-16 short PPO A/B](reports/qknorm-short-ppo-2026-10-16.md) compares the same qk-both u94488 parent against learner-only norm-16/moment transport for at most 120 full updates per arm, one rank on the same health-selected L40S, same initial frozen population/RNG. Fixed-sample history-output sensitivity is checked at 0/30/60/120; no strength claim or production adoption. Main source/jobs/checkpoints unchanged; no long-run or resubmission configured. Results pending.

October 16 short PPO A/B completion: [isolated norm-16 continuation](reports/qknorm-short-ppo-2026-10-16.md), job 7939, completed 120 full updates / 983,040 decisions per arm on one L40S with matched initial pool/RNG/config. Both endpoints and Adam states are finite; all 7,076 actor clipping observations passed (maximum norm 8.752 control / 5.650 norm16, no clipping). Norm16 increased median relative net Q/K direction updates about 7.18×/4.76×, but held-out history-mean-replacement TV changed 1.178%→1.161% versus control 1.178%→1.295%, with queries still highly aligned. No demonstrated history-use or strength gain; single seed/rank only. Keep isolated, do not adopt or extend to long training from these results. Main source/checkpoint/jobs unchanged; no successor.

October 16 user-authorized evaluation: [frozen endpoint FULL/ROUND match ablation](reports/qknorm-match-ablation-2026-10-16.md) runs 512 complete matches per condition against four common frozen parent snapshots, with explicit match seeds, team swaps and paired seed-cluster uncertainty. Both teammates receive the selected view; opponents retain FULL. CPU execution only; cluster evaluation 7966 was canceled before starting because of node reservations. No new training or production changes; results pending.

October 16 match-ablation completion: [FULL/ROUND endpoint evaluation](reports/qknorm-match-ablation-2026-10-16.md) finished all 2,048 matches / 23,196 rounds on local CPU, no exclusions. All 1,024 within-arm first-round action hashes match; identical-policy team-swap sanity gives complementary outcomes and exactly 4/8 wins. FULL-minus-ROUND win-rate gain is control +3.32 pp (paired 95% CI −1.56 to +8.01), norm16 +1.37 pp (−3.52 to +6.25); multiplicity-adjusted intervals also cross zero. Norm16's history-gain interaction is −1.95 pp (−8.98 to +5.27). No established cross-round benefit, equivalence, or norm16 improvement; four closely spaced parent snapshots and one training seed limit generalization. Fixed budget closed; no new training, production adoption or successor.

October 16 public-history prediction check: [public-only predictability report](reports/public-history-predictability-2026-10-16.md) exactly replayed 1,024 prior FULL matches to record public events. Count-table calibration used 512 matches; 512 held-out matches / 128 seed clusters provided 178,737 fourth-and-later-round opponent actions. Adding the first three rounds did not improve categorical Brier score (gain −1.09e−6, 95% CI −3.95e−6 to +1.85e−6). A 256-question identical-deal first-lead control also showed no gain (−1.79e−7, CI −2.90e−6 to +2.40e−6); this controls actual cards at openings only. Identity guessing was 26.95% versus 25% chance (CI 23.24–30.86%). An explicitly exploratory direct residual carryover worsened prediction. These limited statistical probes do not prove history contains no information or isolate a Transformer learning failure; real-game contexts remain public summaries and the four snapshots are closely related. Ten instrumentation tests passed, no private probe fields or forced flags, no neural training/production changes, no successor.

October 16 diverse frozen-opponent replication: [public-history diverse-pool check](reports/public-history-diverse-2026-10-16.md) selected u2623/u20264/u35588/u86800 from eight existing Transformers using 64 independent public-only opening questions. Separate formal questions confirm categorical disagreement 54.82% versus the close pool's 6.40%. All 1,024 full matches and 256 identical-deal questions completed; unchanged count-table analysis on 512 held-out matches finds a small additional-history Brier gain +0.00010981 (95% CI +0.00008467 to +0.00013534; relative error reduction 0.024%). Identical-deal opening gain +0.00212414 (+0.00157719 to +0.00272376), identity accuracy 41.02% (36.91–45.12%, chance 25%). Correct history beats shuffled history; the fixed direct residual rule still worsens prediction. This establishes limited public-history predictability in a more diverse known roster, not current-actor history use or playing-strength benefit. No private probe inputs, neural training, production adoption or automatic successor; eleven instrument tests passed.

October 16 diverse-pool FULL/ROUND match ablation: [frozen endpoint result](reports/qknorm-diverse-match-2026-10-16.md) completed 2,048 matches / 20,593 rounds, 256 paired seed clusters. All 1,024 FULL trajectories reproduce the prior prediction-study matches, all within-arm first-round hashes match, and all 512 paired full-match public paths per endpoint differ under FULL versus ROUND. Control wins 73.05% FULL versus 69.14% ROUND (+3.91 pp; single 95% CI +0.20 to +7.62, but prespecified two-comparison 97.5% CI −0.20 to +8.01). Norm16 wins 71.48% versus 72.66% (−1.17 pp; 95% CI −5.27 to +2.93). Thus context affects behavior and control has a weak positive strength signal, but the prespecified joint positive-benefit gate is unmet; norm16 shows no enhanced history benefit. Fourteen instrument tests passed; no hidden-count reads, neural training, main-lineage change or automatic successor.

October 16 same-state content follow-up: [equal-length public-history replacement](reports/qknorm-history-swap-2026-10-16.md) completed 707 frozen decisions from 256 exactly replayed FULL matches (64 seed clusters per endpoint). Old-history length, current-round events/positions, own observation and candidates stayed fixed. Replacing prior public tokens while also retaining original round/phase metadata gives mean TV 0.401% control / 0.446% norm16, with only 2/357 and 2/350 argmax changes. Original duplicates are exact; fresh-original vs cached logp max errors <3.82e-5. Thus there is weak content sensitivity, not merely a FULL/ROUND length effect, but no established useful opponent adaptation, architecture repair or win gain. Donor length protocol was expanded after explicit failures without dropping target rows; synthetic splicing and opening-of-round sampling limit interpretation. Fourteen tests passed; no training, mainline behavior change or successor.

October 16 frozen stage diagnosis: [encoding/reader/fusion readouts](reports/qknorm-history-stages-2026-10-16.md) exactly replayed 1,024 prior FULL matches and captured each first selectable fourth-round learner decision. Actor weights stayed frozen; only closed-form ridge statistics were calibrated on256 matches and tested on256 per endpoint, with independent identical-card public answers. Old opponent-token memory predicts the known four-opponent roster at48.83%/45.70% (chance25%) and improves same-card Brier; actual FULL readers also outperform ROUND readers (+0.009649/+0.008167 Brier gain). Final states show no established extra gain (−0.000116/−0.000131, intervals cross0). An explicitly post-hoc frozen reconstruction finds the linear signal becomes less readable after query addition and attention_norm. This favors a downstream fusion/use bottleneck over complete encoder failure, but does not prove information destruction, a LayerNorm bug, optimal-action dependence or win benefit. Eighteen tests pass; no neural training, production behavior change or successor.

October 16 offline fusion isolation: [query/LayerNorm factorial and same-state geometry](reports/qknorm-fusion-isolation-2026-10-16.md) reused1,024 saved stage contexts and707 equal-length history swaps, no new matches. Removing query variation restores reader-level public-question predictability; with actual query, row centering reduces the remaining linear-readout gain, whereas additional row scaling/learned affine shows no established major effect. Keeping native LayerNorm with fixed calibration-mean query retains useful signal, especially under the prespecified readout-scaling sensitivity check. Direct content-swap geometry preserves about92%/91% of squared perturbation energy after accounting for units; this is not a percentage of useful information and argues against wholesale history erasure. Evidence favors interaction between query masking and fusion/centering, not a proven LayerNorm implementation bug or adopted repair. Twelve tests passed, checkpoint/source/native identities unchanged; no neural training, mainline changes or successor.

October 17 frozen fusion alternatives: [same-width seven-condition comparison](reports/qknorm-fusion-variants-2026-10-17.md) reused1,024 saved state contexts and707 equal-length content swaps. All original states/decisions reproduce. History-content gain4 increases average policy TV under history swaps from0.401%/0.446% to1.576%/1.802%, with8/357 and9/350 argmax changes versus2 each; original-history policy drift stays0.662%/0.828% on average. Postnorm residual gain4 behaves similarly. Mean restoration and postnorm gain1 give small changes; halving query variation causes about50% policy TV drift and is unsuitable as a low-disruption diagnostic candidate. No condition shows replicated positive FULL−ROUND public-question readout gain in final states, so higher sensitivity is not proven better history use or strength. All weights/widths remain frozen,15 tests pass, no training/mainline change/adoption/successor.

October17 backward-only history/training audit: [training exposure,signal,and11-checkpoint timeline](reports/history-training-audit-2026-10-17.md) confirms full raw-history recomputation and nonzero PPO gradients into prior-round tokens in a saved328-row isolated-PPO minibatch (296 rows contain definitely older rounds); no parameter steps. Current endpoint checks are explicitly reanchored connectivity checks,not real on-policy updates. Recent4 snapshots every2 updates,no archive,and round-terminal GAE limit diverse-opponent practice/direct cross-round exploration credit without disconnecting history features. User-authorized read-only velocity downloads of u75516/u77500/u77748 pass SHA checks. Same-input public next-type accuracy drops74.20%→52.74% from u74028→u77500 and recovers74.09% at u86800; encoded usefulness also drops,but weak final-state history use predates the fault. New remote logs correct the resume boundary: file order goes77584→77501,with source-change anchor77500; duplicate update IDs must be split by run order. Original incident report has an explicit dated correction. Four no-step tests pass,no training/mainline behavior/job changes or successor.

October 18: [private-attention audit tooling](reports/private-attention-audit-2026-10-18.md)
implemented (`eval/private_attention_audit.py`, CPU, multi-minibatch) together with
an opt-in `private_qk_norm` architecture arm (default `none`, main lineage
unchanged, fresh lineages only). Measured only on random-init and synthetic
actors: m7/u86800/latest were not on that VM, so whether their saturation is
global remains unresolved. Next: run the audit where the checkpoints are; any
`qk` A/B lineage needs separate authorization.
