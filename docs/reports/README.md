# Experiment report index

Complete Markdown report catalog, reviewed October 6, 2026. Start with
[project orientation](../STATUS.md) and the [documentation map](../README.md).
Groups are navigation aids; status, settings and limitations live in each
report. Titles below are preserved from the reports, not newly validated claims.
A plan/design is not an implementation receipt or permission to run it.
Dated evidence does not prove current code or live platform state.

For a task, read its latest relevant summary and follow its source reports.
Do not read every report on startup. `.work/` evidence is local and ignored;
check availability before claiming you independently verified artifacts.
Historical records stay at their existing paths to preserve references.

## Current summaries, deployment and research proposals

- [主线长跑指标复核（2026-10-06）](training-review-2026-10-06.md)
- [主线 Transformer 打法演变（2026-10-06）](style-evolution-2026-10-06.md)
- [对 DanLM 的对局复盘（2026-10-06）](danlm-game-review-2026-10-06.md)
- [进贡/还贡怎么学：问题说明和配对数据（2026-10-06）](tribute-learning-brief-2026-10-06.md)
- [Auxiliary heads on the history encoder (October 2, 2026)](aux-heads-2026-10-02.md)
- [Botzone bot, steps 1 to 4 (October 3, 2026)](botzone-bot-2026-10-03.md)
- [Botzone search deployment — October 4, 2026](botzone-search-2026-10-04.md)
- [近期实验复核：方向、证据缺口与优先级（2026-10-04）](experiment-review-2026-10-04.md)
- [Fused test-time search: design (October 5, 2026)](fused-search-design-2026-10-05.md)
- [Learning-rate decay and weight averaging (October 3-4, 2026)](lr-decay-weight-avg-2026-10-04.md)
- [Opponent diversity: plan (October 3, 2026)](opponent-diversity-plan-2026-10-03.md)
- [Search distillation: plan (October 6, 2026)](search-distillation-plan-2026-10-06.md)
- [Strength against DanLM, summary (October 4, 2026, updated with u20264, the October 5 search curve and m1)](strength-summary-2026-10-04.md)
- [VRPO / Q-boosting for the history trainer: design, October 4, 2026](vrpo-design-2026-10-04.md)

## History, opponent response and diagnostics

- [离线行为探针：管线检查与小规模诊断（2026-09-25）](behaviour-probe-2026-09-25.md)
- [主线终点是否在用历史流：输入消融（2026-09-29）](history-ablation-2026-09-29.md)
- [Batch-size recipe test: bounded RTX 4090 run, September 26, 2026 UTC](history-batch-2026-09-26.md)
- [固定预算探索筛查与 B 谱系续训：源码核对与冻结方案](history-budget-protocol-2026-09-27.md)
- [固定预算探索筛查与 B 谱系续训：结果](history-budget-results-2026-09-27.md)
- [容量 × 对手池 2×2（熵系数 0.03 配方）](history-factorial-2026-09-27.md)
- [Cross-round habit diagnostic: plan](history-habit-diagnostic-plan-2026-09-29.md)
- [植入习惯诊断，第 0 阶段：校准（2026-09-30）](history-habit-phase0-2026-09-30.md)
- [植入习惯诊断，第 1a 阶段：ORACLE 与 ROUND（2026-09-30）](history-habit-phase1a-2026-09-30.md)
- [主线终点怎么打牌：u2623 对 B11 与 u0844 的打法统计（2026-09-29）](history-play-style-2026-09-29.md)
- [T7: shared representation versus explicit response input](history-response-plan-2026-09-26.md)
- [T7 results: shared versus explicit opponent-response connection](history-response-results-2026-09-27.md)
- [T4: completed bounded cold-start Transformer pilot](history-t4-pilot-2026-09-26.md)
- [T4 remote pilot: prelaunch contract](history-t4-readiness-2026-09-26.md)
- [Transformer 策略臂：第一步盘点（2026-09-25）](transformer-inventory-2026-09-25-zh.md)
- [Transformer 自博弈实现盘点（2026-09-25 更新）](transformer-inventory-2026-09-25.md)

## Training recipes and continuation

- [GuanZero：Codex 与 Fable 的训练方案讨论](codex-fable-training-discussion-2026-09-27.md)
- [Exploration floor for learner seats (2026-09-26)](exploration-floor-2026-09-26.md)
- [Exploration roadmap after the T7 entropy diagnosis (September 26, 2026)](exploration-roadmap-2026-09-26.md)
- [交接：2026-09-27 晚](handoff-2026-09-27.md)
- [主线续训 8 小时（快照合批 + 显存修剪）与 256 副牌评估（2026-09-29）](history-longrun-batched-2026-09-29.md)
- [大模型通宵训练：1,024 桌 × 7.3 小时（2026-09-28）](history-overnight-large-2026-09-28.md)
- [History Transformer recipe campaign: summary, September 26, 2026 UTC](history-recipe-summary-2026-09-26.md)
- [Recipe run 1: five one-variable arms on CPU, September 26, 2026 UTC](history-recipe1-2026-09-26.md)
- [Recipe run 2: data-parallel collection versus two in-pod controls, September 26, 2026 UTC](history-recipe2-2026-09-26.md)
- [Recipe run 3: replicating two PPO epochs and testing optimizer-step variants, September 26, 2026 UTC](history-recipe3-2026-09-26.md)
- [Recipe run 4: two epochs as the reference, one addition per arm, September 26, 2026 UTC](history-recipe4-2026-09-26.md)
- [Recipe run 5: continuing the best lineages across pods, September 26, 2026 UTC](history-recipe5-2026-09-26.md)
- [Kaggle GuanZero GPU 测速 — 2026-09-24](kaggle-benchmark-2026-09-24.md)
- [Kaggle 免费 GPU 测速：已完成](kaggle-benchmark-plan-2026-09-24.md)
- [GuanZero Kaggle 免费 GPU campaign 总报告 — 2026-09-26](kaggle-campaign-results-2026-09-26.md)
- [Kaggle 约 30 小时续训计划 — 2026-09-24](kaggle-continuation-plan-2026-09-24.md)
- [Kaggle R1 续训结果 — 2026-09-25](kaggle-r1-results-2026-09-25.md)
- [Kaggle R2 复验与 R3 决策 — 2026-09-25](kaggle-r2-results-2026-09-25.md)
- [Kaggle R3 内部最终验证 — 2026-09-26](kaggle-r3-results-2026-09-26.md)
- [长跑第一段：两个学习率同卡对照（2026-09-25）](longrun-segment1-2026-09-25.md)
- [长跑第二段：续跑、EMA 与候选扩展（2026-09-25）](longrun-segment2-2026-09-25.md)
- [B11 main continuation: internal evaluation, September 24, 2026](main-continuation-internal-2026-09-24.md)
- [五小时模型叠加残局搜索：限时独立复验](main-search-confirmation-2026-09-24.md)
- [Overnight training and parallel development plan — September 24, 2026](overnight-plan-2026-09-24.md)
- [GuanZero 夜间实验结果 — 2026-09-24](overnight-results-2026-09-24.md)
- [T7 之后的训练安排：Codex 与 Fable 讨论结论](training-next-steps-2026-09-27.md)

## Performance and compute

- [算力平台补充调研：CPU 隔离（2026-09-24）](compute-cpu-isolation-2026-09-24.md)
- [算力选项调研（2026-09-22）](compute-options.md)
- [Stage C 算力平台筛选：速度优先](compute-stage-c-2026-09-26.md)
- [续训大模型，同时测 actor 进程与快照频率（2026-09-28）](history-actor-ranks-2026-09-28.md)
- [History rollout：RTX 4090 吞吐优化（2026-09-28 UTC）](history-cuda-throughput-2026-09-28.md)
- [T7 execution placement: CPU engine with CUDA model execution](history-device-placement-2026-09-26.md)
- [本地历史策略采样优化：第一轮](history-host-throughput-2026-09-27.md)
- [本地吞吐优化：第二、三轮与 CUDA 接续](history-host-throughput-round3-2026-09-27.md)
- [Learner CUDA benchmark — September 30, 2026](history-learner-cuda-2026-09-30.md)
- [采集提速：快照合并推理与显存回收（2026-09-28）](history-snapshot-batching-2026-09-28.md)
- [History stack optimization: local preparation and verification](history-stack-2026-09-26.md)
- [History stack: bounded RTX 4090 comparison, September 26, 2026 UTC](history-stack-cuda-2026-09-26.md)
- [Training and evaluation performance follow-up](perf-followup-2026-09-24.md)
- [RTX 4090 training and evaluation performance validation](perf-gpu-2026-09-24.md)
- [RTX 4090 profile: where a PPO step goes, and lazy opponent features](perf-gpu-2026-09-24b.md)
- [Learner staging and reference skip: two exact PPO speedups](perf-learner-2026-09-24.md)
- [RTX 4090: in-process rollout pipeline, actors, two arms per GPU](perf-pipeline-2026-09-25.md)
- [Performance scan of training and evaluation, and the first fixes](perf-scan-2026-09-23.md)
- [Training speed deep dive — October 1, 2026](speed-deep-dive-2026-10-01.md)
- [Training stack optimization — October 1, 2026](training-stack-2026-10-01.md)
- [History trainer refactor and speed options — September 29, 2026](training-stack-refactor-2026-09-29.md)

## Historical engine, MLP milestones and integration

- [M0 report: the rules engine](M0.md)
- [M1 pre-training implementation report](M1-preflight.md)
- [M1 RunPod validation and pilot](M1-runpod.md)
- [Stage A2: learned tribute feasibility](M2-A2.md)
- [M2-belief-corrected](M2-belief-corrected.md)
- [Scaled styled-opponent belief probe](M2-belief-scaled.md)
- [M2 Transformer belief gate](M2-belief.md)
- [M2-memory](M2-memory.md)
- [修正后的猜牌与跨回合记忆实验](M2-next-run.md)
- [Stage C candidate-union and advantage-filter pilot — September 24, 2026](candidate-union-pilot-2026-09-24.md)
- [优势过滤实验诊断（2026-09-24）](filter-diagnosis-2026-09-24.md)
- [GuanZero 模型路线研究：Transformer、推断与搜索](model-research-2026-09-23.md)
- [规则网络核对报告（RULES.md vs 公开权威来源）](rules-web-check.md)
- [Stage B B2: perfect-information critic and gate G1](stage-b-critic.md)
- [Stage B B9: exploiter test (gate G4)](stage-b-exploiter.md)
- [Stage B B8: league training versus a single frozen opponent](stage-b-league.md)
- [Stage B B11: a live exploiter in the league (gate G5)](stage-b-live-exploiter.md)
- [Stage B B6: PPO versus continued DMC at equal compute](stage-b-ppo.md)
- [Stage B PPO throughput (B5b)](stage-b-throughput.md)
- [Stage C C0: DanLM external baseline](stage-c-danlm.md)
- [Stage C C1(a): candidate pruning audit](stage-c-pruning-audit.md)
- [Residual-shuffle search: scoring audit and DanLM transfer probe](stage-c-search-transfer-2026-09-24.md)
- [C3 residual-shuffle endgame search v0 — bounded CPU probe](stage-c-search-v0.md)

## Maintaining this index

Add each new report once under the appropriate topic. Keep the date in new
filenames, label proposals and incomplete runs in the report, and update
`../STATUS.md` only when the recorded direction/readiness changes. Preserve
raw result/provenance pointers and negative findings.
