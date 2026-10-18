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

- [Opt-in FP32 paged rollout-attention prototype (October 18, 2026)](paged-attention-prototype-2026-10-18.md) — implementation and local CPU checks; CUDA compilation, integration and speed unverified
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
- [Paired per-deal evaluation, exclusion audit and checkpoint averaging (October 18, 2026)](paired-eval-checkpoint-average-2026-10-18.md)
- [Paired DanLM rerun: u86800, m7 and a four-checkpoint average (October 18, 2026)](paired-eval-mac-2026-10-18.md)
- [Search distillation: plan (October 6, 2026)](search-distillation-plan-2026-10-06.md)
- [Strength against DanLM, summary (October 4, 2026, updated with u20264, the October 5 search curve and m1)](strength-summary-2026-10-04.md)
- [VRPO / Q-boosting for the history trainer: design, October 4, 2026](vrpo-design-2026-10-04.md)

## History, opponent response and diagnostics

- [第 0b 阶段：跨局历史的决策价值（2026-10-18）](habit-crossround-value-2026-10-18.md)：每种风格 700 场（CPU 0–198 + RTX 4090 199–699，$0.50）。残局里"看整场"减"只看本局"：pass b=1 为 −0.57pp [−2.71, +1.57]，bomb b=2 为 0.00pp [−1.43, +1.29]。弱风格的跨局价值约为零；单轴偏置对手里找不到既需要跨局记忆、又值得针对的设定。
- [设计：C++ 批量环境 + GPU 批量推理的搜索 rollout（2026-10-18）](batched-rollout-design-2026-10-18.md)：VecEnv 增加 load / stop_at_round_end / 数组形式的事件导出，共享前缀 KV、单次前向；参考模式下与旧 rollout 逐位一致（CPU 和 CUDA），4090 上第 0b 阶段约 1,900 场/小时，本地 CPU 约 68 场/小时。
- [第 0a 阶段：理想观察者多快能认出风格（2026-10-17）](habit-identifiability-2026-10-17.md)：64 场 × 5 种风格。pass b=4 在本局前半就被认出（ROUND 0.948），不适合测跨局记忆；按预定规则选出 pass b=1（ROUND 0.691 / FULL 0.983）和 bomb b=2（0.747 / 0.975），但它们的整局价值尚未测量。
- [计划：风格标签 vs 公开历史（2026-10-17）](habit-attention-ab-plan-2026-10-17.md)：**仅计划**。ORACLE / FULL / ROUND 三臂从 u2623 续训，对手为强 pass 习惯包，用于区分"训练没学会"和"attention 读不出"；第 0 阶段先用理想观察者确认风格不会在本局内就暴露，再选偏置强度。
- [植入习惯：残局整段针对的价值（2026-10-17）](history-habit-round-2026-10-17.md)：有人剩 ≤10 张后我方每步搜索，"知道风格"对比"不知道风格"，同局配对。pass b=4 为 +8.96pp [+1.99, +15.92]（201 局），说明价值靠多步累积；1a 对手 lead_single b=1 为 +1.49pp [−3.48, +6.47]，分不开。Vast CPU 花费 $1.76，样本少于计划。
- [植入习惯的"作弊版"决策价值（2026-10-17）](history-habit-value-2026-10-17.md)：直接告诉规划器 9 月 30 日对手包的风格 z，在 23 处局面上算一步换招的价值；1a 对手（lead_single b=1）上界仅 +0.02pp，pass b=4 也只有 +0.27pp；方法能测到对手差别（pass b=4 交互 RMS 3.25pp），但最优动作很少改变。1a 的 ORACLE 没赢最可能是因为可赚的东西接近零；整局累积价值未测。
- [身份方案为何变差：Q 噪声与对手应对差异诊断（2026-10-17）](history-identity-diagnosis-2026-10-17.md)：固定新增 368,640 个 CPU 分支；旧身份方案在新配牌上仍 −0.332pp，更多估值后绝对政策提升 +0.578pp。新方案匹配收益 +0.299pp 已有正向证据，总身份增益 +0.165pp 仍跨零；错误换招是负值直接原因，不能说对手完全无需区别，未启动 PPO。
- [更需要针对对手调整打法的中后盘筛查（2026-10-17）](history-opponent-sensitive-tasks-2026-10-17.md)：64 个生成局中选 32 处初筛，锁定 6 处后完成 144,000 个 CPU 评分分支；独立身份胜率增量 −0.68 个百分点，类型匹配对照也未通过。身份门槛失败，停止独立历史阶段；不追加预算、不启动 PPO。
- [千副配牌下的 Q 稳定性与局部决策价值（2026-10-17）](history-decision-precision-2026-10-17.md)：327,680 局本地 CPU 完成；每组 512 副的身份首选一致率 46.88%、排序 78.75%，84.38% 条件的交叉差距不超过 0.1；合并 1,024 副封存后另以 1,024 副评价，身份 −0.00076、公开 posterior 不换动作。保留旧身份方案对弱旧先验的 +0.02356 独立配牌结果；未显示超过更强新先验的身份收益，不启动 PPO。
- [扩大配牌样本后的 Q 稳定性与决策价值（2026-10-17）](history-decision-stability-2026-10-17.md)：40,960 局本地 CPU 完成；两组各 64 副的身份首选一致率 46.88%、排序一致率 66.09%，未稳定；合并 128 副封存动作后独立评价 128 副，身份 +0.03467、公开历史 +0.01099，固定信息集区间均跨零，不启动 PPO。
- [公开历史的局部决策价值 CPU 筛查（2026-10-17）](history-decision-value-2026-10-17.md)：8 个新手牌信息集、2,880 个评分局分支完成；真实身份 Q 选择的独立回报增益 −0.00521，探索性区间跨零；近均匀公开 posterior 未改变先验动作，未建立 PPO 适应验收题，不启动训练。
- [GuanZero 跨局历史利用调查：完整技术报告（2026-10-17）](history-utilization-investigation-summary-2026-10-17.md)：汇总本次 QK-norm、短 PPO、两池完整比赛、公开预测、等长内容替换、融合和训练故障取证；区分信息、策略反应和胜率证据，含恢复点纠正与未完成问题，供外部技术审阅。
- [历史训练经历、收益信号与故障期 checkpoint 审计（2026-10-17）](history-training-audit-2026-10-17.md)
- [两端点不同融合方式的离线对照（2026-10-17）](qknorm-fusion-variants-2026-10-17.md)
- [手牌融合与归一化分离验证（2026-10-16）](qknorm-fusion-isolation-2026-10-16.md)
- [冻结历史编码、读取与出牌状态的分层诊断（2026-10-16）](qknorm-history-stages-2026-10-16.md)
- [两端点等长跨局公开历史替换（2026-10-16）](qknorm-history-swap-2026-10-16.md)
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

- [Training throughput refactor — October 18, 2026](training-throughput-refactor-2026-10-18.md)
- [非有限梯度与归约等待诊断（2026-10-13）](training-gradient-diagnosis-2026-10-13.md)
- [Page-span 基线上的投影优化（2026-10-13，候选验证）](training-stack-next-2026-10-13.md)
- [主线启动 ECC 故障与 page-span 隔离复测（2026-10-13）](training-startup-ecc-2026-10-13.md)
- [Training stack：profile、learner rollout 投影与 page-span（2026-10-13，独立短测）](training-stack-ab-2026-10-13.md)
- [256 宽 × 4 层：完整 PPO 更新吞吐与显存（2026-10-09）](capacity-throughput-2026-10-09.md)
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

- [Rank 1 fixture 逐层反向定位（2026-10-14）](training-fixture-backward-2026-10-14.md)：7323/7324 定位 masked efficient SDPA backward 异常并通过 fixture 修复门槛；7325 三卡 8 ranks × 120 更新稳定性通过。

- [修复基线上的公共历史缓存与归约对照（2026-10-14）](training-cache-reduce-2026-10-14.md)：7366 三卡完整 PPO 完成；等价性通过，缓存吞吐 +1.23%，归约 −1.02% 未证明收益，主线未改。

- [公共历史编码与 rank 等待诊断（2026-10-14）](training-encode-wait-2026-10-14.md)：7392 三卡诊断完成，公共缓存分项与 rank 到达偏差已量化；保留缓存、不采用归约缓冲，诊断非测速。

- [Rollout 缓存分项与调用内索引复用（2026-10-14）](training-cache-reuse-2026-10-14.md)：7452 完成，数值通过但索引复用吞吐 −0.435% 未证明收益；append 为下一诊断重点。

- [Append 内部分项诊断（2026-10-14）](training-append-detail-2026-10-14.md)：7457 完成；attention 最大但无明确可操作优化证据，按用户要求停止。

- [主线梯度爆炸事件与 m8 / u86800 评测（2026-10-15）](main-gradient-blowup-2026-10-15.md)：u74484–77584 actor 梯度范数爆炸（masked efficient SDPA 反向 bug），辅助头退化；10 月 15 日 math SDPA 等补丁上线后恢复；DanLM m8 −0.376、u86800 −0.312（最好点估计，与 m7 −0.334 区间重叠）。

- [主线 SDPA 异常的触发调查（2026-10-15）](training-sdpa-trigger-2026-10-15.md)：窗口内 u74155 已有 norm >100，无同期部署/重启证据；同输入 CPU 对照发现 m6–m7 attention score 大幅增长，支持大 score 下 backward 概率重建不稳定的假设，CUDA 内部机制仍待证实。

- [主线私有注意力诊断（2026-10-15）](private-attention-diagnosis-2026-10-15.md)：同一 fixture、m5–u86800 共 8 组权重的 CPU 对照；所有 head 都是分差极大的硬 one-hot（无 <1 ulp 平局），q 方向几乎不随决策变化，落点按比赛固定（u86800 有 7/8 head 落在常数 BOS）；扰动历史内容的策略 TV 中位数 ≤ 2e-3；m7 起 q/k 的 math 梯度精确为 0，u86800 部分 head 的 Adam √v 已衰减到次正规不动点。只对这一 fixture 成立，未宣布模块退化。

- [QK-norm 三臂 500M 产物核验与机制探针（2026-10-16）](qknorm-mechanism-2026-10-16.md)：同点 +503.84M decisions 与 SHA/源码核验通过；both 解除了极端饱和但 query 仍高度共线，均值替换策略变化小；未启动新训练。

- [Q/K 范数调整的离线可塑性与 Adam 验证（2026-10-16）](qknorm-plasticity-2026-10-16.md)：四档缩放通过 CPU 前向门槛；方向信号保持、步幅提升，保守候选为逐 head 范数 16 配合 m/a、v/a² 且保留 step；清整块 K/V 状态会改变 V，未启动训练或写迁移 checkpoint。

- [Q/K 范数 16 隔离 CUDA 与完整 actor loss 验证（2026-10-16）](qknorm-cuda-gate-2026-10-16.md)：CPU 与 L40S CUDA 全项通过；健康卡诊断 7937 验证完整 actor loss、梯度、主动裁剪、13 个快照与私有 CUDA graph；未启动训练或写迁移 checkpoint。

- [范数 16 隔离短程完整 PPO A/B（2026-10-16）](qknorm-short-ppo-2026-10-16.md)：7939 完成，每臂 120 update / 983,040 decisions，全部有限；方向更新增大但历史均值替换 TV 未增强，不据此长训或采纳主线。

- [两臂端点完整比赛 FULL／ROUND 历史消融（2026-10-16）](qknorm-match-ablation-2026-10-16.md)：完成 2,048 场；FULL−ROUND 为 control +3.32 pp [−1.56, +8.01]、norm16 +1.37 pp [−3.52, +6.25]（95% CI），均未确立跨局收益；不继续训练或改主线。

- [前三局公开出牌的后续预测信息（2026-10-16）](public-history-predictability-2026-10-16.md)：1,024 场原 FULL 比赛精确复跑公开记录；独立 512 场检验和 256 道同牌开局题均未见固定统计预测器的历史增益；直接延续行为偏差反而变差，不能据此证明历史无信息；无神经网络训练或生产改动。

- [更大打法差异的冻结旧模型：公开历史预测复查（2026-10-16）](public-history-diverse-2026-10-16.md)：独立筛选 u2623/u20264/u35588/u86800，正式题分歧率 54.82%（近邻池 6.40%）；1,024 场与 256 道同牌题复查发现小幅历史预测增益，辨认率 41.02%；未测记忆棋力收益，不训练或改主线。

- [差异旧对手池：两端点完整比赛 FULL／ROUND 消融（2026-10-16）](qknorm-diverse-match-2026-10-16.md)：2,048 场完成，两端点全场公共路径均随历史视图变化；control 胜率 +3.91 pp，单项 95% CI [+0.20, +7.62]，双比较区间仍跨零；norm16 −1.17 pp，未见收益，不训练或改主线。

- [Private cross-attention: multi-batch audit tooling and Q/K-norm prototype (2026-10-18)](private-attention-audit-2026-10-18.md): adds `eval/private_attention_audit.py` (CPU, many whole-match minibatches, stratified by round and tribute round) and the config-gated `private_qk_norm` arm (default `none`, the lineage is unchanged). Checkpoints were not available on that VM, so trained-lineage saturation is **unmeasured**. On random init no head saturates. With synthetic trained-like numerics, saturation and exact-zero q/k gradients are global across all strata and minibatches; `qk` bounds the scores and keeps gradients nonzero, `query` does not; neither makes q decision-dependent.
