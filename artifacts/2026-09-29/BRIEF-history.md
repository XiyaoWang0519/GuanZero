# 评审简报：为什么历史 Transformer 不使用公开历史流

本文件可以单独阅读。所有数字都注明来源文件；路径相对于本文件所在目录 `artifacts/2026-09-29/`（仓库文件用 `../../` 开头）。代码描述按分支 `main`（`8e197db`）的源码核对；被测 checkpoint 训练用的源码是 `5b7078e`，评估与消融用的是 `b8c0c54` 的只读导出。

## 1. 目标与现状

- **长期目标**：得到一个会利用整场比赛公开历史的策略，包括在同一场比赛的多局之间记住对手的习惯（[`../../docs/DESIGN.md`](../../docs/DESIGN.md) 第 1 节、第 9.3 节“For opponent adaptation, keep opponent identities stable within a match and compare full versus reduced-history controls”）。
- **现状**：主线终点 u2623（更新 2623，171,900,928 决策，SHA256 `6fd3b94d9ce8f4ae42e5164c3b8645990c09cb4ab1b1e564897ef3732b7972a9`，见 [`../../docs/reports/history-longrun-batched-2026-09-29.md`](../../docs/reports/history-longrun-batched-2026-09-29.md)）在输入消融中**没有可测量的对历史流的依赖**（[`../../docs/reports/history-ablation-2026-09-29.md`](../../docs/reports/history-ablation-2026-09-29.md)，数据在 [`history-ablation/out/`](history-ablation/out/)）。
- 强度背景（同一终点，256 副开发集复式牌，对 B11）：+0.08 [−0.08, +0.24] 净级数/轮，整场胜率 60% [52, 69]（[`evaluation/results.v2.md`](evaluation/results.v2.md)）。B11 是旧的 MLP 模型，只用于评估。

## 2. 证据：输入消融（u2623，对 B11，全部贪心出牌）

来源：[`../../docs/reports/history-ablation-2026-09-29.md`](../../docs/reports/history-ablation-2026-09-29.md)；原始数据 [`history-ablation/out/`](history-ablation/out/)；脚本 [`history-ablation/ablate_lib.py`](history-ablation/ablate_lib.py)。模型权重和观测都不变，只改决策时事件流的内容。

| 条件 | 事件流 |
|---|---|
| FULL | 原样，整场比赛 |
| ROUND | 每局开始时清空，只留本局事件（保留真实局序号） |
| NONE | 始终为空，只有 BOS |
| SWAP | 本局事件不变；之前各局换成另一场比赛同样数量的之前各局事件（仅探针） |

**强度**（[`history-ablation/out/strength_summary.txt`](history-ablation/out/strength_summary.txt)、`strength_summary.json`；牌为 `generate_deals(1000, seed=20260929)`，不是开发集）

| 对局 | 样本 | 净级数/轮 [95%] |
|---|---|---|
| 单局 FULL 对 B11 | 1,000 副 | +0.076 [−0.009, +0.162] |
| 单局 NONE 对 B11 | 1,000 副 | +0.114 [+0.028, +0.198] |
| 单局 FULL − NONE（同牌成对，均对 B11） | 1,000 副 | −0.037 [−0.097, +0.022] |
| 单局 FULL 对 NONE | 1,000 副 | −0.015 [−0.070, +0.042] |
| 整场 FULL 对 ROUND | 185 对，370 场，4,067 局 | −0.010 [−0.071, +0.053]；胜率 48.6% [44.6, 52.7] |

**决策改变**（teacher forcing：沿 FULL 的对局，在同一局面上重算其他条件；只计 ≥2 个合法动作；[`history-ablation/out/probe_analysis.txt`](history-ablation/out/probe_analysis.txt)）

| 比较 | 数据 | 决策数 | 贪心改选率 | 平均 TV |
|---|---|---:|---|---|
| FULL vs NONE | 单局 2,000 局 | 72,192 | 1.48% [1.39, 1.57] | 0.0130 |
| FULL vs NONE | 整场 20 场 226 局 | 8,075 | 1.10% [0.90, 1.32] | 0.0107 |
| FULL vs ROUND | 整场，第 2 局起 | 7,290 | 0.82% [0.59, 1.07] | 0.0072 |
| FULL vs SWAP | 整场，第 2 局起且有供体 | 6,122 | 0.08% [0.02, 0.15] | 0.0009 |

区间来自报告（按牌或按比赛对 bootstrap）。单局中 TV > 0.1 的决策占 1.004%，TV > 0.3 的占 0.003%（`probe_analysis.txt`）。

**按 FULL 自己的第一/第二候选概率差分组**（[`history-ablation/out/margin_analysis.txt`](history-ablation/out/margin_analysis.txt)）：改选几乎全部在概率差 < 0.2 的决策上；概率差 ≥ 0.5 的 48,474（单局 NONE）/ 5,303 / 4,762 / 3,981 个决策中改选为 0。概率差 < 0.05 时的改选率：单局 NONE 21.3%，整场 NONE 17.4%，ROUND 13.9%，SWAP 1.6%。

**SWAP 对照**：在同一批 6,122 个决策上，ROUND 平均 TV 0.0074、改选 55 次；SWAP 平均 TV 0.0009、改选 5 次；TV 差 +0.0065 [+0.0060, +0.0070]（`probe_analysis.txt`）。即事件流的**长度**会轻微改变输出，之前各局里**具体发生了什么**几乎不改变输出。

**随局序号的变化**（`probe_analysis.txt`）：FULL vs ROUND 改选率第 2 局（round_index 1）0.78%（772 个），第 10 局及以后（9+）0.92%（1,851 个）；平均 TV 从 0.0109 到 0.0073。FULL vs SWAP 在 round_index 1–5 改选 0 次，6 及以后 5 次（2,875 个），平均 TV 每组 ≤ 0.0013。分组样本小，没有算分组区间。

**核对**（[`history-ablation/out/verify.json`](history-ablation/out/verify.json)，报告“做法”一节）：FULL 包装与原评估器在 20 副牌 3,015 个决策和一整场比赛（17 局，1,239 个决策，最长前缀 1,747）上逐个决策相同；ROUND 在 613 个决策中没有看到别局事件；NONE 在 448 个决策中前缀全为 0；探针的首次改选位置与实际用 NONE/ROUND 打时轨迹首次分叉位置一致。

**另一处独立迹象**：打法报告的“盲对照”，同样 10 副牌把 u2623 的历史输入关掉，3 副走法不同、7 副相同（[`../../docs/reports/history-play-style-2026-09-29.md`](../../docs/reports/history-play-style-2026-09-29.md)，[`game-review/blind_control.json`](game-review/blind_control.json)）。这说明事件流确实被读入并能改变个别决策，与上面的低改选率一致。

**报告自己声明的局限**：消融后的输入不在训练分布内（测的是依赖程度，不是“不看历史训练出的模型会差多少”）；SWAP 供体也是对 B11 的比赛，所以只检验是否记住具体内容，不检验能否认出对手；对手只有 B11，全部贪心；探针整场部分只有 10 对 20 场。计划中的整场 FULL/ROUND/NONE 各 192 对对 B11 **没有运行**；FULL 对 ROUND 完成 185/192 对。

## 3. 模型看到什么

### 3.1 1,849 维观测（只含本局）

按 [`../../cpp/include/gd/encoder.h`](../../cpp/include/gd/encoder.h)（`kObsDim = 1849`）与 [`../../docs/DESIGN.md`](../../docs/DESIGN.md) 第 6 节核对：自己手牌 108；未见牌 108；本局四家各自已出的牌 4×108（只有集合，无顺序、无牌型）；另三家剩余张数 3×28；出完状态 4×4；级牌与两队级数 3×13；逢人配信息 3+3+12；当前一轮 163（最大牌的动作编码、持有人、连续过牌数、是否领出）；另三家各自最近一手 3×154；阶段与角色 3+6（含自己上一局名次）；本局进贡 4×(54+8)；由进贡推出的已知持牌 3×54。

**只有事件流才有的**（消融报告“问题”一节）：本局的出牌顺序、谁在什么时候用什么牌型、谁在什么牌上过；以及之前各局的全部过程。观测里关于之前各局的只有级数和自己上一局的名次。

### 3.2 事件流 token

[`../../train/history_model.py`](../../train/history_model.py) `PublicStream`：整场比赛每个出牌、过牌（含引擎自动过牌）和公开进贡交换各一个 token，186 维 uint8 = 座位 one-hot 4 + 动作编码 154 + 出牌后剩余张数 one-hot 28；另带局序号和阶段。不含强制过牌标志、私有进贡标志、任何手牌或候选（`append_token` 中有断言；manifest `token_schema`）。

### 3.3 网络连线（`HistoryActor`，[`../../train/history_model.py`](../../train/history_model.py) 第 334–630 行）

主线配置（[`kits/longrun-batched/download/results/segments/main-overnight/manifest.json`](kits/longrun-batched/download/results/segments/main-overnight/manifest.json) 的 `model_config`）：width 128、layers 4、heads 8、max_rounds 16、window 0（看整场）、action_width 128、fusion_width 128、response_mode `auxiliary`；actor 1,416,216 参数，critic 646,913 参数（manifest `parameters`）。

1. **流编码器**（`encode_stream`）：`Linear(186→128)(token) + round_embedding(局序号，clamp 到 0..15) + phase_embedding`，前置可学习 BOS，加正弦位置编码，4 层 pre-norm 因果 `TransformerEncoder`（8 头，FFN 512，dropout 0），最后 LayerNorm。没有私有输入参数。局序号 ≥ 16 的局共用第 16 个嵌入（STAGE_C T1 已记为限制）。
2. **决策状态**（`decision_states`）：query = `MLP(obs 1849→128, 2 层) + seat_embedding`；**一次**交叉注意力读取编码后的流（位置 ≤ 当前前缀，含 BOS），`kv_proj`/`q_proj`/`out_proj`，8 头；`state = LN(query + attended)`，再 `state = LN(state + FFN(state))`。流对动作打分的全部影响都经过这一个带残差的注意力读取；残差意味着 query 本身（只含观测）可以直接通到输出。
3. **候选打分**（`candidate_outputs`）：每个合法候选（完整规范集合，无 top-k）`fused = concat(state, action_tower(候选 154 维))`，`fusion` MLP 输出一个 logit；决策内 softmax。
4. **辅助 response 头**（`response_mode=auxiliary`）：`response_head(fused)` 预测 23 类：本方执行该动作后、本方再次行动或本局结束前，第一个**对手**的 Play 事件（相对座位 1 或 3 × 11 种牌型，或 0 = 无响应）（[`../../train/history_response.py`](../../train/history_response.py)）。标签只来自本谱系自对弈中实际发生的公开事件；只对执行的动作打标签；损失系数 0.1（manifest `response_prediction`）。auxiliary 模式下 `response_bridge` 只接收全零输入，所以预测**不进入**策略 logit，只通过共享的 `fused` 特征影响表示。
5. **critic**（`HistoryCritic`）：`MLP([obs, 另三家手牌逐张计数 3×54])`，3 层宽 256。**不读事件流**，与 actor 不共享参数。

### 3.4 训练配方（manifest `config`/`population`/`reward`，与长跑报告一致）

- PPO：lr 3e-4，clip 0.2，熵系数 0.03，value_coef 1.0，2 个 epoch，每 rank 16 场比赛一个 minibatch（`ddp_global_minibatch`），gamma 1.0，GAE λ 0.95，grad_clip 10，优势按 minibatch 归一化，采样温度 1.0、epsilon 0。
- 布局：4 个 DDP rank × 256 桌，每次更新 64 步，每次更新 65,536 决策（长跑报告）。
- 奖励：`RoundResult.seat_return[team]`（本局净级数），放在该队本局最后一行；**每局结束即 done，不跨局 bootstrap**（manifest `reward`）。所以 RL 目标是逐局的，没有跨局的回报链。
- 进贡/还贡：引擎贪心启发式，这些行不是 PPO 行，但其公开交换事件进入事件流（manifest `reward.tribute`；CLAUDE.md 要求此启发式在各组之间保持一致）。
- 对手池（[`../../train/history_population.py`](../../train/history_population.py) `assignment`）：每场比赛随机选一个座位固定为当前学习者；其余三座各以概率 0.5（`snapshot_probability`）独立抽取最近 4 个快照之一（`population_recent 4`），否则也是当前学习者；座位在整场比赛内固定。快照每 2 次更新取一个（`snapshot_updates 2`），无存档（`population_archive_every 0`）。只有学习者座位产生训练行。
  - 推算（非测量）：最近 4 个快照 × 每 2 次更新 = 对手最多落后约 8 次更新（约 52 万决策）。长跑终点 `population.eligible_snapshots` 为 [1308, 1309, 1310, 1311]（`metrics.jsonl.gz` 最后一行）。
- 训练中的量（rank 0，[`kits/longrun-batched/download/results/segments/main-overnight/metrics.jsonl.gz`](kits/longrun-batched/download/results/segments/main-overnight/metrics.jsonl.gz)，前 31 / 后 31 次更新均值）：`response_accuracy` 0.760 / 0.766；`mean_prefix` 562 / 624 token；`max_prefix` 均值 1,376 / 2,080，全程最大 3,223；熵 0.642 / 0.543（熵在约 1840 次更新处下降，原因未查明，见长跑报告）。

## 4. 已经做过或排除的事（请不要重复提议）

| 做过的事 | 结果 | 来源 |
|---|---|---|
| T7 response 连接：A 纯 PPO、B + 辅助 response 损失、C 再把预测概率显式接入打分 | B − A +0.29 [−0.04, +0.60]（对 B11）；C − B −0.16 [−0.35, +0.04]，C 不再推进；后续都基于 B（即现在的 `auxiliary`） | [`../../docs/reports/history-response-results-2026-09-27.md`](../../docs/reports/history-response-results-2026-09-27.md) |
| 容量 × 对手池 2×2（宽 64/2 层 vs 宽 128/4 层 8 头；最近 4 个 vs 另加本谱系存档，每 32 次更新存一个、最多 16 个、快照座位一半抽存档） | 约 420 万决策下两者都没有收益：容量 −0.081 [−0.202, +0.041]，对手池 −0.095 [−0.210, +0.020]；**没有测历史使用** | [`../../docs/reports/history-factorial-2026-09-27.md`](../../docs/reports/history-factorial-2026-09-27.md) |
| 事后输入消融（本简报第 2 节） | 无可测依赖 | [`../../docs/reports/history-ablation-2026-09-29.md`](../../docs/reports/history-ablation-2026-09-29.md) |
| MLP 时代的离线监督记忆探针（M1 对局数据，memory vs memory_masked） | 整体 CE 改善 +0.000253 [+0.000219, +0.000288]；“late benefit 三个种子均为正”不成立；属于旧 MLP 数据上的诊断，不是新训练入口 | [`../../docs/reports/M2-memory.md`](../../docs/reports/M2-memory.md)；CLAUDE.md 说明离线行为探针不是实验指令 |

**还没有做的**（[`../../docs/STAGE_C_TODO.md`](../../docs/STAGE_C_TODO.md)）：T7 的“history-use controls pending”——从训练开始就对照的 reduced-history 组（模型已支持 `window=k`，只看最近 k 个 token，但没有训练过这样的组）；T6 循环决策模块及其对照：pending；T8 多种子确认与最终测试：pending。消融报告明确说明本次事后消融不算 T7 验收。

## 5. 约束（必须遵守，不是建议）

来源：[`../../docs/DESIGN.md`](../../docs/DESIGN.md) 第 1.1、7.1、7.4 节，[`../../CLAUDE.md`](../../CLAUDE.md)，[`../../docs/PERF_TODO.md`](../../docs/PERF_TODO.md)，[`../../docs/TRAINING.md`](../../docs/TRAINING.md)。

1. 随机初始化，只做自对弈强化学习。不加载任何旧权重、优化器状态；不做蒸馏、模仿、旧 replay；不对 MLP 做 KL 或候选过滤。
2. 训练座位只能是当前 Transformer 和**本谱系**的历史快照（以后可以加入独立的 Transformer 种子，但各组要用同样的种群协议）。
3. MLP、规则机器人、带风格的机器人、外部模型只用于评估；它们的轨迹不能进入学习者、辅助数据集、奖励塑形或训练池。
4. 辅助预测只能用本谱系自对弈中实际发生的事件，因果掩码；不能为未选择的动作编造标签；选择目标的元数据不能泄漏到 actor 输入。隐藏手牌等特权目标只能作为训练时的辅助头或 critic 输入，不能进入 actor 特征。
5. 进贡/还贡启发式在所有对照组之间保持一致。
6. 不能为了省计算而截断、池化或开窗公开历史（PERF_TODO：“Do not silently truncate history or reduce precision”；TRAINING：先减少环境数，再考虑截断）。`window=k` 只能作为**声明的**对照组，单独记录。
7. 推理时遮蔽历史只是敏感性诊断，会引入分布偏移；不能把打乱的、非法的历史当作真实对局（DESIGN 7.4）。“后面几局更好”本身不能证明学会了习惯（DESIGN 9.3）。

## 6. 待检验的假设（均为假设，不是发现）

- **(a) 没东西可记**：训练对手是最近几次更新的本谱系快照，行为几乎相同，跨局记住对手没有回报。消融报告把它列为“一个关于原因的假设”，本次测量无法区分它与其他解释。2×2 实验中加入存档池没有提升强度，但没有测历史使用，且只有约 420 万决策。
- **(b) 观测已足够**：1,849 维观测已包含本局各家出过的牌集合、剩余张数、每家最近一手、当前一轮信息，可能已近似覆盖局内打牌所需的统计量，使事件流对局内决策的边际价值很小。
- **(c) 信用分配太弱**：奖励只在每局结束时出现、不跨局 bootstrap，策略梯度需要通过稀疏的逐局回报去塑造对流的使用；跨局信息的收益（若有）只能通过后面各局的回报间接体现。
- **(d) 结构瓶颈**：流只通过一次带残差的交叉注意力到达决策状态，观测 query 可以绕过它；critic 完全不读流。
- **(e) 辅助任务不需要流**：response 头预测的“下一个对手响应的牌型”可能仅凭观测（当前一轮、剩余张数、最近一手）就能达到约 0.76 的准确率，因此它没有迫使表示去读流。

## 7. 给评审的问题

1. 现有证据更支持哪个假设？哪一个测量能把它们分开？
2. 在第 5 节约束内，什么训练改动能让“使用历史”真正有回报？例如本谱系内部的对手差异（存档快照池、独立 Transformer 种子）、必须读流才能完成的辅助目标（消融报告提到的例子：预测每个座位坐的是哪个快照）、结构改动。请对每个提议说明它检验哪个假设、需要什么对照。
3. 如何衡量成功：SWAP 敏感度（SWAP vs ROUND 的 TV/改选率）、改选率或强度随局序号增长、对带风格对手（只用于评估）的强度、从训练开始就对照的 `window=k` 组。哪些是必要的，门槛怎么定？
4. 哪些零成本诊断应该先在现有 checkpoint 上跑？消融报告列了两个：辅助 response 头的准确率是否依赖事件流（用同样的消融）；直接告诉模型对手身份能值多少。还有哪些？
5. u2623 以外的 checkpoint（例如 u0844）是否值得做同样的消融，以看依赖程度是否随训练变化？

## 8. 文件索引

- 报告：[`../../docs/reports/history-ablation-2026-09-29.md`](../../docs/reports/history-ablation-2026-09-29.md)、[`../../docs/reports/history-longrun-batched-2026-09-29.md`](../../docs/reports/history-longrun-batched-2026-09-29.md)、[`../../docs/reports/history-play-style-2026-09-29.md`](../../docs/reports/history-play-style-2026-09-29.md)、[`../../docs/reports/history-response-results-2026-09-27.md`](../../docs/reports/history-response-results-2026-09-27.md)、[`../../docs/reports/history-factorial-2026-09-27.md`](../../docs/reports/history-factorial-2026-09-27.md)。
- 消融数据：[`history-ablation/out/`](history-ablation/out/)：`strength_summary.*`、`probe_analysis.txt`、`margin_analysis.txt`、`examples.txt`（各类改选中 TV 最大的例子）、`verify.json`、逐决策探针 `probe_*.records.jsonl.gz`（字段：`arg` 各条件的贪心选择、`tv`、`pmax_full`、`prefix`、`round_events`、`cands` 各条件下的候选概率）、`match_*`/`dup_*` 逐场记录。
- 代码：[`../../train/history_model.py`](../../train/history_model.py)（模型）、[`../../train/history_response.py`](../../train/history_response.py)（response 标签）、[`../../train/history_population.py`](../../train/history_population.py)（对手池）、[`../../train/history_ppo.py`](../../train/history_ppo.py)（PPO）、[`../../cpp/include/gd/encoder.h`](../../cpp/include/gd/encoder.h) 与 [`../../cpp/src/encoder.cpp`](../../cpp/src/encoder.cpp)（观测）。
- 训练记录：[`kits/longrun-batched/download/results/segments/main-overnight/`](kits/longrun-batched/download/results/segments/main-overnight/)（`manifest.json`、`metrics.jsonl.gz`）。
- 权重不在仓库中；u2623 的 SHA256 见第 1 节。
