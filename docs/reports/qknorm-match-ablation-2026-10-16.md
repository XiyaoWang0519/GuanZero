# Q/K 两臂端点：完整比赛 FULL／ROUND 消融（2026-10-16）

状态：**2,048 场全部完成；两端点 FULL−ROUND 的配对区间均覆盖零，尚无跨局历史净收益证据。** 用户授权只做冻结端点的完整比赛
历史消融，不继续训练、不改主线。承接[短程完整 PPO A/B](qknorm-short-ppo-2026-10-16.md)。

## 预先固定的设计

- control 与 norm16 均为上一轮 u94608 端点，后者在分叉时做一次 Q/K 联合范数 16
  与 Adam `m/a`、`v/a²` 迁移，然后完成 120 次更新；本次两者均冻结。
- control checkpoint SHA256：
  `a3d5e42ea6f6dc9a4b514f1bce948a061ee4be0b08c376dc8ce659ccf46948c7`。
- norm16 checkpoint SHA256：
  `64d3fbd56bb7ea6968a4a54b3b7fd06f9d505c844920b2aa7c8636e4bcb22bdb`。
- 两名己方玩家使用同一端点，一起切换 FULL／ROUND。FULL 的 encoder 和 reader
  读取整场公开历史；ROUND 使用原生 `RoundEventStore` 和 collector 的 round-stream
  路径，只编码当前局的公开事件（含进贡），每局重新从位置 0/BOS 开始，保留真实局号。
  游戏状态、合法候选、私人观测和当前等级不重置。
- 两名敌方玩家使用同一个冻结 Transformer 快照，整场固定身份，始终 FULL。
  四个对手来自共同的 u94488 parent 的最近四个 eligible snapshot：u94482、
  u94484、u94486、u94488，acting-weight SHA 逐个验证。不用两臂各自的新对手池，
  不加入 MLP、人工风格倾斜、搜索或额外可见信息。
- **256 个预定 match seed × 双方队伍互换 = 每条件 512 场完整比赛**。
  四条件共 2,048 场；每个对手均衡占 64 seed × 2 场。16 wave，每 wave 32 slot。
  主 seed `16102671`，显式 `VecEnv.reset(match_seeds=...)`；四条件完全相同。
  互换队伍使用相同的物理座位发牌，不声称行动分歧后逐局轨迹仍相同。
- 行动从温度 1 的原策略 softmax 采样；每个 seed/seat 有独立 RNG，使用 FP64
  inverse CDF，随机数不受其他比赛完成顺序、batch 宽度或对手调用顺序影响。
  队伍互换、FULL／ROUND、端点之间共享相应的随机流；分歧后各流消耗会自然分歧。
  进贡、还贡保持原生固定 heuristic。
- 每个 slot 只计其预先指定的首场比赛，所有 slot 必须完成；先结束者后续自动重开
  的比赛不推理、不计分。每 wave 条件最多 20,000 vector step，超限即失败，
  不剔除慢场。每个 worker 1,800 秒硬上限，无自动扩大样本或重提。

## 指标与解释边界

主指标是各端点的**整场胜率 FULL − ROUND**。同一种子的两次队伍互换先平均，
再在四个冻结对手层内按种子重采样；20,000 次 paired cluster bootstrap，分析
seed `16102672`。保留四条件联合配对，报告点估计与 95% CI；两项主比较另给
97.5% 区间，作为 Bonferroni family-wise 95% 判断。区间下界大于 0 才作为正收益
证据；包含 0 不等于等效或证明“历史无用”。

次指标为整场平均局数、每场累计 return、先按场计算的每局平均 return、配对胜负
反转数；norm16 相对 control 的历史收益差属于探索性 interaction，不替代各自的
主比较。不能把同一比赛内的局当成独立样本。

四个对手只是单一 parent 最近六次 update 范围内的快照，行为多样性有限。
结论仅限这些短程端点对该冻结池的整体比赛收益，不能排除面对其他风格时的跨局
适应收益，也不能据此宣布范数方案优于原始方案或批准长训。ROUND 同时移除历史
内容并重置跨局 encoder 上下文与位置，所以测的是完整跨局上下文的净收益，
不单独归因为“识别对手习惯”。己方两名玩家同时消融，也不拆分队友/对手建模。

## 实现与检查

入口：[评测脚本](../../bench/qknorm_match_ablation.py)；
[统计脚本](../../bench/qknorm_match_summary.py)。复用匹配 qknorm 分支的模型、
原生 public-history store 与 KV inference，额外 collector 子类仅控制独立采样、
完成场过滤和跳过 replay 保存。不构建 trainer、不调用 backward/optimizer。
实现代码留在独立 bench 脚本，不修改生产 train/eval/C++ 实现。

四个定向测试通过：独立座位 RNG 不随 batch 顺序变化、ragged inverse CDF 的合法
采样、显式种子不被 collector reset 覆盖且等待全部预定比赛完成，以及 bootstrap
保留四条件配对与对手层权重。另有原生 ROUND 路径的五项相关检查通过（一个环境
条件测试跳过）；这些小模型测试不替代端点比赛结果。
CPU 本地小样本每条件 8 场跑通，两个端点各自 FULL／ROUND 的首局动作摘要逐场
完全相同；正式评测也以首局动作摘要相同为 gate。小样本不额外计入 2,048 场。

原计划使用集群 CPU；作业 7966 因节点预留停在 PENDING，未运行即取消。
正式路径为本机 CPU / PyTorch 2.14.0、四个单线程 worker 并行，每个 worker
处理一个 wave；复用匹配 checkpoint 的 qknorm 源模型（SHA256
`8019503df4a65e88583081633ee89b1caaf4b5070b8ba0562c15ccf6d15a0bd4`）。
上一轮 CUDA PPO 使用 PyTorch 2.11；本次四条件在同一本机运行时配对，
不宣称本次 CPU 采样与 CUDA rollout 逐位等同。已有主线作业未被干预。

本地证据目录：`.work/qknorm-match-ablation-2026-10-16/`，包含 runner、日志、
逐场原始记录和 metadata。`.work` 为忽略的本地证据，不是可移植依赖。

## 结果

16 个 worker 均 exit code 0，固定 64 个 wave-condition 全部完成，合计 **2,048 场、
23,196 局、1,626,538 个显式决策**（含 heuristic 交换，不含引擎内部自动 pass）。
耗时 360.50 秒；最长比赛 25 局，无超时、无慢场剔除、无非有限/非法行动异常。
按 seed、队伍、对手核验全部预定记录，四条件的源/配置/对手 metadata 一致。

各端点 FULL／ROUND 的 **1,024 对首局行动摘要全部相同**。额外的评测仪器对称性
检查使用完全相同的 parent actor 分别充当己方和敌方：四个新 seed 各互换队伍，
共八场；每对行动摘要、局数、决策数相同，return 反号、终局胜负互补，恰好 4/8 胜。
该仪器检查不进入正式样本。端点 checkpoint 与所用模型/缓存源码的 SHA 均未改变。

### 主指标：完整比赛胜率

| 端点 | FULL | ROUND | FULL−ROUND | 配对 95% CI | 两比较校正用 97.5% CI |
|---|---:|---:|---:|---:|---:|
| 原始 control | 150/512 = 29.30% | 133/512 = 25.98% | +3.32 pp | [−1.56, +8.01] pp | [−2.15, +8.59] pp |
| norm16 / moment transport | 172/512 = 33.59% | 165/512 = 32.23% | +1.37 pp | [−3.52, +6.25] pp | [−4.30, +6.84] pp |

FULL 胜而 ROUND 负／相反的配对数：control 为 **85／68**，norm16 为 **86／79**。
两臂都是正的点估计，但未通过预定正收益门槛；区间仍允许数个百分点的收益或损失，
因此不能宣布等效、完全不用历史或“跨局历史无用”。

norm16 相对 control 的历史收益 interaction 为 **−1.95 pp**，95% CI
**[−8.98, +5.27] pp**，未见范数方案增强跨局收益。FULL 条件下 norm16 相对
control 的胜率差为 **+4.30 pp**，95% CI **[−0.98, +9.38] pp**，也是探索性、
未确立的优势。两端点对共同父池的绝对胜率均低于 50%；这不构成对主线或其他
对手的棋力结论。本次没有分叉前 learner 对同一四快照池的正式强度对照，不能将
该绝对水平的成因单独归到短程更新、范数迁移或 CPU 运行时。

四个对手层内的 FULL−ROUND 点估计（仅描述，不做层内显著性选择）：

| 对手 update | control | norm16 |
|---|---:|---:|
| u94482 | +3.91 pp | +0.78 pp |
| u94484 | +7.81 pp | +4.69 pp |
| u94486 | −1.56 pp | +1.56 pp |
| u94488 | +3.12 pp | −1.56 pp |

### 次指标（描述性）

| 条件 | 平均局数/场 | 平均累计 return/场 | 先按场求每局平均 return 后取均值 |
|---|---:|---:|---:|
| control FULL | 11.232 | −4.3555 | −0.5079 |
| control ROUND | 11.086 | −5.0117 | −0.5682 |
| norm16 FULL | 11.426 | −3.6582 | −0.4083 |
| norm16 ROUND | 11.561 | −3.6504 | −0.4137 |

**决定：本轮只确立评测完整性，未确立跨局历史的比赛收益，也未确立范数迁移增强
该收益。维持隔离，不改主线、不继续训练、不自动增加比赛预算。** 若后续要提高
识别能力，需要另行明确更窄的效应门槛或更有行为差异的冻结 Transformer 对手池；
本记录不是后续实验授权。

原始记录：`results/wave-00..15/{metadata.json,waves.jsonl,complete.json}`；
分析：`summary.json`；完成状态：`local-workers.json`；仪器对称性：`symmetry.json`；
源与 native extension SHA：`source-manifest.json`，均在上述 `.work` 证据目录。
重算入口：

```sh
.work/external/danlm-venv/bin/python bench/qknorm_match_summary.py \
  --root .work/qknorm-match-ablation-2026-10-16/results \
  --output .work/qknorm-match-ablation-2026-10-16/summary.json
```
