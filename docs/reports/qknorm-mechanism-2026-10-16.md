# QK-norm 三臂 500M 产物核验与机制探针（2026-10-16）

状态：**已完成产物核验与本地 CPU 探针；未启动新训练，未修改主线模型或训练配置。**
本次用户授权只涵盖已有产物核验、三臂 500M 机制探针及下一步判断。

## 结论

`qk-both` 解除了原有的极端 one-hot 和 fixture 上的零 Q/K 梯度；它并非所有
head 都一直近均匀，也没有在终点重新出现原先的极端饱和。但 query 仍高度共线，
历史输出的大部分策略作用可由一个独立校准均值保留。现有数据支持“恢复了部分
读取器可塑性”，不支持“已经学会有收益的跨局适应”或“归一化无效”。

**下一步优先检查读取方向与内容信号的可塑性，而非优先加温度上限或再训 500M。**
建议下一项是无训练的逐 head Q/K 范数调整候选验证：前向策略基本不变、只改 K
不改 V、检查方向梯度尺度及优化器状态处理方案。它只能判断这项手术是否可安全
测试，不能证明续训收益；本次没有实施该手术。随后仍需 FULL/ROUND 完整比赛评测
才能回答跨局收益。没有批准或提交任何后续训练。

## 1. 产物与源码身份

起点 `u86800.pt`：5,688,524,800 decisions，SHA256
`639ccda15452ef101561e836f684f238cb985bfcf412edd48fb063e383d4c720`。

三臂使用已有 `evals/500M-update-094488.pt`，均为 u94488、6,192,365,568 decisions，
从起点实际增加 **503,840,768 decisions**（名义 500M 的保存点超量一致）。
未使用 qk-both 多跑约 200M 后的最终 checkpoint。

| 臂 | query_norm / qk_norm | SHA256 |
|---|---|---|
| control | false / false | `63b497d5efdeb769ac0dae2d2be8aea0fae5dbe8ff8f47f5b8bd03b902369e4f` |
| query | true / false | `cf40f59e37feef00001292d7475b6940b03340daa03fd768dbafc49ddb1e038d` |
| both | true / true | `a4bed2bfb4c3b8c1382a8ed5604e8e797307a37ea2088523dffb1243de6a58bd` |

检查通过：服务器原 `segments/lr-main/update-094488.pt`、evals 副本与本地副本
SHA 一致；已有内部评测的 candidate/baseline SHA 对应这些权重；三臂 lineage
一致、teacher 为空；训练 config 相对起点仅增加两项归一化开关。

三臂记录的 `run_identity.source` 与服务器 `train-src-qknorm/source-identity.json`
完全相同。本地 qknorm 分支的 `history_model`、`history_attention`、`history_rollout`、
`history_aux`、`history_cache`、`history_response`、`logs`、`model` 八个文件哈希均与
该 identity 对应条目一致。探针直接导入该分支，未将模型实现复制进主线。
实际 `train/history_model.py` SHA 为
`8019503df4a65e88583081633ee89b1caaf4b5070b8ba0562c15ccf6d15a0bd4`，
区别于 kit 旧 manifest 中热补丁前的版本；本次以实际训练身份为准。

归一化两臂的 `aux_start_update` 重置到 86800，control 保留 9989；三臂
`aux_warmup_updates=0`、`aux_coef=0.1`，故这项进度差异不会改变辅助损失系数。
population 同步插入归一化的原设计仍是混杂因素：结果属于整套迁移配方，不能
仅归因于 learner 的一个模块。单条 lineage 的一次续训也不能建立多 seed 因果结论。

## 2. 探针范围与检查

新增可复用入口：[bench/qknorm_mechanism.py](../../bench/qknorm_mechanism.py)。
Mac CPU / PyTorch 2.14.0 / FP32 前向和反向，所有调用强制 math SDPA；
attention 分布由捕获的实际 Q/K 在 FP64 重建。它不是 CUDA 内核根因复现。

- 保留原 rank1 fixture：328 决策；SHA256
  `9c3d1018ac52453852536ff5ef497eb1dbcbd553f02e199b30c9570e6c152cd8`。
- 每个终点冻结权重，自博弈推理采样 8 个环境 × 192 vector steps，无优化器。
  三组各抽取 192 决策，共 576 个新决策、24 个比赛身份，覆盖四个座位和 round 0–3；
  新样本前缀 0–263，比原 fixture 短，是补充覆盖而非完整训练分布。
- 起点与三臂均测同一组四个数据集，共 16 次基准前向。
- 每个待测 checkpoint 单独计算校准均值：仅使用 fresh-control 环境 0/1 的
  48 行 `out_proj(attention)`。新样本的干预统计排除各自环境 0/1，均为
  144 行；原 fixture 的 328 行全部用于测试。校准与测试的比赛身份不重叠。
- 均值/零替换都保留原 public prefix、私有观测、合法候选与策略头，只替换
  历史读取输出。它们测离线策略敏感度，不测回报。
- `H / log(prefix+1)` 计算归一化熵；仅 BOS 的行定义为 0。BOS、当前局、此前局
  概率质量逐行和为 1。局别用最后一个可见事件的 round 标签划分，局间尚无本局
  事件时可能归到上一局，不能将该质量统计当作严格的 ROUND 消融。

验证通过：reader 输出的恒等重建、identity 干预 TV 精确为 0、概率质量守恒、
反向梯度有限、actor state_dict 前后逐张量相同。16 次前向的 FP64 显式 attention
参考与 CPU math 输出最大绝对差 6.93e-5。另用原 `attention_probe.py` 独立复测
both 的 fixture，熵一致（中位数定义差异 <0.02），Q/K 梯度逐 head 差 <1e-10。
脚本语法编译通过；未运行与该诊断无关的全仓库测试。

## 3. 机制结果

以下 attention 表为原 fixture，各列按 head 0–7 排列；新采样三组给出同样的
定性结果。梯度仅指同一历史 fixture 的 outcome loss，不代表所有训练 loss。

| 臂 | 归一化熵中位数（head 0–7） | outcome 下 Q/K 非零梯度 head |
|---|---|---|
| start | 全 0 | 0/8 |
| control | 全 0 | 0/8 |
| query | 中位数均约 0，head 7 少量行有梯度 | 1/8 |
| both | .995, 1.000, .236, 1.000, .999, 1.000, .939, .531 | 8/8 |

both 有效尺度 `sqrt(depth) * temperature` 为
`[9.454, 8.872, 7.483, 8.409, 7.837, 6.694, 10.242, 6.637]`，
初值约 6.238。尺度有增长，但终点分布不支持“温度重新导致极端饱和”。
fixture 的 head 2/7 最大概率中位数分别 .267/.200；fresh-control 上为
.513/.355，说明已有选择性，不能把该臂概括为全部均匀读取。

both 的 query 两两余弦中位数仍为 .99875–.99998；Q 投影第一奇异值
235.32，第一方向占平方 Frobenius 范数 97.60%（起点为 97.96%）；
K 对应占比从 93.88% 降到 92.32%。表示仍高度受一个方向支配，但这些
统计本身不证明固定 query 没有用途。

both 在 fixture 的 BOS 平均质量仅约 .005–.023；之前局的质量约 .763–.893。
这只说明它在读取此前编码，不能证明利用了对手习惯。接近均匀时 argmax
可以总落在 BOS，却给 BOS 极少质量：不能用 BOS argmax 比例替代质量统计。

## 4. 历史分支是偏置，还是内容？

下表统一使用 fresh-control 的独立测试环境，共 144 决策。TV 为动作分布的
total variation，均值替换使用本节之前说明的独立校准集。

| 臂 | 均值替换：平均 KL | 平均 TV | argmax 改变 | 零替换：平均 TV |
|---|---:|---:|---:|---:|
| start | .000469 | .00662 | 2.08% | .22296 |
| control | .000670 | .00671 | 2.08% | .27552 |
| query | .000800 | .00842 | 1.39% | .22704 |
| both | .002045 | .01232 | 0.69% | .22052 |

both 在三个新数据集的均值替换平均 TV 为 .0102–.0128、argmax 改变
0–1.39%；其内容敏感度有增加，但仍较小。这些是描述统计，没有做独立
训练重复，也没有给敏感度差异建立置信区间。

在同一 fresh-control 测试集，both 的私有分支去 feature 均值 RMS 中位数
123.54，历史输出为 75.65；历史输出相对校准均值的变化量中位数仅 4.87，
均值替换后 fusion 输出变化 RMS 中位数 .0332，最终 decision state 为 .0207。

**历史分支整体不是数值上接近零。** 零替换变化大、均值替换变化小，支持
“历史分支的常量/共有部分承担了较多策略功能，内容变化的边际作用较弱”。
但这还不能区分 Q/K 的状态依赖、V/输出表示、融合与策略头各自的责任；
不能仅凭两路 raw norm 宣布残差吞没历史，更不能据此直接放大历史分支。

## 5. 已有单局评测核验

仅读取已存在的内部评测；未启动新的 DanLM 或棋力评测。

| 臂 vs control | 净升级/局 | 配对 bootstrap 95% CI |
|---|---:|---|
| query | −.0265 | [−.0770, +.0255] |
| both | +.0060 | [−.0458, +.0580] |

两项都是 2,000 duplicate deals / 4,000 实际单局。区间不能区分，不能称
棋力等效；更不能用独立单局评测证明跨局历史没有价值。

## 6. 后续决策与产物

1. 保留 math fallback；不采纳新架构，不启动新训练。
2. 不把 temperature cap 放在第一优先级：当前证据没有显示重新极端饱和。
3. 下一项建议是读取器可塑性的离线候选检查。若做逐 head Q/K 范数调整，
   同步处理投影 bias、仅处理 kv_proj 的 K 行，先检查 FP32 前向 KL 与梯度变化；
   Adam 的局部状态如何迁移需另行记录，不能随意重置整个 kv_proj 的共享 step。
4. 不因熵恢复就默认继续跑：跨局收益仍需完整比赛 FULL/ROUND；若要评估
   棋力改善，缺失的终点外部评测与多 seed 证据仍未补齐。

本地原始证据位于 `.work/qknorm-mechanism-2026-10-16/`：
`artifact-audit.json`、`results/mechanism.json`、`results/fresh-*.pt`、
`remote/` checkpoint/评测/源码身份、`legacy/both.json`、`validation.json`、
`probe.log` 与 `verify_artifacts.py`。这些文件已存在，但 `.work` 是忽略的本地
证据，不是可移植依赖。关键身份和测量已写入本报告。

复跑命令（需要上述本地 checkpoint、fixture 与匹配的 qknorm 分支）：

```sh
PYTHONPATH=.claude/worktrees/heuristic-kirch-e482e8:python:oracle:. \
  .work/external/danlm-venv/bin/python bench/qknorm_mechanism.py \
  --checkpoint start=.work/velocity-milestones/u86800.pt \
  --checkpoint control=.work/qknorm-mechanism-2026-10-16/remote/500M-update-094488.pt \
  --checkpoint query=.work/qknorm-mechanism-2026-10-16/remote/qk-query/500M-update-094488.pt \
  --checkpoint both=.work/qknorm-mechanism-2026-10-16/remote/qk-both/500M-update-094488.pt \
  --fixture .work/stack-fixture-2026-10-14/rank1-fixture.pt \
  --output .work/qknorm-mechanism-2026-10-16/results
```

服务器仅做读取与下载。本次查看时已有 `gz-main` / `gz-watch` PENDING；
没有提交、取消或恢复这些既有作业。本文不宣称服务器队列已暂停。

2026-10-16 后续：用户批准的离线可塑性检查已完成，见 [Q/K 缩放与 Adam 验证](qknorm-plasticity-2026-10-16.md)。四档 CPU 前向通过；保守候选为范数 16 配合矩坐标迁移，尚无训练/收益结论，未启动训练。
