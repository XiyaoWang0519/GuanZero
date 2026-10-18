# Q/K 范数 16：隔离短程完整 PPO A/B（2026-10-16）

状态：**隔离作业 7939 已完成：两臂 120 次完整更新均数值稳定；未见历史敏感度改善。**
本次范围是原始 qk-both 与范数 16 / Adam 矩迁移的多步稳定性和历史敏感度，
不改主线、不启动长训、不自动续跑。承接 [CUDA 完整 actor loss gate](qknorm-cuda-gate-2026-10-16.md)。

## 预先固定的设计

- 同一 qk-both u94488 / 6,192,365,568 decisions parent，SHA256
  `a4bed2bfb4c3b8c1382a8ed5604e8e797307a37ea2088523dffb1243de6a58bd`。
- control：原权重、原 Adam 状态；norm16：仅 learner 的逐 head Q/K weight+bias
  联合范数在分叉时一次性缩到 16，Adam `m/a`、`v/a²`；保留共享 step、lr、eps、V 与 critic。
  不逐步重归一化，不给后续范数或温度另设硬上界。
- 两臂初始历史快照池逐位相同，不给常驻对手/队友另做迁移；新增快照继续按
  原 snapshot_updates=2 从各自 learner 生成。这样不额外混入初始对手池手术。
- 两个独立进程从 parent 恢复同一 RNG、population RNG、sampler、环境种子。
  行动分歧后的 rollout 会分歧，不宣称全程相同轨迹或严格配对牌局。
- **每臂最多 120 完整 PPO update**，同一健康 L40S 顺序运行。world_size=1，
  原配置每 rank 的 128 env × 64 steps、2 epochs、8 minibatch matches；单臂名义
  983,040 decisions，报告以实际计数为准。有效 batch 是原 8-rank 运行的 1/8，
  因而是单 rank 的短程稳定性筛查，不能代替生产多 rank 验证。
- 保持原 FP32、math fallback、page-span、paged cache、merged snapshot inference、
  辅助头、reward/GAE、学习率、熵与 population 规则。没有新 imitation、旧 MLP
  训练数据、额外监督或参考策略正则。
- 最多预留 2 GPU × 40 分钟，逐卡健康检查后只用同一张健康卡；每臂 900 秒硬
  超时、整个作业 40 分钟，无 requeue、无 resubmit。超时/非有限直接保留失败证据。

## 检查与停止条件

入口：[bench/qknorm_short_ppo.py](../../bench/qknorm_short_ppo.py)，调用真实
`HistoryTrainer.update()`（collect、完整 actor/critic loss、GAE、PPO epochs、
Adam、snapshot、prune），不使用长训 orchestrator。

逐 minibatch 记录原生 actor 裁剪前后范数，与 FP64 范数核对；每次完整 update
检查 actor/critic 参数、Adam 状态有限，并记录 PPO KL、clip fraction、熵、各
辅助损失、内存、Q/K 范数、温度与净切向更新。非有限、裁剪结果不满足阈值、
源身份不匹配或其他异常即停止，不放宽门槛继续跑。

第 **0 / 30 / 60 / 120** 次更新在冻结 CPU 副本上测原 fixture 与前次固定的
三组新样本（328+576 行）。同一 fresh-control 的独立校准行提供每个时点的
历史输出均值；其余环境行测均值/清零替换的 KL、TV、argmax 与融合变化，
以及各 head 的熵、BOS/局别质量、query 相似度与 Q/K 谱。
这些样本和标签不进入 PPO buffer；探针不更新权重。它测固定输入的历史
输出敏感度，不是完整比赛 FULL/ROUND 或跨局适应收益。

单个 continuation seed，没有显著性/等效性或棋力结论。敏感度增加也可能是
噪声或退化，必须结合 loss/KL/数值稳定性判断。以完整 120 次更新是否完成、
所有权重/矩/梯度是否有限为硬稳定性门槛；其余指标报告实际轨迹，不事后挑
某个峰值宣布修复成功。

## 隔离与证据

服务器路径为 `experiments/qknorm-short-ppo-2026-10-16/`，与 main、qknorm
三臂长训和 CUDA gate 分开。源码逐文件按 parent 的 frozen identity 复制并
验 SHA，额外 runner/探针脚本在单独 kit 中记录 SHA。
没有改变源实现或绕过 source/engine identity；使用同一编译引擎扩展的已核验副本。

只在该目录保存端点 checkpoint，并加 `isolated_diagnostic`、parent SHA、迁移
和硬预算标记；不会更新主线 latest 或作为后续长训入口自动采用。
本地 `.work/qknorm-short-ppo-2026-10-16/` 保存 kit、提交/源记录、原始结果与分析。
`.work` 是忽略的本地证据，不是可移植依赖。

迁移和裁剪观测器的本地数值测试：9 passed；这是保护范围与仪器验证，
不代替完整 PPO 运行结果。

<!-- SHORT_PPO_RESULT_BEGIN -->
## 实测结果与结论

**作业 7939 已完成且 exit code 0；两臂均通过短程数值稳定性检查，但没有历史敏感度改善证据。** 范数调整提高了实际参数方向更新幅度，尚未使 query 呈现更多状态依赖；不据此合并主线或启动长训。

### 运行完整性

两臂均从 u94488 到 u94608，各完成 120 次完整 PPO update、983,040 个全部座位决策。初始 config、parent/source 身份、Torch/sampler/population RNG 摘要与恢复后快照池均相同。原 checkpoint 的 13 个 resident snapshot 中，原生 resume/prune 会丢弃旧 match assignment，只保留 4 个 eligible snapshot；两臂这 4 个初始快照的 acting-weight SHA 逐一一致，norm16 没有给它们做手术。后续快照按原规则自然生成。

两臂实际完成轨迹不同，因此 learner rows、样本与 optimizer steps 不完全相同；这不是固定重放 batch，也不应将两条训练轨迹视为逐行配对。

| 指标 | control | norm16 |
|---|---:|---:|
| 完整 updates | 120 | 120 |
| 全部座位 decisions | 983,040 | 983,040 |
| learner rows | 585,898 | 584,613 |
| 完成轨迹训练样本 | 583,396 | 582,017 |
| actor/critic Adam steps | 3,604 | 3,472 |
| actor 裁剪前全局范数均值 | 1.1806 | 1.1649 |
| actor 裁剪前全局范数最大值 | 8.7519 | 5.6503 |
| 原生梯度裁剪触发率（阈值 10） | 0% | 0% |
| PPO approx KL 均值 / 最大 | 0.01371 / 0.03794 | 0.01267 / 0.03149 |
| PPO ratio clip fraction 均值 | 8.95% | 8.41% |
| action entropy 均值 | 0.43816 | 0.45046 |
| 重建完整 actor loss 均值 | 0.50612 | 0.51069 |
| CUDA peak allocated | 1.756 GiB | 1.818 GiB |
| CUDA peak reserved | 4.631 GiB | 4.967 GiB |
| allocation retries | 0 | 0 |

240 次 update 的权重与 Adam 状态检查、7,076 次 actor minibatch 裁剪观测均通过；FP64 原始范数与原生返回范数一致到预设容差，裁剪后范数均满足阈值。完整 loss 均值由各 update 的原生分项均值及固定系数线性重建，不是另写训练目标。两臂数据不同，loss/KL 较低不等于棋力或训练收益更好。端点在本地重新加载后，actor、critic、Adam 张量均有限，且带有 isolated_diagnostic / not_mainline / no_automatic_continuation 标记。

### 方向更新与表示

逐 update、逐 head 的净切向参数更新 / 当前联合参数范数的中位数：Q 为 control 0.00039901、norm16 0.00286654（约 7.18×）；K 为 0.000610023、0.00290638（约 4.76×）。这是各 update 内多个 optimizer 步的净变化，不能当成单步梯度或同数据下的因果倍数。

norm16 终点每 head 的 Q 联合范数为 16.009–16.146，K 为 16.005–16.043。没有快速回涨到原来的约 20–114；有效 logit 尺度全过程最大值为 control 10.373、norm16 10.327。fresh-control 上 norm16 终点逐 head query 余弦中位数仍约 0.998908–0.999982，没有明显打破共线。第 0 次全局 Q/K SVD 的变化含逐 head 重缩放的机械效应，不能把它当成学习恢复。

### 固定样本历史敏感度

下表是三组 fresh 数据独立测试环境的均值替换 TV，各组 144 行，总计 432 行；每时点均值仅由 fresh-control 的另外 48 行校准。按原计划展示所有时点，百分数为动作分布 TV × 100。它测历史读取输出的内容敏感度，TV 更高本身不表示决策更正确。

| 完整更新数 | control 均值替换 TV | norm16 均值替换 TV |
|---|---:|---:|
| 0 | 1.1781% | 1.1781% |
| 30 | 1.1761% | 0.9937% |
| 60 | 1.2945% | 1.1584% |
| 120 | 1.2946% | 1.1614% |

终点 control 为 1.2946%，norm16 为 1.1614%；从各自起点的变化分别为 +0.1165、-0.0167 个百分点。描述性的变化差为 -0.1332 个百分点，方向未支持敏感度增强；未做训练 seed 推断或等效检验，不能据此宣布该方法有害。

均值替换的平均 KL：control 0.002048 → 0.002378；norm16 0.002048 → 0.001712。终点 argmax 改变率约为 control 0.926%、norm16 0.463%。

清零历史读取输出的终点平均 TV 仍显著更大：control 21.85%，norm16 19.07%。这与此前的“共有部分/偏置承担较多策略作用”相容；不能定位为某一个模块的根因，也不证明跨局历史无价值。

原 fixture 与每组 fresh 的逐 head 熵、query cosine、均值/零替换 KL/TV/翻转率、分支尺度与融合变化均保存在四时点 probe JSON 中；没有拿某个临时峰值作为终点结果。

### 结论的边界与后续

这轮完成了单 rank 的短程稳定性筛查，也确认更容易转向的参数并未自动带来更强历史内容依赖。它没有证明长程收益、生产 8-rank 稳定性、跨局适应或棋力变化。建议先保留为隔离候选，下一项证据应转向完整比赛 FULL/ROUND 的历史价值与读取/融合内容信号定位，而不是仅因数值稳定就继续长训；本次没有启动这些后续工作。

主线源码、parent checkpoint 和既有作业未改；结束时原有 7925/7935 仍为 PENDING，短程 7939 已退出且没有 successor。

### 端点和原始证据

- control endpoint SHA256：`a3d5e42ea6f6dc9a4b514f1bce948a061ee4be0b08c376dc8ce659ccf46948c7`。

- norm16 endpoint SHA256：`64d3fbd56bb7ea6968a4a54b3b7fd06f9d505c844920b2aa7c8636e4bcb22bdb`。

- runner SHA256：`7515f22ff88390851ed1ee0fa9bbb7ae72cff85fa057838773acc22ed2f1d2a1`。

本地 `.work/qknorm-short-ppo-2026-10-16/remote/` 保存完整原始 metrics、diagnostic、四时点 probe、源准备/kit 身份、健康检查、作业退出与两个 endpoint；`summary.json`、`validation.json`、`evidence-manifest.json` 保存独立汇总与校验。没有以 `.work` 文件存在本身替代可移植报告中的关键数值和身份。

可复算入口：[bench/qknorm_short_summary.py](../../bench/qknorm_short_summary.py)：

```sh
python3 bench/qknorm_short_summary.py \
  --root .work/qknorm-short-ppo-2026-10-16/remote \
  --output .work/qknorm-short-ppo-2026-10-16/summary.json
```
<!-- SHORT_PPO_RESULT_END -->
