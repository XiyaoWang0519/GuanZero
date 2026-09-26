# GuanZero Kaggle 免费 GPU campaign 总报告 — 2026-09-26

**三轮训练、内部最终验证和 DanLM 最终对照全部完成。** 预先固定的 R2 lineage 主模型在内部测试中优于五小时基线及自己的父模型；对 DanLM 的配对提升区间跨零，未能确认外部对手表现改善。独立 R1 lineage 对父模型的净级差也未确认改善，low-lead 风格弱点仍在。默认模型未由本任务替换。

Kaggle 累计约 **26.75 会话小时**，付费 **$0**。Notebook COMPLETE、两个 W&B run finished，本地评测进程及进程组已退出。全部模型、Adam、league、配置、源码、日志和原始分数已保留。

## 内部最终测试

主候选在 R3 启动前固定为 lineage_r2 的最终终点；lineage_r1 是独立复验，不根据最终结果换选模型，也不选择周期探针峰值。先保存 endpoint SHA，再使用独立最终 seed **2026092691**。每项 4,000 个整 duplicate deals + 1,000 场完整比赛；每项排除数均为 0。

| 终点比较 | 同牌净级差 / 局（95% bootstrap CI） | 完整比赛胜率（95% Wilson CI） |
|---|---|---|
| 独立 R1 lineage 对五小时基线 | +0.111125 [+0.071747, +0.150250] | 594/1000 = 59.40% [56.33%, 62.40%] |
| 独立 R1 lineage 对 R1 父模型 | +0.032000 [-0.006125, +0.071500] | 531/1000 = 53.10% [50.00%, 56.18%] |
| 主候选 R2 lineage 对五小时基线 | +0.160000 [+0.121500, +0.198884] | 566/1000 = 56.60% [53.51%, 59.64%] |
| 主候选 R2 lineage 对 R2 父模型 | +0.040375 [+0.001625, +0.081000] | 565/1000 = 56.50% [53.41%, 59.54%] |

四项主 bootstrap 区间（2,000 次重采样）和比赛 Wilson 区间已独立重算；原始两腿、比赛记录和 68 个 duplicate 单元已核对。R1 对父模型的比赛区间下界仅为 50.0010%，同时净级差区间跨零，不能称为稳健复现。R2 对父模型的净级差区间也仅略高于零。区间条件于固定模型和该测试协议，不涵盖一般训练 seed 方差。

R1 对父模型在 low-lead 风格下出现 -0.02650 [-0.05063, -0.00250] 的退步。R2 对父模型八项风格区间均跨零。全部正负风格结果见 [R3 完整报告](/Users/xiyaowang/Documents/Projects/GuanZero/docs/reports/kaggle-r3-results-2026-09-26.md)；风格均值从原始记录核对，风格区间沿用已校验原始文件，未全部独立重算，且未经多重比较校正。

## DanLM 独立最终对照

使用固定主候选、R2 父模型、五小时基线三个 plain policies，无搜索。最终 seed **2026092699**，4,000 组相同牌，每组两腿互换队伍座位，每模型共 8,000 局。三模型共同完成 4,000 组，共同有效 **3,995 组**；5 组排除（0.125%），没有未完成组。以下三模型分数及配对差值都只使用完全相同的 3,995 组。

| 模型对 DanLM | 净级差 / 局（95%整 deal bootstrap CI） | 单局胜率（95%整 deal bootstrap CI） |
|---|---|---|
| R3 主候选 | -1.942678 [-1.975970, -1.909634] | 13.38% [12.68%, 14.08%] |
| R2 父模型 | -1.948436 [-1.980726, -1.916270] | 13.54% [12.85%, 14.23%] |
| 五小时基线 | -1.975344 [-2.008642, -1.941677] | 13.34% [12.65%, 14.04%] |

| 同牌配对比较 | 净级差变化 / 局（95% CI） | 单局胜率变化，百分点（95% CI） |
|---|---|---|
| 主候选 − 五小时基线 | +0.032666 [-0.006762, +0.072841] | +0.037547 [-0.813517, +0.913642] |
| 主候选 − R2 父模型 | +0.005757 [-0.031045, +0.043680] | -0.162703 [-0.963705, +0.663642] |
| R2 父模型 − 五小时基线 | +0.026909 [-0.011014, +0.063705] | +0.200250 [-0.625782, +1.026283] |

**三项配对净级差和单局胜率变化区间均跨零。** 本次数据不支持“续训已显著改善 DanLM 表现”，也不证明模型完全等效。三个模型对 DanLM 的绝对表现仍明显落后。单局胜率不是完整比赛胜率，不能与上一节 56.6% / 56.5% 的内部完整比赛结果直接比较。

每模型的单组得分为两腿本方净级差平均值；三个模型在同一有效 deal 上相减，再按整 deal 共同索引重采样 5,000 次，统计 RNG seed 2026092872。两腿不当作独立样本。全部分数、差值与区间已独立重算，240 个原始 gzip 分块 / 24,000 条单局记录、74 个冻结代码文件、四个模型 checkpoint SHA、固定配置和生成牌面均核验通过。每 50 组保存原始记录，完整原始数组、排除明细及 SHA 位于 [DanLM 审计](/Users/xiyaowang/Documents/Projects/GuanZero/.work/kaggle-continuation-20260925-r3/danlm-final/audit.json)，独立核验见 [independent-verification.json](/Users/xiyaowang/Documents/Projects/GuanZero/.work/kaggle-continuation-20260925-r3/danlm-final/independent-verification.json)。

排除组（0-based ID）及原因：

| deal ID | 失败模型 / 腿 | 原因 |
|---|---|---|
| 158 | 五小时基线 / a | DanLM 选择无法在镜像引擎中继续（mirror_failed / their_choice_missing） |
| 505 | 五小时基线 / a | DanLM 选择无法在镜像引擎中继续（mirror_failed / their_choice_missing） |
| 720 | 五小时基线 / a | DanLM 选择无法在镜像引擎中继续（mirror_failed / their_choice_missing） |
| 2830 | R3 主候选 / b, R2 父模型 / b | DanLM 选择无法在镜像引擎中继续（mirror_failed / their_choice_missing） |
| 3352 | 五小时基线 / a | DanLM 选择无法在镜像引擎中继续（mirror_failed / their_choice_missing） |

主候选和父模型各有 1 个失败 deal（同为 2830），基线有 4 个；取三者并集排除，因此表中全为共同有效样本。保留这 5 组的两腿和所有错误，不把失败局计成胜局或隐去。

本测试以 DanLM 引擎作裁判，由 GuanZero 引擎同步提供自身观测。适配过程中仍有非致命差异：

| 适配计数 | 主候选 | R2 父模型 | 五小时基线 |
|---|---|---|
| our_candidate_unmatched | 227 | 256 | 243 |
| their_choice_reading | 31 | 27 | 29 |
| their_choice_missing | 1 | 1 | 4 |

`our_candidate_unmatched` 表示我方候选未能映射到 DanLM 合法动作而被过滤；`their_choice_reading` 表示同类型同牌但解析不同的动作被适配；`their_choice_missing` 导致无法继续并排除整组。这些是事件次数，不是失败组数。所有共同有效局的双方终局顺序和奖励核对一致，但不能据此声称规则与动作支持完全等价。结论只适用于此固定适配协议和保留的共同有效牌组，不宣称标准外部榜单成绩、G9 或 SOTA。

本机只执行 CPU 推理与统计，监督脚本共运行 **3251.34 秒（54.19 分钟）**，在 2026-09-26 01:34:36 UTC 正常退出，早于独立硬截止 02:55:06 UTC；没有重启或按结果扩展样本。macOS ARM64 / CPython 3.12 依赖经过短预检，未将 Darwin 二进制上传 Linux。

## R1 / R2 选择过程

两轮均从同一已验证五小时终点和 Adam 起步，分别用训练 seed 2026092601 / 2026092602；每轮原 LR 1e-5 与半 LR 5e-6 两组各 8 小时，critic LR 都为 1e-4。开发 seed 2026092611 / 2026092612，每项 2,000 个整 duplicate deals + 500 场完整比赛。

| 轮次 / 比较 | 净级差 / 局（95% CI） | 完整比赛胜率（95% CI） |
|---|---|---|
| R1 原 LR 对起点 | +0.027250 [-0.026256, +0.085006] | 55.40% [51.02%, 59.70%] |
| R1 半 LR 对起点 | +0.085250 [+0.032738, +0.138506] | 55.00% [50.62%, 59.31%] |
| R1 半 LR 对原 LR | +0.042750 [-0.011750, +0.095256] | 53.20% [48.82%, 57.53%] |
| R2 原 LR 对起点 | +0.043750 [-0.013756, +0.098256] | 57.80% [53.43%, 62.05%] |
| R2 半 LR 对起点 | +0.104250 [+0.050744, +0.159006] | 55.20% [50.82%, 59.50%] |
| R2 半 LR 对原 LR | +0.043750 [-0.007506, +0.098006] | 52.20% [47.82%, 56.54%] |

半 LR 对原 LR 两轮均为正；两轮等权、分层整 deal bootstrap（20,000 次，统计 seed 2026092620）合并为 **+0.043250 [+0.003875, +0.082253]**。满足 R2 启动前冻结的方向、合并下界、随机 heldout 风格及完整比赛不明确退化门槛，因此采用半 LR 续训。该合并区间只条件于两组固定模型对，并非一般训练 seed 方差；两项单轮直接区间各自仍跨零。

反例必须保留：固定 low-lead 风格下，半 LR 对原 LR 两轮分别 **-0.04425 [-0.07825, -0.00850]** 与 **-0.04600 [-0.08025, -0.01225]**。固定风格和冻结门槛中的随机 heldout 风格不是同一组，门槛通过不代表该弱点消失。不能宣称半 LR 对所有风格、或对完整比赛均有优势。

R3 各自继承 R1/R2 半 LR 父模型及 Adam，用独立训练 seed 2026092603 / 2026092604 再训练各 10 小时；共同重建原 league + 五小时基线 + 两个父模型。环境、计数器、league 统计重置，属于 warm start。算法源码冻结于 4940246fbd520ff6bd9e246e729f425b69dfcba8；policy LR 5e-6 / critic LR 1e-4，1024 env / rollout 64 / epochs 4 / minibatch 8192 / engine 1 / torch 1。未混入其他任务优化或此前搜索收益。

## 预算、终态与产物

| 会话 | Kaggle 版本 | 配额会话时长 | Artifact SHA 核验 |
|---|---|---|
| R1 | 4 | 约 8.19h | 173 项，4,159,982,857 字节 |
| R2 | 5 | 约 8.19h | 178 项，4,289,298,341 字节 |
| R3 | 6 | 37,332.6 秒 = 10.37017h | 207 项，5,098,415,347 字节 |

累计约 26.75017h，低于本次 29.75h 上限；前两轮取账户 0.01h 精度的配额增量，因此总数为近似值。以上为 Kaggle T4 x2 会话配额时间，不把两张卡重复计为两次会话。付费支出为 $0。账户已在 9 月 26 日刷新为 30h；本次额度不因刷新增加，也没有为用完额度再启动训练。

最终 CLI 确认 Notebook COMPLETE，W&B 两条 R3 run finished；本机 supervisor/worker PID 和进程组均已消失。结束证据：[云端终态](/Users/xiyaowang/Documents/Projects/GuanZero/.work/kaggle-continuation-20260925-r3/campaign-terminal-check.json)、[provider 时长](/Users/xiyaowang/Documents/Projects/GuanZero/.work/kaggle-continuation-20260925-r3/terminal-provider-proof.json)、[本机终态](/Users/xiyaowang/Documents/Projects/GuanZero/.work/kaggle-continuation-20260925-r3/danlm-final/terminal-process-proof.json)。

保留的原始产物：

- [R1 全量结果（模型/Adam/league/配置/日志/源码）](/Users/xiyaowang/Documents/Projects/GuanZero/.work/kaggle-continuation-20260924/results-r1)
- [R2 全量结果](/Users/xiyaowang/Documents/Projects/GuanZero/.work/kaggle-continuation-20260925-r2/results)
- [R3 全量结果](/Users/xiyaowang/Documents/Projects/GuanZero/.work/kaggle-continuation-20260925-r3/results)
- [DanLM 原始分块和牌面](/Users/xiyaowang/Documents/Projects/GuanZero/.work/kaggle-continuation-20260925-r3/danlm-final)
- [R1 报告](/Users/xiyaowang/Documents/Projects/GuanZero/docs/reports/kaggle-r1-results-2026-09-25.md)
- [R2 报告](/Users/xiyaowang/Documents/Projects/GuanZero/docs/reports/kaggle-r2-results-2026-09-25.md)
- [R3 报告（包含全部风格）](/Users/xiyaowang/Documents/Projects/GuanZero/docs/reports/kaggle-r3-results-2026-09-26.md)

固定模型 SHA256：

- R3 主候选：`60d7bf62e93f6ddc2ea27729dfb033de9801e75bd2340ea69b2684febd4b1a15`，[checkpoint](/Users/xiyaowang/Documents/Projects/GuanZero/.work/kaggle-continuation-20260925-r3/results/results/lineage_r2/run/latest.pt)
- R2 父模型：`6cd944888c26dd8a86bd08ecd7e906a2fff1968955a1944551811773c7baf585`，[checkpoint](/Users/xiyaowang/Documents/Projects/GuanZero/.work/kaggle-continuation-20260925-r3/dataset/lineage_r2.pt)
- 五小时基线：`ebb614f35f825e835c9d5b18bdc5a6efb0f79d5ba5ed9bfcdf44eca98ae139fe`，[checkpoint](/Users/xiyaowang/Documents/Projects/GuanZero/.work/overnight-20260924/final/main-final.pt)
- DanLM 对手：`5341d7316bac1ead4098c796ea03f7e38fb47f001cc647a52be3ed0d15cd7244`。

本任务保留现有私有 Dataset 与 Notebook，不删除用户数据，不自动替换默认模型。本 campaign 已结束，不再启动训练或使用最终测试集调参。自动检查 `guanzero` 已通过应用工具设为 **PAUSED**，并回读配置确认。

[机器可读总报告（含所有原始报告汇总、排除明细与证据 SHA）](/Users/xiyaowang/Documents/Projects/GuanZero/docs/reports/kaggle-campaign-results-2026-09-26.json)。
