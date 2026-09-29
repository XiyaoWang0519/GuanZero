# 评审简报：历史 Transformer PPO 训练提速

本文件可以单独阅读。所有数字都注明了出处文件；仓库文件的链接相对本目录（`artifacts/2026-09-29/`）给出。
仓库状态以 `main` 的 `8e197db` 为准。本文只陈述报告和数据文件里有的内容；报告不支持的说法一律省略。

## 1. 目标

在**一块 RTX 4090（24 GB）**上，提高历史 Transformer 自对弈 PPO 训练器每秒（以及每美元）处理的决策数，
**不减少模型能学到的东西**。

- 模型：宽 128、4 层、8 头的历史 Transformer，从随机初始化开始自对弈 PPO（[`history-longrun-batched-2026-09-29.md`](../../docs/reports/history-longrun-batched-2026-09-29.md)）。
- 租用机器：Vast RTX 4090，约 $0.557/h（机器 67872）到 $0.783/h（机器 20082）（同上报告、[`kits/longrun-batched/README.md`](kits/longrun-batched/README.md)）。
- 最近一次 8 小时主线训练：1,779 次更新、新增 116,588,544 个决策，平均约 4,046 决策/秒，估算 $4.52（`history-longrun-batched-2026-09-29.md`）。

## 2. 约束（必须遵守，不是建议）

### 2.1 提速改动的档位

报告中使用的档位（第 1、2 档见 [`history-snapshot-batching-2026-09-28.md`](../../docs/reports/history-snapshot-batching-2026-09-28.md) 的"档位与 CUDA 验证"表；第 3 档见 [`history-actor-ranks-2026-09-28.md`](../../docs/reports/history-actor-ranks-2026-09-28.md) 对快照间隔 8 的定位；仓库里没有一份单独的档位定义文件，以下是这些用法的归纳）：

| 档位 | 含义 | 验收 |
|---|---|---|
| 第 1 档 | 逐位一致：行、选择、权重、优化器、采样器、指标逐位相同 | CPU/CUDA 逐位测试（例：显存回收、剖析计时） |
| 第 2 档 | 数据分布不变：只有 FP32 归约顺序噪声，或随机流拆分/重排（例：合并快照推理、4 个 DDP rank + 全局 minibatch） | 数值容差门禁；按 refactor 报告，"per the acceptance rules they need no strength A/B"，但下一次 256 副评估要注明源码变化（[`training-stack-refactor-2026-09-29.md`](../../docs/reports/training-stack-refactor-2026-09-29.md)） |
| 第 3 档 | 改变学习动态（例：快照间隔 2→8、改桌数） | 需要 A/B，加独立的强度评估 |

### 2.2 明令禁止

来源：[`docs/TRAINING.md`](../../docs/TRAINING.md)（"Every optimization must preserve model capability, FP32, full event history, canonical legal support and PPO/reward/population semantics"）、[`history-recipe-summary-2026-09-26.md`](../../docs/reports/history-recipe-summary-2026-09-26.md)（"no lower precision, history truncation, smaller legal action set, MLP teacher or filter"）。

- 降低数值精度：FP16、BF16、TF32 都不行（全程 FP32、TF32 关闭，见 [`history-overnight-large-2026-09-28.md`](../../docs/reports/history-overnight-large-2026-09-28.md)）。
- 截断、池化或窗口化公开历史；必须是整场比赛的完整事件流。
- 缩小合法动作集、候选剪枝、MLP 老师或过滤。
- 更少的样本或更弱的训练信号（改 PPO/奖励/对手池语义属于第 3 档，不能以"提速"名义进入）。
- 训练对手池只能是本谱系当前/过去的 Transformer 快照（[`docs/DESIGN.md`](../../docs/DESIGN.md)、仓库 `CLAUDE.md`）。

## 3. 当前测得的状态

### 3.1 布局（主线，更新 845–2623）

来源：`history-longrun-batched-2026-09-29.md`、[`history-actor-ranks-2026-09-28.md`](../../docs/reports/history-actor-ranks-2026-09-28.md)。

- 4 个 DDP rank（`train.history_ddp --world-size 4`）共用一块 GPU，集合通信走主机内存上的 gloo（提交 `a45a324`）。
- 每个 rank 256 桌；每次更新 65,536 个决策（= 4 × 256 × 64 步）。
- PPO：2 个 epoch，每个 rank 每个 minibatch 16 场，`--ddp-global-minibatch`（优势归一化和损失加权在 4 个 rank 的并集上做）。
- lr 3e-4，熵系数 0.03，辅助 response 0.1，快照间隔 2，最近 4 个快照组成对手池。
- 开着的 CUDA 路径：`--causal-sdpa --rollout-kv-cache --rollout-batched-attention --rollout-wide-projection --rollout-private-graphs --rollout-triton-cache --learner-batched-attention`（`history-overnight-large-2026-09-28.md`），外加 `batch_snapshot_policies=true`、`rollout_trim_cuda_cache=auto`（主线长跑）。

### 3.2 每次更新的时间

| 配置 | 每次更新（秒） | 采集 / 学习（秒） | 决策/秒 | 来源 |
|---|---:|---:|---:|---|
| 1 个进程（main-w1，最后 15 次更新） | 35.0 | 26.9 / 8.0 | 1,871 | `history-actor-ranks-2026-09-28.md` |
| 4 个 rank（main-w4） | 19.7 | 13.6 / 6.5 | 3,324 | 同上 |
| 4 rank，合并快照推理 关（A/B，机器 67872） | 20.84 | 14.18 / 8.12（最慢 rank） | 3,145 | `history-snapshot-batching-2026-09-28.md`、[`kits/snapshot-batching/ab-summary.md`](kits/snapshot-batching/ab-summary.md) |
| 4 rank，合并快照推理 开 | 15.39 | 8.87 / 7.00 | 4,263（1.355 倍） | 同上 |
| 8 小时主线（开 + 显存回收），31 次更新窗口 | — | 8.9–10.6 / 7.0–8.0 | 3,693–4,837，平均约 4,046 | `history-longrun-batched-2026-09-29.md`、[`kits/longrun-batched/longrun-summary.md`](kits/longrun-batched/longrun-summary.md) |

A/B 只有两对相邻块（比值 1.384 和 1.326），没有置信区间。两台机器 CPU 不同，绝对速度不能跨机器比。

### 3.3 采集时间花在哪里（合并推理之前，机器 20082，更新 870–884）

来源：[`kits/collect-profile/profile-summary.md`](kits/collect-profile/profile-summary.md)、`history-snapshot-batching-2026-09-28.md`。剖析模式在每个阶段边界同步 CUDA，秒数只用于分摊。

| | 学习者调用 | 快照调用 |
|---|---:|---:|
| 每步调用次数（rank 均值） | 1.00 | 11.57 |
| 每次调用行数（均值） | 155.1 | 8.1 |
| 行数占比 | 62.5% | 37.5% |
| 公开缓存/整理（秒/更新） | 3.18 | 3.84 |
| actor 与采样（秒/更新） | 0.46 | 5.63 |
| actor 每次调用（ms） | 6.92–7.55 | 7.51–7.72 |
| 公开缓存每次调用（ms） | 47.5–53.2 | 5.08–5.24 |

- 采集 13.81 s 中，公开缓存/整理 7.01 s、actor 与采样 6.09 s，其余阶段合计约 0.7 s。
- **actor 调用耗时几乎与行数无关**（155 行和 8 行都是 7–8 ms），这就是合并快照推理的依据。
- 合并后（剖析块）：快照调用从每步 11.44 次、每次 8.2 行变成 1.00 次、93.8 行；快照 actor 时间 5.26 → 0.17 s/更新；快照缓存阶段 4.07 → 4.55 s。**采集的大头现在是公开 KV 缓存编码**（`history-snapshot-batching-2026-09-28.md`）。
- 不剖析参考：3,280 决策/秒，19.98 s/更新，GPU 平均利用率 86.9%，平均前缀 589 个 token，驻留快照 12–14 个（`profile-summary.md`）。

### 3.4 GPU 利用率、功耗、kernel 数

- 8 小时主线：GPU 平均利用率约 90%，平均功耗约 232 W（`history-longrun-batched-2026-09-29.md`；原始采样在 `kits/longrun-batched/download/results/gpu-samples.jsonl`，字段 `util`、`power_w`、`mem_used_mib`，每 5 秒一行）。
- 对 9 月 28 日剖析的另一份分析（分支 `XiyaoWang0519/find-training-speedups`，转述于 `training-stack-refactor-2026-09-29.md`）：利用率 92%，功耗只有约 207 W；每个 rank 每步约 3,100 次 kernel 启动（约 2,750 次在快照编码里）；学习每个 minibatch 约 0.1 TFLOP。结论是 **kernel 数量受限，而不是算力受限**。这一点是分析，不是 A/B。
- 更早的 MLP 训练器上也测到每步约 281–342 次 kernel 启动、单 Python 线程是瓶颈（[`perf-gpu-2026-09-24b.md`](../../docs/reports/perf-gpu-2026-09-24b.md)、[`perf-pipeline-2026-09-25.md`](../../docs/reports/perf-pipeline-2026-09-25.md)）。

### 3.5 显存余量

- 合并推理开、无回收：nvidia-smi 最高 23,578 MiB，离 24 GB 约 0.4 GB（`history-snapshot-batching-2026-09-28.md`）。原因是分配器缓存的空闲块只涨不降（已分配内存并不更高），细节见 [`kits/snapshot-batching/memory-analysis/tables.md`](kits/snapshot-batching/memory-analysis/tables.md)。
- 加显存回收后：8 小时主线 nvidia-smi 12.3–20.9 GB，均值 17.6 GB；每个 31 次更新窗口最高 18,306–20,920 MiB；回收耗时 0.106–0.134 s/更新（`history-longrun-batched-2026-09-29.md`、`longrun-summary.md`）。
- 4 个 rank 已用到 21.5 GB（未合并时），"显存是扩到更多进程的上限"（`history-actor-ranks-2026-09-28.md`）。

### 3.6 历史前缀长度

- 8 小时主线每 31 次更新窗口的平均前缀 562–669 个 token（`longrun-summary.md`）。
- 由 `kits/longrun-batched/download/results/segments/main-overnight/metrics.jsonl.gz`（rank 0，字段 `mean_prefix`、`max_prefix`）重算：1,779 次更新的 `mean_prefix` 平均 620，`max_prefix` 最大 3,223。
- 评估时（历史消融，整场比赛）FULL 的平均前缀约 520–690，最长 2,143（[`history-ablation-2026-09-29.md`](../../docs/reports/history-ablation-2026-09-29.md)）。

### 3.7 单决策评估成本与前缀长度（CPU，1 线程，67 个合法动作的局面）

来源：`history-ablation-2026-09-29.md`（`history-ablation/out/time_decision.txt`）。

| 前缀 token 数 | 100 | 300 | 600 | 1,000 | 1,500 | 2,000 |
|---|---:|---:|---:|---:|---:|---:|
| 毫秒/决策 | 1.8 | 9.2 | 32.3 | 87.2 | 193.5 | 353.9 |

B11（MLP）同一局面 0.5 ms。这是不带 KV 缓存的评估路径；训练采集用 KV 缓存，两者不能直接换算。一次 256 副牌对 B11 的评估在本地 CPU 上约 22–54 分钟（`evaluation/results.v2.md` 的 wall 列）。

## 4. 已经试过的（请不要重复提议）

| 做法 | 结果 | 档位 | 来源 |
|---|---|---|---|
| actor 进程改为同一 GPU 上的 4 个 DDP rank | 1.78 倍（1,871 → 3,324 决策/秒）；GPU 利用率 55% → 82% | 2 | `history-actor-ranks-2026-09-28.md` |
| 合并快照 actor 前向（`--batch-snapshot-policies`，`e7ef4cc`） | 1.355 倍（两对块）；CUDA 生产尺寸 log-prob 最大差 9.537e-07 | 2（仅快照座位） | `history-snapshot-batching-2026-09-28.md` |
| 显存回收（`rollout_trim_cuda_cache`，`0b73d2c`/`06b5f0b`/`5b7078e`） | 修复合并后显存只涨不降；0.120 s/更新，占墙钟 0.76%；8 小时无分配重试 | 1 | 同上、`history-longrun-batched-2026-09-29.md` |
| 在 CPU 上做 rollout 推理 | 小历史模型（T7，同一主机 A/B/C）：CPU 引擎 + CUDA 推理 + CUDA 学习约 1,575 决策/秒；全 CPU 876；CPU 推理 + CUDA 学习 1,359。选定 CUDA 推理。MLP 时代的分析也指出纯 CPU actor 机器会把最贵的神经推理放到最慢的设备上 | — | [`history-device-placement-2026-09-26.md`](../../docs/reports/history-device-placement-2026-09-26.md)、[`perf-learner-2026-09-24.md`](../../docs/reports/perf-learner-2026-09-24.md) |
| 同进程流水线（`rollout_pipeline`，`cbc3a4a`） | 采集慢 1.4–2.2 倍，已删除；原因是单 Python 线程的主机开销 | — | `perf-pipeline-2026-09-25.md`（MLP 训练器） |
| 公开历史 CUDA Graph | 完整采集实验没有一致收益，未进入生产代码 | — | [`history-cuda-throughput-2026-09-28.md`](../../docs/reports/history-cuda-throughput-2026-09-28.md) |
| CUDA 上用 `baddbmm` 做独立投影 | 改变 FP32 末位，81 个 attention 校验失败，恢复逐行线性投影 | 当时未过第 1 档 | 同上 |
| pinned 总输入上传 | 同机对比无稳定收益，默认关闭 | — | 同上 |
| 普通 vmap / 候选 padding 做跨身份合批 | 改变 CPU 浮点位，已排除（后来以堆叠权重的合并推理实现） | — | [`history-host-throughput-round3-2026-09-27.md`](../../docs/reports/history-host-throughput-round3-2026-09-27.md) |
| 快照间隔 8（branch-s8） | 策略调用减半，再快 13%（3,761 决策/秒）；属第 3 档，要强度 A/B 证明不伤棋力才值得用，报告认为优先级低于提高学习速度 | 3 | `history-actor-ranks-2026-09-28.md` |
| 桌数探测（大模型，训练初期） | 256 / 1,024 / 2,048 桌：4,120 / 5,118 / 5,115 决策/秒；2,048 桌显存 23 GB，接近上限 | 3（改配置） | `history-overnight-large-2026-09-28.md` |
| MPS（MLP 时代，4 个 actor） | 在 pod 内能启动，效果在噪声内 | — | `perf-gpu-2026-09-24b.md` |

说明：branch-s8 的强度对比没有写进仓库里的报告，所以这里不给数字；上表只写报告里能找到的部分。同进程流水线、MPS 和 pinned staging 的结论来自旧 MLP 训练器，对历史 Transformer 训练器没有重测。

## 5. 已实现、但尚未在 CUDA 上测速（这些是开放项，不是新点子）

来源：[`training-stack-refactor-2026-09-29.md`](../../docs/reports/training-stack-refactor-2026-09-29.md)、[`docs/TRAINING.md`](../../docs/TRAINING.md) "Speed options after the refactor"。都在 `main` 上，默认关闭，可用 `--resume-set` 切换。

| 选项 | 档位 | 内容 | CPU 结果 |
|---|---|---|---|
| 默认路径重构 | 1 | `PublicStream` 连续数组；PPO 单循环；每个 minibatch 一次打包上传；损失和梯度范数一次传回（此前每 minibatch 约 60 次主机同步） | CPU 上中性（146.4 vs 148.9 s / 10 更新） |
| `--rollout-paged-cache` | CPU 上 1 | 所有身份的公开 K/V 放在一个共享页池；每次编码固定次数的张量操作；替代 `--rollout-triton-cache` | CPU 无收益（CPU 受算力限制） |
| `--batch-snapshot-encoder` | 2（仅快照座位） | 所有快照身份的新公开事件每步一次合并编码（主线上原来约 11 次/步） | 同上；kernel 启动代理计数 1,926 → 466（4.1 倍，但对比的是未融合的逐条目路径） |
| `--learner-length-groups N` | 2 | 学习端按长度分组编码，只编码到各行需要的位置 | 学习时间 2.4 倍（4 组）；真实 minibatch 只有 47% 线性计算、31% attention 计算落在真实 token 上 |
| `--rollout-page-span` | 2 | 注意力键按 64-token 页补齐，而非 2 的幂 | 采集 1.42 倍 |
| 四项合计 | — | — | 87.0 s vs 146.4 s / 10 更新（1.71 倍），M4 Pro CPU，单次运行 |

- 已知风险：页池按 25% 增长、从不收缩；学习期间学习者页仍被保留（每 rank 数百 MB 量级），增长时短暂持有新旧两池（约 2.25 倍）。在已到 23.6/24 GB 的卡上可能显存不足（同报告）。
- CUDA 数值：页缓存给 SDPA 的是跨步视图，在 CUDA 上可能只是第 2 档，要等 CUDA 测试通过。
- 报告给出的 GPU 验收步骤：先跑 `pytest -q tests/test_history_paged_cache.py tests/test_history_length_groups.py tests/test_history_arms_bench.py tests/test_history_snapshot_batch.py`，再用 `python -m bench.history_arms` 从主线 checkpoint 做进程内 ABBA（约 100 次更新，约 $0.30）。

**分支 `XiyaoWang0519/find-training-speedups`**（本地和 `origin` 上都存在；`git log` 前几条：`cb82b93` 合并快照编码调用 cache-wide Triton kernel、`626595b` 一次调用编码所有快照身份、`cdeebe4` cProfile 钩子、`e76754b` PublicStream 连续存储、`573a745` Triton KV 单一 cache-wide 指针表、`df23d7e` 学习端每 minibatch 一次设备读取）。按 refactor 报告：它独立实现了连续公开流、学习端同步合并、基于 Triton 缓存的合并快照编码、稳定 Triton 指针表和一个 **CUDA MPS** 臂，带自己的 GPU A/B kit；其中 Triton 指针表、cProfile 钩子和学习端改动已并入，基于 Triton 的 `--batch-snapshot-encode` 与 `--batch-snapshot-encoder` 竞争，留在该分支，等 GPU A/B 选定缓存方案。两分支在 `history_model.py`、`history_ppo.py`、`history_rollout.py`、`history_snapshot_batch.py` 冲突，只应合并一套。

## 6. 代码地图（均已在 `main` 8e197db 上确认存在）

| 文件 | 作用 |
|---|---|
| [`train/history_rollout.py`](../../train/history_rollout.py) | 公开事件存储 `MatchEventStore`、序列 rollout buffer 和采集器 `HistoryCollector`（每个向量步的各阶段） |
| [`train/history_inference.py`](../../train/history_inference.py) | 冻结采集区间内按策略、按比赛的公开 KV 缓存；权重变化时整体失效并从原始事件重建 |
| [`train/history_triton_cache.py`](../../train/history_triton_cache.py) | 可选的 uint32 Triton kernel，合并 KV 写入和打包，不改任何浮点运算 |
| [`train/history_paged_cache.py`](../../train/history_paged_cache.py) | 可选的页式公开 KV 缓存（`rollout_paged_cache`），尚未在 CUDA 上测速 |
| [`train/history_snapshot_batch.py`](../../train/history_snapshot_batch.py) | 快照座位的合并 actor 前向（堆叠头权重）与合并快照编码器 |
| [`train/history_ppo.py`](../../train/history_ppo.py) | 冷启动 PPO 训练器：配置、`learn()`、checkpoint、`--resume-set` |
| [`train/history_ddp.py`](../../train/history_ddp.py) | 多 rank 数据并行采集、全局 minibatch、`--world-size` 启动器 |
| [`train/history_model.py`](../../train/history_model.py) | `PublicStream` 与历史 Transformer（公开编码器、决策查询、候选打分、critic、辅助 response 头） |
| [`train/history_attention.py`](../../train/history_attention.py) | 用现有层权重的因果 SDPA |
| [`train/history_cuda_graphs.py`](../../train/history_cuda_graphs.py) | 私有决策状态的有预算 FP32 CUDA Graph |
| [`train/history_transfers.py`](../../train/history_transfers.py) | 混合 dtype 小传输的无损打包，一次 H2D |
| [`train/history_population.py`](../../train/history_population.py) | 本谱系快照池，每场比赛抽样一次 |
| [`bench/history_arms.py`](../../bench/history_arms.py) | 从 checkpoint 做进程内多臂 A/B，不写入谱系 |
| [`cpp/src/env.cpp`](../../cpp/src/env.cpp)、[`cpp/src/encoder.cpp`](../../cpp/src/encoder.cpp)、[`cpp/src/movegen.cpp`](../../cpp/src/movegen.cpp) | C++ 向量环境、1,849 维观测编码、合法动作生成（规则标准见 [`docs/RULES.md`](../../docs/RULES.md)） |

## 7. 本目录中可直接分析的数据

- `kits/collect-profile/`：剖析 kit；`profile-summary.md|json`，逐 rank `download/results/segments/profile-diag/metrics.jsonl`（字段 `collection_phase_seconds`、`collection_group_phase_seconds`、`collection_policy_call_rows` 等）。
- `kits/snapshot-batching/`：合并推理 A/B；`ab-summary.md|json`，`memory-analysis/`（逐 rank、逐更新的分配器计数）。
- `kits/trim-check/`：显存回收检查；`download/results/trim-summary.md`，`attempt-1/` 是第一次门禁失败的记录。
- `kits/longrun-batched/`：8 小时主线；`longrun-summary.md|json`，`download/results/gpu-samples.jsonl`，4 个 rank 的 `metrics.jsonl.gz`（每行一次更新，79 个字段，含 `collect_seconds`、`learn_seconds`、`cuda_*`、`mean_prefix`、`max_prefix`、`collection_policy_batches`）。

## 8. 请评审回答的问题

1. **每次更新剩下的约 16–18 秒**（采集 8.9–10.6 s、学习 7.0–8.0 s）里，哪些部分可以去掉？采集侧大头是公开 KV 缓存编码，学习侧是完整前缀的重算。
2. **进程/GPU 布局对不对**：4 个 rank 共用一块 24 GB 卡、利用率约 90% 但功耗只有约 207–232 W。更多 rank、更少 rank、MPS、或别的划分，哪个值得测？显存上限怎么处理？
3. **先测什么**：第 5 节未测的选项（页缓存、合并快照编码、长度分组、页跨度，以及分支上的 Triton 方案）应按什么顺序、用什么指标在 GPU 上验证？
4. 对你的每条提议，请写明：属于第几档；如何验证（逐位测试、数值容差门禁，还是需要 A/B 加独立强度评估）；预期节省的是 kernel 启动、主机同步、FLOP 还是显存。
5. 第 2 节的禁止项是硬约束：请不要提议降低精度、截断/窗口化历史、减少样本或削弱训练信号。
