# 256 宽 × 4 层：完整 PPO 更新吞吐与显存（2026-10-09）

状态：完成。工程短测，没有进行棋力评估或启动正式扩容实验，未修改 Velocity 主线。

## 结果与判断

在同一张 L40S、相同训练配置上，128→256 宽使完整更新平均耗时
**6.711→9.362 秒（+39.5%）**，全局环境决策吞吐
**4,882→3,500/秒（−28.3%）**。
两个种子的更新耗时增幅分别为 +42.7%、+36.4%；这是两次工程重复，不是正式训练多种子的棋力结论。

单卡 4 ranks × 128 桌时，256 宽在 24 GB RTX 4090 上第 16 次更新 OOM；
在 L40S 上两个种子各完成 40 次更新，但最高整卡显存采样达到 **44.16 GiB**。
128 宽在 L40S 上最高为 **24.83 GiB**。显存中位数分别约 13.18、24.77 GiB。

吞吐成本值得继续做受控棋力实验，但当前单卡并发的显存余量很小。
正式实验应先验证较低的每卡并发，或单独测试内存工程优化，同时保持全局训练批量和桌数的对照口径。
参数变大并不证明棋力或样本效率提高；扩容实验仍须同时比较等决策量和等 GPU 时间。

## 配置、计时与原始统计

- 源码：`81cd8e5746f9aff3b60fb727e0dafea0fe1915a3`，干净源树；源码文件集合摘要
  `b512ada2c1f0d521db44cd3cc2811dff0da7988f6dd85c873ea0092d140b396c`。使用项目 `infra.history_artifacts.pack_source` 打包，解包和逐文件身份校验通过。
- 配方从本地 m4 checkpoint 的 config 提取，只读取配置；**没有加载其权重、优化器或人口池**。
  两种宽度分别随机初始化 actor、critic 和辅助头，从各自 Transformer 自博弈建立快照池。
- 模型：4 层、8 heads、全历史；唯一架构变量为 width=128/256。
  action_width=128、fusion_width=128、critic_width=256、critic_layers=3 均保持不变。
  actor 含 response/辅助头为 1,509,870→5,109,614 参数（3.384×）；critic 均为 646,913 参数。
- 4 ranks 共用一张 GPU，每 rank 128 桌、64 步/更新；全局每次更新 32,768 决策。
  PPO 2 epochs，lr=3e-4、entropy=0.03、epsilon=0、每 rank minibatch_matches=8，global minibatch 开启。
  近期 4 快照、snapshot_updates=2、archive_every=0；response auxiliary；next/belief/outcome，aux_coef=0.1。
- 使用当前 paged KV、merged snapshot encoder、learner cache prefill、length_groups=2、
  causal SDPA、batched/wide private attention 和 merged snapshot policies；trim 开启；
  private graphs、Triton cache、cProfile 和同步阶段 profiler 关闭。精确配置见各组 `bench-config.json`。
- 顺序 **128(seed 1)→256(seed 1)→256(seed 2)→128(seed 2)**；种子 2026100901/02。
  每组 40 次真实更新，前 12 次预热，后 28 次计时，总计每种宽度 56 次计时更新。
- 墙钟计时包围 `HistoryDDPTrainer.update()`，前后 CUDA synchronize、rank barrier，并取各 rank 的最大耗时。
  包含采集、PPO 学习、快照更新、日志和既有参数同步检查；不包含初始化、构建和 checkpoint 保存。
  本短测未保存训练 checkpoint。吞吐分子使用 global_step_decisions，不能拿仅以 collect 时间为分母的
  global_decisions_per_sec 代替完整更新吞吐。
- 两种宽度产生各自的真实轨迹，不是固定同一批 replay。两组种子的平均前缀接近，但不宣称轨迹或 PPO rows 完全一致。
- 显存为 1 秒一次的 nvidia-smi **整卡**读数，中位数取计时窗口，峰值覆盖整个 arm。
  采样可能漏掉更短的尖峰。nvidia-smi 总量为 46,068 MiB：最大采样时只余约 851 MiB（同口径）。
  CUDA device properties 报告 44.39 GiB；不同计数器不能混用来推算张量可分配余量。

| 组 | 种子 | 宽度 | 秒/完整更新 | 决策/秒 | 显存中位 GiB | 整卡采样峰值 GiB | 平均前缀 | 最大前缀 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| a128 | 2026100901 | 128 | 6.592 | 4,971 | 13.00 | 21.91 | 713.3 | 2504 |
| b256 | 2026100901 | 256 | 9.407 | 3,483 | 24.75 | 41.93 | 719.2 | 2708 |
| c256 | 2026100902 | 256 | 9.316 | 3,517 | 24.78 | 44.16 | 700.3 | 2921 |
| d128 | 2026100902 | 128 | 6.830 | 4,797 | 13.36 | 24.83 | 704.8 | 2656 |

L40S 主机为 Xeon Gold 6430，实测 cgroup CPU 配额 30.72 核，usable=30；
PyTorch 2.11.0+cu128，CUDA 12.8，FP32、TF32 关闭。引擎 84 test cases 通过；
每组实际进行了 collect/backward/optimizer update 和每次更新的既有 rank 参数同步检查。
160 条 rank-0 scalar timing rows 未发现 NaN/Inf；这不是所有 tensor/所有 rank 的完整数值审计。

## 失败记录、费用与生命周期

1. 首次 4090 启动遗漏 source-identity.json，在更新前退出。已保留日志，补齐并用项目打包器独立验证后重试。
2. 第二次 4090：128 宽完成 40 次更新，256 宽完成 15 次，在第 16 次采集阶段的
   paged KV cache `_append` 打包时 OOM（申请 352 MiB）。失败后未继续使用残存进程；结果保存并销毁实例。
   该组只有 3 次预热后的扩容更新，**不作为完整吞吐结论**；也不能据此说 256 模型在所有 24 GB 配置上都不可运行。
3. L40S：四组各 40 次完整更新，exit code=0；31 个结果文件在销毁前完成大小/SHA256 校验，下载后再次复核通过。

独立本地 guard 已验证，单次租用最长 45 分钟，工作负载 timeout 在 40 分钟以内；
本次总预算上限 $1.25。4090 两次报价约 $0.371/$0.404 每小时，L40S $0.607 每小时。
Vast 初始可用 credit 约 $5.535，销毁后再次读取约 **$5.228**，减少约 $0.307；
余额变化包含平台结算时差，不能替代最终账单。所有本次实例均已销毁，账户实例列表复核为空。

## 产物与局限

- `.work/capacity-throughput-2026-10-09/`：`benchmark.py`、`config.json`、`source.json`、
  `lifecycle.py`、`verify_outputs.py`、`analyze.py`、`summary.json`，以及输出校验、销毁和余额回执。
- `download/`：各组 `timings.jsonl`、`bench-config.json`、manifest、整卡 `gpu.jsonl`、host/hardware facts、console 和引擎测试日志。
- `attempt1/`、`attempt2-4090/` 保留失败和 OOM 记录，不覆盖负结果。

`.work/` 是忽略的本地证据，不是可移植依赖。本结论仅适用于此单卡 4-rank 新初始化工作负载。
虽然平均历史约 700 tokens、最大接近 3,000，仍不能保证成熟策略、长时间运行或更长历史不 OOM。
Velocity 的 8 ranks / 多卡、不同每卡并发、GPU 共享/调度以及 checkpoint 开销需另测，
不能把这里 −28.3% 当作生产训练的精确降速。没有新的棋力、历史利用或主线容量瓶颈结论。
