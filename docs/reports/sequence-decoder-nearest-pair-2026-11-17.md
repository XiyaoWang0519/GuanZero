# 多层序列解码器、记牌器输入裁剪与最近邻对对手池实现及预检（2026-11-17）

状态：**已实现并通过本地与 Velocity 计算节点 CPU 预检；双卡 L40S 训练作业（`22040`）已入队等待资源；GPU 实测吞吐与对局牌力尚未测量。** 暂停中的原主线目录与检查点保持只读未动。

## 1. 范围与动机

针对 11 月 12–15 日诊断定位的三项结构与种群问题（见 [根因诊断](history-root-cause-2026-11-12.md)、[读取机制对照](history-reader-mechanism-2026-11-14.md)、[filtered 池诊断](filtered-history-diagnosis-2026-11-13.md)）：

1. **私有读取器饱和与单步线性平均瓶颈**：原单层无归一化 cross-attention 的 Q/K 范数膨胀导致权重近似 one-hot、梯度消失，且单槽线性平均冲淡稀疏事件。新增可选多层 Pre-LN `PrivateCrossDecoder`（[train/history_decoder.py](../../train/history_decoder.py)），包含逐 head `QK-Norm`、多槽位私有查询（`--private-query-slots`）、Pre-LN ReLU FFN，以及基于查询投影的相对 `(rel_seat, round_diff, step_diff)` 对数注意力偏置。
2. **`obs` 历史汇总捷径剥离与记牌器保留**：原 `1849` 维 `obs`（[cpp/include/gd/encoder.h](../../cpp/include/gd/encoder.h)）内含无序累计已出牌 `Played`、最近一步 `LastAction`、当前桌面快照 `Trick` 及进贡汇总，使策略无需读取 `PublicStream` 即可决策。新增 `--private-obs-mode`：
   - `card_counter`（本次投产配置）：永久保留 **382 维**私有手牌、基本状态与实时记牌器余牌信息（`OwnHand [0:108]`、`Unseen [108:216]`、`CardsLeft [648:732]`、`FinishStatus [732:748]`、`Levels [748:787]`、`WildHeld [787:790]`、`WildUnseen [790:793]`、`WildFlags [793:805]`、`Phase+Roles [1430:1439]`），将全部 **1,467 维**动作历史汇总（`Played [216:648]`、`Trick [805:968]`、`LastAction [968:1430]`、`Tribute+KnownHoldings [1439:1849]`）按 `obs_mask_prob` 在 `--obs-mask-warmup-updates` 内从 `0.0` 线性退火置零至 `1.0`；`SequenceRolloutBuffer` 按决策行保存采样时的 `obs_mask_prob`，保证跨 update 结转未完局样本重算 `ratio == 1.0`；`HistoryCritic` 保持读取完整特权 `obs + hidden`。
   - `private_only`：仅保留 171 维严格私有与级数阶段信息（遮挡 1,678 维）。
   - `full`（默认）：保持原路径逐位不变。
3. **对手池滑窗坍缩与重启探针漂移修复**：原线上 `strength_behavior` 池满时无条件淘汰 `oldest`，在 `u234993` 保存点实测 `archive_replacements = 2460, archive_rejects = 0`，16 个 archive 快照坍缩在最后 `480` 个 update 内，且每次重启从新 rollout buffer 重抽探针。在 [train/history_population.py](../../train/history_population.py) 中新增 `--population-archive-filter nearest_pair`：
   - 先按 `--population-archive-max-age-updates` 淘汰超龄旧快照；
   - 池满时在 $K+1$ 个候选中合并两两策略 TV 最小的一对（淘汰其中 `update` 较老者），保护早期与中期相异流派；
   - `build_behavior_probe` 用独立固定种子环境按 `(round == 0 vs round >= 1) × (早/中/残局手牌张数)` 六层分层抽取非强制决策，并序列化进 `population.state_dict()["behavior_probe"]`，跨 `--resume` 与多进程 DDP 保持探针恒定；`--init-checkpoint` 继承快照时将 `metadata["update"]` 按 `- source_update` 重定基准至 `<= 0`（保留 `source_update`）。
4. **高速推理与无损迁移衔接**：
   - `BatchedHistoryCache`、`ForkedHistoryCache`、`PagedHistoryCache` 及合并快照推理/编码（[train/history_snapshot_batch.py](../../train/history_snapshot_batch.py) `--batch-snapshot-policies --batch-snapshot-encoder`）全部支持 `PrivateCrossDecoder` 与各快照独立 `obs_mask_prob`；`NumpyHistoryActor` 对非 `full` / 多层解码配置显式报错拒绝；`cpp/` 零改动。
   - [train/history_sequence_migrate.py](../../train/history_sequence_migrate.py) 支持校准旧 `private_reader` 输出均值、近恒等初始化 `PrivateCrossDecoder`、按参数名迁移 Adam 一/二阶动量，并转换驻留 `population` 快照。

## 2. 源码、检查点身份与验证结果

| 项目 | 值 |
|---|---|
| 基线提交 | `8e7851088f213dbba0e34e6e927eca4f5bafb32f`（投产打包见下） |
| 引擎摘要 `engine_digest` | `248f048a3a7e56d1feba574fb4358216f5f393503b3eb8629d6415de6e6c31cb` |
| 投产源码包 SHA-256（`gz-seq-cc-np-2026-11-17.tar.gz`，393 文件） | `ea279e0d850efda4f5b512499906fb7c250b7c0f3c53bf4dd703dd3f9b8796f5` |
| 投产源码清单 `source_sha256` | `16da8d29efde60ddfda922d1912e6aa9a27f1ac3e3d5461a5a7992b792936ef3` |
| 只读源检查点 `u234993.pt` SHA-256 | `68df2615d2cc697afe1d61e633d8e867abc01c7e1c795a1065cef48916b3af7b` |
| 迁移后初始检查点 `u234993-seq-cc-np.pt` SHA-256 | `3931c4ca835fdba5b462e615b0197181ed5c1afc8f8abb40f785b0519cc6c613` |

### 本地测试与基线校验

- C++ 单元测试 `./build/cpp/tests/gd_tests`：`87 test cases, 0 failed assertions`。
- Python 全量套件 `pytest -q tests`：`1,764+ passed, 375 skipped`（macOS CPU 主机跳过 CUDA 用例）；新增与扩展专项套件（`test_history_decoder.py`、`test_history_sequence_decoder.py`、`test_history_sequence_migrate.py`、`test_history_population_filter.py`、`test_history_population.py`、`test_history_snapshot_batch.py`）覆盖切片边界、4 路分发一致性（`< 1e-5`）、合并快照推理/编码一致性（`< 2e-5`）、跨 update 结转混合 `obs_mask_prob` 的 `ratio == 1.0`（实测 `ratio_dev = 3.004e-5`）、`--init-checkpoint` 重定基准及 `--resume` 零探针重采样。
- 默认旧路径字节级回归：`bench.training_revision --verify --device cpu --case snapshots --updates 2` 通过（`verification: true`）。
- 1,240 次归档步（`archive_every=32, archive_size=16`，对应 `u155776..195424` 跨度）漂移策略对比模拟：
  - `strength_behavior`：`archive_age_span = 480`、`archive_min_pair_tv = 0.0543`、`archive_mean_pair_tv = 0.1292`；
  - `nearest_pair`（`max_age=40000`）：`archive_age_span = 39,616`、`archive_min_pair_tv = 0.1819`、`archive_mean_pair_tv = 0.6763`。

### Velocity 计算节点预检与排队配置

- 计算节点预检与迁移作业 **`22039`（`gz-seq-cc-np-prep`，`max-compute-11`）已完成（`COMPLETED, ExitCode 0:0`，耗时 `00:05:19`）**：
  - `/dev/shm` 本地构建 Linux `_gd_core` 并通过零 `/home` 映射守卫；
  - 节点运行 11 个核心测试套件：**`233 passed, 50 skipped`**；
  - 从只读 `u234993.pt` 迁移生成 `u234993-seq-cc-np.pt`（`decoder_layers=3, query_slots=4, obs_mode=card_counter, archive_filter=nearest_pair, archive_max_age_updates=40000`）：heldout 248 个决策 `tv_mean = 0.002618`、`tv_median = 0.000465`、240 个多候选决策 `greedy_change_share = 0.0`、全部 24 个头 `one_hot_share_max = 0.0`、`private_decoder`（77/77）与 `stream`（48/48）梯度全部有限非零；
  - 预先构建并写入固定种子 `20261117` 的 **128 行分层探针**（8 场比赛、63 个跨局决策、128 个非强制决策）及 16 个继承 archive 快照指纹。
- 双卡长程训练作业 **`22040`（`gz-seq-cc-np`）**：
  - 依赖 `afterok:22039` 已解除，当前处于 `normal` 分区 `PENDING (Resources)`（`gres/gpu:2, 32 CPUs, 160G, 12:00:00`）；
  - 启动首片前先执行单卡 ECC/矩阵乘健康探针与 4 个套件的 CUDA 数值门禁测试，随后以 8 DDP ranks（`num_envs=128, steps_per_update=64, minibatch_matches=8`，`obs_mask_warmup_updates=1500`，全套 paged/merged-snapshot 加速）运行，12 小时切片优雅落盘并自动续投至 `--updates 40000`（输出目录 `~/guanzero/runs/seq-cc-np-u234993/`，支持 `STOP` 标记停机）。

## 3. 限制与待验证项

1. **CUDA 实测待运行**：截至本记录提交时，`22040` 仍在队列等待 2 张 L40S 释放；CUDA 门禁测试、双卡实际采样/学习吞吐及显存占用将在 `22040` 启动首片时验证并记录在案（`cuda-gate.json`、`metrics.jsonl`）。
2. **牌力与序列利用尚待训练后独立评测**：迁移后 `tv_mean = 0.0026` 与模拟多样性指标仅证明初始化平滑性与归档算法性质，不构成牌力提升证据；需在 `obs_mask_prob = 1.0` 过渡完成后（如 `u3000`、`u10000`、`u20000+`）通过固定牌集对 DanLM 配对评测与序列替换干预独立检验。
3. **产物路径**：本地预检与打包回执位于忽略目录 `.work/filtered-pool-smoke-2026-11-17/` 与 `.work/velocity-seq-np-2026-11-17/velocity-launch-receipt.json`；远端套件与检查点位于 `~/guanzero/seq-cc-np-2026-11-17/`，运行输出位于 `~/guanzero/runs/seq-cc-np-u234993/`。
