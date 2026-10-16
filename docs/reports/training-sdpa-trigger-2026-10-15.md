# 主线 SDPA 异常的触发调查（2026-10-15）

状态：只读服务器取证与本地 CPU 张量对照已完成。未修改训练代码、配置或作业，
未提交 GPU 实验。用户要求调查 u74484 附近的修改及具体触发原因。

结论：没有发现该窗口发生部署、训练重启或配置切换。新的数值证据支持
“Q/K 规模随训练增大，使 efficient backward 重建注意力概率时对浮点误差极端敏感”
这一触发假设。它比单纯的形状猜测更具体，但还不是 CUDA 内核内部因果证明。
已经确定的算子错误仍以 [7323/7324 同输入 VJP](training-fixture-backward-2026-10-14.md)
为准；本调查没有取回 u74155 的原始 minibatch，不能重放其首次故障。

## 时点与变更核对

- 从远端 rank 0–7 的 metrics 分别读取 u73500–75200，共每 rank 1,701 条。
  在此窗口，rank 0 首次 update 平均 actor norm >10 为 u74152（10.7685），
  首次 >100 为 u74155（283.2152），首次 >1,000 为 u74616（11,602.3249）。
  这些是归约后的 update 汇总，不能定位最先产生错误的 rank 或 minibatch。
- watcher 的 update-epochs 将 u74155 标在多伦多 10 月 12 日 19:07:23；
  watcher 以约 10 秒轮询，时间不是精确 optimizer-step 时间。
- u74484 是先前报告中一个 250-update 分组的起点，不是首次异常的精确位置。
  u74150–74233 中已有 8 条 norm >10，随后 u74234–74483 又大体正常，
  中位数 0.5227。u74484–74733 中位数 24.3902，136/250 条 >10；
  u74734–74983 中位数 874.7382，250/250 条 >10。
- 作业 6419 在多伦多 10 月 12 日 10:52:26 启动训练，22:32:57 退出；
  u74155 与 u74484 都在这次同一 attempt 内，没有边界重启或 fallback。
  前一作业与本作业的 train_start settings 一致。
- 两个现存 checkpoint u74028 与 u74648 的 source_changes 最后记录均为
  u53810，且该条前后源码 SHA 相同；config_changes 最后一条均为 u20264。
  该结论是 checkpoint 的审计记录，不等于独立证明所有文件历史从未被修改。
- 远端 10 月 8 日旧 manifest 与 10 月 15 日补丁前 manifest 的文件摘要只差
  `train/ckpt.py` 与 `train/history_population.py`，对应恢复 RNG 和 metrics 精简。
  `history_model.py` 补丁前备份 mtime 为 10 月 4 日，population 当前文件 mtime
  为 10 月 7 日，ckpt 当前文件 mtime 为 10 月 8 日；没有对应故障窗口的新修改证据。
- torch/version.py 记录 2.11.0+cu128、CUDA 12.8、git
  `70d99e998b4955e0049d13a98d77ae1b14db1f45`，mtime 为 10 月 5 日。
  未做完整依赖安装历史取证，mtime 不是不可篡改的环境历史。
- population 在提取窗口每 2 次 update 正常写 snapshot；没有观察到新 population
  策略切换。rank 0 learner 最大前缀的分组中位数约 1,550–1,640，没有对应
  norm 爆发的明显聚合长度跳变；不能排除单个未记录的稀有 mask/形状触发条件。

## 同输入 CPU 数值规模对照

使用已保存的隔离 rank 1 fixture，SHA256
`9c3d1018ac52453852536ff5ef497eb1dbcbd553f02e199b30c9570e6c152cd8`。
只替换 actor 权重，保留同一批 obs、历史、mask 和布局；不运行 engine、采样、
optimizer 或训练。CPU PyTorch 2.14.1，2 threads。
fixture obs 的绝对值最大为 1，没有出现原始输入巨大值。

下表为两个 masked 私有 attention 调用中的可见 score 绝对值最大值的较大者，
score 定义为 QKᵀ / sqrt(head_dim)。FP32 点积误差相对相同 Q/K 的 FP64 对照。
这是后来的诊断输入在不同历史模型上的比较，不是各历史时刻实际 minibatch 的观测。

| 权重 | 最大 score 绝对值 | 最大 FP32 点积绝对误差 |
|---|---:|---:|
| m5 / u58528 | 422,904 | 0.10 |
| m6 / u66092 | 889,640 | 0.14 |
| m7 / u73780 | 91,209,703 | 16.55 |
| u74028 | 97,698,004 | 15.47 |
| u74648 | 144,458,433 | 20.86 |
| m8 / u81344 | 217,562,260 | 42.98 |
| u86800 | 135,668,875 | 22.61 |
| 原故障 fixture 权重 | 427,864,870 | 99.96 |

m6 → m7 的 score 规模已增大约两个数量级。m7 当时的正常 update norm
不能据此证明其 attention 数值没有风险；也不能仅凭这个对照认定 m7 已实际
受到同样严重的错误更新。u86800 在 math 修复后仍有大 score，说明 fallback
绕过了错误 backward，但没有自动消除大 score 的模型状态。

## 具体机制假设及验证边界

与集群安装版本一致的
[PyTorch backward 源码](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_backward.h#L1475)
会重新计算 QKᵀ，减去前向保存的 logsumexp，再取 exp 来重建注意力概率。
该源码使用 float accumulator 与 float logsumexp。

当 score 约为 1e8–4e8 时，原本很小的相对点积误差已经对应十几到上百的绝对误差。
如果重算 score 相对保存的 logsumexp 偏高，exp 会将此误差指数放大，
从而出现远大于 1 的错误“概率”，V 梯度也能被放大；这与此前同输入 VJP 中
前向接近、V 梯度严重失真的现象一致。

补充 CPU 算术演示：前向以 FP32 matmul 计算 scores 并保存 FP32 logsumexp；
随后逐 head dimension 顺序累加，重算数学等价的点积，再减保存的 logsumexp。
故障 fixture 两个调用的重建 log-probability 最大为 +48 和 +64，意味着
错误概率可达 exp(48) 与 exp(64)。m6 同演示最大仅 +0.125。
此演示只证明这批实际张量的浮点累加顺序敏感性，**没有模拟或测量 CUDA 内核的
实际累加顺序、实际保存的 LSE、实际重建概率，也没有证明 CUDA 恰好发生上述误差**。

仍需区分两个未解决问题：

1. CUDA 错误 VJP 的具体内部机制：须在同一 CUDA 环境对故障调用捕获 saved LSE /
   重算 scores，或做 Q/K 规模干预的 bounded 同输入 VJP 对照，才能确认此假设。
2. Q/K 为何在 m6–m7 期间增长：本次仅测到 checkpoint 状态变化，没有归因到
   某一损失、优化器状态或早期错误梯度反馈。训练正常地增大幅度与数值错误形成反馈
   都仍可能；不得把其中任何一条写成已证实原因。

本次不提交后续 GPU 作业，不改变 QK normalization、模型架构、超参或主线权重。

## 产物与复核

本地忽略目录 `.work/m8-trigger-2026-10-15/`：

- `read_remote.py`、`remote-evidence.json`：只读服务器提取脚本、各 rank 标量记录、
  epoch、orchestrator、manifest 及文件元信息。
- `cpu_scale.py`、`cpu-scale.json`：CPU 对照脚本与逐调用测量；脚本显式注明演示边界。
- `update-074028.pt`、`update-074648.pt` 与 `checkpoint-sha256.txt`：从远端现有
  checkpoint 只读下载；本地与远端 SHA256 均已核验。
  摘要分别为 `e626130df9e724e09442e35146390aa72b45900167023be7bff00ec9ea2f1fec`、
  `9272b2129603ec499454dddcfa3cd2f57f3b2d76be9ebdce30be91acafce9b26`。

脚本已实际运行；此次没有运行 CUDA 对照、完整 PPO 或新棋力评测。
