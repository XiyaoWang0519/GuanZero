# 本地吞吐优化：第二、三轮与 CUDA 接续

日期：2026-09-27。工作区：`GuanZero-throughput`，分支 `throughput-2026-09-27`。
基线：`42f70a49032273827a5f36bc71d7d72f765068d4`。

## 当前结论

已继续完成 KV/memory 整理、整步数据搬运和同策略 private attention 合批。
最终 CPU 对比中，默认优化版的 current/recent 吞吐提高约 **10.0%/11.5%**；
启用 private attention 合批后提高约 **18.9%/17.0%**。
wide 的对应观测为 **16.1%/29.8%**，但测试后半段主机明显变慢，不能把该数字当作稳定收益。
18 次运行与基线的完整重放摘要全部一致。

完整 Python 回归为 **901 通过、123 跳过**（33.14 秒）。跳过项包含 CUDA 与可选依赖。
跨策略实验原型另有 **29 项通过**，没有接入正式收集器。

**下一步需要 Vast.ai 的 NVIDIA GPU。** 本机没有 CUDA，无法验证批量计算是否选用不同
CUDA 内核、实际同步成本，以及 CUDA Graph 的逐位等价和收益。
每次决策主机开销降低 5–10 倍的目标仍未完成。以下都是冻结策略采样诊断，
没有优化器更新，也不是正式训练吞吐或棋力结果。

## 已完成的代码

第一轮的候选索引复用、位置表缓存与 cache prune 优化保留；本次新增：

- `train/history_inference.py`：KV 和 memory 的未使用尾部初始化为零，容量足够时直接
  stack 有效视图，减少逐 stream 的完整重写；singleton KV 使用相同形状/stride 的视图。
  输出 memory 仍拥有独立存储，避免后续追加改变先前结果。容量不足时走原填充路径。
  参数签名直接遍历模块与参数，保持共享对象去重、版本变化、替换及设备/dtype 失效检测。
- `train/history_transfers.py`：按字节打包混合 dtype，保持原始位模式与对齐。
  CUDA 使用一次传输再还原各字段视图；CPU 保留直接视图路径。
  公共历史追加字段也使用同一工具。当前传输是 blocking，尚未采用 pinned/异步缓冲。
- `train/history_rollout.py`：先准备所有策略身份的本步输入，一次上传决策字段；
  各策略保持原有权重、缓存和 sorted identity 抽样顺序；最后一次下载所有策略的决策结果。
  熵归约与 Python 累加顺序不变。公共历史追加仍按身份执行，并非整步只有一次 H2D。
- `train/history_model.py`：同一策略的一行一 stream 布局可以合并 private SDPA。
  q/out 投影使用保留原单行 GEMM 形状的 `baddbmm`，CPU 上保持逐位一致。
  普通批量 `Linear` 会改变浮点结果，未采用。
- `train/history_ppo.py`：新增 `--rollout-batched-attention`，配置/manifest/种群恢复兼容。
  此开关默认关闭，等待 CUDA 验收。
- `bench/history_host.py`：支持 CPU/CUDA、CUDA RNG 摘要、显存峰值、设备信息和同步计时。
  不构造 critic 或 optimizer，每块释放诊断 buffer；基线和候选使用相同工具。

完整公共历史、FP32、规范候选、奖励、PPO/GAE 和种群语义保留。
主仓库代码与评估进程未修改。C++ 引擎未修改。

## 最终 CPU 基准

macOS 27 arm64、14 个逻辑 CPU、PyTorch 2.14.0；PyTorch/引擎各 1 线程。
32 env，width 64，2 层，4 头，auxiliary response、完整历史、SDPA + KV。
快照使用不同随机权重并冻结，席位身份使用生产种群分配逻辑。

每次 8 块 × 64 步，前两块暖机，后六块共 12,288 决策计时。
每场景顺序为 **A–B–C–C–B–A**：A 基线，B 默认优化，C 再开 private attention 合批。
每格是该版本两次独立运行吞吐的中位数。计时关闭 profile/cProfile。
最终计时期间代码冻结，本任务没有并发测试或探针；系统其他负载未受控。

| 场景 | A 决策/s | B 决策/s | C 决策/s | B / A | C / A |
|---|---:|---:|---:|---:|---:|
| current | 3203 | 3523 | 3809 | 1.100× | 1.189× |
| recent | 2195 | 2449 | 2569 | 1.115× | 1.170× |
| wide | 1795 | 2084 | 2330 | 1.161× | 1.298× |

wide 的 A 从 1956 降到 1633 决策/s，B 从 2293 降到 1874，C 从 2419 降到 2240。
顺序交错仍不能消除这种非平稳影响，所以 wide 只报告观测值。
前一轮中 C 对 A 的 current/recent/wide 为 1.129×/1.161×/1.211×；
当时 q/out 仍逐行执行，且有短暂本地验证活动，不与最终数据混算。

平均策略批大小分别为 30.77、6.38、3.66；最大公共前缀为 1015、989、955。
private attention 的逐位测试另覆盖长度 2403 的合成编码；完整 KV 历史的
2403 个公共事件通过既有的 cache/dense 容差检验。两者均未用于这张吞吐表。

### 调用数量证据

同一 wide 重放，分别单独运行 cProfile。摘要一致，计数覆盖六个计时块。

| 调用 | 原始 A | 最终 C |
|---|---:|---:|
| SDPA | 18,584 | 9,937 |
| 显式 Tensor.copy_ | 125,706 | 69,778 |
| linear | 89,593 | 69,497 |
| baddbmm | 0 | 2,802 |
| torch.stack | 0 | 8,593 |
| sinusoidal | 3,346 | 195 |
| 整理所有身份缓存 | 384 | 1 |

stack 内部也会复制数据，以上不代表总内存流量下降了相同比例，更不是 CUDA launch 计数。
最终 profile 中约 67.7% 时间在公共 cache/整理，24.7% 在 actor/采样；
SDPA 本身约 2.56 秒。引擎 pending + step 约 0.038 秒。
因此，把引擎改为多进程尚缺少当前瓶颈证据；IPC 的净收益需要单独测量。

## 等价验证

18 次未剖析运行，以及最终 wide profile，均按场景匹配原始基线摘要：

| 场景 | 完整重放 SHA-256 |
|---|---|
| current | `601aaa2c4daca2a19dd2684bd7e0f093eea5f37bbb1eae25f69a4230f203cbd1` |
| recent | `9d268c2e964a744ba15285728166139734e204c6a199c8a0d3cc61815174a6f7` |
| wide | `0424fea16eabe07e43105e8549afef8c0d0c6e078813029ca3671ce6231eb44c` |

摘要覆盖每步动作、compact buffer、logp/behaviour_logp、奖励/轨迹、完整公共 stream、
身份、抽样器/全局/种群 RNG、非计时统计和初始权重；结束时检查权重未变化。
这证明上述 CPU 轨迹的等价，不能替代 CUDA 同设备 A/B/C 验证。

回归覆盖三种 response mode、dense/KV、混合身份、epsilon/temperature、长前缀、
缓存扩容/清理/参数替换，以及混合 dtype、浮点特殊位、非连续/标量/空数组传输。
完整测试在允许共享内存和本地进程通信的环境运行，日志：
`.work/history-host-round3/full-tests.log`。

## 已探索但尚未集成的方向

- **跨身份合批**：普通 vmap 和候选 padding 会改变 CPU 浮点位，已排除。
  不同权重、严格相同形状的 grouped baddbmm 原型通过 29 项测试，覆盖宽 32/64/128、
  三种 response mode、public/private/head/KV 和 RNG；尚无完整 collector/CUDA 验收。
- **真实形状限制**：wide 轨迹的 64 个计时向量步中，private 530 次调用可分为 241 组
  （2.20× 理论调用缩减），KV 为 533→416（1.28×），head 为 530→429（1.24×）。
  尚未计入打包成本；严格同形状跨身份合批不能单独支撑总体 5–10×。
- **引擎分片**：8 env 分 3 个不等长 shard，在含 uint64 溢出边界的两个种子上，
  8,192 决策、11,003 公共事件、107 轮奖励和一次 match 结束逐位相同。
  仅验证非 styled 引擎的 seed 派生，不包含 IPC、神经 actor 或吞吐收益。

## CUDA 接续顺序

1. Vast Secure Cloud 固定 machine ID；先确认实际 cgroup/cpuset 可用 CPU ≥ 16。
   保存 GPU、driver、PyTorch/CUDA、CPU 配额、引擎二进制和 source hashes。
   本报告未查询实时库存/报价，也未开机；旧交接报价不能当作当前价格。
2. 同容器编译一次 Linux `gd`，A/B/C 共享该二进制。FP32、关闭 TF32。
   A 使用基线代码与同一份新 benchmark；B/C 使用冻结候选。
3. 先运行 CUDA 等价测试，再短跑两块验证完整摘要。C 有任何逐位差异则保持关闭，
   定位具体算子；不以 allclose 代替本阶段 exact gate。
4. 通过后，按同机 A–B–C–C–B–A 测 current/recent/wide，剖析单独运行。
   记录 CPU 提交、CUDA launch、kernel、H2D/D2H、同步和显存。
5. 按实际瓶颈优先实现确定性的 private/head CUDA Graph，采样保留在原外层顺序。
   使用固定地址、严格形状、有界图缓存；测试多次 replay、权重更新和身份替换。
   之后评估跨身份 private 合批、历史追加合批及 pinned staging。
6. 最后测完整 collect+learn 的固定工作量；优化训练循环前继续保留版本 barrier、
   固定采样步数与中央 RNG。租机前确定费用上限与独立自动销毁，case 设置超时。

基准命令（从对应源码目录运行；C 另加 `--batched-private-attention`）：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:oracle:. \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m bench.history_host --device cuda --case wide \
  --envs 32 --width 64 --layers 2 --heads 4 --response-mode auxiliary \
  --torch-threads 1 --engine-threads 1 --steps 64 --chunks 8 --warmup 2 \
  --output /workspace/results/wide-A-0
```

## 冻结产物与复现

- 汇总：[history-host-throughput-round3-2026-09-27.json](history-host-throughput-round3-2026-09-27.json)。
- 最终原始结果、脚本、日志和 profile：`.work/history-host-round3/`。
- 上一轮：`.work/history-host-round2/`；[第一轮报告](history-host-throughput-2026-09-27.md)保留历史数据。
- GPU 包：`.work/history-host-round3/gpu-kit/`。包含基线、候选、实验原型和 manifest；
  候选包不含 Mac 扩展或密钥。原型未成为生产代码。
- 候选 source SHA-256：`1a7dc3d9dabc78a058c1847c0d0ef2ce8847636aa2a83d51a10893f04d269f25`。
- 引擎源码 SHA-256：`606ff1e735823cefb9968a974991bfc5ca325c2ef1cb8ea5a5540f174ce84096`。
- 本机扩展 SHA-256：`f7b7af5a34fd4c2e0b5c3c2cc7ed53bec31f3149c477daf63b81ebdfb85911fe`。

当前改动尚未提交或合并。Vast 生命周期脚本、CUDA Graph 和多 actor 收集器尚未实现。
停止本轮的具体依赖是 CUDA 实机验收与剖析，不能据此宣称所有优化空间已耗尽。
