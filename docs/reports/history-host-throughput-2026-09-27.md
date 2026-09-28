# 本地历史策略采样优化：第一轮

日期：2026-09-27。工作区：`GuanZero-throughput`，分支 `throughput-2026-09-27`。
基线：`42f70a49032273827a5f36bc71d7d72f765068d4`。

本页保留第一轮历史结果。当前实现、完整回归和 CUDA 接续见
[第二、三轮报告](history-host-throughput-round3-2026-09-27.md)。

## 结论

保留完整公共历史、FP32、全部规范候选、策略身份、采样顺序和 PPO 数据后，
宽池在两轮本地 A–B–B–A 中分别快 **11.5% 和 8.0%**。
三个场景的基线与修改版均有相同的完整重放摘要。
这是第一轮 CPU 冻结采样诊断，尚未达到每次决策主机开销降低 5–10 倍的目标。
CUDA 吞吐和设备上的逐位等价仍须在固定 Vast 机器上验收。

完整 Python 回归最终为 **803 通过、33 跳过**：沙箱内先有 790 通过，
13 个进程通信相关用例因共享内存/回环端口权限失败或报错；取得本地进程权限后，
这 13 个用例全部通过。跳过项包括 CUDA 与可选环境依赖。未修改 C++ 引擎。

## 先核对交接依据

交接指向的主仓库 `comparison.json` 在检查时尚不存在。
本次只读已有 `screen-202609272{1,2,3}*/local/results/*/train/metrics.jsonl`，
按更新编号合并原运行与续跑记录，取最后 512 次更新：

| 种子后两位 | 大模型最近池采样 s/更新 | 大模型宽池采样 s/更新 | 最近池策略批次/更新 | 宽池策略批次/更新 |
|---|---:|---:|---:|---:|
| 21 | 4.863 | 4.937 | 469.4 | 622.1 |
| 22 | 4.627 | 4.679 | 470.7 | 624.4 |
| 23 | 4.597 | 4.606 | 465.0 | 618.2 |

宽池约多三分之一的策略调用，但晚期采样耗时并没有稳定慢 20%。
这些运行有并发、主机与恢复差异，不能从日志中单独识别池宽带来的耗时。
因此，本次同时测 current、recent、wide，使用受控的冻结策略诊断。

## 改动

- `train/history_model.py`：候选所属决策的整数索引由 collector 生成一次并复用；
  一条 stream 对应一个决策时，用已知行切片替代逐 stream 的设备 `nonzero`。
  保持原来的逐 match 注意力形状与浮点运算顺序；学习器和其他布局保留通用路径。
- `train/history_inference.py`：每个 KV cache 复用一个精确容量、宽度、设备与 dtype
  对应的位置编码和整数位置表。容量变化时替换，清理/参数变化时失效，计入缓存字节数。
- `train/history_rollout.py`：仅在席位身份分配变化后整理 KV cache。
  CUDA 采样结果通过字节视图合并取回，保留 int64 动作、FP32 概率和 FP64 熵统计的位模式。
  每个 learner 策略批次从五次主机读取合为一次，snapshot 从两次合为一次；
  CPU 路径直接读取张量，避免多余打包。熵的设备归约与逐步累计顺序保留。
- `bench/history_host.py`：新增无优化器的冻结策略基准，支持真实的身份分配逻辑、
  不同权重的快照、各身份批次统计、分阶段计时、cProfile 和逐块重放摘要。

跨身份合批和多个 actor 进程本轮尚未实现。不同身份仍使用各自的权重和缓存。

## 基准条件与结果

- macOS 27 arm64，14 个逻辑 CPU，PyTorch 2.14.0；PyTorch/引擎各 1 线程。
- FP32，32 个环境，宽 64、2 层、4 头，`response_mode=auxiliary`，完整历史，SDPA + KV。
- 种子 `2026092701`；current 无快照、recent 4 个快照、wide 16 个快照及最近 4 个尾部。
  快照用不同种子初始化并冻结，通过生产 `HistoryPopulation.assignment` 选择席位。
  它们并非训练后对手，不能据此推断实际策略强度或正式训练吞吐。
- 每次运行 8 块 × 64 步 × 32 决策；前 2 块暖机，后 6 块计时（12,288 次决策）。
  每块重建 learner KV cache；原始历史继续增长，诊断 buffer 每块释放。
- 每个场景按 A–B–B–A 串行执行，A 为归档基线，B 为修改版。
  单格数据为该版本两次运行的吞吐中位数；计时关闭 profile/cProfile。
- 第一轮执行期间有短暂的定向测试；第二轮没有并发测试或其他本任务的性能基准。
  系统其他应用仍可能产生负载，所以保留两轮结果，不把小差异当成稳定收益。

| 场景 | 第一轮 A → B（决策/s） | 第一轮加速 | 第二轮 A → B（决策/s） | 第二轮加速 |
|---|---:|---:|---:|---:|
| current | 3266 → 3415 | 1.046× | 3023 → 3037 | 1.005× |
| recent | 2125 → 2172 | 1.022× | 2003 → 2229 | 1.113× |
| wide | 1841 → 2053 | 1.115× | 1909 → 2062 | 1.080× |

current 没有确认稳定收益；recent 收益有明显波动；wide 两轮均有正收益。
最大采样前缀为 current 1015、recent 989、wide 955 个公共事件。
另有既有长历史测试覆盖 2403 个事件，但该长度没有纳入本轮吞吐测量。

| 场景 | 每个策略批次平均决策数 | 384 个计时向量步中的策略批次数 |
|---|---:|---:|
| current | 30.77 | 384 |
| recent | 6.38 | 1862 |
| wide | 3.66 | 3245 |

计时包含动作日志和批次计数；摘要、权重检查和磁盘输出在计时区间之外。
没有 learner 更新，不能把这些数字同交接中的完整训练更新速度直接比较。

单独的 wide cProfile 显示：优化版约 67.5% 的采样时间在公共历史缓存/整理，
24.5% 在 actor 和采样。计时区间仍有 18,584 次 SDPA 调用和 125,706 次张量拷贝；
本轮没有减少这些调用。CPU 上 SDPA 本身耗时约 2.50 秒，占该 profile 约三分之一，
所以不能把缓存路径的总耗时都算作 Python 主机开销。

## 逐位验证

两个版本的两轮计时使用相同 seed 时，分别得到以下摘要；wide 的单独分阶段诊断也匹配：

| 场景 | 完整重放 SHA-256 |
|---|---|
| current | `601aaa2c4daca2a19dd2684bd7e0f093eea5f37bbb1eae25f69a4230f203cbd1` |
| recent | `9d268c2e964a744ba15285728166139734e204c6a199c8a0d3cc61815174a6f7` |
| wide | `0424fea16eabe07e43105e8549afef8c0d0c6e078813029ca3671ce6231eb44c` |

摘要覆盖每步动作、完整 compact buffer（含两种 logp）、轨迹归属/结束与奖励记录、
公共 stream、席位身份、采样器/全局/种群 RNG、非计时统计和初始策略权重。
不包含 KV 内部存储布局和耗时。每次运行还检查结束权重与初始权重相同。

新增测试检查三种 response mode、dense/KV、混合策略、温度/epsilon、
位置缓存扩容/重置/权重和设备变化，以及字节打包的 int64 极值、浮点特殊位、
非连续张量、标量和空数组。CPU 检查通过；CUDA 对应用例已保留，待实机执行。

## 复现和证据

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:oracle:. \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/Users/xiyaowang/Developer/Projects/GuanZero/.work/external/danlm-venv/bin/python \
  -m bench.history_host --case wide --chunks 8 --warmup 2 \
  --output .work/history-host-reproduce
```

`--case` 支持 `current/recent/wide`。加 `--profile --cprofile` 可定位开销，
但其计时不能混入上表。基线由 `git archive 42f70a4` 解包，复制同一份
`bench/history_host.py` 和当前已验证的 `gd` 扩展后，用相同命令运行。
比较 `result.json` 的 `exact_sha256`；若不同，按 `chunks[].exact.parts` 定位。

- 汇总：[history-host-throughput-2026-09-27.json](history-host-throughput-2026-09-27.json)。
- 原始运行、基线归档、ABBA 脚本、profile 和日志：`.work/throughput-2026-09-27/`。
- 引擎源码摘要：`606ff1e735823cefb9968a974991bfc5ca325c2ef1cb8ea5a5540f174ce84096`。
- 引擎扩展摘要：`f7b7af5a34fd4c2e0b5c3c2cc7ed53bec31f3149c477daf63b81ebdfb85911fe`。

## 下一轮

1. 减少每步对完整 KV/memory 的重新打包，保留 stream 重排、扩容、跨 match 淘汰和权重失效契约。
2. 批量执行 private-query 注意力，随后评估不同策略权重的批推理；保持完整候选和原 RNG 消耗。
   只拼接数据再用一个 actor 会混淆身份，不能作为跨身份合批。
3. 在固定 Vast machine ID 上先执行 CUDA 逐位测试，再做逐项开关和同机 A–B–B–A，
   记录 cgroup CPU 配额、稳定阶段、前缀分布、完整 collect+learn 吞吐与显存。
   多 actor 方案需要在这组数据上判断，不能由本机短基准决定。

以上是后续工作；本轮未租用云机器，也未改变主仓库的代码或评估进程。
