# History rollout：RTX 4090 吞吐优化（2026-09-28 UTC）

状态：已完成。GPU 租用 1 小时 56 分 57 秒；800 个结果文件已校验回传，实例已销毁，Vast API 与控制台均确认剩余 0 台。

主要结果：小模型冻结采集在当前策略 / 最近池 / 宽池分别达到原始版本的 **3.48× / 1.58× / 1.46×**；真实 PPO 短循环整轮约 **1.95×**。最终 GPU 测试 **1,192 passed / 11 skipped**。仍有优化空间，尤其是宽池的小批量策略调用；本轮以两小时预算和完整验收为边界。

## 范围与不变量

从 `42f70a49032273827a5f36bc71d7d72f765068d4` 的 worktree 继续优化。主仓库只读。
保留 FP32、完整公开历史、原候选动作与采样顺序、PPO/GAE/奖励/对手池语义和 dense reference。
主矩阵测量冻结策略的采样吞吐，另有独立的真实 PPO 短循环。二者均不能证明棋力提高；短循环也不能证明长训练总耗时同比下降。

## 机器与预算

- Vast.ai Secure Cloud，machine `20082`，instance `53083880`，美国马里兰。
- 单 RTX 4090 24 GB；50 GB 磁盘；创建时报价含磁盘 $0.783148148/h。
- 创建于 2026-09-28 01:59:27 UTC；03:56:25 UTC 确认销毁，早于 03:59:27 UTC 硬截止。
- cgroup 实测 30.6 CPU，向下取整可用 30；不采用 affinity 的 255 核作为配额。
- Python 3.12.14，PyTorch 2.11.0+cu128，CUDA 12.8，Triton 3.6.0，驱动 590.48.01。
- Linux 引擎源码 SHA256：`606ff1e735823cefb9968a974991bfc5ca325c2ef1cb8ea5a5540f174ce84096`。
- Linux 扩展 SHA256：`3e2c9443bef716ce83a07b3daf7deddba8946b4d46373afeb4a9031311ee4834`。
- 运行期间设有独立本地销毁守护、远程停止守护和每两分钟结果回传；销毁后本地同步及销毁守护均已退出。
- API 与 SSH 凭据只保存在忽略目录的受限权限文件；上传包不含凭据、历史训练数据和检查点。

运行包与原始证据：`.work/vast-throughput-2026-09-28/`；回传结果在其 `download/` 子目录。

## 严格回放的数值边界

原始版本的普通 CUDA A/A 回放并非逐位一致。定位到 `segment_log_softmax` 的 CUDA `index_add_`：相同输入重复 100 次，100 次有差异，最大绝对误差 9.536743e-7；启用 PyTorch deterministic algorithms 后 100 次无差异。
因此本报告的严格 A/B 统一启用 deterministic algorithms、关闭 TF32，并设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。这不证明普通非确定性 CUDA 后端逐位一致。
原始失败记录保留在 `download/aa-wide-01/`，确定性 A/A 在 `download/aa-wide-det-02/`，缩减探针在 `download/results/reduction-probe.json`。

## 第一轮主机端对比

32 环境、width 64、2 层、FP32、seed 2026092701；每次 8 个 64 步 chunk，前 2 个 warmup；每个场景按 `ABDEEDBA` 顺序运行。

- A：原始 `42f70a4`。
- B：本地主机优化 round3。
- D：B + 已上传 prefix 长度复用。
- E：D + pinned 异步上传。

24 次完整回放全部匹配；CPU cgroup throttling 增量均为 0。宿主 load 约 34–48，仍有共享宿主噪声的可能。
B 相对 A 的中位吞吐提高约：当前策略 38.3%，最近池 26.3%，宽池 23.7%。
长度复用相对 B 为 -3.6%/+2.1%/-1.4%；pinned 相对 D 为 -3.7%/-3.8%/+1.2%，暂不支持开启 pinned 默认值。
原始矩阵：`download/experiment-round4-01/report.json`；A 与原始 preflight 的独立来源核对：`matrix-baseline-binding.json`。

## CUDA 特有修正与后续实验

CUDA 上把独立投影换成 `baddbmm` 会改变 FP32 末位，导致 81 个 attention 校验失败。修正为 CUDA 保留原来的逐行线性投影，仅合并可逐位匹配的 attention；修正后 81/81 通过。

随后分别测量了私有决策 CUDA Graph、公开历史 CUDA Graph、Triton KV 字节搬运及其组合。所有建图、失效清理成本均单独记录；最终速度以完整 collector 为准。

## 小模型正式结果

round8、32 环境、width64 / layers2 / heads4，64 steps × 8 chunks，前 2 个 warmup；每个场景 `EAQVWWVQAE`，同一块 RTX4090、同一引擎、同一 seed。30 次完整回放全部与原始 A 逐位匹配。

单位为决策/秒，表内为两次独立进程的中位数，包含采集中的建图和学习器 KV 失效。

| 场景 | 原始 A | 主机优化 E | 私有图+attention Q | Q+Triton min4 V | Q+Triton min8 W |
|---|---:|---:|---:|---:|---:|
| 当前策略 | 1030 | 1373 | 2822 | **3587** | 3487 |
| 最近池 | 611 | 783 | **967** | 943 | 898 |
| 宽池 | 441 | 570 | 645 | 625 | 652 |

- 当前策略 V/A 为 **3.48×**；最近池 Q/A 为 **1.58×**；宽池 Q/A 为 **1.46×**。
- 宽池 W/Q 仅高约 1%，不足以把 W 当作稳定赢家；Triton 的 min4/8 也不能由两个进程选出普遍最优值。多策略池优先保留 Q，Triton 单独按目标 workload 测量。
- 正式版受 512 MiB 总图预算限制，recent/wide 的捕获和淘汰开销真实计入计时。宽池仍有明显优化空间。
- 这些是冻结策略采集速度；不能直接把 3.48× 写成整个 PPO 训练的速度提升。原 handoff 的 5–10× 主机开销目标也尚未证明达到。
- 为优先完成全套验收，本轮没有完成最终大模型性能矩阵。较早 round6 的 width128/layers4/heads8 四次短回放匹配，但冷启动和样本量不足，不据此报告稳定大模型提速。

证据：`download/experiment-production-small-08/report.json`。专用真实 PPO/缓存验收为 **17 passed**；另一个覆盖采样、图、缓存、搬运和逐位 attention 的合集为 **399 passed / 3 skipped**，3 个 skip 是 CPU 参数下不适用的 CUDA stream 项。

## 正式实现

新增能力均为显式开关：

- `--rollout-batched-attention`：合并私有 attention 调用，CUDA 独立投影保留原计算顺序。
- `--rollout-private-graphs`：只捕获私有 decision state；候选打分、归一化、采样保持 eager。按 shape/stride/浮点地址对齐缓存，采集期间验证一次完整模型结构，逐次验证数值后端和 CUDA stream。
- `--rollout-graph-budget-mb 512`：所有策略共享保留图预算，每个策略最多 128 MiB/32 个图；过小预算回退。图捕获期间的瞬时内存可能超过保留预算。
- `--rollout-triton-cache --rollout-triton-min-batch N`：用 uint32 搬运合并 KV 写入和打包，不改变任何浮点计算。小批量可回退原路径；显式启用却缺少 Triton 时直接报错。

图保留参数存储的独立引用，允许原地 Adam / load_state_dict 更新后读取新值；替换模型、存储、结构或数值后端会使图失效。公开 KV 仍随学习器更新清空。同步 `collect()` 区间要求参数和模型结构冻结；不是可与训练并发执行的接口。
Triton 指针表保留存储引用并记录使用流，跨流调用在读写缓存前拒绝；clear/prune 释放退休历史的引用。两类缓存均不写入模型 checkpoint。

旧 checkpoint 恢复时采用 checkpoint 中的训练配置，这是原有行为。仅在命令行加新开关不会把旧 checkpoint 自动切换到新路径；新建训练配置或有记录地迁移恢复配置后，须核验运行 manifest 的实际开关。

### 冻结与验收

- 最终生产候选 round8 源码：`aec0c4fed31e034be978f638a7e8c7258b0eadb072a7a6b15cd3dcb7cc4c35d7`。
- 上传源码包：`1422a6b9a999d03565c02d2d48bbfe138d26517062e2f3e5483ec807564a6377`。
- 原始 A 基准包源码：`4cf71f9690b8cd87df0627a7a3d74be3d5b1ee7d037ac03953ed1d02838521d3`。A 与候选各自绑定诊断 harness 摘要，采用一致的回放协议；harness 并非逐字节相同。
- 正式基准 runner：`4c705b2242c330c310f30653ef6e301144e835673bd8f4fd8b4f7e3e820eaa83`。
- round8 本地完整套件：**1,019 passed / 198 skipped，91.41 秒**。CUDA 条件项的证据以远程结果为准。
- 正式 runner 在每次试验前后绑定源码、引擎二进制、GPU UUID、PyTorch/CUDA、确定性/TF32 设置、真实测试 XML 和脚本摘要；失败或不匹配不会生成 accepted summary。
- 计时包括每个测量 chunk 的采集和学习器 KV 失效；建图开销在采集内。哈希、初始化、PPO 学习及最终清理不计入冻结采集速度。

## 已排除或暂未采用的方案

- pinned 总输入上传：同机镜像对比未显示稳定收益，默认关闭。
- CUDA 独立投影 `baddbmm`：逐位校验失败，恢复原 CUDA 投影路径。
- 公开历史 CUDA Graph：已有完整采集实验收益不足，未进入生产代码。
- 实验原型没有全局私有图预算，recent/wide 保留量约 603/721 MiB；正式默认上限 512 MiB。因此原型数字不能直接代替正式代码数字。
- 普通非确定性 CUDA 的逐位一致性、长训练吞吐和棋力提升，均不从冻结策略实验推断。

## 原型筛选结果（非最终生产速度）

V5 原型 36 次串行进程均完成，完整 digest 与原始 A 的同场景 digest 相同。每场景顺序 `ETQRFSSFRQTE`，其余为上述 small/32env 配置。单位：决策/秒，中位数，含学习器 KV 失效。

| 场景 | E eager | T attention | Q 私有图+T | R Q+Triton | F R+公开图 | S F+scatter |
|---|---:|---:|---:|---:|---:|---:|
| 当前策略 | 1455 | 2014 | 3068 | 3696 | 3347 | 3492 |
| 最近池 | 808 | 913 | 919 | 967 | 881 | 947 |
| 宽池 | 573 | 627 | 708 | 662 | 601 | 618 |

最近池 Q 两次为 1022/816，尽管 capture/hit/miss 次数完全相同，仍有明显时间波动；两次进程不足以对小差别作稳定排序。公开图组合没有一致收益，未进入生产实现。
V5 由 collector 明确建立同步采集区间，把完整结构检查从逐次调用移到区间开始，每次调用保留 stream/autocast/数值后端检查。这不允许区间内并发更新参数；未单独完成受控 V4/V5 速度对比。

原始证据在 `download/experiment-v5-full-02/report.json`，独立分析在 `.work/vast-throughput-2026-09-28/v5-analysis.md`。

## 后续仍可优化的方向

最终 source9 使用 Q + Triton min4，在 current/wide 各跑一个普通参考进程和一个剖析进程：32 环境，32 steps × 3 chunks，前 1 个 warmup。两场景完整回放均匹配。证据在 `download/experiment-profile-source9/report.json`；该报告只接受回放一致性，明确不接受性能结论。

同样 2,048 个测量决策，策略/打分头/公开编码调用数为 current 64 次、wide 556 次，相差 **8.69×**；平均策略批量分别为 **30.875 / 3.543**。wide 的 cProfile 累计时间中，公开编码为 1.989 秒，候选打分为 0.471 秒，采集总计 4.679 秒；其中公开 KV `_signature` 检查 556 次、累计 0.193 秒。这些是主机端归因线索，不能当作 GPU 时间占比或预期提速。

候选打分头单独 shape key 的模拟中，观察 3 次再建图时，测量阶段 replay 比例仅为 current **1.56%**、wide **27.88%**；观察 8 次时降为 **0% / 6.65%**。模拟未包含实际捕获成本、图内存预算与生命周期，不能据此预测速度。

据此保留以下顺序，均需要新的实验验证：

1. 测 env32/64/128 和跨策略身份 attention 合批，优先减少宽池碎批；保留原投影形状与采样顺序。环境数改变训练配置，需在各配置内部做严格 A/B。
2. 试公开 KV 参数元数据检查在同步采集区间内复用；保持模型变更、跨 stream 和学习器更新后的失效契约。
3. 补充更长的宽池候选 shape 统计，再决定是否为候选打分头建图；禁止 padding 改变 FP32 GEMM 结果。
4. 只有最终路径 profile 支持时再引入环境 worker；保留采样/PPO 更新屏障，避免混入旧策略数据。

原始分析在 `.work/vast-throughput-2026-09-28/remaining-optimization-review.md`。这些方向说明优化尚未穷尽；这轮结束由租用预算决定。

## 最终验收中的测试隔离修正

完整 GPU 套件首次在四分钟限时内未完成，并暴露了测试隔离与打包问题，不能算作通过。

- 旧 `test_history_ddp.py` 在 finally 无条件关闭 deterministic，使之后的 CUDA 逐位回放使用非确定性 `index_add_`。最小复现为 27 passed 后 1 failed，差异在 entropy 累计的 FP32 末位。
- 修复 DDP 恢复进入前的 deterministic/warn-only；两组严格回放测试自己设置确定性 FP32，并在退出时恢复后端。
- 六个 OGD fixture 测试需要 `tests/fixtures/ogd_trace_sample.jsonl`。原基准源码白名单没有包含 JSONL，已单独上传并记录 SHA256 `9b89e2133a531a12e2d4485e4de4b147f48d687e1ac943b49c7f7195f50ec2dd`。这是仓库静态测试样例。
- 最终 round9 源码：`c1765e5c239f5d06fc88a94d2b33cd1d3d9649e87b68942183262e48f748276b`；包 SHA256 `79159f4127c2783a88f74914a2193b4bdc8b7928b19a1e38843dcdef1e91d4f1`。
- 与 round8 相比，**仅三个测试文件改变；所有生产代码逐字节相同**。逐文件差异回执在 `source8-to-source9-delta.json`。因此性能表仍归属于其真实运行的 round8，不伪称重跑了 round9 性能矩阵。
- 修正后的本地全套再次为 **1,019 passed / 198 skipped，91.55 秒**。
- 异步原始指针压力测试 **20/20 通过**，覆盖跨分配流下 pending kernel 后立刻释放、重绑、prune/clear 和同尺寸内存填充；检查发生在释放及内存压力之后。证据：`download/results/triton-async-lifetime.json`，总耗时 6.09 秒。

17 份训练配置和一份 JSONL fixture 分别有独立打包清单；共享 gd 扩展通过受检符号链接供 spawn/CLI 子进程加载，测试前后核对源码、引擎、配置和 fixture 摘要。

## 使用与复现

新训练配置的 Q 组合使用 `--causal-sdpa --rollout-kv-cache --rollout-batched-attention --rollout-private-graphs`。
当前策略为主的已测 workload 可再加 `--rollout-triton-cache --rollout-triton-min-batch 4`。
保留默认 FP32；上述 optional 开关默认仍关闭。实际 CUDA 数值后端必须以 manifest 为准。

单进程诊断示例（在有兼容 CUDA/PyTorch/Triton 和已编译 gd 的 Linux 环境运行；输出目录须未使用）：

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 GUANZERO_PINNED_UPLOAD=0 PYTHONPATH=python:oracle:. python - <<'PY'
import torch
from bench.history_host import main
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
main(['--device', 'cuda', '--case', 'current',
      '--output', '.work/throughput-replay-current',
      '--envs', '32', '--width', '64', '--layers', '2', '--heads', '4',
      '--steps', '64', '--chunks', '8', '--warmup', '2', '--seed', '2026092701',
      '--batched-private-attention', '--private-graphs',
      '--triton-cache', '--triton-min-batch', '4'])
PY
```

完整矩阵脚本及说明保留在 `.work/vast-throughput-2026-09-28/production_campaign.py` 与 `PRODUCTION_CAMPAIGN.md`。报告 JSON 保留每个已接受进程的结果摘要和哈希；原始日志、源码包、测试 XML、生命周期回执均保留在同一忽略目录。

最终 round9 的远程完整 `tests/` 套件通过：**1,192 passed / 11 skipped，309.66 秒**；运行前后源码、共享引擎、17 份配置和静态 JSONL fixture 摘要一致。回执：`download/experiment-full-gpu-09/receipt.json`，完整测试 XML 与日志在同目录。首次失败/超时及最小复现日志保留，未替换成成功日志。

## 真实 PPO 短循环

source8、同卡、同 seed、width64/layers2/heads4、32 环境，32 steps × 6 updates，前 2 次 warmup，2 epochs / minibatch_matches4，顺序 `EAQVE`。
五个进程全部严格一致：每次 collect、learn、update 后的完整轨迹、公开历史、模型、critic、Adam 状态、population 和各 RNG hash 均匹配。每个进程的 4 次测量更新均实际学习，共 34 个 minibatch。

| 变体 | 原始整轮计时 DPS（包含诊断） | 扣除单独计量哈希开销后的 DPS |
|---|---:|---:|
| 原始 A | 715 | 749 |
| 主机优化 E（两次中位数） | 969 | 1027 |
| Q 私有图+attention | 1366 | 1481 |
| V Q+Triton min4 | 1392 | 1525 |

V/A 原始整轮约 **1.95×**；扣除已计量诊断区间后约 **2.04×**。仅 E 重复两次，A/Q/V 各一次，因此这仍是短时工程探针，不能推广为稳定长训练倍数。
这里“整轮”指测量阶段的 `update()` wall time，覆盖 collect、learn 和函数内的验证；不含初始化、warmup 及 `update()` 返回后的额外状态哈希。
哈希会额外同步、下载数据；验证后恢复 `buffer.data` 原指针，使正常 compact 工作仍在 learner 计时内。即使扣掉验证 wall，也仍有测量扰动。

六次更新产生了三个真实 snapshot，但这段短运行的 match 尚未结束，128 个席位仍全部绑定 learner；因此它包含真实学习和快照生成，**不覆盖最近池/宽池的实际训练吞吐**。
首次尝试 A 因原基准归档缺少 `source-identity.json` 而在初始化失败。补写前重新计算并匹配原 preflight 的全部源文件摘要，源码未改；原失败记录和补写回执保留。成功重试证据：`download/experiment-ppo-throughput-08-retry/report.json`，runner 为 v2 `f736356d121251e0ca3a744686abc80d83d308e40f70143cba2087176d3fe27f`。

## 回传、销毁与费用

- 回传 **800 个文件、34,201,098 字节**，逐文件 SHA256 与大小均匹配，0 个错误；清单 SHA256 为 `52e7b01f43f28329be4636d012befb8686667afb5140dcc443434b82e42d86b7`。
- 本地校验回执：`.work/vast-throughput-2026-09-28/artifact-verification.json`；远程清单：`download/artifact-final-manifest.json`。失败尝试、测试 XML、原始日志及源码包一并保留。
- 2026-09-28 **03:56:25 UTC**，Vast 返回销毁成功，随后 API 显示 `instances: 0`、`owned: null`；刷新控制台显示 **Instances (0), No instances found**。销毁截图原路径：`.work/vast-throughput-2026-09-28/teardown.png`（2026-10-06 文档检查时，该本地文件不存在）。
- 从创建至确认不存在共 **7,017.46 秒（1 小时 56 分 57 秒）**。按观察到的单价估算计算与磁盘费 **$1.5266**，未单独核实流量费。控制台最终余额 **$8.47**，相对充值后的 $10 约减少 **$1.53**。
- 生命周期与费用回执：`.work/vast-throughput-2026-09-28/teardown-summary.json`、`status-after-destroy.json`、`lifecycle.jsonl`。源码修改保留在当前 worktree，未提交；主仓库未改动。
