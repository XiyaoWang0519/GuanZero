# 主线启动失败与 page-span 复测（2026-10-13）

状态：作业 7174 已完成（exit 0）。四张分配卡中三张通过 CUDA 检查，
GPU 1 复现不可纠正 ECC；使用三张健康 L40S 的 page-span 完整 PPO 吞吐提高 8.50%。未修改主线配置、源码或 checkpoint，
未重新启动主线训练，未租用付费 GPU。

## 当前证据

- 主线作业 6922 在 2026-10-13 08:24:25 UTC 启动，08:43:40 UTC 结束。
  第一次尝试在 CUDA 初始化处报 MPS 805 连接错误；关闭 MPS 后，
  第 2–7 次尝试的日志末尾均包含 `uncorrectable ECC error encountered`。
- rank 1 / rank 5 在 `torch.cuda.set_device()` 失败，尚未进入 trainer 的
  PPO 计算。按 8 ranks / 4 visible GPUs 的轮转映射，两者对应逻辑 GPU 1；
  物理卡身份及当前健康状态须由分配后的逐卡检查确认。
- 自动恢复先关闭 MPS，再回退 cache 等运行选项，仍然在相同初始化阶段失败；
  耗尽 6 次重启后终止。watcher 于 08:58:36 UTC 发现无后续作业并退出。
- attempt 日志跨作业追加，较早的 source-manifest 报错不能作为本次根因。
  本次重新运行完整 `source_identity()` 检查，256 个文件及文件集合均通过，
  source SHA256 为 `ea07296baf000b83a0b0d4a858f07be8be0799280c122b6dbd14e27f0b70e438`；
  主线 manifest 与此前实验冻结记录一致。

这些日志支持“GPU/CUDA 健康故障阻断启动”，不支持将其归因于 PPO、page-span
或模型收敛问题。MPS 805 是否由同一 ECC 故障导致尚未单独证实。
不可纠正 ECC 的进一步处置应由管理员结合 Xid、ECC 与 recovery-action 信息判断；
本任务不在共享节点上执行 GPU reset、重启或全局杀进程。
参考 [NVIDIA GPU Node Triage](https://docs.nvidia.com/deploy/gpu-debug-guidelines/gpu-node-triage.html)。

## 已提交的隔离验证

- 作业 7174，normal 分区，4 GPUs / 96 CPUs，20 分钟上限；
  工作负载另有 1,080 秒 timeout。short 分区的四卡请求因 QoS 限制被拒，
  没有创建重复作业。
- 只对 Slurm 分配的 GPU，分别启动独立 CUDA 进程，执行小矩阵乘与同步，
  并保存 ECC / page retirement / row remapper 信息。
- 若至少两张卡通过，则仅向本作业私有 MPS daemon 暴露通过的卡；
  若不足两张则停止，不反复重试故障卡。实际参与卡数会写入结果。
- 使用此前冻结的同一 source/checkpoint，8 ranks、1,024 环境、FP32、
  完整公共历史和原 PPO 设置。先跑 page-span 真实 rollout parity，
  再做 24 updates warmup + 6 对 AB/BA blocks，每 block 8 updates，
  每次切换排除前 2 次，每臂保留 36 次完整 PPO 更新。
- 如果只有三张卡健康，结果必须标为三卡复测，不能称为四卡生产布局验证。
  投影仍保留此前数值 gate 失败结论，本次不将它加入组合臂。

此前 15:28 UTC 的预计开始时间是 16:21:56 UTC；实际提前于 15:42:31 UTC
启动，完成后保存 exit 0 回执。没有重新提交或追加训练。

## 实测结果与核验

逐卡独立 CUDA 小矩阵乘与同步检查：分配编号 0、2、3 通过，1 失败并复现
`uncorrectable ECC error encountered`。对应 ECC 记录显示 DRAM uncorrectable
volatile 2 / aggregate 4，row remapper uncorrectable 2、pending Yes。
这确认故障卡仍阻断 CUDA；没有执行 reset 或改变共享节点。通过小检查并完成
本次工作负载不等于对三张卡未来可靠性的保证。

实际运行布局为 **3 L40S / 8 ranks**，rank 轮转分配为 3/3/2，96 个分配 CPU。
不是 4 GPU 生产布局。两臂硬件及 PPO 设置相同，均从既定 checkpoint 开始同一
交替试验轨迹，保持 1,024 环境、FP32 与完整公共历史。

| 指标 | 原配置 | page-span |
|---|---:|---:|
| 完整 PPO update 平均秒数 | 4.4350 | 4.0875 |
| 环境决策/秒（包含 learn） | 14,777 | 16,033 |
| 最慢 rank collect 平均秒数 | 2.2790 | 1.9181 |
| 最慢 rank learn 平均秒数 | 2.5481 | 2.4239 |
| rank 0 平均 prefix | 584.1 | 569.0 |
| 单 rank 最大 allocated GiB | 2.056 | 2.003 |

吞吐 **+8.50%**，update 时间 **−7.84%**；六对 AB/BA block bootstrap 的
吞吐改善 95% 区间为 **+6.65%～+10.16%**，六对点估计均改善。
这是单次运行的 block 波动区间，不是跨训练种子或多次作业的总体置信区间。
collect/learn 各自取最慢 rank，不能直接相加替代完整 update 墙钟时间。

真实 rollout parity 通过：19,656 条 compact buffer 记录中非 log-prob 字段
hash 与 sampler RNG 相同，两种 log-prob 最大差为 0；各抽样 256 行完整公共
历史重算误差均为 2.8610e-5，小于既定 1e-4 门槛。

已回收并逐文件验证 **79 个产物**的大小与 SHA256；8 ranks 每个均完成
120 updates（24 warmup + 96 对照），每臂每 rank 保留 36 次更新；rank 和
stage 完成回执齐全。本地重算汇总与远端 summary 完全一致。生产完整
source_identity 通过且 SHA 与冻结记录一致，主线 manifest 未变。

结论：page-span 在此前两卡 +10.28% 之外，本次三卡也复现完整 PPO 提速。
仍未验证四张健康卡的生产布局；不包含 checkpoint 保存开销，不是棋力结果。
主线没有恢复或启用 page-span，硬件故障需另行处理。投影仍未通过此前数值 gate。

## 证据路径

本地 `.work/stack-health-2026-10-13/` 保存脚本、提交回执和失败日志末尾；
集群 `experiments/stack-health-2026-10-13/` 保存同次诊断与验证产物。
源码和 checkpoint 引用此前冻结的 `stack-ab-2026-10-13` 实验副本，
不依赖主线继续写入的文件。前次速度证据见
[训练栈 A/B 报告](training-stack-ab-2026-10-13.md)。

最终产物位于本地 `.work/stack-health-2026-10-13/remote-results/`，
含 `artifact-manifest.json`、`summary.json`、`health.json`、`ecc-before.txt`、
`final-verification.json`、逐 rank timings 与 gate/退出回执。
