# 非有限梯度定位与归约等待诊断（2026-10-13）

状态（October 14 收尾）：7220 已复现 FP32 梯度范数溢出并退出；巨大梯度的算子级根因仍未确定。
用户要求先定位非有限梯度，再继续优化公共历史缓存和梯度归约等待；
使用现有集群、不额外付费、不改主线配置。

## 已有证据

[7215](training-stack-next-2026-10-13.md) 在第 98 次更新、packed 臂退出。
但 actor 的大梯度范数在 packed 启用前已存在：warmup index 0 约 1.73e7，
index 1 约 1.53e9，index 10 约 1.12e10；index 96 约 1.22e11。
同期 critic 范数约为十几，损失和 KL 等记录没有直接暴露对应的非有限项。
这些是 update 汇总，不能代替失败 minibatch 证据，也不能证明 packed 无关。

原退出条件检查 `clip_grad_norm_` 返回的 actor/critic 范数有限性，
不能由这一条异常区分原始梯度 NaN/Inf、归约溢出或 FP32 范数溢出。

## 隔离诊断

- 沿用既有冻结 source / checkpoint 及 7215 的 page-span + packed AB/BA
  调度，8 ranks / 1,024 环境；不改 PPO 目标、精度或训练配置。
- 在原始 backward 后、归约前，以及归约后、clip 前分别记录梯度有限性、
  actor/critic FP32 与 FP64 范数、最大绝对值和最大的参数梯度。
- 所有 rank 通过额外 collective 同步故障信号，异常时停止，不执行该步优化。
- 首次局部范数超过 1e8 或出现非有限值时，在原参数上重算同一 minibatch，
  用 autograd anomaly 与逐损失 autograd.grad 记录 policy、entropy、next、belief、
  outcome、critic 等来源；保留隔离的诊断 fixture，恢复重算前 RNG。
  fixture 不是可恢复主线训练的 checkpoint，不写主线文件。
- 归约保持原有 gloo host-copy、求和、按 real-rank 数量平均的算术；额外 barrier
  将到达等待与实际 all-reduce 分开，记录 pack/D2H、等待、all-reduce、H2D/unpack。
  这些同步和诊断会改变耗时，不能从本次 instrumented run 宣称吞吐改善，
  等待分项也条件于诊断开销，不等同于生产纯通信占比。
- 本轮不启用新的缓存优化，也不放宽已有数值门槛。后续修复/优化必须在根因
  有证据后做独立验证；已有 packed 无提速的负结果继续保留。

## 本地验证

- 单 rank 和双 rank CPU 完整 PPO 调度 smoke 各 7 次更新通过。
- 人工注入有限的 1e20 梯度：FP32 范数为 Inf、FP64 范数有限，分类正确。
  注入 NaN 后能单独标出原始梯度非有限；不是把所有异常都归为数值溢出。
- Python 编译与 shell 语法检查通过。上述检查不证明 GPU 根因已复现。

## 作业与证据

作业 7220 请求 normal 分区 4 GPUs / 96 CPUs，35 分钟硬上限，工作负载
timeout 1,980 秒。先逐卡 CUDA 检查，只使用至少两张通过的卡及本作业私有 MPS。
提交后预计 2026-10-14 03:47:06 UTC（多伦多 10 月 13 日 23:47:06）启动，
该时间随调度变化。不会 reset GPU、重启节点或全局终止其他进程。

本地 `.work/stack-gradient-2026-10-13/` 保存诊断脚本、CPU smoke 与提交 SHA；
集群 `experiments/stack-gradient-2026-10-13/` 保存逐 rank
`gradient-diagnostics.jsonl`、`loss-attribution.jsonl` 与按原因命名的 fixture。
source SHA 为 `ea07296baf000b83a0b0d4a858f07be8be0799280c122b6dbd14e27f0b70e438`，
checkpoint 沿用此前冻结的 `72837de605ce1a49aeaee0ebf55ab7b12ee22d2fa5862edcd456820ac93fae26`。
提交前完整生产 source_identity 通过；结果回收后再次核验主线未改变。

## October 14：7220 诊断结果

作业 exit_code=1，4 张分配卡中 3 张通过健康检查，属于三卡 / 8 ranks
诊断。每个 rank 完成 94 次更新（index 0–93），在 index 94、base 臂、
block 4 / within 6 的第 2715 次归约停止；无完整 120-update 完成回执。
这次失败发生在 page-span 基线，不足以把此前失败归因于 packed。

所有 8 ranks 的 2715 条诊断记录中，归约前后梯度元素均有限。
故障 minibatch 的 rank 1 局部 actor 梯度 FP64 范数约 2.6581e24，
FP32 范数已为 Inf；其余 rank 局部范数约 1e10–1e12。
归约后 actor FP64 范数约 3.3226e23，FP32 范数为 Inf；critic 约 10.26。
因此本次直接退出机制是有限巨大梯度导致 FP32 范数溢出，
不是归约首次制造 NaN/Inf；巨大梯度在 rank 1 局部 backward 后已存在。

rank 1 同 minibatch 的逐损失重算也保持梯度有限，outcome_loss 的
kv_proj.weight 梯度 FP64 范数约 2.2017e25，远高于 next_loss 和 critic。
belief_loss、policy_loss、entropy 路径也有巨大梯度。该证据定位到 actor
反向路径并突出 outcome 辅助损失，但尚未证明具体算子或唯一损失是根因。
逐项 autograd.grad 与原组合 backward 的数值不能视为逐位相同的因果分解。
仅把 norm 改成 FP64 可能避开退出条件，不等于解释或修复巨大梯度。

已回收 67 个文本/脚本文件，逐文件大小和 SHA256 本地验证通过，位于
`.work/stack-gradient-2026-10-13/remote-results`；本地全量诊断检查保存为
同目录上级的 `diagnosis-analysis.json`。fixture 保留远端，未下载模型或 source。
完整生产 source_identity 与冻结源码一致，主线 manifest 字节一致。
新增同步、FP64 统计及 anomaly 重算使本次耗时不适合作为优化收益。

当前自动检查已为 PAUSED，无新作业提交，主线未改。下一步需要对 rank 1
故障 fixture 做最小重放和逐层反向定位，确认具体数值 bug 后再验证修复，
随后才能继续公共历史缓存/归约等待的无诊断开销完整 PPO 测速。
