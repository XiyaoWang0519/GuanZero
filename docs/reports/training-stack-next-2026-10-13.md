# Page-span 基线上的投影优化（2026-10-13）

状态：作业 7215 已结束（exit 1）。packed 通过 CUDA 数值 gate，但完整 PPO
在第 98 次更新的非有限梯度范数检查处停止；未完成计划对照，没有新增提速结论。
用户授权继续优化；沿用不额外付费、不改主线配置的约束。

## 目的与候选

此前 page-span 的完整 PPO 收益已在两卡和三卡测得；全量 key/value 折叠
虽有速度收益，但真实 CUDA log-prob 差超过 1e-4，不能作为已通过的优化。
本轮所有 warmup、profile 和对照臂都开启 page-span，测量新增收益。

1. `packed`：只对各流 `position <= prefix` 的有效编码调用原 `kv_proj`，
   再填回原 attention 张量布局。保留 attention 的形状、mask 和运算路径；
   GEMM 行数变化仍需 CUDA 数值验证。额外 gather/nonzero/scatter 开销可能
   抵消减少 padding 投影的收益，因此 CPU 一致性不代表提速。
2. `value_fold`：保留显式 key 投影，仅把 value 线性投影移到 attention
   加权求和之后；保留 head-depth scale 与 value bias。它是数值验证候选，
   不是对旧全量折叠失败的修正结论。

候选顺序预先固定：packed 若通过则测速，否则检查 value_fold；两者都失败则
只保留 profile 和负结果，不运行未通过 gate 的速度臂。本轮不穷举所有候选。
实例级 wrapper 仅作用于 learner rollout 的 no_grad / one-decision-per-stream
路径；PPO 梯度路径与 snapshot actor 不替换。无模型精度、合法动作或历史截断改动。

## 数值门槛与本地验证

- 实际原路径与候选的 log-prob 绝对差仍为 1e-4，采样选择/RNG 必须一致；
  梯度开启时 log-prob 和 kv_proj 梯度逐位一致；真实轨迹抽样 256 行，
  完整公共历史重算误差不超过 1e-4。
- FP64 attention 是误差诊断参考；不再额外把“候选距离 FP64 小于 1e-4”
  当作门槛，实际路径 parity 门槛没有放宽。旧全量折叠失败记录仍保留。
- CPU 合成输入覆盖 1–64 rows、1–2,049 tokens。真实随机初始化 fixture
  256 次调用、42,343 候选、3,980 次选择：packed 的 log-prob 差为 0，
  value_fold 最大差 4.7684e-7；两者选择/RNG 一致、梯度路径检查通过，
  行为重算误差均为 4.7684e-7。
- packed CPU 完整 PPO 交替 smoke 共 13 次更新通过。CPU 结果只验证脚本，
  不代表成熟 checkpoint 的 CUDA parity 或 GPU 提速。
- 第一份 packed CPU gate 输出沿用了旧脚本的 `initial_2e-5_gate_failed` 字段，
  这是旧全量折叠历史标记，不是该候选的失败；已在提交版改为明确 `variant` 字段，
  原始输出保留，不静默改写证据。

## 冻结与测量方案

沿用此前冻结 source SHA256
`ea07296baf000b83a0b0d4a858f07be8be0799280c122b6dbd14e27f0b70e438`
与 checkpoint SHA256
`72837de605ce1a49aeaee0ebf55ab7b12ee22d2fa5862edcd456820ac93fae26`。
源码与 checkpoint 引用隔离副本，不写主线。

作业 7215 请求 normal 分区 4 GPUs / 96 CPUs，25 分钟上限，工作负载另有
1,380 秒 timeout。先逐卡 CUDA 检查，仅使用至少两张通过的分配卡，私有 MPS
daemon；不足两张则退出。实际卡数必须从 health/hardware 回执确认。

先 warmup 24 次完整 PPO，再做 8 次同步 profile（排除前两次）。随后在训练
checkpoint 上执行候选 gate；通过后重建独立 trainer，24 次 warmup + 6 对
AB/BA blocks，每 block 8 次更新，前两次排除，每臂保留 36 次。
所有臂 page-span=true，8 ranks / 1,024 环境、FP32、原 PPO 和快照配置。
计时包住完整 trainer.update，各 rank 取最大墙钟；不包含 checkpoint 保存。

原脚本输出阶段名仍叫 `projection`，实际候选由 `candidate-decision.json`
指定；其 `base` 明确是 page-span 基线，不能误称主线默认配置，也不能把该收益
直接相加到上一轮 page-span 结果。

## GPU 结果：通过局部 parity，但整段 PPO 未完成

实际使用三张健康 L40S / 8 ranks；四张分配卡中的一张未通过逐卡检查。
作业于 2026-10-14 01:07:36 UTC（多伦多 10 月 13 日 21:07:36）启动，
01:17:05 UTC 附近发生失败，退出回执为 1。

packed 的 CUDA gate 通过：256 次真实调用、308,695 个候选的最大 log-prob
差为 4.577637e-5，小于 1e-4；19,656 次选择无差异，sampler RNG 一致。
梯度开启的 log-prob / kv_proj 梯度逐位不变；256 行完整历史行为重算误差
2.861023e-5。按固定候选顺序选中 packed，因此 value_fold 未运行 CUDA gate
或完整 PPO，不能给出其 GPU 结论。

Profile 阶段 8 ranks 均完成 32 次更新，保留最后 6 次同步分项：

| 分项 | 平均秒数 | 解释 |
|---|---:|---|
| public cache / collation | 0.924 | 仍是采样主要成本 |
| learner prefill | 0.098 | 与 cache 合计约占采样分项 54% |
| actor / sampling | 0.342 | 本轮 page-span 基线 |
| gradient reduce | 0.667 | 约占 learner 分项 30%，含 rank 等待 |
| backward | 0.608 | learner 计算成本 |
| forward | 0.503 | learner 计算成本 |

以上是同步 profile 分项，不是非 profile 更新墙钟，也不能与先前两卡配置
直接相除归因于某个优化。下一处证据目标是 cache 整理和归约/等待的进一步拆分。

完整 PPO 阶段每个 rank 只完成 **97/120** 次更新（index 0–96）。在 index 97、
packed block 4 / within 1，rank 0 于 train.history_ppo.learn 的有限性检查
抛出 `FloatingPointError: non-finite gradient; stopping before the step`。
具体检查对象为 actor / critic 的 clip_grad_norm 返回范数；现有日志不能区分
原始梯度 NaN/Inf 与范数计算溢出，也不能证明是 packed 引起。该 optimizer step
未执行，其他 ranks 被终止，没有保存续训 checkpoint。

不完整数据不能当作计划完成的测速。为保留负结果，只取失败前四组完整 AB/BA
配对（每臂 24 个保留样本），排除未配对的第五组：page-span baseline 4.0590 秒，
page-span + packed 4.1034 秒，诊断吞吐变化 **−1.08%**，配对 block bootstrap
区间 **−2.44%～+0.31%**。没有显示提速；这些是失败运行中完成部分的描述，
不是通过稳定性验证的性能结论。不能采用 packed，也没有启动新试验或放宽 gate。

## 产物与核验

本地 `.work/stack-next-2026-10-13/` 保存脚本、CPU 结果和提交 SHA 清单；
远端 `experiments/stack-next-2026-10-13/` 保存 GPU 产物。已回收至本地
`remote-results/`，100 个文件逐一验证大小/SHA256。profile 完成回执齐全；
projection 的 stage/rank 完成回执缺失，与失败一致，远端 summary 未生成。
本地 `failure-analysis.json` 记录逐 rank 样本数、完整配对诊断和 profile 重算，
不伪称与不存在的远端 summary 对齐。`final-verification.json` 确认生产完整
source_identity 通过、SHA 与冻结值相同、主线 manifest 未改变。
主线配置、源码和 checkpoint 未改；没有付费实例。
