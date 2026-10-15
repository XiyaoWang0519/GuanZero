# Rank 1 故障 fixture 逐层反向定位（2026-10-14）

状态（October 14 更新）：算子级故障已复现，最小修复通过 fixture 门槛；7325 的 120-update 完整 PPO 稳定性验证通过。
用户授权继续定位、验证修复后再优化公共历史缓存和归约等待；仅现有集群，
不额外付费，不修改或重启主线。

## 证据与限制

[7220 诊断](training-gradient-diagnosis-2026-10-13.md) 已确认 rank 1 的巨大有限
actor 梯度在归约前存在，并使 FP32 norm 溢出。outcome_loss 重算中
kv_proj.weight 的范数约 2.20e25；这不是已定位具体算子的证明。

仅下载 rank 1 的故障 fixture（11,389,931 bytes），SHA256 验证为
`9c3d1018ac52453852536ff5ef497eb1dbcbd553f02e199b30c9570e6c152cd8`。
文件保留在忽略目录 `.work/stack-fixture-2026-10-14/rank1-fixture.pt`，不外传。
它包含隔离诊断权重与 minibatch，不是主线可恢复 checkpoint。

本地 CPU / PyTorch 2.14.1 重放 outcome_loss 为 3.1693379879，接近集群
3.1693382263；最大的参数梯度范数约 12.9242（outcome head），未复现异常。
本地只重放模型张量，缺失 native engine 时使用 fixture 维度和 Play=3 的
schema 常量导入 shim，不执行游戏、采样、优化器或训练。CPU 与集群
PyTorch 2.11.0+cu128 的设备和版本都不同，不能据此宣称 CUDA backend 是根因。

## CUDA 隔离对照

7323（gz-fixture-trace），normal / 4 GPU / 96 CPUs / 35 分钟上限，
工作负载 timeout 1,980 秒。逐卡健康检查至少两张通过后使用本作业私有 MPS。
单 fixture 单设备重放依次运行默认 SDPA、数学 SDPA、默认 SDPA 重复；
记录每层及 SDPA q/k/v/output 的前向与梯度有限性、FP64 范数、最大绝对值
及 grad_fn，不输出原始输入内容。数学 SDPA 是诊断对照，不是已推荐修复。
无 optimizer step，无主线 checkpoint 写入。

远端 `experiments/stack-fixture-2026-10-14` 引用原冻结 source，直接读取
7220 rank 1 fixture；本地脚本 SHA、job ID 和 scope 在 `submission.json`。
源码未改。所有新增脚本在 `.work/`；本地 Python 编译与 shell 语法通过。
本地依赖安装在独立 venv，不改变生产环境。

下一步：比较相同 CUDA 环境的梯度首次放大位置、默认重放重复性和数学对照。
只有具体修复经 fixture 与数值门槛验证后才继续稳定性和无诊断开销完整 PPO
性能测试；不得将逐层 hook 耗时当生产吞吐或通信占比。

## 7323 完成：CUDA 算子定位

7323 exit 0，三张健康 L40S，默认 / 全 math / 默认重复三次全部完成。
19 个文本文件已回收、大小及 SHA256 全部核验；完整生产源码和 manifest 未变。
三次 outcome loss 都为 3.1693382263183594，默认两次的 kv_proj.weight 梯度
均为 2.2017281676e25，math 对照最大参数梯度为 12.924197545。

逐层记录把首次异常放大缩小到两组私有 query 对公共历史的 masked SDPA。
第一组 sdpa4 输出梯度范数 0.00318782，但默认
ScaledDotProductEfficientAttentionBackward0 返回 V 梯度 9.444739e23；
全 math 对照 V 梯度约 0.00796285。第二组 sdpa9 同样异常。
编码器梯度异常是该调用上游传播的结果。此定位基于实际调用的 grad_fn，
不是根据库默认配置猜测 backend。尚未定位到 CUDA 内核内部的具体代码行。

进一步作业 7324（gz-fixture-repair）在相同冻结源码和 fixture 上验证：
1. 捕获同一 masked attention 的 q/k/v 和同一输出梯度，分别用 efficient 与
   math 重算 VJP，排除上游梯度与编码器前向差异。
2. 只在需要梯度的 masked attention 中选 math，其余 attention、rollout、权重、
   输入和目标不变；检查梯度有限、fixture 最大参数梯度范数小于 100，
   完整候选和 chosen log-prob 绝对误差均不超过 1e-4。

这是 fixture 级修复门槛；范数 100 只是诊断验收值，不改训练 grad_clip。
本地 CPU 最小修复重放通过，但 CUDA 门槛、完整 PPO 稳定性和性能尚待验证。
7324 同样 35 分钟硬上限、逐卡健康检查和私有 MPS，不运行 optimizer。
脚本 SHA 与状态保存在 `.work/stack-repair-2026-10-14/`；未改主线。

## 7324 完成：同输入 VJP 与最小修复通过

7324 exit 0，18 个文本文件大小/SHA256 本地验证通过，生产源码和 manifest
再次核验一致。同一 q/k/v、mask 和输出梯度的独立 VJP 重算稳定复现异常：
sdpa4 efficient V 梯度范数 9.444739e23，而 math 为 0.00796285348；
两者前向最大差 7.6294e-6。sdpa9 也复现，前向最大差 1.5259e-5。
因此已定位故障 fixture 的直接数值根因为 masked efficient SDPA backward
返回错误量级的梯度，而不是 loss 大、归约污染或 FP32 norm 本身造成巨大梯度。
该结论限定 PyTorch 2.11.0+cu128 / L40S / 本 fixture；内核内部实现原因未查明。

仅 grad-enabled masked attention 切 math 后 fixture 门槛全部通过：
所有梯度有限且最大参数范数小于 100；完整候选 logp 最大绝对差
3.2424927e-5，chosen logp 最大差 1.4424324e-5，均低于原 1e-4 门槛。
这是修复后 forward parity，不要求与错误 backward 的梯度一致。

已提交 7325（gz-fixed-stable）验证隔离修复的完整 PPO 稳定性：
8 ranks × 120 updates，全部 page_span=true，无 packed，无 profile/hooks。
fix.py 仅将 HistoryActor._attend_by_match 在 CUDA 且梯度启用时选 math，
其他 attention、无梯度 rollout 和训练超参保留。35 分钟上限、逐卡健康检查、
私有 MPS。代码/提交 SHA 在 `.work/stack-stable-2026-10-14/`。
本轮没有速度对照臂，不能据此宣称提速；稳定后再继续缓存/归约优化。

## 7325 完成：修复基线稳定性

7325 exit 0，3 张健康 L40S / 8 ranks，每 rank 完整完成 index 0–119，
逐 rank 和顶层完成回执齐全。57 个文本文件的大小及 SHA256 本地核验通过。
每条 update 的 actor/encoder/critic 范数统计均有限；rank 0 actor update
范数最大约 1.1372、critic 约 17.1018。每次 update 自带的跨 rank 参数
checksum 检查均未报错。各 rank 汇总范数可能因实际 minibatch 计数不同而不同，
不能据此声称参数不一致。

完整生产 source_identity 与冻结源码一致，main manifest 字节一致。
分析在 `.work/stack-stable-2026-10-14/stability-analysis.json`，原文在
`remote-results`。这是 120-update 有界稳定性证据，不是长期训练保证，
也没有 A/B 速度比较或棋力证据。主线未改。
