# Q/K 范数 16：隔离 CUDA 与完整 actor loss 验证（2026-10-16）

状态：**CPU 与 L40S CUDA 验证均通过，隔离作业 7937 已完成。**
用户授权为隔离验证，明确不启动训练。未运行训练入口或续训脚本，未写迁移
checkpoint；独立 Adam 步只作用于一次性内存副本。

## 身份与范围

沿用 [离线范数与 Adam 检查](qknorm-plasticity-2026-10-16.md)的 qk-both u94488：
checkpoint SHA256 `a4bed2bfb4c3b8c1382a8ed5604e8e797307a37ea2088523dffb1243de6a58bd`；
目标逐 head 联合 weight/bias 范数 16，Q/K 同步缩放，V 保留；Adam `m/a`、
`v/a²`，step/lr/eps 保留。实际模型源 SHA256
`8019503df4a65e88583081633ee89b1caaf4b5070b8ba0562c15ccf6d15a0bd4`。

入口：[bench/qknorm_cuda_gate.py](../../bench/qknorm_cuda_gate.py)。本地与服务器
均核对实际模型/损失/辅助头/合并快照实现的源身份。服务器使用独立源码副本，
源来自已核验的 qknorm 实现；production source 与 runs/main 均未改。

## 完整 actor loss 的验证方式

调用拥有实现的 `HistoryTrainer.minibatch_loss`，用固定 batch adapter 提供原
rank1 fixture 的输入、动作、隐藏牌训练标签、回报、advantage、response/outcome
标签。**不调用 HistoryTrainer 构造、run、learn 或环境采样。** Critic 使用
checkpoint 权重但冻结；只对 actor 的 policy_total 反向。

actor 总损失逐项核对为 PPO surrogate − entropy coefficient × entropy + response
coefficient × response_loss + aux coefficient × (next + belief + outcome)。
三个场景各测 grouped-default、grouped-math、scalar-default，共 9 组原始/迁移对照：

- stored：保留旧 fixture 行为 logp；这是历史离策略输入，不能当作当前 on-policy batch。
- ratio_one：把旧 logp/behaviour logp 替换成当前原模型在固定动作上的 logp，
  测 ratio≈1 的数值路径；动作、advantage 与标签仍来自历史 fixture。
- ratio_clipped：在上项基础上按 advantage 符号施加 ±.3 的 log-ratio，明确覆盖
  PPO clipping 分支；是合成数值诊断，不是新的训练样本或目标。

每组检查 loss/梯度有限、总损失分解、缩放前后坐标变换后的梯度一致性，以及
default/math/scalar 对照。完整 actor 原生 grad_clip=10，与 FP64 全梯度范数
参考比较；另把同一梯度向量调到范数 20，验证裁剪实际触发并收敛到 10。
恢复正常裁剪后的梯度，在各自原始内存副本独立执行一次 Adam，检查参数/矩
有限、step 仅加一；不串联这些步骤，不保存更新后权重。

前向门槛沿用 max log-prob 差 1e-4、max TV 1e-4、argmax 翻转 0；完整损失差
门槛为 1e-4 + 1e-5 × |参考总损失|，坐标映射后全局梯度相对误差门槛 1e-3。

## 已完成 CPU 预检查

PyTorch 2.14.0 / CPU：原 fixture 328 行与前次保存的 576 个新行做前向，
三种 ratio × 三路径的完整损失、梯度、裁剪、一次性 Adam 检查均通过。
历史 fixture 的 ratio 偏差较大：最大偏离 1 约 11.68，clip fraction 约 .457；
ratio_one 为 0，ratio_clipped 为 1，说明测试确实覆盖了不同 PPO 分支。

grouped-default 原始完整梯度范数分别约 3.08706、2.63628、1.56660；正常
阈值 10 未触发裁剪。主动范数 20 压力检查单独验证了裁剪路径，不能把它
叙述为原始 batch 自发出现大梯度。

另以实际权重检查 learner + checkpoint 中 **13 个常驻历史快照**，每个快照
按自身逐 head 范数缩放，不修改 checkpoint population。合并与逐 identity
推理的最大 log-prob 差：缩放前 4.58e-5，后 6.10e-5；合并路径缩放前后
差 4.58e-5，均通过 1e-4 门槛。快照没有 optimizer 状态，只有独立权重缩放；
这不是接入 production population 的迁移流程。快照/graph 测试使用固定 seed 的合成张量输入和真实 checkpoint 权重，不能当成真实牌局或生产 cache 迁移证据。

CPU 不运行 CUDA graph；不以 CPU 结果替代 CUDA 结果。

## GPU 作业及限制

7936：单卡 15 分钟上限诊断，在健康检查的矩阵乘法阶段遇到不可纠正 ECC，
没有执行模型验证。该失败属于分配卡健康问题，不是模型通过或失败证据。

7937：有界重试，最多预留 2 GPU × 15 分钟；逐个检查已分配设备，只选一张
健康卡运行诊断（模型脚本 600 秒超时）。首个失败产物单独归档。
2026-10-16 19:05 America/Toronto 查看时为 Resources 排队，调度器暂估
20:29 开始；这是当时的调度估计。作业随后提前获得资源，完成且 exit code 0。两张分配卡中一张健康、一张 ECC 故障，模型仅在健康卡运行。没有取消、恢复或改动已有主线/监视作业。

CUDA 已完成上述完整损失对照、默认与 math、scalar-private、实际历史
快照合并，以及 private CUDA graph capture/replay。分组 learner 的私有 attention 保持 math fallback；scalar-default 的反向节点确认实际走了 efficient SDPA。上述有限输入下两者梯度均正常，不构成移除 math fallback 的依据。

DDP、多次 PPO 更新、真实
新 rollout 的 GAE/标签、生产 cache 迁移、长程稳定性与棋力不在本次证据范围。

<!-- CUDA_RESULT_BEGIN -->
**作业 7937 CUDA 验证通过。** 该结论基于完整 gate.json、exit code 0、脚本 SHA、有限值与 9 组完整损失记录的交叉检查。

运行环境：2.11.0+cu128 / NVIDIA L40S；实际只使用一张通过健康检查的分配卡。

- 904 行前向：最大 log-prob 差 3.05e-05；首选动作零翻转。

- 9 组完整 actor loss 对照：最大总损失差 4.77e-07；坐标映射后全局梯度最大相对误差 3.24e-06。

- default/math/scalar 梯度对照：最大全局相对误差 7.62e-06。

- 原生范数计算、正常裁剪和主动范数 20→10 的压力裁剪均通过。

- 18 个独立内存副本各一次 Adam 检查均通过：参数/矩有限，step 仅加一；没有串联更新或保存新 checkpoint。

- learner + 13 个历史快照：迁移后 merged/reference 最大 log-prob 差 6.1e-05；合并路径迁移前后差 4.58e-05。

- private CUDA graph：capture 1、hit 1，最大 log-prob 差 0。

- 原始 checkpoint SHA 与基础 actor 权重保持；该通过结论仍不代表 DDP、完整 PPO 训练稳定性或棋力收益。

- stored 完整梯度范数：原始 3.08707 → 迁移 3.08707；正常裁剪系数 1。

- ratio_one 完整梯度范数：原始 2.63628 → 迁移 2.63628；正常裁剪系数 1。

- ratio_clipped 完整梯度范数：原始 1.5666 → 迁移 1.5666；正常裁剪系数 1。
<!-- CUDA_RESULT_END -->

## 本地产物与回收

`.work/qknorm-cuda-gate-2026-10-16/` 已保存 CPU `gate.json`、log、诊断脚本包、
提交记录与原始源 SHA。7937 的实际结果已由有界回收器下载，校验 gate、exit code、脚本 SHA、有限值、记录数量与 graph capture/hit 后确认通过；回收器已正常退出。
回收器不提交、重启或取消任何作业。Slurm accounting 数据库当时不可用，未取得 sacct 耗时统计，不据此推算实际 GPU 小时。
`.work` 为忽略的本地证据，不是可移植依赖。
