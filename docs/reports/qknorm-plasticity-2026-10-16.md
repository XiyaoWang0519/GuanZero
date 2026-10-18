# Q/K 范数调整：离线可塑性与 Adam 状态验证（2026-10-16）

状态：**本地 CPU 验证完成；未启动训练、未修改或输出训练 checkpoint。**
用户授权范围为离线缩放验证、策略前向检查、方向梯度与 Adam 状态处理检查。
所有一步 Adam 反事实均从原状态独立开始，作用于一次性内存副本，没有重复优化。

## 结论与下一步

四档逐 head Q/K 缩放均通过预设 CPU 前向门槛。只缩放 Q/K 的投影 weight 与
bias，可以基本保留当前策略，同时提高相同学习率下的方向调整幅度。
但内在方向梯度没有新增：这是条件调整，不是已证明修复 query 方向坍缩，
也不是棋力或续训收益证据。

若后续批准更进一步验证，**首选保守候选为每 head 联合范数 16，Q/K Adam
矩按梯度坐标迁移 `m/a`、`v/a²`，保留原 step、lr 与 eps**。这套候选在本次
独立一步反事实中较平稳；16 不是经训练选出的最优数值。更小目标值没有
可见的前向失败，但造成更大的后续方向更新，没有证据必须先降到随机初始化尺度。

本次不把候选接入 resume，不修改 population，不写迁移 checkpoint。
下一道工程验证应为隔离的 CUDA 前向/反向与实际完整 actor loss/梯度裁剪验证，
包括 merged snapshot 路径及快照迁移规则；完整 PPO 和收益仍未验证。
该建议不授权启动任何训练。

## 1. 身份、样本与方法

承接 [500M 机制核验](qknorm-mechanism-2026-10-16.md)，仅处理 qk-both 的
u94488 / 6,192,365,568 decisions checkpoint：

- checkpoint SHA256：`a4bed2bfb4c3b8c1382a8ed5604e8e797307a37ea2088523dffb1243de6a58bd`。
- 实际模型源 SHA256：`8019503df4a65e88583081633ee89b1caaf4b5070b8ba0562c15ccf6d15a0bd4`，
  与 checkpoint `run_identity.source` 对应条目一致。
- fixture SHA256：`9c3d1018ac52453852536ff5ef497eb1dbcbd553f02e199b30c9570e6c152cd8`。
- 原 fixture 328 决策，加前次已保存的三组 fresh 数据各 192 决策，共 **904 个
  独立样本行**。新数据共 24 个比赛身份、四个座位、round 0–3、前缀 0–263；
  样本文件哈希逐一核对前次记录。本次没有重新采样。
- PyTorch 2.14.0 / CPU / FP32，全部 SDPA 强制 math；不代表 L40S CUDA 结果。

每 head 把 weight 行与对应 bias 联合展平为 `theta_h=[W_h,b_h]`，选择
`a_h=min(1, target/||theta_h||)`。正比例同步缩放 Q 的 weight/bias；K 仅处理
`kv_proj` 前 128 行及前 128 个 bias，V 后半块逐位保留。其余模型参数、
query_norm 与温度逐位保留。

缩放前联合范数：Q 各 head 约 22.71–113.72，K 约 19.98–101.16。
测试目标 16、8、4、2.309401；最后一档约为线性层随机初始化时每 head weight
的 Frobenius 范数量级，仅作参考，不意味着适合已训练模型。

三个 CPU 私有 attention 路径分别比较缩放前后：

1. learner 分组 `_attend_by_match`：全部 904 行；
2. 显式 public encode、scalar-private `_attend`：同样 904 行；
3. batched independent-private `_attend_independent`：每数据集等距选 16 行，共 64 行。

共四档 × 四数据集 × 三路径 = **48 组前向比较**。这些路径复用了样本行，
不能把路径数量当作独立样本数量。

## 2. 策略前向

运行前设定门槛：最大绝对 candidate log-prob 差 ≤1e-4、最大逐决策 TV ≤1e-4、
最大逐决策 KL ≤1e-6、argmax 翻转 0。KL/TV 在 FP64 对每个决策重新归一化，
避免 FP32 log-softmax 的微小归一化误差产生假负 KL；log-prob 差使用原输出。

| 逐 head 目标联合范数 | 最大 log-prob 差 | 最大 TV | 最大 KL | argmax 翻转 |
|---|---:|---:|---:|---:|
| 16 | 4.01e-5 | 3.81e-6 | 2.93e-11 | 0 |
| 8 | 4.01e-5 | 3.81e-6 | 2.93e-11 | 0 |
| 4 | 3.62e-5 | 3.81e-6 | 2.91e-11 | 0 |
| 2.309401 | 3.05e-5 | 4.05e-6 | 4.87e-11 | 0 |

全部通过；RMSNorm 有 eps，故不能声称严格逐位不变或对任意输入成立。
KV cache、merged population、CUDA graph、保存/重载后的迁移尚未做专项验证。

## 3. 方向梯度：提高调整速度，没有创造信号

原 fixture 分别计算 outcome、belief、负策略熵三个标量的梯度，不混合 loss。
belief 的隐藏牌标签仅进入辅助损失，不进入 actor 输入；没有策略拟合或训练。

逐 head 联合参数的切向梯度定义为

`g_tan = g - theta * <theta,g>/||theta||²`。

近似尺度不变时，`theta'=a*theta`、`g'=g/a`，因此：

- 原始切向梯度约放大 `1/a`；
- 同学习率 SGD 的方向变化率 `||g_tan||/||theta||` 约放大 `1/a²`；
- 内在方向梯度 `||theta|| * ||g_tan||` 应基本保留。

所有目标、三种 loss、Q/K 各 head 的梯度坐标关系
`||a*g_scaled-g_original||/||g_original||` 最大误差约 **1.91e-4**。
内在方向梯度保持，支持这是近似重参数化。它并不使原先相似的 query
自动变得多样，也没有证明梯度指向更好的策略。

## 4. Adam 状态与一步反事实

实际 actor 使用 Adam：lr=3e-4、betas=(.9,.999)、eps=1e-8、weight_decay=0。
Q weight/bias 与联合 KV weight/bias 的保存 step 均为 **2,787,152**，
它计数 optimizer minibatch steps，不是 checkpoint 的 update 94488。
现有序列化状态依赖相同源的参数注册顺序；本探针核对参数数量、状态形状并
按该 actor 的 named_parameters 映射，没有猜测 K/V 的独立 step。

每种候选从相同保存状态独立执行一次原生 `torch.optim.Adam.step()`：
Q/K 用该候选的 `0.1*outcome_loss` 梯度，所有非 Q/K 元素用原模型同一
cotangent，固定非 Q/K 梯度以隔离迁移效果。原生全 actor grad_clip=10；
各候选 raw norm 约 1.56248，均未触发裁剪。

这是单个辅助损失方向的反事实，**不等于下一次实际 PPO、总辅助损失、
DDP 归约或训练轨迹**。原模型也做一个独立 native Adam 参考步，作为比较基准。

| 状态策略（目标范数 16） | Q 转角倍数中位数 | K 转角倍数中位数 | 一步后相对原 native 步的最大策略 TV | V 更新与原 native 步一致 |
|---|---:|---:|---:|---|
| 原矩不变 | 26.30× | 19.20× | 1.10e-4 | 是，逐位 |
| `m/a, v/a²`，保留 step | **9.42×** | **5.68×** | **3.44e-5** | **是，逐位** |
| 只清 Q/K 矩，保留旧 step | 38.03× | 62.60× | 5.71e-4 | 是，逐位 |
| 清整块投影矩及 step（反例） | 59.25× | 50.18× | 4.00e-3 | **否** |

这里转角是联合参数 weight/bias 的方向变化，并非 attention query 在全部输入上
的转角。倍数较大部分因为原步非常小；目标 16、矩迁移时，Q 绝对转角为
5.7e-5–2.74e-4 rad，K 为 4.1e-5–2.97e-4 rad。

更小目标配合矩迁移：

| 目标范数 | Q 转角倍数中位数 | K 转角倍数中位数 | 相对原 native 步的最大策略 TV |
|---|---:|---:|---:|
| 8 | 20.41× | 11.68× | 8.82e-5 |
| 4 | 42.85× | 23.72× | 1.98e-4 |
| 2.309401 | 75.95× | 41.36× | 3.72e-4 |

### 状态迁移应如何处理

梯度坐标变成 `g/a`，所以保留历史信息的自然转换是 **`m/a`、`v/a²`**，不是
把矩也跟参数一起乘 a。只对 Q 和 K 对应行执行，weight/bias 使用相同逐 head
系数；V、其他参数的矩与共享 step 保持原样。本次逐项断言通过。

这并非“完全保持 Adam 轨迹”。本次固定 lr、eps，意图正是改变后续方向步幅。
若要纯坐标变换、保持原函数轨迹，还需逐 head 的
`lr'=a*lr`、`eps'=eps/a`；当前原生优化器每参数组只有一个标量 lr/eps，
无法直接对同一 weight 的不同 head/KV 行这样设置。

eps 的影响确实不可忽略：目标 16 时，迁移前 Q weight 约 36.5% 的元素
`sqrt(v)<eps`，矩迁移后约 7.68%。因此转角倍数不必严格等于 `1/a`。
这种有效动力学变化必须记录，不能称作无损优化器迁移。

只清矩保留成熟 step 也不是 fresh Adam。在 eps 不主导时，首步因子变为
`(1-beta1)/sqrt(1-beta2) ≈ 3.162`，而正常 fresh Adam 首步因子约 1。
独立标量原生 Adam 测试复现了这项差异。该 head 的实际 eps 仍可能削弱首步，
不能统一假定所有元素恰好放大 3.162 倍。

若未来确实要给 K fresh step，需要独立 K 参数/优化器状态或等价的按切片
时钟实现，并验证所有调用与序列化路径；**不能把联合 kv_proj 的 step 直接清零**。
本次反例同时清整块矩与 step，实测改变 V 更新；不能将这整块重置的差异
全部归因于 step 一项。

## 5. 验证、文件与复跑

实现：[bench/qknorm_plasticity.py](../../bench/qknorm_plasticity.py)。
新增 [数值测试](../../tests/test_qknorm_plasticity.py) **6 passed**，覆盖：
Q/K 缩放不碰 V、三种安全状态操作保留 V/共享时钟/其他状态与源副本、
正确的矩尺度、包含 eps 的 Adam 坐标变换等式、旧时钟清矩与 fresh step 的区别。

主探针另断言：actor 原权重前后相同；原 optimizer state 未变；原 checkpoint
文件 SHA 未变；所有梯度有限。未写出迁移 checkpoint，未调用训练入口或调度器。

本地证据：`.work/qknorm-plasticity-2026-10-16/plasticity.json`、`probe.log`、
`validation.json`、`evidence-manifest.json`。这些文件已存在，但为忽略的本地
产物；报告保留关键源身份、数值和限制，不依赖其成为可移植资源。

```sh
PYTHONPATH=.claude/worktrees/heuristic-kirch-e482e8:python:oracle:. \
  .work/external/danlm-venv/bin/python bench/qknorm_plasticity.py \
  --checkpoint .work/qknorm-mechanism-2026-10-16/remote/qk-both/500M-update-094488.pt \
  --fixture .work/stack-fixture-2026-10-14/rank1-fixture.pt \
  --batches .work/qknorm-mechanism-2026-10-16/results \
  --output .work/qknorm-plasticity-2026-10-16

PYTHONPATH=python:oracle:. .work/external/danlm-venv/bin/python \
  -m pytest -q tests/test_qknorm_plasticity.py
```

2026-10-16 后续：用户批准的 [隔离 CUDA 与完整 actor loss 验证](qknorm-cuda-gate-2026-10-16.md)已在 L40S 通过（7937）；仍未接入 resume、未启动训练，不能推断持续训练或棋力收益。
