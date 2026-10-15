# Training stack：profile、learner rollout 投影与 page-span（2026-10-13）

状态：profile、page-span 完整 PPO 对照及投影诊断测速均已完成；
投影数值 gate 未通过。主线训练配置未修改。用户授权按 profile →
投影对照 → page-span 对照的顺序执行；投影在 gate 被阻断后先完成 page-span，
再单列投影的诊断测速，不能将它解读为已通过采用门槛。

## 范围与来源

- 从 Max-Velocity 实际生产目录 `train-src` 冻结 256 个源码文件和原生引擎扩展，
  从主线 `latest.pt` 复制独立输入；实验只写 `experiments/stack-ab-2026-10-13/`。
- 生产源码记录 revision `fc00e2ecbe34b99cbf0fdb74bf1cae25857f7841`，
  完整源码 SHA256 `ea07296baf000b83a0b0d4a858f07be8be0799280c122b6dbd14e27f0b70e438`。
- 输入 checkpoint SHA256
  `72837de605ce1a49aeaee0ebf55ab7b12ee22d2fa5862edcd456820ac93fae26`。
- 2 张 L40S、8 ranks、每 rank 128 桌，64 steps/update、8 matches/minibatch，
  保持全局 1,024 桌和 65,536 环境决策/update；使用独立 MPS daemon。
  此硬件布局不能替代 4 GPU 生产布局的精确速度。
- 模型宽 128、4 层、8 heads，FP32、TF32 关闭、完整公共历史、完整合法动作支持。
  原 PPO、辅助头、近期快照池和快照频率不变。
- Slurm 作业 6896 完成 profile 后在 gate 失败；6900、6901 保留后续 gate
  失败记录。6909 完成独立 page-span gate 与测速；6910 为投影诊断测速。
  测量作业上限 35 分钟 / 内部 timeout 1,980 秒；诊断作业 15 分钟 / 780 秒。
  不租用付费实例，不重启、取消或改写主线任务。

## 实验方法

1. 从固定 checkpoint 恢复独立 trainer；24 次完整 PPO 更新填充历史后，
   开启现有同步 phase timers 做 8 次更新，排除切换后最初两次。
   各 rank 记录 collect、learn、gradient reduce、host upload 等分项。
   Profile 数据不计入吞吐比较。
2. 在训练过的 checkpoint 上运行 CUDA 数值 gate：比较原路径与折叠投影的
   候选 log-prob、相同随机数下的选择和 sampler 状态；核验有梯度路径不受开关影响；
   用真实采样公共历史比较缓存采样与完整历史重算，并核验 page-span 的有效位置。
3. 投影和 page-span 分别与原配置对照。每项都从同一 checkpoint 重新开始，
   warmup 24 updates，6 对 AB/BA blocks，每 block 8 updates，最初两次排除。
   每臂预期 36 次计时更新。每个 block 清理瞬态缓存，保留进行中的环境及 PPO 参数。
4. 完整更新时间包住 `trainer.update()`，包含采样、真实 PPO 优化、梯度同步、
   cache trim、metrics 写入及 rank 参数一致性检查；取同一 update 各 rank 墙钟最大值。
   分母不使用仅计 collect 的 `global_decisions_per_sec`。
5. 以整个 AB/BA block 对做 bootstrap，保留时间漂移和同进程采样的局限。
   这不是多个独立训练种子的棋力实验，FP32 计算顺序变化也可能使后续轨迹分叉。
6. 原计划在两个候选均通过且单独测速改善后考虑组合臂。投影 gate 失败，
   因此没有运行组合臂，也不能将两个单项收益相加。

`trainer.update()` 不含 checkpoint 保存。实验不保存续训模型，不能把结果表述为
已包含保存、同步、排队及作业切换的全天生产吞吐改善。

## 投影实现

实验代码仅在单独脚本中给 learner rollout actor 安装实例级包装，不修改生产模块。
只在 `no_grad` 且每条公共流一个 query 的采样调用中使用折叠计算：
把 key 投影移到 query 端，value 投影移到 attention 加权结果端；
key bias 在同一 query 的 softmax 中抵消，value bias 正常保留。
有梯度的 PPO forward/backward 始终使用原实现，快照策略路径不变。

## 本地脚本验证

- 单进程投影、profile、组合开关的完整更新检查，以及双进程 page-span 更新检查通过。
  这些小规模 CPU 检查验证脚本和同步流程，不用于估计 GPU 提速。
- FP32 合成输入覆盖 1–64 rows、1–2,049 tokens、变长候选集、不同可见前缀：
  相同随机数下选择和 sampler 状态一致。
- 随机初始化 fixture 的真实采样中，候选 log-prob 最大误差约 `9.54e-7`；
  256 行行为 log-prob 与完整历史重算误差约 `4.77e-7`。
- 已验证有梯度路径的 log-prob 和 `kv_proj.weight` 梯度逐位不变；
  这不声称新采样路径在所有设备、所有参数上逐位一致。
- Resume fixture 的 page-span gate 确实减少 padding：256 → 192 positions，
  所比较的有效编码逐位相同。训练 checkpoint 的 CUDA 结果见下。

## GPU profile

计时来自 8 ranks × 6 次保留更新的分项平均；开启同步 timers，会扰动速度，
不用于 A/B 的吞吐分母。

| 阶段 | 平均秒数 | 解读 |
|---|---:|---|
| public cache / collation | 1.852 | 采样主要成本 |
| learner prefill | 0.165 | 与上一项合计占采样分项约 66% |
| actor / sampling | 0.496 | 已包含策略计算与采样 |
| learner backward | 0.732 | learner 分项之一 |
| gradient reduce | 0.676 | 占 learner 分项约 28%，包括等待，不能全称为网络通信 |
| learner forward | 0.517 | learner 分项之一 |

## Page-span：通过本次 gate，完整 PPO 提速

| 指标 | 原配置 | page-span |
|---|---:|---:|
| 完整 update 平均秒数 | 5.2474 | 4.7584 |
| 环境决策/秒（包含 PPO learn） | 12,489 | 13,773 |
| 各 update 最慢 rank collect 平均秒数 | 2.8819 | 2.4093 |
| 各 update 最慢 rank learn 平均秒数 | 2.4574 | 2.4215 |
| rank 0 平均历史 prefix | 581.1 | 597.4 |
| 所记录峰值 prefix | 2,140 | 2,375 |
| 单 rank 最大 allocated GiB | 2.195 | 1.809 |

每臂 36 次保留更新、6 对 AB/BA blocks。吞吐 **+10.28%**，update 时间 **−9.32%**；
按 block 对 bootstrap 的吞吐改善 95% 区间 **+7.50%～+13.03%**。
该区间只反映这一次交替运行中的 block 波动，不是独立种子/不同作业复现区间。
collect 与 learn 各自取最慢 rank，可能来自不同 rank，二者不能直接相加当作整次 wall time。
内存是单进程 PyTorch allocated 峰值，不是整张 GPU 占用。

CUDA gate：

- 训练 checkpoint 的公共历史缓存比较：padding 512 → 384，真实 prefix 最大 333，
  有效编码最大误差 0。该 fixture 刻意截取公共前缀以覆盖 padding 缩减。
- 现有 CUDA page-span parity 测试通过。
- 同 checkpoint 的两次真实 rollout（分别关闭/开启 page-span，每次 4 × 64 steps）
  对比 19,656 条 compact buffer 记录，所有非 log-prob 字段 hash 一致，
  sampler RNG 一致，log-prob / behaviour log-prob 最大差均为 0。
- 各臂抽取 256 行用完整公共历史重算，最大误差均为 `2.8610e-5`，
  小于本次 `1e-4` 行为重算门槛。没有改变 public-information 边界。

## Learner rollout 投影：数值 gate 未通过

合成与 CPU fixture 通过不代表成熟 checkpoint 通过。保留了以下负结果，
没有把失败的真实 CUDA gate 隐去：

1. 初始 `2e-5` 绝对 log-prob 门槛出现约 `3.05e-5` 的误差。
2. 引入 FP64 attention 参考进行诊断；真实数据也越过初始门槛。
3. 将实际路径比较门槛校准到 `1e-4` 后，曾用“投影必须距离 FP64 小于
   `1e-4`”的额外检查失败（约 `5.70e-4`）。原 FP32 路径自身也有误差，
   因此该检查不能单独用来判断新路径是否更差。
4. 最终检查同时记录原路径和投影对 FP64 的误差，并继续要求两条实际路径
   的绝对差不超过 `1e-4`。第 31 次真实采样比较仍失败：1,127 个候选中
   2 个越界，最大差 `1.373291e-4`。同例原路径对 FP64 最大误差
   `9.155273e-5`，投影对 FP64 `4.577637e-5`。投影该例更接近参考，
   仍不满足约定的实际路径 parity 门槛。

最终 gate 没有完成全套成熟 checkpoint 的采样/RNG/行为重算检查。
投影测速是隔离的性能诊断，**不能作为主线采用资格或训练质量证据**。
没有继续放宽门槛，也没有改动生产模型实现。

投影诊断完整 PPO 结果：

| 指标 | 原配置 | learner rollout 投影 |
|---|---:|---:|
| 完整 update 平均秒数 | 5.2237 | 4.7938 |
| 环境决策/秒（包含 PPO learn） | 12,546 | 13,671 |
| 各 update 最慢 rank collect 平均秒数 | 2.9296 | 2.4716 |
| 各 update 最慢 rank learn 平均秒数 | 2.4324 | 2.3971 |
| rank 0 平均历史 prefix | 570.9 | 580.3 |
| 单 rank 最大 allocated GiB | 2.092 | 2.054 |

每臂 36 次保留更新。吞吐 **+8.97%**，update 时间 **−8.23%**；
配对 block bootstrap 的吞吐改善 95% 区间 **+7.48%～+10.47%**。
保留的 rank 0 投影臂共调用折叠路径 2,304 次（36 updates × 64 steps）。
此速度结果只说明该候选值得进一步解决数值验证问题；PPO 梯度路径未替换，
但采样 FP32 顺序差异可能改变后续轨迹，所以不能宣称整个训练过程完全等价。

## 适用范围与下一步

page-span 已有本次完整 PPO 与真实 rollout parity 证据；下一项应是在生产同型的
4 GPU / 8 rank 隔离布局复现，而不是把 2 GPU 的 10.3% 直接外推给主线。
投影应先解决数值门槛与完整行为重算验证。没有进行棋力评估，不能声称棋力提升。
两项试验均未降低精度、缩短训练历史、裁剪合法动作或改变 PPO 目标。

## 产物

本地 `.work/stack-ab-2026-10-13/`：实验脚本、CPU gate/完整更新日志、
冻结与提交回执。远端同名实验目录保留冻结源码、输入 checkpoint、失败 gate 日志、
各阶段各 rank 的 `timings.jsonl`、完成/退出回执和分析结果。
GPU 结果已回收到本地 `remote-results/`：201 个文件逐一核验大小与 SHA256；
`artifact-manifest.json` 为清单，`summary.json` 为完整汇总。
用本地分析脚本重新计算，结果与远端一致；两项对照都验证为 8 ranks × 120 updates，
每臂每 rank 保留 36 次计时。`span-replay-gate/gates.json` 保留完整 rollout gate，
`gates-relative.log` 保留最终失败的投影 gate。

`measurement-job-exit.json` 和 `projection-diagnostic-job-exit.json` 均记录 exit 0；
6896、6900、6901 的 gate 失败 exit 1 回执也保留，所有实验作业均已退出。
`final-verification.json` 重新核验 256 个生产源码文件与冻结哈希一致，
生产 manifest 与实验前一致。没有保存或替换主线 checkpoint。
提交时脚本哈希与最终脚本哈希分别保留；投影诊断作业排队期间增加的 page-span
完整 rollout replay 检查在其启动前上传，未改动计时 runner 或实验投影实现。

`.work/` 是忽略的本地证据，不是可移植运行依赖。
