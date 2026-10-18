# 公共历史编码与 rank 等待诊断（2026-10-14）

用户决定保留 7366 的缓存 index_select 优化，不采用 pinned host 归约缓冲方案，
授权继续现有集群隔离诊断；主线源码、配置、checkpoint 和运行状态不修改。
7366 缓存吞吐 +1.23%，归约 −1.02% 未证明收益，原负结果保留。

已提交 **7392（gz-encode-wait）**，35 分钟/1980 秒有界工作负载，
normal、4 GPU、96 CPU，逐卡检查后至少两张健康卡才运行；私有 MPS。
结果：exit 0，三张健康 L40S；不将三卡结果称四卡生产复现。

## 已执行方法

始终启用 masked SDPA backward math 修复、page-span、index_select 缓存。
归约保持原 gloo 算术，明确不安装 pinned host 归约候选。两阶段各从原冻结
init.pt 恢复，8 ranks；保持 FP32、公共输入边界、完整历史和训练超参。

- passive：24 warmup + 16 次记录，记录各次 reduce 进入/退出 perf_counter_ns、
  real/count，以及公共 encode_stream 调用形状、grad-enabled 和 CPU 调用时长。
  没有新增归约 barrier 或 CUDA 同步；CPU 调用时长不等于 GPU 执行时长。
  旧 runner 的 base/cache 标签在此两者都启用缓存，不构成 A/B 对照。
- sync：24 warmup + 8 次诊断，启用原 collection/learn 同步分项；公共编码
  前后 CUDA 同步，记录有效 token 数和 padding 形状；原归约 D2H 后插入
  barrier，再测 host all_reduce 和 H2D/unpack。barrier 时间包括到达等待及
  barrier 本身开销，不能声称生产纯等待比例。保留 real/count 与原数学。

同节点 monotonic 时间用于对齐 passive 各 rank 同序号归约事件；先核验事件
序列长度/count，再比较到达偏差、rank 工作量与共享 GPU 映射。sync 数据受
同步和 MPS 竞争扰动，不能当无诊断开销完整 PPO 提速证据。编码 forward
计时不含其 backward，已有 learn 分项补充判断，不能将相嵌套计时直接相加。

脚本通过 Python 编译、shell 语法和原归约变换锚点检查；CUDA 运行尚待验证。
这是定位实验，不直接推出新优化收益；结束回收文本与大小/SHA 清单，核验
每 rank passive 40/sync 32 更新和完成回执，重验主线完整 source_identity/
manifest。失败保留证据，不盲目重试。依据结果再确定具体优化与等价性门槛。

本地 `.work/stack-encode-wait-2026-10-14/` 保存脚本、submission/state 和 SHA；
远端同名 experiments 目录。source/init 沿用 stack-ab 冻结身份，数值修复
和缓存代码与 7366 相同。不含 checkpoint 保存，不作棋力结论。

## 完成与核验（多伦多 2026-10-14 晚间）

SSH 恢复后回收成功。119 个文本/脚本文件均通过大小及 SHA256 校验。
两个阶段均有 8 ranks，passive 各 40 更新、sync 各 32 更新，连续 index 与
逐 rank、阶段、作业完成回执齐全。生产完整 source_identity 与冻结源码一致，
main manifest 字节一致；主线没有修改或重启。
本地 `analysis.json` 由同目录 `analyze.py` 从 remote-results 重算。

### 已测量的瓶颈证据

以下同步分项为每 rank 每 update 平均，不能与不同嵌套层级相加：

| 同步诊断分项 | 秒 |
|---|---:|
| rollout public_cache_or_collation | 0.9973 |
| rollout actor_and_sampling | 0.3382 |
| rollout learner_prefill | 0.1031 |
| learner forward（含公共编码等） | 0.5914 |
| learner backward | 0.6566 |
| learner gradient_reduce（含等待与传输） | 0.8191 |
| 公共 encode_stream forward（嵌套于 learner forward） | 0.2504 |

encode_stream 记录均 grad-enabled：sync 3,520 次调用，passive 7,040 次。
因此这个 hook 覆盖 learner 公共编码，**没有覆盖 rollout 增量 KV 编码的内部**。
同步 public_cache_or_collation 约 1 秒是当前较大的可定位目标，但不能把它全
称为 Transformer 编码；其中仍有 append/gather/metadata/collation 待拆分。
learner 编码按输入 token 槽位加权 padding 比例 22.41%，这不是计算浪费比例，
更不是可直接实现的 22% PPO 提速。

passive 成功对齐 464 个归约位置：各 rank 每 update 事件数相同，count 一致，
且等于 real ranks 总数。归约进入时间的跨 rank 差平均 **16.64ms**，p95
**48.57ms**。整个 reduce 调用平均 0.7366 秒/rank/update，不能全归为通信。
各 rank 平均真实 minibatches 在 27.375–28.0，工作条数相近仍有到达差异。
rank 3、7 分别最后到达 102、117 次；其共享 GPU 映射分别为 0、1。
完整 GPU 映射由 rank % 3 得到：GPU0 ranks 0/3/6，GPU1 ranks 1/4/7，
GPU2 ranks 2/5。最后到达次数分别合计 133/238/93；样本不能单独确定
硬件/MPS 竞争或序列复杂度的因果关系，也不能断言某个 rank 一直最慢。

sync 拆分平均为 pack+D2H 0.0442 秒、arrival barrier 0.2447 秒、host allreduce
0.4607 秒、H2D/unpack 0.0625 秒。234 个位置的 count/real 对齐通过。
新增 barrier、自身同步和 MPS 调度会改变时间；这些不是生产纯通信或纯等待
比例，不能把 0.2447 秒直接当作可节省时间。passive/sync 恢复同 init 但
轨迹和计时有扰动，不把阶段差当速度对照。

### 下一步具体方向（建议，尚未提交新作业）

1. 优先细拆 rollout 缓存的 `_append`/`_merged_append`、`_gather_memory` 和
   元数据构建；用相同真实公共历史输入检查是否存在重复 memory gather 或
   相同调用内可复用的索引/元数据，然后选择一个保持值和生命周期不变的候选。
   当前证据尚不足以直接删除或跨 update 复用编码；权重更新后必须失效。
2. learner 侧可检查当前 length-group 内的 padding 分布，尝试更紧凑的分组
   或精确尾部裁剪；需要完整候选/chosen logp、历史重算、梯度有限性门槛，
   不放宽 1e-4，不改变完整动作历史/公共输入/数学目标。
3. rank 等待优先分析每 minibatch 序列长度、实际 kernel 时间与到达偏差的
   关联，再考虑 GPU 内工作分配。现有平均 minibatch 数不能解释全部差异；
   不采用已失败的 pinned 归约方案，也不据此直接变更 minibatch 顺序或 rank 数。

诊断完成，本轮自动检查暂停；没有新增完整 PPO 提速测量或新作业。
缓存优化继续保留在隔离基线，归约缓冲候选不采用。任何后续收益需要关闭
诊断开销后的独立配对完整 PPO 测试，不包含 checkpoint 保存或棋力结论。
