# Rollout 缓存分项与调用内索引复用（2026-10-14）

用户授权继续现有集群隔离优化；主线配置、源码、checkpoint 与运行状态不改。
唯一新作业 **7452（gz-cache-reuse）**，35 分钟/1980 秒上限，4GPU、96CPU、
normal，逐卡检测后至少两张健康卡才运行，私有 MPS。已 exit 0；最终结果见下方，索引复用未证明完整 PPO 提速。

## 诊断与候选

基线始终保留已验证的 masked backward math 修复、page-span 和 index_select。
不启用被拒绝的 pinned 归约候选，原 gloo 数学与训练超参不变。

短 profile：8 ranks，各 24 warmup + 8 同步记录。分别记录普通 append、
merged append、memory gather 的 inclusive 时间；内部页表构建、行索引、
metadata/event upload、pool index_select 的时间。父子计时不相加；CUDA同步
改变运行节奏，不能当生产纯开销比例。页表构建只是元数据工作的一部分，
其余 Python token/写入布局构造仍在 append 剩余时间内，不把剩余全称计算。

候选复用 append 的精确 pool 行索引，跳过同次 encode 后续 gather 的页表
重建、元数据上传和行索引运算。只匹配相同 entry 对象顺序、长度、pages、
generation 和输出 shape；任何不匹配走原路径。最后一次 append 为子集或
尺寸不同也回退。只缓存索引，不缓存编码输出；同次 encode 结束/异常时
finally 清空，禁止跨 update 或权重变化复用。普通和 merged encode 均覆盖。

## 验证顺序（计划）

本地 CPU tensor-only 45 轮精确对照通过，覆盖页扩容、不同子集/顺序、reset、
prune 和权重变动；19 次命中、71 次回退（包括基线调用）。不以 CPU 结果
代替 CUDA 验证。脚本通过编译与 shell 语法检查。

profile 后从冻结 init 分别恢复基线/候选，各四次真实 collect：验证选择、
prefix、RNG 一致，logp 最大绝对差 <=1e-4，256 行完整历史重算 <=1e-4；
每次复用的 memory 输出必须和原 gather 逐位一致，且必须有实际复用命中。
未过门槛则 set -e 停止，不测速度、不放宽门槛。

通过才进行无 profile/hooks 的完整 PPO 配对对照：8 ranks，各 24 warmup +
6 对 AB/BA，每臂每 block 8 updates、前 2 排除，共 120 更新，每臂保留 36。
每 update 保留参数 checksum 检查。基线是修复+page-span+index_select，
不是主线默认；warmup 和基线不启用新复用。两臂使用相同 wrapper，PROFILE
关闭不额外同步，VALIDATE 关闭不执行参考 gather。

结束回收文本/脚本大小与 SHA 清单，排除 source/init/fixture；核验 profile
8×32、PPO 8×120 及完成回执、命中统计，重算配对 bootstrap 区间并与远端
summary 对齐。重验主线完整 source_identity 和 manifest，保留负结果。
不含 checkpoint 保存、不作棋力结论、不与旧收益简单相加。

本地 `.work/stack-cache-reuse-2026-10-14/` 存 submission/state、脚本 SHA、
local-test.json；远端 experiments 同名目录。source/init 沿用 stack-ab 冻结
身份。仅本轮有界验证，失败不盲目重试，完成后暂停自动检查。

## 2026-10-15 03:23 UTC 进度

7452 RUNNING；profile 阶段完成回执已生成，数值 gate 通过：19,656 条真实
rollout 的选择/prefix/RNG 一致，logp 最大差 0；完整历史重算最大误差
2.86102294921875e-5 < 1e-4。262 次复用 gather 对原路径逐位一致；
762 次 fallback 包括基线调用，不能据此直接计算候选命中率。
完整 PPO 观察到 index 99，尚未结束；不依据尾部样本报告提速。

## 最终结果（多伦多 2026-10-14 晚间）

7452 exit 0，三张健康 L40S。profile 8 ranks 各 32 更新、PPO 8 ranks 各
120 更新，连续 index、逐 rank 和阶段/作业完成回执全部核验通过；每臂36
保留更新、六对 block。113 个文本/脚本文件通过大小与 SHA256 校验。
本地运行回收的 analyze.py，结果与远端 summary JSON 完全一致。
生产完整 source_identity 与冻结源码一致，main manifest 字节一致。

数值检查通过：19,656 条真实 rollout 的选择/prefix/RNG 一致，logp 差为0，
完整历史重算最大差2.86102294921875e-5；262次 gate 复用输出逐位一致。
完整 PPO 候选臂全部48更新/各rank累计24,965次命中、24,187次回退，
命中率50.79%（使用每更新累计计数差，只统计候选臂；不是将gate的混合
fallback当候选命中率）。

| 完整 PPO | 基线 | 索引复用 |
|---|---:|---:|
| 秒/update | 4.078772 | 4.096597 |
| decisions/s | 16,067.58 | 15,997.67 |

吞吐变化 **−0.435%**，配对 block bootstrap 95% 区间 **−1.301% 至 +0.369%**；
更新时间增加0.437%。没有证明收益，不建议采用此复用候选。基线继续保留
此前index_select，不把本次负结果解释为此前缓存优化失效。

### 缓存内部诊断

以下为profile最后6更新×8ranks=48样本，按rank/update平均，同步计时：

| 分项 | 秒 | 调用数 |
|---|---:|---:|
| 普通append inclusive | 0.6264 | 77.375 |
| merged append inclusive | 0.4083 | 64 |
| 最终memory gather inclusive | 0.0858 | 128 |
| metadata/event upload（嵌套） | 0.0694 | 269.375 |
| page_table（嵌套） | 0.0180 | 269.375 |
| row_indices（嵌套） | 0.0596 | 269.375 |
| pool index_select（嵌套，含KV和memory） | 0.1293 | 693.5 |

主要开销集中在普通/merged append，而最终memory gather相对较小。
复用只跳过匹配gather的页表/上传/行索引，不会消除append内部每层KV读取
或编码计算，不能把表中全部元数据时间算作候选可节省量。签名检查等
自身成本可能抵消节省，但本次未单独计时，不能将负收益确定归因于它。

父子计时不能相加；append包含prefill等调用，与collection的单一分项
边界不同。每个timed包装都有CUDA同步、MPS竞争影响，不能按这些数字
计算无诊断开销的生产百分比。page_table只覆盖元数据的一部分，raw token/
write布局的Python构建、layer projection、SDPA、finish和写回仍混在append
剩余，尚未定位到具体算子。

下一步若继续，优先拆append的CPU事件/写布局构建与每层projection、KV读写、
SDPA、FFN/finish，采集每次新增token数、历史span和batch形状；针对占比
最大的部分再设计候选。不能根据本次证据直接缓存跨权重版本的编码、改变
历史边界或缩短历史。未提交新实验；本轮收尾后暂停自动检查。

证据：`.work/stack-cache-reuse-2026-10-14/remote-results/`，其中
artifact-manifest.json、gate/gates.json、final-verification.json及rank原始日志；
本地validation.json、remote-summary.json、local-analysis.txt保存核验和复算。
不含checkpoint保存，不证明棋力或长期稳定性，不与旧page-span收益相加。
