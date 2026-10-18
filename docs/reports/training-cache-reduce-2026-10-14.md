# 修复基线上的公共历史缓存与归约对照（2026-10-14）

## 当前状态与边界

7325 修复稳定性通过：3 张健康 L40S、8 ranks 各 120 次完整 PPO 更新，
梯度范数记录全部有限，逐更新参数 checksum 检查通过；57 个文本证据文件
大小/SHA256 核验通过。详见 [fixture 与稳定性记录](training-fixture-backward-2026-10-14.md)。
这是有界稳定性证据，不是长期稳定性保证。

用户授权继续优化。已提交唯一新作业 **7366（gz-cache-reduce）**；
截至 2026-10-14 21:33 UTC，7366 exit 0，三张健康 L40S。两组完整 PPO
对照和数值门槛均完成，114 个文本/脚本文件通过大小与 SHA256 核验。
仅现有集群，normal、4 GPU、96 CPU、35 分钟上限，工作负载 timeout 1980 秒。
逐卡 CUDA 检查后至少两张健康卡才运行，私有 MPS；最终报告实际健康卡数。
不修改或重启主线，不保存主线 checkpoint，不租 GPU。

## 隔离实现

两项候选分别与相同的修复后 page-span 基线比较，不测组合：

- 公共历史缓存：将 KV/memory pool 的按行高级索引换成 index_select。
  保留行顺序、重复索引、零填充、缓存生命周期、完整历史及公共输入边界。
- 梯度归约：复用 pinned CPU 缓冲区，异步 D2H 后同步当前 CUDA stream，
  再执行原 gloo all_reduce；H2D 返回后按原真实 rank 数归一化。
  保留缺失梯度补零、裁剪与 optimizer 数学。测量缓冲区分配/传输收益，
  不声称消除 rank 到达不均衡，未切换 NCCL。

所有基线和候选保留 fix.py：仅 CUDA grad-enabled 的
HistoryActor._attend_by_match 使用 math SDPA。无 packed 投影，无超参变更。
此修复成本不计作缓存优化收益。

## 验证与测量计划（尚未完成）

本地 tensor-only CPU 检查覆盖重复索引、零填充和分页边界，缓存值完全一致；
脚本通过编译/语法检查。GPU 路径仍必须通过作业内门槛：

1. 从相同冻结 init 分别恢复，四次真实 rollout；候选每次 gather 与原索引
   bitwise 比较，选择/历史前缀/RNG 一致，logp 差及完整历史重算均不超过 1e-4。
2. 两个 gloo ranks 比较原归约与候选，覆盖缺失梯度、无真实 rank、缓冲复用；
   count 和有效梯度必须完全一致。传输完成前不得读取 host 数据。
3. 只有全部通过才开始两项独立完整 PPO 对照。每阶段 8 ranks 各 120 updates：
   24 warmup + 六对 AB/BA，每臂每 block 8 updates、前 2 排除，每臂保留 36。
   profile/diagnostics/hooks 关闭，保留原逐更新 checksum 检查。
4. 回收完成回执与文本 SHA 清单，本地重算 analyze.py 对齐远端 summary；
   按配对 block bootstrap 报告秒/update、吞吐变化及区间。

不包含 checkpoint 保存，不是棋力证据；不将收益与旧 page-span 测量简单相加。
若 gate/稳定性失败，保留负结果，不放宽门槛或自动无界重试。

## 身份与证据位置

冻结 source SHA：`ea07296baf000b83a0b0d4a858f07be8be0799280c122b6dbd14e27f0b70e438`。
init SHA：`72837de605ce1a49aeaee0ebf55ab7b12ee22d2fa5862edcd456820ac93fae26`。
本地 `.work/stack-cache-reduce-2026-10-14/` 保存 submission.json、state.json、
脚本与逐文件 SHA；远端对应 `~/guanzero/experiments/stack-cache-reduce-2026-10-14`。
这些忽略目录是本地证据，不是可移植依赖。结束时再次核验生产完整
source_identity 和 main manifest；7325 收尾时两者均未改变。

提交脚本 SHA256：`f5a27802dc897eccccb716afa4fde95a0137293dac449e98119b9751dcaef7be`。
候选脚本 SHA256：`32eb50c1405f07605f53e4f9b08ffda335025ac6f2facafb9436fe9ea472b675`。

## 21:23 UTC 进度记录

数值门槛通过：19,656 条 rollout 记录的选择、历史前缀和 RNG 一致，
logp 最大差为 0，完整历史重算最大差 2.86102294921875e-5（门槛 1e-4）。
2,572 次缓存 gather 逐位一致；两个归约 ranks 各六个案例逐位一致。
cache 阶段完成，reduce 阶段 warmup 到 index 16；尚未回收最终全 rank
完成记录或计算完整配对区间，此时不据日志尾部宣称提速。

## 最终结果（2026-10-14 21:33 UTC）

两阶段均有 8 ranks，各完成连续 index 0–119；逐 rank 和阶段/作业完成回执齐全。
每阶段每臂保留 36 个更新，六对 AB/BA block；无 profile/诊断 hook。
本地 analyze.py 与回收脚本一致，本地重算 JSON 与远端 summary 完全一致。

| 独立对照 | 基线秒/update | 候选秒/update | decisions/s 基线 → 候选 | 吞吐变化 | 配对 block bootstrap 95% 区间 |
|---|---:|---:|---:|---:|---:|
| 公共缓存 index_select | 4.1072 | 4.0572 | 15,956 → 16,153 | +1.23% | +0.18% 至 +2.19% |
| pinned host 归约缓冲 | 4.0759 | 4.1180 | 16,079 → 15,915 | −1.02% | −2.68% 至 +0.68% |

缓存在本次短测中有小幅正收益，区间下界接近零。归约缓冲优化未证明提速，
保留负结果，不推荐与缓存直接组合。独立运行的基线各自恢复同一冻结 init，
不能将两项差值或历史 page-span 收益相加。归约结果只覆盖 staging/传输实现，
不能据此判断 rank 到达等待已经解决；下一步若继续，应针对实际到达不均衡和
公共历史编码成本设计新证据，而非自动重复本次候选。未提交后续作业。

三个健康卡的布局不是四卡生产复现。完整 PPO 计时不含 checkpoint 保存，
不证明棋力或长期稳定性。保留原参数 checksum 检查，所有更新正常完成。
完整生产 source_identity 与冻结源码一致，main manifest 字节一致；主线未改或重启。

回收：`.work/stack-cache-reduce-2026-10-14/remote-results/`，内含
`artifact-manifest.json`、`final-verification.json`、`summary.json` 和原始记录；
本地 `validation.json`、`remote-summary.json`、`local-analysis.txt` 保存核验/复算证据。
本轮收尾后暂停自动检查。
