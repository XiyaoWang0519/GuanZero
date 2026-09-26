# Kaggle 约 30 小时续训计划 — 2026-09-24

> Historical record. The proposals, budgets and next steps below belong to
> this dated experiment. The active route is now random-start Transformer
> self-play, with old MLPs used only for evaluation; see
> [DESIGN.md](../DESIGN.md) and [STAGE_C_TODO.md](../STAGE_C_TODO.md).
> Preserve the recorded results and artifacts; do not launch an old plan as
> the new experiment or infer current provider state from its snapshot.

用户授权使用剩余免费 GPU 额度推进 GuanZero，并要求 W&B 实时监控。当前账户剩 29.90 小时；付费预算 $0。本轮会话累计实际已用与未完成会话的保守预约之和上限 29.75 小时，不跨配额刷新扩大授权，最后为保存结果留余量。Kaggle 官方单会话上限 12 小时，脚本按不超过约 9.6 小时的分段任务执行，每次启动前重新核对账户额度。

## 下一步与实验问题

从已验证的五小时终点 `main-final.pt` 继续训练，比较原 policy LR `1e-5` 与半 LR `5e-6`。两臂 critic LR 都是 `1e-4`，唯一计划内算法变量为 policy LR。每臂占一张 T4，1024 环境、引擎 1 线程、torch 1 线程，rollout 64、epochs 4、minibatch 8192；各 8 小时同墙钟预算。seed 2026092601，共同 warm start 权重和 Adam 状态，重建相同的 overnight 初始 league 并加入五小时终点。双方各自动态追加训练快照。环境、对手池统计和随机流重新开始，因此不把结果归因于严格完整状态续训。

源码沿用已测的 `4940246fbd520ff6bd9e246e729f425b69dfcba8`。checkpoint SHA256 为 `ebb614f35f825e835c9d5b18bdc5a6efb0f79d5ba5ed9bfcdf44eca98ae139fe`。本轮不再使用已退化的优势过滤；候选扩展与 Transformer 等结构改动留待单独实验。

## 分段分配

1. R1：两臂并行 8 小时，随后做开发评测；包括设置、预检、评测、退出的脚本预算 34,500 秒、通过编辑器提交以保留 Secrets 授权；UI 未提供单独 timeout，脚本独立截止 34,500 秒，另以平台公布的 12 小时上限保守预约额度。
2. R2：根据 R1 结果优先用第二个训练 seed 复验 LR 对照；若明确退化，停止该配置，把时间用于有价值的续训或验证。先冻结下一轮配置和选择规则再启动。
3. R3：用剩余额度推进经复验的配置，并做最终独立评测；如继续训练已经无收益或遇到可靠性问题，保留证据后收尾，不能只为消耗额度运行空任务。

R2/R3 不是已运行任务。每轮结束下载、SHA256 校验后再提交下一轮，不重叠租用多个 GPU Notebook。自动检查依赖本机 Codex 在线；已经提交的单段云端训练可独立完成，但关机期间无法承诺下一段无缝接续。禁止将 Kaggle/W&B 密钥写入源码或 Dataset。

## 评测边界

训练期间每 2 小时跑一次固定内部开发探针（512 duplicate deals、128 full matches，seed 2026092610）。每次至多 240 秒，其耗时计入同墙钟预算。短探针用于画曲线，不充当独立最终验证。

R1 两臂终点各对起点评估 2000 duplicate deals、500 full matches、固定风格和 3 个样本外风格；另做两臂直接对战，开发 seed 2026092611。完整原始 pair scores、置信区间和逐项结果保留。先比较同预算两臂，训练日志胜率不能替代独立棋力证据。

本轮最终内部 seed 2026092691（4000 duplicate deals、1000 full matches），最终 DanLM seed 2026092699（4000 同牌 deals），两者在确定最终候选前不运行，不用最终集调参。DanLM 在后续有已验证依赖的云端 CPU 评测中执行；最终报告区分内部完整比赛与 DanLM 单局对照，不宣称 G9 或外部 SOTA。默认模型不自动替换。

## W&B

[私有项目](https://wandb.ai/mcraenanren-university-of-toronto/guanzero)，工作区已通过现有凭据核验。两个训练 runs 为 `kg26r1-control`、`kg26r1-half_lr`，组 `kaggle-20260924-r1`。独立的 `connection-check-no-training` 已验证指标上传和 API 读回，不代表 GPU 训练已启动。

每个 update 上报全部有限数值指标，包括 policy/value/auxiliary loss、KL、entropy、clip fraction、importance ratio deviation、policy/critic gradient norm、explained variance、return/value、实际 epochs、league 统计、采集/学习耗时、累计及逐 update 吞吐、学习方决策量、CPU RSS 与显存。完整解析后的配置（含默认参数）、模型规模/结构、实际 optimizer LR、源码/模型 SHA 和评测 seed 保留。SDK 同时记录系统 GPU/CPU 指标。默认以学习方决策数对齐训练曲线，评测指标与训练指标分组。

凭据只从该 Notebook 获授权的 Kaggle Secret `WANDB_API_KEY` 读取；开启线上记录失败时，训练不得静默退回无监控模式启动。W&B 不自动上传权重或源码；Kaggle 私有输出保留 checkpoint、optimizer、league、配置、日志和评测原始数据。

## 执行准备

工作目录 `.work/kaggle-continuation-20260924/` 包含 state、worker、probe、runner、构建脚本和代码 SHA 清单。训练 checkpoint 每 300 秒保存，定期保留完整 snapshot。独立普通线程实施截止和子进程清理；磁盘余量不足或任一训练臂失败会停止本轮其他训练臂。每个预检、训练、探针和评测子进程均有超时。云端的 W&B Secrets 联通及 CUDA checkpoint/resume 预检通过后才进入 8 小时训练。

官方规则参考：[Notebook 规格](https://www.kaggle.com/docs/notebooks)、[GPU 配额](https://www.kaggle.com/docs/efficient-gpu-usage)。实际额度和终态以账户 CLI/API 回读为准。

## 启动与实时监控核验

2026-09-24 21:07 UTC 已核验 R1 version 4 正在 Kaggle GPU T4 x2 上运行，两个独立训练进程已经产生真实更新。W&B 回读 control 为 200 updates、约 648 万学习方决策，half_lr 为 196 updates、约 633 万学习方决策；实际 policy LR 分别为 `1e-5`、`5e-6`，critic 都为 `1e-4`。每条 run 有 61 项用户配置以及 130 余项摘要字段（含训练进度、统计和元数据）；这不是棋力结果，首个周期评测尚未到时间。

- [原 LR 训练曲线](https://wandb.ai/mcraenanren-university-of-toronto/guanzero/runs/kg26r1-control)
- [半 LR 训练曲线](https://wandb.ai/mcraenanren-university-of-toronto/guanzero/runs/kg26r1-half_lr)

账户最新回读已用 0.44 小时、剩余 29.56 小时，均包含既往测速。本轮按单会话最大 12 小时保守预约，尚未将已使用的会话时间与完整预约重复计费。Codex 账户共享周额度剩 71%，其他窗口未返回。检查自动化 `guanzero` 已设为每 30 分钟；健康时安静，仅重要变化通知。

证据保存在 `.work/kaggle-continuation-20260924/latest-check.json`、`wandb-live-proof.json` 和 `state.json`。用户授权后，密钥已保存并关联 Kaggle Secret；云端 CPU 联通预检与当前 GPU 训练上传均通过。当前版本使用编辑器的“仅本次以 GPU 运行”，远端 pull 显示的默认 CPU 草稿设置不能代表当前实际分配。

## R1 完成与 R2 复验

2026-09-25 已回收并核验 R1 全部输出，见 [R1 结果与 R2 冻结选择规则](kaggle-r1-results-2026-09-25.md)。第一轮内部结果支持继续复验，但不支持宣布半 LR 优于原 LR。

05:13:59 UTC 已通过编辑器在同一私有 Notebook 提交 version 5，版本名 `R2 Second Seed LR Comparison 8h`。UI 回读 GPU T4 x2；远端代码 AST 与本地冻结代码完全一致，SHA256 `7fdf92cd6b989db52613ea0ae54b6b6bd11c65858352f564ef35443b38510ed4`。Secret 保持关联，输入仍为两个已授权私有 Dataset。

本轮训练 seed 2026092602，终点开发 seed 2026092612；从同一五小时起点和 Adam 状态重新跑原 LR/半 LR 两组，每组 8 小时。当前工作目录 `.work/kaggle-continuation-20260925-r2/`；共享检查脚本与状态仍在原 `.work/kaggle-continuation-20260924/`，已按当前轮动态读取 W&B run IDs。

- [R2 原 LR 曲线](https://wandb.ai/mcraenanren-university-of-toronto/guanzero/runs/kg26r2-control)
- [R2 半 LR 曲线](https://wandb.ai/mcraenanren-university-of-toronto/guanzero/runs/kg26r2-half_lr)

启动前已用 campaign 约 8.19h，当前 R2 保守预约 12h，合计 20.19h；R3 仍未启动。05:15 UTC 账户剩余约 21.69h。所有支出为免费额度，付费 $0。下一轮须先回收、核验本轮输出并根据结果决定，不能跨配额刷新扩大 campaign 的 29.75h 上限。

05:18:40 UTC 已验证 R2 两臂均有 14 个真实训练 updates，各约 46.3 万学习方决策，正确 seed 与实际 LR、有限数值指标、实时 heartbeat 均通过。两张 GPU 的保存/恢复预检完成，正式训练已启动。每条 run 有 61 项用户配置；监控已切换到 R2，按每 30 分钟唤醒，健康时保持安静。账户 Codex 周额度剩 63%，其他窗口未知。证据为 R2 目录下 `wandb-live-proof.json`、`submission-proof.json`；默认模型未替换。

## R2 完成与 R3 续训

2026-09-25 13:31 UTC 已确认 R2 COMPLETE，所有 178 项 artifact SHA 及原始配对数据审计通过。详见 [R2 结果及第三轮冻结决策](kaggle-r2-results-2026-09-25.md)。两轮等权主净级差支持选择半 LR，但 low-lead 风格有复现退化，不宣称全面优势。

13:47:46 UTC 已提交同一私有 Notebook version 6（scriptVersionId 352705092）。实际 UI 为 GPU T4 x2；远端 AST 与本地 SHA256 `4eb5adc00703f36c53d6d313cbc25e4d7178fc702f77e0d43681e4dfc3beb164` 一致。Secret 保持勾选，增加的私有 Dataset 仅含两个已核验父模型与 manifest，约 132 MB。两张 GPU 的 smoke/save/restore 均通过，脚本第 182.804 秒开始正式训练。

R3 分别续训 R1/R2 半 LR 终点各 10 小时，以不同固定 seed 检查进一步训练效果；预先固定 lineage_r2 终点为主要候选。两条线使用同一重建 league，保留各自 Adam，计数器与环境重置。算法源码仍为 4940246。每条线每两小时开发评测现含置信区间；内部最终评测独立 seed 2026092691，结果记录在 final_internal/*。不按最终数据选模型。

- [R1 lineage 续训曲线](https://wandb.ai/mcraenanren-university-of-toronto/guanzero/runs/kg26r3-lineage_r1)
- [R2 lineage 续训曲线（固定主要候选）](https://wandb.ai/mcraenanren-university-of-toronto/guanzero/runs/kg26r3-lineage_r2)

启动前 campaign 累计使用 16.38h，R3 保守预约 12h，合计 28.38h，低于 29.75h 上限；脚本实际截止 11.5h，付费 $0。监控继续每 30 分钟，健康时保持安静。DanLM 本机 CPython3.12/macOS 环境已完成独立 smoke seed2026092680，4/4整deal有效、0排除；最终seed2026092699未使用，需待模型回收后做有截止的 CPU 推理及原始整deal配对审计。

13:51:48 UTC 实际 W&B 回读确认 R3 两条线均完成 4 个正式 updates，各约 13.3 万学习方决策；实际 policy LR 5e-6、critic LR 1e-4，指标有限且 heartbeat 新鲜。每条 run 63 项用户配置。当前 quota 已用 16.55h / 剩 13.45h；已完成会话 16.38h 加当前完整预约 12h 不变。证据在 R3 目录的 submission-proof.json 与 wandb-live-proof.json。自动化已切换为 R3，按半小时检查，健康时安静。
