# Kaggle 约 30 小时续训计划 — 2026-09-24

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
