# Kaggle 免费 GPU 测速：已完成

> Historical record. The proposals, budgets and next steps below belong to
> this dated experiment. The active route is now random-start Transformer
> self-play, with old MLPs used only for evaluation; see
> [DESIGN.md](../DESIGN.md) and [STAGE_C_TODO.md](../STAGE_C_TODO.md).
> Preserve the recorded results and artifacts; do not launch an old plan as
> the new experiment or infer current provider state from its snapshot.

用户随后明确批准私有上传；version 2 已完成，实际脚本耗时 337.94 秒，额度消耗约 0.10 小时。最终结果见 [测速报告](kaggle-benchmark-2026-09-24.md)。下文保留原始审批前计划，状态及 840/900 秒预算已被 version 2 的 720/780 秒预算取代。

# Kaggle 免费 GPU 测速：已准备，等待上传授权

已通过官方 Kaggle CLI 验证账户 `xiyaowang0519`：GPU 配额总计 30 小时，已用 0 小时，剩余 30 小时；API 返回下次刷新时间 2026-09-26T00:00:00。尚未上传数据、创建本次 Notebook 或启动 GPU。

测速包使用昨晚冻结源码 `4940246`、B11 起点、相同初始 league，避免把代码差异误当硬件差异。计划在私有 GPU Notebook 中编译引擎、运行 C++ 检查及一次 CUDA 小批量预检，然后比较：单卡 2,048 环境，与两张卡各自运行 1,024 环境时的总决策吞吐。两组均排除 warmup，记录总决策和学习方决策每秒、collect/learn 耗时、显存峰值、GPU 利用率以及实际 CPU quota。

请求 T4×2，最终型号以运行中查询为准。脚本总预算 840 秒，提交时另设 900 秒 Notebook 上限。无充值、无 RunPod 资源；完成退出并保存结果。本次只测吞吐，不做模型强度结论或替换默认模型。

待上传内容为 **327,706,222 bytes（约 328 MB）**，共 120 个文件：必要的 C++/Python 源码、配置、九份训练/对手模型文件。没有 API 凭据、`.env`、Git 历史、个人文档或本地编译二进制。压缩包 SHA-256 为 `e550ad5ac6fe6da92dd309c50fa867f85cf46c4113cb25f53a6bf674e774d33e`。

目标是用户自己的 **私有** Dataset `xiyaowang0519/guanzero-throughput-kit-20260924` 和 **私有** Notebook `xiyaowang0519/guanzero-t4-benchmark-20260924`，不公开发布。自动审批拒绝了上传命令：它要求用户明确同意将这些私有源码和模型传到 Kaggle。命令在执行前被拦截，尚未产生上传或 GPU 消耗。

- [逐文件上传清单](/Users/xiyaowang/Documents/Projects/GuanZero/.work/kaggle-benchmark-20260924/dataset/payload-manifest.json)
- [准备好的远端测速脚本](/Users/xiyaowang/Documents/Projects/GuanZero/.work/kaggle-benchmark-20260924/kernel/benchmark.py)
- [私有 Notebook 配置](/Users/xiyaowang/Documents/Projects/GuanZero/.work/kaggle-benchmark-20260924/kernel/kernel-metadata.json)
- [当前状态](/Users/xiyaowang/Documents/Projects/GuanZero/.work/kaggle-benchmark-20260924/state.json)

本地仅完成打包、语法和上传范围检查，没有进行本地训练。收到明确上传授权后，才能上传该测试包并执行已准备的测速。
