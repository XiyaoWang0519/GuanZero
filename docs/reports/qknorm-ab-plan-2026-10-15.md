# 私有注意力归一化（query_norm / qk_norm）A/B 计划（2026-10-15）

状态：**计划，未执行**（2026-10-15，多伦多时间）。分支 `qknorm-ab-2026-10-15` 上的代码、
测试和 Velocity kit 已就绪；**没有提交任何训练作业，主线作业与 `train-src` 未改，默认行为
不变**（两个开关都关闭时与此前代码逐位一致）。本文是提案，不是启动授权；启动与否、何时
暂停主线，由用户决定。

> **2026-10-16 更新（多伦多时间）：已执行。** 用户 10 月 15 日批准后，主线在 u88100 暂停，三臂从 u86800 在 Velocity 上训练：qk-control 到 u94843，qk-query 到 u95143（6.235B），qk-both 到 u97554（6.393B，因决策目标只在 slice 末检查而多跑约 200M）。500M 评测点三臂都取 `update-094488.pt`。qk-control 的 500M 内部评测已完成（对起点 +0.043 [−0.008, +0.098]，对 V1T-linux +0.166）；qk-query 与 qk-both 的评测作业已提交（10 月 16 日 18:05），DanLM 评测与注意力探针尚未运行。结果出来后另写报告；下文保持原计划原文。

## 1. 动机

[私有注意力诊断](private-attention-diagnosis-2026-10-15.md)在 2026-10-14 的 rank 1 fixture
上测到：主线从 m5 起所有 head 都是分差极大的硬 one-hot；q 的方向几乎不随决策变化（q_proj 的
输入是未归一化的私有 MLP 输出 + seat，范数 0.9k–5.3k，均值分量占绝大部分，q_proj 最大奇异值
235）；u86800 有 7/8 个 head 固定落在常数 BOS；m7 起 q/k 的精确梯度为 0，u86800 部分 head 的
Adam 二阶矩已衰减到次正规不动点。[触发调查](training-sdpa-trigger-2026-10-15.md)把 10 月 12 日
的梯度爆炸归于 masked efficient SDPA 反向在 1e8 量级 score 下的概率重建误差。两个归一化变体
针对的都是“logit 量级失控”这一个机制；它们是否能带来棋力，只有 A/B 能回答。

## 2. 实现（分支 `qknorm-ab-2026-10-15`，commit 见 `.work/qknorm-ab-2026-10-15/kit-manifest.json`）

`HistoryPolicyConfig` 新增两个独立开关，默认 `False`：

| 开关 | 做什么 | 新参数 | 初始化 |
|---|---|---|---|
| `query_norm` | 私有 query（私有 MLP 输出 + seat）在进入 `q_proj` **之前**过一层 `LayerNorm(width)`；残差路径仍用原始 query | `query_norm.weight/bias` | 单位（weight 1，bias 0），与 `attention_norm` 同风格、同 eps |
| `qk_norm` | q 与 k 逐 head 做 RMSNorm（eps 1e-6，无逐元素仿射），q 再乘可学的逐 head 温度；logit = 温度 × √depth × cos(q, k) | `qk_temperature.log_temperature`（log 空间，[heads]） | 温度 = ln(512)/√depth（width 128、8 head 时 1.559，即最大 logit = ln 512 ≈ 6.24） |

温度初值的依据：归一化后一个与 query 余弦为 1 的 key，对 511 个正交 key，恰好分到一半的
注意力质量（e^L/(e^L + 511) = 0.5 ⇔ L = ln 512）。512 是 fixture 可见前缀长度的量级
（中位数 321，最长 852）。这个尺度“能选中一个位置，但远离 one-hot”，softmax 对 q/k 的梯度
有限；温度可学，所以模型仍可以变得更尖锐，只是要靠梯度走到那里而不是起步就饱和。

覆盖的路径：`_attend`（单 query masked SDPA，rollout 默认路径与 window 对照）、
`_attend_independent`（批量 rollout，逐行投影或 wide projection）、`_attend_by_match`
（learner 的按比赛 padded 调用）三条路径共用 `_query_heads` / `_key_heads`；
`history_snapshot_batch` 的合并快照头（`batch_snapshot_policies=true`，主线在用）：`query_norm`
多叠一层堆叠 LayerNorm，`qk_norm` 因为 key 的归一化对 e 非线性、不能再把 W_k 折进 query，所以
逐行物化每个 head 的 key（[rows, heads, T, depth]）再归一化打分，value 的折叠不变；CUDA graph
的 storage signature 把新方法名纳入；population 载入老快照时缺的归一化参数取学习者的新初值，
并刷新快照摘要（旧摘要保留在 `previous_sha256`）；Botzone 的 numpy actor 不实现归一化，
`eval.botzone.export` 对开了开关的 actor 直接拒绝。

checkpoint 记录：`config_record` 只在开关打开时写入 `query_norm` / `qk_norm`（与
`style_input`、`aux_heads` 同样的规则），旧 checkpoint 的 `model_config` 逐字节不变；
续训时 `--resume-set query_norm=true`（`RESUME_OVERRIDES` 已加入两项）由 `config_changes`
记录 `{"query_norm": [false, true]}`。`extend_actor` 现在允许加 aux head、`query_norm`、
`qk_norm`（只加不减）。Adam 矩的续接改为**按参数名**匹配（`load_extended_optimizers`），
不再要求新参数注册在末尾；新参数没有 Adam 状态，从零开始。

## 3. 从里程碑续训意味着什么

- **注意力选择被重置。** 插入的 LayerNorm 是单位映射、温度是常数，但它们改变了进入 softmax 的
  logit 量级，所以每个 head 落在哪个位置都会变（第 5 节的探针给出了 u86800 插入时刻的实际
  状态）。私有 MLP、k/v 投影、out_proj、决策头和所有辅助头的权重原样继续。
- **常驻快照也被重置。** population 里的快照是学习者架构的副本；载入 u86800 的 14 个常驻快照
  时它们同样拿到新初值的归一化（所有 identity 必须同一架构，合并快照头才能堆叠）。在
  `population_recent=4`、每 2 次更新一个快照的设置下，这些快照约 30 次更新后全部轮换掉。
- **q/k 的 Adam 状态。** u86800 上多数 head 的 q/k 二阶矩在次正规不动点（√v ≈ 8e-22）；归一化
  后梯度重新非零，Adam 的首步大小约为 lr × m̂/√v̂ ≈ 3 × lr（m 与 v 同时从 0 起步），随后自行
  校准；控制臂不受影响。
- **预期先降后升。** 读历史的 head（u86800 只有 head 7）在插入时刻变成近均匀平均，策略会先
  损失一些；这是为什么 100M 处只做安全读数，不做结论。

## 4. 已验证（Mac CPU，2026-10-15）

- 新测试 `tests/test_history_qk_norm.py`（27 项，CUDA 参数化在无 GPU 时跳过）：开关关闭时
  `_attend` 与提交前的代码逐位一致，三条路径与显式参考实现一致；打开时前向有限、所有梯度有限，
  归一化参数与 q_proj 都收到梯度，`qk_norm` 的 logit 受温度 × √depth 约束；`extend_actor` 只加
  不减、其余权重逐位保留、策略确实改变；续训插入记录 `config_changes`、Adam 矩按名字续接、
  快照取新初值并刷新摘要、再次续训与保存/重载正确；合并快照头在三种开关组合下与逐 identity
  的 `candidate_log_probs` 最大差 < 2e-5；Botzone 导出拒绝。
- 现有套件：`pytest tests` 1,404 通过 / 326 跳过（CUDA、Triton 等），`gd_tests` 84 项通过。
  这覆盖了开关关闭时的全部既有行为。
- **fixture 门槛**（`.work/qknorm-ab-2026-10-15/fixture_gate.py`，fixture 自带 actor 权重，
  outcome loss，CPU math SDPA，torch 2.14.0；[10-14 报告](training-fixture-backward-2026-10-14.md)的门槛：
  梯度有限、最大参数梯度范数 < 100）：

| 臂 | outcome loss | 最大参数梯度范数 | q_proj.weight 梯度 | kv_proj.weight 梯度 | 门槛 |
|---|---:|---:|---:|---:|---|
| control（开关关） | 3.1693379879（与 10-14 CPU 重放**逐位相同**） | 12.924（outcome head） | 1.9e-2 | 0.476 | 通过 |
| query_norm | 3.1693890 | 12.926 | 1.6e-4 | 0.474 | 通过 |
| query_norm+qk_norm | 3.1710577 | 12.915 | 6.2e-6 | 0.481 | 通过 |

q_proj 梯度随归一化缩小是归一化的固有效应（梯度被 1/‖q‖ 缩放）；Adam 对逐参数尺度不敏感。

## 5. 插入时刻的机制探针（u86800，同一 fixture，328 个决策行，CPU）

`.work/qknorm-ab-2026-10-15/attention_probe.py`，按 head 统计两次 masked 调用的有效行：
注意力行熵（nats）、最大概率的 1% 分位、最小 top1–top2 分差、落在 BOS 的比例、q 两两余弦中位数、
outcome loss 下 q/k 的 math 梯度范数。

| 配置 | 熵中位数（8 head） | BOS head 数 | 最小分差范围（nats） | q 余弦中位 | q/k 梯度非零的 head |
|---|---|---:|---|---|---|
| u86800 原样 | 全 0（one-hot） | 7/8 | 2.4e3 – 1.3e7 | 1.000 | 0/8 |
| + query_norm | 7 个 head 为 0；head 7 为 0.35 | 7/8 | 170 – 2.3e4（head 7：0.01） | 0.994–1.000 | 1/8（head 7，1.5e-2） |
| + query_norm + qk_norm | 5.4–5.8（≈ 对可见前缀均匀，ln 328 = 5.8） | 0/8（head 2、6 部分） | 0.00–0.02 | 0.994–1.000 | 8/8（2e-7 – 1.2e-2） |

m7 原样与 u86800 相同量级（分差 4.6e3 – 2.3e7，6/8 BOS，梯度全 0）。

**含义与预测（测前写下）：**

1. `query_norm` 单独**不解除饱和**。LayerNorm 把输入压到单位尺度，但 q_proj 的权重尺度
   （σ₁ = 235）本身就把 logit 推到几千 nats，除 head 7 外仍是精确 one-hot、仍落在 BOS、q/k 梯度
   仍为 0。预测：`qk-query` 臂与控制臂在棋力上不可区分，机制指标也不会移动；它保留在设计里是为了
   把“LayerNorm 本身”与“温度约束”分开，并作为代价最小的变体。如果想省一臂，可把它换成
   `qk-only`（只开 `qk_norm`，kit 已定义）。
2. `query_norm + qk_norm` 在插入时刻把注意力变成**近均匀**（q 的方向仍由 q_proj 的主奇异方向
   主导，与所有 key 的余弦都接近 0），所有 head 的 q/k 梯度重新非零。预测：插入后内部尺子先下降
   （head 7 的历史读取变成平均），随后 q_proj 必须旋转出与 key 相关的方向才能重新选位置；
   500M 决策是否够、能否超过控制臂，未知。这是本 A/B 要回答的问题。
3. 温度可学，所以 `qk_norm` 可能重新走向尖锐；机制探针在每个评测点上重跑，看熵与 BOS 比例的
   走向，而不只看终点。

## 6. A/B 设计（kit：`.work/qknorm-ab-2026-10-15/`，README 有逐步命令）

- **起点**：u86800（`.work/velocity-milestones/u86800.pt`，SHA256 `639ccda1…c720`，5.689B 决策，
  DanLM −0.312 [−0.350, −0.277]），修复后的主线最新评测点，三臂同一起点。备选 m7
  （SHA256 `fa10bd1a…fefb`）：爆炸前的点，但 q/k 的 Adam 状态里带着受污染的动量。
- **臂**：`qk-control`（无改动）、`qk-query`（`query_norm=true`）、`qk-both`
  （`query_norm=true qk_norm=true`）；可选 `qk-only`。其余一切（`fast-lg2`、8 ranks、lr 3e-4、
  entropy 0.03、2 epochs、辅助头、快照节奏、paged cache、page span、合并快照推理）由 checkpoint
  与主线 `train.sbatch` 的逻辑决定，kit 只加 `--set` 开关。
- **长度**：每臂 500M 决策（= 一个主线里程碑，约 7,630 次更新）；评测点 100M / 250M / 500M。
- **预算**：按作业 7464 的 2 卡 15.6k 决策/s，每臂约 9 h 墙钟 = 18 GPU·h，三臂约 54 GPU·h，
  约 1.1 天整机。每用户上限 2,880 运行 GPU·分钟（4 卡 × 12 h）：**主线在跑时臂跑不起来**；
  建议暂停主线一天（`runs/main/STOP`），先并行跑 `qk-control` 与 `qk-both`（各 2 卡），再跑
  `qk-query`，然后恢复主线。这是用户决定。
- **顺序**：`make_kit.py` → 上传 kit 与起点 checkpoint → `prepare_source.py`（`train-src` 复制为
  `train-src-qknorm`，换入分支的 7 个文件，**重新套用 10 月 15 日的 math-SDPA 热补丁**，重建
  source-identity）→ 服务器 CPU 跑新测试 → `gate_qknorm.sbatch`（1 卡 ≤ 1 h：分支测试的 CUDA
  用例 + 每臂 12 次更新的单 rank 冒烟）→ 三臂训练 → `eval_arms.sbatch` / `eval_mac.sh`。
- **尺子**（2026-10-03 起的尺子，`scripts/eval_night.sh`）：内部 `lineage_eval` 2,000 局
  对起点、对同点控制臂（CI ≈ ±0.05）；B11 256 局只作 sanity；服务器 V1T 4,000 局只与服务器
  数字比；外部 DanLM 4,000 局、seed 20260929、duplicate（CI ≈ ±0.037）在 Mac 上跑终点。
  机制尺子：`attention_probe.py` 在每个评测点。
- **判据**：以 500M 的 DanLM 为主。归一化臂只有在同时满足 (a) 对控制臂的内部 2,000 局 CI 不含 0，
  (b) DanLM 领先量大于控制臂与主线同点之间的差（run-to-run 噪声的估计），(c) `qk-both` 的机制
  指标按预测方向移动（熵 > 0、BOS 比例 < 1、多数 head 的 q/k 梯度 > 0）时才采纳；棋力涨而机制
  不动，不归因于归一化。100M 处任一臂低于控制臂 0.3 以上即停（期待的是下沉，不是崩塌）。
  按“没有功效分析不说平台”的规矩，区间重叠就写“不能区分”。

## 7. 风险与未验证

- **CUDA 未测。** Mac 没有 CUDA；新代码的 CUDA 路径（含合并快照头中 `qk_norm` 的物化 key）
  只有 `gate_qknorm.sbatch` 跑过才算数。合并快照头物化 key 的额外显存约 rows × T × width
  FP32（百行、850 token 时约 40 MB），速度代价未测。
- **q 方向常数。** 归一化不改变 q 的方向由 q_proj 主奇异方向主导这一事实；是否能在 500M 内学出
  与 key 相关的 query，是实验问题。如果 `qk-both` 终点仍近均匀，结论是“本设置下归一化不够”，
  下一步才考虑 q_proj 的重初始化或谱归一化，不在本计划内。
- **温度回涨。** 温度可学，可能再饱和；探针跟踪，不设上限。
- **兼容性。** 带归一化的 checkpoint 只能在分支代码下加载（`HistoryPolicyConfig(**model_config)`
  在旧代码下会因未知字段报错）；主线代码加载旧 checkpoint 不受影响。Botzone 导出对归一化 actor
  拒绝；若归一化被采纳，需要再实现 numpy actor。
- **热补丁重叠。** 服务器 `train-src` 的 math-SDPA 补丁在 `_attend_by_match` 里，与分支改动相邻；
  `prepare_source.py` 以精确字符串重新套用并断言只有列出的 7 个文件变化。

## 8. 产物

- 分支 `qknorm-ab-2026-10-15`（基于 d05d75a）：`train/history_model.py`、`train/history_ppo.py`、
  `train/history_population.py`、`train/history_snapshot_batch.py`、`train/history_cuda_graphs.py`、
  `eval/botzone/export.py`、`tests/test_history_qk_norm.py`、本文。
- `.work/qknorm-ab-2026-10-15/`（本地，未提交）：`README.md`、`kit-manifest.json`、`src/`、
  `make_kit.py`、`prepare_source.py`、`gate_qknorm.sbatch`、`train_qknorm.sbatch`、`submit_arm.sh`、
  `eval_arms.sbatch`、`eval_mac.sh`、`fixture_gate.py` + `fixture-gate-cpu.json`、
  `attention_probe.py` + `probe/*.json`（u86800 原样 / +query_norm / +两者，m7 原样）。
