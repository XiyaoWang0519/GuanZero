# 实验产物：2026-09-28/29（主线续训、提速、评估、打法统计、历史消融）

本目录是从本地未入库的 `.work/` 目录挑选出来的原始数据和脚本，供外部审阅者分析。所有文件都是复制件，未做修改；只有超过 5 MB 的文本文件被 gzip 压缩（见第 3 节）。本 README 里的数字都取自本目录的文件或 `docs/reports/` 中的四份报告，出处在括号里注明。

另有两份给审阅者的独立简报，各自可以单独阅读：

- [`BRIEF-performance.md`](BRIEF-performance.md)：训练提速（每秒 / 每美元决策数）。
- [`BRIEF-history.md`](BRIEF-history.md)：为什么 Transformer 没有用上历史事件流。

## 1. 项目与单位

GuanZero 是一个掼蛋（Guandan，四人两队的中国扑克）AI 项目：C++20 规则引擎（`cpp/`，验收标准是 [`docs/RULES.md`](../../docs/RULES.md)）加 Python 训练和评估。当前模型是**历史 Transformer**：宽 128、4 层、8 头，每次决策输入 1,849 维观测和整场比赛的公开事件流，从随机初始化开始用自对弈 PPO 训练，训练对手只来自本谱系的当前或过去版本（[`docs/DESIGN.md`](../../docs/DESIGN.md) v0.6，[`docs/STAGE_C_TODO.md`](../../docs/STAGE_C_TODO.md)）。**B11**（B11 main）是之前最强的 MLP 策略，只用于评估，不参与训练（权重 SHA256 `25e9e0bf549957b9…`，见 `evaluation/out/*.json` 的 `baseline_sha256`）。

单位：
- **净级数/轮**（net levels per round）：复式牌（duplicate），同一副牌两队换座各打一次，每副牌得分 = (第 1 腿 A 方净升级数 + 第 2 腿 A 方净升级数) / 2（`evaluation/out/u2201.json` 的 `score_definition`），再对 256 副取平均。正数表示被测模型占优。
- **256 副开发集复式牌**：固定的开发集（`split: "development"`），不是最终测试集；最终测试集没有打开。
- **整场胜率**：64 个整场比赛种子，每个种子两队换座各打一场（64 对），被测方赢下整场比赛的比例。

区间：单点区间是评估器对 256 副牌的 bootstrap（2,000 次）；成对差值区间是逐副差值的 bootstrap（4,000 次）（`evaluation/results.v2.json` 的 `method`）。

## 2. 时间线（时间为 UTC，取自各 kit 的 `teardown-summary.json` / `lifecycle.jsonl` / `*.time`）

| 时间 | 实验 | 目录 | 报告 |
|---|---|---|---|
| 09-28 21:38 | 采集阶段剖析（提交 `8303769`），机器 20082，更新 870–884 | `kits/collect-profile/` | [快照合批报告](../../docs/reports/history-snapshot-batching-2026-09-28.md) 结果一 |
| 09-29 01:45 | 快照合并推理 A/B（提交 `e7ef4cc`），机器 67872，更新 845–919 | `kits/snapshot-batching/` | 同上，结果二、三 |
| 09-29 03:03 / 03:12 | 显存回收检查：第一次在门禁处中止（`06b5f0b`，`attempt-1/`），第二次完成（`5b7078e`） | `kits/trim-check/` | 同上，结果三 |
| 09-29 03:30–12:01 | 8 小时主线续训（`5b7078e`，合批 + 回收），更新 845–2623；前两次启动失败，第三次（03:54 创建）完成 | `kits/longrun-batched/` | [长跑报告](../../docs/reports/history-longrun-batched-2026-09-29.md) |
| 09-29 | 13 个 checkpoint 对 B11 的 256 副评估（本地 CPU，`b8c0c54` 导出） | `evaluation/` | 长跑报告"强度评估" |
| 09-29 | 云端 CPU 评估 kit，已准备、**从未启动** | `kits/eval-cloud/` | 无 |
| 09-29 | u2623 对 B11 / u0844 的打法统计与回放 | `game-review/` | [打法报告](../../docs/reports/history-play-style-2026-09-29.md) |
| 09-29 | 历史输入消融（FULL / ROUND / NONE / SWAP） | `history-ablation/` | [消融报告](../../docs/reports/history-ablation-2026-09-29.md) |

背景文档：[`docs/DESIGN.md`](../../docs/DESIGN.md)（系统设计与约束）、[`docs/STAGE_C_TODO.md`](../../docs/STAGE_C_TODO.md)（T0–T8 任务队列）、[`docs/TRAINING.md`](../../docs/TRAINING.md)（入口就绪状态与 GPU 流程）、[`docs/RULES.md`](../../docs/RULES.md)（规则）。

## 3. 各子目录

所有提交都是 `main` 的祖先（本分支从 `8e197db` 分出）。

### 3.1 `kits/`：云端 kit 的共同格式

每个 kit 是在一台租用的 Vast.ai RTX 4090 上跑一次的封闭流程。共同文件：
- `README.md`：kit 的计划、保护措施和命令（写于启动前，其中"Unverified"是当时的状态）。
- `lifecycle.py`：本地控制脚本（租机、上传、同步、销毁）；`remote_setup.sh` / `remote_start.sh` / `remote_guard.template.py`：远端脚本；`summarize.py` 等：汇总脚本。
- `kit-manifest.json`：`source.revision`（代码提交）、`source_sha256`、kit 文件哈希；`kit-files.sha256`。
- `lifecycle.jsonl`：本地事件日志（每行一个 JSON：`time`、`event`，如 `launch`、`created`、`teardown`、`absent_confirmed`）。
- `create-intent.json`、`instance-*.closed.json`、`last-instance.json`：租用的实例（机器号、offer 号、报价、时间上限；`last-instance.json` 含 SSH 主机/端口和公网 IP）。
- `offers.json`、`dry-run*.json`：选机结果和演练输出。`teardown-summary.json`：实例存活秒数、按报价估算的费用、账户额度前后、销毁确认。`artifact-verification.json`：下载校验结果。
- `download/results/`：从远端下载的原始结果。`gates.json` 与 `gate-*.log|xml`（训练前 GPU 门禁测试，pytest junit XML）、`gd-tests.log`（C++ 单元测试）、`host-facts.json`（CPU/cgroup/GPU）、`nvidia-smi.txt`、`pip-freeze.txt`、`setup.log`、`orchestrator.jsonl`（训练进程的启动/退出/重启事件）、`gpu-samples.jsonl`（每 5 秒：`epoch`、`util`、`mem_util`、`mem_used_mib`、`power_w`、`sm_mhz`）、`update-epochs.jsonl`（每次更新的墙钟：`epoch`、`update`、`attempt`）、`results-manifest.json`（结果文件哈希）。
- `download/results/segments/<名字>/`：训练器输出，rank 0 在该目录，rank 1–3 在 `rank-r/`。
  - `metrics.jsonl`（长跑中为 `metrics.jsonl.gz`）：每次更新一行，79 个字段。主要字段：`update`、`global_decisions`（谱系累计决策）、`decisions`、`step_decisions`、`global_decisions_per_sec`、`collect_seconds` / `learn_seconds`（本 rank）与 `global_collect_seconds` / `global_learn_seconds`、`entropy`、`approx_kl`、`clip_fraction`、`policy_loss`、`value_loss`、`explained_variance`、`response_loss` / `response_accuracy`（辅助 response 头）、`mean_prefix` / `max_prefix`（采集时历史前缀 token 数）、`learn_mean_prefix` / `learn_max_prefix`、`cuda_*_bytes`（分配器计数）、`cuda_allocation_retries`、`cuda_trim_seconds`、`collection_phase_seconds`（剖析时的分阶段秒数）、`collection_policy_batches` / `collection_policy_call_rows`、`population`、`round_gain`、`world_size`。
  - `population.jsonl`（长跑中为 `.gz`）：对手池事件。`assignment`（`env`、`match`、`version`、`seats`：四个座位的策略身份，0 = 当前学习者，其他数字 = 快照身份）、`snapshot`（`identity`、`update`、`sha256`）、`resume`、`ddp`。
  - `manifest.json`：checkpoint 元数据（配方、布局、`config_changes`、`source_changes`）。`*.log`：训练器标准输出。

### 3.2 `kits/collect-profile/`（提交 `8303769`）

采集阶段剖析：4 个 rank，先 25 次不剖析的更新热身，再剖析 15 次（更新 870–884）。
- 主要数字：`profile-summary.md`（同样的内容在 `profile-summary.json`）。剖析模式下采集 13.81 s/更新、学习 7.17 s/更新（4 rank 均值）；公共缓存/整理 7.01 s 和 actor 与采样 6.09 s 是采集的两大块。剖析模式在每个阶段边界同步 CUDA，秒数只用于分摊，不代表绝对速度（`profile-summary.md` 第 3 行）。
- 机器 20082；费用约 $0.19（`teardown-summary.json`）。

### 3.3 `kits/snapshot-batching/`（提交 `e7ef4cc`）

合并快照推理的同进程 A/B（排程 `35:off,10:on,10:off,10:on,5:off,5:on`，每块丢掉前 2 次更新）。
- 主要数字：`ab-summary.md|json`：关 / 开 20.84 / 15.39 s/更新，3,145 / 4,263 决策/秒（比值 1.355），每步策略调用 12.63 / 2.00，nvidia-smi 最高 21,250 / 23,578 MiB。
- `memory-analysis/`：显存上涨的诊断。`tables.md`（按块、rank 的分配器计数表）、`analysis.json`、`per_update.csv`（每 rank 每次更新一行）、`timeline.svg`、脚本 `analyze_memory.py`。
- `cpu-measure.json` / `cpu_measure.py`：本地 CPU 上的测量；`flagoff-check/`：开关关闭时与父提交 `8303769` 的摘要对比（`branch.json`、`parent-8303769.json`）。
- `download/results/gate-snapshot.log|xml`：snapshot 门禁 49 项。

### 3.4 `kits/trim-check/`（提交 `5b7078e`；`attempt-1/` 为 `06b5f0b`）

分配器回收检查（排程 `10:off,30:on`）。
- 主要数字：`trim-summary.md|json`（顶层与 `download/results/` 中各一份，内容相同）：判定 FLAT；nvidia-smi 每次更新最高值后 10 次比前 10 次高 700 MiB，已分配峰值合计高 892 MiB；回收耗时 0.120 s/更新，占墙钟 0.76%。
- `attempt-1/`：第一次启动，`download/results/FAILED.json` 为 `{"stage":"setup","ok":false}`；`gate-trim.log|xml` 是失败的门禁（原因见快照合批报告"回收门禁第一次失败"：测试设计问题，进程级分配器计数）。

### 3.5 `kits/longrun-batched/`（提交 `5b7078e`）

8 小时主线续训：从 main-w4 的 `latest.pt`（update 844，SHA256 `b1c5503743fe69d7…`，kit 里的 `init.pt`，未包含）续到 update 2623。
- 主要数字：`longrun-summary.md|json`：1,779 次更新，新增 116,588,544 决策，谱系累计 171,900,928；训练 8.00 h；重启 0，fallback 0；每 31 次更新一行（速度、采集/学习秒数、快照数、前缀、熵、KL、显存、回收）。
- 原始训练日志：`download/results/segments/main-overnight/metrics.jsonl.gz`（原名 `metrics.jsonl`，1,779 行）与 `rank-1..3/metrics.jsonl.gz`；`population.jsonl.gz`（原名 `population.jsonl`）；`main-overnight-attempt-1.log`；`MAIN_LINEAGE.txt`。
- 启动记录：`lifecycle.jsonl` 含三次启动（实例 53298552 未开始训练、第二次未创建实例、53300701 完成）；`RECOVER-53298552.stale.md`、`instance-*.closed.json`、`remote-started.json`。
- 费用：实例存活 29,206 秒，按 $0.557/h 估算 $4.52（`teardown-summary.json`）。
- 训练 checkpoint（57 个 `update-*.pt` 和 `latest.pt`，共约 5.9 GB）未包含，哈希见第 4 节。

### 3.6 `kits/eval-cloud/`（源码 `b8c0c54`，**从未启动**）

在一台租用机器的 CPU 上评估 7 个 checkpoint 的 kit，已做 freeze 和 dry run，没有启动，没有结果。保留它是因为 `README.md` 记录了评估器源码身份的核对（`git archive b8c0c54` 与导出目录都哈希为 `14e72581…`，213 个文件）和跨平台可比性的设计。`payload/assets/development.json` 是 256 副开发集复式牌，`payload/assets/freeze.json` 是评估冻结文件（原 freeze，SHA256 `f77ea34d…`），`payload/expected.json` 是 payload 各文件的期望哈希。`offers-search.json` 是选机时的 offer 列表。

### 3.7 `evaluation/`（评估器：`b8c0c54` 的只读导出，源码指纹 `14e72581…`）

13 个 checkpoint 对 B11 的 256 副复式牌评估，本地 CPU。
- 主要数字：`results.v2.md`（表）和 `results.v2.json`。`results.json|md`、`results.v1.*` 是较早版本（点数较少），`summarize*.py` 是生成它们的脚本。
- `results.v2.json` 字段：`method`（评估器、freeze、牌、区间算法）；`b11_only_verification`（只对 B11 的 freeze 与原 freeze 的逐位核对，`verdict: "PASS"`）；`checkpoints[]`（每个点：`name`、`update`、`file`、`run`、`global_decisions`、`entropy_mean_31`（该点之前 31 次更新的训练熵均值）、`candidate_sha256`、`evaluation_source_sha256`、`freeze_sha256`、`vs_b11`、`vs_b11_ci95`、`match_win_rate`、`match_win_ci95`、成对差值、`wall_seconds`、`concurrent`）；`late_segment`（u2108 起 8 个点的合并值、OLS 斜率、后半减前半、各点减终点）。
- `out/uNNNN.json`（双对手 freeze：B11 和一个 MLP 续训终点 `longrun-segment2-raw-endpoint`）与 `out-b11/uNNNN.json`（只含 B11 的 freeze `b11only/freeze.b11-only.json`）：评估器原始输出。顶层 `candidate_sha256`、`freeze_sha256`、`evaluation_source_sha256`、`split`、`selection`、`claim`；`reports.<对手>.duplicates`：`deals`、`rounds`、`score_definition`、`mean_net_levels_per_round`、`bootstrap_95_ci`、`banker_rate`（头游率）、`double_win_rate`（双上率）、`opponent_double_win_rate`、`pair_scores`（256 个逐副得分）、`results`（每副两腿的名次 `order`、`winning_team`、`gain`、`seat_return`）；`reports.<对手>.full_matches`：`pairs`（64 对整场的回合数、头游、双上等）、`win_rate`、`bootstrap_95_ci`。`*.log` / `*.time` 是运行日志和起止时间。
- `out-b11/u2201.json` 是只对 B11 的核对重跑；`b11only/verify.json|log` 与 `VERDICT`（PASS）是核对结果。`out-verify/` 是一次没有跑完的核对（只有开始时间）。
- 运行脚本：`run_one*.sh`、`lane_*.sh`、`queue*.sh`、`run_point.py`、`watch_v2.sh`；它们引用本机绝对路径，不能直接在别处运行。

### 3.8 `game-review/`（`b8c0c54` 导出 + kit 内的记录器副本 `review_lib.py`）

u2623 对 B11（1,000 副 × 两腿）和对 u0844（500 副 × 两腿）的贪心打法统计，另录 5 场完整比赛供回放。
- 主要数字：`analysis_b11.txt`、`analysis_u0844.txt`（`analyze.py` 的输出：每项比率、Wilson 区间、按牌聚类的 bootstrap 差值区间）。
- `game-review.html`：自包含的回放页面，数据以 `const DATA = {...}` 内嵌（与 `games.json` 逐项相同，已核对：5 场、49 回合）。唯一的外部引用是 Google Fonts 字体，离线时回落到系统字体。用浏览器直接打开即可。
- `games.json`：`generated`、`rules`、`note`、`games[]`（每场：`title`、`seed`、`team_names`、`seat_labels`、`winner_team`、`match_id`、`rounds[]`；每回合：`level_rank`、`levels_before/after`、`hands`（四家起手牌，每张 `{r, s, w}`：点数、花色、是否逢人配）、`tribute`、`leader`、`steps[]`（`seat`、`pass`、`cards`、`type`、`type_zh`、`new_trick`、`left`）、`finish_order`、`winning_team`、`gain`）。`viewA.games.json`、`viewB.games.json` 是合并前的两部分。
- `statsB11.decisions.jsonl.gz`（206,912 行）、`statsU0844.decisions.jsonl.gz`、`viewA|viewB.decisions.jsonl`：每个决策一行。字段 `match`（如 `D0L0` = 第 0 副牌第 0 腿，`M1` = 第 1 场比赛）、`round`、`step`、`seat`、`label`（该座位的模型）、`level`（引擎级牌编号）、`forced`、`lead`、`hand`（出牌前手牌，引擎牌编号 0–107）、`left`（四家剩余张数）、`top_seat`、`top_rel`（`partner`/`opponent`）、`n_legal`、`has_nonpass`、`has_nonbomb`、`pass`、`type`、`cards`、`bomb_size`、`probs`（仅历史策略：概率最高的 4 个候选 `[描述, 概率]`；B11 为 `null`）。
- `statsB11.rounds.json`、`statsU0844.rounds.json`：每个单回合一条（`deal`、`leg`、`keys`、`level`、`finish_order`、`winning_team`、`gain`、`seat_return`、`steps`）。
- `verify.json|log`：记录器与评估器逐个决策一致的核对；`blind_control.json`：关掉历史输入后 10 副牌中 3 副走法不同（负对照）。
- 脚本：`record.py`（阵容和种子）、`review_lib.py`、`analyze.py`、`moments.py`、`build_html.py`、`viewer.template.html`。

### 3.9 `history-ablation/`（`b8c0c54` 导出 + kit 内的子类 `ablate_lib.py`）

u2623 的历史输入消融，对手只有 B11，全部贪心。
- 主要数字：`out/strength_summary.txt|json`（强度）、`out/probe_analysis.txt`（改选率与 TV，按局面分组）、`out/margin_analysis.txt`（按 FULL 第一、第二候选概率差分组）、`out/time_decision.txt`（前缀长度与每决策耗时）、`out/verify.json|log`（包装器核对）、`out/examples.txt`（TV 最大的例子）。
- `out/probe_*.records.jsonl.gz`：探针逐决策记录。字段 `tag`（`D…` 单局复式、`P…` 整场比赛对）、`round`、`seat`、`level`、`lead`、`n_legal`、`prefix`（FULL 事件流长度）、`round_events`（本局事件数）、`own_left`、`partner_left`、`opp_left`、`pass`、`type`、`has_nonpass`、`arg`（各条件的贪心选择下标：`full`/`none`/`round`/`swap`）、`pmax_full`、`tv`（FULL 与 `none`/`round`/`swap` 的总变差距离，及 `swap_vs_round`）、`donor`（SWAP 供体编号，-1 为无）、`cands`（FULL 前 3 名与各条件贪心选择的并集，每项 `i`、`a`（描述）和各条件概率）、`hand`、`history_len`。`probe_*.meta.json`：每场/每副的元数据（`tag`、`seed`、`team`、`won`、`rounds`、`round_tokens`、`events`）。
- `out/dup_*.json`、`out/match_*.json`、`out/match_*.pairs.jsonl`：强度对局的逐副 / 逐对记录；`out/donor_full_b11_0.*`：SWAP 供体（12 场 FULL 对 B11）。
- `jobs*.txt`：计划的任务列表（其中 `jobs_match.txt` 的整场 FULL / ROUND / NONE 对 B11 没有运行，见消融报告"计划中但没有做的测量"）。`env.sh` 记录运行时的 `PYTHONPATH`（本机路径）。

## 4. 未包含的内容

| 内容 | 原因 |
|---|---|
| 模型权重 `*.pt`（kit 的 `init.pt`、训练 checkpoint、评估用 `ckpt/`、B11 等） | 体积（单个 87–116 MB，长跑共约 5.9 GB）。哈希见下表，可用来把结果对应到权重。 |
| 源码包 `source.tar.gz` 和导出源码树 `src-b8c0c54/`、`local-src/`、`memory-analysis/src/` | 与仓库重复。`src-b8c0c54/` 和 `local-src/` 对应提交 `b8c0c54`（源码指纹 `14e72581…`，见 `kits/eval-cloud/README.md`）；`memory-analysis/src/train/` 与 `git archive e7ef4cc train` 逐文件相同（整理时核对）；各 kit 的 `source.tar.gz` 对应其 `kit-manifest.json` 的 `source.revision`。 |
| `evaluation/out-tainted-mixed-source/`（u2480、u2623 的第一次评估结果） | 已废弃：它们在后续提交（`49797b0`、`f757ba2`、`ce24c5d`）进入工作区后从工作区运行，评估器源码指纹变成 `0e935769…`，与其他点不可比。上表用的是从 `b8c0c54` 导出重跑的结果（长跑报告"评估方法的两点说明"第 2 条）。为避免被当成结果，未复制。 |
| `smoke*/`、`smoke-init/`、`eval-cloud/smoke/`（含伪造的 `fake-same`/`fake-diff` 输入） | 本地 CPU 冒烟测试，用微型 checkpoint 和假数据，不是结果。 |
| `known_hosts`、`*.env`（`deadlines.env`）、`generated/`、`dry-run*/`、`dry-run-generated/` | 主机密钥记录和渲染好的远端脚本。凭证（Vast API key、SSH 私钥）从不在 kit 目录中：`lifecycle.py` 从仓库外的路径读取，`dry-run*.json` 只记录 `"present, 0600"`。 |
| `__pycache__/`、`*.pyc`、`.mon_err.*` | 编译缓存与监视脚本的临时输出。 |

权重哈希（SHA256，整理时由本地文件计算；与 `evaluation/results.v2.json` 的 `candidate_sha256` 一致）：

| 权重 | 更新 | SHA256 |
|---|---:|---|
| main-w4 `latest.pt`（长跑 `init.pt`，评估中的 u0844） | 844 | `b1c5503743fe69d7d997612ffeab139cb9d5ef2ecb55e02ee2c76f259e112bd7` |
| u1333 | 1333 | `4058b8ed8f2a5553cf0e9b212f1ed6c513cd386aeec59be18c16cf21ee85a7ef` |
| u1612 | 1612 | `e380212005dd0f2d4b8c763d8fa310184aaf7d00dd5e80011f252372dbd926eb` |
| u1829 | 1829 | `448a79d092bde85d5ead0d5126375505b3f2fcd9dbb6f835f389fce3be0be7bc` |
| u2015 | 2015 | `89502270020e6994fd641d82e7843a22eb9a587f2ee4f694841d1f923d559baa` |
| u2108 | 2108 | `2c91e70b41c68d0ecaa29ae841c3facf01d9ec2340336a194d3be305a734c89e` |
| u2201 | 2201 | `8dd52066c0cfa4692ff9038c7d9bd985845070e4733a5e739abe5f79f179627d` |
| u2263 | 2263 | `a99024158a24e960dfe843ffbebafadfa6b1eea5a8661cc48a5371e53acd35d1` |
| u2356 | 2356 | `a48e8725f0d6059b4d3a46a9ddbf64524d196a4307a79643da3525e33811fa54` |
| u2418 | 2418 | `5bdc3b66dfc7e07b2d30f9db3ae36b1fd10968cbe07a8f10eff142264feeeb74` |
| u2480 | 2480 | `47260eb1884db3d3b3160343db77ea1e81a8f5caefe9a9e756e538813bd5486f` |
| u2542 | 2542 | `642fe4e4c6fa87fcd7e5d1969dc13b3a5259b7c1524f25c8d3994a6a2d37e892` |
| **主线终点 `latest.pt` = u2623** | 2623 | `6fd3b94d9ce8f4ae42e5164c3b8645990c09cb4ab1b1e564897ef3732b7972a9` |
| B11 main（对手） | — | `25e9e0bf549957b9d8b8aeb426b7225259cca714ec23d77de8eeda45bef063df` |
| MLP 续训终点 `longrun-segment2-raw-endpoint`（双对手 freeze 的第二个对手） | — | `eda2efc844d3831615b8fa99b3102591a95f9c1a6dfc03f3b0ede0479b8f6aa6` |

u1333–u2542 的评估文件与长跑目录中同名的 `update-NNNNNN.pt` 哈希相同；u2623 与长跑的 `latest.pt` 相同。

## 5. 审阅时需要记住的限制（摘自报告）

1. **单条谱系、单个种子。** 长跑不是预先声明的对照实验（长跑报告开头）。13 个 checkpoint 属于同一条谱系、用同一批牌，后段合并值是平滑估计，不是独立重复。
2. **开发集、单一对手。** 强度只在 256 副开发集复式牌上对 B11 测过，终测牌没有打开；不构成强度晋升（长跑报告）。
3. **评估必须从 `b8c0c54` 运行才可比。** 之后的提交改了 `train/history_model.py` 和 `eval/history_policy.py`，源码指纹会变（长跑报告"评估方法的两点说明"）。所有保留结果的 `evaluation_source_sha256` 都是 `14e72581…`。
4. **后段趋势分辨不出。** u2108 起 8 个点：对 B11 合并 +0.10 [+0.02, +0.18]，OLS 斜率每 100 次更新 −0.01 [−0.04, +0.03]，后 4 点减前 4 点 −0.04 [−0.15, +0.07]；从 u844 到后段的平均增速约每 100 次更新 +0.018，落在斜率区间内，所以既分辨不出"走平"，也分辨不出"仍按之前速度上升"（`evaluation/results.v2.md`、长跑报告）。
5. **只有贪心出牌。** 评估、打法统计和消融都取概率最大的单个动作；训练时是按分布采样（打法报告"局限"）。同一手牌型的多个花色组合会分摊概率，按前 4 个候选约 0.48% 的决策受影响，严格下界 0.16%（打法报告"补充发现二"）。
6. **消融后的输入不在训练分布内。** 模型训练时总看到完整事件流，消融测的是依赖程度，不是"不看历史训练出来的模型会差多少"；SWAP 供体也是对 B11 的比赛（消融报告"局限"）。
7. **速度 A/B 只有 2 对块**，在一台机器、一个进程内，比值没有区间；不同机器的绝对速度不能直接比（快照合批报告"局限"）。
8. **费用数字**是"存活时长 × 报价"，不含流量（`teardown-summary.json` 的 `cost_scope`）。

## 6. 项目方希望审阅者思考的问题

1. 消融显示 u2623 对历史事件流没有可测量的依赖：去掉事件流时贪心选择只在 1.1–1.5% 的决策上改变，把之前各局换成别场内容只改变 0.08%（消融报告）。原因是什么？在项目约束下（只用本谱系自对弈，MLP 只用于评估）怎样才能让模型用上历史？详见 [`BRIEF-history.md`](BRIEF-history.md)。
2. 训练熵在约 1840 次更新处从 845–1836 段的均值 0.622 降到 1837–2623 段的 0.550，原因没有查明（长跑报告）。这个变化是否重要，应当和哪些量对照？
3. 合批之后每次更新仍是采集约 9–10 s、学习约 7–8 s，采集的大头已转到公共缓存编码（快照合批报告、`kits/longrun-batched/longrun-summary.md`）。剩下的时间怎样用得更好？详见 [`BRIEF-performance.md`](BRIEF-performance.md)。
4. 评估取概率最大的单个动作，而同一手牌型的多个花色组合会分摊概率（打法报告"补充发现二"）。评估是否应当先把花色变体的概率合并，再取最大？如果要改，怎样定口径才能与已有结果衔接？
