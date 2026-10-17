# 差异旧对手池：两端点完整比赛 FULL／ROUND 消融（2026-10-16）

状态：**2,048 场全部完成；两端点的公开路径均受历史范围影响。control 有 +3.91 pp 的弱正向胜率信号，但未通过预定双主比较门槛；norm16 未见跨局胜率收益。** 用户只授权冻结
评测，不训练、不改生产实现、主线 checkpoint 或既有作业。

## 已固定的设计

承接[差异旧模型的公开历史预测检查](public-history-diverse-2026-10-16.md)。那轮
确认前三局公开行为有小幅后续预测信息，但没有检验当前 actor 是否能将其转为
比赛收益。本轮只更换原[近邻池 FULL／ROUND 消融](qknorm-match-ablation-2026-10-16.md)
的对手池，保留两端点、历史范围、采样、配对与主统计方法。

- 两端点仍为隔离 120-update continuation 的 u94608：原始 qk-both control、
  一次范数 16 / Adam `m/a、v/a²` 迁移后的 norm16。两者均冻结，不再缩放、
  优化或新增快照。checkpoint SHA256 分别是
  `a3d5e42ea6f6dc9a4b514f1bce948a061ee4be0b08c376dc8ce659ccf46948c7`、
  `64d3fbd56bb7ea6968a4a54b3b7fd06f9d505c844920b2aa7c8636e4bcb22bdb`。
- 固定上轮独立选择的 **u2623、u20264、u35588、u86800**，不按本轮胜率重选。
  四者每场固定身份，两名对手使用同一模型且始终 FULL；保留其各自原配置/权重。
  `47241..47244` 只是兼容的评测类别标签，非原生近邻 snapshot 身份；真实
  checkpoint 路径、update、checkpoint SHA、acting SHA 由上轮 `selected.json`
  及本轮每 wave 的 metadata 固定。
- 两名己方一起使用 FULL（整场公开序列）或 ROUND（当前局公开序列，含交换）；
  ROUND 的 encoder/reader 从当前局重新开始，不携带跨局 KV cache，但保留真实
  局号。游戏状态、等级、上一局公开角色和正常私人观测不重置。对手不跟着截断。
- **256 个 match seed × 两次队伍互换 = 每条件 512 场**，四条件共 2,048 场。
  每个对手每条件 64 seed × 2 场，均衡。16 wave，每 wave 32 slot，主 seed
  16102671；每 seed/seat 有独立采样流，温度 1，固定 tribute heuristic。
- 沿用前一轮公开预测检查的种子，四条件逐场配对；FULL 必须精确复跑其原有
  1,024 场记录。**FULL 回放不是新增独立棋力样本**，本轮新增的是同病例 ROUND
  反事实及完整配对比较。没有按先前胜率选择样本，也未因预测检查改过 actor。
- 所有预定 slot 的首场比赛必须完整结束，不挑先完成者；每 wave/条件最多
  20,000 vector step，每 worker 1,800 秒，四个单线程本地 CPU worker，固定
  16 wave、无自动扩样或重提、无新训练/GPU 作业。

## 检查“影响出牌”和“帮助赢”

硬检查：同一端点 FULL／ROUND 的首局行动摘要必须逐场完全相同；FULL 与上一轮
同对手的记录逐字段相同。另对整场**实际公开动作**（含自动 pass）做 hash，
只用局号、座位、阶段、公开余牌数、已公开动作编码，不读 forced 标志。整场 hash
不同表示固定 seed 下两种历史范围产生了不同公共路径，不等于证明学会了有用的
对手建模；hash 相同也不能证明所有策略概率完全一致。

主指标：各端点完整比赛胜率 **FULL−ROUND**；以完整 seed 为 cluster，保留
双方队伍互换及四条件联合配对，在四个对手层内 bootstrap 20,000 次。报告
95% 区间，另外报告两个主比较的 97.5% 区间，用于 Bonferroni family-wise
95% 判断。区间包含零表示未确立正收益，不代表等效或完全不用历史。

范数方案相对 control 的历史收益 interaction、FULL 条件下两端点差异、各对手
分层点估计和 return/比赛长度是次指标，不以其中某一个替换主胜率结论。

FULL／ROUND 同时改变跨局内容、encoder 上下文及位置；即使净收益为正，也不能
仅解释成“认出了对手习惯”。四个旧模型有棋力差异；只对这个冻结池、两个固定
短程端点及当前采样规则成立，不代表主线或更大训练预算的表现。

## 实现、隐私边界与证据

入口：[bench/qknorm_diverse_match.py](../../bench/qknorm_diverse_match.py)。
沿用原 `EvalCollector` 的采样、ROUND store/key/cache 与 `run_wave`，以已有
不读 `hidden_counts` 的 metadata 路径构造 actor 输入，再仅将己方 prefix
切到当前 round；对手仍是 full prefix。玩家正常看自己的牌，分析/轨迹记录器
不读私牌、候选集或 forced，不导出私人观测/候选；无 trainer 或 optimizer。

新增三项检查：FULL/ROUND 下仅己方 prefix 变化（两种参数化情形），以及公开
hash 不读取隐私属性且能检测公开出牌变化；与已有采样/统计/隐私检查共 **14 passed**。
旧的统计脚本和训练/模型/cache/native 源码保持原 SHA。

主统计直接调用不改动的 `bench/qknorm_match_summary.py`，外层先核对新池与选择
manifest 完全相符。metadata 的 `parent` 兼容字段仅指两个 learner 的共同 parent，
不代表四个旧对手来自该 parent 的 resident pool；已另写明 `parent_note`。

本地证据：`.work/qknorm-diverse-match-2026-10-16/`，含 `protocol.json`、
`results/wave-00..15/`、日志、manifest 和汇总；该目录被忽略，非可移植依赖。

## 结果

### 完整性与行为影响

16 个 worker 均 exit code 0，耗时 **289.73 秒**。全部预定 **2,048 场、20,593 局、
1,442,292 个显式决策**完成（含固定交换动作，不含引擎内部自动 pass），最长
比赛 24 局，无超时、非法行动或非有限概率异常，无剔除/替换比赛。

两个端点 FULL 各 512 场均与前一轮不同旧对手的完整记录逐字段相同，合计
**1,024 场精确 FULL 回放**；再从上轮 944,584 条原始公开事件独立重算整场 hash，
与本轮 FULL 的 1,024 场 hash 逐场全部相同。FULL／ROUND 的 **1,024 对首局行动摘要全部相同**。
同端点同 seed/队伍的整场公开动作 hash：control **512/512 对不同**，norm16
**512/512 对不同**。因此改变跨局上下文确实改变了实际公共路径；不能再以“两个
视图出牌完全一样”解释胜率结果。该计数不单独证明模型在识别对手习惯，且 hash
相同/不同不直接度量策略概率差异或该差异的价值。

模型/端点/对手 checkpoint 和拥有实现/native extension 的 SHA 未变；每个 wave
的四条件 roster/源/采样 metadata 相同。新增 prefix 测试确认 ROUND 只截断己方，
对手 prefix 仍为 FULL，且采样器不读 hidden counts；总计 14 项仪器测试通过。

### 主指标：完整比赛胜率

| 端点 | FULL | ROUND | FULL−ROUND | 配对 95% CI | 预定双比较用 97.5% CI |
|---|---:|---:|---:|---:|---:|
| 原始 qk-both control | 374/512 = 73.05% | 354/512 = 69.14% | +3.91 pp | [+0.20, +7.62] pp | [−0.20, +8.01] pp |
| norm16 / moment transport | 366/512 = 71.48% | 372/512 = 72.66% | −1.17 pp | [−5.27, +2.93] pp | [−5.86, +3.52] pp |

control 的 FULL 胜而 ROUND 负／相反为 **65／45**，净多赢 20 场；norm16 为
**50／56**，净少赢 6 场。前者的单项 95% 区间略高于零，属于正向迹象；但本轮
事先约定同时查两端点，以 97.5% 区间控制两个主比较，该区间仍跨零，**尚未通过
预定正收益门槛**。不能事后改用单项门槛宣布已经确立收益；也不应抹去正向信号，
写成“原版完全没有收益迹象”。

norm16 的点估计为负，但区间同时允许正负几个百分点，不能宣布其历史有害或
ROUND 与 FULL 等效。两端点当前都不足以支撑长训或主线采纳决定。

### 端点比较与对手分层（次要/描述性）

norm16 相对 control 的历史收益 interaction 为 **−5.08 pp**，95% CI
**[−10.16, 0.00] pp**，97.5% CI **[−10.94, +0.78] pp**；没有范数 16 增强
跨局收益的证据。FULL 下 norm16 相对 control 的胜率差为 **−1.56 pp**，95%
CI **[−5.66, +2.54] pp**，未确立两端点棋力差异。

四个对手各占每条件 128 场，以下只报告 FULL−ROUND 点估计，不挑某一层作为主结论：

| 冻结对手 | control | norm16 |
|---|---:|---:|
| u2623 | +0.78 pp | 0.00 pp |
| u20264 | +3.12 pp | −4.69 pp |
| u35588 | +7.81 pp | −1.56 pp |
| u86800 | +3.91 pp | +1.56 pp |

| 条件 | 平均局数/场 | 平均累计 return/场 | 先按场求每局平均 return 后取均值 |
|---|---:|---:|---:|
| control FULL | 10.174 | +5.3574 | +0.7310 |
| control ROUND | 10.234 | +5.0391 | +0.6952 |
| norm16 FULL | 10.043 | +5.6094 | +0.7601 |
| norm16 ROUND | 9.770 | +5.9668 | +0.8209 |

这些 return 是描述性指标，没有用其替代预定完整比赛胜率。两端点均高于此前
近邻池的绝对胜率，不能据此称模型突然变强：本轮换了明显不同、且强弱不同的旧
对手。比较历史价值必须看同一池内 FULL−ROUND，而不是跨池绝对胜率。

### 用人话说与停止点

**历史会改变出牌；原版有一点“记住整场更能赢”的迹象，范数 16 没看到。**
原版多赢 20/512 场，但按事先约定同时查两臂的标准，证据仍差一点，不能宣布
已经坐实。范数 16 没显示更会把历史信息转成胜率。

结合上一轮，当前可以区分三件事：这批对手的公开历史确实含有可预测线索；
actor 的公共路径确实受历史范围影响；但**可靠的净比赛收益仍未确立**。这不是
“模型完全不看历史”，也不是“模型已经有效识别并利用对手习惯”的证明。
本轮不训练、不改生产实现或主线、不追加比赛、不自动安排下一步。

证据目录中的 `summary.json` 保存原统计代码输出与整场路径计数，`workers.json`
保存各 worker 退出状态；`results/wave-00..15/{metadata.json,waves.jsonl,complete.json}`
为原始逐场记录及完成标志。`source-manifest.json`、`evidence-manifest.json` 和
`validation.json` 保存源/证据 SHA 与独立计数核验。

重算：

```sh
PYTHONPATH=.claude/worktrees/heuristic-kirch-e482e8:python:oracle:. \
  .work/external/danlm-venv/bin/python bench/qknorm_diverse_match.py \
  --output .work/qknorm-diverse-match-2026-10-16 --summarize
```
