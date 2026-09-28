# Transformer 策略臂：第一步盘点（2026-09-25）

状态：只读盘点，没有改代码，没有启动训练。目的是回答"哪些能直接复用，哪些要新写，
哪些事要先决定"，为第二步（离线试跑）和第四步（RL 三臂）铺路。决定本身记录在
`docs/DESIGN.md` 7.2、7.3 和 `STAGE_C_TODO.md` C5、C8；本文不重复论证。

约定的目标模型：公开事件流（每手牌一个 token，整场比赛不压缩）经因果 Transformer 编码，
自己的手牌等私有信息作为一个 query 去 cross-attend，得到状态向量；状态向量接现有的
打分头和辅助头。"猜对手下一手"是可开关的辅助损失。策略头做固定词表和候选打分两种对照。

## 1. 能直接复用的

| 组件 | 位置 | 现状 | 缺什么 |
|---|---|---|---|
| 公开流 + 私有 query 的 Transformer | `train/belief_model.py` `HistoryBelief` | 因果掩码、padding、BOS、私有 query 的 cross-attention 和 FF 块都有，6 个单元测试 | 没有局数嵌入；位置编码是固定正弦；输出头只有猜暗牌 |
| 单环境 KV cache | `train/history_cache.py` | 逐 token 追加、每个前缀与整段前向一致，7 个测试 | batch 固定为 1；每局 `clear()`；权重一变就拒绝服务 |
| 公开事件流 | `VecEnv(log_public_actions=True)` + `drain_public_actions()` | 每次出牌、过牌、强制过牌、进贡都有事件，带 env、match、round、seat、phase、编码后的动作、剩余张数、forced 位；每个环境内有序，`pending()` 之后 drain 到的就是每个待决策的完整前缀 | PPO 从没打开这个开关；没有"一局结束"事件，也没有局末亮牌 |
| 整场比赛的环境 | `cpp/src/env.cpp` `advance()` | 一个环境槽自动打完整场再开下一场；`match_id` 变化就是流的边界；`fork` 无人调用，`reset` 只在初始化 | 无 |
| token 编码 | `train/logs.py` `public_token`，186 维 | 座位 one-hot、154 维动作、剩余张数 one-hot | 没有局数、阶段、forced 位 |
| 抽象动作词表 | `cpp/include/gd/action.h` `kNumAbstract = 393`，`Action.abstract_id` 已绑定到 Python | 类型 + 主值 + 炸弹张数 + 三带二的对子档 + 同花顺花色 | 没有"抽象动作到具体牌"的反向映射；`DecisionBatch` 里没有每个候选的抽象编号 |
| 状态向量的接入口 | `GuandanModel.score_candidates(..., state=None)`，`StageBPolicy.logits(..., state=None)` | 已经允许外部算好状态向量再打分 | 打分头、辅助头和 `_score_static_candidates` 的权重切片都按 `state_width` 取值，新塔输出宽度要么对齐要么改这一个常量 |
| 离线数据 | `.work/belief-styled/collect-100k-m1`，schema 2 | 十万局、555 万个决策，来自真正的 M1；每局带 `match_id` 和 `round_index`，跨局的流可以离线拼回来 | 无 |

canonical 模式下 `wild_usage = minimal` 让每个抽象动作的逢人配用量由手牌唯一决定，
所以 393 个编号不必再乘以逢人配数量。

## 2. 要新写的，按工作量排

1. **rollout buffer 和 learner 的序列重算。最大的一块。** 现在每行只存 obs、hidden、候选、
   选择、logp 等，行与行独立打散，没有 `match_id`、`round_index`、前缀位置，也没有 token 存储。
   PPO 多轮 epoch 要重算状态向量，就必须按 (env, match) 把 token 流存下来，每行记一个前缀位置，
   learner 按比赛分组跑一次流编码再按位置取。`train/rollout_buffer.py`、`train/ppo.py` 的
   `minibatch_loss` 和 `policy_terms` 都要动。
2. **评测钩子。这是阻塞项，不是锦上添花。** `eval/duplicate.py`、`eval/danlm/arena.py`、
   `eval/arena.py` 都是一次给策略一个局面，对手的动作通过 `engine.apply` 直接落地，策略
   不知情。带流的模型在这些地方拿不到任何历史。要给 `eval/policies.py` 加一种能看到每次
   落地动作的策略类型，三个评测入口都要调用它；`load_policy` 目前只认 dmc、a2、ppo 三种 stage。
3. **批量 KV cache，或者干脆每步重算前缀。** actor 每次 `collect_once` 都重载权重，缓存必须
   从存下来的 token 重建，每次更新一次。是缓存还是重算，由第三步的吞吐测量决定，这里不定。
4. **token 布局。** 加局数（嵌入或 one-hot）、阶段、forced 位。186 这个数被 `tests/test_encoder_golden.py`
   之外的 `history_cache` 输入校验和几个 belief 测试钉住，改了要一起改。
5. **每个候选的抽象编号进 `DecisionBatch`。** 小的 C++ 加法，不改规则。Python 侧从候选的 154 维
   读类型、主值、炸弹张数、逢人配数是可以的，但三带二的对子档和同花顺花色要重新推导，不如引擎直接给。
6. **抽象动作到具体牌的选法。** 引擎只枚举，没有反向映射。同一个抽象动作在 canonical 模式下
   仍可能有多个变体，差别在保留哪些同花顺相关的花色。唯一的现成先例是 `tribute_bot` 的
   "避开同花顺相关牌"规则。建议就用这条：多个变体里选占用同花顺相关牌最少的那个。这条动
   canonical 生成，按 CLAUDE.md 第 6 条要先在 RULES.md 加测试向量再实现。
7. **NTP 辅助头。** 标签是流里下一个非强制事件的抽象编号，按座位分开记分。forced 位事件里有，
   token 里现在没有，回到第 4 项。

## 3. 要先决定的事

- **联赛对手怎么办。** 公开流对四家相同，但编码依赖权重，每一个 Transformer 对手都要自己的缓存。
  第一臂建议联赛对手全部保持 MLP checkpoint，Transformer 快照进联赛往后放。
- **去掉 top-32 剪枝牵连什么。** 剪枝不只是候选集合：`ref_log_probs`、`cached_reference_kl`
  的 KL 正则，以及对手和 learner 共用的冻结参考前向，都建在它上面。固定词表头天然不需要剪枝，
  但 KL 项要另做决定，不能只删候选集。
- **进贡阶段。** 保持现有路径不动。启发式进贡下 learner 行本来就只有出牌阶段。
- **critic 不动。** 它是 obs 加暗牌的独立 MLP，不共享参数。
- **局末亮牌 token。** C8 设计里有，引擎现在没有这个事件，也没有把剩余手牌暴露给 Python。要就是
  一个新的 C++ 事件，第一臂可以先不要。
- **引擎源码摘要。** `engine_source_digest` 哈希了 `config.h` 和编码器，任何 C++ 改动都会让
  旧的 belief 和 memory 实验校验失败。这是后果，不是阻碍，但要在报告里写明。

## 4. 顺手发现的一个隐患

`cpp/src/movegen.cpp` 的 `drop_duplicates` 比较时忽略 `fh_pair_rank`，排序又不稳定。手里有
三张 X 加两张逢人配时，XXXWW 会对每个对子档各生成一次，去重后留下哪一个的对子档是任意的，
它的抽象编号也就是任意的。现在没影响，因为没人用抽象编号；固定词表一上就是真 bug。另外
`gd.abstract_id(tuple)` 对三带二给的编号是错的（对子档传了 -1），不要用它。

## 5. 第二步现在就能做的

不需要任何 C++ 改动，本地就能开始：

- 用 `collect-100k-m1` 按 (env, match) 把各局 token 拼成跨局的流，Python 侧加局数列。
- NTP 标签用下一个 token 的类型、主值、炸弹张数列，194 个条目的最小词表就够，不需要对子档和花色。
- 在同一份数据上训三样：flat 对照、Transformer 纯黑盒（只拟合 M1 的决策）、Transformer 加 NTP。
  再把策略头换成固定词表跑一遍。看损失曲线是否正常、掩码有没有漏、两种头哪个拟合得好。
- 子采样在 Mac 上跑通；十万局的完整探针要一张卡，几个 GPU 小时。

第三步的吞吐测量和第四步的三臂都要先租卡。
