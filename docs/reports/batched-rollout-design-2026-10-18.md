# 设计：C++ 批量环境 + GPU 批量推理的搜索 rollout（2026-10-18）

状态：已实现并验证（2026-10-18）。用于[第 0b 阶段](habit-attention-ab-plan-2026-10-17.md)和以后需要大量搜索的评估。不改训练代码，不改主线。

## 为什么要改

实测（[CUDA 基准](../../.work/gpu-bench-2026-10-17/bench.json)，2026-10-17）：现有 `team_rollout` 把模型搬到 CUDA 后，单个进程快 4.7 倍。但批次再加大，每条线也只从 11.7 降到 8.4 毫秒，因为每条线、每一步都要在 Python 里算合法动作、编码候选、编码观测、追加事件。结果是 12 个进程共用一张 4090，吞吐只有 607 条线/秒，和 64 核 CPU 机器差不多。

## 三处改动

**1. 环境推进放进 C++（`gd.VecEnv`）**

- `VecEnv.load(states)`（新增）：把任意局面放进槽位，比如每个重配后的手牌世界一个槽位。
- `EnvConfig.stop_at_round_end`（新增）：一局结束后记录 `RoundResult`，然后这个槽位进入空闲，不再出现在 `pending()` 里。
- `drain_public_arrays()`（新增）：公开事件直接以 numpy 数组返回，供向量化写入历史 token。
- 合法动作、观测和候选的编码、只剩"过"时自动打掉，都已经在 C++ 里批量完成（`pending()` / `step()`）。

**2. 一个网络、一次前向**

风格对手和我方用的是同一个 u2623 权重，风格只是在最后的 log 概率上加 `z·b·f(a)` 再归一化。所以每一步对所有活跃的线只做**一次**前向，然后按行对对手座位加上各自 z 的偏置。原来要做三次前向、建三个缓存。

**3. 共享前缀的 KV 缓存**

现有的 `ForkedHistoryCache` 会把根局面的整段历史 KV 复制给每条线。第 3 局时历史有几百个事件，每条线约 2.4 MB，几千条线放不下。新缓存的做法是：

- 根历史的 KV 和编码输出，**每个搜索只存一份**；
- 每条线只存它在 rollout 里新增的事件（通常不到 100 个）；
- 流编码器的每一层、以及决策时的私有 attention，都分成"根部分"和"后缀部分"两段算分数，合并后再做 softmax。数学上与整段 attention 相同，只是浮点累加顺序不同。

## 跨比赛批处理

每场比赛写成一个生成器，需要模型的时候就 yield 一个请求：

- `probs`：现场的单个决策；
- `search`：一次残局配对搜索（根局面、候选、世界种子）。

调度器同时推进几十场比赛，把同一轮里所有比赛的 `search` 请求合成一个大批次一起跑：所有线一起 `pending` / 前向 / `step`，直到全部打完。`probs` 请求合成一次前向。

## 随机数

- **参考模式**：沿用 `MatchSampler`，每条线、每个座位一个独立的 numpy 生成器。用于和现有 `team_rollout` 逐位对照。
- **快速模式**：按（线种子、座位、步数）做计数器哈希（splitmix64），完全向量化。抽样分布相同，具体数值和参考模式不同。正式运行用快速模式。

## 验证（Mac 上，在定算力之前完成）

1. **C++**：`load` + `stop_at_round_end` 的贪心 rollout 与标量引擎逐位一致（`cpp/tests/test_env_search.cpp`，已通过）。
2. **单步**：批量前向的候选 log 概率与 `decision_probabilities` 一致，误差 ≤ 1e-5，覆盖我方和两种风格。
3. **单个搜索**：参考模式下，Q 表与 `team_rollout` 一致（小模型和真实 u2623 都测）。
4. **整场比赛**：参考模式下，生成器版与 `habit_crossround_value.play_match` 的结果一致。
5. **速度**：在 Mac CPU 上与旧版对比，记录每秒条数。CUDA 速度要上 GPU 才能测。

## 实测结果（2026-10-18）

- 等价性：小模型和 u2623 在参考模式下，Q 表与 `team_rollout` 逐位一致；CPU 与 CUDA 结果逐位一致；调度器的精确模式与 `play_match` 逐位一致（[tests/test_batched_rollout.py](../../tests/test_batched_rollout.py)、[tests/test_habit_crossround_batched.py](../../tests/test_habit_crossround_batched.py)，C++ 部分见 `cpp/tests/test_env_search.cpp`）。
- 速度（同一个 384 条线的搜索）：RTX 4090 上 CUDA 用 1.57 秒，同机 CPU 用 12.8 秒，旧版 CUDA 用 4.5 秒。16 个搜索合成一批时单进程约 530 条/秒，主要被每步的固定开销拖住。在第 0b 阶段的实际任务里，4 个进程共用一张 4090，约 1,900 场/小时；本地 Mac CPU 3 进程约 68 场/小时。
- 注意：4 个进程、每批 4096 条线时，显存接近 24 GB 上限，需要设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，或者把 `--max-branches` 调小。
- 还能改进的地方：每步按搜索个数循环的小运算、零散的主机到显卡拷贝。
