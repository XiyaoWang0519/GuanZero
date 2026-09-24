# GuanZero 夜间实验结果 — 2026-09-24

五小时续训终点在内部比赛和独立 DanLM 对照中都优于 B11；残局搜索也取得了小幅独立增益。优势过滤明显退化，候选扩展没有证明提升。开发集挑出的主训练“最佳”检查点没有通过独立增益检验，因此不能把开发集排名当成最终强度排名。

本次实测 RunPod 余额减少 **$7.34834**，低于 **$20** 总上限。12:33 UTC 的供应商核验显示账户 **0 个 pod、$0/小时**，两个训练 pod 和三个探针全部释放；没有为了用完预算增加实验。训练、评测、备份和报告归档已完成，定时检查 `guanzero` 已关闭（PAUSED）。

**训练与对照**

从 06:21:44 UTC 开始，截止时间为 13:21:44 UTC。主 GPU 完成 18,000.28 秒训练，4,450 updates、583.16M 总决策，其中 284.80M 是学习方决策。主训练使用冻结源码 `4940246`、seed 240924、B11 warm start 和显式重建的对手池。对手池没有完整沿用原 B11 训练状态，所以这证明的是本次续训配置的效果，不能严格归因于“仅增加训练量”。

另一张 GPU 使用冻结源码 `cf40250`，依次运行 control、filter、union，各 4,500 秒，同训练 seed 2026092431、同 B11 起点、同对手池。filter 使用优势绝对值分位阈值 0.8、最低阈值 0.05；union 使用参考策略 top-32 加 8 个均匀随机候选。两项功能均默认关闭。各臂完成 1,153 / 1,137 / 1,088 updates；union 在同墙钟预算内少处理约 5.6% 决策。因此比较包含计算开销，不能单独分离候选质量的影响。主训练不是这三臂的匹配对照。

**最终 DanLM 结果**

开发集固定为 1,000 duplicate deals、seed 2026092401；最终集为另一个 seed 2026092477 的 4,000 deals，50% 含进贡，每副牌交换双方后打两局，由 DanLM 引擎裁判。最终种子没有用于挑选检查点。负的净升级分代表仍落后于 DanLM，表中的胜率是单局先出完一方的比例，不是完整比赛胜率。

| 模型 | 有效 deals / 4,000 | 每局净升级分 | 单局胜率 |
|---|---:|---:|---:|
| B11 起点 | 3,998 | −2.0562 | 11.67% |
| 75 分钟 control 终点 | 3,998 | −2.0271 | 12.33% |
| 75 分钟 filter 终点 | 3,997 | −2.2761 | 8.81% |
| 75 分钟 union 终点 | 3,998 | −2.0479 | 11.96% |
| 主训练开发集最佳 | 3,999 | −2.0320 | 11.89% |
| 主训练五小时终点 | 3,997 | **−1.9601** | **13.21%** |

以下均为同牌、双方共同有效整副牌的差值，95% percentile bootstrap CI 使用 5,000 次整副牌重采样。不能用上表不同有效集合的均值相减替代配对差值。

| 对照 | 共同有效 deals | 净升级分差值 [95% CI] | 解释 |
|---|---:|---:|---|
| 五小时终点 − B11 | 3,995 | **+0.0974 [+0.0567, +0.1368]** | 本次续训有支持的提升 |
| 开发集最佳 − B11 | 3,997 | +0.0246 [−0.0148, +0.0630] | 未证明提升 |
| control − B11 | 3,996 | +0.0293 [−0.0095, +0.0673] | 未证明提升 |
| filter − control | 3,995 | **−0.2489 [−0.2876, −0.2116]** | 拒绝当前过滤配置 |
| union − control | 3,996 | −0.0210 [−0.0589, +0.0186] | 未证明提升或明确退化 |

五小时终点相对 B11 的配对单局胜率提高 **1.56 个百分点 [0.70, 2.44]**。预先指定的五小时终点与开发集最佳都在看到这两份最终结果前冻结。开发集最佳来自第 2,000 次更新、约 2.22 小时，开发集分数 −1.9860 高于终点的 −2.0145，但最终集上终点更强。终点减开发集最佳为 +0.0727 [+0.0343, +0.1130]，这是检查点间的补充比较；不能事后把终点改称“开发集最佳”，或继续用这份最终集调参后宣称独立验证。两份模型及原始结果都保留，项目默认模型没有自动替换。

五小时终点与 B11 比较排除的 deal IDs 为 846、1009、1361、2719、3984；开发集最佳与 B11 为 846、1361、2417。原因均为桥接引擎 `mirror_failed`，完整失败 leg、所有其他对照的排除 ID 和逐 deal 差值保存在原始文件中。有效局中仍存在合法动作表示差异，因此这是固定桥接协议下的外部校准结果。三个搜索/训练方向和多个对照的区间均为逐比较区间，未做跨所有实验的多重比较校正；单一训练 seed 的区间也不衡量重新训练的波动。

**内部比赛与残局搜索**

五小时终点对 B11 的内部 4,000 副 duplicate 得分差为 **+0.1386 [+0.0979, +0.1779]**；1,000 场完整比赛赢 **594 场，59.4% [56.3%, 62.4%]**。Greedy、bomb-happy、bomb-shy 三格区间为正，其余五格包含零，包括三个 heldout styles；没有证据证明每种风格都提升或都非劣。完整审计见 [主训练内部报告](/Users/xiyaowang/Documents/Projects/GuanZero/docs/reports/main-continuation-internal-2026-09-24.md)。

filter 内部对 control 为 −0.3284 [−0.3660, −0.2894]，完整比赛 197/1,000；union 为 +0.0020 [−0.0353, +0.0411]，490/1,000。独立与内部结果方向一致。过滤诊断发现保留样本中 pass 占比明显升高、梯度范数和 KL 增大，但尚未识别唯一失败机制。详见 [过滤诊断](/Users/xiyaowang/Documents/Projects/GuanZero/docs/reports/filter-diagnosis-2026-09-24.md) 和 [候选扩展报告](/Users/xiyaowang/Documents/Projects/GuanZero/docs/reports/candidate-union-pilot-2026-09-24.md)。

残局搜索使用自己的手牌和公开约束重采样隐藏牌，没有读取真实暗牌；它是均匀合法采样原型，尚不是学习到的信念分布。以 B11 为底座的内部独立 4,000 deals 得到 **+0.0490 [+0.0381, +0.0601]**。另一组独立 DanLM seed 2026092497、3,998 共同有效 deals 得到 **+0.01588 [+0.00925, +0.02251]**。这支持固定协议下的小幅搜索增益；没有测过“新五小时模型 + 搜索”的叠加效果。

内部搜索记录 38,995 次触发全部完成、774 次改变动作，额外搜索延迟 p50 **9.66 ms**、p95 **30.41 ms**、最大 **114.68 ms**。这是内部运行聚合延迟，不能当作 DanLM 运行逐决策延迟，也不是正式 G9 验收。详见 [搜索实现与延迟](/Users/xiyaowang/Documents/Projects/GuanZero/docs/reports/stage-c-search-v0.md) 和 [独立迁移报告](/Users/xiyaowang/Documents/Projects/GuanZero/docs/reports/stage-c-search-transfer-2026-09-24.md)。

**费用、额度与资源回收**

| 资源 | 按 manifest 时长估计费用 |
|---|---:|
| 主 RTX 4090，分配至确认释放约 5.326 小时 | $3.9733 |
| 三臂实验 RTX 4090，约 4.236 小时 | $3.1600 |
| 三个清理机制探针合计 | $0.2235 |
| 估计合计，含保守存储费用 | $7.3567 |

供应商实际余额 **$50.3131313317 → $42.9647907151**，观测扣费 **$7.3483406166**；时长估计比观测值高约 $0.0084，保留两种口径，不把估计当账单。实时核验为 0 pods、$0/小时，预约余额全部释放。每个开发代理关联的云成本都小于 $15，没有自动充值或购买 Codex 额度。

结束检查时 Codex **7 天共享额度剩余 83%**；未返回第二个窗口，所以五小时余量未知。开始时剩余 96%，变化属于账户共享用量，不能精确归因到本任务或换算成每个代理美元费用。三个 GPT-6-sol 使用共享 Codex 额度；上面的美元核算是可观测的 RunPod 云成本。

供应商定时删除和容器自删探针均未通过，实际采用本机独立守护、备份脚本和定时检查。主备份末尾两次超时已恢复，第 95 次备份成功，最终状态 complete；实验共 79 次备份、状态 complete。资源已由这些脚本释放，不能据此宣称供应商自身具有硬期限保护。完成后本任务的开发集监控也已停止。

**交付位置与复核**

- [机器可读完整摘要](/Users/xiyaowang/Documents/Projects/GuanZero/docs/reports/overnight-results-2026-09-24.json)：各臂结果、配对区间、排除局、冻结选择、费用与供应商确认。
- [五小时终点 checkpoint](/Users/xiyaowang/Documents/Projects/GuanZero/.work/overnight-20260924/final/main-final.pt)：SHA-256 `ebb614f35f825e835c9d5b18bdc5a6efb0f79d5ba5ed9bfcdf44eca98ae139fe`。
- [开发集最佳 checkpoint](/Users/xiyaowang/Documents/Projects/GuanZero/.work/overnight-20260924/evals/main/20260924T091154Z/checkpoint.pt)：SHA-256 `c282f13aff3f194c531c945068ac268c55f24731253b13eba59875a565d6e83d`。选择记录在 [main-final-selection.json](/Users/xiyaowang/Documents/Projects/GuanZero/.work/overnight-20260924/main-final-selection.json)。
- [最终文件哈希清单](/Users/xiyaowang/Documents/Projects/GuanZero/.work/overnight-20260924/final-artifact-manifest.json)：最终模型、每臂原始 `.raw.json.gz`、结果/配对 JSON、评测脚本、冻结源码归档及来源信息。
- [主 GPU 备份清单](/Users/xiyaowang/Documents/Projects/GuanZero/.work/overnight-20260924/main-results/sha256-manifest.json) 和 [实验 GPU 备份清单](/Users/xiyaowang/Documents/Projects/GuanZero/.work/overnight-20260924/experiments-results/sha256-manifest.json)：94 / 139 个文件，覆盖 cfg、训练/评测日志、checkpoint 和 league。哈希是本地下载文件的 SHA-256；没有声称独立远端哈希比对。
- [费用账本](/Users/xiyaowang/Documents/Projects/GuanZero/.work/overnight-20260924/budget-ledger.json) 和 [供应商最终状态](/Users/xiyaowang/Documents/Projects/GuanZero/.work/overnight-20260924/final-provider-confirmation.json)。

功能代码已集成在 `codex/overnight-20260924`：优势过滤 `b521030`、候选扩展 `413fcd8`、独立评测 `cf40250`、安全搜索 `fdf684d`。运行中的源码归档始终未修改。云端训练/恢复预检通过，主冻结源码 C++ 76 cases、Python 27 tests 通过；搜索集成另有 C++ 81 cases、相关 Python 47 tests 和主代理集成检查。DanLM 最终批次全部正常退出，主结果另经独立代理逐 deal 重算核对。

下一轮优先验证五小时终点在新训练 seed、固定同一对手池下是否复现，再用新的保留牌局评测“终点 + 搜索”。开发集选模型的方差需要降低。当前过滤配置不继续投入，候选扩展保留为未证实实验。今晚不再新开训练或付费资源。
