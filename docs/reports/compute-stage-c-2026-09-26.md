# Stage C 算力平台筛选：速度优先

查询时间：2026-09-26 03:58–04:00 UTC（多伦多 09-25 深夜至 09-26 零点）。
用户偏好：优先速度，每小时超过 US$2 可以接受。本文只完成只读调研，没有租机、充值或启动训练。

当前首选试测 **Vast.ai 的 RTX PRO 6000 Blackwell Workstation 96GB + Ryzen 9950X/9950X3D 单卡主机**；用同平台的 **H100 NVL 94GB + Ryzen 9900X** 做速度对照。**Hyperstack RTX PRO 6000 SE** 是另一平台的备选。这个排序基于当前代码、公开配置和报价，尚不是实测训练速度排名。

## 之前慢在哪里

已重新核对本地实验报告：

- [CPU 配额调查](compute-cpu-isolation-2026-09-24.md)：曾有标称 64 vCPU 的 RunPod，实际 cgroup 配额只有 13.6 或 17.85；同一工作负载在不同主机和时段有明显波动。不能据此断言每一台 RunPod 都存在 CPU 争用。
- [CUDA profile](perf-gpu-2026-09-24b.md)：旧 PPO 每个采集 step 约 281–342 次 kernel launch，包含 Python 调度、数据整理、上传、引擎和 GPU 运算。相同 learner 工作曾出现 19–70 ms 的波动。四 actor 在特定旧模型试验中提升了吞吐，不代表新 Transformer 已支持同样拓扑。
- [流水线试验](perf-pipeline-2026-09-25.md)：一台 54.4 CPU 配额、未观察到明显主机争用的 4090 上，单 Python 控制流仍是限制。拆成更多流水线组反而变慢；四 actor 的 GPU SM busy 也只有 24–33%。不能把所有慢都归咎于平台或 GPU 型号。

## 最新训练需求不同

[Stage C](../STAGE_C_TODO.md) 是随机初始化、完整公开对局历史的 Transformer 自博弈 PPO，旧 MLP 仅用于评估。当前 [试跑方案](history-t4-readiness-2026-09-26.md) 使用 width=64、2 层、FP32、关闭 TF32，先扫 4/8/16 环境；正常架构默认值另有 width=128、4 层，二者不能混作同一个性能结论。

当前收集路径每次决策重算完整历史，尚未接入增量 KV cache。历史增长会增加运算及激活开销；CPU 上已记录的 2,500→580 decisions/s 不能预测 GPU 的绝对速度。

整理期间，仓库新增了 [T4 已完成 GPU 试跑报告](history-t4-pilot-2026-09-26.md)。已直接读取并复算下载的 `sweep/envs-{4,8,16}/metrics.jsonl`：

| RTX 4090，width=64 / 2 层 | 长历史 collect decisions/s | 长历史 collect+learn 唯一训练 rows/s | 最大 reserved GiB |
|---|---:|---:|---:|
| 4 环境，平均前缀 760 | 532 | 479 | 1.41 |
| 8 环境，平均前缀 754 | 663 | 557 | 6.69 |
| 16 环境，平均前缀 729 | 751 | 649 | 17.31 |

16 环境比 8 环境的该项端到端吞吐约高 16.4%，但超过预设的 70% reserved 显存门槛，所以试跑选了 8 环境。报告记录后续 population 试跑完成 200 updates，最长前缀 2,403；allocator reserved 峰值 24.31GB，而 allocated 峰值仅 1.55GB，未 OOM。**预留量不等于活跃张量占用，也不是模型确实需要 24GB 的证明**；需排查变长序列的分配器行为，再判断扩大显存及环境数的收益。这是短程新模型基线，长期学习曲线和跨平台速度排名仍未建立。

因此选型看四项：CPU 单线程与有效配额、GPU 实际计算速度、长历史/多环境/历史策略快照占用的显存、完整 collect+learn 的耗时。当前入口没有已验证的多 GPU 训练扩展，买四卡不会自动使一个训练任务快四倍。

## 当前具体候选

Vast 数据来自官方公开 [search offers API](https://docs.vast.ai/api-reference/search/search-offers)，是当时可租、verified、按需实例的快照。保存的必要字段见 [报价快照](compute-stage-c-2026-09-26.json)。Vast 下表价格包含 50GB 容器磁盘，不含流量和税；CPU 线程数不是物理核心数，RAM 为近似标称容量，实际可见值见 JSON。库存会变化，创建前必须重查。

| 优先级 | 平台及配置 | CPU / 主机内存 | 当时报价 US$/h | 判断 |
|---|---|---|---:|---|
| 首选试测 | Vast，美国宾州，RTX PRO 6000 WS 96GB；offer 52591556 / machine 135371 | Ryzen 9950X3D，16 核/32 线程；约 128GB RAM | 1.5037 | 新 CPU、96GB 显存、报告 GPU 功率上限 600W；适合检验更大环境批量。注意流量较贵 |
| 同类备选 | Vast，澳大利亚，RTX PRO 6000 WS 96GB；offer 50850184 / machine 93097 | Ryzen 9950X，16 核/32 线程；约 96GB RAM | 1.3339 | 相同 GPU 档位，流量便宜；加拿大访问距离远，但单机自博弈循环本身无需跨洋通信 |
| 速度对照 | Vast，美国堪萨斯，H100 NVL 94GB；offer 27615974 / machine 34267 | Ryzen 9900X，12 核/24 线程；约 128GB RAM | 2.1421 | 显存带宽更高，是否能胜过 PRO 6000 要看实际算子和批量 |
| 平台备选 | Hyperstack，RTX PRO 6000 SE 96GB | 官网页面列最大每 GPU 31 pCPU / 180GB RAM | 1.85 | 明确按需价；实际 flavor、CPU 隔离、地区库存尚未确认 |
| 另一速度对照 | Hyperstack，H100 SXM 80GB | 官网页面列最大每 GPU 24 pCPU / 240GB RAM | 3.20 | 值得在吞吐实测支持时使用；不能凭型号认定最快 |

Hyperstack 来源：[当前价格表](https://www.hyperstack.cloud/gpu-pricing)。最大配置不代表任意实例都分配这些资源；额外卷和公网 IP 另计，价格表列公网 IP 为 $0.00672043/h，流量免费。本次没有登录验证账户额度或可立即交付的实例。

按上述 Vast 报价，宾州 PRO 6000 连续 10 小时约 $15.04，H100 NVL 约 $21.42，另计传输与税。宾州候选的 API 流量价为上行 $81.92/TB、下行 $53.33/TB，约 $0.08/GB 与 $0.0521/GB（API 的换算按 1024）；频繁全量下载 checkpoint 会增加费用。澳洲候选相应为 $4.00/TB 与 $2.6667/TB；H100 候选均约 $0.6827/TB。应按实际导出量计入总成本，优先增量同步。

三台主要 Vast 候选都报告 `num_gpus=1`、`gpu_frac=1`。依据 [官方分配规则](https://docs.vast.ai/guides/instances/docker-environment)，CPU 基线按所租 GPU 占该机器的比例分配。这比在八卡机器上只租一张更有利于控制资源份额，**但这些字段不能证明底层物理服务器绝无其他工作负载，也不是裸金属专属合同**。verified 和 reliability 是平台测试/历史指标，不是将来运行的保证；入机后仍需测有效配额、节流和吞吐波动。

## 为什么先比较 PRO 6000 和 H100

当前严格 FP32 的方案不会直接获得宣传中的低精度 Tensor Core 峰值。NVIDIA 给出的 [PRO 6000 Workstation](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf) 理论 FP32 为 125 TFLOPS、显存 96GB；[H100 NVL](https://www.nvidia.com/en-eu/data-center/h100/) 为 60 TFLOPS、94GB，显存带宽 3.9TB/s。PRO 6000 Server Edition 则是另一 SKU，官方 FP32 为 120 TFLOPS，不能混用规格。

这些参数只解释为什么两者都应列入测试：算术密集路径可能偏向 PRO 6000，带宽密集路径可能偏向 H100；小算子和主机调度也可能盖过两者的差别。更大显存可以允许更多并行环境或更长历史，但只有批量实现和学习设置确实利用它时才会转化为吞吐。没有以降低数值精度或截断历史来换速度的建议。

## 其他平台的取舍

- [RunPod](https://www.runpod.io/pricing)：当前 Secure 4090 $0.74/h、5090 $0.99/h；换卡仍应检查 CPU 配额。现有部署和回收工具支持它，迁移工程成本低。旧主机结果并不排除一台配置合适的新 RunPod。
- [TensorDock](https://console.tensordock.com/deploy)：当前公开入口 4090 从 $0.50/h、5090 从 $0.60/h，CPU/RAM 可配置。起价不是我们目标配置的完整报价，也未核实 CPU 独占，因此暂不排在速度优先的前列。
- [Lambda](https://lambda.ai/instances)：单卡 H100 SXM $4.29/h，26 vCPU、225GiB RAM；H100 PCIe $3.29/h。值得作为预算更宽的替代，但目前无证据证明其在本项目上快过上表。
- [CloudRift](https://www.cloudrift.ai/pricing)：5090 按需 $0.60/h；PRO 6000/H100 当前按需显示需询价，$1.16/$1.70 属于月度预留价，不能当作短时试跑价。
- Hetzner GEX131 是真正独立服务器路线，但当前 [官方调整表](https://docs.hetzner.com/general/infrastructure-and-availability/price-adjustment/) 列 GEX131-1 月价 $1,397.10、设置费 $699，均未含税和 IPv4；[交付表](https://docs.hetzner.com/general/infrastructure-and-availability/order-processing/) 列德国约 10–12 个工作日。老报告里的低价快开预期已不适合当前短试验。
- Latitude.sh 页面只有 H100 “from $1.68/h”的公开起价，本次未确认对应区域、付费期限和实际交付报价，不把它当作已核实的短时租用价。

## 建议的下一步

建议单独批准一个**总计不超过 US$10 的顺序对照测试**，而不是先确定长跑平台。这个数是建议预算，尚未获得租机授权。

1. 固定当前源代码快照、依赖、精度、种子、模型大小和环境数，先在 PRO 6000、H100 NVL 上各跑相同的 `bench.history_ppo`；5090 可作为较便宜的第三对照。每个硬件上重复短测，报告中保留主机差异，不能将所有差异归因于 GPU。
2. 覆盖平均历史长度约 144、720 及更长的真实对局，分别记录 collect、learn、完成一批相同训练工作的时间、唯一 learner rows/s、GPU 内存、CPU 配额/节流、GPU 利用率。population 前后分别测，避免只看新局短历史。
3. 先做相同配置比较，再在各卡容量允许范围内找合适环境数。修改 batch/环境数的吞吐不等于学习效率，另保留学习曲线验收。
4. 速度优先选重复测试中长历史端到端最快的合格配置；同时列每百万唯一训练样本成本和包含安装、存储、导出在内的总账单。两卡速度相近时取更便宜且波动更小者。
5. 如果都明显随历史长度变慢，先优化并验证批量 KV cache 及权重更新后的重建，再重复测试。当前试跑门槛是长历史采集吞吐至少保留早期的 50%；4090 上 4/8 环境已通过短 sweep，population 长跑的显存余量和新平台仍需独立验证。

迁移至 Vast 或 Hyperstack 前，需要对应平台的预算/时限回收保护、checkpoint 导出及删除后计费确认。现有 `infra/runpod.py` 不能直接管理其他供应商。本轮没有修改训练程序，也没有开展付费测试。
