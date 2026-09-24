# Kaggle GuanZero GPU 测速 — 2026-09-24

Kaggle 免费 GPU 可以运行现有 GuanZero 训练代码。本次单 T4 实测 18,502 次决策/秒；两张 T4 各跑一个独立训练任务，合计 28,862 次决策/秒，相对本次单卡配置增加 56%。适合后续并行比较两个实验；这不是一个模型使用双卡 DDP 的加速结果，也不是棋力结论。

## 实测硬件与配置

- 2 × Tesla T4，每卡 CUDA 可见显存 15,636,037,632 bytes（14.56 GiB）。
- Intel Xeon 2.00 GHz，4 个可用 CPU；affinity 4，cgroup `400000 100000`，实际 quota 4 CPU。
- Python 3.12.13，PyTorch 2.10.0+cu128，CUDA 12.8。
- 冻结源码 `4940246fbd520ff6bd9e246e729f425b69dfcba8`，B11 warm start、相同初始 15 项 league；2 次 update warmup 不计入测速，每臂约 120 秒。
- rollout 64，PPO epochs 4，minibatch 8192，torch threads 1。单卡 2048 环境、引擎 3 线程；双卡每任务 1024 环境、引擎各 1 线程，共享 4 CPU。

## 结果

| 配置 | 决策/秒 | 学习方决策/秒 | 测量秒数 | collect 秒 | learn 秒 | CUDA 峰值 MiB |
|---|---:|---:|---:|---:|---:|---:|
| 单 T4，2048 env | 18,502 | 9,162 | 120.43 | 52.79 | 67.62 | 1190.37 |
| 双卡任务 0，1024 env | 14,037 | 6,931 | 121.39 | 64.51 | 56.87 | 1186.17 |
| 双卡任务 1，1024 env | 14,825 | 7,323 | 123.78 | 69.98 | 53.77 | 1184.11 |
| 双任务吞吐合计 | 28,862 | 14,254 | — | — | — | — |

双任务合计为各任务测量速率之和，两个任务同步启动但完成时间略有不同。改变了每任务环境数、线程分配及 update 大小，因此 1.560× 是这两种实际部署配置的吞吐比，不是严格相同优化器工作的硬件缩放率。短测速只做了一轮，没有多次重复的误差区间；不足以证明长训练稳定性、棋力提升或与 RTX 4090 的严格倍速关系。显存峰值为 PyTorch allocated，非整卡全部占用。

## 执行、额度及保留结果

官方 CLI 查询 Notebook version 2 为 `COMPLETE`；completion 标记成功且所有子进程已停止。C++ 76 个测试全部通过，CUDA smoke、单卡和双任务均成功。脚本从启动到清理耗时 337.94 秒（约 5.63 分钟），配置的脚本预算 720 秒、Notebook timeout 780 秒；没有继续启动训练或替换默认模型。

GPU 额度由 0.00/30.00 小时变为 0.10/30.00 小时，剩余 29.90 小时；这是账户级四舍五入读数，包含本次尝试，不能据此精确推导双 GPU 的长期计费规则。CLI 返回下次刷新时间 `2026-09-26T00:00:00`。本次付费支出 $0。

Version 1 在约 5 秒时因 Kaggle 自动展开上传 tar 包而失败，未进入训练；修复后 version 2 按 120 个文件的 SHA256 验证展开内容并成功运行。首次失败输出保留在 `.work/kaggle-benchmark-20260924/attempt1/`。

用户明确授权约 328 MB 源码和模型私有上传后才执行。[私有 Notebook](https://www.kaggle.com/code/xiyaowang0519/guanzero-t4-benchmark-20260924) 与 [私有 Dataset](https://www.kaggle.com/datasets/xiyaowang0519/guanzero-throughput-kit-20260924) 保留供复现。没有公开或上传额外私有材料。

原始 JSON、配置、各阶段日志和 GPU 采样保存在 `.work/kaggle-benchmark-20260924/results/`；[机器可读报告](kaggle-benchmark-2026-09-24.json) 含硬件、结果、completion、额度以及下载文件和运行脚本的 SHA256。

- 上传包 SHA256：`e550ad5ac6fe6da92dd309c50fa867f85cf46c4113cb25f53a6bf674e774d33e`
- version 2 脚本 SHA256：`1adcc5cefcef26db33cc0d684a25250d1e6459eb22f856e0464bd22e849e4393`

后续可优先用这两张免费 T4 并行跑两个独立实验。本次授权已完成，尚未启动后续实验。
