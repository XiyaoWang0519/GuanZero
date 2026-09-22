# 算力选项调研（2026-09-22）

目的：为掼蛋 RL 训练找比现有 RunPod 更便宜或免费的算力，并按我们的负载严格判断是否合适。

## 我们的负载（判断标准）

- 单次训练 6-10 小时无人值守；C++ 模拟器吃 CPU 线程，约 5M 参数的 PyTorch 模型用一张 GPU。
- **CPU 是瓶颈**：上次 64 vCPU + RTX 5090，GPU 只有 44-73% 忙。有用区间 16-64 vCPU；GPU 有 4090/5090/A40/L4 级别就够；内存 32 GB 以上。
- 需要：Linux、SSH 或脚本控制、cmake/ninja/编译器、pip、CUDA 12.8 兼容的 PyTorch、能下载几 GB 结果。能用程序创建和删除实例、或能可靠自动关机，这一点很重要。
- 基准：RunPod secure 5090 + 64 vCPU $0.99/h（实际拿到）；RunPod CPU pod $0.03-0.035/vCPU-h；A40 $0.49/h。

一个重要结论先说：**按 vCPU 算，现有的 5090 pod 已经很便宜**（64 vCPU 合 $0.015/vCPU-h，比 RunPod 纯 CPU pod 的一半还低，GPU 等于白送）。付费渠道里很难再便宜一半以上，真正能省钱的是学术集群。

## 对比表

| 选项 | 价格 | GPU | vCPU | 时长限制 | 是否适合我们 | 申请门槛 | 链接 |
|---|---|---|---|---|---|---|---|
| **Alliance（原 Compute Canada）默认分配**，Nibi 在 Waterloo，由 SHARCNET 运营 | 免费 | H100 80GB，可切 MIG（1g.10gb / 2g.20gb / 3g.40gb） | 每 GPU 建议不超过 14 核（Nibi）、12 核（Fir/Narval）、16 核（Rorqual）；要更多核会按更多 GPU 计费、降低优先级 | Fir 最长 7 天；Nibi 最长时限**未查到**（官方 wiki 被反爬挡住） | **适合，但需改用法**：Slurm 批处理，不是 SSH 进 pod；默认分配是最低优先级、排队时间无保证；计算节点可能不通外网，pip 要在登录节点用 Alliance 的 wheelhouse 装。也可以单独申请纯 CPU 作业（32-64 核）跑 rollout 密集部分 | 需要一位加拿大高校教职（PI）注册 CCDB，然后把学生加为 sponsored user；本科生可以被 sponsor | [Nibi](https://www.alliancecan.ca/en/services/compute/nibi)、[账号管理](https://www.alliancecan.ca/en/our-services/advanced-research-computing/account-management)、[默认分配/RAS](https://www.alliancecan.ca/en/our-services/advanced-research-computing/accessing-resources/rapid-access-service)、[CPU/GPU 比例](https://mint.westdri.ca/python/gpu_clusters) |
| **WATGPU**（UW CS 学院 GPU 集群） | 免费 | 未列出具体型号 | 未列出 | 未列出 | 可能适合；Slurm 集群，GPU 归出资的教授组，空闲时所有人可用 | 必须属于 CS（或交叉任命）研究组；本科生需教授替你发邮件申请 | [watgpu.cs.uwaterloo.ca](https://watgpu.cs.uwaterloo.ca/) |
| **MFCF 教学 GPU 集群**（数学院） | 免费 | 8×GTX1080Ti、5×RTX 6000 Ada、1×RTX PRO 6000、3×L40S | 全集群 288 核，每作业只能用 1 个节点 | **每作业最长 12 小时** | 勉强：时长能装下 6-10 小时，但它是教学用途，要有课程关联的 Slurm 账号；排队和每用户限额不明 | 修数学院课程（官方写的是非 CS 的 Math 课程）并由课程开通账号 | [分区与限制](https://uwaterloo.ca/math-faculty-computing-facility/services/service-catalogue-teaching-linux/teaching-gpu-cluster-slurm-partitions)、[访问](https://uwaterloo.ca/math-faculty-computing-facility/services/service-catalogue-teaching-linux/access-teaching-gpu-cluster) |
| Vector Institute / Killarney | 免费 | H100、L40S | 未查 | 未查 | 对我们基本不现实 | 需要 CCDB 账号，并且 PI 由 AI 研究所授予 AIP 类 RAP；UW 只有少数 Vector 教职 | [Killarney](https://www.alliancecan.ca/en/services/compute/killarney) |
| **Vast.ai** | 5090 按需最低约 $0.41-0.43/h，spot 低到约 $0.09；4090 可中断约 $0.29，按需 $0.35-0.50 | 4090/5090 等 | **按主机不同**，要筛选 ≥32 核；带多核的报价通常更贵，**实际价格没查到** | 无硬上限；可中断实例可能被抢 | 适合，前提是筛到多核、高可靠性的按需主机；有 CLI，可在实例内自毁 | 注册、信用卡 | [5090](https://vast.ai/pricing/gpu/RTX-5090)、[4090](https://vast.ai/pricing/gpu/RTX-4090)、[比价](https://getdeploying.com/gpus/nvidia-rtx-5090) |
| RunPod（现用） | 5090：community $0.69，secure $0.99；4090：$0.34 / $0.74；A40：$0.35 / $0.49；L4：$0.44 / $0.49 | 同左 | 定价页标的是**每 GPU 最低规格**（5090 9 vCPU、A40 9、L4 12），实际分到多少核按机器而定 | 无 | 适合，是现有流程。community cloud 便宜约 30%，但要确认能拿到 ≥32 vCPU | 已有 | [pricing](https://www.runpod.io/pricing) |
| TensorDock | 4090 约 $0.35-0.37/h | 4090 等 | 可按需配 vCPU（按核单独计价，**具体价格没查到**） | 无 | 可能适合，VM 形式、可 SSH；主机质量参差 | 注册、信用卡 | [4090](https://www.tensordock.com/gpu-4090.html) |
| GCP Spot g2-standard-16 | 约 $0.83/h（us-central1，2026-06 数据），按需 $1.15 | L4 | 16 | 可能被抢占 | 比现有贵，CPU 少；不推荐 | 需申请 GPU 配额 | [holori](https://calculator.holori.com/gcp/vm/g2-standard-16) |
| AWS g6.4xlarge | 按需 $1.32/h，spot 约 $1.11/h（us-east-1） | L4 | 16 | spot 会被抢 | 贵，不推荐 | GPU 配额 | [vantage](https://instances.vantage.sh/aws/ec2/g6.4xlarge) |
| GCP Spot c2d-highcpu-56（纯 CPU） | 约 $0.61/h 起 | 无 | 56 | 可被抢占 | 只适合纯 rollout；单价已不低于现有 5090 pod | GCP 账号 | [spare cores](https://sparecores.com/server/gcp/c2d-highcpu-56) |
| Hetzner | 2026-06 涨价后：AX102-1 €257/月 + €129 开通费；云 CCX63（48 vCPU）约 €1.63/h；GPU 服务器 GEX131-1 €1197/月 | GEX 系列有 RTX PRO 卡 | AX102 16 核 | 按月（独服） | **不推荐**：涨价后既不便宜也不按小时 | 注册 | [价格调整](https://docs.hetzner.com/general/infrastructure-and-availability/price-adjustment/) |
| Lambda Cloud | A100 $1.99+/h，H100 $3.99+/h | 数据中心卡 | 多 | 无 | 太贵 | — | [pricing](https://lambda.ai/pricing) |
| Salad | 4090 约 $0.16-0.20/h（批处理档 4 vCPU） | 消费级卡 | 约 4 | 随时被抢 | **不适合**：CPU 太少、只能跑容器、没有 SSH | — | [pricing](https://salad.com/pricing/) |
| Google Colab 免费 / Pro / Pro+ | 免费 / $9.99/月（100 CU）/ $49.99/月（500-600 CU，来源说法不一） | T4、L4、A100 | 约 2 个（高内存档约 8 个，**未核实**） | 免费版最长 12 小时，空闲会断开；Pro+ 后台最长 24 小时 | **不适合**：CPU 太少；免费版禁止 SSH | Google 账号 | [FAQ](https://research.google.com/colaboratory/faq.html) |
| Kaggle | 免费，GPU 约 30 小时/周 | T4×2 / P100 | 约 4 个（**常见说法，未核实**） | 每会话 12 小时 | 不适合：CPU 太少 | 手机验证 | [文档](https://www.kaggle.com/docs/efficient-gpu-usage) |
| Lightning AI | 免费 15 credits/月（约 22 小时 T4） | T4/L4/A10G/L40S | 免费 Studio 4 核 | Studio 会重启 | 不适合正式训练；可用于编译和冒烟测试 | 注册 | [pricing](https://lightning.ai/pricing/) |
| SageMaker Studio Lab | 免费 | T4 级别 | 少 | 每天 4 小时 GPU | 不适合；据报道 2026-07-30 已停止新注册（**第三方来源，官网 403 未核实**） | — | [来源](https://www.thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025) |
| Modal | 免费 $30/月；学术额度最高 $10k（需申请） | T4 到 B300 | CPU 按核计费（$0.0000131/核·秒，约 $0.047/核·h） | 函数有超时 | 学术额度值得申请；但它是无服务器形态，要改成 Modal 函数，且按核计费比 RunPod 贵 | 学术申请 | [pricing](https://modal.com/pricing) |
| Azure for Students / GitHub 学生包 | $100 Azure 额度 | 实际拿不到 GPU | 学生订阅默认配额只有 3 vCPU | — | **不适合** | 学校邮箱 | [Q&A](https://learn.microsoft.com/en-us/answers/questions/2141487/request-for-gpu-enabled-vm-quota-increase-for-azur) |
| DigitalOcean（原学生包） | 2026-08-01 起已退出学生包，此前 6 月起就不能用于 GPU | — | — | — | 已无 | — | [来源](https://aistudentdiscount.com/digitalocean-github-student-developer-pack-credits/) |
| GCP 教育额度 / AWS Educate | GCP 新用户 $300（90 天）；教育额度一般通过课程或教授发放，面向课程的约 $50；AWS Educate 额度**未核实** | GCP 默认没有 GPU 配额 | — | — | 只能当一次性小额补贴 | — | [GCP edu](https://cloud.google.com/edu/students) |

## 推荐排序

**最佳免费 / 学术路径：Alliance 账号，首选 Nibi（就在 Waterloo，由 SHARCNET 运营）**
- 免费，H100 加大量 CPU；Fir 单作业最长 7 天，我们 6-10 小时的作业很宽松。
- 要注意的代价：
  1. 默认分配是最低优先级，GPU 排队时间没有保证。
  2. 每张 GPU 只配 12-16 核。我们要 32 核以上，要么多占 GPU 份额（降低优先级），要么把 rollout 拆到纯 CPU 作业。实际可以先试「1 张 GPU 或 MIG 3g.40gb + 14 核」测吞吐。
  3. 要把 `preflight.sh` / launch 脚本改成 sbatch 形式。依赖用 `module load gcc cmake cuda` 加 Alliance wheelhouse 里的 torch；装 CUDA 12.8 版 torch 前要先确认 wheelhouse 里有没有这个版本。
- WATGPU 作为补充：需要找一位 CS 教授 sponsor，找的人和 Alliance 可以是同一位。

**最佳付费路径：继续用 RunPod，并尝试更便宜的档位**
- 现有 secure 5090 + 64 vCPU $0.99/h，按每 vCPU 算已经是本次调研里最便宜的实测报价。
- 可以试：
  - RunPod community 5090（$0.69）或 A40（$0.35），前提是分到 ≥32 vCPU，下单前核对 vCPU 数。
  - Vast.ai 按需 4090/5090，筛选 `cpu_cores_effective>=32`、`reliability>0.98`、verified 主机。预计 $0.4-0.7/h，**没实测**。
- 两家都能用 CLI 创建/删除，也能在实例内自毁，符合成本安全要求。

**避免**
- Colab、Kaggle、Lightning 免费版、Salad：CPU 只有 2-4 核，是我们的瓶颈，而且有会话或抢占限制。
- Azure 学生版：3 vCPU 配额，开不了 GPU。
- Hetzner：2026-06 涨价，独服按月计费还要开通费。
- AWS / GCP 的 L4 实例：贵 10-70%，CPU 只有 16。
- Lambda：数据中心卡太贵，对 5M 参数的模型是浪费。

## 下一步（学生可操作）

1. **找教授 sponsor Alliance 账号**：优先找做 RL、游戏 AI 或 ML 的 CS/Math/ECE 教授（可以借 URA、USRA、课程项目或 directed study 的名义）。邮件要点：
   - 一段话介绍项目（掼蛋 RL，C++ 引擎加 PPO，已有可复现结果和报告）。
   - 算力需求：每次 6-10 小时，1 GPU + 16-32 核，每周几次。
   - 请求：请教授在 CCDB 以 PI 身份注册（如果还没有），并把你加为 sponsored user。你需要他的 CCRI（格式类似 `abc-123-01`）。
   - 顺带问能否帮你申请 WATGPU（watgpu-admin@lists.uwaterloo.ca，需要教授发出或抄送）。
2. 拿到 CCRI 后在 [ccdb.alliancecan.ca](https://ccdb.alliancecan.ca/account_application) 注册，角色选本科生，填教授的 CCRI，等教授批准。
3. 在 Nibi 登录节点确认三件事：torch wheel 版本、CUDA module 版本、计算节点能不能上外网。然后用 `sbatch --gpus=h100:1 --cpus-per-task=14 --time=10:00:00` 跑一次 preflight，记录排队时间和吞吐。
4. 付费侧同时做一次对照：Vast.ai 筛 ≥32 核的 4090/5090，跑 30 分钟吞吐测试，和 RunPod $0.99 那档比每千局的成本。
5. 可选：申请 Modal 学术额度（上限 $10k），作为不排队的备用。

## 未能确认的数字

- Nibi 最长作业时限、GPU 节点每节点核数、默认分配下的排队时间（docs.alliancecan.ca 被反爬挡住）。
- Vast.ai 上带 ≥32 vCPU 的 4090/5090 实际报价；TensorDock 按 vCPU 计价的单价。
- Colab 和 Kaggle 的 vCPU 数；Colab Pro+ 每月是 500 还是 600 CU（来源不一致）。
- SageMaker Studio Lab 停止新注册（仅有第三方来源）；AWS Educate 的 $100 额度。
- WATGPU 的硬件和时限。
