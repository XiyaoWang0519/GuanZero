# Append 内部分项诊断（2026-10-14）

用户要求不采用7452索引复用，保留已验证的index_select，并继续拆事件整理、
KV读写、attention、FFN；如果没有明显瓶颈，直接报告停止，不继续实验。
仅现有集群，不付费、不改或重启主线。7452负结果保留。

唯一新作业 **7457（gz-append-detail）**，normal、4GPU、96CPU、35分钟，
工作负载1980秒上限；逐卡检查至少两健康卡、私有MPS。已完成 exit 0；最终结论见下方。
基线保留math backward修复、page-span、index_select，原gloo；没有索引复用
或pinned归约，没有新优化候选、没有测速臂。

## 方法与验证范围

两个阶段各从冻结init恢复，8ranks，各24warmup+8记录=32完整更新。
event阶段使用当前stream CUDA events，update末统一同步读取event；CPU
提交时间同时记录，原collection/learn分项同步关闭。sync阶段每个scope前后
同步并开启原profile。两者都有探针开销，不当无诊断开销PPO速度，也不能
将二者差值称为收益。共享GPU/MPS会影响event与wall时间。

AST只为原ordinary/merged append语句添加计时上下文，不重排算术；KV读取
保留此前index_select。分项：events_layout（含分配/新事件切片/页表/写布局）、
upload、embedding、mask_rows、qkv_projection、kv_write、query_layout、
kv_read、attention、output_projection、ffn、final_norm_write。ordinary.finish
是output_projection/ffn的父项，不能重复加；metadata CPU范围里的event时间
可能含host提交空隙，不能当GPU计算。记录每次append batch、新token数、
chunk_width、max_old_length、span，便于解释padding和历史长度。

普通路径CPU tensor-only45轮对照通过，包括分页、reset/prune、权重变更；
merged GPU路径仍待作业内执行验证。Python/shell语法通过。此轮是原算术诊断，
不是新候选的数值门槛；若诊断后产生候选，必须另过原1e-4 logp/完整历史重算、
选择/RNG、梯度有限性检查，再无探针完整PPO配对测速。

## 决策与停止条件

完成后核验两阶段8×32更新/回执及源码SHA，回收文本大小SHA清单，复核生产
完整source_identity/main manifest。对比event/sync各rank与各update：寻找
稳定较大的具体分项，并检查代码中是否存在可消除的重复工作或等价实现。
仅一个大数字、计时扰动、或正常必需计算本身不构成继续试验理由。

若没有明确且可操作的瓶颈，或证据不足以支持具体候选，直接说明并停止，
暂停自动检查，不再追加探索性profile/盲目新试验。如果有明确瓶颈，用户已
授权做隔离修复/优化及数值门槛后的完整PPO；只提交一个有界验证，保持公共
输入边界、完整历史、FP32、数学目标、训练超参和原数值阈值，不改主线。

本地 `.work/stack-append-detail-2026-10-14/` 保存脚本SHA、submission/state、
local-test.json；远端experiments同名目录。source/init沿用stack-ab冻结身份。

## 2026-10-15 完成：按用户停止条件收尾

7457 exit 0，三张健康 L40S，event/sync 两阶段均8ranks各32更新，连续index、
逐rank及阶段/作业回执核验通过。103个文本/脚本文件大小/SHA256核验通过。
生产完整source_identity与冻结源码一致，main manifest字节一致。没有主线
修改、重启或checkpoint保存。原文remote-results，复算脚本analyze.py及
analysis.json在本地同一实验目录。

分析取各阶段最后6次更新×8ranks=48样本。下表合并ordinary和merged的同名
叶子分项，单位秒/rank/update；排除ordinary.finish父项，避免重复计入。

| 分项 | event CUDA时间 | sync墙钟时间 |
|---|---:|---:|
| attention | 0.3671 | 0.3680 |
| FFN | 0.1298 | 0.2341 |
| KV读取 | 0.0978 | 0.1094 |
| KV写入 | 0.0351 | 0.0518 |
| 事件/布局整理 | 0.0483 | 0.0677 |
| 上传 | 0.0507 | 0.0540 |
| 输出投影 | 0.1042 | 0.1327 |
| QKV投影 | 0.0543 | 0.1246 |

CPU事件整理的event数值含host空隙，不是GPU计算时间；不同计时列不能当
相同物理量比较或相减。sync改变调度，CUDA events也有开销/MPS竞争；表中
不是生产占比或可节约预算。FFN/QKV两种方式的差异尤其说明不能用单列推算
完整PPO收益。

attention在两种计时下均为最大单项。ordinary各rank均值约0.199–0.274秒
(event)、0.201–0.282秒(sync)，merged约0.115–0.138/0.123–0.157秒。
每阶段记录约6,800次append，平均chunk_width约27、span约1,500、batch约61，
是长历史下的必要增量编码工作；平均值不能证明具体padding比例或可删计算。

代码已复核：append每层写入新KV、读取完整所需历史、执行一次masked SDPA
及FFN，保持公共历史语义；本次没有证据表明重复算同层相同输入或可安全跳过
这些操作。事件整理/上传相对较小；前次最终gather索引复用已证明无明显收益。
因此虽有最大耗时项，**尚无明显、可操作且有收益证据支持的新优化点**。

按用户明确要求停止：不追加profile、不猜测kernel替代收益、不提交新候选
或PPO测速。本轮不是新数值门槛/新速度收益；只保留此前验证的index_select
隔离基线，拒绝索引复用和pinned归约。自动检查暂停。以后若有新证据或用户
重新授权再评估，不声称不存在任何理论优化空间。
