# Botzone 计分与动作合法性复核（2026-10-22）

## 范围与来源

用户要求查明天梯换算规则及 GuanZero 是否因非法牌型受罚。本次为只读调查和本地验证，未修改运行代码、权重、线上版本或天梯设置。
源码基准 HEAD：`09d793f877471d83cc4a0b6a155c0a5dbeaa5035`；工作区原有文档修改保留。
裁判来源为本次用户粘贴的完整 Python 文件；测试仅移除未使用的 NumPy import 后执行其函数。
脚本和结果保存在 `.work/botzone/audit-20261022/{check.py,result.json,candidates.py,candidates.json}`，属于本机忽略证据。

官方[天梯说明](https://wiki.botzone.org.cn/index.php?title=天梯)于本次通过浏览器读取：公开规则为对参与 Bot 的所有无序二元组进行 Elo 更新。页面最后修改于 2015-10-11，未给出 K、得分映射、多人归一化或掼蛋换座成组计分细节。不能据此断言赢 3 分的天梯增量是赢 1 分的三倍，也不能断言两者相同。[DouZero 补充材料 E.2](https://proceedings.mlr.press/v139/zha21a/zha21a-supp.pdf)说明的是斗地主双局总分决定胜负，不能直接套用到掼蛋。

## 已保存线上记录的重新验证

扫描 `.work/botzone/{platform_logs,v5_logs,v5_logs_oct6}/*.html`，按 match ID 去重，共 60 局。GuanZero 共 1,341 次响应，全部平台 verdict 为 OK，未见 debug notes。752 次响应带 u20264 checkpoint 前缀 `3191f5826b2d`，589 次未标记；这批不是 v7 的近期审计。

使用原始裁判 request/response 前缀和初始发牌恢复输入，对其中 1,281 次 GuanZero 出牌执行粘贴裁判的 `main()`：未产生任何 error。记录均为无进贡局，该脚本的 initdata 重建只适用于此范围。

有一局结束原因是 `ILLEGAL_CLAIM`：[6ac4dc36e2453c4f4720b779](https://www.botzone.org.cn/match/6ac4dc36e2453c4f4720b779)，违规为座位 3，GuanZero 是座位 2；不能算作我们的违规。平台 verdict OK 本身不足以证明动作合法，必须另外检查裁判 display.error。

## 当前候选生成与声明转换

固定种子 20261022，13 个级牌各 100 手牌，共 1,300 个随机 27 张手牌。对当前纯 Python 引擎生成的 canonical 领出候选，经 `claim_faces`、实体 ID 转换、`physical_claim` 后，调用用户裁判的 `isLegalClaim` 和 `checkPokerType`。

- 244,332 个可表达候选：声明合法且识别牌型全部符合预期，0 错误。
- 26 个无法表达候选被现有 None 过滤屏蔽，未发送给裁判。
- 包含 97,805 个三带二、70,663 个顺子、5,005 个炸弹、2,830 个同花顺及 2 个王炸。

这不是穷举，不测试完整跟牌状态机、所有比较 key、运行时限或贡还阶段。已保存线上动作回放则包含真实跟牌场景。
本地 v7 提交包 `.work/botzone/dist/gz_bot_u73780_search.py` 解包后，protocol/pyengine/mirror/search/bot 五模块与当前源码逐字节相同；numpy_actor 已有变化。没有重新下载线上源码逐字节核验。

## 发现的异常兜底风险

`eval/botzone/bot.py:154` 的 `_fallback` 并不保证合法：进贡/还贡直接返回剩余 hand 的首牌，未检查最大非配子/还贡上限或抗贡；手牌重建也未完整纳入收取的贡还牌。正常路径使用合法候选，这个分支只在前面的处理抛异常后运行。

本地合成反例：tribute 阶段 hand 含实体 ID `[4, 53]`（红桃 2、大王），调用 `_fallback` 返回 `[4]`；用户裁判 `isValidTribute(..., level='2')` 判 `INVALID_MOVE`，因为应贡大王。此简化反例验证分支缺少规则约束，不代表线上曾出现该状态或曾触发兜底。保存的 60 局未见该分支触发。

## 当前线上状态与限制

本次公开[账号页](https://www.botzone.org.cn/account/6ac14300e2453c4f471cb216)显示最新版本仍为 v7，描述为 u73780 + root search。两种浏览器均未登录；公开全局比赛列表中未完成 v7 最近对局的逐步日志审计。因此不能把旧记录或候选采样宣称为“当前线上所有局零违规”。旧 v7 本地 parity artifact `judge-u73780-parity.json` 可读，记载 30 局、2,411 次决策、0 illegal、无 note 样本，属于历史本地裁判证据。

结论：已检查记录未发现 GuanZero 非法牌型/声明或运行错误；当前正常候选路径有较强的采样兼容证据，但异常兜底存在可复现规则漏洞。未在本次调查中修复或部署。
