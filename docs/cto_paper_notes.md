# CTO 论文阅读记录

论文：Jing-Zhi Huang, Zhijian (James) Huang, Zhuo Li, Fenghua Wen, “Selling at the Opening: The ‘T+1’ Rule, Short-term Speculation, and Stock Returns”。文件为 SSRN preprint，69 页，未同行评审。

## 样本区间与样本构造

- 中国全部 A 股，2000 年 1 月至 2020 年 12 月。
- 基准单变量检验：4,047 家上市公司、518,122 个 firm-month observations。
- 2000 年作为起点，理由是保证组合形成的观察数和会计数据口径一致。
- 主要数据来自 CSMAR；机构持股来自 WindDB；股吧数据来自 CNRDS；2011 年以前的管理层业绩预告来自 RESSET，2010Q4 起由 CSMAR 补充。
- 月度 CTO 是该月每日 close-to-open overnight return 的平均值：
  `Daily_CTO(i,d) = (OPEN(i,d) / CLOSE(i,d-1) - 1) × 100`。
- 计算中剔除：上市首日；前一交易日收盘触及涨跌停的观察。价格按拆股和分红调整。
- 含会计变量的检验从 2002 年 4 月起，因为此前上市公司只需披露半年报和年报；股吧变量可用期从 2008 年起。

## 分组方法

- 每月底 t 按 CTO 横截面排序为 10 个等权 deciles，D1 为最低 CTO、D10 为最高 CTO。
- 持有并计算下月 t+1 的等权（EW）和市值加权（VW）组合收益；论文主要比较 D10 - D1，即 long high CTO / short low CTO。
- 双排序：先按 14 个公司特征分成 terciles，再在每个 tercile 内按 CTO 分成 10 组，共 30 组；把三个特征 tercile 中相同 CTO decile 合并为 10 个组合，持有一个月。
- 控制的 14 个特征：SIZE、BM、BETA、MOM、REV、MAX、-MIN、ROE、GPTA、COSKEW、ISKEW、PRICE、IVOL、TURN。
- Fama-MacBeth：用 t 月 CTO 及控制变量解释 t+1 月个股收益，Newey-West t 统计量使用 5 个滞后项。

## Table 1：组合特征

正文页只保留 “Insert Table 1” 占位，但论文后部附录页实际嵌入了完整 Table 1；关键数值如下：

- 10 个 CTO 组合中有 7 个的平均 overnight return 为负；全样本平均 CTO 约 -0.11%。
- 各 decile 的平均 CTO 从 -0.79% 到 0.56%，D10-D1 的 CTO 差为 135 bps。
- 低 CTO 组表现为小市值、低 BM（growth/glamour）、高 MOM；同时具有较高 IVOL、极端日收益和 TURN，呈现 lottery-like 特征。
- 低 CTO 组的 East Money 股吧 posts、views、comments 均明显高于其他组，说明个人投资者讨论/活跃度更高。
- 论文强调这些特征是同月描述性统计，不能据此建立 CTO 与特征之间的因果关系。
- Table 1 的完整关键行：CTO D1–D10 = -0.79, -0.36, -0.25, -0.18, -0.12, -0.07, -0.02, 0.04, 0.13, 0.56；H-L = 1.35，t=17.15。SIZE H-L=0.35；BM=0.05；MOM H-L=-9.64；REV=3.73；IVOL H-L=-1.30；TURN H-L=-0.71。股吧变量按表注缩放展示。

## 主结果：Table 2

正文页只保留 “Insert Table 2” 占位，但论文后部附录页实际嵌入了完整 Table 2；正文与表格一致。基准数字是：

- EW：D10 高 CTO 组合下月平均收益 1.34%，D1 低 CTO 组合 0.37%；多空（D10-D1）为 **+0.97%/月**，t = 4.62。
- VW：D10-D1 收益差为 **+1.07%/月**，t = 3.51。
- EW 下不同因子模型的 alpha 差约 **+0.88% 至 +1.10%/月**，t 值约 3.87 至 4.91。
- 经济解释：低 CTO 代表短期投机者集中买入、次日开盘卖出压力大；低 CTO 股票随后收益较低，符合投机导致的暂时高估和后续修正。

Table 2 的 H-L（D10-D1）完整结果：EW raw 0.97%（t=4.62），Carhart alpha 1.05%（4.76），FF5 1.06%（4.91），Novy-Marx 0.88%（3.87），q-factor 1.10%（4.16），Stambaugh-Yuan 1.10%（4.76），Daniel-Hirshleifer-Sun 0.92%（4.10）；VW raw 1.07%（3.51），Carhart alpha 1.30%（3.87）。

## 排除 firm-specific news 的检验

### 他们怎么定义“有新闻”

论文不是按媒体报道关键词抓新闻，而是使用公司层面的公告/事件日期。四类 firm-specific news：

1. earnings announcements（EA，业绩/财报公告）；
2. management earnings forecasts（MF，管理层业绩预告）；
3. dividend and stock split announcements and ex-dates（分红、拆股公告及除权除息日）；
4. other important corporate events：外部融资、并购、股权质押、诉讼、公司违规、管理层变动、突发事件。

数据来源：主要新闻日期来自 CSMAR；management forecast dates 由 RESSET 补充（Section 3.1 还说明 2010Q4 起 CSMAR 有管理层预告数据，2011 年以前使用 RESSET 补齐）。涉及第 4 类 other events 的检验从 2002 年 4 月开始，因为该类事件日期数据从此才可用；其他新闻检验仍为 2000/01–2020/12。

### 日级排除窗口

- 对每一类新闻分别做一次，并额外做四类同时排除。
- 构造 modified CTO：剔除新闻事件交易日，以及事件日前一个交易日和后一个交易日，即以公告日为中心的 3 个交易日窗口 `[t-1, t, t+1]`。
- 若公告发生在收盘后，将下一个交易日定义为实际公告日，再按该日展开三日窗口。
- 在排除后的日数据上重新计算月度 CTO，再按 CTO 分十组，观察 CTO 分布与下月收益多空差。

### 月级排除窗口

为处理新闻影响可能持续超过三天，论文另做更强的月份剔除：

- 新闻发生在当月前十个交易日（first half）：剔除公告月；
- 新闻发生在当月后半段：剔除下一个月，因为下月更可能受该新闻影响；
- 每个月每个 decile 至少需要 10 只股票，否则该月不纳入组合统计；这是因为中国财报公告存在月份聚集。
- 论文还报告，简单剔除“包含公告日的月份”时，结论也不变。

### 检验结果

- Table 8：排除三日窗口后，CTO D10-D1 基线 1.35%；分别排除 EA、MF、Dividend、Other 后为 1.36%、1.35%、1.34%、1.33%；四类同时排除为 1.33%。
- 月份剔除法下，D10-D1 分别为 1.35%、1.41%、1.32%、1.36%、1.33%。CTO 分布几乎不变。
- Table 9 的三日窗口、四类同时排除：EW raw 多空 0.87%（t=4.52），Carhart alpha 0.99%（4.80）；VW raw 0.99%（3.26），alpha 1.29%（4.46）。
- 月份剔除法、四类同时排除：EW raw 1.23%（4.32），alpha 1.32%（4.45）；VW raw 1.49%（3.64），alpha 1.79%（3.92）。
- 结论：低 CTO 溢价不是由公司特定新闻驱动；用日级三日窗口或月级剔除，结果均保持。

### 对第三部分数据管道的直接模板

建议至少实现两套口径：

- `event_window_3d`：公告日及前后各一个交易日剔除；盘后公告日期顺延到下一交易日。
- `event_month`：前十交易日公告剔除当月，后半月公告剔除次月；每个组合月设最小股票数阈值（论文为 10）。

事件表至少需要：`stock_code`、`event_type`、`raw_announcement_datetime/date`、`effective_trading_date`、`source`。公告日期缺失时不能静默归入“无新闻”，应单独标记为 `date_missing` 并从排除检验中统计出来。

## 其他主结果

- 双排序 Table 3：控制单个公司特征后，Carhart 四因子 alpha 的 D10-D1 差仍显著；EW 为 53–111 bps/月，VW 为 82–138 bps/月。
- Fama-MacBeth Table 4：无控制变量时 CTO 系数 0.809，Newey-West t = 6.64；对应 D10-D1 的隐含月度收益差约 **1.09%**。加入控制变量后 CTO 系数为 0.508–0.824，隐含收益差约 **0.69%–1.11%/月**，t 值均高于 3.0。
- 结果在剔除公司新闻、控制风险/注意力/反应不足、双排序、WLS、样本截取和不同 CTO 形成窗口等检验下保持。

## 第 2 周对标基准

建议把以下数字作为复现/策略开发的第一基准：

> 月末按月度 CTO 分 D1–D10，持有下月；多空 = long D10、short D1。基准收益：EW **+0.97%/月**（t=4.62），VW **+1.07%/月**（t=3.51）。

注意：正文页的表格位置是空占位，但完整 Table 1/2 在论文后部表格页中出现；上述数字已按后部完整表格和正文交叉核对。
