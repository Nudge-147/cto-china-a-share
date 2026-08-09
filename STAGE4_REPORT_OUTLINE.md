# A股5分钟横截面预测：归因、成本与可实现性

## 1. 动机与数据

- 研究问题、100只流动性样本及 2020 年以后的5分钟数据。
- 图表：样本覆盖时间线；行业/板块分布；日均成交额分布。

## 2. 数据质量调查

- 2019 覆盖缺口、48-bar 完整性、日分钟对账的 regime 及 volume 重标定。
- 图表：regime 边界图；修复前后对账误差；异常股票清单。

## 3. 特征与实验设计

- 因果特征、48-bar 序列、六个日内采样点、延迟一根 bar 入场。
- 图表：特征和标签时间轴；八个滚动窗口划分表。

## 4. 泄漏审计

- label 打乱测试、purge gap 时间戳审计、窗口内标准化。
- 图表：审计 checkpoint 汇总；打乱后 RankIC 分布。

## 5. 四模型对比

- Ridge / LightGBM / MLP / GRU 的 RankIC、ICIR 和十分组多空。
- 图表：全期汇总表；按窗口 RankIC 时序；四模型十档曲线。

## 6. GRU 增量与失效窗口

- 五种子独立性证据、预注册 0.005 门槛、Window 2/8 失效解剖。
- 图表：seed mean±std；GRU-LGB 差值与指数20日波动/涨跌的描述性对照图。
- 回填结论：W2（2022-10-05—2023-04-04）和 W8（2025-10-05—2026-04-04）的
  GRU−LightGBM RankIC 分别为 0.003041 和 0.001464，均未达到 0.005。两期市场方向相反、
  波动率不处于共同极端，且均不与分钟数据修复 regime 重合；八点证据只支持“增量具有
  时间不稳定性”，不支持单一市场状态解释。详见
  [`docs/STAGE4_WINDOW_2_8_ANALYSIS.md`](docs/STAGE4_WINDOW_2_8_ANALYSIS.md)。

## 7. 归因

- GRU Integrated Gradients 和 MLP permutation importance；检验近端 `ret_5m` 依赖。
- 图表：48×F IG 热图；`ret_5m` 时间位置曲线；GRU/MLP 特征排名。

## 8. 成本与可实现性

- 30分钟调仓的 5/10/20bp 单边成本、盈亏平衡成本、换手/持仓重合度。
- 日内六信号聚合、收盘到次日收盘的降频对照。
- 图表：四模型税前/税后表；成本曲线；日内与日频方案对照。

## 9. 结论与局限

- 将“预测力”与“可交易性”分开下结论。
- 局限：100只流动性样本、Baostock regime 修复、冲击成本静态假设、未做样本外实盘。

## 附录

- 附录 A：同刻收盘与延迟一根 bar 两口径对比，见
  [`docs/APPENDIX_A_EXECUTION_PROTOCOLS.md`](docs/APPENDIX_A_EXECUTION_PROTOCOLS.md)。
- 定稿图表及数据来源清单见 [`docs/figures/README.md`](docs/figures/README.md)。
