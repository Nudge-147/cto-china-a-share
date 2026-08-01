# Full-A CTO pipeline

主脚本：[cto_pipeline.py](cto_pipeline.py)

## 运行顺序

```bash
# 1) AkShare 退市清单 + 2019 年后退市股行情覆盖探测
MPLCONFIGDIR=/private/tmp/cto-mpl python3 cto_pipeline.py --validate-delisted

# 2) 全 A 后复权日线（断点续跑；当前 AkShare 代码清单约 5,530 只）
MPLCONFIGDIR=/private/tmp/cto-mpl python3 cto_pipeline.py --download

# 3) 提供历史股票池元数据后生成日度 CTO、月度 CTO 和诊断输出
MPLCONFIGDIR=/private/tmp/cto-mpl python3 cto_pipeline.py --build-cto --metadata path/to/metadata.csv
```

下载目录为 `data/cto/daily_hfq/`，每只股票一个 CSV；日志为 `data/cto/outputs/download_log.json`，支持中断后跳过已完成文件。

元数据 CSV 至少包含：`code,name,list_date,is_st,is_suspended`。当前脚本不会把“当前股票名称”冒充历史 ST 状态；要严格做 2010 至今过滤，需要补入历史 ST/停牌状态表。

## 已实现过滤

- ST：依赖元数据中的历史 `is_st`。
- 上市一年内：`date < list_date + 1 year`。
- 停牌：成交量为 0 或元数据 `is_suspended`。
- 上市首日：每只股票历史序列的第一交易日。
- 前收盘触及涨跌停：以 `abs(close_t / close_{t-1} - 1)` 与规则阈值比较，容差 0.2 个百分点。
- 规则：主板 10%；科创板（688）20%；创业板（300/301）在 2020-08-24 前 10%、之后 20%；ST 5%。

CTO 日度为 `open_t / close_{t-1} - 1`，月度为同一股票当月通过过滤的有效日度 CTO 简单平均，和论文定义一致。月度输出为 `data/cto/monthly_cto.csv`。

## 诊断输出

- `data/cto/outputs/cto_diagnostics.png`：月度 CTO 横截面 P10/P50/P90、年度样本量、前收盘涨跌停剔除比例。
- `data/cto/outputs/yearly_sample_counts.csv`
- `data/cto/outputs/limit_exclusion_timeseries.csv`
- `data/cto/outputs/delisted_coverage.json`

## Baostock 备选源

Baostock 已安装并完成探测。平安银行、贵州茅台和退市拉夏均能返回 2010 至今/全历史日线；退市拉夏可返回到 2022-05-24。全量下载使用 [baostock_pipeline.py](baostock_pipeline.py)，日线只请求一次后复权数据（`adjustflag=1`）；不复权 OHLC 由 `query_adjust_factor` 返回的稀疏 `backAdjustFactor` 按事件日向前填充后还原，即 `raw_price = hfq_price / backAdjustFactor`。茅台 4,018 个交易日对既有不复权文件的最大还原误差为 `4.55e-13` 元。价格和复权因子仍全部来自 Baostock，不与 AkShare 行情拼接。

```bash
# 先以两个独立会话做分片试跑；各分片写自己的日志并按代码取模，不会重叠。
python3 baostock_pipeline.py --shard-index 0 --shard-count 2
python3 baostock_pipeline.py --shard-index 1 --shard-count 2
MPLCONFIGDIR=/private/tmp/cto-mpl python3 cto_pipeline.py --build-cto --source baostock --metadata path/to/metadata.csv
```

Baostock 股票基础清单当前约 5,566 只，另补入已落盘的交易所退市清单代码；价格数据全程仍只走 Baostock。

当后复权与原始价文件对达到 2,000 只时，运行 `python3 src/run_preview.py` 可在 `data/cto_baostock/preview/` 生成预览诊断图、流通市值及首版十分组回测；正式全样本结果仍以全量下载完成后为准。

## 市值权重

[market_cap_pipeline.py](market_cap_pipeline.py) 用不复权价和 Baostock 日线换手率构造逐日流通市值：`float_shares = volume / (turn / 100)`，`float_market_cap = raw_close × float_shares`。当 `turn < 0.01%`、缺失或停牌时不反推，直接持有上一个有效股本；有效反推值先取 20 日滚动中位数，只有偏离当前阶梯值超过 5% 才更新。这样保留真实解禁/增发跳变，压低成交量和换手率舍入造成的日噪声。若未来改为 AkShare 历史市值，允许仅将其作为权重数据源混入；价格与收益仍必须保持 Baostock 单源，避免复权基准跳变。

## ST 与停牌状态

曾用名重构方案已退役。Baostock 历史日线直接返回逐日 `isST` 和 `tradestatus`：前者用于当日 ST 剔除，后者用于停牌剔除。这比名称变更史更精确，不需要推断区间。

## 当前验证状态

- 退市清单：SSE 159 条、SZSE 208 条，字段已落盘。
- 2019 年后退市股行情探测：本轮 6 只样本均被东方财富行情接口远端断开，因此只能确认“清单可见”，尚不能确认 AkShare 历史后复权行情对退市股的完整覆盖。
- 已用现有三只样本（000001、600519、603157）跑通过滤、月度聚合和图形诊断；这不是全 A 结果。
- 全 A 下载器已实现，但本轮因上游深交所清单/行情接口连接不稳定，没有把未完成的请求伪装成全量完成。
