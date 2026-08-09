# Close-to-Open (CTO) in China A-shares

**Question and headline finding.** Does the close-to-open return identify a persistent short-term-speculation state under China’s T+1 rule? In a 2010–2026 all-A-share replication, high-CTO stocks outperform low-CTO stocks by 1.53% per month (EW); the return is concentrated in the low-CTO leg and in overnight trading, but largely disappears after realistic opening-limit and trading-cost frictions.

This repository is a research replication and extension of *Selling at the Opening: The “T+1” Rule, Short-term Speculation, and Stock Returns*. It is designed for a five-minute read of the research question and a quick verification of the included outputs. A full vendor-data rebuild is materially longer because it downloads 5,538 stock histories from free endpoints.

## Main results

### CTO long–short return (D10 − D1)

| Period | EW mean monthly return | NW(5) t | VW mean monthly return | NW(5) t |
| --- | ---: | ---: | ---: | ---: |
| Full sample, 2010–2026 | 1.53% | 7.42 | 1.25% | 3.54 |
| Paper-overlap, 2010–2020 | 1.46% | 5.88 | 1.42% | 3.25 |
| Out of sample, 2021–2026 | 1.65% | 4.47 | 0.89% | 1.56 |

![Long-short NAV](results/main/long_short_nav_log.png)

### From academic return to implementable return

The paper-overlap, cost-and-tradability-adjusted EW return is only 0.25% per month (NW t=0.98) at 25 bps one-side costs. This supports the interpretation that implementation frictions can protect the apparent mispricing from arbitrage.

![Implementability waterfall](results/week3/implementation/implementability_waterfall.png)

### Time-of-day decomposition

| Component, D10 − D1 | Full sample monthly return | NW(5) t |
| --- | ---: | ---: |
| Overnight | 3.93% | 25.82 |
| Intraday | −2.77% | −11.30 |
| Total | 1.53% | 7.42 |

The low-CTO leg loses overnight and partially recovers intraday, consistent with a persistent “speculation magnet” state rather than one isolated event. See [`results/week3/overnight_intraday_decomposition.csv`](results/week3/overnight_intraday_decomposition.csv).

### Fixed nonlinear extension

An expanding-window comparison asks whether the formation month's daily return sequence adds information beyond mean CTO, and whether that increment is nonlinear. A five-feature CTO-family Ridge raises mean IC from 0.040 to 0.077, but its out-of-sample long-short return is almost unchanged; the much larger eight-feature Ridge return is mainly associated with added size, turnover, and cumulative-return exposures. Fixed-parameter LightGBM underperforms the full Ridge (2.48% vs. 2.72% gross per month) and turns over more. See the [extension analysis](results/ml_extension/ML_EXTENSION_ANALYSIS.md).

## Data and method

- **Price source:** Baostock daily A-share history, 2010-01-01 to 2026-07-22. Post-adjusted prices construct returns; unadjusted prices perform exchange-price-limit checks. The same vendor is used for both price series.
- **Universe:** 5,538 mainland A-share codes, including delisted stocks. B shares are excluded.
- **Daily filters:** ST (`isST`), non-trading days (`tradestatus`), listing first day, first year after IPO, and observations whose prior close touched the applicable limit. The rule is 5% for ST, 10% for main board, 20% for STAR, and 10%/20% for ChiNext before/after 2020-08-24. The theoretical limit is rounded to RMB 0.01 before comparison.
- **Signal:** daily CTO is `open_t / close_(t-1) − 1`; monthly CTO is the mean eligible daily CTO in the formation month.
- **Portfolio test:** at each month end, sort stocks into CTO deciles; hold next month; report EW and float-market-cap VW returns. Long–short is D10 minus D1. Newey–West uses five lags, matching the paper.
- **Market-cap weight:** inferred float shares use `volume / turnover`, invalid `turn < 0.01%` observations are forward-filled, then shares are 20-day-median smoothed with a 5% step threshold.

### Daily-price lineage note

The files under `data/cto_baostock/daily_raw/` are usually reconstructed from
Baostock post-adjusted OHLC divided by the vendor's back-adjustment factor. Some
therefore retain `adjustflag=1` even though their price levels are reconstructed
to raw levels. This has negligible impact on the CTO return construction, but it
means those files are not treated as native unadjusted observations for the new
five-minute reconciliation pipeline. That pipeline downloads a separate native
`adjustflag=3` daily companion table for every selected stock. See
[`DATA_NOTES.md`](DATA_NOTES.md), DN-005.

## Reproduction

### Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

For a compact cold-start gate after cloning, run the same environment setup and then:

```bash
bash scripts/cold_start_smoke.sh .venv/bin/python
```

This checks dependency imports, compiles all source files, opens the principal CLI entry points,
and runs the complete unit-test suite without downloading vendor data. The Stage-4 final figures
additionally require the locally archived V17 Dataset and are regenerated with
`python finalize_stage4.py`; cloud-only artifacts are not treated as the primary copy.

On macOS, LightGBM also needs an OpenMP runtime (for example, `brew install libomp`).

The unit tests and inspection of included `results/` complete in a few minutes. Full raw-data reconstruction is slower because free Baostock/Eastmoney endpoints are rate-limited.

### Full rebuild order

Run from the repository root. All raw/intermediate files are recreated below `data/`, which is intentionally ignored by Git.

```bash
# 1. Download post-adjusted and raw daily prices (single worker; resumable)
python src/baostock_pipeline.py

# Optional: multiple independent, resumable shards
python src/baostock_pipeline.py --shard-index 0 --shard-count 2
python src/baostock_pipeline.py --shard-index 1 --shard-count 2

# 2. Construct eligible daily/monthly CTO and diagnostics
python src/cto_pipeline.py --build-cto --source baostock

# 3. Build float market capitalization
python src/market_cap_pipeline.py

# 4. Main replication, audits, mechanism, and implementation tests
python src/run_formal_backtest.py
python src/run_week2_audits.py
python src/audit_delisting_returns.py
python src/run_week3_mechanism.py
python src/run_week3_implementation.py

# 5. Announcement extension and frozen Week-4 tests
python src/download_disclosure_events.py --start-year 2010 --end-year 2026
bash src/run_week4_label_pipeline.sh
python src/run_week4_2d_backtest.py
python src/run_week4_power_posthoc.py

# 6. Fixed eight-feature linear-versus-nonlinear extension
python src/ml_extension.py
```

Expected elapsed time: price download roughly 8–12 hours with two shards (endpoint-dependent); disclosure download roughly 1–3 hours; all local construction/backtests several hours on a modern laptop. The scripts are checkpointed where remote retrieval is involved.

### Five-minute pilot pipeline

The pilot universe is selected mechanically by average daily traded amount from
2024 onward. The same rule is intended for the later 1,500-stock expansion.

```bash
pip install -r requirements-5min.txt

# Rebuild the top-100 code-only universe from CTO daily files
python build_stock_list.py

# Conservative two-session download; each process logs in independently
python download_5min.py --limit 10 --workers 2

# Three-layer QC, CSV details, and five diagnostic figures
python quality_check.py --limit 10
```

Minute data are requested by natural year and first saved as
`data/raw_5min/segments/{code}_{year}.parquet`. Historical completed segments are
checkpointed in `progress.json`; the current year is refreshed on every run.
After all requested segments finish, each stock is merged once into
`data/raw_5min/{code}.parquet`. Direct unadjusted daily companion tables are in
`data/daily_reference_unadjusted/`, and QC output is in `data/qc_report/`.

The exact 48 timestamps, the 2019 coverage gap, reconciliation evidence, and the
unresolved high/low vendor difference are documented in
[`DATA_NOTES.md`](DATA_NOTES.md). Comments beside QC assertions reference those
note identifiers.

The first 10-stock end-to-end run found no OHLC, duplicate, negative-value,
invalid-time, or missing-bar assertion failures across 14,326 covered stock-days.
It did find a vendor reconciliation regime concentrated in 2023–2025: daily and
five-minute volume/amount, and occasionally the final close, do not always match
even though every covered day has all 48 bars. These observations remain strict
QC exceptions rather than being silently tolerated; both exact issue lists and
relative-error quantiles are retained.

The 10-stock gate investigation rejected post-market fixed-price and block trades
as the main explanation: they explain only 2.23% of 5,203 volume-mismatch days,
while 75.03% of gaps have the opposite sign required by that hypothesis. An
independent Sina daily feed agrees with Baostock daily volume on every compared
day in 2023–2026, and non-STAR minute volumes switch to exact 100-share granularity
throughout 2024–2025. This evidence localizes the unresolved regime to Baostock's
historical minute archive. Scaling from 10 to 100 stocks was therefore frozen pending repair; see
[`docs/RECONCILIATION_INVESTIGATION.md`](docs/RECONCILIATION_INVESTIGATION.md) and
DN-007/008 in [`DATA_NOTES.md`](DATA_NOTES.md). The follow-up materiality test
rejects a single round-lot-noise explanation for the main regime: the 2024 median
absolute volume gap is 215,292 shares versus a 2,400-share theoretical rounding
bound. It does not mandate a full pipeline migration; a 2024-capable second-source
sample or a validated affected-period repair is the next gate.

That gate is now resolved by DN-009. The known CTO adjustment factor does not
explain the minute-volume ratios, although daily volume and amount ratios move
almost identically. Raw files remain immutable. `repair_5min.py` creates
research-ready files in `data/processed_5min/`, proportionally allocates each
daily reference volume to integer 5-minute shares, preserves `volume_raw`, and
adds `volume_rescaled`, `volume_scale_factor`, and `close_anomaly_flag`. The
10-stock repaired sample reconciles daily volume with zero residual days, so the
100-stock pilot gate is open.

```bash
# After downloading any stock prefix, build the research-ready layer
python repair_5min.py --limit 100

# QC the processed layer; price work must honor close_anomaly_flag
python quality_check.py --limit 100 --raw-dir data/processed_5min \
  --output-dir data/qc_report/processed_100
```

### Five-minute features and baselines

The research layer keeps feature values unstandardized and uses only the current
bar and prior history. Cross-sectional labels require at least 30 eligible
stocks. Each sequence contains 48 bars and nine numeric features plus a
sample-relative `is_cross_day` flag.

```bash
# Build per-bar causal features and same-day forward labels
python features_5min.py

# Build compressed sequence arrays and flat LightGBM/Ridge snapshots
python build_5min_dataset.py

# Run purged 2y/3m/6m rolling baselines in the isolated arm64 environment
MPLCONFIGDIR=/tmp/mpl-5min .venv-5min/bin/python run_baseline.py

# Re-check whether the 10-stock archive regimes generalize to the expanded pool
python validate_regime_100.py
```

The requested 14:35 sampling point is intentionally retained for sequence and
60-minute-label use. Under the separately requested rule that the 30-minute
target may not cross the close, however, 14:35 is one of the final six bars and
therefore has no `fwd_ret_30m` or `label_rank`; the 30-minute baseline uses the
other five daily sampling points. This eligibility loss is explicit in
`data/dataset/flat/dataset_manifest.csv` rather than silently changing either
specification.

The completed 100-stock build contains 866,208 sequence/flat samples, of which
711,780 have an eligible cross-sectional rank label. All eight rolling training
windows contain about 210k observations, below the requested 500k warning level,
and are explicitly flagged. The canonical same-close target produces mean test
RankIC of 0.1016 for LightGBM and 0.0912 for Ridge. A leakage/execution audit
finds that this is dominated by very short-horizon reversal and an optimistic
same-close execution convention: entering one bar later reduces mean RankIC to
0.0397 and 0.0297. Both values are reported; strategy claims should use the
delayed-entry sensitivity. See DN-011 and `data/baseline_report/`.

Stage three adds `run_neural_models.py`: a two-layer 128-unit BatchNorm/Dropout MLP on the exact
baseline snapshots and a one-layer hidden-64 GRU on the 48-bar sequence. Window-one CPU smoke tests
completed without an overfit alert. The provisional delayed RankIC values are 0.0211 for MLP and
0.0344 for GRU versus 0.0263 for LightGBM. Five-seed claims remain gated; use the interruption-safe
Kaggle command in [`KAGGLE_STAGE3.md`](KAGGLE_STAGE3.md). See DN-012 and
[`docs/checkpoints/STAGE3_CHECKPOINT.md`](docs/checkpoints/STAGE3_CHECKPOINT.md).

### Stage-four finalization

Stage four freezes the five-seed GRU attribution, failed-window comparison, execution-timing
appendix, and transaction-cost layer. No model fitting is required for the final publication pack:

```bash
MPLCONFIGDIR=/tmp/mpl-stage4-final .venv-5min/bin/python finalize_stage4.py
```

The command reads the verified local V17 primary copy under `data/archive/`, writes the exact source
tables to [`docs/tables/`](docs/tables/), and writes the unified 220-DPI figures to
[`docs/figures/`](docs/figures/). Window 2/8 interpretation is in
[`docs/STAGE4_WINDOW_2_8_ANALYSIS.md`](docs/STAGE4_WINDOW_2_8_ANALYSIS.md); the same-close versus
one-bar-delayed comparison is in
[`docs/APPENDIX_A_EXECUTION_PROTOCOLS.md`](docs/APPENDIX_A_EXECUTION_PROTOCOLS.md). Research and
engineering provenance is archived in [`docs/checkpoints/`](docs/checkpoints/).

## Data-quality controls

![CTO diagnostics](results/diagnostics/cto_diagnostics.png)

- **Cross-sectional CTO distribution / annual sample count / limit exclusion:** the diagnostic figure documents the 2015 limit-hit spike (28.78%) and the expected post-2020-08 ChiNext rule shift.
- **Timing-overlap audit:** the final formation-month CTO uses the formation-day open and prior close; the held return begins from that formation-day close, so the next-month opening return is not in the signal.
- **IPO audit:** applying the paper’s first-day-only IPO rule increases, rather than explains, the long–short result; the baseline first-year filter is retained as a documented design difference.
- **Delisting audit:** all 106 D1 holding-month delisting events remain in the return panel; removing them changes EW long–short by only 0.002 percentage points.

Supporting CSV files are in [`results/audits/`](results/audits/) and [`results/diagnostics/`](results/diagnostics/).

## Pre-registration and extension

The frozen label definition and predictions are in [`docs/WEEK4_PREREGISTERED_DESIGN.md`](docs/WEEK4_PREREGISTERED_DESIGN.md). The repository is being initialized after the research work; therefore its commit order documents the repository snapshot sequence, **not** an externally timestamped pre-registration. The design snapshot is committed before the two-dimensional result snapshot, but the original pre-Git filesystem timestamps—not Git history—are the weaker evidence that the design file preceded the results.

The resulting Q1 comparison is informative: attention-type, low-CTO stocks have a negative future return, while information-type low-CTO stocks do not. See [`docs/WEEK4_PREREGISTRATION_COMPARISON.md`](docs/WEEK4_PREREGISTRATION_COMPARISON.md).

## Limitations

- Delisting-month returns use the last available quote, not a final liquidation value.
- Float shares are inferred from turnover and can lag true share changes; quarterly balance-sheet data would remain subject to disclosure lag.
- Cost estimates are a lower bound: they do not fully model borrow availability, impact, or all short-sale constraints.
- The announcement extension omits the paper’s fourth category of “other important corporate events,” because no equivalent all-market structured AkShare feed was available.
- The strict non-limit, market-adjusted extreme-return label is sparse in the 2015 bull market: broad market gains are removed and many large jumps reached price limits. Results are therefore less representative of that environment.

## Repository layout

```text
src/       Downloaders, signal construction, backtests, audits, and extensions
tests/     Unit tests for filters, timing alignment, costs, and feature processing
docs/      Research design, audit notes, pre-registration, and result interpretation
results/   Versioned figures and compact CSV result tables; no raw daily histories
```
