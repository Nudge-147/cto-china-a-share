"""Test the round-lot hypothesis and quantify downstream materiality."""
from __future__ import annotations

import argparse
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

from config import (
    EXPECTED_BARS_PER_DAY, PRECISION_INVESTIGATION_DIR, PRECISION_REGIME_START_DATE,
    RAW_5MIN_DIR, RECONCILIATION_INVESTIGATION_DIR, ROUNDING_LIKE_REGIME_START_DATE,
    ROUNDING_MAX_ERROR_PER_BAR, ROUNDING_RANDOM_WALK_STD_PER_BAR,
    SECOND_SOURCE_CODES, SECOND_SOURCE_PROBE_DATE, STOCK_LIST_PATH,
    VOLUME_FEATURE_LOOKBACK_DAYS,
)


RECONCILIATION_PATH = RAW_5MIN_DIR.parent / "qc_report" / "reconciliation_daily.csv"
ROUNDING_HARD_BOUND = EXPECTED_BARS_PER_DAY * ROUNDING_MAX_ERROR_PER_BAR
ROUNDING_DAILY_STD = np.sqrt(EXPECTED_BARS_PER_DAY) * ROUNDING_RANDOM_WALK_STD_PER_BAR


def load_codes(limit: int) -> list[str]:
    """Read the tested stock prefix."""
    return pd.read_csv(STOCK_LIST_PATH, dtype=str)["code"].head(limit).tolist()


def load_reconciliation(codes: list[str]) -> pd.DataFrame:
    """Load reconciliation rows with signed volume gaps."""
    frame = pd.read_csv(RECONCILIATION_PATH, parse_dates=["date"])
    frame = frame[frame["code"].isin(codes)].copy()
    frame["year"] = frame["date"].dt.year
    frame["daily_minus_minute_volume"] = frame["daily_volume"] - frame["minute_volume"]
    frame["abs_volume_gap"] = frame["daily_minus_minute_volume"].abs()
    frame["within_rounding_bound"] = frame["abs_volume_gap"] <= ROUNDING_HARD_BOUND
    return frame


def rounding_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize mismatch magnitude against round-lot theory."""
    mismatch = frame[frame["abs_volume_gap"].gt(0)]
    rows = [summarize_gap("overall", mismatch)]
    rows.extend(summarize_gap(str(year), group) for year, group in mismatch.groupby("year"))
    return pd.DataFrame(rows)


def regime_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare the three observed archive regimes."""
    periods = [
        ("pre_precision_switch", "2020-01-01", "2023-08-08"),
        ("large_systematic_gap", PRECISION_REGIME_START_DATE, "2025-03-19"),
        ("mostly_rounding_like", ROUNDING_LIKE_REGIME_START_DATE, "2025-12-31"),
        ("current_archive", "2026-01-01", "2026-08-07"),
    ]
    rows: list[dict[str, object]] = []
    for label, start, end in periods:
        selected = frame[frame["date"].between(start, end)]
        mismatch = selected[selected["abs_volume_gap"].gt(0)]
        row = summarize_gap(label, mismatch)
        row["stock_days"] = len(selected)
        row["mismatch_share"] = len(mismatch) / len(selected)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_gap(label: str, frame: pd.DataFrame) -> dict[str, object]:
    """Produce one rounding-bound decision row."""
    gap = frame["abs_volume_gap"]
    return {
        "period": label, "mismatch_days": len(frame),
        "rounding_hard_bound_shares": ROUNDING_HARD_BOUND,
        "rounding_random_walk_std_shares": ROUNDING_DAILY_STD,
        "median_abs_gap": gap.median(), "p95_abs_gap": gap.quantile(0.95),
        "max_abs_gap": gap.max(), "within_bound_days": int(frame["within_rounding_bound"].sum()),
        "within_bound_share": frame["within_rounding_bound"].mean(),
    }


def monthly_precision(codes: list[str]) -> pd.DataFrame:
    """Locate archive precision changes by month and divisibility."""
    rows: list[dict[str, object]] = []
    for code in codes:
        if code.startswith("sh.688"):
            continue
        frame = pd.read_parquet(RAW_5MIN_DIR / f"{code}.parquet", columns=["date", "volume"])
        frame["month"] = pd.to_datetime(frame["date"]).dt.to_period("M").astype(str)
        for month, group in frame.groupby("month"):
            rows.append({"code": code, "month": month, "bars": len(group),
                         "multiple_10_share": group["volume"].mod(10).eq(0).mean(),
                         "multiple_100_share": group["volume"].mod(100).eq(0).mean()})
    return pd.DataFrame(rows)


def close_materiality(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure absolute and relative size of final-close mismatches."""
    close = frame[frame["close_absolute_error"].ge(0.01)].copy()
    close["relative_error_bps"] = close["close_absolute_error"] / close["daily_close"] * 10_000
    return pd.DataFrame([{
        "covered_stock_days": len(frame), "close_mismatch_days": len(close),
        "close_mismatch_share": len(close) / len(frame),
        "median_abs_error_yuan": close["close_absolute_error"].median(),
        "p95_abs_error_yuan": close["close_absolute_error"].quantile(0.95),
        "max_abs_error_yuan": close["close_absolute_error"].max(),
        "median_relative_error_bps": close["relative_error_bps"].median(),
        "p95_relative_error_bps": close["relative_error_bps"].quantile(0.95),
        "over_10bps_days": int(close["relative_error_bps"].gt(10).sum()),
        "over_10bps_share_of_all_days": close["relative_error_bps"].gt(10).sum() / len(frame),
    }])


def load_minutes(codes: list[str], reconciliation: pd.DataFrame) -> pd.DataFrame:
    """Load minute bars and add proportional daily-volume correction."""
    frames: list[pd.DataFrame] = []
    daily = reconciliation[["code", "date", "minute_volume", "daily_volume"]]
    for code in codes:
        frame = pd.read_parquet(RAW_5MIN_DIR / f"{code}.parquet", columns=["date", "time", "volume"])
        frame["code"] = code
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    result["bar_time"] = pd.to_datetime(result["time"]).dt.strftime("%H:%M:%S")
    result = result.merge(daily, on=["code", "date"], how="left")
    result["corrected_volume"] = result["volume"] * result["daily_volume"] / result["minute_volume"]
    return result


def add_relative_volume(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate raw and proportionally corrected 20-day same-time ratios."""
    frame = frame.sort_values(["code", "bar_time", "date"]).copy()
    keys = ["code", "bar_time"]
    for source, baseline in [("volume", "raw_baseline"), ("corrected_volume", "corrected_baseline")]:
        frame[baseline] = frame.groupby(keys)[source].transform(
            lambda values: values.shift(1).rolling(VOLUME_FEATURE_LOOKBACK_DAYS,
                                                    min_periods=VOLUME_FEATURE_LOOKBACK_DAYS).mean()
        )
    frame["raw_relative_volume"] = frame["volume"] / frame["raw_baseline"]
    frame["corrected_relative_volume"] = frame["corrected_volume"] / frame["corrected_baseline"]
    return frame.dropna(subset=["raw_relative_volume", "corrected_relative_volume"])


def rank_correlation(group: pd.DataFrame) -> float:
    """Compute Spearman correlation without a SciPy dependency."""
    left = group["raw_relative_volume"].rank()
    right = group["corrected_relative_volume"].rank()
    return left.corr(right)


def sensitivity_row(label: str, frame: pd.DataFrame) -> dict[str, object]:
    """Summarize a proportional-correction sensitivity scenario."""
    frame = frame.copy()
    keys = ["date", "bar_time"]
    frame["absolute_feature_change"] = (
        frame["raw_relative_volume"] - frame["corrected_relative_volume"]
    ).abs()
    correlations = frame.groupby(keys).apply(rank_correlation, include_groups=False).dropna()
    return {
        "period": label, "bar_stock_observations": len(frame),
        "median_abs_feature_change": frame["absolute_feature_change"].median(),
        "p95_abs_feature_change": frame["absolute_feature_change"].quantile(0.95),
        "median_cross_section_rank_corr": correlations.median(),
        "p05_cross_section_rank_corr": correlations.quantile(0.05),
    }


def volume_feature_sensitivity(minutes: pd.DataFrame) -> pd.DataFrame:
    """Test materiality under a daily proportional-correction scenario."""
    frame = add_relative_volume(minutes)
    before = frame[frame["date"].between(PRECISION_REGIME_START_DATE, "2025-03-19")]
    after = frame[frame["date"].between(ROUNDING_LIKE_REGIME_START_DATE, "2025-12-31")]
    return pd.DataFrame([sensitivity_row("2023-08-09_to_2025-03-19", before),
                         sensitivity_row("2025-03-20_to_2025-12-31", after),
                         sensitivity_row("all_available", frame)])


def fetch_sina_minutes(code: str, refresh: bool) -> pd.DataFrame:
    """Cache the recent AkShare/Sina minute sample."""
    cache = PRECISION_INVESTIGATION_DIR / "sina_cache" / f"{code}.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not refresh:
        return pd.read_csv(cache)
    symbol = code.replace(".", "")
    frame = ak.stock_zh_a_minute(symbol=symbol, period="5", adjust="")
    frame.to_csv(cache, index=False, encoding="utf-8")
    return frame


def second_source_spotcheck(reconciliation: pd.DataFrame, refresh: bool) -> pd.DataFrame:
    """Compare three boards on a recent common archive-regime anomaly."""
    rows: list[dict[str, object]] = []
    for code in SECOND_SOURCE_CODES:
        frame = fetch_sina_minutes(code, refresh)
        frame["day"] = pd.to_datetime(frame["day"])
        sample = frame[frame["day"].dt.strftime("%Y-%m-%d").eq(SECOND_SOURCE_PROBE_DATE)]
        reference = reconciliation[(reconciliation["code"].eq(code)) &
                                   (reconciliation["date"].eq(SECOND_SOURCE_PROBE_DATE))].iloc[0]
        rows.append({"date": SECOND_SOURCE_PROBE_DATE, "code": code, "sina_bars": len(sample),
                     "sina_volume": pd.to_numeric(sample["volume"]).sum(),
                     "baostock_minute_volume": reference["minute_volume"],
                     "daily_volume": reference["daily_volume"],
                     "sina_last_close": pd.to_numeric(sample["close"]).iloc[-1],
                     "baostock_last_close": reference["minute_close"],
                     "daily_close": reference["daily_close"]})
    return pd.DataFrame(rows)


def write_outputs(tables: dict[str, pd.DataFrame]) -> None:
    """Write auditable CSV evidence."""
    PRECISION_INVESTIGATION_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(PRECISION_INVESTIGATION_DIR / f"{name}.csv", index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--skip-second-source", action="store_true")
    parser.add_argument("--refresh-second-source", action="store_true")
    args = parser.parse_args()
    codes = load_codes(args.limit)
    reconciliation = load_reconciliation(codes)
    tables = {"rounding_bound_summary": rounding_summary(reconciliation),
              "regime_summary": regime_summary(reconciliation),
              "monthly_precision": monthly_precision(codes),
              "close_materiality": close_materiality(reconciliation),
              "volume_feature_sensitivity": volume_feature_sensitivity(
                  load_minutes(codes, reconciliation))}
    if not args.skip_second_source:
        tables["second_source_spotcheck"] = second_source_spotcheck(
            reconciliation, args.refresh_second_source)
    write_outputs(tables)
    for name, table in tables.items():
        print(f"\n{name}\n{table.head(20).to_string(index=False)}", flush=True)


if __name__ == "__main__":
    main()
