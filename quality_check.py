"""Three-layer quality checks for Baostock 5-minute parquet files."""
from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path
from typing import Any

from config import PLOT_CACHE_DIR

PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    AMOUNT_ABSOLUTE_TOLERANCE, DAILY_REFERENCE_DIR, EXPECTED_BARS_PER_DAY,
    EXPECTED_BAR_TIMES, HIGH_LOW_EXPLANATION_RATIO, HIGH_LOW_OPEN_MATCH_TOLERANCE,
    PLOT_BINS, PLOT_DPI, PRICE_TOLERANCE, QC_REPORT_DIR, RAW_5MIN_DIR,
    SPECIAL_DAY_MIN_COVERAGE_RATIO, STOCK_LIST_PATH, VOLUME_ABSOLUTE_TOLERANCE,
)


EXPECTED_TIME_SET = set(EXPECTED_BAR_TIMES)
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
DETAIL_NAMES = [
    "assertion_ohlc", "assertion_duplicates", "assertion_nonnegative",
    "assertion_invalid_time", "bar_count_issues", "completeness_daily",
    "coverage_gaps", "zero_volume_by_stock_time", "reconciliation_daily",
    "strict_reconciliation_issues", "high_low_open_hypothesis",
]


def load_codes(path: Path, limit: int | None) -> list[str]:
    """Load the requested prefix of the canonical stock list."""
    codes = pd.read_csv(path, dtype=str)["code"].dropna().drop_duplicates().tolist()
    return codes[:limit] if limit else codes


def load_minutes(path: Path) -> pd.DataFrame:
    """Load one canonical minute parquet with normalized timestamps."""
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_daily(path: Path) -> pd.DataFrame:
    """Load one directly downloaded unadjusted daily reference."""
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for column in NUMERIC_COLUMNS + ["tradestatus"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("date")


def assertion_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return row-level hard assertion violations (DN-001)."""
    ohlc_bad = ((frame["low"] > frame[["open", "close"]].min(axis=1)) |
                (frame["high"] < frame[["open", "close"]].max(axis=1)) |
                (frame["low"] > frame["high"]))
    duplicate_bad = frame.duplicated(["code", "date", "time"], keep=False)
    negative_bad = frame[NUMERIC_COLUMNS].lt(0).any(axis=1)
    clocks = frame["time"].dt.strftime("%H:%M:%S")
    invalid_time_bad = frame["time"].isna() | ~clocks.isin(EXPECTED_TIME_SET)
    return {
        "assertion_ohlc": label_rows(frame[ohlc_bad], "DN-001", "invalid_ohlc_relation"),
        "assertion_duplicates": label_rows(frame[duplicate_bad], "DN-001", "duplicate_key"),
        "assertion_nonnegative": label_rows(frame[negative_bad], "DN-001", "negative_value"),
        "assertion_invalid_time": label_rows(frame[invalid_time_bad], "DN-001", "invalid_bar_end_time"),
    }


def label_rows(frame: pd.DataFrame, note: str, issue: str) -> pd.DataFrame:
    """Add traceable issue metadata to detail rows."""
    result = frame.copy()
    result["data_note"] = note
    result["issue"] = issue
    return result


def collect_day_counts(code: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Count bars by stock-date for cross-sectional market-day inference."""
    counts = frame.groupby("date", as_index=False).size().rename(columns={"size": "bars"})
    counts["code"] = code
    return counts


def build_market_profile(day_counts: pd.DataFrame, stock_count: int) -> pd.DataFrame:
    """Identify normal versus possible special market dates (DN-001)."""
    if day_counts.empty:
        return pd.DataFrame(columns=["date", "market_mode_bars", "observed_stocks", "is_normal_market_day"])
    grouped = day_counts.groupby("date")
    profile = grouped["bars"].agg(market_mode_bars=lambda x: int(x.mode().max()), observed_stocks="size").reset_index()
    ratio = profile["observed_stocks"] / max(stock_count, 1)
    supported_special = (profile["market_mode_bars"] != EXPECTED_BARS_PER_DAY) & (ratio >= SPECIAL_DAY_MIN_COVERAGE_RATIO)
    profile["is_normal_market_day"] = ~supported_special
    return profile


def effective_start(minutes: pd.DataFrame, daily: pd.DataFrame) -> pd.Timestamp | None:
    """Use the later of reference start and first available minute date (DN-002)."""
    if minutes.empty or daily.empty:
        return None
    return max(minutes["date"].min(), daily["date"].min())


def completeness_table(code: str, minutes: pd.DataFrame, daily: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Classify suspension, coverage gap, whole-day and intraday missingness (DN-010)."""
    counts = minutes.groupby("date").size().rename("bars")
    activity = minutes.groupby("date")[["volume", "amount"]].sum().sum(axis=1)
    table = daily[["date", "tradestatus"]].drop_duplicates("date").copy()
    table["bars"] = table["date"].map(counts).fillna(0).astype(int)
    table["zero_activity_bars"] = table["date"].map(activity).fillna(0).eq(0)
    table = table.merge(market, on="date", how="left")
    table["is_normal_market_day"] = table["is_normal_market_day"].map(
        lambda value: True if pd.isna(value) else bool(value)
    ).astype(bool)
    table["market_mode_bars"] = table["market_mode_bars"].fillna(EXPECTED_BARS_PER_DAY).astype(int)
    start = effective_start(minutes, daily)
    table["classification"] = table.apply(lambda row: classify_day(row, start), axis=1)
    table["code"] = code
    missing_map = missing_clock_map(minutes)
    table["missing_times"] = table["date"].map(missing_map).fillna("")
    table["data_note"] = "DN-006"
    return table


def classify_day(row: pd.Series, start: pd.Timestamp | None) -> str:
    """Apply the documented daily completeness decision tree."""
    bars, status = int(row["bars"]), int(row["tradestatus"])
    if status == 0:
        if bars == 0:
            return "suspended"
        return "suspended_placeholder_bars" if bool(row["zero_activity_bars"]) else \
            "suspension_status_conflict"
    if start is None or row["date"] < start:
        return "coverage_gap"
    if not bool(row["is_normal_market_day"]):
        return "special_market_day"
    if bars == EXPECTED_BARS_PER_DAY:
        return "complete"
    if bars == 0:
        return "whole_day_missing"
    if bars < EXPECTED_BARS_PER_DAY:
        return "intraday_missing"
    return "extra_bars"


def missing_clock_map(minutes: pd.DataFrame) -> dict[pd.Timestamp, str]:
    """List missing canonical clocks for every observed incomplete day."""
    clock_frame = minutes[["date"]].copy()
    clock_frame["clock"] = minutes["time"].dt.strftime("%H:%M:%S")
    stats = clock_frame.groupby("date")["clock"].agg(["size", "nunique"])
    candidates = stats[(stats["size"] != EXPECTED_BARS_PER_DAY) |
                       (stats["nunique"] != EXPECTED_BARS_PER_DAY)].index
    result: dict[pd.Timestamp, str] = {}
    for day in candidates:
        observed = set(clock_frame.loc[clock_frame["date"].eq(day), "clock"])
        result[day] = ";".join(sorted(EXPECTED_TIME_SET - observed))
    return result


def bar_count_issues(table: pd.DataFrame) -> pd.DataFrame:
    """Hard-flag only normal trading days after effective coverage (DN-001)."""
    bad_classes = {"whole_day_missing", "intraday_missing", "extra_bars"}
    result = table[table["classification"].isin(bad_classes)].copy()
    result["issue"] = "normal_trading_day_not_48_bars"
    result["data_note"] = "DN-001"
    return result


def aggregate_minutes(minutes: pd.DataFrame) -> pd.DataFrame:
    """Aggregate canonical daily values from minute bars."""
    return minutes.sort_values("time").groupby("date", as_index=False).agg(
        minute_volume=("volume", "sum"), minute_amount=("amount", "sum"),
        minute_high=("high", "max"), minute_low=("low", "min"),
        minute_close=("close", "last"), bars=("time", "size"),
    )


def reconciliation_table(code: str, minutes: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Compare minute aggregates with direct unadjusted daily data (DN-003/004)."""
    reference = daily.rename(columns={column: f"daily_{column}" for column in NUMERIC_COLUMNS})
    table = aggregate_minutes(minutes).merge(reference, on="date", how="inner")
    table = table[table["tradestatus"].eq(1)].copy()
    table["code"] = code
    table["volume_error"] = table["minute_volume"] - table["daily_volume"]
    table["amount_error"] = table["minute_amount"] - table["daily_amount"]
    table["volume_relative_error"] = safe_relative(table["volume_error"], table["daily_volume"])
    table["amount_relative_error"] = safe_relative(table["amount_error"], table["daily_amount"])
    for field in ["close", "high", "low"]:
        table[f"{field}_absolute_error"] = (table[f"minute_{field}"] - table[f"daily_{field}"]).abs()
    return table


def safe_relative(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Compute relative errors without infinite values."""
    return numerator.div(denominator.replace(0, np.nan))


def strict_reconciliation_issues(table: pd.DataFrame) -> pd.DataFrame:
    """Keep strict volume, amount, and final-close mismatches (DN-003/007)."""
    volume_bad = table["volume_error"].abs() > VOLUME_ABSOLUTE_TOLERANCE
    amount_bad = table["amount_error"].abs() > AMOUNT_ABSOLUTE_TOLERANCE
    close_bad = table["close_absolute_error"] >= PRICE_TOLERANCE
    rows: list[pd.DataFrame] = []
    for mask, issue in [(volume_bad, "volume_mismatch"), (amount_bad, "amount_mismatch"),
                        (close_bad, "close_mismatch")]:
        selected = table[mask].copy()
        selected["issue"] = issue
        rows.append(selected)
    result = pd.concat(rows, ignore_index=True) if rows else table.iloc[0:0].copy()
    result["data_note"] = "DN-003/DN-007"
    return result


def high_low_hypothesis(code: str, table: pd.DataFrame) -> pd.DataFrame:
    """Test whether daily extremes equal the opening-auction price (DN-004)."""
    ordered = table.sort_values("date").copy()
    ordered["previous_daily_close"] = ordered["daily_close"].shift(1)
    ordered["opening_gap_pct"] = safe_relative(
        ordered["daily_open"] - ordered["previous_daily_close"], ordered["previous_daily_close"]
    )
    rows: list[pd.DataFrame] = []
    for field in ["high", "low"]:
        selected = ordered[ordered[f"{field}_absolute_error"] >= PRICE_TOLERANCE].copy()
        selected["extreme"] = field
        selected["absolute_error"] = selected[f"{field}_absolute_error"]
        selected["extreme_near_open"] = (
            selected[f"daily_{field}"] - selected["daily_open"]
        ).abs() < HIGH_LOW_OPEN_MATCH_TOLERANCE
        rows.append(selected[["date", "opening_gap_pct", "extreme", "absolute_error",
                              "extreme_near_open", f"daily_{field}", f"minute_{field}", "daily_open"]])
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    result["code"] = code
    result["data_note"] = "DN-004"
    return result


def zero_volume_table(code: str, minutes: pd.DataFrame) -> pd.DataFrame:
    """Summarize zero-volume ratios by stock and bar end time."""
    frame = minutes.assign(bar_time=minutes["time"].dt.strftime("%H:%M:%S"), zero=minutes["volume"].eq(0))
    result = frame.groupby("bar_time", as_index=False).agg(zero_volume_bars=("zero", "sum"), bars=("zero", "size"))
    result["zero_volume_ratio"] = result["zero_volume_bars"] / result["bars"]
    result["code"] = code
    return result


def summarize_stock(code: str, minutes: pd.DataFrame, assertions: dict[str, pd.DataFrame], completeness: pd.DataFrame, reconciliation: pd.DataFrame, strict: pd.DataFrame, hypothesis: pd.DataFrame) -> dict[str, Any]:
    """Build the required one-row-per-stock QC summary."""
    classes = completeness["classification"].value_counts()
    traded = completeness[completeness["tradestatus"].eq(1) & ~completeness["classification"].eq("coverage_gap")]
    missing = classes.get("whole_day_missing", 0) + classes.get("intraday_missing", 0)
    bar_issues = missing + classes.get("extra_bars", 0)
    assertion_count = sum(len(value) for value in assertions.values()) + bar_issues
    volume_abs = reconciliation["volume_relative_error"].abs()
    amount_abs = reconciliation["amount_relative_error"].abs()
    explanation = float(hypothesis["extreme_near_open"].mean()) if not hypothesis.empty else np.nan
    strict_counts = strict["issue"].value_counts() if not strict.empty else pd.Series(dtype=int)
    return {
        "code": code, "total_rows": len(minutes), "first_date": boundary(minutes, "min"),
        "last_date": boundary(minutes, "max"), "assertion_anomalies": assertion_count,
        "ohlc_anomalies": len(assertions["assertion_ohlc"]),
        "duplicate_rows": len(assertions["assertion_duplicates"]),
        "negative_rows": len(assertions["assertion_nonnegative"]),
        "invalid_time_rows": len(assertions["assertion_invalid_time"]),
        "normal_day_bar_count_anomalies": bar_issues, "trading_days_expected": len(traded),
        "complete_days": int(classes.get("complete", 0)),
        "intraday_missing_days": int(classes.get("intraday_missing", 0)),
        "whole_day_missing_days": int(classes.get("whole_day_missing", 0)),
        "suspended_days": int(classes.get("suspended", 0) +
                              classes.get("suspended_placeholder_bars", 0)),
        "missing_bar_day_ratio": missing / len(traded) if len(traded) else np.nan,
        "zero_volume_ratio": float(minutes["volume"].eq(0).mean()) if len(minutes) else np.nan,
        "volume_error_median": reconciliation["volume_relative_error"].median(),
        "volume_error_p95_abs": volume_abs.quantile(0.95),
        "amount_error_median": reconciliation["amount_relative_error"].median(),
        "amount_error_p95_abs": amount_abs.quantile(0.95),
        "close_error_max": reconciliation["close_absolute_error"].max(),
        "strict_reconciliation_issues": len(strict),
        "volume_mismatch_days": int(strict_counts.get("volume_mismatch", 0)),
        "amount_mismatch_days": int(strict_counts.get("amount_mismatch", 0)),
        "close_mismatch_days": int(strict_counts.get("close_mismatch", 0)),
        "volume_rescaled_days": flagged_days(minutes, "volume_rescaled"),
        "close_anomaly_flag_days": flagged_days(minutes, "close_anomaly_flag"),
        "total_anomaly_records": assertion_count + len(strict),
        "high_error_median": reconciliation["high_absolute_error"].median(),
        "low_error_median": reconciliation["low_absolute_error"].median(),
        "high_low_open_explanation_ratio": explanation,
        "high_low_hypothesis_supported": bool(explanation >= HIGH_LOW_EXPLANATION_RATIO) if not np.isnan(explanation) else False,
    }


def flagged_days(frame: pd.DataFrame, column: str) -> int:
    """Count flagged stock-days when processing metadata is present (DN-009)."""
    if column not in frame.columns:
        return 0
    return int(frame.loc[frame[column].fillna(False).astype(bool), "date"].nunique())


def boundary(frame: pd.DataFrame, operation: str) -> str | None:
    """Return a display-safe date boundary."""
    if frame.empty:
        return None
    value = frame["date"].min() if operation == "min" else frame["date"].max()
    return value.date().isoformat()


def add_detail(store: dict[str, list[pd.DataFrame]], name: str, frame: pd.DataFrame) -> None:
    """Collect non-empty detail frames for final concatenation."""
    if not frame.empty or not store[name]:
        store[name].append(frame)


def process_stock(code: str, minutes: pd.DataFrame, daily: pd.DataFrame, market: pd.DataFrame, store: dict[str, list[pd.DataFrame]]) -> dict[str, Any]:
    """Run all three QC layers for one stock."""
    assertions = assertion_frames(minutes)
    completeness = completeness_table(code, minutes, daily, market)
    count_issues = bar_count_issues(completeness)
    reconciliation = reconciliation_table(code, minutes, daily)
    strict = strict_reconciliation_issues(reconciliation)
    hypothesis = high_low_hypothesis(code, reconciliation)
    zero = zero_volume_table(code, minutes)
    for name, frame in assertions.items():
        add_detail(store, name, frame)
    for name, frame in [("bar_count_issues", count_issues), ("completeness_daily", completeness),
                        ("coverage_gaps", completeness[completeness["classification"].eq("coverage_gap")]),
                        ("zero_volume_by_stock_time", zero), ("reconciliation_daily", reconciliation),
                        ("strict_reconciliation_issues", strict), ("high_low_open_hypothesis", hypothesis)]:
        add_detail(store, name, frame)
    return summarize_stock(code, minutes, assertions, completeness, reconciliation, strict, hypothesis)


def combine_details(store: dict[str, list[pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    """Combine collected issue tables while preserving empty outputs."""
    return {name: pd.concat(store[name], ignore_index=True) if store[name] else pd.DataFrame()
            for name in DETAIL_NAMES}


def write_csv_reports(summary: pd.DataFrame, details: dict[str, pd.DataFrame], output: Path) -> None:
    """Write summary and every issue class as CSV."""
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "qc_summary.csv", index=False, encoding="utf-8")
    for name, frame in details.items():
        frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8")


def empty_plot(message: str) -> None:
    """Render a readable placeholder when a diagnostic has no observations."""
    plt.text(0.5, 0.5, message, ha="center", va="center", transform=plt.gca().transAxes)
    plt.xticks([])
    plt.yticks([])


def save_plot(path: Path, title: str) -> None:
    """Apply common layout and save one diagnostic image."""
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()


def create_diagnostic_plots(details: dict[str, pd.DataFrame], output: Path) -> None:
    """Create five compact diagnostics requested by the QC specification."""
    plot_bar_counts(details["completeness_daily"], output / "01_bar_count_distribution.png")
    plot_missing_month(details["completeness_daily"], output / "02_missing_dates_by_month.png")
    plot_volume_errors(details["reconciliation_daily"], output / "03_volume_error_histogram.png")
    plot_high_low(details["reconciliation_daily"], output / "04_high_low_error_histogram.png")
    plot_opening_hypothesis(details["high_low_open_hypothesis"], output / "05_opening_gap_vs_extreme_error.png")


def plot_bar_counts(frame: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    if frame.empty:
        empty_plot("No completeness observations")
    else:
        frame["bars"].value_counts().sort_index().plot(kind="bar", color="#2563EB")
        plt.xlabel("5-minute bars per reference day")
        plt.ylabel("Stock-days")
    save_plot(path, "Bar-count distribution")


def plot_missing_month(frame: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(10, 4.5))
    issues = frame[frame["classification"].isin(["whole_day_missing", "intraday_missing"])] if not frame.empty else frame
    if issues.empty:
        empty_plot("No missing-bar dates")
    else:
        issues = issues.assign(month=pd.to_datetime(issues["date"]).dt.to_period("M").astype(str))
        pivot = issues.pivot_table(index="month", columns="classification", values="code", aggfunc="count", fill_value=0)
        pivot.plot(kind="bar", stacked=True, ax=plt.gca(), color=["#F59E0B", "#DC2626"])
        plt.xlabel("Month")
        plt.ylabel("Stock-days")
    save_plot(path, "Missing-bar dates over time")


def plot_volume_errors(frame: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    values = frame["volume_relative_error"].dropna() if not frame.empty else pd.Series(dtype=float)
    if values.empty:
        empty_plot("No reconciliation observations")
    else:
        plt.hist(values, bins=PLOT_BINS, color="#0F766E", alpha=0.85)
        plt.xlabel("(minute volume - daily volume) / daily volume")
        plt.ylabel("Trading days")
    save_plot(path, "Daily volume reconciliation errors")


def plot_high_low(frame: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    if frame.empty:
        empty_plot("No high/low observations")
    else:
        plt.hist(frame["high_absolute_error"].dropna(), bins=PLOT_BINS, alpha=0.65, label="High")
        plt.hist(frame["low_absolute_error"].dropna(), bins=PLOT_BINS, alpha=0.65, label="Low")
        plt.xlabel("Absolute price error")
        plt.ylabel("Trading days")
        plt.legend()
    save_plot(path, "Daily versus 5-minute high/low errors")


def plot_opening_hypothesis(frame: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    usable = frame.dropna(subset=["opening_gap_pct", "absolute_error"]) if not frame.empty else frame
    if usable.empty:
        empty_plot("No high/low mismatch observations")
    else:
        for label, group in usable.groupby("extreme"):
            plt.scatter(group["opening_gap_pct"].abs(), group["absolute_error"], s=12, alpha=0.5, label=label)
        plt.xlabel("Absolute opening gap")
        plt.ylabel("High/low absolute error")
        plt.legend()
    save_plot(path, "Opening gap versus high/low discrepancy")


def run_quality_check(codes: list[str], raw_dir: Path, reference_dir: Path, output: Path) -> pd.DataFrame:
    """Run the full QC workflow and write reports."""
    available = [code for code in codes if (raw_dir / f"{code}.parquet").exists()]
    count_frames = [collect_day_counts(code, pd.read_parquet(
        raw_dir / f"{code}.parquet", columns=["date"])) for code in available]
    counts = pd.concat(count_frames, ignore_index=True) if count_frames else pd.DataFrame()
    market = build_market_profile(counts, len(available))
    store: dict[str, list[pd.DataFrame]] = defaultdict(list)
    summaries: list[dict[str, Any]] = []
    for code in available:
        daily_path = reference_dir / f"{code}.parquet"
        if not daily_path.exists():
            continue
        minutes = load_minutes(raw_dir / f"{code}.parquet")
        summaries.append(process_stock(code, minutes, load_daily(daily_path), market, store))
        print(f"qc {code} rows={len(minutes)}", flush=True)
    summary = pd.DataFrame(summaries).sort_values("code") if summaries else pd.DataFrame()
    details = combine_details(store)
    write_csv_reports(summary, details, output)
    create_diagnostic_plots(details, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stock-list", type=Path, default=STOCK_LIST_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_5MIN_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DAILY_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=QC_REPORT_DIR)
    args = parser.parse_args()
    summary = run_quality_check(load_codes(args.stock_list, args.limit), args.raw_dir, args.reference_dir, args.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
