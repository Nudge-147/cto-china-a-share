"""Probe Baostock 5-minute formats, coverage, and daily reconciliation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import baostock as bs
import pandas as pd

from config import (
    CTO_DAILY_RAW_DIR,
    MINUTE_FIELDS,
    MINUTE_FREQUENCY,
    PRICE_MATCH_TOLERANCE,
    PROBE_COVERAGE_START_YEAR,
    PROBE_COVERAGE_WINDOW_END,
    PROBE_COVERAGE_WINDOW_START,
    PROBE_EARLIEST_CANDIDATE_END,
    PROBE_EARLIEST_CANDIDATE_START,
    PROBE_RECENT_TRADING_DAYS,
    PROBE_REPORT_DIR,
    STOCK_LIST_PATH,
    UNADJUSTED_FLAG,
)


NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
DAILY_FIELDS = "date,code,open,high,low,close,volume,amount,adjustflag,tradestatus"


def query_minutes(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch one raw Baostock minute response without altering time strings."""
    response = bs.query_history_k_data_plus(
        code,
        MINUTE_FIELDS,
        start_date=start_date,
        end_date=end_date,
        frequency=MINUTE_FREQUENCY,
        adjustflag=UNADJUSTED_FLAG,
    )
    if response.error_code != "0":
        raise RuntimeError(f"{response.error_code}: {response.error_msg}")
    rows: list[list[str]] = []
    while response.next():
        rows.append(response.get_row_data())
    return pd.DataFrame(rows, columns=response.fields)


def query_daily(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch direct unadjusted daily rows for an independent reference."""
    response = bs.query_history_k_data_plus(
        code, DAILY_FIELDS, start_date=start_date, end_date=end_date,
        frequency="d", adjustflag=UNADJUSTED_FLAG,
    )
    if response.error_code != "0":
        raise RuntimeError(f"{response.error_code}: {response.error_msg}")
    rows: list[list[str]] = []
    while response.next():
        rows.append(response.get_row_data())
    return pd.DataFrame(rows, columns=response.fields)


def scan_coverage(code: str, last_year: int) -> tuple[pd.DataFrame, dict[int, int]]:
    """Probe the same liquid trading-week window in every target year."""
    frames: list[pd.DataFrame] = []
    counts: dict[int, int] = {}
    for year in range(PROBE_COVERAGE_START_YEAR, last_year + 1):
        start = f"{year}-{PROBE_COVERAGE_WINDOW_START}"
        end = f"{year}-{PROBE_COVERAGE_WINDOW_END}"
        frame = query_minutes(code, start, end)
        counts[year] = len(frame)
        if not frame.empty:
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=MINUTE_FIELDS.split(","))
    return combined, counts


def parse_minutes(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse the observed Baostock long timestamp and numeric fields."""
    parsed = raw.copy()
    timestamp_text = parsed["time"].astype(str).str.strip()
    parsed["datetime"] = pd.to_datetime(
        timestamp_text.str[:14], format="%Y%m%d%H%M%S", errors="coerce"
    )
    parsed["date"] = pd.to_datetime(parsed["date"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        parsed[column] = pd.to_numeric(parsed[column], errors="coerce")
    return parsed


def load_probe_daily(code: str) -> pd.DataFrame:
    """Load the matching unadjusted CTO daily file."""
    path = CTO_DAILY_RAW_DIR / f"{code.split('.', 1)[1]}.csv"
    daily = pd.read_csv(path, encoding="utf-8-sig")
    daily["日期"] = pd.to_datetime(daily["日期"], errors="coerce")
    for column in ["最高", "最低", "收盘", "成交量", "成交额", "交易状态"]:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    return daily


def recent_probe_range(daily: pd.DataFrame) -> tuple[str, str]:
    """Choose a recent fully overlapping daily window."""
    traded = daily[daily["交易状态"].eq(1)].sort_values("日期")
    selected = traded.tail(PROBE_RECENT_TRADING_DAYS)
    return selected["日期"].min().date().isoformat(), selected["日期"].max().date().isoformat()


def reconcile(minutes: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate minutes and compare them with raw daily observations."""
    grouped = minutes.sort_values("datetime").groupby("date", as_index=False).agg(
        minute_volume=("volume", "sum"),
        minute_amount=("amount", "sum"),
        minute_high=("high", "max"),
        minute_low=("low", "min"),
        minute_close=("close", "last"),
        bars=("datetime", "size"),
    )
    reference = daily.rename(columns={
        "日期": "date", "成交量": "daily_volume", "成交额": "daily_amount",
        "最高": "daily_high", "最低": "daily_low", "收盘": "daily_close",
    })
    result = grouped.merge(reference, on="date", how="left")
    result["volume_relative_error"] = (
        result["minute_volume"] - result["daily_volume"]
    ) / result["daily_volume"]
    for name in ["high", "low", "close"]:
        result[f"{name}_absolute_error"] = (
            result[f"minute_{name}"] - result[f"daily_{name}"]
        ).abs()
    return result


def build_report(code: str, coverage: pd.DataFrame, recent: pd.DataFrame, matched: pd.DataFrame, coverage_counts: dict[int, int]) -> dict[str, object]:
    """Summarize evidence needed before finalizing QC assertions."""
    clocks = recent["datetime"].dt.strftime("%H:%M:%S")
    volume_error = matched["volume_relative_error"].abs()
    price_columns = [f"{name}_absolute_error" for name in ["high", "low", "close"]]
    return {
        "code": code,
        "raw_time_samples": recent["time"].head(5).tolist(),
        "raw_time_lengths": sorted(recent["time"].astype(str).str.len().unique().tolist()),
        "coverage_window_rows_by_year": coverage_counts,
        "earliest_observed_minute_date": coverage["date"].min().date().isoformat() if not coverage.empty else None,
        "recent_first_bar": clocks.min(),
        "recent_last_bar": clocks.max(),
        "recent_unique_clocks": int(clocks.nunique()),
        "bar_count_distribution": recent.groupby("date").size().value_counts().sort_index().to_dict(),
        "volume_error_abs_max": float(volume_error.max()),
        "volume_error_median": float(matched["volume_relative_error"].median()),
        "price_error_abs_max": float(matched[price_columns].max().max()),
        "prices_within_tolerance": bool((matched[price_columns] < PRICE_MATCH_TOLERANCE).all().all()),
    }


def write_outputs(raw_coverage: pd.DataFrame, raw_recent: pd.DataFrame, matched: pd.DataFrame, report: dict[str, object]) -> None:
    """Persist compact, auditable probe artifacts."""
    PROBE_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    raw_coverage.to_csv(PROBE_REPORT_DIR / "raw_coverage_windows.csv", index=False)
    raw_recent.to_csv(PROBE_REPORT_DIR / "raw_recent.csv", index=False)
    matched.to_csv(PROBE_REPORT_DIR / "daily_reconciliation.csv", index=False)
    (PROBE_REPORT_DIR / "probe_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def probe_earliest(code: str) -> None:
    """Check the start of the first year with observed minute coverage."""
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"login failed: {login.error_code} {login.error_msg}")
    try:
        raw = query_minutes(code, PROBE_EARLIEST_CANDIDATE_START, PROBE_EARLIEST_CANDIDATE_END)
    finally:
        bs.logout()
    print(f"rows={len(raw)}")
    print(raw.head(5).to_string(index=False))
    if not raw.empty:
        earliest = str(raw["date"].min())
        print(f"earliest={earliest} latest={raw['date'].max()}")
        report_path = PROBE_REPORT_DIR / "probe_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        report["earliest_confirmed_minute_date"] = earliest
        report["earliest_candidate_window_rows"] = int(len(raw))
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--earliest-only", action="store_true")
    args = parser.parse_args()
    code = pd.read_csv(STOCK_LIST_PATH, dtype=str)["code"].iloc[0]
    if args.earliest_only:
        probe_earliest(code)
        return
    daily = load_probe_daily(code)
    recent_start, recent_end = recent_probe_range(daily)
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"login failed: {login.error_code} {login.error_msg}")
    try:
        raw_coverage, coverage_counts = scan_coverage(code, int(daily["日期"].max().year))
        raw_recent = query_minutes(code, recent_start, recent_end)
        direct_daily = query_daily(code, recent_start, recent_end)
    finally:
        bs.logout()
    coverage = parse_minutes(raw_coverage)
    recent = parse_minutes(raw_recent)
    direct_daily = direct_daily.rename(columns={
        "date": "日期", "code": "代码", "open": "开盘", "high": "最高", "low": "最低",
        "close": "收盘", "volume": "成交量", "amount": "成交额",
        "adjustflag": "复权标记", "tradestatus": "交易状态",
    })
    direct_daily["日期"] = pd.to_datetime(direct_daily["日期"])
    for column in ["最高", "最低", "收盘", "成交量", "成交额", "交易状态"]:
        direct_daily[column] = pd.to_numeric(direct_daily[column], errors="coerce")
    matched = reconcile(recent, direct_daily)
    report = build_report(code, coverage, recent, matched, coverage_counts)
    report["cto_adjustflags_recent"] = sorted(daily.tail(PROBE_RECENT_TRADING_DAYS)["复权标记"].astype(str).unique().tolist())
    report["direct_daily_adjustflags"] = sorted(direct_daily["复权标记"].astype(str).unique().tolist())
    write_outputs(raw_coverage, raw_recent, matched, report)
    print("RAW RESPONSE HEAD")
    print(raw_recent.head(5).to_string(index=False))
    print("\nPROBE REPORT")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nDAILY RECONCILIATION")
    print(matched.to_string(index=False))


if __name__ == "__main__":
    main()
