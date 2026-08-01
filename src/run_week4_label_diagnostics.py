"""Create the pre-registered information/attention labels before any 2D returns.

This script deliberately stops at label diagnostics. It never reads CTO returns
or creates sorted portfolio performance, preserving the Week-4 pre-registration
firewall.

Inputs: complete announcement partitions and post-adjusted/raw daily price data.
Outputs: event-coverage, label-share, and label-count diagnostics.
Role: enforces the pre-return feasibility check before the Week-4 sort.
"""
from __future__ import annotations

import json
import gc
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cto_pipeline import as_bool, limit_pct


BASE = Path("data/cto_baostock")
DISCLOSURES = BASE / "disclosures"
OUT = BASE / "formal_backtest" / "week4"
HFQ = BASE / "daily_hfq"
RAW = BASE / "daily_raw"
REQUIRED_TYPES = {"earnings_forecast", "periodic_report", "earnings_flash", "dividend"}


def require_complete_download() -> dict:
    manifest_path = DISCLOSURES / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = set(manifest["completed"])
    expected = {f"{kind}/{year}{month_day}" for kind in REQUIRED_TYPES for year in range(2010, 2027) for month_day in ("0331", "0630", "0930", "1231") if not (year == 2026 and month_day in {"0930", "1231"})}
    missing = expected - completed
    if missing:
        raise RuntimeError(f"announcement download incomplete: {len(missing)} partitions missing; retry before labelling")
    return manifest


def write_gap_audit(manifest: dict) -> None:
    """Document confirmed-zero partitions separately from unresolved gaps."""
    rows = []
    for key, detail in manifest["completed"].items():
        if detail.get("rows", 0) == 0:
            event_type, report_date = key.split("/")
            rows.append({"event_type": event_type, "report_date": report_date,
                         "status": "confirmed_zero_events", "event_rows": 0,
                         "potentially_affected_stock_months": 0,
                         "note": "provider returned an explicit empty period"})
    for key, error in manifest["failures"].items():
        event_type, report_date = key.split("/")
        rows.append({"event_type": event_type, "report_date": report_date,
                     "status": "unresolved_download_gap", "event_rows": np.nan,
                     "potentially_affected_stock_months": np.nan, "note": error})
    columns = ["event_type", "report_date", "status", "event_rows", "potentially_affected_stock_months", "note"]
    pd.DataFrame(rows, columns=columns).to_csv(OUT / "announcement_gap_audit.csv", index=False)


def load_events() -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    frames = []
    for event_type in REQUIRED_TYPES:
        for path in sorted((DISCLOSURES / event_type).glob("*.csv.gz")):
            x = pd.read_csv(path, usecols=["stock_code", "event_type", "event_date", "event_role"])
            frames.append(x)
    events = pd.concat(frames, ignore_index=True)
    events["stock_code"] = events["stock_code"].astype(str).str.zfill(6)
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce").dt.normalize()
    coverage = (events.groupby(["event_type", "event_role"], dropna=False)
                .agg(event_rows=("stock_code", "size"), missing_event_date=("event_date", lambda x: x.isna().sum()),
                     first_event_date=("event_date", "min"), last_event_date=("event_date", "max"))
                .reset_index())
    events = events.dropna(subset=["event_date"]).drop_duplicates(["stock_code", "event_type", "event_role", "event_date"])
    by_stock = {code: group["event_date"].to_numpy(dtype="datetime64[ns]") for code, group in events.groupby("stock_code")}
    return by_stock, coverage


def market_medians(start_year: int, end_year: int) -> pd.Series:
    """Exact daily median adjusted return, using a disk-backed float32 matrix."""
    paths = sorted(HFQ.glob("*.csv"))
    calendar = pd.to_datetime(pd.read_csv(HFQ / "600519.csv", usecols=["日期"])["日期"])
    dates = pd.Index(calendar.drop_duplicates().sort_values())
    dates = dates[(dates.year >= start_year) & (dates.year <= end_year)]
    location = pd.Series(np.arange(len(dates)), index=dates)
    scratch = OUT / "market_returns_float32.memmap"
    values = np.memmap(scratch, mode="w+", dtype=np.float32, shape=(len(paths), len(dates)))
    values[:] = np.nan
    for row, path in enumerate(paths):
        d = pd.read_csv(path, usecols=["日期", "收盘", "前收盘"])
        d["date"] = pd.to_datetime(d["日期"])
        close = pd.to_numeric(d["收盘"], errors="coerce")
        previous = pd.to_numeric(d["前收盘"], errors="coerce")
        idx = location.reindex(d["date"]).to_numpy()
        valid = ~pd.isna(idx) & previous.gt(0).to_numpy()
        values[row, idx[valid].astype(int)] = (close.to_numpy()[valid] / previous.to_numpy()[valid] - 1).astype(np.float32)
        if (row + 1) % 500 == 0:
            print(f"market-median pass: {row + 1}/{len(paths)} stocks", flush=True)
    # Median is calculated in narrow date blocks so the temporary working
    # array stays a few MB even on memory-constrained machines.
    medians = np.empty(len(dates), dtype=np.float32)
    for start in range(0, len(dates), 64):
        end = min(start + 64, len(dates))
        medians[start:end] = np.nanmedian(values[:, start:end], axis=0)
    print("market-median pass complete", flush=True)
    values.flush()
    del values
    scratch.unlink(missing_ok=True)
    return pd.Series(medians, index=dates, name="market_median_return")


def build_medians_stage(start_year: int, end_year: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    medians = market_medians(start_year, end_year)
    target = OUT / "market_median_returns.csv"
    new = medians.rename_axis("date").reset_index()
    if target.exists():
        old = pd.read_csv(target, parse_dates=["date"])
        new = pd.concat([old, new], ignore_index=True).drop_duplicates("date", keep="last")
    new.sort_values("date").to_csv(target, index=False)
    print(f"wrote market medians for {start_year}-{end_year}: {len(medians)} dates", flush=True)


def median_scratch(start_year: int, end_year: int) -> tuple[Path, pd.Index, list[Path], pd.Series]:
    paths = sorted(HFQ.glob("*.csv"))
    calendar = pd.to_datetime(pd.read_csv(HFQ / "600519.csv", usecols=["日期"])["日期"])
    dates = pd.Index(calendar.drop_duplicates().sort_values())
    dates = dates[(dates.year >= start_year) & (dates.year <= end_year)]
    return OUT / f"market_returns_{start_year}_{end_year}.memmap", dates, paths, pd.Series(np.arange(len(dates)), index=dates)


def fill_medians_stage(start_year: int, end_year: int, start_index: int, end_index: int | None) -> None:
    """Fill a bounded row range, keeping each invocation below the runner limit."""
    OUT.mkdir(parents=True, exist_ok=True)
    scratch, dates, paths, location = median_scratch(start_year, end_year)
    shape = (len(paths), len(dates))
    if scratch.exists():
        values = np.memmap(scratch, mode="r+", dtype=np.float32, shape=shape)
    else:
        values = np.memmap(scratch, mode="w+", dtype=np.float32, shape=shape)
        values[:] = np.nan
    stop = min(end_index or len(paths), len(paths))
    for row in range(start_index, stop):
        d = pd.read_csv(paths[row], usecols=["日期", "收盘", "前收盘"])
        d["date"] = pd.to_datetime(d["日期"])
        close = pd.to_numeric(d["收盘"], errors="coerce")
        previous = pd.to_numeric(d["前收盘"], errors="coerce")
        idx = location.reindex(d["date"]).to_numpy()
        valid = ~pd.isna(idx) & previous.gt(0).to_numpy()
        values[row, idx[valid].astype(int)] = (close.to_numpy()[valid] / previous.to_numpy()[valid] - 1).astype(np.float32)
        if (row + 1) % 500 == 0:
            print(f"median fill {start_year}-{end_year}: {row + 1}/{len(paths)}", flush=True)
    values.flush(); del values
    print(f"median fill complete {start_year}-{end_year}: rows {start_index}:{stop}", flush=True)


def finalize_medians_stage(start_year: int, end_year: int) -> None:
    scratch, dates, paths, _ = median_scratch(start_year, end_year)
    values = np.memmap(scratch, mode="r", dtype=np.float32, shape=(len(paths), len(dates)))
    medians = np.empty(len(dates), dtype=np.float32)
    for start in range(0, len(dates), 64):
        end = min(start + 64, len(dates))
        medians[start:end] = np.nanmedian(values[:, start:end], axis=0)
    del values
    target = OUT / "market_median_returns.csv"
    new = pd.DataFrame({"date": dates, "market_median_return": medians})
    if target.exists():
        old = pd.read_csv(target)
        if "date" not in old and "日期" in old:
            old = old.rename(columns={"日期": "date"})
        old["date"] = pd.to_datetime(old["date"])
        new = pd.concat([old, new], ignore_index=True).drop_duplicates("date", keep="last")
    new.sort_values("date").to_csv(target, index=False)
    scratch.unlink(missing_ok=True)
    print(f"finalized market medians {start_year}-{end_year}: {len(dates)} dates", flush=True)


def event_window(dates: pd.Series, event_dates: np.ndarray | None) -> np.ndarray:
    result = np.zeros(len(dates), dtype=bool)
    if event_dates is None or len(event_dates) == 0:
        return result
    array = dates.to_numpy(dtype="datetime64[ns]")
    for event in event_dates:
        pos = array.searchsorted(event)
        if pos == len(array):
            continue
        result[max(0, pos - 1): min(len(array), pos + 2)] = True
    return result


def touches_up_limit(raw: pd.DataFrame, code: str) -> pd.DataFrame:
    dates = pd.to_datetime(raw["日期"])
    high = pd.to_numeric(raw["最高"], errors="coerce")
    prev = pd.to_numeric(raw["前收盘"], errors="coerce")
    st = raw["是否ST"].map(as_bool)
    pct = np.array([limit_pct(code, d, flag) for d, flag in zip(dates, st)])
    up = (prev.to_numpy() * (1 + pct)).round(2)
    return pd.DataFrame({"date": dates, "touches_up_limit": np.abs(high.to_numpy() - up) <= 0.005})


def classify_stock(path: Path, medians: pd.Series, events: dict[str, np.ndarray]) -> pd.DataFrame:
    code = path.stem
    hfq = pd.read_csv(path, usecols=["日期", "收盘", "前收盘", "换手率"])
    raw = pd.read_csv(RAW / path.name, usecols=["日期", "最高", "前收盘", "是否ST"])
    hfq["date"] = pd.to_datetime(hfq["日期"])
    hfq["close"] = pd.to_numeric(hfq["收盘"], errors="coerce")
    hfq["prev_close"] = pd.to_numeric(hfq["前收盘"], errors="coerce")
    hfq["turnover"] = pd.to_numeric(hfq["换手率"], errors="coerce")
    hfq["daily_return"] = hfq["close"] / hfq["prev_close"] - 1
    hfq["market_adjusted_return"] = hfq["daily_return"] - hfq["date"].map(medians)
    hfq["p975_prior250"] = hfq["market_adjusted_return"].shift(1).rolling(250, min_periods=250).quantile(.975)
    hfq["turnover_mean_prior60"] = hfq["turnover"].shift(1).rolling(60, min_periods=60).mean()
    # Delisted names can have different raw/HFQ row counts.  Align the
    # tradability flag by calendar date, never by incidental row position.
    raw_limits = touches_up_limit(raw, code)
    hfq = hfq.merge(raw_limits, on="date", how="left")
    hfq["touches_up_limit"] = hfq["touches_up_limit"].fillna(False)
    hfq["announcement_window"] = event_window(hfq["date"], events.get(code))
    hfq["extreme_day"] = hfq["market_adjusted_return"].gt(hfq["p975_prior250"]) & ~hfq["touches_up_limit"]
    hfq["information_extreme"] = hfq["extreme_day"] & hfq["announcement_window"]
    hfq["attention_extreme"] = (hfq["extreme_day"] & ~hfq["announcement_window"] &
                                 hfq["turnover"].gt(2 * hfq["turnover_mean_prior60"]))
    hfq["unclassified_extreme"] = hfq["extreme_day"] & ~hfq["information_extreme"] & ~hfq["attention_extreme"]
    x = hfq[hfq["extreme_day"]].copy()
    if x.empty:
        return x
    x["code"] = code
    x["month"] = x["date"].dt.to_period("M")
    return x[["code", "date", "month", "market_adjusted_return", "p975_prior250", "turnover", "turnover_mean_prior60", "touches_up_limit", "announcement_window", "information_extreme", "attention_extreme", "unclassified_extreme"]]


def label_stage(start_index: int, end_index: int | None, reset: bool = False) -> None:
    manifest = require_complete_download()
    OUT.mkdir(parents=True, exist_ok=True)
    write_gap_audit(manifest)
    events, coverage = load_events()
    coverage.to_csv(OUT / "announcement_coverage.csv", index=False)
    median_path = OUT / "market_median_returns.csv"
    medians = pd.read_csv(median_path, parse_dates=["date"]).set_index("date")["market_median_return"]
    temp_labels = OUT / "extreme_day_labels.tmp.csv"
    if reset:
        temp_labels.unlink(missing_ok=True)
    paths = sorted(HFQ.glob("*.csv"))[start_index:end_index]
    first = not temp_labels.exists()
    for n, path in enumerate(paths, start_index + 1):
        labels = classify_stock(path, medians, events)
        if not labels.empty:
            labels.to_csv(temp_labels, mode="w" if first else "a", header=first, index=False)
            first = False
        del labels
        if n % 100 == 0:
            gc.collect()
        if n % 500 == 0:
            print(f"labelled {n} stocks", flush=True)
    print(f"label stage complete: {start_index}:{end_index or 'end'}", flush=True)


def finalize_stage() -> None:
    temp_labels = OUT / "extreme_day_labels.tmp.csv"
    source = temp_labels if temp_labels.exists() else OUT / "extreme_day_labels.csv.gz"
    if not source.exists():
        raise RuntimeError("no staged extreme-day file found")
    days = pd.read_csv(source, parse_dates=["date"], dtype={"code": str})
    days["code"] = days["code"].str.zfill(6)
    if temp_labels.exists():
        temp_labels.unlink(missing_ok=True)
    days.to_csv(OUT / "extreme_day_labels.csv.gz", index=False, compression="gzip")
    labels = np.select([days["attention_extreme"], days["information_extreme"]], ["attention", "information"], default="unclassified")
    days["extreme_label"] = labels
    summary = (days.groupby("extreme_label", as_index=False)
               .agg(extreme_days=("code", "size"), stocks=("code", "nunique"), first_date=("date", "min"), last_date=("date", "max")))
    summary["share_of_extreme_days"] = summary["extreme_days"] / summary["extreme_days"].sum()
    summary.to_csv(OUT / "extreme_label_distribution.csv", index=False)
    monthly = days.groupby(["code", "month"], as_index=False).agg(
        attention_days=("attention_extreme", "sum"), information_days=("information_extreme", "sum"), unclassified_days=("unclassified_extreme", "sum"))
    monthly["month"] = pd.PeriodIndex(monthly["month"].astype(str), freq="M")
    monthly["stock_label"] = np.select(
        [monthly["attention_days"].gt(0), monthly["information_days"].gt(0)], ["attention", "information"], default="baseline")
    monthly.to_csv(OUT / "monthly_stock_labels_nonbaseline.csv", index=False)
    # Use the existing eligible CTO stock-month universe only for feasibility
    # counts; no future return is read at this stage.
    universe = pd.read_csv(BASE / "monthly_cto.csv", usecols=["code", "month"], dtype={"code": str})
    universe["code"] = universe["code"].str.zfill(6)
    universe["month"] = pd.PeriodIndex(universe["month"].astype(str), freq="M")
    stock_month = universe.merge(monthly, on=["code", "month"], how="left")
    for column in ("attention_days", "information_days", "unclassified_days"):
        stock_month[column] = stock_month[column].fillna(0).astype(int)
    stock_month["stock_label"] = stock_month["stock_label"].fillna("baseline")
    stock_month.to_csv(OUT / "monthly_stock_labels.csv", index=False)
    stock_summary = (stock_month.groupby("stock_label", as_index=False)
                     .agg(stock_months=("code", "size"), stocks=("code", "nunique"),
                          first_month=("month", "min"), last_month=("month", "max")))
    stock_summary["share_of_stock_months"] = stock_summary["stock_months"] / stock_summary["stock_months"].sum()
    stock_summary.to_csv(OUT / "stock_month_label_distribution.csv", index=False)
    stock_month["year"] = stock_month["month"].dt.year
    yearly = stock_month.groupby(["year", "stock_label"]).size().rename("stock_months").reset_index()
    yearly["share"] = yearly["stock_months"] / yearly.groupby("year")["stock_months"].transform("sum")
    yearly.to_csv(OUT / "annual_stock_month_label_shares.csv", index=False)
    chart = yearly.pivot(index="year", columns="stock_label", values="share").fillna(0)
    chart = chart.reindex(columns=["attention", "information", "baseline"], fill_value=0)
    ax = chart.plot(kind="area", stacked=True, figsize=(12, 5), color=["#d95f02", "#1b9e77", "#bdbdbd"])
    ax.set_ylim(0, 1); ax.set_ylabel("Share of eligible CTO stock-months")
    ax.set_title("Pre-return diagnostic: annual information / attention / baseline label shares")
    ax.grid(axis="y", alpha=.25); ax.legend(title="stock label", loc="upper left")
    ax.figure.tight_layout(); ax.figure.savefig(OUT / "annual_stock_month_label_shares.png", dpi=180); plt.close(ax.figure)
    print(summary.to_string(index=False))
    print(stock_summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["medians", "median-fill", "median-finalize", "labels", "finalize"], required=True)
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--reset-labels", action="store_true")
    args = parser.parse_args()
    if args.stage == "medians":
        if args.start_year is None or args.end_year is None:
            parser.error("medians requires --start-year and --end-year")
        build_medians_stage(args.start_year, args.end_year)
    elif args.stage == "median-fill":
        if args.start_year is None or args.end_year is None:
            parser.error("median-fill requires --start-year and --end-year")
        fill_medians_stage(args.start_year, args.end_year, args.start_index, args.end_index)
    elif args.stage == "median-finalize":
        if args.start_year is None or args.end_year is None:
            parser.error("median-finalize requires --start-year and --end-year")
        finalize_medians_stage(args.start_year, args.end_year)
    elif args.stage == "labels":
        label_stage(args.start_index, args.end_index, args.reset_labels)
    else:
        finalize_stage()
