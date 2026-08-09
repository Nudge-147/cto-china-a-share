"""Investigate systematic daily-versus-minute reconciliation differences."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import akshare as ak
import numpy as np
import pandas as pd

from config import (
    BLOCK_MATCH_ABSOLUTE_SHARES, BLOCK_MATCH_RELATIVE_TOLERANCE,
    BLOCK_TRADE_PAGE_LIMIT, BLOCK_TRADE_START_DATE, CLOSE_SAMPLE_SIZE,
    CROSS_VENDOR_VOLUME_TOLERANCE_SHARES, CTO_DAILY_RAW_DIR,
    EVENT_REQUEST_SLEEP_SECONDS, EVENT_RETRY_ATTEMPTS,
    FACTOR_CHANGE_TOLERANCE, FIXED_PRICE_RELATIVE_TOLERANCE,
    PRICE_TOLERANCE, RAW_5MIN_DIR, RECONCILIATION_INVESTIGATION_DIR, STOCK_LIST_PATH,
)


RECONCILIATION_PATH = RAW_5MIN_DIR.parent / "qc_report" / "reconciliation_daily.csv"
CTO_DAILY_HFQ_DIR = CTO_DAILY_RAW_DIR.parent / "daily_hfq"


def load_codes(limit: int) -> list[str]:
    """Read the tested stock prefix."""
    return pd.read_csv(STOCK_LIST_PATH, dtype=str)["code"].head(limit).tolist()


def board_name(code: pd.Series) -> pd.Series:
    """Classify the three boards represented in the pilot."""
    return pd.Series(np.select(
        [code.str.startswith("sh.60"), code.str.startswith("sh.688"), code.str.startswith("sz.300")],
        ["SH main", "STAR", "ChiNext"], default="Other",
    ), index=code.index)


def load_reconciliation(codes: list[str]) -> pd.DataFrame:
    """Load the QC daily reconciliation and add signed gaps."""
    frame = pd.read_csv(RECONCILIATION_PATH, parse_dates=["date"])
    frame = frame[frame["code"].isin(codes)].copy()
    frame["year"] = frame["date"].dt.year
    frame["board"] = board_name(frame["code"])
    frame["daily_minus_minute_volume"] = frame["daily_volume"] - frame["minute_volume"]
    frame["daily_minus_minute_amount"] = frame["daily_amount"] - frame["minute_amount"]
    return frame


def direction_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify whether gaps have the direction required by the hypothesis."""
    mismatch = frame[frame["daily_minus_minute_volume"].ne(0)].copy()
    mismatch["daily_gt_minute"] = mismatch["daily_minus_minute_volume"].gt(0)
    overall = mismatch.groupby("board", as_index=False).agg(
        mismatch_days=("date", "size"), daily_gt_minute_days=("daily_gt_minute", "sum"),
        hypothesis_direction_share=("daily_gt_minute", "mean"),
        median_gap_relative=("volume_relative_error", lambda values: -values.median()),
    )
    by_year = mismatch.groupby(["year", "board"], as_index=False).agg(
        mismatch_days=("date", "size"), daily_gt_minute_days=("daily_gt_minute", "sum"),
        hypothesis_direction_share=("daily_gt_minute", "mean"),
        median_gap_relative=("volume_relative_error", lambda values: -values.median()),
    )
    return overall, by_year


def volume_granularity(codes: list[str]) -> pd.DataFrame:
    """Measure the share of minute bars reported in exact 100-share lots."""
    rows: list[dict[str, object]] = []
    for code in codes:
        frame = pd.read_parquet(RAW_5MIN_DIR / f"{code}.parquet", columns=["date", "volume"])
        frame["year"] = pd.to_datetime(frame["date"]).dt.year
        for year, group in frame.groupby("year"):
            rows.append({"code": code, "board": board_name(pd.Series([code])).iloc[0],
                         "year": year, "bars": len(group),
                         "exact_100_share_ratio": group["volume"].mod(100).eq(0).mean()})
    return pd.DataFrame(rows)


def retry(request: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Retry one AkShare request with bounded exponential backoff."""
    error: Exception | None = None
    for attempt in range(EVENT_RETRY_ATTEMPTS):
        try:
            result = request()
            time.sleep(EVENT_REQUEST_SLEEP_SECONDS)
            return result
        except Exception as exc:
            error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"AkShare request failed: {error}")


def month_windows(start: str, end: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split a long event request into safe calendar-month windows."""
    starts = pd.date_range(pd.Timestamp(start).replace(day=1), pd.Timestamp(end), freq="MS")
    return [(value, min(value + pd.offsets.MonthEnd(0), pd.Timestamp(end))) for value in starts]


def fetch_block_window(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch a block-trade window and bisect if the API page cap is reached."""
    request = lambda: ak.stock_dzjy_mrmx(
        symbol="A股", start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d")
    )
    frame = retry(request)
    if len(frame) < BLOCK_TRADE_PAGE_LIMIT or start >= end:
        return frame
    midpoint = start + (end - start) / 2
    left = fetch_block_window(start, midpoint.normalize())
    right = fetch_block_window(midpoint.normalize() + pd.Timedelta(days=1), end)
    return pd.concat([left, right], ignore_index=True)


def download_block_trades(codes: list[str], end: str) -> pd.DataFrame:
    """Download and cache monthly AkShare block-trade detail."""
    cache = RECONCILIATION_INVESTIGATION_DIR / "block_cache"
    cache.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    six_digit = {code.split(".", 1)[1] for code in codes}
    for start, stop in month_windows(BLOCK_TRADE_START_DATE, end):
        path = cache / f"{start:%Y-%m}.csv"
        if path.exists():
            selected = pd.read_csv(path, dtype={"证券代码": str}, parse_dates=["交易日期"])
        else:
            raw = fetch_block_window(start, stop)
            if raw.empty:
                selected = pd.DataFrame(columns=["交易日期", "证券代码", "成交量", "成交额"])
            else:
                raw["证券代码"] = raw["证券代码"].astype(str).str.zfill(6)
                selected = raw[raw["证券代码"].isin(six_digit)].copy()
            selected.to_csv(path, index=False, encoding="utf-8")
        frames.append(selected)
        print(f"block trades {start:%Y-%m}: {len(selected)} pilot rows", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def normalize_block_trades(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate AkShare block trades by stock-date."""
    if frame.empty:
        return pd.DataFrame(columns=["date", "code", "block_volume", "block_amount", "block_trades"])
    frame["date"] = pd.to_datetime(frame["交易日期"])
    frame["code"] = np.where(frame["证券代码"].str.startswith("6"), "sh.", "sz.") + frame["证券代码"]
    frame["成交量"] = pd.to_numeric(frame["成交量"], errors="coerce")
    frame["成交额"] = pd.to_numeric(frame["成交额"], errors="coerce")
    return frame.groupby(["date", "code"], as_index=False).agg(
        block_volume=("成交量", "sum"), block_amount=("成交额", "sum"), block_trades=("成交量", "size")
    )


def add_mechanism_matches(reconciliation: pd.DataFrame, block: pd.DataFrame) -> pd.DataFrame:
    """Test block-volume matching and a close-price after-hours proxy."""
    frame = reconciliation.merge(block, on=["date", "code"], how="left")
    for column in ["block_volume", "block_amount", "block_trades"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    gap = frame["daily_minus_minute_volume"]
    amount_gap = frame["daily_minus_minute_amount"]
    frame["hypothesis_direction"] = gap.gt(0)
    frame["implied_gap_price"] = amount_gap.div(gap.replace(0, np.nan))
    frame["implied_price_close_error"] = (
        frame["implied_gap_price"] - frame["daily_close"]
    ).abs().div(frame["daily_close"].replace(0, np.nan))
    tolerance = np.maximum(BLOCK_MATCH_ABSOLUTE_SHARES, gap.abs() * BLOCK_MATCH_RELATIVE_TOLERANCE)
    frame["block_volume_match"] = gap.gt(0) & frame["block_volume"].gt(0) & (gap.sub(frame["block_volume"]).abs() <= tolerance)
    frame["fixed_price_proxy_match"] = gap.gt(0) & amount_gap.gt(0) & (
        frame["implied_price_close_error"] <= FIXED_PRICE_RELATIVE_TOLERANCE
    )
    frame["mechanism_explained"] = frame["block_volume_match"] | frame["fixed_price_proxy_match"]
    return frame


def mechanism_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Report the explainable share of true volume mismatches."""
    mismatch = frame[frame["daily_minus_minute_volume"].ne(0)].copy()
    return mismatch.groupby(["year", "board"], as_index=False).agg(
        mismatch_days=("date", "size"), hypothesis_direction_days=("hypothesis_direction", "sum"),
        block_trade_overlap_days=("block_trades", lambda values: values.gt(0).sum()),
        block_volume_match_days=("block_volume_match", "sum"),
        fixed_price_proxy_days=("fixed_price_proxy_match", "sum"),
        mechanism_explained_days=("mechanism_explained", "sum"),
        mechanism_explained_share=("mechanism_explained", "mean"),
    )


def investigation_overview(frame: pd.DataFrame, block_rows: int) -> pd.DataFrame:
    """Create one decision-ready row for the scale/no-scale gate."""
    mismatch = frame[frame["daily_minus_minute_volume"].ne(0)]
    explained = int(mismatch["mechanism_explained"].sum())
    return pd.DataFrame([{
        "volume_mismatch_days": len(mismatch),
        "daily_gt_minute_days": int(mismatch["hypothesis_direction"].sum()),
        "minute_gt_daily_days": int(mismatch["daily_minus_minute_volume"].lt(0).sum()),
        "akshare_block_trade_rows": block_rows,
        "block_trade_overlap_days": int(mismatch["block_trades"].gt(0).sum()),
        "block_volume_match_days": int(mismatch["block_volume_match"].sum()),
        "fixed_price_proxy_days": int(mismatch["fixed_price_proxy_match"].sum()),
        "mechanism_explained_days": explained,
        "mechanism_explained_share": explained / len(mismatch),
        "unexplained_days": len(mismatch) - explained,
        "scale_gate": "BLOCKED",
    }])


def download_cross_vendor_daily(codes: list[str], end: str) -> pd.DataFrame:
    """Download independent Sina daily bars through AkShare."""
    cache = RECONCILIATION_INVESTIGATION_DIR / "akshare_daily"
    cache.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for code in codes:
        path = cache / f"{code}.csv"
        if path.exists():
            frame = pd.read_csv(path, parse_dates=["date"])
        else:
            raw = retry(lambda: ak.stock_zh_a_daily(symbol=code.replace(".", ""),
                                                    start_date=BLOCK_TRADE_START_DATE.replace("-", ""),
                                                    end_date=end.replace("-", ""), adjust=""))
            frame = raw.rename(columns={"volume": "ak_volume_shares", "amount": "ak_amount"})
            frame = frame[["date", "ak_volume_shares", "ak_amount"]]
            frame["date"] = pd.to_datetime(frame["date"])
            frame["code"] = code
            frame.to_csv(path, index=False, encoding="utf-8")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def cross_vendor_summary(reconciliation: pd.DataFrame, akshare_daily: pd.DataFrame) -> pd.DataFrame:
    """Check which Baostock side agrees with independent Sina daily bars."""
    merged = reconciliation.merge(akshare_daily, on=["date", "code"], how="inner")
    merged["ak_volume_shares"] = pd.to_numeric(merged["ak_volume_shares"])
    merged["ak_daily_volume_match"] = (
        merged["ak_volume_shares"] - merged["daily_volume"]
    ).abs() < CROSS_VENDOR_VOLUME_TOLERANCE_SHARES
    merged["ak_minute_volume_match"] = (
        merged["ak_volume_shares"] - merged["minute_volume"]
    ).abs() < CROSS_VENDOR_VOLUME_TOLERANCE_SHARES
    return merged.groupby(["year", "board"], as_index=False).agg(
        compared_days=("date", "size"), ak_matches_daily_share=("ak_daily_volume_match", "mean"),
        ak_matches_minute_share=("ak_minute_volume_match", "mean"),
    )


def factor_event_dates(code: str) -> set[pd.Timestamp]:
    """Infer adjustment-factor change dates from CTO HFQ/raw pairs."""
    stem = code.split(".", 1)[1]
    hfq = pd.read_csv(CTO_DAILY_HFQ_DIR / f"{stem}.csv", encoding="utf-8-sig")
    raw = pd.read_csv(CTO_DAILY_RAW_DIR / f"{stem}.csv", encoding="utf-8-sig")
    pair = hfq[["日期", "收盘"]].merge(raw[["日期", "收盘"]], on="日期", suffixes=("_hfq", "_raw"))
    pair["date"] = pd.to_datetime(pair["日期"])
    pair["factor"] = pd.to_numeric(pair["收盘_hfq"]) / pd.to_numeric(pair["收盘_raw"])
    changed = pair["factor"].pct_change().abs() > FACTOR_CHANGE_TOLERANCE
    return set(pair.loc[changed, "date"])


def last_bar_table(codes: list[str]) -> pd.DataFrame:
    """Extract the actual last bar used in daily close reconciliation."""
    frames: list[pd.DataFrame] = []
    for code in codes:
        frame = pd.read_parquet(RAW_5MIN_DIR / f"{code}.parquet")
        frame = frame.sort_values("time").groupby("date", as_index=False).tail(1).copy()
        frame["code"] = code
        frames.append(frame.rename(columns={column: f"last_bar_{column}" for column in
                                            ["open", "high", "low", "close", "volume", "amount", "time"]}))
    return pd.concat(frames, ignore_index=True)


def close_anomaly_tables(mechanisms: pd.DataFrame, codes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a 20-row close audit and a grouped pattern summary."""
    close = mechanisms[mechanisms["close_absolute_error"] >= PRICE_TOLERANCE].copy()
    close = close.merge(last_bar_table(codes), on=["date", "code"], how="left")
    events = {code: factor_event_dates(code) for code in codes}
    close["factor_change_date"] = [day in events[code] for code, day in zip(close["code"], close["date"])]
    close["cross_stock_anomalies_same_date"] = close.groupby("date")["code"].transform("size")
    summary = close.groupby(["year", "board"], as_index=False).agg(
        close_mismatch_days=("date", "size"), volume_mismatch_share=("volume_error", lambda x: x.ne(0).mean()),
        block_trade_overlap_days=("block_trades", lambda x: x.gt(0).sum()),
        factor_change_overlap_days=("factor_change_date", "sum"),
        fixed_price_proxy_days=("fixed_price_proxy_match", "sum"),
        median_same_date_stock_count=("cross_stock_anomalies_same_date", "median"),
    )
    top = close.nlargest(CLOSE_SAMPLE_SIZE, "close_absolute_error")
    return summary, top


def write_outputs(tables: dict[str, pd.DataFrame]) -> None:
    """Write auditable CSV evidence tables."""
    RECONCILIATION_INVESTIGATION_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(RECONCILIATION_INVESTIGATION_DIR / f"{name}.csv", index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--end-date", default="2026-08-06")
    args = parser.parse_args()
    codes = load_codes(args.limit)
    reconciliation = load_reconciliation(codes)
    direction, direction_year = direction_summary(reconciliation)
    blocks_raw = download_block_trades(codes, args.end_date)
    blocks = normalize_block_trades(blocks_raw)
    mechanisms = add_mechanism_matches(reconciliation, blocks)
    cross = cross_vendor_summary(reconciliation, download_cross_vendor_daily(codes, args.end_date))
    close_summary, close_top = close_anomaly_tables(mechanisms, codes)
    write_outputs({
        "investigation_overview": investigation_overview(mechanisms, len(blocks_raw)),
        "volume_direction_by_board": direction,
        "volume_direction_by_year_board": direction_year,
        "minute_volume_granularity": volume_granularity(codes),
        "block_trades_akshare": blocks_raw,
        "mechanism_match_detail": mechanisms[mechanisms["daily_minus_minute_volume"].ne(0)],
        "mechanism_match_summary": mechanism_summary(mechanisms),
        "cross_vendor_daily_summary": cross,
        "close_anomaly_summary": close_summary,
        "close_anomaly_top20": close_top,
    })
    print(mechanism_summary(mechanisms).to_string(index=False))
    print("\nCLOSE ANOMALIES\n", close_summary.to_string(index=False))


if __name__ == "__main__":
    main()
