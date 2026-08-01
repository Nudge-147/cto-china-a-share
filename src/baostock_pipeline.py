"""Baostock-only price downloader.

Uses Baostock for both post-adjusted and raw prices so CTO and price-limit checks
never mix vendors. The ticker universe comes from Baostock's stock_basic; the
optional exchange delisting lists only supplement symbols missing from that
current universe.

Inputs: Baostock API responses and optional exchange delisting lists.
Outputs: raw and post-adjusted daily CSVs plus a checkpointed universe/log.
Role: first full-rebuild stage; all price-series calculations use this vendor.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import baostock as bs
import pandas as pd

BASE = Path("data/cto_baostock")
HFQ = BASE / "daily_hfq"
RAW = BASE / "daily_raw"
OUT = BASE / "outputs"
for p in (HFQ, RAW, OUT):
    p.mkdir(parents=True, exist_ok=True)


def fetch_rows(code: str, start: str, end: str, adjustflag: str) -> pd.DataFrame:
    rs = bs.query_history_k_data_plus(
        code,
        "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,adjustflag,tradestatus,isST",
        start_date=start,
        end_date=end,
        frequency="d",
        adjustflag=adjustflag,
    )
    if rs.error_code != "0":
        raise RuntimeError(f"{code}: {rs.error_code} {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    cols = ["日期", "代码", "开盘", "最高", "最低", "收盘", "前收盘", "成交量", "成交额", "换手率", "涨跌幅", "复权标记", "交易状态", "是否ST"]
    return pd.DataFrame(rows, columns=cols)


def fetch_adjust_factors(code: str, start: str, end: str) -> pd.DataFrame:
    """Fetch sparse adjustment events; backAdjustFactor is forward-filled by date."""
    rs = bs.query_adjust_factor(code, start_date=start, end_date=end)
    if rs.error_code != "0":
        raise RuntimeError(f"{code}: {rs.error_code} {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    factors = pd.DataFrame(rows, columns=rs.fields)
    if factors.empty:
        raise RuntimeError(f"{code}: no adjustment-factor history returned")
    factors = factors.rename(columns={"dividOperateDate": "日期", "backAdjustFactor": "后复权因子"})
    factors["日期"] = pd.to_datetime(factors["日期"])
    factors["后复权因子"] = pd.to_numeric(factors["后复权因子"], errors="coerce")
    return factors[["日期", "后复权因子"]].dropna().sort_values("日期")


def reconstruct_raw_prices(hfq: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Recover raw OHLC from Baostock post-adjusted prices and sparse factors."""
    x = hfq.copy()
    x["日期"] = pd.to_datetime(x["日期"])
    x = pd.merge_asof(x.sort_values("日期"), factors, on="日期", direction="backward")
    if x["后复权因子"].isna().any():
        missing = int(x["后复权因子"].isna().sum())
        raise RuntimeError(f"missing adjustment factor for {missing} trading days")
    for column in ("开盘", "最高", "最低", "收盘", "前收盘"):
        x[column] = pd.to_numeric(x[column], errors="coerce") / x["后复权因子"]
    x["日期"] = x["日期"].dt.strftime("%Y-%m-%d")
    return x.drop(columns="后复权因子")


def login_or_raise():
    """Open a fresh Baostock session, including after an idle disconnect."""
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"Baostock login failed: {login.error_code} {login.error_msg}")


def fetch_price_pair(code, start, end, retries=5):
    """Fetch one post-adjusted history plus sparse factors, with reconnect retries."""
    hfq_seconds = 0.0
    factor_seconds = 0.0
    raw_fallback_seconds = 0.0
    used_raw_fallback = False
    reconnect_seconds = 0.0
    reconnects = 0
    for attempt in range(retries):
        query_start = time.perf_counter()
        try:
            hfq = fetch_rows(code, start, end, "1")
            hfq_seconds += time.perf_counter() - query_start
            factor_start = time.perf_counter()
            factors = fetch_adjust_factors(code, "1990-01-01", end)
            factor_seconds += time.perf_counter() - factor_start
            try:
                raw = reconstruct_raw_prices(hfq, factors)
            except RuntimeError as exc:
                # Some securities have no factor before the requested history
                # (or no factor history at all).  Preserve full coverage with a
                # same-vendor raw query only for these exceptional codes.
                if "missing adjustment factor" not in str(exc):
                    raise
                raw_start = time.perf_counter()
                raw = fetch_rows(code, start, end, "3")
                raw_fallback_seconds += time.perf_counter() - raw_start
                used_raw_fallback = True
            return hfq, raw, hfq_seconds, factor_seconds, raw_fallback_seconds, used_raw_fallback, reconnect_seconds, reconnects
        except RuntimeError as exc:
            hfq_seconds += time.perf_counter() - query_start
            retryable = ("10001001", "10002007")
            if not any(code in str(exc) for code in retryable) or attempt == retries - 1:
                raise
            try:
                bs.logout()
            except Exception:
                pass
            # Back off substantially: repeated immediate requests after a socket
            # error cause Baostock to reject an entire stretch of symbols.
            reconnect_start = time.perf_counter()
            time.sleep(min(30, 2**attempt * 2) + random.uniform(0, 2))
            login_or_raise()
            reconnect_seconds += time.perf_counter() - reconnect_start
            reconnects += 1


def stock_universe():
    rs = bs.query_stock_basic()
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    # Baostock type=1 includes B shares (sh.900xxx and sz.200xxx).  The
    # research universe is mainland A shares only, so remove them explicitly.
    df = df[(df["type"] == "1") & df["code"].str.match(r"^(sh|sz)\.")].copy()
    df = df[~df["code"].str.match(r"^(sh\.900|sz\.200)")].copy()
    # Supplement Baostock's current list with exchange delisted symbols already
    # validated by the AkShare coverage step. Prices still come exclusively
    # from Baostock.
    extra = []
    for path, prefix, code_col, name_col, date_col in [
        (Path("data/cto/sse_delist_coverage.csv"), "sh", "公司代码", "公司简称", "上市日期"),
        (Path("data/cto/szse_delist_coverage.csv"), "sz", "证券代码", "证券简称", "上市日期"),
    ]:
        if path.exists():
            d = pd.read_csv(path, dtype=str).fillna("")
            for _, r in d.iterrows():
                code = str(r[code_col]).zfill(6)
                extra.append({"code": f"{prefix}.{code}", "code_name": r[name_col], "ipoDate": r.get(date_col, ""), "type": "1", "status": "0"})
    if extra:
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True).drop_duplicates("code")
    df.to_csv(BASE / "baostock_universe.csv", index=False, encoding="utf-8-sig")
    return df


def download_all(start="2010-01-01", end="2026-07-22", limit=None, shard_index=0, shard_count=1):
    universe = stock_universe()
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    universe = universe[universe["code"].str.split(".").str[-1].astype(int).mod(shard_count).eq(shard_index)]
    if limit:
        universe = universe.head(limit)
    log = []
    log_path = OUT / f"download_log_shard{shard_index:02d}.json"
    for _, row in universe.iterrows():
        vendor_code = row["code"]
        code = vendor_code.split(".", 1)[1]
        if (HFQ / f"{code}.csv").exists() and (RAW / f"{code}.csv").exists():
            continue
        item = {"code": vendor_code, "name": row.get("code_name", "")}
        stock_start = time.perf_counter()
        try:
            # Baostock: 1=后复权, 2=前复权, 3=不复权.
            hfq, raw, hfq_seconds, factor_seconds, raw_fallback_seconds, used_raw_fallback, reconnect_seconds, reconnects = fetch_price_pair(vendor_code, start, end)
            write_start = time.perf_counter()
            hfq.to_csv(HFQ / f"{code}.csv", index=False, encoding="utf-8-sig")
            raw.to_csv(RAW / f"{code}.csv", index=False, encoding="utf-8-sig")
            write_seconds = time.perf_counter() - write_start
            item.update({"status": "ok", "hfq_rows": len(hfq), "raw_rows": len(raw),
                         "first_date": hfq["日期"].min() if len(hfq) else None,
                         "last_date": hfq["日期"].max() if len(hfq) else None,
                         "hfq_seconds": round(hfq_seconds, 3),
                         "factor_seconds": round(factor_seconds, 3),
                         "raw_fallback_seconds": round(raw_fallback_seconds, 3),
                         "used_raw_fallback": used_raw_fallback,
                         "reconnect_seconds": round(reconnect_seconds, 3),
                         "reconnects": reconnects,
                         "write_seconds": round(write_seconds, 3)})
        except Exception as exc:
            item.update({"status": "failed", "error": repr(exc)})
        sleep_seconds = random.uniform(1, 3)
        item["sleep_seconds"] = round(sleep_seconds, 3)
        item["total_seconds"] = round(time.perf_counter() - stock_start + sleep_seconds, 3)
        log.append(item)
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(sleep_seconds)
    return log


def download_codes(codes, start="2010-01-01", end="2026-07-22"):
    """Download explicit six-digit symbols for validation and recovery."""
    log = []
    for code in codes:
        code = str(code).zfill(6)
        vendor_code = f"sh.{code}" if code.startswith(("6", "9")) else f"sz.{code}"
        item = {"code": vendor_code}
        try:
            hfq, raw, *_ = fetch_price_pair(vendor_code, start, end)
            hfq.to_csv(HFQ / f"{code}.csv", index=False, encoding="utf-8-sig")
            raw.to_csv(RAW / f"{code}.csv", index=False, encoding="utf-8-sig")
            item.update({"status": "ok", "hfq_rows": len(hfq), "raw_rows": len(raw)})
        except Exception as exc:
            item.update({"status": "failed", "error": repr(exc)})
        log.append(item)
    return log


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--codes", help="comma-separated six-digit codes")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()
    login_or_raise()
    try:
        result = download_codes(args.codes.split(",")) if args.codes else download_all(limit=args.limit, shard_index=args.shard_index, shard_count=args.shard_count)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        bs.logout()
