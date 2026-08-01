"""AkShare full-A daily pipeline and CTO diagnostics.

The downloader is checkpointed per symbol because the upstream endpoint is
rate-limited. The cleaner expects a metadata CSV with code, name, list_date,
is_st, and is_suspended when those historical fields are available.

Inputs: vendor daily price CSVs, Baostock IPO metadata, and optional metadata.
Outputs: eligible daily/monthly CTO panels and three diagnostic artifacts.
Role: applies time-correct filters and constructs the research signal.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import akshare as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA = Path("data/cto")
RAW = DATA / "daily_hfq"
RAW_UNADJ = DATA / "daily_raw"
OUT = DATA / "outputs"
for p in (RAW, RAW_UNADJ, OUT):
    p.mkdir(parents=True, exist_ok=True)


def retry_hist(code: str, start: str, end: str, adjust: str = "hfq", tries: int = 3) -> pd.DataFrame:
    last = None
    for i in range(tries):
        try:
            return ak.stock_zh_a_hist(code, "daily", start, end, adjust, timeout=20)
        except Exception as exc:
            last = exc
            time.sleep(min(60, (2 ** i) * random.uniform(1, 3)))
    raise last


def validate_delisted_coverage(start="20100101", end="20260722"):
    """Record exchange-list coverage and probe known post-2019 delisted codes."""
    result = {"checked_at": pd.Timestamp.now().isoformat(), "start": start, "end": end, "lists": {}, "probes": []}
    candidates = []
    for market, fn in (("SSE", ak.stock_info_sh_delist), ("SZSE", ak.stock_info_sz_delist)):
        try:
            df = fn()
            df.to_csv(DATA / f"{market.lower()}_delist_coverage.csv", index=False, encoding="utf-8-sig")
            result["lists"][market] = {"status": "ok", "rows": len(df), "columns": list(df.columns)}
            code_col = "公司代码" if market == "SSE" else "证券代码"
            name_col = "公司简称" if market == "SSE" else "证券简称"
            date_col = "暂停上市日期" if market == "SSE" else "终止上市日期"
            if code_col in df:
                for _, r in df.iterrows():
                    d = str(r.get(date_col, ""))
                    if d[:4].isdigit() and int(d[:4]) >= 2019:
                        candidates.append((str(r[code_col]).zfill(6), market, str(r.get(name_col, "")), d))
        except Exception as exc:
            result["lists"][market] = {"status": "failed", "error": repr(exc)}
    # Probe up to 20 post-2019 candidates, rather than claiming every delisted
    # ticker is covered by the historical endpoint.
    # A short probe is intentional: the endpoint is one-symbol-at-a-time and
    # rate-limited. This verifies coverage without pretending it is exhaustive.
    for code, market, name, delist_date in candidates[:6]:
        item = {"code": code, "market": market, "name": name, "delist_date": delist_date}
        try:
            d = retry_hist(code, start, end, "hfq", tries=2)
            item.update({"status": "ok" if len(d) else "empty", "rows": len(d),
                         "first_date": str(d["日期"].min()) if len(d) else None,
                         "last_date": str(d["日期"].max()) if len(d) else None})
        except Exception as exc:
            item.update({"status": "failed", "error": repr(exc)})
        result["probes"].append(item)
        (OUT / "delisted_coverage.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "delisted_coverage.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def get_universe():
    """Get current A-share code/name universe; historical metadata is supplied separately."""
    df = ak.stock_info_a_code_name()
    df.columns = ["code", "name"]
    df["code"] = df["code"].astype(str).str.zfill(6)
    df.to_csv(DATA / "universe_code_name.csv", index=False, encoding="utf-8-sig")
    return df


def download_all(start="20100101", end="20260722", limit=None, sleep_min=1.0, sleep_max=3.0):
    universe = get_universe()
    if limit:
        universe = universe.head(limit)
    log = []
    for n, code in enumerate(universe["code"], 1):
        path = RAW / f"{code}.csv"
        if path.exists():
            continue
        item = {"code": code}
        try:
            df = retry_hist(code, start, end, "hfq")
            if len(df):
                df.to_csv(path, index=False, encoding="utf-8-sig")
                # Raw prices are needed for exchange price-limit detection;
                # adjusted prices alone can create false limit classifications.
                raw = retry_hist(code, start, end, "")
                if len(raw):
                    raw.to_csv(RAW_UNADJ / f"{code}.csv", index=False, encoding="utf-8-sig")
            item.update({"status": "ok" if len(df) else "empty", "rows": len(df)})
        except Exception as exc:
            item.update({"status": "failed", "error": repr(exc)})
        log.append(item)
        if n % 50 == 0:
            (OUT / "download_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(random.uniform(sleep_min, sleep_max))
    (OUT / "download_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return log


def limit_pct(code: str, date: object, is_st: bool = False) -> float:
    """Price-limit rule used for previous-close limit filtering."""
    d = pd.Timestamp(date)
    if as_bool(is_st):
        return 0.05
    if code.startswith("688"):
        return 0.20
    if code.startswith(("300", "301")):
        return 0.20 if d >= pd.Timestamp("2020-08-24") else 0.10
    return 0.10


def as_bool(v: object) -> bool:
    return v is True or str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def clean_one(df: pd.DataFrame, code: str, meta: dict | None = None, raw_df: pd.DataFrame | None = None):
    meta = meta or {}
    x = df.copy()
    x["date"] = pd.to_datetime(x["日期"])
    x = x.sort_values("date").drop_duplicates("date")
    x["open"] = pd.to_numeric(x["开盘"], errors="coerce")
    x["close"] = pd.to_numeric(x["收盘"], errors="coerce")
    x["pct_change"] = pd.to_numeric(x.get("涨跌幅"), errors="coerce") / 100
    x["prev_close"] = x["close"].shift(1)
    x["prev_close_ret"] = x["close"] / x["prev_close"] - 1
    if raw_df is not None and len(raw_df):
        r = raw_df.copy()
        r["date"] = pd.to_datetime(r["日期"])
        r["raw_close"] = pd.to_numeric(r["收盘"], errors="coerce")
        r = r.sort_values("date").drop_duplicates("date")[["date", "raw_close"]]
        x = x.merge(r, on="date", how="left")
        x["raw_prev_close"] = x["raw_close"].shift(1)
    else:
        x["raw_close"] = x["close"]
        x["raw_prev_close"] = x["prev_close"]
    source_is_st = x.get("是否ST", pd.Series(False, index=x.index)).map(as_bool)
    x["is_st"] = source_is_st | as_bool(meta.get("is_st", False))
    x["limit_pct"] = [limit_pct(code, d, is_st) for d, is_st in zip(x["date"], x["is_st"])]
    up_limit = (x["raw_prev_close"] * (1 + x["limit_pct"])).round(2)
    down_limit = (x["raw_prev_close"] * (1 - x["limit_pct"])).round(2)
    x["close_at_limit"] = x["raw_close"].sub(up_limit).abs().le(0.005) | x["raw_close"].sub(down_limit).abs().le(0.005)
    # CTO(t) is excluded when the previous close, at t-1, hit a limit.
    x["prev_close_at_limit"] = x["close_at_limit"].shift(1, fill_value=False)
    x["listing_first_day"] = x["date"].eq(x["date"].min())
    trade_status = x.get("交易状态", pd.Series("1", index=x.index)).astype(str)
    x["suspended"] = trade_status.ne("1") | x.get("成交量", pd.Series(index=x.index)).fillna(0).eq(0) | as_bool(meta.get("is_suspended", False))
    x["listed_less_1y"] = False
    if meta.get("list_date"):
        x["listed_less_1y"] = x["date"] < pd.Timestamp(meta["list_date"]) + pd.DateOffset(years=1)
    x["cto_daily"] = x["open"] / x["prev_close"] - 1
    x["eligible"] = ~(x[["is_st", "listed_less_1y", "suspended", "listing_first_day", "prev_close_at_limit"]].any(axis=1))
    return x


def build_cto(metadata_path=None, source_dir=RAW, raw_dir=RAW_UNADJ, output_base=DATA):
    meta = {}
    if metadata_path and Path(metadata_path).exists():
        md = pd.read_csv(metadata_path, dtype={"code": str}).fillna("")
        meta = {str(r.code).zfill(6): r.to_dict() for _, r in md.iterrows()}
    elif (Path(source_dir).parent / "baostock_universe.csv").exists():
        # Baostock's IPO dates provide the time-correct listing-age filter for
        # the downloaded A-share universe without needing a second metadata feed.
        md = pd.read_csv(Path(source_dir).parent / "baostock_universe.csv", dtype=str).fillna("")
        for _, row in md.iterrows():
            code = str(row.get("code", "")).split(".")[-1].zfill(6)
            meta[code] = {"list_date": row.get("ipoDate", "")}
    daily_parts, monthly_parts = [], []
    for path in sorted(Path(source_dir).glob("*.csv")):
        code = path.stem
        raw = pd.read_csv(path)
        raw_path = Path(raw_dir) / path.name
        raw_unadj = pd.read_csv(raw_path) if raw_path.exists() else None
        c = clean_one(raw, code, meta.get(code), raw_unadj)
        c["code"] = code
        daily_parts.append(c[["date", "code", "open", "close", "prev_close", "cto_daily", "limit_pct", "close_at_limit", "prev_close_at_limit", "is_st", "listed_less_1y", "suspended", "listing_first_day", "eligible"]])
        e = c[c["eligible"]].copy()
        if len(e):
            m = e.assign(month=e["date"].dt.to_period("M")).groupby(["code", "month"], as_index=False).agg(
                cto_month=("cto_daily", "mean"), valid_days=("cto_daily", "count"),
                limit_excluded=("prev_close_at_limit", "sum"))
            monthly_parts.append(m)
    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    monthly = pd.concat(monthly_parts, ignore_index=True) if monthly_parts else pd.DataFrame()
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_base / "daily_cto.csv", index=False)
    monthly.to_csv(output_base / "monthly_cto.csv", index=False)
    make_diagnostics(daily, monthly, output_base / "outputs")


def make_diagnostics(daily, monthly, output_dir=OUT):
    if daily.empty:
        return
    daily["year"] = daily["date"].dt.year
    yearly = daily.groupby("year").agg(total_rows=("code", "size"), eligible_rows=("eligible", "sum"), stocks=("code", "nunique"))
    yearly["limit_excluded_pct"] = daily.groupby("year")["prev_close_at_limit"].mean() * 100
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(output_dir / "yearly_sample_counts.csv")
    limit_ts = daily.groupby(daily["date"].dt.to_period("M")).agg(
        excluded=("prev_close_at_limit", "sum"), observations=("code", "size"))
    limit_ts["excluded_pct"] = limit_ts["excluded"] / limit_ts["observations"] * 100
    limit_ts.to_csv(output_dir / "limit_exclusion_timeseries.csv")
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))
    if not monthly.empty:
        monthly["month"] = monthly["month"].astype(str)
        monthly.groupby("month")["cto_month"].quantile([.1, .5, .9]).unstack().plot(ax=axes[0], lw=0.9)
        axes[0].set_title("Monthly CTO cross-sectional distribution (P10/P50/P90)")
    yearly["stocks"].plot(ax=axes[1], marker="o", color="tab:blue", label="stocks")
    axes[1].set_ylabel("stocks")
    ax_right = axes[1].twinx()
    yearly["eligible_rows"].plot(ax=ax_right, marker="o", color="tab:orange", label="eligible_rows")
    ax_right.set_ylabel("eligible rows")
    axes[1].legend(loc="upper left")
    ax_right.legend(loc="upper right")
    axes[1].set_title("Annual sample counts")
    limit_ts["excluded_pct"].plot(ax=axes[2])
    axes[2].set_title("Previous-close price-limit exclusion rate")
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output_dir / "cto_diagnostics.png", dpi=160)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-delisted", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--build-cto", action="store_true")
    ap.add_argument("--metadata")
    ap.add_argument("--source", choices=["akshare", "baostock"], default="akshare")
    args = ap.parse_args()
    if args.validate_delisted:
        print(json.dumps(validate_delisted_coverage(), ensure_ascii=False, indent=2))
    if args.download:
        print(json.dumps(download_all(limit=args.limit), ensure_ascii=False, indent=2))
    if args.build_cto:
        if args.source == "baostock":
            build_cto(args.metadata, Path("data/cto_baostock/daily_hfq"), Path("data/cto_baostock/daily_raw"), Path("data/cto_baostock"))
        else:
            build_cto(args.metadata)
