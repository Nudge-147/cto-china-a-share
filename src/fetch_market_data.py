"""Probe three-stock AkShare price coverage and adjustment continuity.

Inputs: AkShare price and exchange-delisting endpoints.
Outputs: small local validation CSVs and an adjustment chart under ``data/``.
Role: exploratory data-source validation; not the production Baostock pipeline.
"""

from pathlib import Path
import json
import time
import akshare as ak
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("data")
OUT.mkdir(exist_ok=True)

def save_csv(df, name):
    path = OUT / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path

def fetch_hist(code, adjust):
    last = None
    for attempt in range(3):
        try:
            return ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20190101", end_date="20260722", adjust=adjust, timeout=20)
        except Exception as exc:
            last = exc
            time.sleep(2 + attempt * 2)
    raise last

def main():
    # Start from exchange delisting lists; choose the first post-2019 candidate
    # with a usable historical daily series.
    lists = {}
    for market, fn in [("SSE", ak.stock_info_sh_delist), ("SZSE", ak.stock_info_sz_delist)]:
        try:
            lists[market] = fn()
            save_csv(lists[market], f"{market.lower()}_delist_list.csv")
        except Exception as exc:
            lists[market] = pd.DataFrame()
            print(f"{market} delist list failed: {exc}")

    candidates = []
    for market, df in lists.items():
        if df.empty:
            continue
        print(f"\n{market} columns: {list(df.columns)}")
        print(df.head(3).to_string(index=False))
        for _, row in df.iterrows():
            vals = " ".join(str(x) for x in row.tolist())
            # The exchange lists usually include a termination date. Keep broad
            # candidates and validate by requesting daily history below.
            if any(str(y) in vals for y in range(2019, 2027)):
                code = next((str(x).zfill(6) for x in row.tolist() if str(x).isdigit() and len(str(x)) <= 6), None)
                if code:
                    candidates.append((code, market, vals))

    # Prefer a known post-2019 delisting candidate, then fall back to the list.
    ordered = [("603157", "SSE"), ("603996", "SSE"), ("600074", "SSE"), ("600680", "SSE"), ("000670", "SZSE"), ("000751", "SZSE")] + [(c, m) for c, m, _ in candidates]
    chosen = None
    for code, market in ordered:
        try:
            raw = fetch_hist(code, "")
            if len(raw) >= 20:
                chosen = (code, market, raw)
                break
        except Exception as exc:
            print(f"candidate {code} failed: {exc}")
    if chosen is None:
        raise RuntimeError("No usable post-2019 delisted stock found")

    stocks = [("000001", "平安银行"), ("600519", "贵州茅台"), (chosen[0], f"退市股-{chosen[0]}")]
    summary = {"chosen_delisted": {"code": chosen[0], "market": chosen[1]}, "stocks": {}}
    fig, axes = plt.subplots(len(stocks), 1, figsize=(13, 11), sharex=False)
    if len(stocks) == 1:
        axes = [axes]
    for ax, (code, name) in zip(axes, stocks):
        for adjust, label in [("", "raw"), ("qfq", "qfq"), ("hfq", "hfq")]:
            try:
                df = fetch_hist(code, adjust)
            except Exception as exc:
                print(f"{code} {label} failed: {exc}")
                continue
            save_csv(df, f"{code}_{label}.csv")
            date_col, close_col = "日期", "收盘"
            df[date_col] = pd.to_datetime(df[date_col])
            df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
            ret = df[close_col].pct_change().dropna()
            summary["stocks"].setdefault(code, {"name": name, "adjustments": {}})["adjustments"][label] = {
                "rows": int(len(df)), "start": str(df[date_col].min().date()), "end": str(df[date_col].max().date()),
                "close_start": float(df[close_col].iloc[0]), "close_end": float(df[close_col].iloc[-1]),
                "missing_close": int(df[close_col].isna().sum()), "max_abs_daily_return": float(ret.abs().max()),
                "jumps_over_30pct": int((ret.abs() > 0.30).sum()),
            }
            ax.plot(df[date_col], df[close_col], label=label, linewidth=0.8)
        ax.set_title(f"{code} {name}: raw / qfq / hfq")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "adjusted_price_check.png", dpi=160)
    (OUT / "market_data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
