"""Construct daily float market capitalization from Baostock raw prices.

Baostock's daily turnover is a percentage. For traded observations,
float_shares = volume / (turn / 100), then float_market_cap = raw_close ×
float_shares. This uses unadjusted prices exclusively for weights.

Inputs: Baostock unadjusted daily price, volume, turnover, and trading-status CSVs.
Outputs: daily and month-end float-market-cap CSVs plus a smoothing diagnostic.
Role: supplies VW weights and size controls without mixing price vendors.
"""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path("data/cto_baostock")


def stepify_float_shares(raw_shares: pd.Series, window: int = 20, threshold: float = 0.05) -> pd.Series:
    """Remove rounding noise while retaining material float-share jumps."""
    smoothed = raw_shares.rolling(window=window, min_periods=1).median()
    stepped, current = [], None
    for candidate in smoothed:
        if pd.isna(candidate):
            stepped.append(current)
        elif current is None or abs(candidate / current - 1) > threshold:
            current = candidate
            stepped.append(current)
        else:
            stepped.append(current)
    return pd.Series(stepped, index=raw_shares.index, dtype="float64").ffill()


def build_market_caps(raw_dir=BASE / "daily_raw", output=BASE / "market_caps_daily.csv"):
    parts = []
    for p in sorted(Path(raw_dir).glob("*.csv")):
        d = pd.read_csv(p, dtype={"代码": str})
        if "换手率" not in d.columns:
            raise ValueError(f"{p} lacks 换手率; redownload with current baostock_pipeline.py")
        d["date"] = pd.to_datetime(d["日期"])
        d["raw_close"] = pd.to_numeric(d["收盘"], errors="coerce")
        d["volume"] = pd.to_numeric(d["成交量"], errors="coerce")
        d["turn_pct"] = pd.to_numeric(d["换手率"], errors="coerce")
        d["tradestatus"] = d["交易状态"].astype(str)
        d["float_shares_raw"] = d["volume"] / (d["turn_pct"] / 100)
        valid = d["tradestatus"].eq("1") & d["turn_pct"].ge(0.01) & d["float_shares_raw"].gt(0)
        d.loc[~valid, "float_shares_raw"] = pd.NA
        d["float_shares"] = stepify_float_shares(d["float_shares_raw"])
        d["float_market_cap"] = d["raw_close"] * d["float_shares"]
        d["code"] = p.stem
        parts.append(d[["date", "code", "raw_close", "volume", "turn_pct", "float_shares_raw", "float_shares", "float_market_cap"]])
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out.to_csv(output, index=False)
    return out


def build_market_caps_stream(raw_dir=BASE / "daily_raw", output=BASE / "market_caps_daily.csv"):
    """Write daily caps incrementally; retain only monthly endpoints in memory."""
    output = Path(output)
    temp = output.with_suffix(output.suffix + ".tmp")
    monthly_parts, first = [], True
    for p in sorted(Path(raw_dir).glob("*.csv")):
        d = pd.read_csv(p, dtype={"代码": str})
        if "换手率" not in d.columns:
            raise ValueError(f"{p} lacks 换手率; redownload with current baostock_pipeline.py")
        d["date"] = pd.to_datetime(d["日期"])
        d["raw_close"] = pd.to_numeric(d["收盘"], errors="coerce")
        d["volume"] = pd.to_numeric(d["成交量"], errors="coerce")
        d["turn_pct"] = pd.to_numeric(d["换手率"], errors="coerce")
        d["tradestatus"] = d["交易状态"].astype(str)
        d["float_shares_raw"] = d["volume"] / (d["turn_pct"] / 100)
        valid = d["tradestatus"].eq("1") & d["turn_pct"].ge(0.01) & d["float_shares_raw"].gt(0)
        d.loc[~valid, "float_shares_raw"] = pd.NA
        d["float_shares"] = stepify_float_shares(d["float_shares_raw"])
        d["float_market_cap"] = d["raw_close"] * d["float_shares"]
        d["code"] = p.stem
        daily = d[["date", "code", "raw_close", "volume", "turn_pct", "float_shares_raw", "float_shares", "float_market_cap"]]
        daily.to_csv(temp, mode="w" if first else "a", header=first, index=False)
        first = False
        month_end = daily.assign(month=daily["date"].dt.to_period("M")).groupby("month", as_index=False).tail(1)
        monthly_parts.append(month_end[["code", "month", "float_market_cap"]])
    temp.replace(output)
    monthly = pd.concat(monthly_parts, ignore_index=True) if monthly_parts else pd.DataFrame()
    return monthly.rename(columns={"float_market_cap": "market_cap"})


def plot_share_estimates(daily_caps, code, output=BASE / "outputs" / "float_shares_raw_vs_step.png"):
    x = daily_caps[daily_caps["code"].eq(code)].sort_values("date")
    if x.empty:
        raise ValueError(f"No market-cap observations for {code}")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x["date"], x["float_shares_raw"] / 1e8, alpha=.35, lw=.6, label="raw inferred")
    ax.step(x["date"], x["float_shares"] / 1e8, where="post", lw=1.2, label="20d median + 5% step")
    ax.set_title(f"{code}: inferred float shares")
    ax.set_ylabel("100 million shares")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)


def monthly_caps(daily_caps):
    x = daily_caps.copy()
    x["month"] = x["date"].dt.to_period("M")
    return x.sort_values("date").groupby(["code", "month"], as_index=False).tail(1)[["code", "month", "float_market_cap"]].rename(columns={"float_market_cap": "market_cap"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(BASE / "daily_raw"))
    ap.add_argument("--output", default=str(BASE / "market_caps_daily.csv"))
    ap.add_argument("--plot-code", default="603157")
    args = ap.parse_args()
    monthly = build_market_caps_stream(args.raw_dir, args.output)
    monthly.to_csv(BASE / "market_caps_monthly.csv", index=False)
    print(monthly.groupby("code")["market_cap"].last().sort_values(ascending=False).head(10).to_string())
