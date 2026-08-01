"""Run Week-3 implementation stress tests.

Inputs: formal portfolios, raw prices, and turnover-derived market-cap data.
Outputs: cost/tradability CSVs and the implementability waterfall figure.
Role: translates the academic gross return into an execution-aware lower bound.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest_cto import month_end_returns, nw_mean_t
from cto_pipeline import limit_pct


BASE = Path("data/cto_baostock")
FORMAL = BASE / "formal_backtest"
WEEK3 = FORMAL / "week3"
OUT = WEEK3 / "implementation"
NW_LAGS = 5
PERIODS = {"full_2010_2026": ("2010-01", "2026-12"), "paper_overlap_2010_2020": ("2010-01", "2020-12"), "out_of_sample_2021_2026": ("2021-01", "2026-12")}


def summarize(x: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    months = pd.PeriodIndex(x["holding_month"].astype(str), freq="M")
    for label, (start, end) in PERIODS.items():
        s = x.loc[(months >= pd.Period(start, "M")) & (months <= pd.Period(end, "M")), column]
        t, mean = nw_mean_t(s, lags=NW_LAGS)
        rows.append({"period": label, "months": s.notna().sum(), "mean_monthly_return": mean, "newey_west_t_lag5": t})
    return pd.DataFrame(rows)


def cost_stress() -> pd.DataFrame:
    spread = pd.read_csv(FORMAL / "long_short_monthly.csv")
    turnover = pd.read_csv(WEEK3 / "portfolio_turnover.csv")
    turnover["month"] = turnover["month"].astype(str)
    turn = turnover[turnover["decile"].isin([1, 10])].pivot(index="month", columns="decile", values="one_way_turnover").rename(columns={1: "d1_turnover", 10: "d10_turnover"})
    spread["formation_month"] = spread["formation_month"].astype(str)
    x = spread.merge(turn, left_on="formation_month", right_index=True, how="left")
    # One-way turnover is the fraction replaced. Each replacement generates a
    # sell and a buy; both long and short legs pay the stated one-side cost.
    x["two_leg_buy_sell_turnover"] = 2 * (x["d1_turnover"] + x["d10_turnover"])
    for bps in (10, 15, 25):
        cost = bps / 10_000
        x[f"cost_{bps}bps"] = x["two_leg_buy_sell_turnover"] * cost
        for weighting in ("ew", "vw"):
            x[f"{weighting}_net_{bps}bps"] = x[f"{weighting}_long_short"] - x[f"cost_{bps}bps"]
    rows = []
    for bps in (10, 15, 25):
        for weighting in ("EW", "VW"):
            q = summarize(x, f"{weighting.lower()}_net_{bps}bps")
            q["weighting"] = weighting; q["one_side_cost_bps"] = bps
            rows.append(q)
    result = pd.concat(rows, ignore_index=True)
    x.to_csv(OUT / "cost_stress_monthly.csv", index=False)
    result.to_csv(OUT / "cost_stress_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(13, 5))
    x = x.sort_values("holding_month")
    for bps, color in zip((0, 10, 15, 25), ("black", "tab:green", "tab:orange", "tab:red")):
        series = x["ew_long_short"] if bps == 0 else x[f"ew_net_{bps}bps"]
        ax.plot(pd.PeriodIndex(x["holding_month"].astype(str), freq="M").to_timestamp(), (1 + series.fillna(0)).cumprod(), label="EW gross" if bps == 0 else f"EW net, {bps} bps/side", color=color)
    ax.set_yscale("log"); ax.set_title("CTO long-short EW NAV after turnover costs"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(OUT / "ew_cost_net_nav.png", dpi=160); plt.close(fig)
    return result


def rebalance_open_limits(assignments: pd.DataFrame) -> pd.DataFrame:
    a = assignments[assignments["decile"].isin([1, 10])][["code", "month", "decile"]].copy()
    a["month"] = pd.PeriodIndex(a["month"].astype(str), freq="M")
    a["holding_month"] = a["month"] + 1
    wanted = {code: set(g.holding_month) for code, g in a.groupby("code")}
    parts = []
    for path in (BASE / "daily_raw").glob("*.csv"):
        code = path.stem
        if code not in wanted:
            continue
        d = pd.read_csv(path, usecols=["日期", "开盘", "前收盘", "是否ST"])
        d["date"] = pd.to_datetime(d["日期"]); d["holding_month"] = d["date"].dt.to_period("M")
        d = d[d["holding_month"].isin(wanted[code])].sort_values("date").groupby("holding_month", as_index=False).head(1)
        if d.empty:
            continue
        d["raw_open"] = pd.to_numeric(d["开盘"], errors="coerce")
        d["raw_prev_close"] = pd.to_numeric(d["前收盘"], errors="coerce")
        st = d["是否ST"].astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes", "y"])
        limits = [limit_pct(code, date, is_st) for date, is_st in zip(d["date"], st)]
        up = (d["raw_prev_close"] * (1 + pd.Series(limits, index=d.index))).round(2)
        down = (d["raw_prev_close"] * (1 - pd.Series(limits, index=d.index))).round(2)
        d["open_at_limit"] = d["raw_open"].sub(up).abs().le(.005) | d["raw_open"].sub(down).abs().le(.005)
        d["code"] = code
        parts.append(d[["code", "holding_month", "date", "open_at_limit"]])
    return pd.concat(parts, ignore_index=True)


def tradability_stress() -> pd.DataFrame:
    a = pd.read_csv(FORMAL / "assignments.csv", dtype={"code": str})
    a = a[a["decile"].isin([1, 10])].copy(); a["month"] = pd.PeriodIndex(a["month"].astype(str), freq="M")
    a["holding_month"] = a["month"] + 1
    limits = rebalance_open_limits(a)
    a = a.merge(limits, on=["code", "holding_month"], how="left")
    d = pd.read_csv(BASE / "daily_cto.csv", usecols=["code", "date", "close"], dtype={"code": str})
    returns = month_end_returns(d)
    x = a.merge(returns, on=["code", "month"], how="left").dropna(subset=["next_month_return"])
    x["open_at_limit"] = x["open_at_limit"].fillna(False)
    exclusion = x.groupby(["month", "decile"], as_index=False).agg(stocks=("code", "size"), excluded_open_limit=("open_at_limit", "sum"))
    exclusion["excluded_pct"] = exclusion["excluded_open_limit"] / exclusion["stocks"]
    p = x[~x["open_at_limit"]].groupby(["month", "holding_month", "decile"], as_index=False).agg(return_=("next_month_return", "mean"))
    pivot = p.pivot(index=["month", "holding_month"], columns="decile", values="return_").reset_index()
    pivot["ew_long_short_open_tradable"] = pivot[10] - pivot[1]
    result = summarize(pivot, "ew_long_short_open_tradable")
    exclusion.to_csv(OUT / "rebalance_open_limit_exclusions.csv", index=False)
    pivot.to_csv(OUT / "rebalance_open_tradable_long_short.csv", index=False)
    result.to_csv(OUT / "rebalance_open_tradable_summary.csv", index=False)
    return result


def combined_cost_and_tradability() -> pd.DataFrame:
    gross = pd.read_csv(FORMAL / "long_short_monthly.csv")
    costs = pd.read_csv(OUT / "cost_stress_monthly.csv")
    tradable = pd.read_csv(OUT / "rebalance_open_tradable_long_short.csv")
    tradable = tradable.rename(columns={"month": "formation_month"})
    for x in (gross, costs, tradable):
        x["formation_month"] = x["formation_month"].astype(str)
        x["holding_month"] = x["holding_month"].astype(str)
    x = tradable.merge(costs[["formation_month", "holding_month", "cost_10bps", "cost_15bps", "cost_25bps"]], on=["formation_month", "holding_month"], how="left")
    rows = []
    for bps in (10, 15, 25):
        column = f"net_after_open_limit_{bps}bps"
        x[column] = x["ew_long_short_open_tradable"] - x[f"cost_{bps}bps"]
        q = summarize(x, column)
        q["one_side_cost_bps"] = bps
        rows.append(q)
    summary = pd.concat(rows, ignore_index=True)
    x.to_csv(OUT / "combined_open_limit_cost_monthly.csv", index=False)
    summary.to_csv(OUT / "combined_open_limit_cost_summary.csv", index=False)

    # Full-sample waterfall: sequentially apply the two independent frictions.
    gross_mean = gross["ew_long_short"].mean()
    tradable_mean = x["ew_long_short_open_tradable"].mean()
    values = [gross_mean, tradable_mean, tradable_mean - x["cost_10bps"].mean(), tradable_mean - x["cost_15bps"].mean(), tradable_mean - x["cost_25bps"].mean()]
    labels = ["Gross", "Open-limit\ntradable", "Net\n10 bps", "Net\n15 bps", "Net\n25 bps"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(labels, [v * 100 for v in values], color=["tab:blue", "tab:orange", "tab:green", "tab:olive", "tab:red"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .025, f"{value * 100:.2f}%", ha="center", va="bottom")
    ax.set_ylabel("Mean monthly return (%)")
    ax.set_title("CTO EW long-short: academic gross to implementable net")
    ax.axhline(0, color="black", lw=.8); ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(OUT / "implementability_waterfall.png", dpi=180); plt.close(fig)
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(cost_stress().to_string(index=False))
    print(tradability_stress().to_string(index=False))
    print(combined_cost_and_tradability().to_string(index=False))


if __name__ == "__main__":
    main()
