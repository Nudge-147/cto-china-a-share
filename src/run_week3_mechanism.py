"""Run Week-3 mechanism audits: intraday split, decay, size control, and turnover.

Inputs: formal assignments, daily CTO prices, and monthly market caps.
Outputs: mechanism and decay CSVs under the formal-backtest directory.
Role: tests the proposed persistent overnight-selling mechanism.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest_cto import assign_deciles, month_end_returns, nw_mean_t, portfolio_returns


BASE = Path("data/cto_baostock")
FORMAL = BASE / "formal_backtest"
OUT = FORMAL / "week3"
NW_LAGS = 5
PERIODS = {"full_2010_2026": ("2010-01", "2026-12"), "paper_overlap_2010_2020": ("2010-01", "2020-12"), "out_of_sample_2021_2026": ("2021-01", "2026-12")}


def in_period(x: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    month = pd.PeriodIndex(x["holding_month"].astype(str), freq="M")
    return x[(month >= pd.Period(start, "M")) & (month <= pd.Period(end, "M"))]


def mechanism_panel(assignments: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = assignments[["code", "month", "decile"]].copy()
    a["month"] = pd.PeriodIndex(a["month"].astype(str), freq="M")
    a["holding_month"] = a["month"] + 1
    d = daily[["code", "date", "open", "close", "prev_close"]].copy()
    d["holding_month"] = d["date"].dt.to_period("M")
    x = a.merge(d, on=["code", "holding_month"], how="inner").sort_values(["code", "month", "date"])
    x["overnight_log"] = np.log(pd.to_numeric(x["open"], errors="coerce") / pd.to_numeric(x["prev_close"], errors="coerce"))
    x["intraday_log"] = np.log(pd.to_numeric(x["close"], errors="coerce") / pd.to_numeric(x["open"], errors="coerce"))
    keys = ["code", "month", "holding_month", "decile"]
    stock = x.groupby(keys, as_index=False).agg(overnight_log=("overnight_log", "sum"), intraday_log=("intraday_log", "sum"))
    stock["overnight_return"] = np.exp(stock["overnight_log"]) - 1
    stock["intraday_return"] = np.exp(stock["intraday_log"]) - 1
    stock["total_return"] = (1 + stock["overnight_return"]) * (1 + stock["intraday_return"]) - 1
    monthly = stock.groupby(["month", "holding_month", "decile"], as_index=False)[["overnight_return", "intraday_return", "total_return"]].mean()
    rows = []
    for label, (start, end) in PERIODS.items():
        z = in_period(monthly, start, end)
        for component in ("overnight_return", "intraday_return", "total_return"):
            pivot = z.pivot(index="holding_month", columns="decile", values=component)
            for decile in (1, 10):
                t, mean = nw_mean_t(pivot[decile], lags=NW_LAGS)
                rows.append({"period": label, "series": f"D{decile}", "component": component, "mean_monthly_return": mean, "newey_west_t_lag5": t})
            spread = pivot[10] - pivot[1]
            t, mean = nw_mean_t(spread, lags=NW_LAGS)
            rows.append({"period": label, "series": "D10-D1", "component": component, "mean_monthly_return": mean, "newey_west_t_lag5": t})
    # Holding-day decay from formation close (first holding-day prior close).
    x["holding_day"] = x.groupby(["code", "month"]).cumcount() + 1
    x["cum_return"] = x["close"] / x.groupby(["code", "month"])["prev_close"].transform("first") - 1
    decay_rows = []
    for horizon in (1, 5, 10, 20):
        z = x[x["holding_day"].eq(horizon)].groupby(["month", "holding_month", "decile"], as_index=False)["cum_return"].mean()
        pivot = z.pivot(index="holding_month", columns="decile", values="cum_return")
        s = pivot[10] - pivot[1]
        for label, (start, end) in PERIODS.items():
            q = s[(s.index >= pd.Period(start, "M")) & (s.index <= pd.Period(end, "M"))]
            t, mean = nw_mean_t(q, lags=NW_LAGS)
            decay_rows.append({"period": label, "horizon_trading_days": horizon, "mean_d10_minus_d1_return": mean, "newey_west_t_lag5": t, "months": q.notna().sum()})
    return pd.DataFrame(rows), pd.DataFrame(decay_rows)


def medium_cap_control(monthly: pd.DataFrame, daily: pd.DataFrame, caps: pd.DataFrame) -> pd.DataFrame:
    x = monthly.merge(caps[["code", "month", "market_cap"]], on=["code", "month"], how="inner")
    x["month"] = pd.PeriodIndex(x["month"].astype(str), freq="M")
    x["keep"] = x["market_cap"] >= x.groupby("month")["market_cap"].transform(lambda s: s.quantile(.30))
    a = assign_deciles(x[x["keep"]].drop(columns="keep"))
    r = month_end_returns(daily[["code", "date", "close"]])
    p = portfolio_returns(a, r, market_caps=None)
    rows = []
    for label, (start, end) in PERIODS.items():
        z = in_period(p, start, end).pivot(index="holding_month", columns="decile", values="ew_return")
        s = z[10] - z[1]
        t, mean = nw_mean_t(s, lags=NW_LAGS)
        rows.append({"period": label, "universe": "exclude_bottom_30pct_market_cap", "months": s.notna().sum(), "ew_d10_minus_d1": mean, "newey_west_t_lag5": t})
    return pd.DataFrame(rows)


def portfolio_turnover(assignments: pd.DataFrame) -> pd.DataFrame:
    a = assignments[["code", "month", "decile"]].copy()
    a["month"] = pd.PeriodIndex(a["month"].astype(str), freq="M")
    rows = []
    for decile, g in a.groupby("decile"):
        groups = {m: set(h.code) for m, h in g.groupby("month")}
        months = sorted(groups)
        for prev, current in zip(months, months[1:]):
            if current != prev + 1:
                continue
            overlap = len(groups[prev] & groups[current])
            rows.append({"month": current, "decile": decile, "stocks": len(groups[current]),
                         "overlap_with_prior_month": overlap, "one_way_turnover": 1 - overlap / len(groups[current])})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assignments = pd.read_csv(FORMAL / "assignments.csv", dtype={"code": str})
    daily = pd.read_csv(BASE / "daily_cto.csv", dtype={"code": str})
    daily["date"] = pd.to_datetime(daily["date"])
    monthly = pd.read_csv(BASE / "monthly_cto.csv", dtype={"code": str})
    caps = pd.read_csv(BASE / "market_caps_monthly.csv", dtype={"code": str})
    mechanism, decay = mechanism_panel(assignments, daily)
    mechanism.to_csv(OUT / "overnight_intraday_decomposition.csv", index=False)
    decay.to_csv(OUT / "decay_curve.csv", index=False)
    medium_cap_control(monthly, daily, caps).to_csv(OUT / "medium_cap_control.csv", index=False)
    turnover = portfolio_turnover(assignments)
    turnover.to_csv(OUT / "portfolio_turnover.csv", index=False)
    turnover.groupby("decile", as_index=False)["one_way_turnover"].mean().to_csv(OUT / "mean_portfolio_turnover.csv", index=False)
    print(mechanism.to_string(index=False))
    print(decay.to_string(index=False))


if __name__ == "__main__":
    main()
