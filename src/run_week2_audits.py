"""Run Week-2 CTO audits: tail shape, characteristics, drawdowns, and IPO control.

Inputs: formal assignments, daily CTO, market-cap, and raw-price data.
Outputs: audit CSVs under ``data/cto_baostock/formal_backtest/audits``.
Role: validates the interpretation of the baseline long-short result.
"""
from __future__ import annotations

from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd

from backtest_cto import month_end_returns, nw_mean_t, run


BASE = Path("data/cto_baostock")
FORMAL = BASE / "formal_backtest"
OUT = FORMAL / "audits"
NW_LAGS = 5


def as_bool(x: pd.Series) -> pd.Series:
    return x.astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes", "y"])


def d1_subbins(assignments: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    x = assignments[assignments["decile"].eq(1)].copy()
    x["month"] = pd.PeriodIndex(x["month"].astype(str), freq="M")
    x["d1_subbin"] = x.groupby("month")["cto_month"].rank(method="first", pct=True)
    x["d1_subbin"] = np.ceil(x["d1_subbin"] * 5).astype(int)
    x = x.merge(returns, on=["code", "month"], how="left")
    rows = []
    for subbin, g in x.groupby("d1_subbin"):
        t, mean = nw_mean_t(g["next_month_return"], lags=NW_LAGS)
        rows.append({"d1_subbin": subbin, "months": g.groupby("month")["next_month_return"].count().gt(0).sum(),
                     "mean_monthly_return": mean, "newey_west_t_lag5": t,
                     "mean_stocks_per_month": g.groupby("month").size().mean()})
    return pd.DataFrame(rows), x


def formation_profiles(assignments: pd.DataFrame) -> pd.DataFrame:
    select = assignments[assignments["decile"].isin([1, 5])][["code", "month", "decile"]].copy()
    select["month"] = pd.PeriodIndex(select["month"].astype(str), freq="M")
    caps = pd.read_csv(BASE / "market_caps_monthly.csv", dtype={"code": str})
    caps["month"] = pd.PeriodIndex(caps["month"].astype(str), freq="M")
    selected = select.merge(caps, on=["code", "month"], how="left")
    keys = {code: set(g.month) for code, g in select.groupby("code")}
    daily_parts = []
    for p in (BASE / "daily_raw").glob("*.csv"):
        code = p.stem
        if code not in keys:
            continue
        d = pd.read_csv(p, usecols=["日期", "换手率", "是否ST"])
        d["month"] = pd.to_datetime(d["日期"]).dt.to_period("M")
        d = d[d["month"].isin(keys[code])]
        if d.empty:
            continue
        d["code"] = code
        d["turnover"] = pd.to_numeric(d["换手率"], errors="coerce")
        d["is_st_day"] = as_bool(d["是否ST"])
        daily_parts.append(d[["code", "month", "turnover", "is_st_day"]])
    daily = pd.concat(daily_parts, ignore_index=True)
    chars = daily.groupby(["code", "month"], as_index=False).agg(
        mean_turnover_pct=("turnover", "mean"), st_any_in_month=("is_st_day", "max"))
    selected = selected.merge(chars, on=["code", "month"], how="left")
    rows = []
    for decile, g in selected.groupby("decile"):
        rows.append({"portfolio": f"D{decile}", "stock_months": len(g),
                     "mean_float_market_cap_rmb": g["market_cap"].mean(),
                     "median_float_market_cap_rmb": g["market_cap"].median(),
                     "mean_turnover_pct": g["mean_turnover_pct"].mean(),
                     "st_month_share": g["st_any_in_month"].mean()})
    return pd.DataFrame(rows)


def index_monthly_return(code: str) -> pd.DataFrame:
    rs = bs.query_history_k_data_plus(code, "date,close", start_date="2010-01-01", end_date="2026-07-22", frequency="d", adjustflag="3")
    if rs.error_code != "0":
        raise RuntimeError(f"{code}: {rs.error_msg}")
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    x = pd.DataFrame(rows, columns=["date", "close"])
    x["date"] = pd.to_datetime(x["date"]); x["close"] = pd.to_numeric(x["close"])
    x["month"] = x.date.dt.to_period("M")
    x = x.sort_values("date").groupby("month", as_index=False).tail(1).sort_values("month")
    x["return"] = x.close.pct_change()
    return x[["month", "return"]]


def drawdown_months(portfolios: pd.DataFrame, spread: pd.DataFrame) -> pd.DataFrame:
    x = spread.nsmallest(6, "ew_long_short").copy()
    x["formation_month"] = pd.PeriodIndex(x["formation_month"].astype(str), freq="M")
    x["holding_month"] = pd.PeriodIndex(x["holding_month"].astype(str), freq="M")
    legs = portfolios[portfolios["decile"].isin([1, 10])][["formation_month", "decile", "ew_return", "vw_return"]].copy()
    legs["formation_month"] = pd.PeriodIndex(legs["formation_month"].astype(str), freq="M")
    legs = legs.pivot(index="formation_month", columns="decile", values="ew_return").rename(columns={1: "d1_return", 10: "d10_return"})
    x = x.merge(legs, left_on="formation_month", right_index=True, how="left")
    for code, name in [("sh.000300", "csi300_return"), ("sh.000905", "csi500_return"), ("sz.399006", "chinext_return")]:
        idx = index_monthly_return(code).rename(columns={"return": name, "month": "holding_month"})
        x = x.merge(idx, on="holding_month", how="left")
    return x.sort_values("ew_long_short")


def ipo_first_day_control(daily: pd.DataFrame, caps: pd.DataFrame) -> pd.DataFrame:
    x = daily.copy()
    only_first_day = ~(x[["is_st", "suspended", "listing_first_day", "prev_close_at_limit"]].any(axis=1))
    alt = x[only_first_day].assign(month=x.loc[only_first_day, "date"].dt.to_period("M")).groupby(["code", "month"], as_index=False).agg(
        cto_month=("cto_daily", "mean"), valid_days=("cto_daily", "count"))
    _, _, spread, _ = run(alt, daily, caps, nw_lags=NW_LAGS)
    rows = []
    for period, start, end in [("full_2010_2026", "2010-01", "2026-12"), ("paper_overlap_2010_2020", "2010-01", "2020-12")]:
        month = pd.PeriodIndex(spread.holding_month.astype(str), freq="M")
        g = spread[(month >= pd.Period(start, "M")) & (month <= pd.Period(end, "M"))]
        t, mean = nw_mean_t(g.ew_long_short, lags=NW_LAGS)
        rows.append({"period": period, "ipo_filter": "first_day_only", "months": g.ew_long_short.notna().sum(),
                     "ew_d10_minus_d1": mean, "newey_west_t_lag5": t})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assignments = pd.read_csv(FORMAL / "assignments.csv", dtype={"code": str})
    portfolios = pd.read_csv(FORMAL / "portfolio_returns.csv")
    spread = pd.read_csv(FORMAL / "long_short_monthly.csv")
    daily = pd.read_csv(BASE / "daily_cto.csv", dtype={"code": str})
    daily["date"] = pd.to_datetime(daily["date"])
    caps = pd.read_csv(BASE / "market_caps_monthly.csv", dtype={"code": str})
    returns = month_end_returns(daily[["code", "date", "close"]])
    sub_summary, sub_detail = d1_subbins(assignments, returns)
    sub_summary.to_csv(OUT / "d1_subbin_returns.csv", index=False)
    sub_detail.to_csv(OUT / "d1_subbin_assignments.csv", index=False)
    formation_profiles(assignments).to_csv(OUT / "d1_vs_d5_profile.csv", index=False)
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(login.error_msg)
    try:
        drawdown_months(portfolios, spread).to_csv(OUT / "six_worst_ew_months.csv", index=False)
    finally:
        bs.logout()
    ipo_first_day_control(daily, caps).to_csv(OUT / "ipo_filter_control.csv", index=False)
    print(sub_summary.to_string(index=False))


if __name__ == "__main__":
    main()
