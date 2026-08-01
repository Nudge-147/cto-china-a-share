"""Monthly CTO portfolio backtest framework.

Formation: sort month-t CTO into deciles; hold the assigned stocks in t+1.
The framework handles small samples by exposing the effective number of bins,
rather than pretending three stocks form ten populated deciles.

Inputs: monthly CTO, daily adjusted close data, and optional monthly market cap.
Outputs: assignments, portfolio returns, long-short returns, and NW summaries.
Role: shared core used by the formal replication and downstream audits.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def nw_mean_t(x: pd.Series, lags: int = 5) -> tuple[float, float]:
    """Newey-West t-statistic for a time-series mean."""
    x = pd.Series(x).dropna().astype(float)
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    demeaned = x - x.mean()
    lr_var = (demeaned @ demeaned) / n
    for lag in range(1, min(lags, n - 1) + 1):
        cov = (demeaned.iloc[lag:].to_numpy() @ demeaned.iloc[:-lag].to_numpy()) / n
        lr_var += 2 * (1 - lag / (lags + 1)) * cov
    se = np.sqrt(lr_var / n)
    return float(x.mean() / se) if se else np.nan, float(x.mean())


def month_end_returns(daily: pd.DataFrame) -> pd.DataFrame:
    x = daily.copy()
    x["date"] = pd.to_datetime(x["date"])
    x["close"] = pd.to_numeric(x["close"], errors="coerce")
    x["month"] = x["date"].dt.to_period("M")
    last = x.sort_values("date").groupby(["code", "month"], as_index=False).tail(1)
    last = last.sort_values(["code", "month"])
    last["next_month_return"] = last.groupby("code")["close"].shift(-1) / last["close"] - 1
    return last[["code", "month", "next_month_return"]]


def assign_deciles(cto: pd.DataFrame) -> pd.DataFrame:
    x = cto.copy()
    x["month"] = pd.PeriodIndex(x["month"].astype(str), freq="M")
    x = x.dropna(subset=["cto_month"])
    x["rank"] = x.groupby("month")["cto_month"].rank(method="first")
    x["n_month"] = x.groupby("month")["code"].transform("size")
    # Full sample: ten deciles. Small samples: labels are still 1..10-scale,
    # while effective bins are explicitly retained for diagnostics.
    x["decile"] = np.ceil(10 * x["rank"] / x["n_month"]).astype(int)
    x["effective_bins"] = x.groupby("month")["decile"].transform("nunique")
    return x


def portfolio_returns(assignments: pd.DataFrame, returns: pd.DataFrame, market_caps: pd.DataFrame | None = None):
    x = assignments.merge(returns, on=["code", "month"], how="left").dropna(subset=["next_month_return"])
    if market_caps is not None:
        caps = market_caps.copy()
        caps["month"] = pd.PeriodIndex(caps["month"].astype(str), freq="M")
        x = x.merge(caps[["code", "month", "market_cap"]], on=["code", "month"], how="left")
    else:
        x["market_cap"] = np.nan
    market = x.groupby("month")["next_month_return"].mean().rename("ew_market_return")
    rows = []
    for (month, decile), g in x.groupby(["month", "decile"]):
        ew = g["next_month_return"].mean()
        if g["market_cap"].notna().all() and g["market_cap"].sum() > 0:
            vw = np.average(g["next_month_return"], weights=g["market_cap"])
            vw_available = True
        else:
            vw, vw_available = np.nan, False
        rows.append({"formation_month": month, "holding_month": month + 1, "decile": decile, "n": len(g),
                     "ew_return": ew, "ew_market_return": market.loc[month], "ew_market_adjusted": ew - market.loc[month],
                     "vw_return": vw, "vw_available": vw_available,
                     "effective_bins": g["effective_bins"].iloc[0]})
    return pd.DataFrame(rows)


def long_short(portfolios: pd.DataFrame):
    rows = []
    for month, g in portfolios.groupby("formation_month"):
        lo = g.loc[g["decile"].idxmin()]
        hi = g.loc[g["decile"].idxmax()]
        rows.append({"formation_month": month, "holding_month": month + 1,
                     "low_decile": lo.decile, "high_decile": hi.decile,
                     "effective_bins": hi.effective_bins,
                     "ew_long_short": hi.ew_return - lo.ew_return,
                     "vw_long_short": hi.vw_return - lo.vw_return if hi.vw_available and lo.vw_available else np.nan})
    out = pd.DataFrame(rows).sort_values("holding_month")
    out["ew_nav"] = (1 + out["ew_long_short"].fillna(0)).cumprod()
    out["vw_nav"] = (1 + out["vw_long_short"].fillna(0)).cumprod()
    return out


def run(monthly_cto, daily, market_caps=None, nw_lags: int = 5):
    assigned = assign_deciles(monthly_cto)
    returns = month_end_returns(daily)
    portfolios = portfolio_returns(assigned, returns, market_caps)
    spread = long_short(portfolios)
    ew_t, ew_mean = nw_mean_t(spread["ew_long_short"], lags=nw_lags)
    vw_t, vw_mean = nw_mean_t(spread["vw_long_short"], lags=nw_lags)
    summary = pd.DataFrame([{"series": "EW high-low", "months": spread["ew_long_short"].notna().sum(),
                             "mean_monthly_return": ew_mean, f"newey_west_t_lag{nw_lags}": ew_t},
                            {"series": "VW high-low", "months": spread["vw_long_short"].notna().sum(),
                             "mean_monthly_return": vw_mean, f"newey_west_t_lag{nw_lags}": vw_t}])
    return assigned, portfolios, spread, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--monthly-cto", default="data/cto/monthly_cto.csv")
    ap.add_argument("--daily-cto", default="data/cto/daily_cto.csv")
    ap.add_argument("--market-caps")
    ap.add_argument("--output", default="data/backtest")
    args = ap.parse_args()
    monthly = pd.read_csv(args.monthly_cto, dtype={"code": str})
    daily = pd.read_csv(args.daily_cto, dtype={"code": str})
    caps = pd.read_csv(args.market_caps, dtype={"code": str}) if args.market_caps else None
    assigned, portfolios, spread, summary = run(monthly, daily, caps)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    assigned.to_csv(out / "assignments.csv", index=False)
    portfolios.to_csv(out / "portfolio_returns.csv", index=False)
    spread.to_csv(out / "long_short.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))
