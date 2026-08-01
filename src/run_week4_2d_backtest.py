"""Run frozen Week-4 label × CTO-quintile tests and pre-registered decay checks.

Inputs: frozen monthly stock labels, CTO panels, and next-month returns.
Outputs: cell counts, returns, Q5−Q1 tests, and decay diagnostics.
Role: preregistered information-versus-attention conditional analysis.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest_cto import month_end_returns, nw_mean_t


BASE = Path("data/cto_baostock")
OUT = BASE / "formal_backtest" / "week4" / "two_dimensional"
NW_LAGS = 5
LABELS = ["attention", "information", "baseline"]
QUINTILES = [1, 2, 3, 4, 5]
HORIZONS = [1, 5, 10, 20]


def assign_quintiles() -> pd.DataFrame:
    cto = pd.read_csv(BASE / "monthly_cto.csv", dtype={"code": str})
    labels = pd.read_csv(BASE / "formal_backtest" / "week4" / "monthly_stock_labels.csv", dtype={"code": str})
    for x in (cto, labels):
        x["code"] = x["code"].str.zfill(6)
        x["month"] = pd.PeriodIndex(x["month"].astype(str), freq="M")
    x = cto.merge(labels[["code", "month", "stock_label"]], on=["code", "month"], how="left")
    x["stock_label"] = x["stock_label"].fillna("baseline")
    x = x[x["stock_label"].isin(LABELS)].dropna(subset=["cto_month"]).copy()
    x["rank"] = x.groupby(["month", "stock_label"])["cto_month"].rank(method="first")
    x["n_assigned"] = x.groupby(["month", "stock_label"])["code"].transform("size")
    x["quintile"] = np.ceil(5 * x["rank"] / x["n_assigned"]).astype(int)
    return x[["code", "month", "stock_label", "quintile", "n_assigned"]]


def return_panel(assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(BASE / "daily_cto.csv", usecols=["code", "date", "close", "prev_close"], dtype={"code": str})
    daily["code"] = daily["code"].str.zfill(6)
    daily["date"] = pd.to_datetime(daily["date"])
    returns = month_end_returns(daily[["code", "date", "close"]])
    x = assignments.merge(returns, on=["code", "month"], how="left").dropna(subset=["next_month_return"])
    portfolio = (x.groupby(["month", "stock_label", "quintile"], as_index=False)
                 .agg(ew_return=("next_month_return", "mean"), n=("code", "size")))
    portfolio["holding_month"] = portfolio["month"] + 1
    return portfolio, daily


def summarize_cells(portfolio: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for label in LABELS:
        for q in QUINTILES:
            x = portfolio[(portfolio.stock_label == label) & (portfolio.quintile == q)]
            t, mean = nw_mean_t(x["ew_return"], lags=NW_LAGS)
            rows.append({"stock_label": label, "quintile": q, "months": x.ew_return.notna().sum(),
                         "mean_monthly_return": mean, "newey_west_t_lag5": t,
                         "mean_stocks": x.n.mean(), "min_stocks": x.n.min(), "months_n_lt_10": (x.n < 10).sum()})
    cells = pd.DataFrame(rows)
    count_audit = (portfolio.groupby(["stock_label", "quintile"], as_index=False)["n"].agg(
        months="size", mean_stocks="mean", p10_stocks=lambda s: s.quantile(.10), min_stocks="min",
        months_n_lt_10=lambda s: (s < 10).sum(), months_n_lt_30=lambda s: (s < 30).sum()))
    return cells, count_audit


def long_short_and_contrast(portfolio: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = portfolio.pivot(index=["month", "holding_month", "stock_label"], columns="quintile", values="ew_return").reset_index()
    p["q5_minus_q1"] = p[5] - p[1]
    rows = []
    for label, g in p.groupby("stock_label"):
        t, mean = nw_mean_t(g["q5_minus_q1"], lags=NW_LAGS)
        rows.append({"series": f"{label}_Q5-Q1", "stock_label": label, "months": g.q5_minus_q1.notna().sum(),
                     "mean_monthly_return": mean, "newey_west_t_lag5": t})
    wide = p.pivot(index="holding_month", columns="stock_label", values="q5_minus_q1")
    contrast = wide["attention"] - wide["information"]
    t, mean = nw_mean_t(contrast, lags=NW_LAGS)
    rows.append({"series": "attention_minus_information_Q5-Q1", "stock_label": "contrast",
                 "months": contrast.notna().sum(), "mean_monthly_return": mean, "newey_west_t_lag5": t})
    return p, pd.DataFrame(rows)


def decay(assignments: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = assignments.copy()
    a["holding_month"] = a["month"] + 1
    d = daily.copy(); d["holding_month"] = d["date"].dt.to_period("M")
    x = a.merge(d, on=["code", "holding_month"], how="inner").sort_values(["code", "month", "date"])
    x["holding_day"] = x.groupby(["code", "month"]).cumcount() + 1
    x["cum_return"] = pd.to_numeric(x["close"], errors="coerce") / pd.to_numeric(x.groupby(["code", "month"])["prev_close"].transform("first"), errors="coerce") - 1
    paths, rows = [], []
    for horizon in HORIZONS:
        z = x[x.holding_day.eq(horizon)].groupby(["month", "holding_month", "stock_label", "quintile"], as_index=False).agg(
            ew_cumulative_return=("cum_return", "mean"), n=("code", "size"))
        z["horizon_trading_days"] = horizon
        paths.append(z)
        pivot = z.pivot(index=["month", "holding_month", "stock_label"], columns="quintile", values="ew_cumulative_return").reset_index()
        pivot["q5_minus_q1"] = pivot[5] - pivot[1]
        for label, g in pivot.groupby("stock_label"):
            t, mean = nw_mean_t(g["q5_minus_q1"], lags=NW_LAGS)
            rows.append({"stock_label": label, "horizon_trading_days": horizon, "months": g.q5_minus_q1.notna().sum(),
                         "mean_q5_minus_q1_return": mean, "newey_west_t_lag5": t})
    all_paths = pd.concat(paths, ignore_index=True)
    # A horizon-specific curve has fewer observations at 20 days.  Recompute
    # it on the common 20-day-capable months before interpreting persistence.
    matched_rows = []
    for label in LABELS:
        h20 = all_paths[(all_paths.stock_label == label) & (all_paths.horizon_trading_days == 20)]
        h20_pivot = h20.pivot(index=["month", "holding_month"], columns="quintile", values="ew_cumulative_return")
        common_months = h20_pivot.index[h20_pivot[1].notna() & h20_pivot[5].notna()]
        for horizon in HORIZONS:
            z = all_paths[(all_paths.stock_label == label) & (all_paths.horizon_trading_days == horizon)]
            pivot = z.pivot(index=["month", "holding_month"], columns="quintile", values="ew_cumulative_return")
            s = (pivot[5] - pivot[1]).reindex(common_months)
            t, mean = nw_mean_t(s, lags=NW_LAGS)
            matched_rows.append({"stock_label": label, "horizon_trading_days": horizon, "months": s.notna().sum(),
                                 "mean_q5_minus_q1_return": mean, "newey_west_t_lag5": t,
                                 "sample": "matched_20day_capable_months"})
    summary = pd.DataFrame(rows)
    summary["sample"] = "horizon_specific_months"
    return all_paths, pd.concat([summary, pd.DataFrame(matched_rows)], ignore_index=True)


def make_count_plot(portfolio: pd.DataFrame) -> None:
    x = portfolio.copy(); x["date"] = x["holding_month"].dt.to_timestamp()
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for ax, label in zip(axes, LABELS):
        for q, g in x[x.stock_label.eq(label)].groupby("quintile"):
            ax.plot(g.date, g.n, lw=.7, label=f"Q{q}")
        ax.axhline(10, color="red", lw=.8, ls="--", label="n = 10")
        ax.set_title(f"{label}: realized stocks per CTO quintile")
        ax.legend(ncol=6); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "stocks_per_cell_timeseries.png", dpi=180); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assignments = assign_quintiles()
    assignments.to_csv(OUT / "assignments.csv", index=False)
    portfolio, daily = return_panel(assignments)
    portfolio.to_csv(OUT / "portfolio_returns.csv", index=False)
    cells, audit = summarize_cells(portfolio)
    cells.to_csv(OUT / "cell_return_summary.csv", index=False)
    audit.to_csv(OUT / "cell_count_audit.csv", index=False)
    spreads, spread_summary = long_short_and_contrast(portfolio)
    spreads.to_csv(OUT / "monthly_q5_minus_q1.csv", index=False)
    spread_summary.to_csv(OUT / "q5_minus_q1_summary.csv", index=False)
    decay_paths, decay_summary = decay(assignments, daily)
    decay_paths.to_csv(OUT / "decay_paths.csv", index=False)
    decay_summary.to_csv(OUT / "decay_summary.csv", index=False)
    make_count_plot(portfolio)
    print(cells.to_string(index=False))
    print(spread_summary.to_string(index=False))
    print(audit.to_string(index=False))
    print(decay_summary.to_string(index=False))


if __name__ == "__main__":
    main()
