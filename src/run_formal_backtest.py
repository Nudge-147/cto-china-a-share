"""Run formal CTO decile results for full, overlap, and out-of-sample periods.

Inputs: eligible monthly/daily CTO panels and monthly float market caps.
Outputs: decile/long-short CSVs and figures under ``data/cto_baostock/formal_backtest``.
Role: main paper-replication result stage.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtest_cto import nw_mean_t, run


BASE = Path("data/cto_baostock")
OUT = BASE / "formal_backtest"
NW_LAGS = 5  # Paper's portfolio tables use Newey-West standard errors with five lags.
PERIODS = {
    "full_2010_2026": ("2010-01", "2026-12"),
    "paper_overlap_2010_2020": ("2010-01", "2020-12"),
    "out_of_sample_2021_2026": ("2021-01", "2026-12"),
}


def segment(x: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    month = pd.PeriodIndex(x["holding_month"].astype(str), freq="M")
    return x[(month >= pd.Period(start, "M")) & (month <= pd.Period(end, "M"))].copy()


def decile_summary(portfolios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, (start, end) in PERIODS.items():
        x = segment(portfolios, start, end)
        for weighting, column in (("EW", "ew_return"), ("VW", "vw_return")):
            for decile, g in x.groupby("decile"):
                t, mean = nw_mean_t(g[column], lags=NW_LAGS)
                rows.append({"period": period, "weighting": weighting, "decile": decile,
                             "months": g[column].notna().sum(), "mean_monthly_return": mean,
                             "newey_west_t_lag5": t})
    return pd.DataFrame(rows)


def long_short_summary(spread: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, (start, end) in PERIODS.items():
        x = segment(spread, start, end)
        for weighting, column in (("EW", "ew_long_short"), ("VW", "vw_long_short")):
            t, mean = nw_mean_t(x[column], lags=NW_LAGS)
            rows.append({"period": period, "weighting": weighting, "months": x[column].notna().sum(),
                         "mean_monthly_return": mean, "newey_west_t_lag5": t})
    return pd.DataFrame(rows)


def make_plots(spread: pd.DataFrame, portfolios: pd.DataFrame) -> None:
    x = spread.copy()
    x["holding_month"] = pd.PeriodIndex(x["holding_month"].astype(str), freq="M").to_timestamp()
    x = x.sort_values("holding_month")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x["holding_month"], x["ew_nav"], label="EW D10-D1", lw=1.1)
    ax.plot(x["holding_month"], x["vw_nav"], label="VW D10-D1", lw=1.1)
    ax.axvline(pd.Timestamp("2021-01-01"), color="black", lw=.9, ls="--", label="2021 out-of-sample")
    ax.set_yscale("log")
    ax.set_title("CTO long-short cumulative NAV (log scale)")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(OUT / "long_short_nav_log.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(x["holding_month"], x["ew_long_short"], lw=.75, label="EW D10-D1")
    axes[1].plot(x["holding_month"], x["vw_long_short"], lw=.75, label="VW D10-D1", color="tab:orange")
    for ax in axes:
        ax.axhline(0, color="black", lw=.7); ax.axvline(pd.Timestamp("2021-01-01"), color="black", lw=.8, ls="--")
        ax.legend(); ax.grid(alpha=.25)
    fig.suptitle("Monthly CTO long-short returns")
    fig.tight_layout(); fig.savefig(OUT / "long_short_monthly_returns.png", dpi=160); plt.close(fig)

    n = portfolios.copy()
    n["holding_month"] = pd.PeriodIndex(n["holding_month"].astype(str), freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(13, 5))
    for decile, g in n.groupby("decile"):
        ax.plot(g["holding_month"], g["n"], lw=.75, label=f"D{decile}")
    low = n[n["n"] < 30]
    ax.scatter(low["holding_month"], low["n"], color="red", s=10, label="n < 30", zorder=3)
    ax.axhline(30, color="red", lw=.8, ls="--")
    ax.set_title("Monthly stocks per CTO decile")
    ax.legend(ncol=6); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(OUT / "decile_stock_counts.png", dpi=160); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    monthly = pd.read_csv(BASE / "monthly_cto.csv", dtype={"code": str})
    daily = pd.read_csv(BASE / "daily_cto.csv", dtype={"code": str})
    caps = pd.read_csv(BASE / "market_caps_monthly.csv", dtype={"code": str})
    assigned, portfolios, spread, _ = run(monthly, daily, caps, nw_lags=NW_LAGS)
    deciles = decile_summary(portfolios)
    long_short = long_short_summary(spread)
    counts = portfolios[["formation_month", "holding_month", "decile", "n"]].copy()
    counts["below_30"] = counts["n"] < 30
    assigned.to_csv(OUT / "assignments.csv", index=False)
    portfolios.to_csv(OUT / "portfolio_returns.csv", index=False)
    spread.to_csv(OUT / "long_short_monthly.csv", index=False)
    counts.to_csv(OUT / "decile_stock_counts.csv", index=False)
    deciles.to_csv(OUT / "decile_mean_returns.csv", index=False)
    long_short.to_csv(OUT / "long_short_summary.csv", index=False)
    make_plots(spread, portfolios)
    print(deciles.to_string(index=False))
    print(long_short.to_string(index=False))


if __name__ == "__main__":
    main()
