"""Audit delisting-month CTO returns.

Inputs: reconstructed CTO panels and the Baostock universe under ``data/``.
Outputs: compact audit CSVs under ``data/cto_baostock/formal_backtest/audits``.
Role: post-main-result check that delisting observations were not silently lost.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path("data/cto_baostock")
FORMAL = BASE / "formal_backtest"
OUT = FORMAL / "audits"


def month_end_with_next(daily: pd.DataFrame) -> pd.DataFrame:
    x = daily.copy()
    x["date"] = pd.to_datetime(x["date"])
    x["close"] = pd.to_numeric(x["close"], errors="coerce")
    x["month"] = x["date"].dt.to_period("M")
    last = x.sort_values("date").groupby(["code", "month"], as_index=False).tail(1)
    last = last.sort_values(["code", "month"])
    last["holding_end_date"] = last.groupby("code")["date"].shift(-1)
    last["next_month_return"] = last.groupby("code")["close"].shift(-1) / last["close"] - 1
    return last[["code", "month", "next_month_return", "holding_end_date"]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(FORMAL / "assignments.csv", dtype={"code": str})
    a["month"] = pd.PeriodIndex(a["month"].astype(str), freq="M")
    a["holding_month"] = a["month"] + 1
    d = pd.read_csv(BASE / "daily_cto.csv", usecols=["code", "date", "close"], dtype={"code": str})
    r = month_end_with_next(d)
    a = a.merge(r, on=["code", "month"], how="left")

    universe = pd.read_csv(BASE / "baostock_universe.csv", dtype=str).fillna("")
    universe["code"] = universe["code"].str.split(".").str[-1].str.zfill(6)
    universe["out_date"] = pd.to_datetime(universe["outDate"], errors="coerce")
    universe["out_month"] = universe["out_date"].dt.to_period("M")
    a = a.merge(universe[["code", "out_date", "out_month"]], on="code", how="left")
    a["delists_in_holding_month"] = a["out_month"].eq(a["holding_month"])
    a["delisted_before_holding_month"] = a["out_month"].lt(a["holding_month"])
    a["return_observed"] = a["next_month_return"].notna()
    a["holding_end_is_before_month_end"] = (
        a["holding_end_date"].notna()
        & a["holding_end_date"].dt.to_period("M").eq(a["holding_month"])
        & a["holding_end_date"].dt.is_month_end.eq(False)
    )

    # Exclude the terminal formation month, which necessarily lacks a future return.
    analyzable = a[a["holding_month"] <= pd.Period("2026-06", "M")].copy()
    event = analyzable[analyzable["delists_in_holding_month"]].copy()
    yearly = event.assign(year=event["holding_month"].dt.year).groupby(["year", "decile"], as_index=False).agg(
        delisting_stock_months=("code", "size"),
        returns_observed=("return_observed", "sum"),
        partial_month_end_quotes=("holding_end_is_before_month_end", "sum"),
        mean_return=("next_month_return", "mean"),
    )
    summary = event.groupby("decile", as_index=False).agg(
        delisting_stock_months=("code", "size"),
        returns_observed=("return_observed", "sum"),
        dropped_missing_return=("return_observed", lambda s: (~s).sum()),
        partial_month_end_quotes=("holding_end_is_before_month_end", "sum"),
        mean_return=("next_month_return", "mean"),
        median_return=("next_month_return", "median"),
    )
    d1_examples = event[event["decile"].eq(1)].sort_values("holding_month")[[
        "code", "month", "holding_month", "out_date", "holding_end_date", "next_month_return", "return_observed", "holding_end_is_before_month_end"
    ]]
    # All D1 stock-months with no return, separated from normal terminal-date loss.
    d1_missing = analyzable[(analyzable["decile"].eq(1)) & (~analyzable["return_observed"])][[
        "code", "month", "holding_month", "out_date", "out_month", "delisted_before_holding_month"
    ]]

    summary.to_csv(OUT / "delisting_return_summary_by_decile.csv", index=False)
    yearly.to_csv(OUT / "delisting_events_by_year_decile.csv", index=False)
    d1_examples.to_csv(OUT / "d1_delisting_holding_month_examples.csv", index=False)
    d1_missing.to_csv(OUT / "d1_missing_next_return_observations.csv", index=False)
    print(summary.to_string(index=False))
    print("D1 missing returns", len(d1_missing))


if __name__ == "__main__":
    main()
