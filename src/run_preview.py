"""Run a partial-universe CTO preview during downloading.

Inputs: currently completed Baostock raw/post-adjusted price pairs.
Outputs: preview CTO, market-cap, and backtest artifacts under ``data/cto_baostock/preview``.
Role: development smoke test only; it is never used as a formal result.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtest_cto import run
from cto_pipeline import build_cto
from market_cap_pipeline import build_market_caps, monthly_caps


BASE = Path("data/cto_baostock")


def available_pairs() -> list[str]:
    hfq = {p.stem for p in (BASE / "daily_hfq").glob("*.csv")}
    raw = {p.stem for p in (BASE / "daily_raw").glob("*.csv")}
    return sorted(hfq & raw)


def run_preview(threshold: int = 2000) -> bool:
    n = len(available_pairs())
    if n < threshold:
        print(f"Preview not run: {n}/{threshold} complete price pairs.")
        return False

    build_cto(
        source_dir=BASE / "daily_hfq",
        raw_dir=BASE / "daily_raw",
        output_base=BASE / "preview",
    )
    caps_daily = build_market_caps(BASE / "daily_raw", BASE / "preview" / "market_caps_daily.csv")
    caps_monthly = monthly_caps(caps_daily)
    caps_monthly.to_csv(BASE / "preview" / "market_caps_monthly.csv", index=False)

    monthly = pd.read_csv(BASE / "preview" / "monthly_cto.csv", dtype={"code": str})
    daily = pd.read_csv(BASE / "preview" / "daily_cto.csv", dtype={"code": str})
    assigned, portfolios, spread, summary = run(monthly, daily, caps_monthly)
    output = BASE / "preview" / "backtest"
    output.mkdir(parents=True, exist_ok=True)
    assigned.to_csv(output / "assignments.csv", index=False)
    portfolios.to_csv(output / "portfolio_returns.csv", index=False)
    spread.to_csv(output / "long_short.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    # Preview diagnostics only: do not interpret its incomplete-universe returns.
    monthly.groupby("month", as_index=False).agg(
        eligible_stocks=("code", "nunique"),
        eligible_stock_months=("code", "size"),
    ).to_csv(output / "monthly_eligible_stock_counts.csv", index=False)
    assigned.groupby(["month", "decile"], as_index=False).agg(
        stocks=("code", "nunique"),
    ).to_csv(output / "monthly_decile_counts.csv", index=False)
    print(f"Preview complete with {n} price pairs.\n{summary.to_string(index=False)}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=2000)
    args = parser.parse_args()
    run_preview(args.threshold)
