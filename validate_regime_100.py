"""Validate the previously observed volume-gap regimes on the expanded stock pool."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    EXPECTED_BARS_PER_DAY, PRECISION_REGIME_START_DATE, QC_REPORT_DIR,
    REGIME_MAIN_MIN_LARGE_GAP_SHARE, REGIME_MIN_STOCK_DAYS,
    REGIME_POST_MAX_LARGE_GAP_SHARE, REGIME_PRE_MAX_LARGE_GAP_SHARE,
    REGIME_REPORT_DIR, REGIME_START_SEARCH_WINDOW, REGIME_END_SEARCH_WINDOW,
    ROUNDING_LIKE_REGIME_START_DATE, ROUNDING_MAX_ERROR_PER_BAR,
    STOCK_LIST_PATH,
)


ROUNDING_BOUND = EXPECTED_BARS_PER_DAY * ROUNDING_MAX_ERROR_PER_BAR
PERIODS = {
    "pre": ("2020-01-01", "2023-08-08"),
    "main": (PRECISION_REGIME_START_DATE, "2025-03-19"),
    "post": (ROUNDING_LIKE_REGIME_START_DATE, "2025-12-31"),
}


def load_inputs(stock_list: Path, reconciliation: Path) -> pd.DataFrame:
    """Load the requested pool and calculate a common large-gap indicator."""
    codes = pd.read_csv(stock_list, dtype=str)["code"].dropna().drop_duplicates()
    frame = pd.read_csv(reconciliation, parse_dates=["date"])
    frame = frame[frame["code"].isin(codes)].copy()
    frame["signed_volume_gap"] = frame["daily_volume"] - frame["minute_volume"]
    frame["absolute_volume_gap"] = frame["signed_volume_gap"].abs()
    frame["large_gap"] = frame["absolute_volume_gap"].gt(ROUNDING_BOUND)
    return frame.sort_values(["code", "date"])


def period_metrics(group: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    """Summarize one fixed archive period for one stock."""
    selected = group[group["date"].between(start, end)]
    return {"days": len(selected), "large_gap_share": selected["large_gap"].mean(),
            "median_abs_gap": selected["absolute_volume_gap"].median(),
            "p95_relative_error": selected["volume_relative_error"].abs().quantile(0.95)}


def infer_transition(group: pd.DataFrame, start: str, end: str, rising: bool) -> object:
    """Locate the first 20-session rolling change in large-gap prevalence."""
    selected = group[group["date"].between(start, end)].copy()
    rolling = selected["large_gap"].rolling(REGIME_MIN_STOCK_DAYS,
                                            min_periods=REGIME_MIN_STOCK_DAYS).mean()
    hit = rolling.ge(REGIME_MAIN_MIN_LARGE_GAP_SHARE) if rising else rolling.le(
        REGIME_POST_MAX_LARGE_GAP_SHARE)
    positions = np.flatnonzero(hit.to_numpy())
    if not len(positions):
        return pd.NaT
    position = int(positions[0]) - REGIME_MIN_STOCK_DAYS + 1
    return selected.iloc[max(position, 0)]["date"]


def stock_row(code: str, group: pd.DataFrame) -> dict[str, object]:
    """Classify one stock while exempting listings without enough history."""
    metrics = {name: period_metrics(group, *bounds) for name, bounds in PERIODS.items()}
    enough = all(value["days"] >= REGIME_MIN_STOCK_DAYS for value in metrics.values())
    stable = enough and metrics["pre"]["large_gap_share"] <= REGIME_PRE_MAX_LARGE_GAP_SHARE
    stable &= metrics["main"]["large_gap_share"] >= REGIME_MAIN_MIN_LARGE_GAP_SHARE
    stable &= metrics["post"]["large_gap_share"] <= REGIME_POST_MAX_LARGE_GAP_SHARE
    row: dict[str, object] = {"code": code, "classification":
        "STABLE" if stable else ("INSUFFICIENT_HISTORY" if not enough else "ANOMALY")}
    for name, values in metrics.items():
        row.update({f"{name}_{key}": value for key, value in values.items()})
    row["inferred_start"] = infer_transition(group, "2023-01-01", "2024-03-31", True)
    row["inferred_end"] = infer_transition(group, "2025-01-01", "2025-12-31", False)
    return row


def validate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build per-stock, anomaly-only, and aggregate stability reports."""
    detail = pd.DataFrame([stock_row(code, group) for code, group in frame.groupby("code")])
    anomalies = detail[detail["classification"].ne("STABLE")].copy()
    summary = detail.groupby("classification", as_index=False).agg(stocks=("code", "size"))
    summary["share"] = summary["stocks"] / len(detail)
    return detail, anomalies, summary


def cross_section_boundaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect synchronized archive boundaries from cross-sectional gap prevalence."""
    daily = frame.groupby("date", as_index=False).agg(
        covered_stocks=("code", "nunique"), large_gap_share=("large_gap", "mean"))
    daily["share_change"] = daily["large_gap_share"].diff()
    rows: list[dict[str, object]] = []
    specifications = [
        ("regime_start", REGIME_START_SEARCH_WINDOW, "increase", PRECISION_REGIME_START_DATE),
        ("regime_end", REGIME_END_SEARCH_WINDOW, "decrease", ROUNDING_LIKE_REGIME_START_DATE),
    ]
    for label, bounds, direction, expected in specifications:
        selected = daily[daily["date"].between(*bounds)]
        index = selected["share_change"].idxmax() if direction == "increase" else \
            selected["share_change"].idxmin()
        row = daily.loc[index]
        rows.append({"boundary": label, "expected_date": expected,
            "observed_date": row["date"], "exact_match": row["date"].strftime("%Y-%m-%d") == expected,
            "covered_stocks": int(row["covered_stocks"]),
            "large_gap_share": row["large_gap_share"], "share_change": row["share_change"]})
    return daily, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock-list", type=Path, default=STOCK_LIST_PATH)
    parser.add_argument("--reconciliation", type=Path,
                        default=QC_REPORT_DIR / "reconciliation_daily.csv")
    parser.add_argument("--output-dir", type=Path, default=REGIME_REPORT_DIR)
    args = parser.parse_args()
    source = load_inputs(args.stock_list, args.reconciliation)
    detail, anomalies, summary = validate(source)
    daily, boundaries = cross_section_boundaries(source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "regime_by_stock.csv", index=False, encoding="utf-8")
    anomalies.to_csv(args.output_dir / "regime_anomalies.csv", index=False, encoding="utf-8")
    summary.to_csv(args.output_dir / "regime_summary.csv", index=False, encoding="utf-8")
    daily.to_csv(args.output_dir / "cross_section_daily.csv", index=False, encoding="utf-8")
    boundaries.to_csv(args.output_dir / "boundary_summary.csv", index=False, encoding="utf-8")
    print(summary.to_string(index=False))
    print(boundaries.to_string(index=False))
    print(f"anomalies={len(anomalies)}")


if __name__ == "__main__":
    main()
