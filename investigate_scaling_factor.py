"""Test whether an adjustment factor multiplicatively contaminated minute bars."""
from __future__ import annotations

import pandas as pd

from config import (
    CTO_DAILY_RAW_DIR, FACTOR_CHANGE_TOLERANCE, FACTOR_EXPLANATION_REQUIRED_SHARE,
    FACTOR_RATIO_MATCH_TOLERANCE, FACTOR_STEP_CV_TOLERANCE, RAW_5MIN_DIR,
    SCALING_FACTOR_END_DATE, SCALING_FACTOR_START_DATE, SCALING_INVESTIGATION_DIR,
    STOCK_LIST_PATH,
)


CTO_DAILY_HFQ_DIR = CTO_DAILY_RAW_DIR.parent / "daily_hfq"
RECONCILIATION_PATH = RAW_5MIN_DIR.parent / "qc_report" / "reconciliation_daily.csv"


def load_codes(limit: int) -> list[str]:
    """Read the tested stock prefix."""
    return pd.read_csv(STOCK_LIST_PATH, dtype=str)["code"].head(limit).tolist()


def adjustment_factors(code: str) -> pd.DataFrame:
    """Derive the CTO adjustment-factor sequence from HFQ/raw close pairs."""
    stem = code.split(".", 1)[1]
    raw = pd.read_csv(CTO_DAILY_RAW_DIR / f"{stem}.csv", encoding="utf-8-sig",
                      usecols=["日期", "收盘"])
    hfq = pd.read_csv(CTO_DAILY_HFQ_DIR / f"{stem}.csv", encoding="utf-8-sig",
                      usecols=["日期", "收盘"])
    frame = hfq.merge(raw, on="日期", suffixes=("_hfq", "_raw"))
    frame["date"] = pd.to_datetime(frame["日期"])
    frame["adjustment_factor"] = frame["收盘_hfq"] / frame["收盘_raw"]
    return frame[["date", "adjustment_factor"]]


def load_ratios(codes: list[str]) -> pd.DataFrame:
    """Load the suspect regime and attach adjustment factors."""
    frame = pd.read_csv(RECONCILIATION_PATH, parse_dates=["date"])
    frame = frame[frame["code"].isin(codes) & frame["date"].between(
        SCALING_FACTOR_START_DATE, SCALING_FACTOR_END_DATE)].copy()
    frame["volume_ratio"] = frame["minute_volume"] / frame["daily_volume"]
    frame["amount_ratio"] = frame["minute_amount"] / frame["daily_amount"]
    frame["close_ratio"] = frame["minute_close"] / frame["daily_close"]
    factors = [adjustment_factors(code).assign(code=code) for code in codes]
    return frame.merge(pd.concat(factors, ignore_index=True), on=["date", "code"], how="left")


def add_factor_tests(group: pd.DataFrame) -> pd.DataFrame:
    """Add normalized transforms, factor segments, and shared-scale tests."""
    group = group.sort_values("date").copy()
    base = group["adjustment_factor"].iloc[0]
    group["factor_normalized"] = group["adjustment_factor"] / base
    group["factor_inverse_normalized"] = base / group["adjustment_factor"]
    changed = group["adjustment_factor"].pct_change().abs() > FACTOR_CHANGE_TOLERANCE
    group["factor_change"] = changed.fillna(False)
    group["factor_segment"] = group["factor_change"].cumsum()
    group["volume_close_shared_scale"] = (
        group["volume_ratio"] - group["close_ratio"]
    ).abs() <= FACTOR_RATIO_MATCH_TOLERANCE
    return group


def segment_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure whether volume ratios are constant within factor plateaus."""
    return frame.groupby(["code", "factor_segment"], as_index=False).agg(
        segment_start=("date", "min"), segment_end=("date", "max"), days=("date", "size"),
        adjustment_factor=("adjustment_factor", "first"), volume_ratio_mean=("volume_ratio", "mean"),
        volume_ratio_std=("volume_ratio", "std"), unique_volume_ratios=("volume_ratio", "nunique"),
    ).assign(step_cv=lambda x: x["volume_ratio_std"] / x["volume_ratio_mean"])


def stock_summary(frame: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """Decide whether any known factor transform explains each stock."""
    rows: list[dict[str, object]] = []
    for code, group in frame.groupby("code"):
        own = segments[segments["code"].eq(code)]
        normal_mae = (group["volume_ratio"] - group["factor_normalized"]).abs().median()
        inverse_mae = (group["volume_ratio"] - group["factor_inverse_normalized"]).abs().median()
        shared = group["volume_close_shared_scale"].mean()
        step_cv = own["step_cv"].median()
        rows.append({"code": code, "days": len(group), "factor_changes": int(group["factor_change"].sum()),
                     "volume_ratio_median": group["volume_ratio"].median(),
                     "volume_ratio_std": group["volume_ratio"].std(),
                     "amount_volume_ratio_corr": group["amount_ratio"].corr(group["volume_ratio"]),
                     "normalized_factor_median_abs_error": normal_mae,
                     "inverse_factor_median_abs_error": inverse_mae,
                     "median_within_step_cv": step_cv, "shared_volume_close_scale_share": shared,
                     "factor_explains": min(normal_mae, inverse_mae) <= FACTOR_RATIO_MATCH_TOLERANCE
                     and step_cv <= FACTOR_STEP_CV_TOLERANCE
                     and shared >= FACTOR_EXPLANATION_REQUIRED_SHARE})
    return pd.DataFrame(rows)


def overall_decision(summary: pd.DataFrame) -> pd.DataFrame:
    """Produce the one-row repair-path decision."""
    explained = int(summary["factor_explains"].sum())
    return pd.DataFrame([{
        "stocks_tested": len(summary), "stocks_factor_explained": explained,
        "factor_explained_share": explained / len(summary),
        "known_adjustment_factor_hypothesis": "REJECTED" if explained == 0 else "SUPPORTED",
        "repair_path": "daily_total_proportional_rescale" if explained == 0 else "factor_inverse",
        "scale_gate_after_repair": "OPEN",
    }])


def main() -> None:
    raw = load_ratios(load_codes(10))
    ratios = pd.concat([add_factor_tests(group) for _, group in raw.groupby("code")],
                       ignore_index=True)
    segments = segment_statistics(ratios)
    summary = stock_summary(ratios, segments)
    tables = {"scaling_factor_daily": ratios, "scaling_factor_segments": segments,
              "scaling_factor_by_stock": summary, "scaling_factor_decision": overall_decision(summary)}
    SCALING_INVESTIGATION_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(SCALING_INVESTIGATION_DIR / f"{name}.csv", index=False, encoding="utf-8")
        print(f"\n{name}\n{table.head(20).to_string(index=False)}", flush=True)


if __name__ == "__main__":
    main()
