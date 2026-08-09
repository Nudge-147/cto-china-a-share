"""Build leakage-safe per-bar features and cross-sectional labels."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    EXPECTED_BARS_PER_DAY, EXPECTED_BAR_TIMES, FEATURE_MIN_VOL_HISTORY_DAYS,
    FEATURE_VOL_LOOKBACK_DAYS,
    FEATURES_5MIN_DIR, LABEL_30M_BARS, LABEL_60M_BARS, LABEL_RANK_MIN_STOCKS,
    PROCESSED_5MIN_DIR, SAMPLE_BAR_TIMES, SEQUENCE_FEATURE_COLUMNS, STOCK_LIST_PATH,
)
from download_5min import atomic_parquet


IDENTIFIER_COLUMNS = ["code", "date", "time", "bar_index"]
LABEL_COLUMNS = ["fwd_ret_30m", "fwd_ret_60m", "label_rank"]


def load_codes(path: Path, limit: int | None) -> list[str]:
    """Load the canonical stock prefix."""
    codes = pd.read_csv(path, dtype=str)["code"].dropna().drop_duplicates().tolist()
    return codes[:limit] if limit else codes


def prior_same_bar_mean(volume: pd.Series, bar_index: pd.Series) -> pd.Series:
    """Use only prior trading days for the same bar position."""
    frame = pd.DataFrame({"volume": volume, "bar_index": bar_index})
    return frame.groupby("bar_index")["volume"].transform(
        lambda values: values.shift(1).rolling(
            FEATURE_VOL_LOOKBACK_DAYS, min_periods=FEATURE_MIN_VOL_HISTORY_DAYS).mean()
    )


def add_price_volume_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate causal per-bar price, volume, and time features."""
    frame = frame.sort_values("time").copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["time"] = pd.to_datetime(frame["time"])
    clock_to_index = {clock: index for index, clock in enumerate(EXPECTED_BAR_TIMES)}
    frame["bar_index"] = frame["time"].dt.strftime("%H:%M:%S").map(clock_to_index)
    if frame["bar_index"].isna().any():
        raise ValueError("processed input contains a non-canonical bar timestamp")
    frame["bar_index"] = frame["bar_index"].astype("int8")
    frame["is_first_bar"] = frame["bar_index"].eq(0).astype("int8")
    previous_close = frame["close"].shift(1)
    frame["ret_5m"] = np.log(frame["close"] / previous_close)
    frame["range_rel"] = (frame["high"] - frame["low"]) / previous_close
    spread = frame["high"] - frame["low"]
    frame["pos_in_bar"] = np.where(spread.eq(0), 0.5, (frame["close"] - frame["low"]) / spread)
    frame["vol_log"] = np.log1p(frame["volume"])
    baseline = prior_same_bar_mean(frame["volume"], frame["bar_index"])
    frame["vol_rel_20d"] = frame["volume"] / baseline.replace(0, np.nan)
    frame["ret_overnight"] = 0.0
    first = frame["is_first_bar"].eq(1)
    frame.loc[first, "ret_overnight"] = np.log(frame.loc[first, "open"] / previous_close[first])
    angle = 2 * np.pi * frame["bar_index"] / EXPECTED_BARS_PER_DAY
    frame["time_of_day_sin"], frame["time_of_day_cos"] = np.sin(angle), np.cos(angle)
    return frame.iloc[1:].copy()


def remove_suspension_placeholders(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove DN-010 zero-activity suspension bars before price transforms."""
    frame = frame.copy()
    counts = frame.groupby("date")["date"].transform("size")
    activity = frame.groupby("date")[["volume", "amount"]].transform("sum").sum(axis=1)
    placeholder = counts.eq(EXPECTED_BARS_PER_DAY) & activity.eq(0)
    days = int(frame.loc[placeholder, "date"].nunique())
    return frame.loc[~placeholder].copy(), days


def add_incomplete_day_flag(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag observed trading days that do not contain all 48 bars (DN-010)."""
    frame = frame.copy()
    counts = frame.groupby("date")["date"].transform("size")
    frame["incomplete_day_flag"] = counts.ne(EXPECTED_BARS_PER_DAY)
    return frame


def add_forward_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Add same-day forward returns without crossing the close."""
    frame = frame.copy()
    grouped = frame.groupby("date")["close"]
    frame["fwd_ret_30m"] = np.log(grouped.shift(-LABEL_30M_BARS) / frame["close"])
    frame["fwd_ret_60m"] = np.log(grouped.shift(-LABEL_60M_BARS) / frame["close"])
    frame["label_rank"] = np.nan
    return frame


def validate_feature_frame(frame: pd.DataFrame, code: str) -> None:
    """Fail on non-history-related NaNs and malformed days."""
    required = [column for column in SEQUENCE_FEATURE_COLUMNS if column != "vol_rel_20d"]
    missing = frame[required].isna().sum()
    if missing.sum():
        raise ValueError(f"{code} unexpected feature NaNs: {missing[missing.gt(0)].to_dict()}")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError(f"{code} contains infinite core features")
    if frame["bar_index"].gt(EXPECTED_BARS_PER_DAY - 1).any():
        raise ValueError(f"{code} contains an out-of-range bar index")


def compute_stock_features(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    """Build one stock's raw-valued feature and label table."""
    required = {"open", "high", "low", "close", "volume", "close_anomaly_flag"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{code} processed input missing columns: {sorted(missing)}")
    cleaned, _ = remove_suspension_placeholders(frame)
    result = add_forward_labels(add_price_volume_features(add_incomplete_day_flag(cleaned)))
    validate_feature_frame(result, code)
    keep = IDENTIFIER_COLUMNS + ["open", "high", "low", "close", "volume", "volume_raw",
                                 "volume_rescaled", "volume_scale_factor", "close_anomaly_flag",
                                 "incomplete_day_flag",
                                 *SEQUENCE_FEATURE_COLUMNS, *LABEL_COLUMNS]
    return result[keep]


def rank_one_timestamp(group: pd.DataFrame) -> pd.Series:
    """Normalize a valid cross-section's average ranks to [-1, 1]."""
    valid = group["fwd_ret_30m"].notna() & ~group["close_anomaly_flag"].astype(bool)
    valid &= ~group["incomplete_day_flag"].astype(bool)
    result = pd.Series(np.nan, index=group.index, dtype=float)
    if int(valid.sum()) < LABEL_RANK_MIN_STOCKS:
        return result
    ranks = group.loc[valid, "fwd_ret_30m"].rank(method="average")
    result.loc[valid] = 2 * (ranks - 1) / (len(ranks) - 1) - 1
    return result


def cross_sectional_ranks(paths: list[Path]) -> pd.DataFrame:
    """Build ranks only at the six configured downstream sampling times."""
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path, columns=["code", "date", "time", "fwd_ret_30m",
                                               "close_anomaly_flag", "incomplete_day_flag"])
        frame = frame[frame["time"].dt.strftime("%H:%M:%S").isin(SAMPLE_BAR_TIMES)]
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["label_rank"] = np.nan
    valid = combined["fwd_ret_30m"].notna()
    valid &= ~combined["close_anomaly_flag"].astype(bool)
    valid &= ~combined["incomplete_day_flag"].astype(bool)
    eligible = combined.loc[valid]
    grouped = eligible.groupby(["date", "time"])["fwd_ret_30m"]
    counts = grouped.transform("count")
    ranks = grouped.rank(method="average")
    labels = 2 * (ranks - 1) / (counts - 1) - 1
    combined.loc[eligible.index, "label_rank"] = labels.where(
        counts.ge(LABEL_RANK_MIN_STOCKS))
    return combined[["code", "date", "time", "label_rank"]]


def apply_ranks(paths: list[Path], ranks: pd.DataFrame) -> None:
    """Merge cross-sectional ranks back into per-stock feature files."""
    rank_by_code = {code: frame for code, frame in ranks.groupby("code", sort=False)}
    for path in paths:
        frame = pd.read_parquet(path).drop(columns="label_rank")
        own = rank_by_code.get(frame["code"].iloc[0], ranks.iloc[0:0])
        frame = frame.merge(own, on=["code", "date", "time"], how="left", validate="one_to_one")
        atomic_parquet(frame, path)


def finalize_features(paths: list[Path], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add cross-sectional ranks and refresh NaN reports."""
    apply_ranks(paths, cross_sectional_ranks(paths))
    detail, summary = nan_report(paths)
    detail.to_csv(output_dir / "nan_rate_by_stock.csv", index=False, encoding="utf-8")
    summary.to_csv(output_dir / "nan_rate_summary.csv", index=False, encoding="utf-8")
    return detail, summary


def nan_report(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report per-stock and overall NaN rates for all model fields."""
    rows: list[dict[str, object]] = []
    columns = [*SEQUENCE_FEATURE_COLUMNS, *LABEL_COLUMNS]
    for path in paths:
        frame = pd.read_parquet(path, columns=["code", *columns])
        for column in columns:
            rows.append({"code": frame["code"].iloc[0], "column": column,
                         "rows": len(frame), "nan_count": int(frame[column].isna().sum())})
    detail = pd.DataFrame(rows)
    detail["nan_rate"] = detail["nan_count"] / detail["rows"]
    summary = detail.groupby("column", as_index=False).agg(rows=("rows", "sum"),
        nan_count=("nan_count", "sum"))
    summary["nan_rate"] = summary["nan_count"] / summary["rows"]
    summary["expected_reason"] = summary["column"].map({
        "vol_rel_20d": "fewer_than_5_prior_same-time days",
        "fwd_ret_30m": "same-day horizon unavailable",
        "fwd_ret_60m": "same-day horizon unavailable",
        "label_rank": "non-sample time, small pool, masked day, or unavailable horizon",
    }).fillna("")
    return detail, summary


def build_features(codes: list[str], input_dir: Path, output_dir: Path) -> list[Path]:
    """Build per-stock files, add ranks, and write NaN reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    quality_rows: list[dict[str, object]] = []
    for code in codes:
        source = input_dir / f"{code}.parquet"
        if not source.exists():
            continue
        raw = pd.read_parquet(source)
        _, placeholder_days = remove_suspension_placeholders(raw)
        features = compute_stock_features(raw, code)
        path = output_dir / f"{code}.parquet"
        atomic_parquet(features, path)
        paths.append(path)
        quality_rows.append({"code": code, "input_rows": len(raw), "feature_rows": len(features),
            "suspension_placeholder_days_removed": placeholder_days,
            "incomplete_trading_days": int(features.loc[
                features["incomplete_day_flag"], "date"].nunique())})
        print(f"features {code}", flush=True)
    finalize_features(paths, output_dir)
    pd.DataFrame(quality_rows).to_csv(output_dir / "data_quality_filter_report.csv",
                                      index=False, encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stock-list", type=Path, default=STOCK_LIST_PATH)
    parser.add_argument("--input-dir", type=Path, default=PROCESSED_5MIN_DIR)
    parser.add_argument("--output-dir", type=Path, default=FEATURES_5MIN_DIR)
    parser.add_argument("--ranks-only", action="store_true")
    args = parser.parse_args()
    codes = load_codes(args.stock_list, args.limit)
    if args.ranks_only:
        paths = [args.output_dir / f"{code}.parquet" for code in codes
                 if (args.output_dir / f"{code}.parquet").exists()]
        finalize_features(paths, args.output_dir)
    else:
        paths = build_features(codes, args.input_dir, args.output_dir)
    print(pd.read_csv(args.output_dir / "nan_rate_summary.csv").to_string(index=False))
    print(f"feature files={len(paths)}")


if __name__ == "__main__":
    main()
