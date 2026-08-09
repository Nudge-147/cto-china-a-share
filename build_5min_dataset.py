"""Assemble leakage-audited sequence and flat datasets from per-bar features."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    FEATURES_5MIN_DIR, FLAT_CURRENT_FEATURE_COLUMNS, FLAT_DATASET_DIR,
    SAMPLE_BAR_TIMES, SEQ_DATASET_DIR, SEQUENCE_FEATURE_COLUMNS,
    SEQUENCE_LOOKBACK_BARS, STOCK_LIST_PATH,
)


FLAT_AGGREGATE_COLUMNS = (
    "momentum_1h", "momentum_2h", "momentum_1d", "vol_rel_mean_1h",
    "vol_rel_mean_1d", "day_cumulative_return", "range_mean_1h",
    "range_mean_2h", "range_mean_1d", "pos_mean_1h", "pos_mean_1d",
    "vol_log_mean_1h", "vol_log_mean_1d", "ret_volatility_1h",
    "ret_volatility_1d", "cross_day_share",
)


def load_codes(path: Path, limit: int | None) -> list[str]:
    """Load the canonical stock prefix."""
    codes = pd.read_csv(path, dtype=str)["code"].dropna().drop_duplicates().tolist()
    return codes[:limit] if limit else codes


def safe_nanmean(values: np.ndarray) -> float:
    """Return NaN when an allowed-history feature has no observations."""
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def safe_nanstd(values: np.ndarray) -> float:
    """Return NaN when an allowed-history feature has no observations."""
    return float(np.nanstd(values)) if np.isfinite(values).any() else np.nan


def flat_aggregates(window: pd.DataFrame, current_day: pd.Timestamp) -> dict[str, float]:
    """Compress one causal 48-bar window into interpretable aggregates."""
    ret = window["ret_5m"].to_numpy(float)
    vol = window["vol_rel_20d"].to_numpy(float)
    ranges = window["range_rel"].to_numpy(float)
    pos = window["pos_in_bar"].to_numpy(float)
    logs = window["vol_log"].to_numpy(float)
    same_day = window["date"].eq(current_day).to_numpy()
    return {
        "momentum_1h": float(ret[-12:].sum()), "momentum_2h": float(ret[-24:].sum()),
        "momentum_1d": float(ret.sum()), "vol_rel_mean_1h": safe_nanmean(vol[-12:]),
        "vol_rel_mean_1d": safe_nanmean(vol), "day_cumulative_return": float(ret[same_day].sum()),
        "range_mean_1h": float(ranges[-12:].mean()), "range_mean_2h": float(ranges[-24:].mean()),
        "range_mean_1d": float(ranges.mean()), "pos_mean_1h": float(pos[-12:].mean()),
        "pos_mean_1d": float(pos.mean()), "vol_log_mean_1h": float(logs[-12:].mean()),
        "vol_log_mean_1d": float(logs.mean()), "ret_volatility_1h": safe_nanstd(ret[-12:]),
        "ret_volatility_1d": safe_nanstd(ret), "cross_day_share": float((~same_day).mean()),
    }


def sequence_array(window: pd.DataFrame, current_day: pd.Timestamp) -> np.ndarray:
    """Append a sample-relative cross-day flag to the configured features."""
    base = window[list(SEQUENCE_FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    cross_day = (~window["date"].eq(current_day)).to_numpy(dtype=np.float32)[:, None]
    return np.concatenate([base, cross_day], axis=1)


def sample_indices(frame: pd.DataFrame) -> list[int]:
    """Select only configured times with a complete causal lookback."""
    clocks = frame["time"].dt.strftime("%H:%M:%S")
    return [int(index) for index in np.flatnonzero(clocks.isin(SAMPLE_BAR_TIMES))
            if index >= SEQUENCE_LOOKBACK_BARS - 1]


def sample_record(frame: pd.DataFrame, index: int) -> tuple[np.ndarray, dict[str, object]]:
    """Build one sequence and flat record with audit metadata."""
    start = index - SEQUENCE_LOOKBACK_BARS + 1
    window = frame.iloc[start:index + 1]
    current = frame.iloc[index]
    if window["time"].max() > current["time"]:
        raise AssertionError("future bar entered a sequence")
    record: dict[str, object] = {
        "code": current["code"], "date": current["date"], "time": current["time"],
        "window_start_time": window["time"].iloc[0], "window_end_time": window["time"].iloc[-1],
        "fwd_ret_30m": current["fwd_ret_30m"], "fwd_ret_60m": current["fwd_ret_60m"],
        "label_rank": current["label_rank"],
        "price_mask_flag": bool(window["close_anomaly_flag"].astype(bool).any()),
        "data_quality_mask_flag": bool(window["incomplete_day_flag"].astype(bool).any()),
    }
    for column in FLAT_CURRENT_FEATURE_COLUMNS:
        record[column] = current[column]
    record.update(flat_aggregates(window, current["date"]))
    return sequence_array(window, current["date"]), record


def build_stock_dataset(frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Build all configured samples for one stock."""
    frame = frame.sort_values("time").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["time"] = pd.to_datetime(frame["time"])
    sequences: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for index in sample_indices(frame):
        sequence, record = sample_record(frame, index)
        sequences.append(sequence)
        records.append(record)
    shape = (0, SEQUENCE_LOOKBACK_BARS, len(SEQUENCE_FEATURE_COLUMNS) + 1)
    return (np.stack(sequences) if sequences else np.empty(shape, dtype=np.float32),
            pd.DataFrame(records))


def save_stock_dataset(code: str, sequences: np.ndarray, flat: pd.DataFrame,
                       seq_dir: Path, flat_dir: Path) -> None:
    """Persist arrays, identifiers, and the flat comparison table."""
    seq_dir.mkdir(parents=True, exist_ok=True)
    flat_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        seq_dir / f"{code}.npz", X=sequences,
        feature_names=np.array([*SEQUENCE_FEATURE_COLUMNS, "is_cross_day"]),
        code=flat["code"].astype(str).to_numpy(),
        date=flat["date"].to_numpy(dtype="datetime64[D]"),
        time=flat["time"].to_numpy(dtype="datetime64[ns]"),
        fwd_ret_30m=flat["fwd_ret_30m"].to_numpy(float),
        fwd_ret_60m=flat["fwd_ret_60m"].to_numpy(float),
        label_rank=flat["label_rank"].to_numpy(float),
        price_mask_flag=flat["price_mask_flag"].to_numpy(bool),
        data_quality_mask_flag=flat["data_quality_mask_flag"].to_numpy(bool),
    )
    flat.to_parquet(flat_dir / f"{code}.parquet", index=False, compression="snappy")


def manifest_row(code: str, flat: pd.DataFrame) -> dict[str, object]:
    """Summarize sample eligibility and the intentional tail-label gap."""
    clocks = flat["time"].dt.strftime("%H:%M:%S")
    return {"code": code, "samples": len(flat),
            "label_rank_samples": int(flat["label_rank"].notna().sum()),
            "price_mask_samples": int(flat["price_mask_flag"].sum()),
            "data_quality_mask_samples": int(flat["data_quality_mask_flag"].sum()),
            "fwd30_missing_samples": int(flat["fwd_ret_30m"].isna().sum()),
            "fwd60_missing_samples": int(flat["fwd_ret_60m"].isna().sum()),
            "sample_1435_count": int(clocks.eq("14:35:00").sum()),
            "sample_1435_fwd30_nonnull": int(flat.loc[clocks.eq("14:35:00"), "fwd_ret_30m"].notna().sum())}


def build_datasets(codes: list[str], feature_dir: Path, seq_dir: Path,
                   flat_dir: Path) -> pd.DataFrame:
    """Build per-stock datasets and an aggregate audit manifest."""
    rows: list[dict[str, object]] = []
    for code in codes:
        path = feature_dir / f"{code}.parquet"
        if not path.exists():
            continue
        sequences, flat = build_stock_dataset(pd.read_parquet(path))
        save_stock_dataset(code, sequences, flat, seq_dir, flat_dir)
        rows.append(manifest_row(code, flat))
        print(f"dataset {code} samples={len(flat)}", flush=True)
    manifest = pd.DataFrame(rows)
    flat_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(flat_dir / "dataset_manifest.csv", index=False, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stock-list", type=Path, default=STOCK_LIST_PATH)
    parser.add_argument("--feature-dir", type=Path, default=FEATURES_5MIN_DIR)
    parser.add_argument("--seq-dir", type=Path, default=SEQ_DATASET_DIR)
    parser.add_argument("--flat-dir", type=Path, default=FLAT_DATASET_DIR)
    args = parser.parse_args()
    manifest = build_datasets(load_codes(args.stock_list, args.limit), args.feature_dir,
                              args.seq_dir, args.flat_dir)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
