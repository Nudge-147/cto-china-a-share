"""Create auditable research-ready minute files from immutable raw inputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DAILY_REFERENCE_DIR, PILOT_SCALE_STOCK_COUNT, PRICE_TOLERANCE, PROCESSED_5MIN_DIR,
    RAW_5MIN_DIR, REPAIR_REPORT_PATH, SCALE_GATE_PATH, STOCK_LIST_PATH,
)
from download_5min import atomic_parquet


def load_codes(path: Path, limit: int | None) -> list[str]:
    """Load the requested canonical stock prefix."""
    codes = pd.read_csv(path, dtype=str)["code"].dropna().drop_duplicates().tolist()
    return codes[:limit] if limit else codes


def proportional_integer_allocation(values: pd.Series, target: float) -> np.ndarray:
    """Allocate an integer daily target proportionally by largest remainder."""
    weights = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    target_int = int(round(target))
    if weights.sum() == 0:
        if target_int != 0:
            raise ValueError("cannot allocate a positive target from zero raw volume")
        return np.zeros(len(weights), dtype=np.int64)
    exact = weights * target_int / weights.sum()
    result = np.floor(exact).astype(np.int64)
    remainder = target_int - int(result.sum())
    if remainder:
        order = np.argsort(-(exact - result), kind="stable")
        result[order[:remainder]] += 1
    return result


def daily_reference(path: Path) -> pd.DataFrame:
    """Load tradable daily volume and close targets."""
    frame = pd.read_parquet(path, columns=["date", "volume", "close", "tradestatus"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame[frame["tradestatus"].eq(1)].drop_duplicates("date").set_index("date")


def repair_stock(code: str, raw_dir: Path, reference_dir: Path, output_dir: Path) -> dict[str, object]:
    """Rescale volume by daily targets and flag close-anomaly stock-days."""
    frame = pd.read_parquet(raw_dir / f"{code}.parquet").sort_values("time").copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    reference = daily_reference(reference_dir / f"{code}.parquet")
    frame["volume_raw"] = pd.to_numeric(frame["volume"], errors="raise").astype(np.int64)
    frame["volume_scale_factor"] = 1.0
    frame["volume_rescaled"] = False
    frame["close_anomaly_flag"] = False
    for day, index in frame.groupby("date", sort=False).groups.items():
        if day not in reference.index:
            continue
        raw = frame.loc[index, "volume_raw"]
        target = reference.at[day, "volume"]
        corrected = proportional_integer_allocation(raw, target)
        factor = float(target) / float(raw.sum()) if raw.sum() else 1.0
        changed = not np.array_equal(corrected, raw.to_numpy())
        frame.loc[index, "volume"] = corrected
        frame.loc[index, "volume_scale_factor"] = factor
        frame.loc[index, "volume_rescaled"] = changed
        last_close = float(frame.loc[index].sort_values("time")["close"].iloc[-1])
        frame.loc[index, "close_anomaly_flag"] = abs(last_close - reference.at[day, "close"]) >= PRICE_TOLERANCE
    frame["volume"] = pd.to_numeric(frame["volume"], errors="raise").astype(np.int64)
    atomic_parquet(frame, output_dir / f"{code}.parquet")
    daily_sum = frame.groupby("date")["volume"].sum()
    common = daily_sum.index.intersection(reference.index)
    residual = daily_sum.loc[common] - reference.loc[common, "volume"]
    return {"code": code, "rows": len(frame),
            "volume_rescaled_days": frame.loc[frame["volume_rescaled"], "date"].nunique(),
            "close_anomaly_days": frame.loc[frame["close_anomaly_flag"], "date"].nunique(),
            "post_rescale_nonzero_days": int(residual.ne(0).sum()),
            "max_post_rescale_abs_shares": float(residual.abs().max()) if len(residual) else 0.0}


def run_repair(codes: list[str], raw_dir: Path, reference_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Repair every available stock and write a one-row audit."""
    available = [code for code in codes if (raw_dir / f"{code}.parquet").exists()
                 and (reference_dir / f"{code}.parquet").exists()]
    report = pd.DataFrame([repair_stock(code, raw_dir, reference_dir, output_dir) for code in available])
    REPAIR_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPAIR_REPORT_PATH, index=False, encoding="utf-8")
    write_scale_gate(report)
    return report


def write_scale_gate(report: pd.DataFrame) -> None:
    """Open the 100-stock gate only after exact repaired-volume reconciliation."""
    residual_days = int(report["post_rescale_nonzero_days"].sum()) if len(report) else -1
    status = "OPEN" if len(report) and residual_days == 0 else "BLOCKED"
    gate = pd.DataFrame([{
        "gate": "100_stock_pilot", "status": status,
        "target_stocks": PILOT_SCALE_STOCK_COUNT, "validated_stocks": len(report),
        "post_rescale_nonzero_days": residual_days,
        "required_processing": "daily_total_proportional_rescale_and_close_anomaly_mask",
    }])
    gate.to_csv(SCALE_GATE_PATH, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stock-list", type=Path, default=STOCK_LIST_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_5MIN_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DAILY_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_5MIN_DIR)
    args = parser.parse_args()
    report = run_repair(load_codes(args.stock_list, args.limit), args.raw_dir,
                        args.reference_dir, args.output_dir)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
