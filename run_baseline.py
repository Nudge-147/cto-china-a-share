"""Run purged rolling LightGBM and Ridge baselines on flat intraday samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    BASELINE_RANDOM_SEED, BASELINE_REPORT_DIR, DECILE_COUNT, FLAT_CURRENT_FEATURE_COLUMNS,
    FEATURES_5MIN_DIR, FLAT_DATASET_DIR, LABEL_30M_BARS, MIN_EXPECTED_TRAIN_SAMPLES,
    PURGE_CALENDAR_DAYS, SAMPLE_BAR_TIMES,
    ROLL_STEP_MONTHS, TEST_MONTHS, TRAIN_YEARS, VALID_MONTHS,
)
from build_5min_dataset import FLAT_AGGREGATE_COLUMNS


MODEL_FEATURES = [*FLAT_CURRENT_FEATURE_COLUMNS, *FLAT_AGGREGATE_COLUMNS]


def delayed_entry_returns(feature_dir: Path) -> pd.DataFrame:
    """Build an execution-lag sensitivity label using entry at the next bar close."""
    rows: list[pd.DataFrame] = []
    for path in feature_dir.glob("*.parquet"):
        frame = pd.read_parquet(path, columns=["code", "date", "time", "close"])
        frame = frame.sort_values("time")
        grouped = frame.groupby("date")["close"]
        time_grouped = frame.groupby("date")["time"]
        frame["fwd_ret_30m_entry_lag1"] = np.log(
            grouped.shift(-LABEL_30M_BARS) / grouped.shift(-1))
        frame["label_end_time"] = time_grouped.shift(-LABEL_30M_BARS)
        frame["entry_lag1_time"] = time_grouped.shift(-1)
        clocks = frame["time"].dt.strftime("%H:%M:%S")
        rows.append(frame.loc[clocks.isin(SAMPLE_BAR_TIMES),
                              ["code", "time", "fwd_ret_30m_entry_lag1",
                               "label_end_time", "entry_lag1_time"]])
    return pd.concat(rows, ignore_index=True)


def load_flat_dataset(directory: Path, feature_dir: Path | None = None) -> pd.DataFrame:
    """Load all per-stock flat partitions and enforce timestamp types."""
    paths = sorted(path for path in directory.glob("*.parquet") if path.name != "dataset_manifest.parquet")
    if not paths:
        raise FileNotFoundError(f"no flat parquet files in {directory}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["time"] = pd.to_datetime(frame["time"])
    if feature_dir is not None:
        frame = frame.merge(delayed_entry_returns(feature_dir), on=["code", "time"],
                            how="left", validate="one_to_one")
    valid = frame["label_rank"].notna() & frame["fwd_ret_30m"].notna()
    valid &= ~frame["price_mask_flag"] & ~frame["data_quality_mask_flag"]
    return frame[valid].sort_values(["time", "code"]).reset_index(drop=True)


def rolling_windows(dates: pd.Series) -> list[dict[str, pd.Timestamp]]:
    """Create two-year/three-month/six-month windows with one-day gaps."""
    anchor, maximum = dates.min().normalize(), dates.max().normalize()
    rows: list[dict[str, pd.Timestamp]] = []
    while True:
        train_end = anchor + pd.DateOffset(years=TRAIN_YEARS) - pd.Timedelta(days=1)
        valid_start = train_end + pd.Timedelta(days=PURGE_CALENDAR_DAYS + 1)
        valid_end = valid_start + pd.DateOffset(months=VALID_MONTHS) - pd.Timedelta(days=1)
        test_start = valid_end + pd.Timedelta(days=PURGE_CALENDAR_DAYS + 1)
        test_end = test_start + pd.DateOffset(months=TEST_MONTHS) - pd.Timedelta(days=1)
        if test_end > maximum:
            break
        rows.append({"train_start": anchor, "train_end": train_end,
                     "valid_start": valid_start, "valid_end": valid_end,
                     "test_start": test_start, "test_end": test_end})
        anchor += pd.DateOffset(months=ROLL_STEP_MONTHS)
    return rows


def split_frame(frame: pd.DataFrame, window: dict[str, pd.Timestamp]) -> tuple[pd.DataFrame, ...]:
    """Slice one window and assert chronological separation."""
    train = frame[frame["date"].between(window["train_start"], window["train_end"])]
    valid = frame[frame["date"].between(window["valid_start"], window["valid_end"])]
    test = frame[frame["date"].between(window["test_start"], window["test_end"])]
    if train.empty or valid.empty or test.empty:
        raise ValueError("rolling window contains an empty split")
    if not (train["date"].max() < valid["date"].min() < test["date"].min()):
        raise AssertionError("rolling split chronology violated")
    return train, valid, test


def validation_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate the validation objective used for coarse tuning."""
    return float(np.mean((actual - predicted) ** 2))


def fit_lightgbm(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[Any, dict[str, Any]]:
    """Coarsely tune only the three authorized LightGBM parameters."""
    try:
        from lightgbm import LGBMRegressor
    except Exception as exc:
        raise RuntimeError("LightGBM runtime unavailable; install lightgbm and libomp") from exc
    best_model: Any | None = None
    best: dict[str, Any] | None = None
    for leaves in [15, 31]:
        for rate in [0.03, 0.08]:
            for minimum in [100, 500]:
                params = {"num_leaves": leaves, "learning_rate": rate,
                          "min_child_samples": minimum, "n_estimators": 250}
                model = LGBMRegressor(objective="regression", random_state=BASELINE_RANDOM_SEED,
                                      n_jobs=4, verbosity=-1, **params)
                model.fit(train[MODEL_FEATURES], train["label_rank"])
                score = validation_mse(valid["label_rank"].to_numpy(),
                                       model.predict(valid[MODEL_FEATURES]))
                if best is None or score < best["validation_mse"]:
                    best_model, best = model, {**params, "validation_mse": score}
    return best_model, best or {}


def fit_ridge(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[Any, dict[str, Any]]:
    """Fit a fixed regularized linear comparison with training-only transforms."""
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError("scikit-learn runtime unavailable") from exc
    model = Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    model.fit(train[MODEL_FEATURES], train["label_rank"])
    score = validation_mse(valid["label_rank"].to_numpy(), model.predict(valid[MODEL_FEATURES]))
    return model, {"alpha": 10.0, "validation_mse": score}


def rank_ic(group: pd.DataFrame) -> float:
    """Compute Spearman RankIC without relying on SciPy's helper."""
    if len(group) < 30:
        return np.nan
    return float(group["prediction"].rank().corr(group["fwd_ret_30m"].rank()))


def evaluate_predictions(frame: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Calculate timestamp RankIC, ICIR, and decile long-short returns."""
    keys = ["date", "time"]
    ic = frame.groupby(keys).apply(rank_ic, include_groups=False).rename("rank_ic").dropna().reset_index()
    ranked = frame.copy()
    percentiles = ranked.groupby(keys)["prediction"].rank(method="first", pct=True)
    ranked["decile"] = np.minimum(np.ceil(percentiles * DECILE_COUNT), DECILE_COUNT).astype(int)
    deciles = ranked.groupby("decile", as_index=False)["fwd_ret_30m"].mean()
    spreads = ranked.groupby(keys).apply(lambda group:
        group.loc[group["decile"].eq(DECILE_COUNT), "fwd_ret_30m"].mean()
        - group.loc[group["decile"].eq(1), "fwd_ret_30m"].mean(), include_groups=False)
    mean_ic, std_ic = ic["rank_ic"].mean(), ic["rank_ic"].std(ddof=1)
    metrics = {"rank_ic": mean_ic, "icir": mean_ic / std_ic if std_ic else np.nan,
               "decile_long_short_return": spreads.mean(), "ic_observations": len(ic)}
    return metrics, ic, deciles


def sanity_flag(rankic: float) -> str:
    """Flag suspicious baseline performance instead of celebrating it."""
    if rankic < 0:
        return "NEGATIVE_REVIEW"
    if rankic > 0.08:
        return "HIGH_REVIEW"
    return "OK"


def run_one_model(name: str, fitter: Callable[..., tuple[Any, dict[str, Any]]],
                  train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame,
                  window_id: int, prediction_dir: Path | None = None
                  ) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Fit and evaluate one model in one rolling window."""
    model, params = fitter(train, valid)
    score_columns = ["code", "date", "time", "fwd_ret_30m"]
    if "fwd_ret_30m_entry_lag1" in test:
        score_columns.append("fwd_ret_30m_entry_lag1")
    scored = test[score_columns].copy()
    scored["prediction"] = model.predict(test[MODEL_FEATURES])
    if prediction_dir is not None:
        import joblib
        prediction_dir.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(prediction_dir / f"window_{window_id}_{name.lower()}_predictions.parquet",
                          index=False)
        joblib.dump(model, prediction_dir / f"window_{window_id}_{name.lower()}.joblib")
    metrics, ic, deciles = evaluate_predictions(scored)
    metrics["entry_lag1_rank_ic"] = np.nan
    metrics["entry_lag1_icir"] = np.nan
    metrics["entry_lag1_decile_long_short_return"] = np.nan
    if "fwd_ret_30m_entry_lag1" in scored:
        delayed = scored.drop(columns="fwd_ret_30m").rename(
            columns={"fwd_ret_30m_entry_lag1": "fwd_ret_30m"}).dropna()
        delayed_metrics, delayed_ic, delayed_deciles = evaluate_predictions(delayed)
        metrics["entry_lag1_rank_ic"] = delayed_metrics["rank_ic"]
        metrics["entry_lag1_icir"] = delayed_metrics["icir"]
        metrics["entry_lag1_decile_long_short_return"] = delayed_metrics[
            "decile_long_short_return"]
        for _, decile_row in delayed_deciles.iterrows():
            metrics[f"decile_{int(decile_row['decile'])}"] = decile_row["fwd_ret_30m"]
        ic, deciles = delayed_ic, delayed_deciles
    row: dict[str, object] = {"window": window_id, "model": name,
        "train_samples": len(train), "valid_samples": len(valid), "test_samples": len(test),
        "train_below_500k": len(train) < MIN_EXPECTED_TRAIN_SAMPLES,
        "parameters": json.dumps(params, ensure_ascii=False), **metrics,
        "rankic_sanity_flag": sanity_flag(float(metrics["rank_ic"]))}
    ic["window"], ic["model"] = window_id, name
    deciles["window"], deciles["model"] = window_id, name
    return row, ic, deciles


def plot_reports(ic: pd.DataFrame, deciles: pd.DataFrame, output: Path) -> None:
    """Create the requested IC time-series and decile monotonicity figures."""
    output.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 5))
    daily = ic.groupby(["date", "model"], as_index=False)["rank_ic"].mean()
    for model, group in daily.groupby("model"):
        plt.plot(group["date"], group["rank_ic"].rolling(20, min_periods=5).mean(), label=model)
    plt.axhline(0, color="black", linewidth=0.8); plt.legend(); plt.ylabel("20-day mean RankIC")
    plt.tight_layout(); plt.savefig(output / "rankic_timeseries.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 5))
    means = deciles.groupby(["model", "decile"], as_index=False)["fwd_ret_30m"].mean()
    for model, group in means.groupby("model"):
        plt.plot(group["decile"], group["fwd_ret_30m"], marker="o", label=model)
    plt.xlabel("Prediction decile"); plt.ylabel("Mean forward 30m log return"); plt.legend()
    plt.tight_layout(); plt.savefig(output / "decile_monotonicity.png", dpi=150); plt.close()


def run_baselines(frame: pd.DataFrame, output: Path,
                  save_predictions: bool = False) -> pd.DataFrame:
    """Run every complete rolling window and write auditable outputs."""
    rows: list[dict[str, object]] = []; ic_tables: list[pd.DataFrame] = []
    decile_tables: list[pd.DataFrame] = []; audits: list[dict[str, object]] = []
    fitters = {"LightGBM": fit_lightgbm, "Ridge": fit_ridge}
    for window_id, window in enumerate(rolling_windows(frame["date"]), start=1):
        train, valid, test = split_frame(frame, window)
        audits.append({"window": window_id, **window, "train_max": train["date"].max(),
                       "valid_min": valid["date"].min(), "valid_max": valid["date"].max(),
                       "test_min": test["date"].min()})
        for name, fitter in fitters.items():
            prediction_dir = output / "artifacts" if save_predictions else None
            row, ic, decile = run_one_model(name, fitter, train, valid, test, window_id,
                                            prediction_dir)
            rows.append({**window, **row}); ic_tables.append(ic); decile_tables.append(decile)
            print(f"window={window_id} model={name} rankic={row['rank_ic']:.4f}", flush=True)
    metrics = pd.DataFrame(rows); ic = pd.concat(ic_tables, ignore_index=True)
    deciles = pd.concat(decile_tables, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "window_metrics.csv", index=False, encoding="utf-8")
    ic.to_csv(output / "rankic_timeseries.csv", index=False, encoding="utf-8")
    deciles.to_csv(output / "decile_returns.csv", index=False, encoding="utf-8")
    pd.DataFrame(audits).to_csv(output / "split_audit.csv", index=False, encoding="utf-8")
    summary = metrics.groupby("model", as_index=False).agg(
        windows=("window", "size"), mean_rank_ic=("rank_ic", "mean"),
        mean_entry_lag1_rank_ic=("entry_lag1_rank_ic", "mean"),
        mean_entry_lag1_icir=("entry_lag1_icir", "mean"),
        mean_entry_lag1_long_short=("entry_lag1_decile_long_short_return", "mean"),
        mean_icir=("icir", "mean"), mean_long_short=("decile_long_short_return", "mean"),
        low_sample_windows=("train_below_500k", "sum"))
    summary.to_csv(output / "full_period_summary.csv", index=False, encoding="utf-8")
    plot_reports(ic, deciles, output)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat-dir", type=Path, default=FLAT_DATASET_DIR)
    parser.add_argument("--feature-dir", type=Path, default=FEATURES_5MIN_DIR)
    parser.add_argument("--output-dir", type=Path, default=BASELINE_REPORT_DIR)
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    metrics = run_baselines(load_flat_dataset(args.flat_dir, args.feature_dir), args.output_dir,
                            args.save_predictions)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
