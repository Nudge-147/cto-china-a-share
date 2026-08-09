"""Run the pre-neural label-shuffle and exact timestamp purge gates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (BASELINE_RANDOM_SEED, BASELINE_REPORT_DIR, FEATURES_5MIN_DIR,
                    FLAT_DATASET_DIR, LABEL_SHUFFLE_ABS_RANKIC_GATE,
                    STAGE3_REPORT_DIR, STAGE3_SEEDS)
from run_baseline import (MODEL_FEATURES, evaluate_predictions, load_flat_dataset,
                          rolling_windows, split_frame)


def purge_timestamp_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Verify that prior-split label endpoints precede next-split samples."""
    rows: list[dict[str, object]] = []
    for number, window in enumerate(rolling_windows(frame["date"]), start=1):
        train, valid, test = split_frame(frame, window)
        train_end, valid_start = train["label_end_time"].max(), valid["time"].min()
        valid_end, test_start = valid["label_end_time"].max(), test["time"].min()
        rows.append({"window": number, "train_label_end_max": train_end,
            "valid_sample_time_min": valid_start, "train_valid_gap_hours":
            (valid_start - train_end).total_seconds() / 3600,
            "train_valid_pass": train_end < valid_start,
            "valid_label_end_max": valid_end, "test_sample_time_min": test_start,
            "valid_test_gap_hours": (test_start - valid_end).total_seconds() / 3600,
            "valid_test_pass": valid_end < test_start,
            "valid_context_starts_before_train_end":
            valid["window_start_time"].min() <= train["time"].max()})
    return pd.DataFrame(rows)


def shuffle_labels(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Shuffle labels only within each sampling timestamp."""
    result = frame.copy(); values = result["label_rank"].to_numpy(copy=True)
    rng = np.random.default_rng(seed)
    for indices in result.groupby("time", sort=False).indices.values():
        values[indices] = rng.permutation(values[indices])
    result["label_rank"] = values
    return result


def fixed_models(parameters: dict[str, object]) -> dict[str, object]:
    """Instantiate prior-selected window-one models without retuning on noise."""
    from lightgbm import LGBMRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    lgb = LGBMRegressor(objective="regression", random_state=BASELINE_RANDOM_SEED,
        n_jobs=4, verbosity=-1, **parameters)
    ridge = Pipeline([("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    return {"LightGBM": lgb, "Ridge": ridge}


def label_shuffle_audit(frame: pd.DataFrame, parameters: dict[str, object]) -> pd.DataFrame:
    """Fit five within-timestamp label shuffles and score delayed returns."""
    train, _, test = split_frame(frame, rolling_windows(frame["date"])[0])
    delayed = test[["date", "time", "fwd_ret_30m_entry_lag1"]].rename(
        columns={"fwd_ret_30m_entry_lag1": "fwd_ret_30m"}).dropna()
    rows: list[dict[str, object]] = []
    for seed in STAGE3_SEEDS:
        shuffled = shuffle_labels(train, seed)
        for name, model in fixed_models(parameters).items():
            model.fit(shuffled[MODEL_FEATURES], shuffled["label_rank"])
            scored = delayed.copy()
            scored["prediction"] = model.predict(test.loc[delayed.index, MODEL_FEATURES])
            metrics, _, _ = evaluate_predictions(scored)
            rows.append({"seed": seed, "model": name, **metrics})
    result = pd.DataFrame(rows)
    means = result.groupby("model")["rank_ic"].transform("mean").abs()
    result["gate_pass"] = means.lt(LABEL_SHUFFLE_ABS_RANKIC_GATE)
    return result


def selected_lgb_parameters() -> dict[str, object]:
    """Load the already selected first-window hyperparameters."""
    metrics = pd.read_csv(BASELINE_REPORT_DIR / "window_metrics.csv")
    row = metrics[(metrics["window"].eq(1)) & metrics["model"].eq("LightGBM")].iloc[0]
    parameters = json.loads(row["parameters"]); parameters.pop("validation_mse", None)
    return parameters


def main() -> None:
    STAGE3_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_flat_dataset(FLAT_DATASET_DIR, FEATURES_5MIN_DIR)
    purge = purge_timestamp_audit(frame)
    shuffle = label_shuffle_audit(frame, selected_lgb_parameters())
    purge.to_csv(STAGE3_REPORT_DIR / "purge_timestamp_audit.csv", index=False)
    shuffle.to_csv(STAGE3_REPORT_DIR / "label_shuffle_audit.csv", index=False)
    print(purge.to_string(index=False)); print(shuffle.groupby("model").agg(
        rank_ic_mean=("rank_ic", "mean"), rank_ic_std=("rank_ic", "std"),
        all_pass=("gate_pass", "all")).to_string())


if __name__ == "__main__":
    main()
