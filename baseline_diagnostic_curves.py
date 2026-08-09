"""Create train-versus-validation RankIC diagnostics for non-neural baselines."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import (BASELINE_REPORT_DIR, FEATURES_5MIN_DIR, FLAT_DATASET_DIR,
                    LIGHTGBM_CURVE_ITERATION_STEP, STAGE3_REPORT_DIR)
from run_baseline import MODEL_FEATURES, fit_ridge, load_flat_dataset, rolling_windows, split_frame
from run_neural_models import RankICScorer


def selected_parameters() -> dict[str, object]:
    """Load the previously selected window-one LightGBM configuration."""
    frame = pd.read_csv(BASELINE_REPORT_DIR / "window_metrics.csv")
    row = frame[(frame["window"].eq(1)) & frame["model"].eq("LightGBM")].iloc[0]
    result = json.loads(row["parameters"]); result.pop("validation_mse", None)
    return result


def lightgbm_curve(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Evaluate delayed RankIC along the fixed boosting path."""
    from lightgbm import LGBMRegressor
    parameters = selected_parameters()
    model = LGBMRegressor(objective="regression", random_state=20260807,
                          n_jobs=4, verbosity=-1, **parameters)
    model.fit(train[MODEL_FEATURES], train["label_rank"])
    train_scorer, valid_scorer = RankICScorer(train), RankICScorer(valid)
    rows: list[dict[str, float]] = []
    total = int(parameters["n_estimators"])
    for iteration in range(LIGHTGBM_CURVE_ITERATION_STEP, total + 1,
                           LIGHTGBM_CURVE_ITERATION_STEP):
        rows.append({"step": iteration, "train_rank_ic": train_scorer.score(
            model.predict(train[MODEL_FEATURES], num_iteration=iteration)),
            "valid_rank_ic": valid_scorer.score(
            model.predict(valid[MODEL_FEATURES], num_iteration=iteration))})
    return pd.DataFrame(rows)


def ridge_point(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Represent the non-iterative Ridge fit as one diagnostic point."""
    model, _ = fit_ridge(train, valid)
    return pd.DataFrame([{"step": 1, "train_rank_ic": RankICScorer(train).score(
        model.predict(train[MODEL_FEATURES])), "valid_rank_ic": RankICScorer(valid).score(
        model.predict(valid[MODEL_FEATURES]))}])


def plot_curve(frame: pd.DataFrame, model: str) -> None:
    """Save the common train/validation diagnostic format."""
    plt.figure(figsize=(8, 4.5)); plt.plot(frame["step"], frame["train_rank_ic"],
        marker="o", label="train"); plt.plot(frame["step"], frame["valid_rank_ic"],
        marker="o", label="validation"); plt.axhline(0, color="black", linewidth=0.7)
    plt.xlabel("Boosting iteration" if model == "LightGBM" else "Single fitted model")
    plt.ylabel("Delayed RankIC"); plt.title(f"window_1_{model.lower()}_diagnostic")
    plt.legend(); plt.tight_layout(); plt.savefig(
        STAGE3_REPORT_DIR / f"window_1_{model.lower()}_curve.png", dpi=150); plt.close()


def main() -> None:
    STAGE3_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_flat_dataset(FLAT_DATASET_DIR, FEATURES_5MIN_DIR)
    train, valid, _ = split_frame(frame, rolling_windows(frame["date"])[0])
    for model, curve in (("LightGBM", lightgbm_curve(train, valid)),
                         ("Ridge", ridge_point(train, valid))):
        curve.to_csv(STAGE3_REPORT_DIR / f"window_1_{model.lower()}_curve.csv", index=False)
        plot_curve(curve, model); print(model, curve.tail(1).to_dict("records")[0])


if __name__ == "__main__":
    main()
