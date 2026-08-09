"""Run label-free six-signal inference from saved Stage-3/4 models."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from config import (FEATURES_5MIN_DIR, FLAT_DATASET_DIR, SAMPLE_BAR_TIMES,
                    SEQ_DATASET_DIR, STAGE3_SEEDS)
from run_baseline import MODEL_FEATURES, rolling_windows
from run_neural_models import GRURegressor, MLP, predict


def load_inference_frame(directory: Path) -> pd.DataFrame:
    """Load all six sample times without requiring a forward label."""
    paths = sorted(path for path in directory.glob("*.parquet")
                   if path.name != "dataset_manifest.parquet")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["time"] = pd.to_datetime(frame["time"])
    clocks = frame["time"].dt.strftime("%H:%M:%S")
    valid = clocks.isin(SAMPLE_BAR_TIMES) & ~frame["price_mask_flag"]
    valid &= ~frame["data_quality_mask_flag"]
    return frame[valid].sort_values(["time", "code"]).reset_index(drop=True)


def test_rows(frame: pd.DataFrame, window: dict[str, pd.Timestamp]) -> pd.DataFrame:
    """Slice only a rolling window's test dates."""
    return frame[frame["date"].between(window["test_start"], window["test_end"])]


def sequence_values(frame: pd.DataFrame, directory: Path) -> np.ndarray:
    """Load sequences in exactly the supplied frame order."""
    chunks = []
    for code, group in frame.groupby("code", sort=False):
        with np.load(directory / f"{code}.npz") as archive:
            positions = pd.Index(archive["time"]).get_indexer(
                group["time"].to_numpy(dtype="datetime64[ns]"))
            if (positions < 0).any():
                raise KeyError(f"missing inference sequence: {code}")
            chunks.append(pd.DataFrame({"index": group.index,
                "value": list(archive["X"][positions])}))
    ordered = pd.concat(chunks).sort_values("index")
    return np.stack(ordered["value"].to_numpy())


def torch_predictions(model: torch.nn.Module, values: np.ndarray, payload: dict,
                      device: torch.device) -> np.ndarray:
    """Apply checkpoint scaling and bounded-batch inference."""
    scaled = np.nan_to_num((values - payload["mean"]) / payload["scale"]).astype(np.float32)
    outputs = []; model.to(device).eval()
    with torch.no_grad():
        for start in range(0, len(scaled), 4096):
            outputs.append(model(torch.from_numpy(scaled[start:start + 4096]).to(device)).cpu().numpy())
    return np.concatenate(outputs)


def save_prediction(meta: pd.DataFrame, prediction: np.ndarray, path: Path) -> None:
    """Persist the six-signal identifier/prediction panel."""
    result = meta[["code", "date", "time"]].copy(); result["prediction"] = prediction
    result.to_parquet(path, index=False)


def run(args: argparse.Namespace) -> None:
    """Score all four model families at all six daily sample times."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_inference_frame(args.flat_dir); windows = rolling_windows(frame["date"])
    device = torch.device(args.device)
    for window_id, window in enumerate(windows, start=1):
        test = test_rows(frame, window)
        for model in ("ridge", "lightgbm"):
            fitted = joblib.load(args.baseline_artifact_dir / f"window_{window_id}_{model}.joblib")
            save_prediction(test, fitted.predict(test[MODEL_FEATURES]),
                args.output_dir / f"window_{window_id}_{model}_six_signals.parquet")
        sequences = sequence_values(test, args.seq_dir)
        for model_name, values in (("mlp", test[MODEL_FEATURES].to_numpy(np.float32)),
                                   ("gru", sequences)):
            for seed in STAGE3_SEEDS:
                stem = f"window_{window_id}_{model_name}_seed_{seed}"
                payload = torch.load(args.neural_artifact_dir / f"{stem}.pt",
                                     map_location=device, weights_only=False)
                model = GRURegressor(payload["inputs"]) if model_name == "gru" else MLP(payload["inputs"])
                model.load_state_dict(payload["state_dict"])
                save_prediction(test, torch_predictions(model, values, payload, device),
                    args.output_dir / f"{stem}_six_signals.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-artifact-dir", type=Path, required=True)
    parser.add_argument("--neural-artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--flat-dir", type=Path, default=FLAT_DATASET_DIR)
    parser.add_argument("--feature-dir", type=Path, default=FEATURES_5MIN_DIR)
    parser.add_argument("--seq-dir", type=Path, default=SEQ_DATASET_DIR)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
