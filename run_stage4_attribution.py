"""Generate Stage-4 GRU integrated gradients and MLP permutation importance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import (ATTRIBUTION_EXTREME_SHARE, ATTRIBUTION_HEARTBEAT_EVERY,
                    ATTRIBUTION_SAMPLES_PER_WINDOW, ATTRIBUTION_STEPS,
                    ATTRIBUTION_WINDOW_TIMEOUT_SECONDS, FLAT_DATASET_DIR, FEATURES_5MIN_DIR,
                    SEQ_DATASET_DIR, SEQUENCE_FEATURE_COLUMNS,
                    STAGE4_ATTRIBUTION_SEED, STAGE4_REPORT_DIR)
from run_baseline import MODEL_FEATURES, load_flat_dataset, rolling_windows, split_frame
from run_neural_models import (GRURegressor, MLP, flat_arrays, sequence_arrays,
                               StandardizedDataset)

ATTRIBUTION_SEQUENCE_COLUMNS = (*SEQUENCE_FEATURE_COLUMNS, "is_cross_day")


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Restore a Stage-4 checkpoint without refitting."""
    payload = torch.load(path, map_location=device, weights_only=False)
    model = GRURegressor(payload["inputs"]) if payload["model"] == "GRU" else MLP(payload["inputs"])
    model.load_state_dict(payload["state_dict"]); model.to(device).eval()
    return model, payload


def extreme_indices(predictions: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Sample equally from the predeclared top and bottom prediction deciles."""
    rng = np.random.default_rng(seed); half = count // 2
    low, high = np.quantile(predictions, [ATTRIBUTION_EXTREME_SHARE,
                                         1 - ATTRIBUTION_EXTREME_SHARE])
    tails = [np.flatnonzero(predictions <= low), np.flatnonzero(predictions >= high)]
    chosen = [rng.choice(index, min(half, len(index)), replace=False) for index in tails]
    return np.concatenate(chosen)


def standardized(values: np.ndarray, payload: dict) -> torch.Tensor:
    """Apply checkpointed training-window statistics only."""
    scaled = (values - payload["mean"]) / payload["scale"]
    return torch.from_numpy(np.nan_to_num(scaled).astype(np.float32))


def integrated_gradients(model: torch.nn.Module, values: torch.Tensor,
                         device: torch.device, steps: int, heartbeat_every: int,
                         window_id: int) -> np.ndarray:
    """Run Captum IG in bounded chunks to control GPU memory."""
    try:
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise RuntimeError("captum is required: pip install captum") from exc
    if isinstance(model, GRURegressor) and device.type == "cuda":
        model.gru.train(); model.head.eval()
    method = IntegratedGradients(model); chunks: list[np.ndarray] = []; started = time.monotonic()
    for start in range(0, len(values), heartbeat_every):
        batch = values[start:start + heartbeat_every].to(device)
        attrs = method.attribute(batch, baselines=torch.zeros_like(batch),
                                 n_steps=steps, internal_batch_size=heartbeat_every * 2)
        chunks.append(attrs.detach().cpu().numpy())
        done = min(start + heartbeat_every, len(values))
        print(f"IG_HEARTBEAT window={window_id} completed={done}/{len(values)} "
              f"elapsed_seconds={time.monotonic() - started:.1f}", flush=True)
    return np.concatenate(chunks)


def summarize_gru(attrs: np.ndarray, window: int, output: Path) -> pd.DataFrame:
    """Write 48-by-feature and marginal absolute attribution summaries."""
    mean_abs = np.abs(attrs).mean(axis=0)
    heat = pd.DataFrame(mean_abs, columns=ATTRIBUTION_SEQUENCE_COLUMNS)
    heat.insert(0, "lookback_step", np.arange(-len(heat) + 1, 1)); heat.insert(0, "window", window)
    by_feature = heat[list(ATTRIBUTION_SEQUENCE_COLUMNS)].sum().sort_values(ascending=False)
    ranking = by_feature.rename("absolute_attribution").rename_axis("feature").reset_index()
    ranking["share"] = ranking["absolute_attribution"] / ranking["absolute_attribution"].sum()
    ranking.insert(0, "window", window)
    heat.to_csv(output / f"window_{window}_gru_attribution_heatmap.csv", index=False)
    ranking.to_csv(output / f"window_{window}_gru_feature_ranking.csv", index=False)
    return heat


def plot_gru(heatmaps: pd.DataFrame, output: Path) -> None:
    """Render the requested temporal-feature heatmap and recent-return profile."""
    matrix = heatmaps.groupby("lookback_step")[list(ATTRIBUTION_SEQUENCE_COLUMNS)].mean().T
    plt.figure(figsize=(13, 5)); plt.imshow(matrix, aspect="auto", cmap="magma")
    plt.yticks(range(len(matrix)), matrix.index); plt.xticks(range(0, 48, 4), range(-47, 1, 4))
    plt.colorbar(label="Mean absolute IG"); plt.xlabel("Lookback bar (0=current)")
    plt.tight_layout(); plt.savefig(output / "gru_ig_heatmap.png", dpi=150); plt.close()
    ret = heatmaps.groupby("lookback_step")["ret_5m"].mean()
    plt.figure(figsize=(9, 4.5)); plt.plot(ret.index, ret.values, marker="o", markersize=2)
    plt.xlabel("Lookback bar (0=current)"); plt.ylabel("Mean absolute IG: ret_5m")
    plt.tight_layout(); plt.savefig(output / "gru_ret5m_temporal_attribution.png", dpi=150); plt.close()


def permutation_importance(model: torch.nn.Module, values: np.ndarray, payload: dict,
                           prediction: np.ndarray, seed: int, device: torch.device) -> pd.DataFrame:
    """Measure MLP feature importance as prediction disruption after permutation."""
    rng = np.random.default_rng(seed); base = pd.Series(prediction).rank().to_numpy()
    rows = []
    for column, feature in enumerate(MODEL_FEATURES):
        shuffled = values.copy(); shuffled[:, column] = rng.permutation(shuffled[:, column])
        tensor = standardized(shuffled, payload); outputs = []
        with torch.no_grad():
            for start in range(0, len(tensor), 4096):
                outputs.append(model(tensor[start:start + 4096].to(device)).cpu().numpy())
        correlation = pd.Series(base).corr(pd.Series(np.concatenate(outputs)).rank())
        rows.append({"feature": feature, "rank_disruption": 1 - correlation})
    return pd.DataFrame(rows).sort_values("rank_disruption", ascending=False)


def run_window(window_id: int, flat: pd.DataFrame, artifact: Path,
               output: Path, device: torch.device, sample_count: int,
               steps: int, heartbeat_every: int, include_mlp: bool) -> pd.DataFrame:
    """Attribute the fixed representative seed for one test window."""
    window = rolling_windows(flat["date"])[window_id - 1]; parts = split_frame(flat, window)
    arrays = sequence_arrays(flat, window, SEQ_DATASET_DIR)
    stem = f"window_{window_id}_gru_seed_{STAGE4_ATTRIBUTION_SEED}"
    model, payload = load_checkpoint(artifact / f"{stem}.pt", device)
    pred = pd.read_parquet(artifact / f"{stem}_predictions.parquet")["prediction"].to_numpy()
    selected = extreme_indices(pred, sample_count,
                               STAGE4_ATTRIBUTION_SEED + window_id)
    attrs = integrated_gradients(model, standardized(arrays["test"][0][selected], payload),
                                 device, steps, heartbeat_every, window_id)
    heat = summarize_gru(attrs, window_id, output)
    if not include_mlp:
        return heat
    mlp_stem = f"window_{window_id}_mlp_seed_{STAGE4_ATTRIBUTION_SEED}"
    mlp, mlp_payload = load_checkpoint(artifact / f"{mlp_stem}.pt", device)
    mlp_pred = pd.read_parquet(artifact / f"{mlp_stem}_predictions.parquet")["prediction"].to_numpy()
    mlp_selected = extreme_indices(mlp_pred, sample_count,
                                   STAGE4_ATTRIBUTION_SEED + window_id)
    importance = permutation_importance(mlp, flat_arrays(parts)["test"][0][mlp_selected],
                                        mlp_payload, mlp_pred[mlp_selected], window_id, device)
    importance.insert(0, "window", window_id)
    importance.to_csv(output / f"window_{window_id}_mlp_permutation.csv", index=False)
    return heat


def run_worker(args: argparse.Namespace) -> None:
    """Run exactly one window in one process on one device."""
    args.output_dir.mkdir(parents=True, exist_ok=True); device = torch.device(args.device)
    flat = load_flat_dataset(args.flat_dir, args.feature_dir)
    run_window(args.worker_window, flat, args.artifact_dir, args.output_dir, device,
               args.sample_count, args.steps, args.heartbeat_every, args.include_mlp)


def worker_command(args: argparse.Namespace, window: int) -> list[str]:
    """Build an isolated single-window worker command."""
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
        "--worker-window", str(window), "--artifact-dir", str(args.artifact_dir),
        "--flat-dir", str(args.flat_dir), "--feature-dir", str(args.feature_dir),
        "--output-dir", str(args.output_dir), "--device", args.device,
        "--sample-count", str(args.sample_count), "--steps", str(args.steps),
        "--heartbeat-every", str(args.heartbeat_every)]
    return command + (["--include-mlp"] if args.include_mlp else [])


def run(args: argparse.Namespace) -> None:
    """Run isolated windows, skipping failures and enforcing a hard timeout."""
    args.output_dir.mkdir(parents=True, exist_ok=True); failures = []
    for window in args.windows:
        started = time.monotonic(); print(f"IG_WINDOW_START window={window}", flush=True)
        try:
            subprocess.run(worker_command(args, window), check=True,
                           timeout=args.window_timeout_seconds)
            print(f"IG_WINDOW_COMPLETE window={window} elapsed_seconds="
                  f"{time.monotonic() - started:.1f}", flush=True)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            failures.append({"window": window, "error": type(exc).__name__,
                             "elapsed_seconds": time.monotonic() - started})
            print(f"IG_WINDOW_SKIPPED window={window} reason={type(exc).__name__}", flush=True)
    paths = sorted(args.output_dir.glob("window_*_gru_attribution_heatmap.csv"))
    if not paths:
        raise RuntimeError("all attribution windows failed")
    combined = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    combined.to_csv(args.output_dir / "gru_attribution_all_windows.csv", index=False)
    plot_gru(combined, args.output_dir)
    (args.output_dir / "attribution_failures.json").write_text(json.dumps(failures, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", nargs="+", type=int, default=list(range(1, 9)))
    parser.add_argument("--worker-window", type=int)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--flat-dir", type=Path, default=FLAT_DATASET_DIR)
    parser.add_argument("--feature-dir", type=Path, default=FEATURES_5MIN_DIR)
    parser.add_argument("--output-dir", type=Path, default=STAGE4_REPORT_DIR / "attribution")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--sample-count", type=int, default=ATTRIBUTION_SAMPLES_PER_WINDOW)
    parser.add_argument("--steps", type=int, default=ATTRIBUTION_STEPS)
    parser.add_argument("--heartbeat-every", type=int, default=ATTRIBUTION_HEARTBEAT_EVERY)
    parser.add_argument("--window-timeout-seconds", type=int,
                        default=ATTRIBUTION_WINDOW_TIMEOUT_SECONDS)
    parser.add_argument("--include-mlp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_worker(arguments) if arguments.worker_window is not None else run(arguments)
