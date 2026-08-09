"""Train leakage-audited MLP and GRU models on rolling five-minute windows."""
from __future__ import annotations

import argparse
import copy
import hashlib
import random
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import (FEATURES_5MIN_DIR, FLAT_DATASET_DIR, GRU_BATCH_SIZE,
                    GRU_HIDDEN_SIZE, GRU_LAYERS, MLP_BATCH_SIZE, MLP_HIDDEN_LAYERS,
                    MLP_HIDDEN_SIZE, NEURAL_CPU_THREADS, NEURAL_DROPOUT,
                    NEURAL_EARLY_STOP_PATIENCE, NEURAL_LEARNING_RATE,
                    NEURAL_MAX_EPOCHS, NEURAL_WEIGHT_DECAY, GRU_INCREMENT_RANKIC_GATE,
                    SCALING_FACTOR_END_DATE, SCALING_FACTOR_START_DATE,
                    SEQ_DATASET_DIR, STAGE3_REPORT_DIR, STAGE3_SEEDS,
                    BASELINE_REPORT_DIR)
from run_baseline import (MODEL_FEATURES, evaluate_predictions, load_flat_dataset,
                          rolling_windows, split_frame)


class StandardizedDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Apply training-only feature statistics lazily to an in-memory array."""

    def __init__(self, features: np.ndarray, target: np.ndarray,
                 mean: np.ndarray, scale: np.ndarray) -> None:
        self.features, self.target = features, target.astype(np.float32)
        self.mean = torch.from_numpy(mean.astype(np.float32))
        self.scale = torch.from_numpy(scale.astype(np.float32))

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.from_numpy(self.features[index].astype(np.float32, copy=False))
        values = torch.nan_to_num((values - self.mean) / self.scale)
        return values, torch.tensor(self.target[index], dtype=torch.float32)


class MLP(nn.Module):
    """Batch-normalized snapshot network with the preregistered capacity."""

    def __init__(self, inputs: int) -> None:
        super().__init__(); layers: list[nn.Module] = []
        for index in range(MLP_HIDDEN_LAYERS):
            layers.extend([nn.Linear(inputs if index == 0 else MLP_HIDDEN_SIZE,
                                     MLP_HIDDEN_SIZE), nn.BatchNorm1d(MLP_HIDDEN_SIZE),
                           nn.ReLU(), nn.Dropout(NEURAL_DROPOUT)])
        layers.append(nn.Linear(MLP_HIDDEN_SIZE, 1)); self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


class GRURegressor(nn.Module):
    """One-layer GRU followed by a small regularized regression head."""

    def __init__(self, inputs: int) -> None:
        super().__init__()
        self.gru = nn.GRU(inputs, GRU_HIDDEN_SIZE, GRU_LAYERS, batch_first=True)
        self.head = nn.Sequential(nn.Linear(GRU_HIDDEN_SIZE, GRU_HIDDEN_SIZE), nn.ReLU(),
                                  nn.Dropout(NEURAL_DROPOUT), nn.Linear(GRU_HIDDEN_SIZE, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(values)
        return self.head(output[:, -1]).squeeze(-1)


class RankICScorer:
    """Calculate mean timestamp Spearman correlation with cached target ranks."""

    def __init__(self, metadata: pd.DataFrame) -> None:
        self.codes, _ = pd.factorize(metadata["time"], sort=False)
        self.counts = np.bincount(self.codes).astype(float)
        target = metadata.groupby("time")["fwd_ret_30m_entry_lag1"].rank().to_numpy(float)
        self.target = target
        self.target_mean = np.bincount(self.codes, weights=target) / self.counts

    def score(self, prediction: np.ndarray) -> float:
        ranked = pd.Series(prediction).groupby(self.codes).rank().to_numpy(float)
        pred_mean = np.bincount(self.codes, weights=ranked) / self.counts
        pred_center = ranked - pred_mean[self.codes]
        target_center = self.target - self.target_mean[self.codes]
        numerator = np.bincount(self.codes, weights=pred_center * target_center)
        pred_ss = np.bincount(self.codes, weights=pred_center ** 2)
        target_ss = np.bincount(self.codes, weights=target_center ** 2)
        correlations = numerator / np.sqrt(pred_ss * target_ss)
        return float(np.nanmean(correlations[self.counts >= 30]))


def set_seed(seed: int) -> None:
    """Make local and Kaggle CPU runs repeatable."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(NEURAL_CPU_THREADS)


def feature_statistics(values: np.ndarray, sequence: bool) -> tuple[np.ndarray, np.ndarray]:
    """Fit imputation and scaling statistics on one training split only."""
    axes = (0, 1) if sequence else 0
    mean = np.nanmean(values, axis=axes)
    scale = np.nanstd(values, axis=axes)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return mean, scale


def flat_arrays(parts: tuple[pd.DataFrame, ...]) -> dict[str, tuple[np.ndarray, pd.DataFrame]]:
    """Convert the exact baseline snapshots into model arrays."""
    result: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for name, frame in zip(("train", "valid", "test"), parts):
        meta = frame[["code", "date", "time", "fwd_ret_30m",
                      "fwd_ret_30m_entry_lag1", "label_rank"]].reset_index(drop=True)
        result[name] = (frame[MODEL_FEATURES].to_numpy(np.float32), meta)
    return result


def sequence_arrays(frame: pd.DataFrame, window: dict[str, pd.Timestamp],
                    directory: Path) -> dict[str, tuple[np.ndarray, pd.DataFrame]]:
    """Load only one rolling window from the per-stock compressed sequences."""
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in ("train", "valid", "test")}
    metas: dict[str, list[pd.DataFrame]] = {name: [] for name in chunks}
    parts = split_frame(frame, window)
    for code, eligible in frame.groupby("code", sort=False):
        path = directory / f"{code}.npz"
        if not path.exists() or eligible.empty:
            continue
        with np.load(path) as archive:
            positions = pd.Index(archive["time"]).get_indexer(
                eligible["time"].to_numpy(dtype="datetime64[ns]"))
            if (positions < 0).any():
                raise KeyError(f"sequence metadata mismatch for {code}")
            stock_x = archive["X"][positions]
        for name, part in zip(chunks, parts):
            selected = eligible.index.isin(part.index)
            if selected.any():
                chunks[name].append(stock_x[selected])
                metas[name].append(eligible.loc[selected, ["code", "date", "time",
                    "fwd_ret_30m", "fwd_ret_30m_entry_lag1", "label_rank"]])
    return {name: (np.concatenate(chunks[name]), pd.concat(metas[name], ignore_index=True))
            for name in chunks}


def make_loaders(arrays: dict[str, tuple[np.ndarray, pd.DataFrame]], batch_size: int,
                 sequence: bool) -> tuple[dict[str, DataLoader], np.ndarray, np.ndarray]:
    """Create loaders using only training-window normalization statistics."""
    mean, scale = feature_statistics(arrays["train"][0], sequence)
    loaders: dict[str, DataLoader] = {}
    for name, (values, meta) in arrays.items():
        dataset = StandardizedDataset(values, meta["label_rank"].to_numpy(), mean, scale)
        loaders[name] = DataLoader(dataset, batch_size=batch_size,
                                   shuffle=name == "train", num_workers=0)
    return loaders, mean, scale


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Run deterministic CPU inference."""
    model.eval(); outputs: list[np.ndarray] = []
    with torch.no_grad():
        for values, _ in loader:
            outputs.append(model(values.to(device)).cpu().numpy())
    return np.concatenate(outputs)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: Any,
                device: torch.device) -> float:
    """Run one MSE training epoch."""
    model.train(); total, observations = 0.0, 0
    loss_function = nn.MSELoss()
    for values, target in loader:
        values, target = values.to(device), target.to(device)
        optimizer.zero_grad(); prediction = model(values)
        loss = loss_function(prediction, target); loss.backward(); optimizer.step()
        total += float(loss) * len(target); observations += len(target)
    return total / observations


def model_hash(model: nn.Module) -> str:
    """Hash initialized parameters before any optimizer step."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8")); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit_neural(model: nn.Module, loaders: dict[str, DataLoader],
               arrays: dict[str, tuple[np.ndarray, pd.DataFrame]],
               device: torch.device) -> pd.DataFrame:
    """Early-stop on delayed-entry validation RankIC and restore the best epoch."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=NEURAL_LEARNING_RATE,
                                  weight_decay=NEURAL_WEIGHT_DECAY)
    scorers = {name: RankICScorer(arrays[name][1]) for name in ("train", "valid")}
    train_evaluation = DataLoader(loaders["train"].dataset,
                                  batch_size=loaders["train"].batch_size, shuffle=False)
    rows: list[dict[str, float]] = []; best_score = -np.inf; patience = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, NEURAL_MAX_EPOCHS + 1):
        loss = train_epoch(model, loaders["train"], optimizer, device)
        train_ic = scorers["train"].score(predict(model, train_evaluation, device))
        valid_ic = scorers["valid"].score(predict(model, loaders["valid"], device))
        rows.append({"epoch": epoch, "train_loss": loss,
                     "train_rank_ic": train_ic, "valid_rank_ic": valid_ic})
        print(f"epoch={epoch} loss={loss:.5f} train_ic={train_ic:.4f} val_ic={valid_ic:.4f}",
              flush=True)
        if valid_ic > best_score:
            best_score, patience = valid_ic, 0; best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= NEURAL_EARLY_STOP_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(rows)


def score_predictions(model: nn.Module, loader: DataLoader,
                      meta: pd.DataFrame) -> pd.DataFrame:
    """Build a stable prediction artifact with sample identifiers."""
    scored = meta[["code", "date", "time", "fwd_ret_30m",
                   "fwd_ret_30m_entry_lag1"]].copy()
    scored["prediction"] = predict(model, loader, next(model.parameters()).device)
    return scored


def evaluation_row(model: nn.Module, loader: DataLoader, meta: pd.DataFrame,
                   name: str, seed: int, window_id: int, curve: pd.DataFrame,
                   device: torch.device) -> dict[str, object]:
    """Report canonical diagnostics and the preregistered delayed-entry metrics."""
    scored = score_predictions(model, loader, meta)
    canonical, _, _ = evaluate_predictions(scored[["date", "time", "fwd_ret_30m",
                                                    "prediction"]])
    delayed = scored.drop(columns="fwd_ret_30m").rename(
        columns={"fwd_ret_30m_entry_lag1": "fwd_ret_30m"}).dropna()
    main, _, deciles = evaluate_predictions(delayed)
    result: dict[str, object] = {"window": window_id, "seed": seed, "model": name,
        "rank_ic": main["rank_ic"], "icir": main["icir"],
        "decile_long_short_return": main["decile_long_short_return"],
        "canonical_rank_ic": canonical["rank_ic"], "best_epoch":
        int(curve.loc[curve["valid_rank_ic"].idxmax(), "epoch"]),
        "max_train_rank_ic": curve["train_rank_ic"].max(),
        "max_valid_rank_ic": curve["valid_rank_ic"].max(),
        "overfit_alert": curve["train_rank_ic"].max() > 0.15
        and curve["valid_rank_ic"].max() < 0.05}
    for _, row in deciles.iterrows():
        result[f"decile_{int(row['decile'])}"] = row["fwd_ret_30m"]
    return result


def plot_curve(curve: pd.DataFrame, path: Path, title: str) -> None:
    """Render the requested train-versus-validation overfitting diagnostic."""
    plt.figure(figsize=(8, 4.5)); plt.plot(curve["epoch"], curve["train_rank_ic"], label="train")
    plt.plot(curve["epoch"], curve["valid_rank_ic"], label="validation")
    plt.axhline(0, color="black", linewidth=0.7); plt.xlabel("Epoch"); plt.ylabel("Delayed RankIC")
    plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def train_model(name: str, arrays: dict[str, tuple[np.ndarray, pd.DataFrame]],
                seed: int, window_id: int, output: Path,
                device: torch.device) -> dict[str, object]:
    """Train one seed and checkpoint its curve immediately."""
    set_seed(seed); sequence = name == "GRU"
    batch = GRU_BATCH_SIZE if sequence else MLP_BATCH_SIZE
    loaders, _, _ = make_loaders(arrays, batch, sequence)
    inputs = arrays["train"][0].shape[-1]
    model = (GRURegressor(inputs) if sequence else MLP(inputs)).to(device)
    initial_hash = model_hash(model)
    curve = fit_neural(model, loaders, arrays, device)
    stem = f"window_{window_id}_{name.lower()}_seed_{seed}"
    curve.to_csv(output / f"{stem}_curve.csv", index=False)
    plot_curve(curve, output / f"{stem}_curve.png", stem)
    if getattr(train_model, "save_artifacts", False):
        artifact = output / "artifacts"; artifact.mkdir(exist_ok=True)
        scored = score_predictions(model, loaders["test"], arrays["test"][1])
        scored.to_parquet(artifact / f"{stem}_predictions.parquet", index=False)
        _, mean, scale = make_loaders(arrays, batch, sequence)
        torch.save({"state_dict": model.state_dict(), "mean": mean, "scale": scale,
                    "inputs": inputs, "model": name, "seed": seed,
                    "window": window_id, "initial_weight_hash": initial_hash,
                    "first_epoch_loss": float(curve.iloc[0]["train_loss"])},
                   artifact / f"{stem}.pt")
    row = evaluation_row(model, loaders["test"], arrays["test"][1],
                         name, seed, window_id, curve, device)
    row["initial_weight_hash"] = initial_hash
    row["first_epoch_loss"] = float(curve.iloc[0]["train_loss"])
    return row


def completed_keys(path: Path) -> set[tuple[int, int, str]]:
    """Support interruption-safe Kaggle restarts."""
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    return set(zip(frame["window"], frame["seed"], frame["model"]))


def append_result(path: Path, row: dict[str, object]) -> None:
    """Atomically checkpoint a completed model/seed/window row."""
    current = pd.read_csv(path) if path.exists() else pd.DataFrame()
    updated = pd.concat([current, pd.DataFrame([row])], ignore_index=True)
    temporary = path.with_suffix(".tmp.csv"); updated.to_csv(temporary, index=False)
    temporary.replace(path)


def write_summaries(path: Path, output: Path) -> None:
    """Refresh per-window seed statistics and the four-model comparison table."""
    neural = pd.read_csv(path)
    aggregations: dict[str, tuple[str, str]] = {
        "seeds": ("seed", "nunique"), "rank_ic_mean": ("rank_ic", "mean"),
        "rank_ic_std": ("rank_ic", "std"), "icir_mean": ("icir", "mean"),
        "icir_std": ("icir", "std"),
        "long_short_mean": ("decile_long_short_return", "mean"),
        "long_short_std": ("decile_long_short_return", "std")}
    for decile in range(1, 11):
        column = f"decile_{decile}"
        if column in neural:
            aggregations[f"{column}_mean"] = (column, "mean")
            aggregations[f"{column}_std"] = (column, "std")
    grouped = neural.groupby(["window", "model"], as_index=False).agg(**aggregations)
    grouped.to_csv(output / "neural_seed_summary.csv", index=False)
    baseline_path = BASELINE_REPORT_DIR / "window_metrics.csv"
    if not baseline_path.exists():
        return
    baseline = pd.read_csv(baseline_path)
    required = {"entry_lag1_rank_ic", "entry_lag1_icir",
                "entry_lag1_decile_long_short_return"}
    if not required.issubset(baseline.columns):
        return
    rename = {"entry_lag1_rank_ic": "rank_ic_mean",
        "entry_lag1_icir": "icir_mean",
        "entry_lag1_decile_long_short_return": "long_short_mean"}
    rename.update({f"decile_{value}": f"decile_{value}_mean" for value in range(1, 11)})
    base = baseline.rename(columns=rename)
    columns = ["window", "model", "test_start", "test_end", "rank_ic_mean",
               "icir_mean", "long_short_mean", *[f"decile_{d}_mean" for d in range(1, 11)]]
    base = base[[column for column in columns if column in base]]
    base["seeds"], base["rank_ic_std"], base["icir_std"], base["long_short_std"] = 1, 0.0, 0.0, 0.0
    dates = base[["window", "test_start", "test_end"]].drop_duplicates("window")
    grouped = grouped.merge(dates, on="window", how="left")
    comparison = pd.concat([base, grouped], ignore_index=True).sort_values(["window", "model"])
    comparison.to_csv(output / "four_model_comparison.csv", index=False)
    plot_model_summaries(comparison, output)


def plot_model_summaries(table: pd.DataFrame, output: Path) -> None:
    """Render window chronology and full ten-decile model profiles."""
    plt.figure(figsize=(10, 5)); table = table.copy()
    table["test_midpoint"] = pd.to_datetime(table["test_start"]) + (
        pd.to_datetime(table["test_end"]) - pd.to_datetime(table["test_start"])) / 2
    for model, group in table.groupby("model"):
        plt.errorbar(group["test_midpoint"], group["rank_ic_mean"],
            yerr=group["rank_ic_std"].fillna(0), marker="o", capsize=3, label=model)
    plt.axvspan(pd.Timestamp(SCALING_FACTOR_START_DATE), pd.Timestamp(SCALING_FACTOR_END_DATE),
                color="gray", alpha=0.15, label="volume regime")
    plt.axhline(0, color="black", linewidth=0.7); plt.ylabel("Delayed RankIC")
    plt.legend(); plt.tight_layout(); plt.savefig(output / "rankic_by_window.png", dpi=150); plt.close()
    deciles = [f"decile_{value}_mean" for value in range(1, 11)]
    if not set(deciles).issubset(table.columns):
        return
    plt.figure(figsize=(8, 5))
    for model, group in table.groupby("model"):
        plt.plot(range(1, 11), group[deciles].mean().to_numpy(), marker="o", label=model)
    plt.axhline(0, color="black", linewidth=0.7); plt.xlabel("Prediction decile")
    plt.ylabel("Mean delayed 30m return"); plt.legend(); plt.tight_layout()
    plt.savefig(output / "four_model_decile_curve.png", dpi=150); plt.close()


def write_full_seed_summary(neural: pd.DataFrame, output: Path) -> None:
    """List model seed volatility after averaging each seed across windows."""
    by_seed = neural.groupby(["model", "seed"], as_index=False).agg(
        windows=("window", "nunique"), rank_ic=("rank_ic", "mean"),
        icir=("icir", "mean"), long_short=("decile_long_short_return", "mean"))
    summary = by_seed.groupby("model", as_index=False).agg(seeds=("seed", "nunique"),
        windows_min=("windows", "min"), rank_ic_mean=("rank_ic", "mean"),
        rank_ic_seed_std=("rank_ic", "std"), icir_mean=("icir", "mean"),
        long_short_mean=("long_short", "mean"))
    by_seed.to_csv(output / "full_period_by_seed.csv", index=False)
    summary.to_csv(output / "full_period_seed_summary.csv", index=False)


def write_gru_gate(output: Path) -> None:
    """Apply the preregistered increment and five-seed stability rule."""
    path = output / "four_model_comparison.csv"
    if not path.exists():
        return
    table = pd.read_csv(path); rows: list[dict[str, object]] = []
    for window, group in table.groupby("window"):
        lgb = group[group["model"].eq("LightGBM")]
        gru = group[group["model"].eq("GRU")]
        if lgb.empty or gru.empty:
            continue
        increment = float(gru.iloc[0]["rank_ic_mean"] - lgb.iloc[0]["rank_ic_mean"])
        seed_std = float(gru.iloc[0]["rank_ic_std"])
        stable = int(gru.iloc[0]["seeds"]) == len(STAGE3_SEEDS)
        stable &= seed_std <= increment
        rows.append({"window": window, "gru_increment": increment,
            "gru_seed_std": seed_std, "seed_std_not_greater_than_increment": seed_std <= increment,
            "five_seed_stable": stable, "increment_gate_pass":
            increment >= GRU_INCREMENT_RANKIC_GATE, "sequence_value_add_pass":
            stable and increment >= GRU_INCREMENT_RANKIC_GATE})
    pd.DataFrame(rows).to_csv(output / "gru_increment_gate.csv", index=False)


def write_full_gru_gate(output: Path) -> None:
    """Apply the same rule after averaging each seed across all windows."""
    seed_path = output / "full_period_seed_summary.csv"
    base_path = BASELINE_REPORT_DIR / "full_period_summary.csv"
    if not seed_path.exists() or not base_path.exists():
        return
    neural, baseline = pd.read_csv(seed_path), pd.read_csv(base_path)
    gru = neural[neural["model"].eq("GRU")]
    lgb = baseline[baseline["model"].eq("LightGBM")]
    if gru.empty or lgb.empty:
        return
    increment = float(gru.iloc[0]["rank_ic_mean"] - lgb.iloc[0]["mean_entry_lag1_rank_ic"])
    seed_std = float(gru.iloc[0]["rank_ic_seed_std"])
    complete = int(gru.iloc[0]["seeds"]) == len(STAGE3_SEEDS)
    complete &= int(gru.iloc[0]["windows_min"]) == 8
    row = {"gru_increment": increment, "gru_seed_std": seed_std,
        "five_seeds_eight_windows": complete,
        "seed_std_not_greater_than_increment": seed_std <= increment,
        "increment_gate_pass": increment >= GRU_INCREMENT_RANKIC_GATE,
        "sequence_value_add_pass": complete and seed_std <= increment
        and increment >= GRU_INCREMENT_RANKIC_GATE}
    pd.DataFrame([row]).to_csv(output / "full_period_gru_gate.csv", index=False)


def run(args: argparse.Namespace) -> None:
    """Run selected windows and seeds; reload sequence data one window at a time."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "neural_metrics.csv"
    if args.summarize_only:
        neural = pd.read_csv(result_path); write_full_seed_summary(neural, args.output_dir)
        write_summaries(result_path, args.output_dir); write_gru_gate(args.output_dir)
        write_full_gru_gate(args.output_dir)
        return
    device = torch.device(args.device)
    train_model.save_artifacts = args.save_artifacts
    flat = load_flat_dataset(args.flat_dir, args.feature_dir)
    windows = rolling_windows(flat["date"])
    done = completed_keys(result_path)
    for window_id in args.windows:
        window = windows[window_id - 1]; parts = split_frame(flat, window)
        cached: dict[str, dict[str, tuple[np.ndarray, pd.DataFrame]]] = {}
        for model_name in args.models:
            cached[model_name] = (flat_arrays(parts) if model_name == "MLP" else
                                  sequence_arrays(flat, window, args.seq_dir))
            for seed in args.seeds:
                if (window_id, seed, model_name) in done:
                    continue
                row = train_model(model_name, cached[model_name], seed, window_id,
                                  args.output_dir, device)
                append_result(result_path, row); write_full_seed_summary(
                    pd.read_csv(result_path), args.output_dir)
                write_summaries(result_path, args.output_dir)
                write_gru_gate(args.output_dir); write_full_gru_gate(args.output_dir)
                print(row, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=["MLP", "GRU"], default=["MLP", "GRU"])
    parser.add_argument("--windows", nargs="+", type=int, default=[1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[STAGE3_SEEDS[0]])
    parser.add_argument("--flat-dir", type=Path, default=FLAT_DATASET_DIR)
    parser.add_argument("--feature-dir", type=Path, default=FEATURES_5MIN_DIR)
    parser.add_argument("--seq-dir", type=Path, default=SEQ_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=STAGE3_REPORT_DIR)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--save-artifacts", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
