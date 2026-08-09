"""Stage-4 seed audit, failed-window context, and transaction-cost analysis."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import (DAILY_MIN_SIGNALS, DAILY_REFERENCE_DIR, FAILED_GRU_WINDOWS,
                    SCALING_FACTOR_END_DATE, SCALING_FACTOR_START_DATE,
                    STAGE3_SEEDS, STAGE4_REPORT_DIR, TRANSACTION_COST_BPS)


def file_sha256(path: Path) -> str:
    """Hash a model artifact for provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_seeds(artifact_dir: Path, output: Path) -> pd.DataFrame:
    """Verify five GRU seeds using recorded pre-training hashes and first losses."""
    rows = []
    for path in sorted(artifact_dir.glob("window_*_gru_seed_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows.append({"window": payload["window"], "seed": payload["seed"],
                     "initial_weight_hash": payload["initial_weight_hash"],
                     "first_epoch_loss": payload["first_epoch_loss"],
                     "checkpoint_sha256": file_sha256(path)})
    frame = pd.DataFrame(rows)
    checks = frame.groupby("window").agg(seeds=("seed", "nunique"),
        unique_initial_hashes=("initial_weight_hash", "nunique"),
        unique_first_epoch_losses=("first_epoch_loss", "nunique")).reset_index()
    checks["five_seeds_present"] = checks["seeds"].eq(len(STAGE3_SEEDS))
    checks["hashes_independent"] = checks["unique_initial_hashes"].eq(checks["seeds"])
    checks["losses_independent"] = checks["unique_first_epoch_losses"].gt(1)
    frame.to_csv(output / "gru_seed_evidence.csv", index=False)
    checks.to_csv(output / "gru_seed_independence_audit.csv", index=False)
    if not checks[["five_seeds_present", "hashes_independent", "losses_independent"]].all().all():
        raise AssertionError("GRU seed independence audit failed")
    return checks


def load_predictions(baseline_dir: Path, neural_dir: Path) -> pd.DataFrame:
    """Load four-model predictions, averaging neural predictions across seeds."""
    tables = []
    for path in sorted(baseline_dir.glob("window_*_predictions.parquet")):
        model = "LightGBM" if "lightgbm" in path.name else "Ridge"
        frame = pd.read_parquet(path); frame["model"] = model
        frame["window"] = int(path.name.split("_")[1]); tables.append(frame)
    neural = []
    for path in sorted(neural_dir.glob("window_*_predictions.parquet")):
        frame = pd.read_parquet(path); pieces = path.stem.split("_")
        frame["window"], frame["model"] = int(pieces[1]), pieces[2].upper()
        neural.append(frame)
    if neural:
        keys = ["window", "model", "code", "date", "time", "fwd_ret_30m",
                "fwd_ret_30m_entry_lag1"]
        ensemble = pd.concat(neural).groupby(keys, as_index=False, dropna=False).agg(
            prediction=("prediction", "mean"), seeds=("prediction", "size"))
        if ensemble["seeds"].min() != len(STAGE3_SEEDS):
            raise AssertionError("neural prediction ensemble is missing seeds")
        tables.append(ensemble.drop(columns="seeds"))
    if not tables:
        raise FileNotFoundError("no prediction artifacts found")
    return pd.concat(tables, ignore_index=True)


def load_daily_predictions(directory: Path) -> pd.DataFrame:
    """Load six-signal panels and average neural seeds."""
    baseline, neural = [], []
    for path in sorted(directory.glob("*_six_signals.parquet")):
        pieces = path.stem.split("_"); frame = pd.read_parquet(path)
        frame["window"] = int(pieces[1]); model = pieces[2]
        frame["model"] = {"lightgbm": "LightGBM", "ridge": "Ridge",
                          "mlp": "MLP", "gru": "GRU"}[model]
        (neural if model in {"mlp", "gru"} else baseline).append(frame)
    tables = baseline
    if neural:
        keys = ["window", "model", "code", "date", "time"]
        ensemble = pd.concat(neural).groupby(keys, as_index=False).agg(
            prediction=("prediction", "mean"), seeds=("prediction", "size"))
        if ensemble["seeds"].min() != len(STAGE3_SEEDS):
            raise AssertionError("six-signal neural ensemble is missing seeds")
        tables.append(ensemble.drop(columns="seeds"))
    return pd.concat(tables, ignore_index=True)


def assign_extremes(group: pd.DataFrame) -> tuple[set[str], set[str]]:
    """Return equal-count bottom and top decile memberships."""
    ordered = group.sort_values(["prediction", "code"])
    count = max(1, len(ordered) // 10)
    return set(ordered.head(count)["code"]), set(ordered.tail(count)["code"])


def leg_turnover(current: set[str], previous: set[str] | None) -> tuple[float, float]:
    """Calculate equal-weight one-way turnover and membership overlap."""
    if previous is None or not current:
        return 1.0, 0.0
    names = current | previous
    distance = sum(abs((1 / len(current) if code in current else 0) -
                       (1 / len(previous) if code in previous else 0)) for code in names)
    overlap = len(current & previous) / len(current)
    return 0.5 * distance, overlap


def intraday_portfolios(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build lag-one 30-minute long-short periods and realized turnover."""
    rows = []
    for (window, model), panel in predictions.groupby(["window", "model"]):
        prior_time = None; prior_low = prior_high = None
        for timestamp, group in panel.dropna(subset=["fwd_ret_30m_entry_lag1"]).groupby("time"):
            low, high = assign_extremes(group)
            consecutive = prior_time is not None and timestamp - prior_time == pd.Timedelta(minutes=30)
            low_turn, low_overlap = leg_turnover(low, prior_low if consecutive else None)
            high_turn, high_overlap = leg_turnover(high, prior_high if consecutive else None)
            returns = group.set_index("code")["fwd_ret_30m_entry_lag1"]
            gross = returns.reindex(list(high)).mean() - returns.reindex(list(low)).mean()
            rows.append({"window": window, "model": model, "time": timestamp,
                "gross_return": gross, "low_one_way_turnover": low_turn,
                "high_one_way_turnover": high_turn, "low_overlap": low_overlap,
                "high_overlap": high_overlap,
                "two_leg_buy_sell_turnover": 2 * (low_turn + high_turn)})
            prior_time, prior_low, prior_high = timestamp, low, high
    return pd.DataFrame(rows)


def daily_close_returns(directory: Path) -> pd.DataFrame:
    """Build next-close returns from the unadjusted companion daily table."""
    rows = []
    for path in directory.glob("*.parquet"):
        frame = pd.read_parquet(path, columns=["date", "code", "close", "tradestatus"])
        frame = frame[frame["tradestatus"].eq(1)].sort_values("date")
        frame["next_close_return"] = np.log(frame["close"].shift(-1) / frame["close"])
        rows.append(frame[["date", "code", "next_close_return"]])
    return pd.concat(rows, ignore_index=True)


def daily_portfolios(predictions: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Average six intraday signals, then hold close-to-next-close."""
    signal = predictions.groupby(["window", "model", "code", "date"], as_index=False).agg(
        prediction=("prediction", "mean"), signals=("prediction", "size"))
    signal = signal[signal["signals"].ge(DAILY_MIN_SIGNALS)].merge(
        daily, on=["code", "date"], how="left").dropna(subset=["next_close_return"])
    rows = []
    for (window, model), panel in signal.groupby(["window", "model"]):
        prior_low = prior_high = None
        for date, group in panel.groupby("date"):
            low, high = assign_extremes(group)
            low_turn, low_overlap = leg_turnover(low, prior_low)
            high_turn, high_overlap = leg_turnover(high, prior_high)
            returns = group.set_index("code")["next_close_return"]
            rows.append({"window": window, "model": model, "time": date,
                "gross_return": returns.reindex(list(high)).mean() - returns.reindex(list(low)).mean(),
                "low_one_way_turnover": low_turn, "high_one_way_turnover": high_turn,
                "low_overlap": low_overlap, "high_overlap": high_overlap,
                "two_leg_buy_sell_turnover": 2 * (low_turn + high_turn)})
            prior_low, prior_high = low, high
    return pd.DataFrame(rows)


def cost_summary(periods: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Summarize gross, cost-adjusted returns, turnover, and breakeven cost."""
    rows = []
    for (window, model), group in periods.groupby(["window", "model"]):
        gross, turnover = group["gross_return"].mean(), group["two_leg_buy_sell_turnover"].mean()
        row = {"frequency": frequency, "window": window, "model": model,
               "periods": len(group), "gross_return": gross,
               "mean_two_leg_buy_sell_turnover": turnover,
               "mean_membership_overlap": group[["low_overlap", "high_overlap"]].mean().mean(),
               "breakeven_one_way_bps": gross / turnover * 10_000 if turnover else np.nan}
        for bps in TRANSACTION_COST_BPS:
            row[f"net_{bps}bps"] = (group["gross_return"] -
                                     group["two_leg_buy_sell_turnover"] * bps / 10_000).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def market_context(index: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    """Create the eight-point descriptive window-state comparison."""
    data = index.sort_values("date").copy()
    data["return"] = np.log(data["close"] / data["close"].shift())
    data["volatility_20d"] = data["return"].rolling(20, min_periods=20).std()
    pivot = comparison.pivot(index=["window", "test_start", "test_end"],
                             columns="model", values="rank_ic_mean").reset_index()
    rows = []
    regime_start, regime_end = pd.Timestamp(SCALING_FACTOR_START_DATE), pd.Timestamp(SCALING_FACTOR_END_DATE)
    for _, window in pivot.iterrows():
        start, end = pd.Timestamp(window["test_start"]), pd.Timestamp(window["test_end"])
        sample = data[data["date"].between(start, end)]
        overlap_start, overlap_end = max(start, regime_start), min(end, regime_end)
        overlap = max(0, (overlap_end - overlap_start).days + 1) / ((end - start).days + 1)
        rows.append({"window": window["window"], "test_start": start, "test_end": end,
            "gru_minus_lightgbm": window["GRU"] - window["LightGBM"],
            "gru_gate_pass": int(window["window"]) not in FAILED_GRU_WINDOWS,
            "index_log_return": sample["return"].sum(),
            "mean_20d_volatility": sample["volatility_20d"].mean(),
            "regime_calendar_overlap_share": overlap})
    return pd.DataFrame(rows)


def plot_context(context: pd.DataFrame, output: Path) -> None:
    """Plot descriptive GRU increment against the two market-state variables."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, column, label in zip(axes, ["mean_20d_volatility", "index_log_return"],
                                  ["Mean 20d index volatility", "Index log return"]):
        colors = np.where(context["gru_gate_pass"], "tab:blue", "tab:red")
        axis.scatter(context[column], context["gru_minus_lightgbm"], c=colors)
        for _, row in context.iterrows():
            axis.annotate(f"W{int(row['window'])}", (row[column], row["gru_minus_lightgbm"]))
        axis.axhline(0.005, linestyle="--", color="gray"); axis.set_xlabel(label)
        axis.set_ylabel("GRU - LightGBM RankIC")
    fig.tight_layout(); fig.savefig(output / "gru_increment_market_state.png", dpi=150); plt.close(fig)


def run(args: argparse.Namespace) -> None:
    """Generate all non-attribution Stage-4 reports from auditable artifacts."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_seeds(args.neural_artifact_dir, args.output_dir)
    predictions = load_predictions(args.baseline_artifact_dir, args.neural_artifact_dir)
    intraday = intraday_portfolios(predictions); intraday.to_parquet(
        args.output_dir / "intraday_portfolio_periods.parquet", index=False)
    daily_predictions = load_daily_predictions(args.daily_prediction_dir)
    daily = daily_portfolios(daily_predictions, daily_close_returns(args.daily_dir)); daily.to_parquet(
        args.output_dir / "daily_portfolio_periods.parquet", index=False)
    summary = pd.concat([cost_summary(intraday, "intraday_30m"),
                         cost_summary(daily, "daily_close")], ignore_index=True)
    summary.to_csv(args.output_dir / "cost_summary_by_window.csv", index=False)
    aggregate = summary.groupby(["frequency", "model"], as_index=False).mean(numeric_only=True)
    aggregate.to_csv(args.output_dir / "cost_summary_full_period.csv", index=False)
    index = pd.read_parquet(args.index_path) if args.index_path.suffix == ".parquet" else pd.read_csv(args.index_path)
    index["date"] = pd.to_datetime(index["date"])
    comparison = pd.read_csv(args.comparison_path)
    context = market_context(index, comparison); context.to_csv(
        args.output_dir / "window_market_context.csv", index=False)
    plot_context(context, args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-artifact-dir", type=Path, required=True)
    parser.add_argument("--neural-artifact-dir", type=Path, required=True)
    parser.add_argument("--comparison-path", type=Path, required=True)
    parser.add_argument("--daily-prediction-dir", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--daily-dir", type=Path, default=DAILY_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=STAGE4_REPORT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
