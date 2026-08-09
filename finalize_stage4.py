"""Build the publication-ready Stage-4 tables and figures from archived outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ("Ridge", "LightGBM", "MLP", "GRU")
COLORS = {"Ridge": "#6B7280", "LightGBM": "#D97706", "MLP": "#2563EB", "GRU": "#059669"}
LABELS = {"canonical": "Same-bar close", "delayed": "One-bar delayed entry"}


def set_style() -> None:
    """Apply one restrained style to every final figure."""
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 220, "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.2,
        "legend.frameon": False, "figure.facecolor": "white"})


def rank_ic(group: pd.DataFrame, target: str) -> float:
    """Compute cross-sectional Spearman correlation for one timestamp."""
    valid = group[["prediction", target]].dropna()
    if len(valid) < 30:
        return np.nan
    return float(valid["prediction"].rank().corr(valid[target].rank()))


def score_panel(panel: pd.DataFrame, target: str) -> tuple[dict[str, float], pd.DataFrame]:
    """Score a prediction panel and return timestamp-balanced decile returns."""
    panel = panel.dropna(subset=[target]).copy()
    keys = ["date", "time"]
    ic = panel.groupby(keys).apply(rank_ic, target=target, include_groups=False).dropna()
    panel["decile"] = panel.groupby(keys)["prediction"].rank(method="first", pct=True)
    panel["decile"] = np.ceil(panel["decile"] * 10).clip(1, 10).astype(int)
    per_time = panel.groupby(keys + ["decile"], as_index=False)[target].mean()
    deciles = per_time.groupby("decile", as_index=False)[target].mean()
    spread = deciles.set_index("decile").loc[10, target] - deciles.set_index("decile").loc[1, target]
    dispersion = ic.std(ddof=1)
    metrics = {"rank_ic": ic.mean(), "icir": ic.mean() / dispersion if dispersion else np.nan,
               "long_short": spread, "ic_observations": len(ic)}
    return metrics, deciles.rename(columns={target: "mean_return"})


def baseline_panel(root: Path, window: int, model: str) -> pd.DataFrame:
    """Load one baseline panel, including both realized-return definitions."""
    slug = model.lower()
    path = root / "stage4_baseline_clean" / "artifacts" / f"window_{window}_{slug}_predictions.parquet"
    return pd.read_parquet(path)


def neural_panels(root: Path, window: int, model: str) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Load neural seeds, attach baseline labels, and form the five-seed ensemble."""
    directory = root / "stage4_six_signal_clean"
    paths = sorted(directory.glob(f"window_{window}_{model.lower()}_seed_*_six_signals.parquet"))
    labels = baseline_panel(root, window, "LightGBM").drop(columns="prediction")
    seeds = [pd.read_parquet(path).merge(labels, on=["code", "date", "time"], how="inner")
             for path in paths]
    keyed = [frame.assign(seed=path.stem.split("_")[4]) for frame, path in zip(seeds, paths)]
    all_seeds = pd.concat(keyed, ignore_index=True)
    keys = ["code", "date", "time", "fwd_ret_30m", "fwd_ret_30m_entry_lag1"]
    ensemble = all_seeds.groupby(keys, as_index=False, dropna=False)["prediction"].mean()
    return ensemble, seeds


def build_model_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recompute four-model metrics, execution appendix, and delayed deciles."""
    rows, seed_rows, decile_rows = [], [], []
    for window in range(1, 9):
        for model in MODELS:
            if model in {"Ridge", "LightGBM"}:
                panel, seeds = baseline_panel(root, window, model), []
            else:
                panel, seeds = neural_panels(root, window, model)
            for protocol, target in (("canonical", "fwd_ret_30m"),
                                     ("delayed", "fwd_ret_30m_entry_lag1")):
                panels = seeds or [panel]; scored = [score_panel(item, target) for item in panels]
                values = pd.DataFrame([item[0] for item in scored])
                metrics = values.mean(numeric_only=True).to_dict()
                metrics["rank_ic_std"] = values["rank_ic"].std(ddof=1) if seeds else 0.0
                rows.append({"window": window, "model": model, "protocol": protocol,
                             "seeds": len(panels), **metrics})
                for seed_id, item in enumerate(scored, start=1):
                    seed_rows.append({"window": window, "model": model, "protocol": protocol,
                                      "seed_index": seed_id, **item[0]})
                    if protocol == "delayed":
                        deciles = item[1]; deciles["window"], deciles["model"] = window, model
                        deciles["seed_index"] = seed_id; decile_rows.append(deciles)
    return pd.DataFrame(rows), pd.DataFrame(seed_rows), pd.concat(decile_rows, ignore_index=True)


def plot_four_models(metrics: pd.DataFrame, seeds: pd.DataFrame, output: Path) -> None:
    """Plot delayed-entry RankIC by rolling test window."""
    delayed = metrics[metrics["protocol"].eq("delayed")]
    fig, axis = plt.subplots(figsize=(9.2, 5.0))
    for model in MODELS:
        group = delayed[delayed["model"].eq(model)].sort_values("window")
        errors = group["rank_ic_std"].fillna(0).to_numpy()
        axis.errorbar(group["window"], group["rank_ic"], yerr=errors, color=COLORS[model],
                      marker="o", linewidth=1.8, capsize=3, label=model)
    axis.axhline(0, color="#111827", linewidth=0.8)
    axis.set(title="Four-model out-of-sample RankIC", xlabel="Rolling test window",
             ylabel="Mean cross-sectional RankIC (one-bar delayed entry)", xticks=range(1, 9))
    axis.legend(ncol=4, loc="upper left"); fig.tight_layout()
    fig.savefig(output / "four_model_rankic_by_window.png", bbox_inches="tight"); plt.close(fig)


def plot_deciles(deciles: pd.DataFrame, output: Path) -> None:
    """Plot timestamp-balanced delayed returns for all ten prediction deciles."""
    averaged = deciles.groupby(["model", "decile"], as_index=False)["mean_return"].mean()
    fig, axis = plt.subplots(figsize=(9.2, 5.0))
    for model in MODELS:
        group = averaged[averaged["model"].eq(model)]
        axis.plot(group["decile"], group["mean_return"] * 10_000, color=COLORS[model],
                  marker="o", linewidth=1.8, label=model)
    axis.axhline(0, color="#111827", linewidth=0.8)
    axis.set(title="Delayed 30-minute return by prediction decile", xlabel="Prediction decile",
             ylabel="Mean forward return (bp)", xticks=range(1, 11))
    axis.legend(ncol=4); fig.tight_layout()
    fig.savefig(output / "four_model_decile_returns.png", bbox_inches="tight"); plt.close(fig)


def plot_costs(root: Path, output: Path) -> None:
    """Plot net long-short returns as a function of assumed one-way cost."""
    path = root / "stage4_final_report" / "cost_summary_full_period.csv"
    summary = pd.read_csv(path); costs = np.linspace(0, 20, 81)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), sharey=True)
    for axis, frequency, title in zip(axes, ["intraday_30m", "daily_close"],
                                     ["30-minute holding", "Daily aggregation"]):
        for model in MODELS:
            row = summary[(summary["frequency"].eq(frequency)) & (summary["model"].eq(model))].iloc[0]
            net = row["gross_return"] * 10_000 - row["mean_two_leg_buy_sell_turnover"] * costs
            axis.plot(costs, net, color=COLORS[model], linewidth=1.8, label=model)
        axis.axhline(0, color="#111827", linewidth=0.8); axis.set_title(title)
        axis.set_xlabel("Assumed one-way cost (bp)")
    axes[0].set_ylabel("Net long-short return per rebalance (bp)")
    axes[0].legend(ncol=2); fig.suptitle("Transaction-cost stress test", y=1.01)
    fig.tight_layout(); fig.savefig(output / "four_model_cost_curves.png", bbox_inches="tight"); plt.close(fig)


def plot_attribution(root: Path, output: Path) -> None:
    """Plot mean absolute integrated gradients across the eight windows."""
    path = root / "stage4_attribution_clean" / "gru_attribution_all_windows.csv"
    frame = pd.read_csv(path); features = [c for c in frame if c not in {"window", "lookback_step"}]
    mean = frame.groupby("lookback_step")[features].mean().sort_index()
    shares = mean / mean.to_numpy().sum() * 100
    fig, axis = plt.subplots(figsize=(11.0, 5.2))
    image = axis.imshow(shares.T, aspect="auto", origin="lower", cmap="YlGnBu")
    ticks = np.arange(0, 48, 6); axis.set_xticks(ticks, mean.index.to_numpy()[ticks])
    axis.set_yticks(range(len(features)), features)
    axis.set(title="GRU integrated-gradients attribution", xlabel="Lookback bar (0 = signal bar)",
             ylabel="Sequence feature")
    colorbar = fig.colorbar(image, ax=axis, pad=0.02); colorbar.set_label("Share of total attribution (%)")
    fig.tight_layout(); fig.savefig(output / "gru_integrated_gradients_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def window_context(root: Path, tables: Path) -> pd.DataFrame:
    """Save the predeclared failed-window table plus passed-window benchmarks."""
    source = pd.read_csv(root / "stage4_final_report" / "window_market_context.csv")
    passed = source[source["gru_gate_pass"].astype(bool)].copy()
    selected = source[source["window"].isin([2, 8])].copy()
    selected["passed_window_mean_gru_increment"] = passed["gru_minus_lightgbm"].mean()
    selected["passed_window_mean_index_return"] = passed["index_log_return"].mean()
    selected["passed_window_mean_volatility"] = passed["mean_20d_volatility"].mean()
    selected.to_csv(tables / "window_2_8_market_context.csv", index=False)
    return source


def plot_context(context: pd.DataFrame, output: Path) -> None:
    """Plot the eight descriptive observations without fitting a regression."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    for axis, column, label in zip(axes, ["mean_20d_volatility", "index_log_return"],
                                  ["Mean 20-day index volatility", "Index log return"]):
        colors = np.where(context["gru_gate_pass"], "#2563EB", "#DC2626")
        axis.scatter(context[column], context["gru_minus_lightgbm"], c=colors, s=55)
        for _, row in context.iterrows():
            axis.annotate(f"W{int(row['window'])}", (row[column], row["gru_minus_lightgbm"]),
                          xytext=(4, 4), textcoords="offset points")
        axis.axhline(0.005, color="#6B7280", linestyle="--", linewidth=1)
        axis.set(xlabel=label, ylabel="GRU − LightGBM RankIC")
    fig.suptitle("GRU increment and market state: descriptive eight-window comparison")
    fig.tight_layout(); fig.savefig(output / "gru_increment_market_state.png", bbox_inches="tight")
    plt.close(fig)


def save_tables(metrics: pd.DataFrame, seeds: pd.DataFrame, deciles: pd.DataFrame,
                tables: Path) -> None:
    """Persist the exact data behind the final figures and Appendix A."""
    metrics.to_csv(tables / "four_model_two_protocol_metrics.csv", index=False)
    seeds.to_csv(tables / "model_seed_metrics.csv", index=False)
    deciles.to_csv(tables / "four_model_delayed_deciles.csv", index=False)
    appendix = metrics.groupby(["model", "protocol"], as_index=False).agg(
        windows=("window", "nunique"), mean_rank_ic=("rank_ic", "mean"),
        mean_icir=("icir", "mean"), mean_long_short=("long_short", "mean"),
        min_rank_ic=("rank_ic", "min"), max_rank_ic=("rank_ic", "max"))
    appendix.to_csv(tables / "appendix_a_execution_protocols.csv", index=False)


def run(root: Path, figures: Path, tables: Path) -> None:
    """Create all final Stage-4 engineering handoff artifacts."""
    figures.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    set_style(); metrics, seeds, deciles = build_model_tables(root)
    save_tables(metrics, seeds, deciles, tables)
    plot_four_models(metrics, seeds, figures); plot_deciles(deciles, figures)
    plot_costs(root, figures); plot_attribution(root, figures)
    context = window_context(root, tables); plot_context(context, figures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path,
        default=Path("data/archive/stage4_dataset_v17/files"))
    parser.add_argument("--figures-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--tables-dir", type=Path, default=Path("docs/tables"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args(); run(args.archive_root, args.figures_dir, args.tables_dir)
