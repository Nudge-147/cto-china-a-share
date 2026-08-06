"""Run the frozen nonlinear CTO extension analysis.

Inputs
------
``data/cto_baostock/daily_cto.csv`` (the main-analysis eligible panel),
``daily_raw/*.csv`` (turnover), and ``market_caps_monthly.csv`` (the
step-smoothed float market cap used by the main analysis).

Outputs
-------
Feature-cache data under ``data/cto_baostock/ml_extension/`` and all public
tables/figures under ``results/ml_extension/``.

Pipeline position
-----------------
This is an isolated extension after the frozen main CTO backtest. It compares
the original CTO ranking, a five-feature CTO-family Ridge, an eight-feature
Ridge, and a prespecified LightGBM model in the same expanding-window, strictly
chronological prediction exercise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from backtest_cto import nw_mean_t


BASE = Path("data/cto_baostock")
DEFAULT_CACHE = BASE / "ml_extension" / "feature_panel.csv.gz"
DEFAULT_OUTPUT = Path("results/ml_extension")
FEATURES = [
    "cto_mean",
    "cto_std",
    "cto_min",
    "cto_max",
    "intraday_mean",
    "month_cumulative_return",
    "turnover_mean",
    "log_float_market_cap",
]
CTO_FAMILY_FEATURES = FEATURES[:5]
MODELS = ("cto_sort", "ridge_cto_family", "ridge", "lightgbm")
MODEL_LABELS = {
    "cto_sort": "CTO sort",
    "ridge_cto_family": "Ridge: CTO family (1-5)",
    "ridge": "Ridge: all 8 features",
    "lightgbm": "LightGBM: all 8 features",
}
PERIODS = {
    "full_available": ("1900-01", "2100-12"),
    "paper_overlap_2010_2020": ("2010-01", "2020-12"),
    "out_of_sample_2021_2026": ("2021-01", "2026-12"),
}
LIGHTGBM_PARAMS = {
    "num_leaves": 31,
    "max_depth": 5,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}


def _as_bool(series: pd.Series) -> pd.Series:
    """Parse persisted boolean fields without relying on CSV inference."""
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def _complete_code_blocks(path: Path, chunksize: int = 500_000) -> Iterator[pd.DataFrame]:
    """Yield code-complete blocks from the code-sorted main daily panel."""
    columns = ["date", "code", "open", "close", "prev_close", "cto_daily", "eligible"]
    carry = pd.DataFrame()
    for chunk in pd.read_csv(path, usecols=columns, dtype={"code": str}, chunksize=chunksize):
        chunk["code"] = chunk["code"].str.zfill(6)
        if not carry.empty:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        final_code = chunk["code"].iloc[-1]
        complete = chunk.loc[chunk["code"].ne(final_code)].copy()
        carry = chunk.loc[chunk["code"].eq(final_code)].copy()
        if not complete.empty:
            yield complete
    if not carry.empty:
        yield carry


def _one_code_features(daily: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """Construct the seven daily-data features and next-month raw return."""
    code = str(daily["code"].iloc[0]).zfill(6)
    daily = daily.sort_values("date").copy()
    daily["date"] = pd.to_datetime(daily["date"])
    for column in ("open", "close", "prev_close", "cto_daily"):
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    daily["month"] = daily["date"].dt.to_period("M")

    # The return target exactly follows the main backtest's adjusted month-end
    # close-to-close convention. Require the next observed month to be t+1.
    endpoints = daily.groupby("month", as_index=False).tail(1).sort_values("month")
    endpoints["next_close"] = endpoints["close"].shift(-1)
    endpoints["target_end_date"] = endpoints["date"].shift(-1)
    endpoints["next_observed_month"] = endpoints["month"].shift(-1)
    endpoints["raw_next_month_return"] = endpoints["next_close"] / endpoints["close"] - 1
    consecutive = endpoints["next_observed_month"].eq(endpoints["month"] + 1)
    endpoints.loc[~consecutive, "raw_next_month_return"] = np.nan
    targets = endpoints.rename(columns={"date": "target_start_date"})[
        ["month", "raw_next_month_return", "target_start_date", "target_end_date"]
    ]

    eligible = daily.loc[_as_bool(daily["eligible"])].copy()
    if eligible.empty:
        return pd.DataFrame()
    raw_path = raw_dir / f"{code}.csv"
    if not raw_path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(raw_path, usecols=["日期", "换手率"])
    raw["date"] = pd.to_datetime(raw["日期"])
    raw["turnover_mean"] = pd.to_numeric(raw["换手率"], errors="coerce") / 100.0
    eligible = eligible.merge(raw[["date", "turnover_mean"]], on="date", how="left")
    eligible["intraday_return"] = eligible["close"] / eligible["open"] - 1
    eligible["daily_total_return"] = eligible["close"] / eligible["prev_close"] - 1

    grouped = eligible.groupby("month", sort=True)
    features = grouped.agg(
        cto_mean=("cto_daily", "mean"),
        cto_std=("cto_daily", "std"),
        cto_min=("cto_daily", "min"),
        cto_max=("cto_daily", "max"),
        intraday_mean=("intraday_return", "mean"),
        turnover_mean=("turnover_mean", "mean"),
        feature_start_date=("date", "min"),
        feature_end_date=("date", "max"),
        valid_days=("cto_daily", "count"),
    ).reset_index()
    cumulative = grouped["daily_total_return"].agg(lambda s: np.prod(1.0 + s.dropna()) - 1.0)
    features = features.merge(
        cumulative.rename("month_cumulative_return").reset_index(), on="month", how="left"
    )
    features["code"] = code
    return features.merge(targets, on="month", how="left")


def build_feature_panel(
    daily_path: Path = BASE / "daily_cto.csv",
    raw_dir: Path = BASE / "daily_raw",
    monthly_cto_path: Path = BASE / "monthly_cto.csv",
    caps_path: Path = BASE / "market_caps_monthly.csv",
) -> pd.DataFrame:
    """Build the fixed eight-feature panel from the frozen main sample."""
    parts: list[pd.DataFrame] = []
    processed = 0
    for block in _complete_code_blocks(daily_path):
        for _, stock in block.groupby("code", sort=False):
            part = _one_code_features(stock, raw_dir)
            if not part.empty:
                parts.append(part)
            processed += 1
        if processed % 500 < block["code"].nunique():
            print(f"feature build: {processed:,} stocks", flush=True)
    features = pd.concat(parts, ignore_index=True)

    # Anchor to the exact stock-month pool and CTO aggregate already produced
    # by the main pipeline; this is the central no-second-universe safeguard.
    anchor = pd.read_csv(monthly_cto_path, dtype={"code": str})
    anchor["code"] = anchor["code"].str.zfill(6)
    anchor["month"] = pd.PeriodIndex(anchor["month"].astype(str), freq="M")
    features["month"] = pd.PeriodIndex(features["month"].astype(str), freq="M")
    panel = anchor[["code", "month", "cto_month"]].merge(features, on=["code", "month"], how="left")
    diff = (panel["cto_month"] - panel["cto_mean"]).abs()
    if diff.dropna().empty or diff.dropna().max() > 1e-12:
        raise AssertionError(f"Rebuilt CTO does not match main pipeline; max difference={diff.max()}")

    caps = pd.read_csv(caps_path, dtype={"code": str})
    caps["code"] = caps["code"].str.zfill(6)
    caps["month"] = pd.PeriodIndex(caps["month"].astype(str), freq="M")
    panel = panel.merge(caps[["code", "month", "market_cap"]], on=["code", "month"], how="left")
    panel["log_float_market_cap"] = np.log(pd.to_numeric(panel["market_cap"], errors="coerce"))
    panel["holding_month"] = panel["month"] + 1

    # Demean against the full main-analysis pool before complete-case model
    # filtering, so the target never changes with feature availability.
    market_mean = panel.groupby("month")["raw_next_month_return"].transform("mean")
    panel["target_excess_return"] = panel["raw_next_month_return"] - market_mean
    panel["feature_complete"] = panel[FEATURES].notna().all(axis=1)
    panel["target_complete"] = panel["target_excess_return"].notna()
    return panel


def winsorize_and_standardize(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply monthly 1/99 winsorization and cross-sectional z-scores."""
    out = panel.copy()
    for feature in FEATURES:
        def transform(group: pd.Series) -> pd.Series:
            lo, hi = group.quantile([0.01, 0.99])
            clipped = group.clip(lo, hi)
            std = clipped.std(ddof=0)
            return (clipped - clipped.mean()) / std if std > 0 else pd.Series(np.nan, index=group.index)

        out[f"z_{feature}"] = out.groupby("month", group_keys=False)[feature].transform(transform)
    return out


def assert_prediction_timing(train: pd.DataFrame, test: pd.DataFrame, formation_month: pd.Period) -> None:
    """Fail fast if a model can see a target unavailable at prediction time."""
    if train.empty or test.empty:
        raise AssertionError("Training and prediction samples must both be non-empty")
    train_holding = pd.PeriodIndex(train["holding_month"].astype(str), freq="M")
    test_formation = pd.PeriodIndex(test["month"].astype(str), freq="M")
    test_holding = pd.PeriodIndex(test["holding_month"].astype(str), freq="M")
    assert train_holding.max() <= formation_month, "Training uses a not-yet-realized target"
    assert test_formation.min() == test_formation.max() == formation_month
    assert (test_holding == formation_month + 1).all(), "Prediction target is not next month"
    assert (pd.to_datetime(test["target_end_date"]) > pd.to_datetime(test["feature_end_date"])).all()


def _load_lightgbm_regressor():
    """Import LightGBM lazily so feature construction needs no native runtime."""
    try:
        from lightgbm import LGBMRegressor
    except OSError as exc:  # pragma: no cover - platform-specific dependency message
        raise RuntimeError(
            "LightGBM requires OpenMP. On macOS install libomp or launch with "
            "DYLD_LIBRARY_PATH pointing to a compatible libomp.dylib."
        ) from exc
    return LGBMRegressor


def expanding_predictions(panel: pd.DataFrame, warmup_months: int = 36) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate monthly CTO, two Ridge, and fixed-LightGBM predictions."""
    zfeatures = [f"z_{f}" for f in FEATURES]
    sample = panel.loc[panel["feature_complete"] & panel["target_complete"]].copy()
    sample = sample.dropna(subset=zfeatures).sort_values(["month", "code"])
    months = sorted(sample["month"].unique())
    if len(months) <= warmup_months:
        raise ValueError("The sample is shorter than the 36-month warm-up")

    LGBMRegressor = _load_lightgbm_regressor()
    prediction_parts: list[pd.DataFrame] = []
    importance_parts: list[pd.DataFrame] = []
    pdp_parts: list[pd.DataFrame] = []
    cto_grid = np.linspace(sample["z_cto_mean"].quantile(0.01), sample["z_cto_mean"].quantile(0.99), 41)

    for number, formation_month in enumerate(months[warmup_months:], start=1):
        # At the end of formation month t, only targets with holding month <= t
        # have been observed. The test target is holding month t+1.
        train = sample.loc[sample["holding_month"] <= formation_month]
        test = sample.loc[sample["month"] == formation_month].copy()
        assert_prediction_timing(train, test, formation_month)
        X_train, y_train = train[zfeatures], train["target_excess_return"]
        X_test = test[zfeatures]

        ridge_cto_family = Ridge(alpha=1.0)
        ridge_cto_family.fit(X_train[[f"z_{f}" for f in CTO_FAMILY_FEATURES]], y_train)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        # ``verbosity`` controls console output only; all model hyperparameters
        # remain exactly the prespecified values in LIGHTGBM_PARAMS.
        lightgbm = LGBMRegressor(**LIGHTGBM_PARAMS, verbosity=-1)
        lightgbm.fit(X_train, y_train)

        test["pred_cto_sort"] = test["z_cto_mean"]
        test["pred_ridge_cto_family"] = ridge_cto_family.predict(
            X_test[[f"z_{f}" for f in CTO_FAMILY_FEATURES]]
        )
        test["pred_ridge"] = ridge.predict(X_test)
        test["pred_lightgbm"] = lightgbm.predict(X_test)
        prediction_parts.append(test)

        gain = lightgbm.booster_.feature_importance(importance_type="gain")
        importance_parts.append(pd.DataFrame({"formation_month": str(formation_month), "feature": FEATURES, "gain": gain}))

        # Rolling partial dependence: replace CTO for the current cross-section,
        # average predictions, then aggregate these monthly curves ex post.
        for value in cto_grid:
            X_pdp = X_test.copy()
            X_pdp["z_cto_mean"] = value
            pdp_parts.append(pd.DataFrame({
                "formation_month": [str(formation_month)],
                "cto_zscore": [value],
                "mean_prediction": [float(lightgbm.predict(X_pdp).mean())],
            }))
        if number == 1 or number % 12 == 0 or formation_month == months[-1]:
            print(
                f"rolling fit {number}/{len(months) - warmup_months}: formation={formation_month}, "
                f"train={len(train):,}, test={len(test):,}",
                flush=True,
            )

    return (
        pd.concat(prediction_parts, ignore_index=True),
        pd.concat(importance_parts, ignore_index=True),
        pd.concat(pdp_parts, ignore_index=True),
    )


def assign_prediction_deciles(predictions: pd.DataFrame) -> pd.DataFrame:
    """Assign monthly deciles separately for each of the four scores."""
    parts = []
    for model in MODELS:
        x = predictions[["code", "month", "holding_month", "raw_next_month_return", "target_excess_return", f"pred_{model}"]].copy()
        x = x.rename(columns={f"pred_{model}": "prediction"})
        x["rank"] = x.groupby("month")["prediction"].rank(method="first")
        x["n_month"] = x.groupby("month")["code"].transform("size")
        x["decile"] = np.ceil(10 * x["rank"] / x["n_month"]).astype(int)
        x["model"] = model
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


def monthly_metrics(assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute monthly Spearman ICs and equal-weighted decile/spread returns."""
    ic_rows = []
    portfolio_rows = []
    for (model, month), group in assignments.groupby(["model", "month"], sort=True):
        ic = spearmanr(group["prediction"], group["target_excess_return"], nan_policy="omit").statistic
        ic_rows.append({"model": model, "formation_month": month, "holding_month": month + 1, "ic": ic, "n": len(group)})
        for decile, cell in group.groupby("decile"):
            portfolio_rows.append({
                "model": model,
                "formation_month": month,
                "holding_month": month + 1,
                "decile": decile,
                "n": len(cell),
                "mean_return": cell["raw_next_month_return"].mean(),
            })
    return pd.DataFrame(ic_rows), pd.DataFrame(portfolio_rows)


def portfolio_turnover(assignments: pd.DataFrame) -> pd.DataFrame:
    """Calculate one-way membership turnover for each model's two extreme legs."""
    rows = []
    extremes = assignments.loc[assignments["decile"].isin([1, 10])]
    for (model, decile), group in extremes.groupby(["model", "decile"]):
        members = {month: set(cell["code"]) for month, cell in group.groupby("month")}
        months = sorted(members)
        for previous, current in zip(months, months[1:]):
            if current != previous + 1:
                continue
            overlap = len(members[previous] & members[current])
            rows.append({
                "model": model,
                "formation_month": current,
                "decile": decile,
                "stocks": len(members[current]),
                "overlap": overlap,
                "one_way_turnover": 1 - overlap / len(members[current]),
            })
    return pd.DataFrame(rows)


def build_spreads(portfolios: pd.DataFrame, turnover: pd.DataFrame) -> pd.DataFrame:
    """Build gross and main-framework cost-adjusted long-short returns."""
    pivot = portfolios.pivot(index=["model", "formation_month", "holding_month"], columns="decile", values="mean_return").reset_index()
    pivot["gross_long_short"] = pivot[10] - pivot[1]
    turn = turnover.pivot(index=["model", "formation_month"], columns="decile", values="one_way_turnover").reset_index()
    turn = turn.rename(columns={1: "low_one_way_turnover", 10: "high_one_way_turnover"})
    out = pivot.merge(turn, on=["model", "formation_month"], how="left")
    out["mean_leg_one_way_turnover"] = (out["low_one_way_turnover"] + out["high_one_way_turnover"]) / 2
    out["two_leg_one_way_turnover"] = out["low_one_way_turnover"] + out["high_one_way_turnover"]
    out["two_leg_buy_sell_turnover"] = 2 * out["two_leg_one_way_turnover"]
    for bps in (10, 15, 25):
        out[f"cost_{bps}bps"] = out["two_leg_buy_sell_turnover"] * bps / 10_000
        out[f"net_{bps}bps"] = out["gross_long_short"] - out[f"cost_{bps}bps"]
    return out.sort_values(["model", "holding_month"])


def _period_mask(frame: pd.DataFrame, start: str, end: str) -> pd.Series:
    holding = pd.PeriodIndex(frame["holding_month"].astype(str), freq="M")
    return (holding >= pd.Period(start, "M")) & (holding <= pd.Period(end, "M"))


def summarize_results(ics: pd.DataFrame, spreads: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize IC, gross/net returns, and turnover in the fixed periods."""
    ic_rows, return_rows, turnover_rows = [], [], []
    for period, (start, end) in PERIODS.items():
        for model in MODELS:
            i = ics.loc[(ics["model"] == model) & _period_mask(ics, start, end), "ic"].dropna()
            ic_std = i.std(ddof=1)
            ic_rows.append({
                "period": period,
                "model": model,
                "months": len(i),
                "mean_ic": i.mean(),
                "std_ic": ic_std,
                "icir": i.mean() / ic_std if ic_std > 0 else np.nan,
            })
            s = spreads.loc[(spreads["model"] == model) & _period_mask(spreads, start, end)]
            for return_type, column, cost_bps in [
                ("gross", "gross_long_short", 0),
                ("net", "net_10bps", 10),
                ("net", "net_15bps", 15),
                ("net", "net_25bps", 25),
            ]:
                t_value, mean = nw_mean_t(s[column], lags=5)
                return_rows.append({
                    "period": period,
                    "model": model,
                    "return_type": return_type,
                    "one_side_cost_bps": cost_bps,
                    "months": s[column].notna().sum(),
                    "mean_monthly_return": mean,
                    "newey_west_t_lag5": t_value,
                })
            turnover_rows.append({
                "period": period,
                "model": model,
                "months": s["mean_leg_one_way_turnover"].notna().sum(),
                "mean_low_leg_one_way_turnover": s["low_one_way_turnover"].mean(),
                "mean_high_leg_one_way_turnover": s["high_one_way_turnover"].mean(),
                "mean_portfolio_one_way_turnover": s["mean_leg_one_way_turnover"].mean(),
            })
    return pd.DataFrame(ic_rows), pd.DataFrame(return_rows), pd.DataFrame(turnover_rows)


def timeline_example(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    preferred_code: str = "000001",
) -> dict[str, object]:
    """Return a concrete stock-month chronology for the no-overlap audit."""
    first_month = predictions["month"].min()
    candidates = predictions.loc[(predictions["month"] == first_month) & (predictions["code"] == preferred_code)]
    prediction_row = candidates.iloc[0] if not candidates.empty else predictions.loc[predictions["month"] == first_month].iloc[0]
    code = prediction_row["code"]
    test_row = panel.loc[(panel["code"] == code) & (panel["month"] == first_month)].iloc[0]
    available = panel.loc[
        panel["feature_complete"]
        & panel["target_complete"]
        & (panel["holding_month"] <= first_month)
    ]
    latest_realized_month = available["holding_month"].max()
    training_row = available.loc[
        (available["code"] == code) & (available["holding_month"] == latest_realized_month)
    ].iloc[0]
    return {
        "code": code,
        "latest_training_feature_month": str(training_row["month"]),
        "latest_training_feature_start_date": str(pd.Timestamp(training_row["feature_start_date"]).date()),
        "latest_training_feature_end_date": str(pd.Timestamp(training_row["feature_end_date"]).date()),
        "latest_training_target_month": str(training_row["holding_month"]),
        "latest_training_target_end_date": str(pd.Timestamp(training_row["target_end_date"]).date()),
        "prediction_formation_month": str(test_row["month"]),
        "prediction_feature_start_date": str(pd.Timestamp(test_row["feature_start_date"]).date()),
        "prediction_feature_end_date": str(pd.Timestamp(test_row["feature_end_date"]).date()),
        "test_target_holding_month": str(test_row["holding_month"]),
        "test_target_return_start_date": str(pd.Timestamp(test_row["target_start_date"]).date()),
        "test_target_return_end_date": str(pd.Timestamp(test_row["target_end_date"]).date()),
        "assertion": "training holding_month <= formation_month < test holding_month",
    }


def make_figures(spreads: pd.DataFrame, importance: pd.DataFrame, pdp: pd.DataFrame, output: Path) -> None:
    """Write the required NAV, gain-importance, and CTO-PDP figures."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for model in MODELS:
        group = spreads.loc[spreads["model"] == model].sort_values("holding_month")
        dates = pd.PeriodIndex(group["holding_month"].astype(str), freq="M").to_timestamp()
        nav = (1 + group["gross_long_short"].fillna(0)).cumprod()
        ax.plot(dates, nav, lw=1.2, label=MODEL_LABELS[model])
    ax.axvline(pd.Timestamp("2021-01-01"), color="black", ls="--", lw=0.9)
    ax.set_yscale("log")
    ax.set_title("CTO extension: gross long-short cumulative NAV")
    ax.set_ylabel("Cumulative NAV (log scale)")
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(output / "long_short_nav_comparison_log.png", dpi=180)
    plt.close(fig)

    gain = importance.groupby("feature", as_index=False)["gain"].sum()
    gain["gain_share"] = gain["gain"] / gain["gain"].sum()
    gain = gain.sort_values("gain_share")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(gain["feature"], gain["gain_share"], color="tab:blue")
    ax.set_xlabel("Share of aggregate rolling gain")
    ax.set_title("LightGBM feature importance (gain)")
    ax.grid(axis="x", alpha=0.25); fig.tight_layout()
    fig.savefig(output / "lightgbm_feature_importance_gain.png", dpi=180)
    plt.close(fig)

    curve = pdp.groupby("cto_zscore", as_index=False)["mean_prediction"].mean()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(curve["cto_zscore"], curve["mean_prediction"] * 100, lw=1.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Monthly CTO (winsorized cross-sectional z-score)")
    ax.set_ylabel("Predicted next-month excess return (%)")
    ax.set_title("Rolling LightGBM partial dependence on monthly CTO")
    ax.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(output / "lightgbm_cto_partial_dependence.png", dpi=180)
    plt.close(fig)


def run(cache: Path = DEFAULT_CACHE, output: Path = DEFAULT_OUTPUT, rebuild_features: bool = False) -> None:
    """Execute the full prespecified extension and save public artifacts."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if rebuild_features or not cache.exists():
        panel = build_feature_panel()
        panel.to_csv(cache, index=False, compression="gzip")
    else:
        panel = pd.read_csv(cache, dtype={"code": str})
        for column in ("month", "holding_month"):
            panel[column] = pd.PeriodIndex(panel[column].astype(str), freq="M")
        for column in ("feature_start_date", "feature_end_date", "target_start_date", "target_end_date"):
            panel[column] = pd.to_datetime(panel[column])
        panel["feature_complete"] = _as_bool(panel["feature_complete"])
        panel["target_complete"] = _as_bool(panel["target_complete"])

    standardized = winsorize_and_standardize(panel)
    predictions, importance, pdp = expanding_predictions(standardized, warmup_months=36)
    assignments = assign_prediction_deciles(predictions)
    ics, portfolios = monthly_metrics(assignments)
    turnover = portfolio_turnover(assignments)
    spreads = build_spreads(portfolios, turnover)
    ic_summary, return_summary, turnover_summary = summarize_results(ics, spreads)

    prediction_columns = [
        "code", "month", "holding_month", "raw_next_month_return", "target_excess_return",
        "pred_cto_sort", "pred_ridge_cto_family", "pred_ridge", "pred_lightgbm",
    ]
    predictions[prediction_columns].to_csv(
        output / "rolling_predictions.csv.gz", index=False, compression="gzip"
    )
    ics.to_csv(output / "monthly_ic.csv", index=False)
    ic_summary.to_csv(output / "ic_summary.csv", index=False)
    portfolios.to_csv(output / "decile_monthly_returns.csv", index=False)
    spreads.to_csv(output / "long_short_monthly.csv", index=False)
    return_summary.to_csv(output / "long_short_summary.csv", index=False)
    turnover.to_csv(output / "portfolio_turnover_monthly.csv", index=False)
    turnover_summary.to_csv(output / "portfolio_turnover_summary.csv", index=False)

    gain = importance.groupby("feature", as_index=False)["gain"].sum()
    gain["gain_share"] = gain["gain"] / gain["gain"].sum()
    gain.sort_values("gain_share", ascending=False).to_csv(output / "lightgbm_feature_importance_gain.csv", index=False)
    pdp.groupby("cto_zscore", as_index=False)["mean_prediction"].mean().to_csv(
        output / "lightgbm_cto_partial_dependence.csv", index=False
    )
    timeline = timeline_example(predictions, standardized)
    (output / "timeline_no_overlap_example.json").write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    coverage = {
        "main_stock_months": int(len(panel)),
        "complete_feature_stock_months": int(panel["feature_complete"].sum()),
        "complete_feature_and_target_stock_months": int((panel["feature_complete"] & panel["target_complete"]).sum()),
        "prediction_stock_months_after_36_month_warmup": int(len(predictions)),
        "first_prediction_formation_month": str(predictions["month"].min()),
        "last_prediction_formation_month": str(predictions["month"].max()),
        "models": {model: MODEL_LABELS[model] for model in MODELS},
        "lightgbm_hyperparameters": LIGHTGBM_PARAMS,
        "hyperparameter_policy": "Prespecified to avoid in-sample tuning bias; no grid search was performed.",
    }
    (output / "sample_and_model_audit.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    make_figures(spreads, importance, pdp, output)
    print("TIMELINE AUDIT")
    print(json.dumps(timeline, indent=2))
    print("\nIC SUMMARY")
    print(ic_summary.to_string(index=False))
    print("\nLONG-SHORT SUMMARY")
    print(return_summary.to_string(index=False))
    print("\nTURNOVER SUMMARY")
    print(turnover_summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()
    run(args.cache, args.output, args.rebuild_features)


if __name__ == "__main__":
    main()
