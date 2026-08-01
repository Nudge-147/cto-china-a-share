"""Run a power audit and explicitly exploratory pooled CTO×label regression.

Inputs: frozen Week-4 Q5−Q1 series, labels, CTO, market-cap, and daily returns.
Outputs: HAC power calculation and post-hoc regression CSVs.
Role: supplementary analysis; it does not replace preregistered portfolio sorts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from backtest_cto import month_end_returns


BASE = Path("data/cto_baostock")
WEEK4 = BASE / "formal_backtest" / "week4"
TWO_D = WEEK4 / "two_dimensional"
OUT = TWO_D / "posthoc"
NW_LAGS = 5


def hac_lrv(x: pd.Series, lags: int = NW_LAGS) -> float:
    x = pd.Series(x).dropna().astype(float)
    centered = x - x.mean()
    lrv = float(centered @ centered) / len(centered)
    for lag in range(1, min(lags, len(centered) - 1) + 1):
        cov = float(centered.iloc[lag:].to_numpy() @ centered.iloc[:-lag].to_numpy()) / len(centered)
        lrv += 2 * (1 - lag / (lags + 1)) * cov
    return lrv


def power_audit() -> pd.DataFrame:
    x = pd.read_csv(TWO_D / "monthly_q5_minus_q1.csv")
    pivot = x.pivot(index="holding_month", columns="stock_label", values="q5_minus_q1")
    difference = (pivot["attention"] - pivot["information"]).dropna()
    effect = float(difference.mean())
    n = len(difference)
    lrv = hac_lrv(difference)
    se = np.sqrt(lrv / n)
    noncentrality = effect / se
    critical = norm.ppf(.975)
    power = norm.cdf(-critical - noncentrality) + (1 - norm.cdf(critical - noncentrality))
    n_80 = int(np.ceil(((critical + norm.ppf(.80)) ** 2) * lrv / effect**2))
    result = pd.DataFrame([{
        "test": "attention_minus_information_Q5-Q1",
        "months": n,
        "observed_difference": effect,
        "hac_long_run_sd": np.sqrt(lrv),
        "hac_se": se,
        "observed_nw_t": noncentrality,
        "two_sided_alpha": .05,
        "approx_power_at_observed_effect": power,
        "months_for_80pct_power_at_observed_effect": n_80,
        "method": "Normal approximation using the observed Newey-West(5) long-run variance; exploratory power audit",
    }])
    result.to_csv(OUT / "power_audit.csv", index=False)
    return result


def exploratory_regression() -> pd.DataFrame:
    """Month-FE pooled OLS; explicitly post-hoc and not a replacement for sorts."""
    labels = pd.read_csv(WEEK4 / "monthly_stock_labels.csv", dtype={"code": str})
    cto = pd.read_csv(BASE / "monthly_cto.csv", dtype={"code": str})
    caps = pd.read_csv(BASE / "market_caps_monthly.csv", dtype={"code": str})
    daily = pd.read_csv(BASE / "daily_cto.csv", usecols=["code", "date", "close"], dtype={"code": str})
    for x in (labels, cto, caps, daily):
        x["code"] = x["code"].str.zfill(6)
    for x in (labels, cto, caps):
        x["month"] = pd.PeriodIndex(x["month"].astype(str), freq="M")
    daily["date"] = pd.to_datetime(daily["date"])
    next_returns = month_end_returns(daily)
    # The return ending in the formation month is a conventional momentum
    # control.  It is shifted one month forward, so it never contains t+1.
    prior_return = next_returns.copy()
    prior_return["month"] = prior_return["month"] + 1
    prior_return = prior_return.rename(columns={"next_month_return": "formation_month_return"})
    x = (cto.merge(labels[["code", "month", "stock_label"]], on=["code", "month"], how="left")
           .merge(caps[["code", "month", "market_cap"]], on=["code", "month"], how="left")
           .merge(next_returns, on=["code", "month"], how="left")
           .merge(prior_return[["code", "month", "formation_month_return"]], on=["code", "month"], how="left"))
    x["stock_label"] = x["stock_label"].fillna("baseline")
    x["attention"] = x.stock_label.eq("attention").astype(float)
    x["information"] = x.stock_label.eq("information").astype(float)
    x["cto_pct"] = x["cto_month"] * 100
    x["attention_x_cto"] = x.attention * x.cto_pct
    x["information_x_cto"] = x.information * x.cto_pct
    x["log_market_cap"] = np.log(x.market_cap.where(x.market_cap.gt(0)))
    cols = ["next_month_return", "cto_pct", "attention", "information", "attention_x_cto", "information_x_cto", "log_market_cap", "formation_month_return", "month"]
    x = x.dropna(subset=cols).copy()
    regressors = ["cto_pct", "attention", "information", "attention_x_cto", "information_x_cto", "log_market_cap", "formation_month_return"]
    # Within-month demeaning implements month fixed effects without forming
    # hundreds of dummy columns.  SEs are clustered by formation month.
    y = x.next_month_return - x.groupby("month").next_month_return.transform("mean")
    X = x[regressors] - x.groupby("month")[regressors].transform("mean")
    fit = sm.OLS(y, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": x["month"].astype(str)})
    result = pd.DataFrame({"term": fit.params.index, "coefficient": fit.params.values, "cluster_month_se": fit.bse.values,
                           "t_stat": fit.tvalues.values, "p_value": fit.pvalues.values})
    result["n_obs"] = int(fit.nobs); result["n_month_clusters"] = x.month.nunique()
    result["specification"] = "POST-HOC: month-FE pooled OLS, month-clustered SE; controls log float market cap and formation-month return"
    result.to_csv(OUT / "posthoc_pooled_interaction_regression.csv", index=False)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(power_audit().to_string(index=False))
    print(exploratory_regression().to_string(index=False))


if __name__ == "__main__":
    main()
