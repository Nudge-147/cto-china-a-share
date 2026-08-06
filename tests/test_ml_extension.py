"""Unit tests for the frozen CTO nonlinear-extension mechanics."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_extension import (
    CTO_FAMILY_FEATURES,
    FEATURES,
    assert_prediction_timing,
    build_spreads,
    winsorize_and_standardize,
)


class MLExtensionTests(unittest.TestCase):
    def test_cto_family_control_is_frozen_to_features_one_through_five(self):
        self.assertEqual(CTO_FAMILY_FEATURES, FEATURES[:5])
        self.assertNotIn("month_cumulative_return", CTO_FAMILY_FEATURES)
        self.assertNotIn("turnover_mean", CTO_FAMILY_FEATURES)
        self.assertNotIn("log_float_market_cap", CTO_FAMILY_FEATURES)

    def test_monthly_winsorization_then_standardization(self):
        rows = 100
        panel = pd.DataFrame({"month": [pd.Period("2020-01", "M")] * rows})
        for number, feature in enumerate(FEATURES, start=1):
            panel[feature] = np.arange(rows, dtype=float) * number
        out = winsorize_and_standardize(panel)
        for feature in FEATURES:
            raw = panel[feature]
            lo, hi = raw.quantile([0.01, 0.99])
            clipped = raw.clip(lo, hi)
            expected = (clipped - clipped.mean()) / clipped.std(ddof=0)
            self.assertAlmostEqual(out[f"z_{feature}"].mean(), 0.0, places=12)
            self.assertAlmostEqual(out[f"z_{feature}"].std(ddof=0), 1.0, places=12)
            np.testing.assert_allclose(out[f"z_{feature}"], expected, rtol=0, atol=1e-12)

    def test_timing_assertion_accepts_only_realized_training_targets(self):
        train = pd.DataFrame({"holding_month": [pd.Period("2012-12", "M"), pd.Period("2013-01", "M")]})
        test = pd.DataFrame({
            "month": [pd.Period("2013-01", "M")],
            "holding_month": [pd.Period("2013-02", "M")],
            "feature_end_date": [pd.Timestamp("2013-01-31")],
            "target_end_date": [pd.Timestamp("2013-02-28")],
        })
        assert_prediction_timing(train, test, pd.Period("2013-01", "M"))
        leaked = train.copy()
        leaked.loc[len(leaked)] = [pd.Period("2013-02", "M")]
        with self.assertRaises(AssertionError):
            assert_prediction_timing(leaked, test, pd.Period("2013-01", "M"))

    def test_main_cost_formula_is_applied_to_both_legs_and_trades(self):
        portfolios = pd.DataFrame({
            "model": ["cto_sort", "cto_sort"],
            "formation_month": [pd.Period("2020-01", "M")] * 2,
            "holding_month": [pd.Period("2020-02", "M")] * 2,
            "decile": [1, 10],
            "mean_return": [-0.01, 0.02],
        })
        turnover = pd.DataFrame({
            "model": ["cto_sort", "cto_sort"],
            "formation_month": [pd.Period("2020-01", "M")] * 2,
            "decile": [1, 10],
            "one_way_turnover": [0.5, 0.75],
        })
        row = build_spreads(portfolios, turnover).iloc[0]
        self.assertAlmostEqual(row["gross_long_short"], 0.03)
        self.assertAlmostEqual(row["cost_10bps"], 2 * (0.5 + 0.75) * 0.001)
        self.assertAlmostEqual(row["net_10bps"], 0.0275)


if __name__ == "__main__":
    unittest.main()
