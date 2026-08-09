"""Tests for Stage-4 attribution sampling and cost accounting."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from run_stage4_analysis import (cost_summary, intraday_portfolios, leg_turnover,
                                 market_context)
from run_stage4_attribution import extreme_indices


class Stage4Tests(unittest.TestCase):
    def test_extreme_sampling_is_balanced_and_reproducible(self) -> None:
        predictions = np.arange(1000, dtype=float)
        first = extreme_indices(predictions, 200, 7)
        second = extreme_indices(predictions, 200, 7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual((predictions[first] <= 99.9).sum(), 100)
        self.assertEqual((predictions[first] >= 899.1).sum(), 100)

    def test_leg_turnover_uses_equal_weight_l1_distance(self) -> None:
        turnover, overlap = leg_turnover({"a", "b"}, {"b", "c"})
        self.assertAlmostEqual(turnover, 0.5)
        self.assertAlmostEqual(overlap, 0.5)

    def test_intraday_gap_resets_portfolio_turnover(self) -> None:
        rows = []
        for time in [pd.Timestamp("2024-01-02 11:05"), pd.Timestamp("2024-01-02 13:35")]:
            for index in range(40):
                rows.append({"window": 1, "model": "Ridge", "code": f"c{index:02d}",
                    "time": time, "prediction": index,
                    "fwd_ret_30m_entry_lag1": index / 10000})
        result = intraday_portfolios(pd.DataFrame(rows))
        self.assertTrue(result["two_leg_buy_sell_turnover"].eq(4.0).all())

    def test_cost_summary_applies_one_way_bps_to_actual_trades(self) -> None:
        periods = pd.DataFrame({"window": [1], "model": ["GRU"],
            "gross_return": [0.001], "two_leg_buy_sell_turnover": [4.0],
            "low_overlap": [0.0], "high_overlap": [0.0]})
        row = cost_summary(periods, "intraday_30m").iloc[0]
        self.assertAlmostEqual(row["net_10bps"], -0.003)
        self.assertAlmostEqual(row["breakeven_one_way_bps"], 2.5)

    def test_market_context_marks_predeclared_failed_windows(self) -> None:
        index = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=40),
                              "close": np.linspace(100, 104, 40)})
        comparison = pd.DataFrame([
            {"window": 2, "test_start": "2023-01-01", "test_end": "2023-02-09",
             "model": "GRU", "rank_ic_mean": 0.03},
            {"window": 2, "test_start": "2023-01-01", "test_end": "2023-02-09",
             "model": "LightGBM", "rank_ic_mean": 0.028},
        ])
        row = market_context(index, comparison).iloc[0]
        self.assertFalse(row["gru_gate_pass"])
        self.assertAlmostEqual(row["gru_minus_lightgbm"], 0.002)


if __name__ == "__main__":
    unittest.main()
