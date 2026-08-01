"""Unit tests for CTO price-limit, ST/trading-status, and market-cap filters."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from cto_pipeline import clean_one, limit_pct
from market_cap_pipeline import stepify_float_shares


class LimitRuleTests(unittest.TestCase):
    def test_round_to_cent_before_comparison(self):
        # 14.25 * 1.10 = 15.675, but the exchange price is 15.68.
        d = pd.DataFrame({"日期": ["2020-07-03", "2020-07-06"], "开盘": [14.0, 15.68], "收盘": [14.25, 15.68], "成交量": [1, 1]})
        out = clean_one(d, "000001", {"is_st": False, "list_date": "2007-01-01"}, d)
        self.assertEqual(round(14.25 * 1.10, 2), 15.68)
        self.assertTrue(bool(out.loc[out["date"].eq("2020-07-06"), "close_at_limit"].iloc[0]))

    def test_known_ping_an_2020_07_06(self):
        # Minimal public fixture for the known 2020-07-06 Ping An Bank limit day.
        raw = pd.DataFrame({
            "日期": ["2020-07-03", "2020-07-06", "2020-07-07"],
            "开盘": [13.57, 14.60, 16.30], "收盘": [14.25, 15.68, 15.48],
            "最高": [14.32, 15.68, 16.63], "最低": [13.56, 14.59, 15.03],
            "成交量": [3768334, 4711461, 3964428], "交易状态": ["1", "1", "1"], "是否ST": ["0", "0", "0"],
        })
        out = clean_one(raw, "000001", {"is_st": False, "list_date": "2007-01-01"}, raw)
        row = out[out["date"].eq(pd.Timestamp("2020-07-06"))]
        self.assertEqual(len(row), 1)
        self.assertTrue(bool(row["close_at_limit"].iloc[0]))
        next_row = out[out["date"].eq(pd.Timestamp("2020-07-07"))]
        self.assertTrue(bool(next_row["prev_close_at_limit"].iloc[0]))

    def test_rule_change(self):
        self.assertEqual(limit_pct("300001", "2020-08-23"), 0.10)
        self.assertEqual(limit_pct("300001", "2020-08-24"), 0.20)
        self.assertEqual(limit_pct("688001", "2020-08-24"), 0.20)

    def test_baostock_daily_st_and_trade_status(self):
        d = pd.DataFrame({"日期": ["2021-01-04", "2021-01-05"], "开盘": [10, 10], "收盘": [10, 10],
                          "成交量": [1, 1], "是否ST": ["0", "1"], "交易状态": ["1", "0"]})
        out = clean_one(d, "000001", {"list_date": "2000-01-01"}, d)
        row = out.iloc[1]
        self.assertTrue(bool(row["is_st"]))
        self.assertTrue(bool(row["suspended"]))
        self.assertFalse(bool(row["eligible"]))

    def test_share_stepification(self):
        raw = pd.Series([100.0, 101.0, 99.0, None, 102.0, 130.0, 130.0])
        stepped = stepify_float_shares(raw, window=3, threshold=.05)
        self.assertEqual(stepped.iloc[0], 100.0)
        self.assertEqual(stepped.iloc[4], 100.0)  # rounding noise / missing day is held
        self.assertGreater(stepped.iloc[6], 120.0)  # material jump is retained


if __name__ == "__main__":
    unittest.main()
