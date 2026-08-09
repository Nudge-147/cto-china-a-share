"""Offline tests for the five-minute downloader and QC rules."""
from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from config import EXPECTED_BAR_TIMES, EXPECTED_BARS_PER_DAY
from download_5min import atomic_parquet, parse_minutes, run_tasks
from investigate_reconciliation import add_mechanism_matches
from investigate_precision_materiality import ROUNDING_HARD_BOUND, rounding_summary
from repair_5min import proportional_integer_allocation
from quality_check import (
    assertion_frames, completeness_table, flagged_days, high_low_hypothesis,
    reconciliation_table, run_quality_check, strict_reconciliation_issues,
)


def minute_day(day: str, code: str = "sh.600000") -> pd.DataFrame:
    """Build one complete synthetic 48-bar trading day."""
    timestamps = [pd.Timestamp(f"{day} {clock}") for clock in EXPECTED_BAR_TIMES]
    return pd.DataFrame({
        "date": pd.Timestamp(day), "time": timestamps, "code": code,
        "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0,
        "volume": 100.0, "amount": 1_000.0,
    })


def daily_rows(days: list[str], statuses: list[int]) -> pd.DataFrame:
    """Build a direct-unadjusted daily reference."""
    return pd.DataFrame({
        "date": pd.to_datetime(days), "code": "sh.600000", "open": 10.0,
        "high": 10.2, "low": 9.8, "close": 10.0, "volume": 4_800.0,
        "amount": 48_000.0, "tradestatus": statuses,
    })


class DownloadParsingTests(unittest.TestCase):
    def test_expected_time_table(self) -> None:
        self.assertEqual(len(EXPECTED_BAR_TIMES), EXPECTED_BARS_PER_DAY)
        self.assertEqual(EXPECTED_BAR_TIMES[0], "09:35:00")
        self.assertEqual(EXPECTED_BAR_TIMES[-1], "15:00:00")

    def test_long_time_parsing_and_parquet(self) -> None:
        raw = pd.DataFrame([[
            "2020-01-02", "20200102093500000", "sh.600000", "10", "10.2",
            "9.8", "10.1", "100", "1000",
        ]], columns="date,time,code,open,high,low,close,volume,amount".split(","))
        parsed = parse_minutes(raw)
        self.assertEqual(parsed.loc[0, "time"], pd.Timestamp("2020-01-02 09:35:00"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.parquet"
            atomic_parquet(parsed, path)
            self.assertEqual(len(pd.read_parquet(path)), 1)

    def test_each_task_result_is_emitted_immediately(self) -> None:
        emitted: list[dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = run_tasks(executor, lambda value: {"status": "ok", "value": value},
                                [(1,), (2,)], emitted.append)
        self.assertCountEqual([item["value"] for item in results], [1, 2])
        self.assertEqual(len(emitted), 2)


class QualityRuleTests(unittest.TestCase):
    def test_assertion_layers_find_each_issue(self) -> None:
        frame = minute_day("2024-01-02").iloc[:3].copy()
        frame.loc[0, "low"] = 10.1
        frame.loc[1, "volume"] = -1
        frame.loc[2, "time"] = pd.Timestamp("2024-01-02 09:31:00")
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        issues = assertion_frames(frame)
        self.assertGreater(len(issues["assertion_ohlc"]), 0)
        self.assertGreater(len(issues["assertion_duplicates"]), 0)
        self.assertEqual(len(issues["assertion_nonnegative"]), 1)
        self.assertEqual(len(issues["assertion_invalid_time"]), 1)

    def test_completeness_classification(self) -> None:
        complete = minute_day("2024-01-02")
        incomplete = minute_day("2024-01-03").iloc[:-1]
        minutes = pd.concat([complete, incomplete], ignore_index=True)
        daily = daily_rows(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            [1, 1, 0, 1],
        )
        market = pd.DataFrame({
            "date": daily["date"], "market_mode_bars": 48,
            "observed_stocks": 1, "is_normal_market_day": True,
        })
        table = completeness_table("sh.600000", minutes, daily, market)
        classes = dict(zip(table["date"].dt.strftime("%Y-%m-%d"), table["classification"]))
        self.assertEqual(classes["2024-01-02"], "complete")
        self.assertEqual(classes["2024-01-03"], "intraday_missing")
        self.assertEqual(classes["2024-01-04"], "suspended")
        self.assertEqual(classes["2024-01-05"], "whole_day_missing")

    def test_zero_volume_bars_on_nontrading_day_are_suspension_placeholders(self) -> None:
        minutes = minute_day("2024-01-02")
        minutes[["open", "high", "low", "close", "volume", "amount"]] = 0
        daily = daily_rows(["2024-01-02"], [0])
        market = pd.DataFrame({"date": daily["date"], "market_mode_bars": 48,
                               "observed_stocks": 1, "is_normal_market_day": True})
        table = completeness_table("sh.600000", minutes, daily, market)
        self.assertEqual(table.loc[0, "classification"], "suspended_placeholder_bars")

    def test_strict_reconciliation_and_open_hypothesis(self) -> None:
        minutes = minute_day("2024-01-02")
        daily = daily_rows(["2024-01-02"], [1])
        table = reconciliation_table("sh.600000", minutes, daily)
        self.assertTrue(strict_reconciliation_issues(table).empty)
        table.loc[:, "daily_high"] = 11.0
        table.loc[:, "high_absolute_error"] = 0.8
        hypothesis = high_low_hypothesis("sh.600000", table)
        self.assertFalse(bool(hypothesis.loc[0, "extreme_near_open"]))

    def test_end_to_end_report_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, reference, output = root / "raw", root / "daily", root / "qc"
            raw.mkdir()
            reference.mkdir()
            minute_day("2024-01-02").to_parquet(raw / "sh.600000.parquet", index=False)
            daily_rows(["2024-01-02"], [1]).to_parquet(reference / "sh.600000.parquet", index=False)
            summary = run_quality_check(["sh.600000"], raw, reference, output)
            self.assertEqual(len(summary), 1)
            self.assertTrue((output / "qc_summary.csv").exists())
            self.assertEqual(len(list(output.glob("*.png"))), 5)


class ReconciliationInvestigationTests(unittest.TestCase):
    def mechanism_input(self, gap: float, amount_gap: float, close: float = 10.0) -> pd.DataFrame:
        """Build one synthetic reconciliation row with explicit signed gaps."""
        return pd.DataFrame({
            "date": [pd.Timestamp("2024-01-02")], "code": ["sh.600000"],
            "daily_minus_minute_volume": [gap], "daily_minus_minute_amount": [amount_gap],
            "daily_close": [close],
        })

    def test_opposite_direction_cannot_be_explained(self) -> None:
        block = pd.DataFrame(columns=["date", "code", "block_volume", "block_amount", "block_trades"])
        result = add_mechanism_matches(self.mechanism_input(-100, -1_000), block)
        self.assertFalse(bool(result.loc[0, "hypothesis_direction"]))
        self.assertFalse(bool(result.loc[0, "mechanism_explained"]))

    def test_matching_block_volume_requires_positive_gap(self) -> None:
        block = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-02")], "code": ["sh.600000"],
            "block_volume": [1_000.0], "block_amount": [9_000.0], "block_trades": [1],
        })
        result = add_mechanism_matches(self.mechanism_input(1_000, 9_000), block)
        self.assertTrue(bool(result.loc[0, "block_volume_match"]))

    def test_close_price_proxy_is_explicit_not_an_event_label(self) -> None:
        block = pd.DataFrame(columns=["date", "code", "block_volume", "block_amount", "block_trades"])
        result = add_mechanism_matches(self.mechanism_input(1_000, 10_000), block)
        self.assertTrue(bool(result.loc[0, "fixed_price_proxy_match"]))

    def test_round_lot_hard_bound_is_48_half_lots(self) -> None:
        self.assertEqual(ROUNDING_HARD_BOUND, 2_400)
        frame = pd.DataFrame({
            "abs_volume_gap": [2_400.0, 2_401.0], "within_rounding_bound": [True, False],
            "year": [2024, 2024],
        })
        summary = rounding_summary(frame)
        overall = summary[summary["period"].eq("overall")].iloc[0]
        self.assertEqual(overall["within_bound_days"], 1)

    def test_proportional_allocation_is_integer_and_exact(self) -> None:
        result = proportional_integer_allocation(pd.Series([100, 200, 300]), 1_001)
        self.assertEqual(result.sum(), 1_001)
        self.assertTrue((result >= 0).all())
        self.assertEqual(result.dtype.kind, "i")

    def test_processing_flags_count_unique_days(self) -> None:
        frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
                              "volume_rescaled": [True, True, False]})
        self.assertEqual(flagged_days(frame, "volume_rescaled"), 1)
        self.assertEqual(flagged_days(frame, "close_anomaly_flag"), 0)


if __name__ == "__main__":
    unittest.main()
