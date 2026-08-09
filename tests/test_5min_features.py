"""Offline leakage and shape tests for five-minute research datasets."""
from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from build_5min_dataset import build_stock_dataset
from features_5min import compute_stock_features, rank_one_timestamp
from run_baseline import delayed_entry_returns, evaluate_predictions, rolling_windows, split_frame
from run_neural_models import (GRURegressor, MLP, RankICScorer, feature_statistics,
                               write_gru_gate)
from tests.test_5min_pipeline import minute_day


def processed_days(count: int = 30) -> pd.DataFrame:
    """Build complete processed-form trading days with evolving prices."""
    days = pd.bdate_range("2024-01-02", periods=count)
    frames: list[pd.DataFrame] = []
    for number, day in enumerate(days):
        frame = minute_day(day.strftime("%Y-%m-%d"))
        prices = 10 + number * 0.01 + np.arange(len(frame)) * 0.001
        frame["open"] = prices
        frame["close"] = prices + 0.0005
        frame["high"] = prices + 0.002
        frame["low"] = prices - 0.002
        frame["volume"] = 100 + number
        frame["volume_raw"] = frame["volume"]
        frame["volume_rescaled"] = False
        frame["volume_scale_factor"] = 1.0
        frame["close_anomaly_flag"] = False
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class FeatureTests(unittest.TestCase):
    def test_features_are_causal_and_tail_labels_do_not_cross_days(self) -> None:
        features = compute_stock_features(processed_days(), "sh.600000")
        core = ["ret_5m", "range_rel", "pos_in_bar", "vol_log", "ret_overnight",
                "is_first_bar", "time_of_day_sin", "time_of_day_cos"]
        self.assertFalse(features[core].isna().any().any())
        self.assertTrue(features.loc[features["bar_index"].ge(42), "fwd_ret_30m"].isna().all())
        self.assertTrue(features.loc[features["bar_index"].ge(36), "fwd_ret_60m"].isna().all())
        non_first = features["is_first_bar"].eq(0)
        self.assertTrue(features.loc[non_first, "ret_overnight"].eq(0).all())
        day_six = features[features["date"].eq(features["date"].drop_duplicates().iloc[5])]
        self.assertTrue(day_six["vol_rel_20d"].notna().all())

    def test_cross_section_rank_spans_minus_one_to_one(self) -> None:
        group = pd.DataFrame({"fwd_ret_30m": np.arange(30, dtype=float),
                              "close_anomaly_flag": False,
                              "incomplete_day_flag": False})
        ranks = rank_one_timestamp(group)
        self.assertAlmostEqual(float(ranks.min()), -1.0)
        self.assertAlmostEqual(float(ranks.max()), 1.0)
        self.assertAlmostEqual(float(ranks.mean()), 0.0)

    def test_bar_index_comes_from_clock_not_daily_row_count(self) -> None:
        source = processed_days(2)
        second_day = source["date"].drop_duplicates().iloc[1]
        keep = ~source["date"].eq(second_day) | source["time"].dt.strftime(
            "%H:%M:%S").ge("10:00:00")
        features = compute_stock_features(source[keep], "sh.600000")
        first_observed = features[features["date"].eq(second_day)].iloc[0]
        self.assertEqual(first_observed["time"].strftime("%H:%M:%S"), "10:00:00")
        self.assertEqual(int(first_observed["bar_index"]), 5)
        self.assertTrue(bool(first_observed["incomplete_day_flag"]))

    def test_suspension_placeholder_bars_are_removed(self) -> None:
        source = processed_days(3)
        middle_day = source["date"].drop_duplicates().iloc[1]
        columns = ["open", "high", "low", "close", "volume", "amount"]
        source.loc[source["date"].eq(middle_day), columns] = 0
        features = compute_stock_features(source, "sh.600000")
        self.assertFalse(features["date"].eq(middle_day).any())

    def test_sequences_end_at_sample_and_1435_has_no_30m_label(self) -> None:
        features = compute_stock_features(processed_days(10), "sh.600000")
        sequences, flat = build_stock_dataset(features)
        self.assertEqual(sequences.shape[1:], (48, 10))
        self.assertTrue(flat["window_end_time"].eq(flat["time"]).all())
        late = flat["time"].dt.strftime("%H:%M:%S").eq("14:35:00")
        self.assertGreater(int(late.sum()), 0)
        self.assertTrue(flat.loc[late, "fwd_ret_30m"].isna().all())


class BaselineTests(unittest.TestCase):
    def test_neural_shapes_and_rankic_scorer(self) -> None:
        import torch
        self.assertEqual(tuple(MLP(25)(torch.zeros(4, 25)).shape), (4,))
        self.assertEqual(tuple(GRURegressor(10)(torch.zeros(4, 48, 10)).shape), (4,))
        times = pd.to_datetime(["2024-01-02 10:05"] * 30 + ["2024-01-02 10:35"] * 30)
        target = np.tile(np.arange(30, dtype=float), 2)
        meta = pd.DataFrame({"time": times, "fwd_ret_30m_entry_lag1": target})
        self.assertAlmostEqual(RankICScorer(meta).score(target), 1.0)

    def test_sequence_statistics_use_train_values_only(self) -> None:
        train = np.arange(24, dtype=float).reshape(2, 3, 4)
        mean, scale = feature_statistics(train, sequence=True)
        np.testing.assert_allclose(mean, train.mean(axis=(0, 1)))
        np.testing.assert_allclose(scale, train.std(axis=(0, 1)))

    def test_gru_seed_std_must_not_exceed_its_lead(self) -> None:
        table = pd.DataFrame([
            {"window": 1, "model": "LightGBM", "rank_ic_mean": 0.03,
             "rank_ic_std": 0.0, "seeds": 1},
            {"window": 1, "model": "GRU", "rank_ic_mean": 0.04,
             "rank_ic_std": 0.011, "seeds": 5},
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            table.to_csv(path / "four_model_comparison.csv", index=False)
            write_gru_gate(path)
            gate = pd.read_csv(path / "gru_increment_gate.csv").iloc[0]
        self.assertFalse(bool(gate["seed_std_not_greater_than_increment"]))
        self.assertFalse(bool(gate["sequence_value_add_pass"]))

    def test_delayed_entry_label_enters_at_next_bar_close(self) -> None:
        times = pd.date_range("2024-01-02 10:05", periods=7, freq="5min")
        frame = pd.DataFrame({"code": "sh.600000", "date": times.normalize(),
                              "time": times, "close": np.arange(1, 8, dtype=float)})
        with tempfile.TemporaryDirectory() as directory:
            frame.to_parquet(Path(directory) / "sh.600000.parquet", index=False)
            result = delayed_entry_returns(Path(directory))
        first = result[result["time"].eq(times[0])].iloc[0]
        self.assertAlmostEqual(first["fwd_ret_30m_entry_lag1"], np.log(7 / 2))

    def test_rolling_windows_and_purge_are_disjoint(self) -> None:
        dates = pd.Series(pd.date_range("2020-01-01", "2024-12-31", freq="D"))
        window = rolling_windows(dates)[0]
        frame = pd.DataFrame({"date": dates})
        train, valid, test = split_frame(frame, window)
        self.assertGreater((valid["date"].min() - train["date"].max()).days, 1)
        self.assertGreater((test["date"].min() - valid["date"].max()).days, 1)

    def test_prediction_metrics_use_timestamp_cross_sections(self) -> None:
        frames = []
        for time in pd.to_datetime(["2024-01-02 10:05", "2024-01-02 10:35"]):
            values = np.arange(30, dtype=float)
            frames.append(pd.DataFrame({"date": time.normalize(), "time": time,
                "prediction": values, "fwd_ret_30m": values}))
        metrics, ic, deciles = evaluate_predictions(pd.concat(frames, ignore_index=True))
        self.assertAlmostEqual(metrics["rank_ic"], 1.0)
        self.assertEqual(len(ic), 2)
        self.assertEqual(set(deciles["decile"]), set(range(1, 11)))


if __name__ == "__main__":
    unittest.main()
