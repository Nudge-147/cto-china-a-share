"""Tests for deterministic Stage-4 publication tables."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from finalize_stage4 import score_panel, window_context


class FinalizeStage4Tests(unittest.TestCase):
    def test_score_panel_balances_cross_sections_by_timestamp(self) -> None:
        rows = []
        for day, scale in (("2024-01-02", 1.0), ("2024-01-03", 10.0)):
            for index in range(40):
                rows.append({"date": day, "time": f"{day} 10:05", "prediction": index,
                             "target": scale * index / 10_000})
        metrics, deciles = score_panel(pd.DataFrame(rows), "target")
        self.assertAlmostEqual(metrics["rank_ic"], 1.0)
        self.assertEqual(metrics["ic_observations"], 2)
        self.assertEqual(deciles["decile"].tolist(), list(range(1, 11)))
        self.assertGreater(metrics["long_short"], 0)

    def test_window_context_preserves_predeclared_failed_rows(self) -> None:
        frame = pd.DataFrame({"window": range(1, 9), "gru_gate_pass": [1, 0, 1, 1, 1, 1, 1, 0],
            "gru_minus_lightgbm": np.arange(8) / 1000, "index_log_return": np.zeros(8),
            "mean_20d_volatility": np.ones(8) / 100, "test_start": "2024-01-01",
            "test_end": "2024-06-30", "regime_calendar_overlap_share": np.zeros(8)})
        with tempfile.TemporaryDirectory() as directory:
            root, tables = Path(directory) / "root", Path(directory) / "tables"
            (root / "stage4_final_report").mkdir(parents=True); tables.mkdir()
            frame.to_csv(root / "stage4_final_report" / "window_market_context.csv", index=False)
            result = window_context(root, tables)
            selected = pd.read_csv(tables / "window_2_8_market_context.csv")
        self.assertEqual(len(result), 8)
        self.assertEqual(selected["window"].tolist(), [2, 8])
        self.assertTrue(selected["passed_window_mean_gru_increment"].notna().all())


if __name__ == "__main__":
    unittest.main()
