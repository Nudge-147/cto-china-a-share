# Kaggle stage-three batch

The runner checkpoints every completed `(window, seed, model)` row to
`neural_metrics.csv`. Re-running the same command skips completed combinations.

Enable a Kaggle GPU, attach the project code plus the `features_5min`, `dataset/flat`, and
`dataset/seq` directories, install the pinned requirements, then run:

```bash
python -m pip install -r requirements-5min.txt

python run_neural_models.py \
  --models MLP GRU \
  --windows 1 2 3 4 5 6 7 8 \
  --seeds 20260807 20260808 20260809 20260810 20260811 \
  --device cuda \
  --flat-dir /kaggle/input/a-share-5min/dataset/flat \
  --feature-dir /kaggle/input/a-share-5min/features_5min \
  --seq-dir /kaggle/input/a-share-5min/dataset/seq \
  --output-dir /kaggle/working/stage3_report
```

The MLP always uses the exact 25 flat columns imported from `run_baseline.MODEL_FEATURES`.
The GRU uses the stored `(48, 10)` arrays. Both fit imputation/scaling statistics separately
inside every training window. Early stopping uses delayed-entry validation RankIC.

Outputs include `neural_metrics.csv`, per-window `neural_seed_summary.csv`,
`full_period_seed_summary.csv`, `four_model_comparison.csv`, `rankic_by_window.png`,
`four_model_decile_curve.png`, and both per-window and full-period GRU increment gates. The fixed
robustness rule requires five complete seeds over eight windows, a GRU lead of at least 0.005, and
seed-level RankIC standard deviation no greater than the lead; otherwise the conclusion is
“no robust evidence.”
