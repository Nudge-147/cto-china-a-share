# Stage three: MLP and GRU checkpoint

Run date: 2026-08-07; final five-seed status was superseded by the Stage-4 checkpoint.

## Preflight gates

- Five within-timestamp label shuffles passed: delayed-entry test RankIC was
  `-0.00244 ± 0.00449` for LightGBM and `-0.00083 ± 0.00980` for Ridge.
- All eight exact timestamp purge checks passed. The minimum prior-label-end to next-sample gap
  was 43.5 hours. Historical sequence context may precede a split boundary, but no training label
  endpoint reaches a validation sample and no validation label endpoint reaches a test sample.

## Window-one local CPU result

Both neural models used training-window-only imputation and standardization. Early stopping
monitored delayed-entry validation RankIC with patience 8.

| Model | Seed | Best epoch | Test delayed RankIC | ICIR | Decile L/S | Max train IC | Max validation IC | Overfit alert |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MLP | 20260807 | 2 | 0.02108 | 0.14597 | 0.0000625 | 0.04069 | 0.05958 | No |
| GRU | 20260807 | 14 | 0.03436 | 0.23373 | 0.0000705 | 0.07439 | 0.05763 | No |

The common delayed-entry comparison was:

| Model | Test RankIC | ICIR | Decile L/S |
|---|---:|---:|---:|
| Ridge | 0.01614 | 0.11212 | -0.000123 |
| LightGBM | 0.02635 | 0.19877 | 0.000235 |
| MLP | 0.02108 | 0.14597 | 0.0000625 |
| GRU | 0.03436 | 0.23373 | 0.0000705 |

In this single seed, MLP trailed LightGBM by `0.00527`. GRU exceeded it by `0.00802`, clearing
the `0.005` threshold provisionally. No sequence-value conclusion was made until the five-seed
Kaggle batch. The predeclared rule required all five seeds and all eight windows, a GRU lead of at
least 0.005, and seed-level full-period RankIC standard deviation no greater than that lead.

All four models had train-versus-validation diagnostics. LightGBM training RankIC reached 0.1167
while validation peaked at 0.0504; this showed pressure but did not trigger the predeclared
`train > 0.15 and validation < 0.05` alert. Ridge was represented by one fitted point.

## Batch handoff

`run_neural_models.py` provided interruption-safe checkpoints for eight windows, two neural
models, and five fixed seeds. The completed batch, seed audit, attribution, and cost conclusions
are preserved in [`STAGE4_CHECKPOINT.md`](STAGE4_CHECKPOINT.md).

