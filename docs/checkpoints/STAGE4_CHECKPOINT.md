# Stage 4 final checkpoint: attribution, costs, and engineering closeout

## Run provenance

- Clean reproducibility notebook: `kaggle_stage4_attribution_cost_clean.ipynb`.
- Kaggle notebook: `nudge147/a-share-5min-stage4-clean-reproducibility` (Version 2).
- Kaggle recovery source: checkpoint Dataset Version 5.
- Restored artifacts: 80 neural checkpoints and 80 neural prediction files.
- GRU attribution: one representative seed, 400 tail samples per window, 32 IG steps.
- All eight windows completed; each window was archived independently before the final upload.
- Final Kaggle archive: `stage4_cost_complete.zip` (about 60 MB).
- Attribution archive: `stage4_attribution_complete.zip` (about 136 KB).
- Final clean-clone smoke gate: 41/41 tests passed.

## Seed independence audit

Every window contains five GRU seeds, five distinct initial-weight hashes, and five
distinct first-epoch losses. All three audit checks passed in all eight windows.

## GRU integrated-gradients summary

Mean attribution share across windows:

| Feature | Share |
|---|---:|
| pos_in_bar | 29.57% |
| range_rel | 16.21% |
| ret_5m | 16.08% |
| vol_rel_20d | 10.53% |
| vol_log | 6.16% |
| is_cross_day | 5.88% |
| time_of_day_cos | 5.44% |
| time_of_day_sin | 5.31% |
| is_first_bar | 3.60% |
| ret_overnight | 1.22% |

Within the `ret_5m` attribution alone, the latest 1/3/6/12 bars account for
15.42% / 36.75% / 58.86% / 77.88%. The GRU therefore relies strongly on recent
returns, but not only on the immediately preceding bar.

## Transaction-cost result

Full-period means across eight windows:

| Frequency | Model | Gross return | Breakeven one-way cost | Net at 5 bp |
|---|---|---:|---:|---:|
| 30-minute | GRU | 1.029 bp | 0.314 bp | -15.236 bp |
| 30-minute | LightGBM | 0.870 bp | 0.255 bp | -16.005 bp |
| 30-minute | MLP | 1.100 bp | 0.327 bp | -15.544 bp |
| 30-minute | Ridge | -0.896 bp | -0.275 bp | -17.895 bp |
| Daily close | GRU | -1.522 bp | -0.988 bp | -14.552 bp |
| Daily close | LightGBM | -13.964 bp | -5.532 bp | -27.338 bp |
| Daily close | MLP | -13.241 bp | -5.702 bp | -26.159 bp |
| Daily close | Ridge | -18.119 bp | -7.115 bp | -31.646 bp |

The 30-minute signals are positive before costs for GRU, MLP, and LightGBM, but
their breakeven costs are only about 0.25--0.33 bp per side. None survives even
the optimistic 5 bp assumption. Averaging the six intraday signals into a daily
close-to-close portfolio does not rescue performance; all four daily variants
have negative gross returns.

## Market-state comparison

The two predeclared failed windows are:

- Window 2: 2022-10-05 to 2023-04-04; GRU minus LightGBM RankIC = 0.00304.
- Window 8: 2025-10-05 to 2026-04-04; failed the predeclared GRU increment gate.

The final context table merges both GPU comparison shards and contains all eight
windows. It is descriptive only; eight observations are insufficient for a
regression claim.

Window 2 and Window 8 have opposite index-return signs, neither overlaps the
minute-data repair regime, and neither shares an extreme volatility state. The
evidence supports time variation in GRU value-add, but not a stable market-state
explanation. The exact comparison is archived in
`docs/STAGE4_WINDOW_2_8_ANALYSIS.md` and `docs/tables/window_2_8_market_context.csv`.

## Important data note

The Kaggle Stage-3 input did not contain the unadjusted companion daily table.
For the daily-frequency cost experiment only, daily closes were reconstructed as
the last available processed 5-minute close per stock-day and `tradestatus=1`
was assigned to observed days. This is adequate for the close-to-next-close
portfolio calculation but must not be described as a fresh Baostock daily-source
cross-check. The intraday cost results do not depend on this fallback.

## Local primary-copy provenance

- Kaggle Dataset Version 16: 178 files, ZIP CRC and SHA-256 verified locally.
- Kaggle Dataset Version 17: 163 files, ZIP CRC and SHA-256 verified locally.
- All 41 V16 CSV files and all 27 V17 CSV files parsed successfully.
- `stage4_cost_complete.zip`: reconstructed from the 143-file V16 directory and CRC-verified.
- Full hashes and byte counts: `data/archive/STAGE4_ARCHIVE_MANIFEST.md`.
- Persistence state: `COMPLETE_LOCAL_PRIMARY_VERIFIED` (DATA_NOTES DN-013).

## Final engineering outputs

- Deterministic builder: `finalize_stage4.py`.
- Window 2/8 analysis: `docs/STAGE4_WINDOW_2_8_ANALYSIS.md`.
- Two-protocol appendix: `docs/APPENDIX_A_EXECUTION_PROTOCOLS.md`.
- Publication figures: `docs/figures/` with exact source tables in `docs/tables/`.
- Repository cold-start and full-test status are recorded in
  `docs/checkpoints/COLD_START_REPRODUCIBILITY.md`.
