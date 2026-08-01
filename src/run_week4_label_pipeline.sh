#!/usr/bin/env bash
# Rebuild the frozen Week-4 labels in memory-safe stages.
# Inputs: downloaded announcements and Baostock price data under data/.
# Outputs: Week-4 label diagnostics under data/cto_baostock/formal_backtest/week4.
# Role: orchestration only; it fixes no research parameter and runs no returns.
set -euo pipefail

for period in "2010 2012" "2013 2015" "2016 2018" "2019 2021" "2022 2024" "2025 2026"; do
  read -r start_year end_year <<< "$period"
  for range in "0 2000" "2000 4000" "4000 5538"; do
    read -r start_index end_index <<< "$range"
    python src/run_week4_label_diagnostics.py --stage median-fill \
      --start-year "$start_year" --end-year "$end_year" \
      --start-index "$start_index" --end-index "$end_index"
  done
  python src/run_week4_label_diagnostics.py --stage median-finalize \
    --start-year "$start_year" --end-year "$end_year"
done

for range in "0 1000" "1000 2000" "2000 3000" "3000 4000" "4000 5000" "5000 5538"; do
  read -r start_index end_index <<< "$range"
  extra=()
  if [[ "$start_index" == "0" ]]; then extra+=(--reset-labels); fi
  python src/run_week4_label_diagnostics.py --stage labels \
    --start-index "$start_index" --end-index "$end_index" "${extra[@]}"
done

python src/run_week4_label_diagnostics.py --stage finalize
