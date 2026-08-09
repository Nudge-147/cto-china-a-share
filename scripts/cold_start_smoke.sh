#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${1:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/a_share_5min_mpl"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/a_share_5min_cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

echo "[1/4] Python and dependency imports"
"$PYTHON_BIN" -c "import baostock, lightgbm, matplotlib, numpy, pandas, pyarrow, sklearn, torch"

echo "[2/4] Source compilation"
"$PYTHON_BIN" -m compileall -q src tests ./*.py

echo "[3/4] CLI entry-point smoke checks"
for script in download_5min.py quality_check.py repair_5min.py features_5min.py \
              build_5min_dataset.py run_baseline.py run_neural_models.py \
              run_stage4_analysis.py run_stage4_attribution.py \
              run_stage4_inference.py finalize_stage4.py; do
  "$PYTHON_BIN" "$script" --help >/dev/null
done

echo "[4/4] Unit tests"
"$PYTHON_BIN" -m unittest discover -s tests -v
echo "COLD_START_SMOKE=PASS"
