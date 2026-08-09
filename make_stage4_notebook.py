"""Build the clean, self-contained Kaggle Stage-4 recovery notebook."""
from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "kaggle_stage4_attribution_cost_clean.ipynb"
EMBEDDED = {
    "config.py": ROOT / "config.py",
    "run_baseline.py": ROOT / "run_baseline.py",
    "run_stage4_attribution.py": ROOT / "run_stage4_attribution.py",
    "run_stage4_inference.py": ROOT / "run_stage4_inference.py",
    "run_stage4_analysis.py": ROOT / "run_stage4_analysis.py",
    "sh.000300_daily.parquet": ROOT / "data/stage4_report/sh.000300_daily.parquet",
}


def payloads() -> dict[str, str]:
    """Encode current local sources for lossless reconstruction on Kaggle."""
    return {name: base64.b64encode(gzip.compress(path.read_bytes())).decode()
            for name, path in EMBEDDED.items()}


def code_cell(source: str) -> dict[str, object]:
    """Create a notebook code cell."""
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def markdown_cell(source: str) -> dict[str, object]:
    """Create a notebook markdown cell."""
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def bootstrap_cell(encoded: dict[str, str]) -> str:
    """Return environment and source reconstruction code."""
    serial = json.dumps(encoded, separators=(",", ":"))
    return f'''from pathlib import Path
import base64, gzip, json, os, shutil, subprocess, sys, time, zipfile
import kagglehub, pandas as pd, torch

print("STAGE4_CLEAN_BOOT", flush=True)
sources = list(Path("/kaggle/input").rglob("run_neural_models.py"))
assert sources, "Attach the Stage-3 100-stock pilot Dataset"
source = max((p.parent for p in sources), key=lambda p: int((p/"data/dataset/flat").exists()) + int((p/"data/dataset/seq").exists()))
project = Path("/kaggle/working/stage4_clean_project")
project.mkdir(parents=True, exist_ok=True)
for path in source.glob("*.py"):
    shutil.copy2(path, project/path.name)
payloads = {serial}
for name, payload in payloads.items():
    (project/name).write_bytes(gzip.decompress(base64.b64decode(payload)))
data_link = project/"data"
if data_link.is_symlink():
    data_link.unlink()
if not data_link.exists():
    data_link.symlink_to(source/"data", target_is_directory=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "captum>=0.7"])
from captum.attr import IntegratedGradients
assert torch.cuda.is_available(), "Enable GPU T4 x2"
print("STAGE4_CLEAN_ENV_OK", torch.__version__, torch.cuda.get_device_name(0), flush=True)
'''


def restore_cell() -> str:
    """Return immutable Version-5 checkpoint restoration code."""
    return '''v5 = Path("/kaggle/working/stage4_v5_download")
if v5.exists():
    shutil.rmtree(v5)
downloaded = Path(kagglehub.dataset_download("nudge147/a-share-5min-stage4-v5-immutable"))
shutil.copytree(downloaded, v5)
for archive_path in list(v5.rglob("*.zip")):
    try:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(v5/"unzipped")
    except zipfile.BadZipFile:
        pass
artifacts = Path("/kaggle/working/stage4_v5_artifacts")
if artifacts.exists():
    shutil.rmtree(artifacts)
artifacts.mkdir()
for path in v5.rglob("*"):
    if not path.is_file() or "artifacts" not in path.parts:
        continue
    index = path.parts.index("artifacts")
    relative = Path(*path.parts[index + 1:])
    if relative.parts:
        target = artifacts/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
assert len(list(artifacts.glob("*.pt"))) == 80
assert len(list(artifacts.glob("*.parquet"))) == 80
print("VERSION5_RESTORED", 80, "checkpoints", 80, "predictions", flush=True)
'''


def attribution_cell() -> str:
    """Return bounded, restart-safe attribution runner code."""
    return '''attr_out = Path("/kaggle/working/stage4_attribution_clean")
attr_out.mkdir(exist_ok=True)
env = os.environ.copy()
# Kaggle currently pairs torch 2.10/cu128 with legacy P100 runners.  That
# wheel has no executable Pascal kernel, so attribution deliberately uses
# CPU for reproducibility across accelerator assignments.
env["CUDA_VISIBLE_DEVICES"] = ""
command = [sys.executable, "-u", str(project/"run_stage4_attribution.py"),
    "--windows", *map(str, range(1, 9)), "--artifact-dir", str(artifacts),
    "--flat-dir", str(project/"data/dataset/flat"),
    "--feature-dir", str(project/"data/features_5min"),
    "--output-dir", str(attr_out), "--device", "cpu",
    "--sample-count", "400", "--steps", "32", "--heartbeat-every", "50",
    "--window-timeout-seconds", "1200"]
subprocess.check_call(command, env=env)
assert len(list(attr_out.glob("window_*_gru_attribution_heatmap.csv"))) == 8
print("ATTRIBUTION_COMPLETE 8/8", flush=True)
'''


def cost_cell() -> str:
    """Return baseline, inference, cost, and durable upload code."""
    return '''baseline = Path("/kaggle/working/stage4_baseline_clean")
if len(list((baseline/"artifacts").glob("*.joblib"))) != 16:
    subprocess.check_call([sys.executable, "-u", str(project/"run_baseline.py"),
        "--flat-dir", str(project/"data/dataset/flat"),
        "--feature-dir", str(project/"data/features_5min"),
        "--output-dir", str(baseline), "--save-predictions"])
prediction_dir = Path("/kaggle/working/stage4_six_signal_clean")
prediction_dir.mkdir(exist_ok=True)
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = ""
print("INFERENCE_DEVICE cpu (portable fallback)", flush=True)
subprocess.check_call([sys.executable, "-u", str(project/"run_stage4_inference.py"),
    "--baseline-artifact-dir", str(baseline/"artifacts"),
    "--neural-artifact-dir", str(artifacts), "--output-dir", str(prediction_dir),
    "--flat-dir", str(project/"data/dataset/flat"),
    "--feature-dir", str(project/"data/features_5min"),
    "--seq-dir", str(project/"data/dataset/seq"), "--device", "cpu"], env=env)
assert len(list(prediction_dir.glob("*.parquet"))) == 96

daily = Path("/kaggle/working/daily_reference_from_5min")
daily.mkdir(exist_ok=True)
for path in (project/"data/features_5min").glob("*.parquet"):
    frame = pd.read_parquet(path, columns=["date", "code", "close", "time"])
    frame = frame.sort_values("time").groupby(["date", "code"], as_index=False).tail(1)
    frame["tradestatus"] = 1
    frame[["date", "code", "close", "tradestatus"]].to_parquet(daily/path.name, index=False)

comparison_parts = list(v5.rglob("four_model_comparison.csv"))
assert comparison_parts, "Version 5 comparison shards are missing"
comparison = Path("/kaggle/working/four_model_comparison_all_windows.csv")
pd.concat([pd.read_csv(p) for p in comparison_parts], ignore_index=True).drop_duplicates().to_csv(comparison, index=False)
index_path = project/"sh.000300_daily.parquet"
report = Path("/kaggle/working/stage4_final_report")
subprocess.check_call([sys.executable, "-u", str(project/"run_stage4_analysis.py"),
    "--baseline-artifact-dir", str(baseline/"artifacts"),
    "--neural-artifact-dir", str(artifacts), "--comparison-path", str(comparison),
    "--daily-prediction-dir", str(prediction_dir), "--index-path", str(index_path),
    "--daily-dir", str(daily), "--output-dir", str(report)])

durable = Path("/kaggle/working/stage4_clean_outputs")
durable.mkdir(exist_ok=True)
with zipfile.ZipFile(durable/"stage4_clean_complete.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for root in (attr_out, report, prediction_dir, baseline):
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to("/kaggle/working"))
kagglehub.dataset_upload("nudge147/a-share-5min-stage4-checkpoints", str(durable),
                         version_notes="clean notebook reproducibility run")
print("STAGE4_CLEAN_COMPLETE", flush=True)
'''


def summary_cell() -> str:
    """Return concise result inspection code."""
    return '''summary = pd.read_csv(report/"cost_summary_full_period.csv")
display(summary)
ranking = pd.concat([pd.read_csv(p) for p in attr_out.glob("window_*_gru_feature_ranking.csv")])
display(ranking.groupby("feature", as_index=False)["share"].mean().sort_values("share", ascending=False))
display(pd.read_csv(report/"gru_seed_independence_audit.csv"))
'''


def build() -> dict[str, object]:
    """Assemble the notebook document."""
    cells = [markdown_cell("# A-share 5-minute Stage 4 — clean reproducibility run\n\n"
        "Restores immutable checkpoint Dataset Version 5, runs bounded GRU IG, "
        "rebuilds baselines, executes cost analysis, and uploads durable outputs.\n"),
        code_cell(bootstrap_cell(payloads())), code_cell(restore_cell()),
        code_cell(attribution_cell()), code_cell(cost_cell()), code_cell(summary_cell())]
    return {"cells": cells, "metadata": {"accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(build(), ensure_ascii=False, indent=1))
    print(OUTPUT)
