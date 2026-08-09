"""Fail-fast Captum/cuDNN backward smoke test for the first GRU checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from config import FLAT_DATASET_DIR, FEATURES_5MIN_DIR, SEQ_DATASET_DIR
from run_baseline import load_flat_dataset, rolling_windows
from run_neural_models import sequence_arrays
from run_stage4_attribution import integrated_gradients, load_checkpoint, standardized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args(); device = torch.device(args.device)
    flat = load_flat_dataset(FLAT_DATASET_DIR, FEATURES_5MIN_DIR)
    window = rolling_windows(flat["date"])[0]
    arrays = sequence_arrays(flat, window, SEQ_DATASET_DIR)
    model, payload = load_checkpoint(args.checkpoint, device)
    values = standardized(arrays["test"][0][:args.samples], payload)
    attrs = integrated_gradients(model, values, device)
    if attrs.shape != values.shape or not np.isfinite(attrs).all():
        raise AssertionError("invalid integrated-gradients smoke output")
    print(f"ATTRIBUTION_SMOKE_PASS samples={len(values)} shape={attrs.shape}")


if __name__ == "__main__":
    main()
