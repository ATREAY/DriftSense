#!/usr/bin/env python
"""Classical normalized cross-correlation (NCC) template-matching
baseline -- no training, no learned weights. Given the known ~10x
magnification ratio, the reference is downsampled to its expected
apparent size in the search image and matched via `skimage.feature.
match_template` (which computes NCC in the Fourier domain).

This exists to answer honestly whether DriftSense's learned model
actually beats plain template matching, per the README's core claim, and
to give a concrete number for the "why not just NCC" presentation slide.

Usage:
  python baseline_ncc.py --data-dir data/self_eval --tolerance-px 100
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import match_template
from skimage.transform import resize


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def ncc_localize(reference: np.ndarray, search: np.ndarray, mag_ratio: int = 10):
    ref_h, ref_w = reference.shape
    template_size = max(8, round(ref_h / mag_ratio)), max(8, round(ref_w / mag_ratio))
    template = resize(reference, template_size, anti_aliasing=True)

    result = match_template(search, template, pad_input=True)
    peak = np.unravel_index(np.argmax(result), result.shape)
    return float(peak[1]), float(peak[0])  # (x, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--tolerance-px", type=float, default=100.0)
    ap.add_argument("--mag-ratio", type=int, default=10)
    args = ap.parse_args()

    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    errors, times = [], []
    for rec in manifest["pairs"]:
        reference = load_gray(args.data_dir / rec["reference_path"])
        search = load_gray(args.data_dir / rec["search_path"])
        t0 = time.time()
        px, py = ncc_localize(reference, search, args.mag_ratio)
        times.append(time.time() - t0)
        err = float(np.hypot(px - rec["gt_x"], py - rec["gt_y"]))
        errors.append(err)

    errors = np.array(errors)
    times = np.array(times)
    summary = {
        "num_pairs": len(errors),
        "tolerance_px": args.tolerance_px,
        "accuracy_within_tolerance": float(np.mean(errors <= args.tolerance_px)),
        "mean_error_px": float(errors.mean()),
        "median_error_px": float(np.median(errors)),
        "mean_time_sec": float(times.mean()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
