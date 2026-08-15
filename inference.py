#!/usr/bin/env python
"""Standalone localization inference.

Given a reference image and a search image, prints the predicted (x, y)
pixel center of the reference pattern inside the search image. No manual
edits required -- this is the exact script Applied Materials runs on the
Phase 2 test set.

Usage:
  python inference.py --reference path/to/reference.png --search path/to/search.png \
      --weights weights/driftsense.pt

Prints a single line: "x,y" (floating point, in the search image's own
pixel coordinate system -- rescaled automatically if the input image
isn't exactly 1000x1000).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.siamese_localizer import SiameseLocalizer, localize


def load_gray_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, required=True, help="Path to reference image.")
    ap.add_argument("--search", type=Path, required=True, help="Path to search image.")
    ap.add_argument("--weights", type=Path, default=Path(__file__).resolve().parent / "weights" / "driftsense.pt",
                     help="Path to trained model checkpoint (.pt).")
    ap.add_argument("--mag-ratio", type=int, default=10,
                     help="Known magnification ratio between reference and search (dataset parameter).")
    ap.add_argument("--quiet", action="store_true", help="Print only 'x,y' with no extra text.")
    args = ap.parse_args()

    device = pick_device()
    ckpt = torch.load(args.weights, map_location=device)
    cfg = ckpt.get("cfg", {})
    model = SiameseLocalizer(feat_ch=cfg.get("feat_ch", 128), temperature=cfg.get("temperature", 1.0))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    reference = load_gray_tensor(args.reference)
    search = load_gray_tensor(args.search)

    t0 = time.time()
    result = localize(model, reference, search, mag_ratio=args.mag_ratio, device=device)
    dt = time.time() - t0

    if args.quiet:
        print(f"{result.x:.2f},{result.y:.2f}")
    else:
        print(f"device: {device}")
        print(f"coarse prediction: ({result.coarse_x:.2f}, {result.coarse_y:.2f})")
        print(f"predicted (x, y): ({result.x:.2f}, {result.y:.2f})")
        print(f"inference time: {dt * 1000:.1f} ms")


if __name__ == "__main__":
    main()
