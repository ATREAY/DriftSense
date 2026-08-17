#!/usr/bin/env python
"""Self-evaluation on a generated manifest: accuracy within tolerance,
per-pair inference time, and one SUCCESS + one HONEST-FAILURE
visualization for the presentation (Slide 6 requirements).

Usage:
  python evaluate.py --data-dir data/self_eval --weights weights/driftsense.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.siamese_localizer import SiameseLocalizer, localize


def load_gray_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def make_visualization(ref_path, search_path, gt_xy, pred_xy, out_path, title):
    ref = Image.open(ref_path).convert("RGB").resize((320, 320))
    search = Image.open(search_path).convert("RGB").resize((640, 640))
    scale = 640 / 1000
    d = ImageDraw.Draw(search)
    gx, gy = gt_xy[0] * scale, gt_xy[1] * scale
    px, py = pred_xy[0] * scale, pred_xy[1] * scale
    r = 10
    d.ellipse([gx - r, gy - r, gx + r, gy + r], outline=(0, 200, 0), width=3)
    d.ellipse([px - r, py - r, px + r, py + r], outline=(255, 0, 0), width=3)
    d.line([gx - 2 * r, gy, gx + 2 * r, gy], fill=(0, 200, 0), width=1)
    d.line([px, py - 2 * r, px, py + 2 * r], fill=(255, 0, 0), width=1)

    canvas = Image.new("RGB", (320 + 640 + 20, 660), (255, 255, 255))
    canvas.paste(ref, (10, 10))
    canvas.paste(search, (330, 10))
    d2 = ImageDraw.Draw(canvas)
    d2.text((10, 335), f"{title}  (green=ground truth, red=prediction)", fill=(0, 0, 0))
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=Path("weights/driftsense.pt"))
    ap.add_argument("--tolerance-px", type=float, default=30.0,
                     help="Prediction counts as correct if within this L2 pixel distance of ground truth.")
    ap.add_argument("--output-dir", type=Path, default=Path("evaluation_report"))
    ap.add_argument("--no-ncc-snap", action="store_true",
                     help="Disable the final local-NCC snap, to measure the learned model alone.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.weights, map_location=device)
    cfg = ckpt.get("cfg", {})
    model = SiameseLocalizer(feat_ch=cfg.get("feat_ch", 128), temperature=cfg.get("temperature", 1.0))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    pairs = manifest["pairs"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    best_success, worst_failure = None, None

    for rec in pairs:
        ref_path = args.data_dir / rec["reference_path"]
        search_path = args.data_dir / rec["search_path"]
        reference = load_gray_tensor(ref_path)
        search = load_gray_tensor(search_path)

        t0 = time.time()
        res = localize(model, reference, search, device=device, ncc_snap=not args.no_ncc_snap,
                        mag_ratio=rec.get("mag_ratio", 10))
        dt = time.time() - t0

        gt = np.array([rec["gt_x"], rec["gt_y"]])
        pred = np.array([res.x, res.y])
        err = float(np.linalg.norm(gt - pred))
        model_err = float(np.linalg.norm(gt - np.array([res.model_x, res.model_y])))
        success = err <= args.tolerance_px

        row = {"id": rec["id"], "architecture": rec["architecture"],
               "ambiguous_periodic": rec["ambiguous_periodic"],
               "gt_x": rec["gt_x"], "gt_y": rec["gt_y"],
               "pred_x": res.x, "pred_y": res.y,
               "model_x": res.model_x, "model_y": res.model_y, "model_error_px": model_err,
               "error_px": err, "success": success, "time_sec": dt}
        results.append(row)

        if success and (best_success is None or err < best_success["error_px"]):
            best_success = row
        if not success and (worst_failure is None or err > worst_failure["error_px"]):
            worst_failure = row

    errors = np.array([r["error_px"] for r in results])
    model_errors = np.array([r["model_error_px"] for r in results])
    times = np.array([r["time_sec"] for r in results])
    accuracy = float(np.mean([r["success"] for r in results]))
    model_accuracy = float(np.mean(model_errors <= args.tolerance_px))

    summary = {
        "num_pairs": len(results),
        "tolerance_px": args.tolerance_px,
        "accuracy_within_tolerance": accuracy,
        "mean_error_px": float(errors.mean()),
        "median_error_px": float(np.median(errors)),
        "model_only_accuracy_within_tolerance": model_accuracy,
        "model_only_mean_error_px": float(model_errors.mean()),
        "mean_time_sec": float(times.mean()),
        "median_time_sec": float(np.median(times)),
    }

    print(json.dumps(summary, indent=2))
    (args.output_dir / "results.json").write_text(json.dumps({"summary": summary, "per_pair": results}, indent=2))

    if best_success:
        make_visualization(
            args.data_dir / f"reference/{best_success['id']}.png",
            args.data_dir / f"search/{best_success['id']}.png",
            (best_success["gt_x"], best_success["gt_y"]), (best_success["pred_x"], best_success["pred_y"]),
            args.output_dir / "success_example.png",
            f"SUCCESS: {best_success['id']} err={best_success['error_px']:.1f}px",
        )
    if worst_failure:
        make_visualization(
            args.data_dir / f"reference/{worst_failure['id']}.png",
            args.data_dir / f"search/{worst_failure['id']}.png",
            (worst_failure["gt_x"], worst_failure["gt_y"]), (worst_failure["pred_x"], worst_failure["pred_y"]),
            args.output_dir / "failure_example.png",
            f"HONEST FAILURE: {worst_failure['id']} err={worst_failure['error_px']:.1f}px "
            f"(ambiguous_periodic={worst_failure['ambiguous_periodic']})",
        )
    print(f"Report written to {args.output_dir}")


if __name__ == "__main__":
    main()
