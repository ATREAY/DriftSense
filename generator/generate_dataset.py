#!/usr/bin/env python
"""Standalone synthetic dataset generator for DriftSense.

Generates (reference, search, ground-truth-center) triples for the
Applied Materials "Drift-Sense" localization task. Every pair:
  - Reference: 1000x1000 px, native-magnification DRAM-style or
    FinFET-style layout, drawn from one of a small fixed pool of
    structural presets (see generator/presets.py) rather than fully
    randomized geometry -- see the module docstring there for why.
  - Search: 1000x1000 px, rendered from the *same* physical region at
    MAG_RATIO x lower magnification (i.e. covering MAG_RATIO^2 more
    physical area), so the reference content reappears shrunk by
    MAG_RATIO somewhere inside it -- exactly the ~10x scale relationship
    described in the problem statement.
  - Ground truth: the (x, y) pixel center of the reference pattern inside
    the search image, computed analytically from the shared coordinate
    system (not estimated), then carried through independent geometric
    jitter applied to the search image so it stays exact. A gt_box
    (top-left x, y, w, h) is also recorded, matching the reference
    scaffold's manifest schema.

Usage:
  python generator/generate_dataset.py --architecture dram --num-pairs 40 \
      --output-dir data/train

Writes both manifest.json (rich, nested -- used by this repo's own
training/eval code) and manifest.csv (flat columns: id, reference_path,
search_path, gt_x, gt_y, gt_box_x, gt_box_y, gt_box_w, gt_box_h,
architecture, family, seed, ...) for compatibility with the reference
scaffold's conventions.

See docs/CITATIONS.md for the references backing every noise/augmentation
choice made in generator/noise.py and generator/augment.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generator import augment, noise, patterns  # noqa: E402
from generator.presets import preset_names  # noqa: E402

MAG_RATIO = 10
REF_SIZE = 1000
SEARCH_SIZE = 1000
GT_BOX_SIZE = REF_SIZE // MAG_RATIO  # 100 -- matches reference scaffold
MARGIN_PX = 60  # keep embedded reference center away from search borders


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _biased_crop_origin(rng: np.random.Generator, boundary_bias: float) -> tuple[float, float]:
    """With probability `boundary_bias`, bias an axis to land near a
    mat/strip zone boundary (see patterns.MAT_SIZE_NM/STRIP_WIDTH_NM) so
    the reference crop straddles it -- non-periodic structure that is the
    actual disambiguating signal (diagnosed against the reference
    scaffold's src/patterns/zones.py + baseline_solution/zncc.py: its
    60-75%@5px accuracy comes from exactly this boundary-straddling bias,
    not from noise calibration or a smooth shading cue). Otherwise
    uniform-random, which -- since REF_SIZE (1000) is a meaningful
    fraction of MAT_SIZE_NM (2600) -- still lands near a boundary somewhat
    often by chance, matching the reference implementation's behavior.
    """
    def one_axis() -> float:
        if rng.random() < boundary_bias:
            period = patterns.MAT_SIZE_NM + patterns.STRIP_WIDTH_NM
            k = int(rng.integers(0, 8))
            boundary = k * period + patterns.MAT_SIZE_NM
            return max(0.0, boundary + float(rng.uniform(-REF_SIZE / 2, REF_SIZE / 2)))
        return float(rng.uniform(0, 20000))
    return one_axis(), one_axis()


def generate_pair(arch: str, idx: int, rng: np.random.Generator, *,
                   harder_noise: bool, ambiguous_prob: float,
                   max_rotation_deg: float, max_scale_pct: float,
                   boundary_bias: float = 0.35,
                   mag_ratio: float = MAG_RATIO, mag_ratio_jitter_pct: float = 0.0) -> dict:
    family = str(rng.choice(preset_names(arch)))
    ambiguous = rng.random() < ambiguous_prob
    field_amp = 0.0 if ambiguous else float(rng.uniform(0.03, 0.09))
    params = patterns.default_params(arch, rng, family=family, field_amplitude=field_amp)

    # Sample this pair's own magnification ratio. Default 0 jitter -> always
    # exactly MAG_RATIO (the problem statement's stated ~10x scenario);
    # mag_ratio_jitter_pct > 0 draws a per-pair ratio around it, for
    # explicitly testing robustness to scale variation as its own dimension
    # (see judge Q&A on Track 2 dataset composition, docs/CITATIONS.md).
    this_mag_ratio = mag_ratio
    if mag_ratio_jitter_pct > 0:
        lo = mag_ratio * (1 - mag_ratio_jitter_pct / 100.0)
        hi = mag_ratio * (1 + mag_ratio_jitter_pct / 100.0)
        this_mag_ratio = float(rng.uniform(lo, hi))

    ref_ox, ref_oy = _biased_crop_origin(rng, 0.0 if ambiguous else boundary_bias)
    ref_center_x = ref_ox + REF_SIZE / 2.0
    ref_center_y = ref_oy + REF_SIZE / 2.0

    # Target pixel location for the reference's center inside the search
    # canvas, chosen uniformly at random away from the border.
    target_px = float(rng.uniform(MARGIN_PX, SEARCH_SIZE - MARGIN_PX))
    target_py = float(rng.uniform(MARGIN_PX, SEARCH_SIZE - MARGIN_PX))
    search_ox = ref_center_x - target_px * this_mag_ratio
    search_oy = ref_center_y - target_py * this_mag_ratio

    ref_clean = patterns.render_region(arch, params, ref_ox, ref_oy, REF_SIZE, step=1.0)
    search_clean = patterns.render_region(arch, params, search_ox, search_oy, SEARCH_SIZE, step=this_mag_ratio)

    ref_warped, ref_jitter, _ = augment.geometric_jitter(
        ref_clean, rng, max_rotation_deg=max_rotation_deg, max_scale_pct=max_scale_pct)
    search_warped, search_jitter, search_tform = augment.geometric_jitter(
        search_clean, rng, max_rotation_deg=max_rotation_deg, max_scale_pct=max_scale_pct)

    gt_x, gt_y = augment.transform_point(search_tform, (target_px, target_py))

    ref_deg_params = noise.reference_degradation_params(rng)
    search_deg_params = noise.search_degradation_params(rng, harder=harder_noise)
    # Independent RNGs -> independent noise realizations for the two images.
    ref_final = noise.degrade(ref_warped, np.random.default_rng(rng.integers(0, 2**32 - 1)), **ref_deg_params)
    search_final = noise.degrade(search_warped, np.random.default_rng(rng.integers(0, 2**32 - 1)), **search_deg_params)

    in_bounds = 0 <= gt_x < SEARCH_SIZE and 0 <= gt_y < SEARCH_SIZE
    if not in_bounds:
        gt_x = float(np.clip(gt_x, 0, SEARCH_SIZE - 1))
        gt_y = float(np.clip(gt_y, 0, SEARCH_SIZE - 1))

    gt_box_size = REF_SIZE / this_mag_ratio
    gt_box_x = gt_x - gt_box_size / 2.0
    gt_box_y = gt_y - gt_box_size / 2.0

    return {
        "id": f"{arch}_{idx:05d}",
        "architecture": arch,
        "family": family,
        "ambiguous_periodic": ambiguous,
        "mag_ratio": this_mag_ratio,
        "gt_x": gt_x,
        "gt_y": gt_y,
        "gt_box_x": gt_box_x,
        "gt_box_y": gt_box_y,
        "gt_box_w": float(gt_box_size),
        "gt_box_h": float(gt_box_size),
        "pattern_params": {k: v for k, v in params.items() if k != "field"},
        "field_amplitude": field_amp,
        "reference_jitter": ref_jitter,
        "search_jitter": search_jitter,
        "reference_degradation": ref_deg_params,
        "search_degradation": search_deg_params,
        "reference_image": ref_final,
        "search_image": search_final,
    }


def _worker(args_tuple):
    (idx, arch, seed_seq, output_dir, harder_noise, ambiguous_prob,
     max_rotation_deg, max_scale_pct, boundary_bias, mag_ratio, mag_ratio_jitter_pct) = args_tuple
    rng = np.random.default_rng(seed_seq)
    rec = generate_pair(
        arch, idx, rng,
        harder_noise=harder_noise, ambiguous_prob=ambiguous_prob,
        max_rotation_deg=max_rotation_deg, max_scale_pct=max_scale_pct,
        boundary_bias=boundary_bias, mag_ratio=mag_ratio, mag_ratio_jitter_pct=mag_ratio_jitter_pct,
    )
    ref_path = output_dir / "reference" / f"{rec['id']}.png"
    search_path = output_dir / "search" / f"{rec['id']}.png"
    Image.fromarray(to_uint8(rec.pop("reference_image")), mode="L").save(ref_path)
    Image.fromarray(to_uint8(rec.pop("search_image")), mode="L").save(search_path)
    rec["reference_path"] = str(ref_path.relative_to(output_dir))
    rec["search_path"] = str(search_path.relative_to(output_dir))
    return rec


def _flatten_for_csv(rec: dict, seed: int) -> dict:
    flat = {
        "id": rec["id"], "reference_path": rec["reference_path"], "search_path": rec["search_path"],
        "gt_x": rec["gt_x"], "gt_y": rec["gt_y"],
        "gt_box_x": rec["gt_box_x"], "gt_box_y": rec["gt_box_y"],
        "gt_box_w": rec["gt_box_w"], "gt_box_h": rec["gt_box_h"],
        "architecture": rec["architecture"], "family": rec["family"], "seed": seed,
        "ambiguous_periodic": rec["ambiguous_periodic"], "field_amplitude": rec["field_amplitude"],
        "mag_ratio": rec["mag_ratio"],
    }
    for k, v in rec["pattern_params"].items():
        flat[f"struct_{k}"] = v
    for k, v in rec["reference_degradation"].items():
        flat[f"ref_deg_{k}"] = v
    for k, v in rec["search_degradation"].items():
        flat[f"search_deg_{k}"] = v
    for k, v in rec["reference_jitter"].items():
        flat[f"ref_jitter_{k}"] = v
    for k, v in rec["search_jitter"].items():
        flat[f"search_jitter_{k}"] = v
    return flat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--architecture", choices=["dram", "finfet", "both"], default="both",
                     help="Die architecture style to generate.")
    ap.add_argument("--num-pairs", type=int, required=True, help="Number of image pairs to generate.")
    ap.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--harder-noise", action="store_true",
                     help="Widen search-image noise range to emulate the organizers' noisier held-out test set.")
    ap.add_argument("--ambiguous-prob", type=float, default=0.15,
                     help="Fraction of samples with ~0 shading cue -> genuinely hard, fully periodic case.")
    ap.add_argument("--max-rotation-deg", type=float, default=3.0)
    ap.add_argument("--max-scale-pct", type=float, default=5.0)
    ap.add_argument("--boundary-bias", type=float, default=0.35,
                     help="Probability of deliberately biasing the reference crop to straddle a "
                          "mat/strip zone boundary (see generator/patterns.py) -- this, not noise "
                          "calibration, is what makes samples locally disambiguable.")
    ap.add_argument("--mag-ratio", type=float, default=float(MAG_RATIO),
                     help="Reference:search magnification ratio (problem statement's ~10x scenario).")
    ap.add_argument("--mag-ratio-jitter-pct", type=float, default=0.0,
                     help="Draw each pair's own magnification ratio uniformly within +/- this percent "
                          "of --mag-ratio, e.g. 20 -> U(8, 12) for the default 10x. Use this to build a "
                          "dedicated scale-variation test set; 0 (default) always uses exactly --mag-ratio.")
    ap.add_argument("--workers", type=int, default=1,
                     help="Parallel worker processes (each pair is CPU-bound; use ~num cores).")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reference").mkdir(exist_ok=True)
    (args.output_dir / "search").mkdir(exist_ok=True)

    archs = ["dram", "finfet"] if args.architecture == "both" else [args.architecture]
    seed_seqs = np.random.SeedSequence(args.seed).spawn(args.num_pairs)

    def arch_for(i):
        return archs[i % len(archs)] if args.architecture == "both" else archs[0]

    jobs = [
        (i, arch_for(i), seed_seqs[i], args.output_dir, args.harder_noise,
         args.ambiguous_prob, args.max_rotation_deg, args.max_scale_pct, args.boundary_bias,
         args.mag_ratio, args.mag_ratio_jitter_pct)
        for i in range(args.num_pairs)
    ]

    manifest = [None] * args.num_pairs
    t0 = time.time()
    done = 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, job): job[0] for job in jobs}
            for fut in as_completed(futures):
                idx = futures[fut]
                manifest[idx] = fut.result()
                done += 1
                if done % 10 == 0 or done == args.num_pairs:
                    print(f"[{done}/{args.num_pairs}] generated ({time.time() - t0:.1f}s elapsed)")
    else:
        for job in jobs:
            manifest[job[0]] = _worker(job)
            done += 1
            if done % 10 == 0 or done == args.num_pairs:
                print(f"[{done}/{args.num_pairs}] generated ({time.time() - t0:.1f}s elapsed)")

    with open(args.output_dir / "manifest.json", "w") as f:
        json.dump({
            "mag_ratio": args.mag_ratio, "mag_ratio_jitter_pct": args.mag_ratio_jitter_pct,
            "ref_size": REF_SIZE, "search_size": SEARCH_SIZE,
            "num_pairs": args.num_pairs, "seed": args.seed, "harder_noise": args.harder_noise,
            "pairs": manifest,
        }, f, indent=2)

    flat_rows = [_flatten_for_csv(rec, args.seed) for rec in manifest]
    fieldnames = sorted({k for row in flat_rows for k in row})
    fieldnames = [c for c in ["id", "reference_path", "search_path", "gt_x", "gt_y",
                               "gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h",
                               "architecture", "family", "seed"] if c in fieldnames] + \
        sorted(c for c in fieldnames if c not in
               {"id", "reference_path", "search_path", "gt_x", "gt_y",
                "gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h", "architecture", "family", "seed"})
    with open(args.output_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    print(f"Wrote {args.num_pairs} pairs + manifest.json + manifest.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
