#!/usr/bin/env python
"""Standalone synthetic dataset generator for DriftSense.

Generates (reference, search, ground-truth-center) triples for the
Applied Materials "Drift-Sense" localization task. Every pair:
  - Reference: 1000x1000 px, native-magnification DRAM-style or
    FinFET-style layout.
  - Search: 1000x1000 px, rendered from the *same* physical region at
    MAG_RATIO x lower magnification (i.e. covering MAG_RATIO^2 more
    physical area), so the reference content reappears shrunk by
    MAG_RATIO somewhere inside it -- exactly the ~10x scale relationship
    described in the problem statement.
  - Ground truth: the (x, y) pixel center of the reference pattern inside
    the search image, computed analytically from the shared coordinate
    system (not estimated), then carried through independent geometric
    jitter applied to the search image so it stays exact.

Usage:
  python generator/generate_dataset.py --architecture dram --num-pairs 40 \
      --output-dir data/train

See docs/CITATIONS.md for the references backing every noise/augmentation
choice made in generator/noise.py and generator/augment.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generator import augment, noise, patterns  # noqa: E402

MAG_RATIO = 10
REF_SIZE = 1000
SEARCH_SIZE = 1000
MARGIN_PX = 60  # keep embedded reference center away from search borders


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def generate_pair(arch: str, idx: int, rng: np.random.Generator, *,
                   harder_noise: bool, ambiguous_prob: float,
                   max_rotation_deg: float, max_scale_pct: float) -> dict:
    ambiguous = rng.random() < ambiguous_prob
    field_amp = 0.0 if ambiguous else float(rng.uniform(0.10, 0.20))
    params = patterns.default_params(arch, rng, field_amplitude=field_amp)

    ref_ox = float(rng.uniform(0, 20000))
    ref_oy = float(rng.uniform(0, 20000))
    ref_center_x = ref_ox + REF_SIZE / 2.0
    ref_center_y = ref_oy + REF_SIZE / 2.0

    # Target pixel location for the reference's center inside the search
    # canvas, chosen uniformly at random away from the border.
    target_px = float(rng.uniform(MARGIN_PX, SEARCH_SIZE - MARGIN_PX))
    target_py = float(rng.uniform(MARGIN_PX, SEARCH_SIZE - MARGIN_PX))
    search_ox = ref_center_x - target_px * MAG_RATIO
    search_oy = ref_center_y - target_py * MAG_RATIO

    ref_clean = patterns.render_region(arch, params, ref_ox, ref_oy, REF_SIZE, step=1.0)
    search_clean = patterns.render_region(arch, params, search_ox, search_oy, SEARCH_SIZE, step=float(MAG_RATIO))

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

    return {
        "id": f"{arch}_{idx:05d}",
        "architecture": arch,
        "ambiguous_periodic": ambiguous,
        "gt_x": gt_x,
        "gt_y": gt_y,
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
     max_rotation_deg, max_scale_pct) = args_tuple
    rng = np.random.default_rng(seed_seq)
    rec = generate_pair(
        arch, idx, rng,
        harder_noise=harder_noise, ambiguous_prob=ambiguous_prob,
        max_rotation_deg=max_rotation_deg, max_scale_pct=max_scale_pct,
    )
    ref_path = output_dir / "reference" / f"{rec['id']}.png"
    search_path = output_dir / "search" / f"{rec['id']}.png"
    Image.fromarray(to_uint8(rec.pop("reference_image")), mode="L").save(ref_path)
    Image.fromarray(to_uint8(rec.pop("search_image")), mode="L").save(search_path)
    rec["reference_path"] = str(ref_path.relative_to(output_dir))
    rec["search_path"] = str(search_path.relative_to(output_dir))
    return rec


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
         args.ambiguous_prob, args.max_rotation_deg, args.max_scale_pct)
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
            "mag_ratio": MAG_RATIO, "ref_size": REF_SIZE, "search_size": SEARCH_SIZE,
            "num_pairs": args.num_pairs, "seed": args.seed, "harder_noise": args.harder_noise,
            "pairs": manifest,
        }, f, indent=2)
    print(f"Wrote {args.num_pairs} pairs + manifest.json to {args.output_dir}")


if __name__ == "__main__":
    main()
