"""Independent geometric jitter applied per capture (reference and search
are warped separately), modeling stage-rotation and magnification drift
between two SEM acquisitions of the same die region.

Reference:
  - C. Shorten & T. M. Khoshgoftaar, "A survey on Image Data Augmentation
    for Deep Learning", Journal of Big Data 6, 60 (2019) -- rotation and
    scale jitter as standard, justified augmentations for viewpoint/
    acquisition-drift robustness.
"""
from __future__ import annotations

import numpy as np
from skimage.transform import AffineTransform, warp


def geometric_jitter(img: np.ndarray, rng: np.random.Generator, *,
                      max_rotation_deg: float, max_scale_pct: float) -> tuple[np.ndarray, dict]:
    angle = np.deg2rad(rng.uniform(-max_rotation_deg, max_rotation_deg))
    scale = 1.0 + rng.uniform(-max_scale_pct, max_scale_pct) / 100.0
    h, w = img.shape
    center = np.array([w / 2.0, h / 2.0])

    tform = (
        AffineTransform(translation=-center)
        + AffineTransform(rotation=angle, scale=(scale, scale))
        + AffineTransform(translation=center)
    )
    warped = warp(img, tform.inverse, mode="edge", preserve_range=True)
    info = {"rotation_deg": float(np.rad2deg(angle)), "scale": float(scale)}
    return warped.astype(np.float32), info, tform


def transform_point(tform: AffineTransform, point_xy: tuple[float, float]) -> tuple[float, float]:
    """Map a point defined in the pre-warp image into post-warp pixel
    coordinates, so ground truth stays exact after geometric jitter.
    """
    out = tform(np.array([point_xy]))[0]
    return float(out[0]), float(out[1])
