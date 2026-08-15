"""SEM-realistic degradation pipeline.

Ideal layout -> edge brightening -> blur -> Poisson-Gaussian (shot+read)
noise -> speckle -> contrast/gamma jitter -> final image.

Every call generates an *independent* noise realization (fresh RNG draws),
so calling this twice on the same clean image -- once for the reference,
once for the search patch -- never reuses the same noise pattern. This
mirrors two physically separate SEM captures, per the dataset generator's
mandatory "independent sensor noise" requirement.

References:
  - A. Foi, M. Trimeche, V. Katkovnik, K. Egiazarian, "Practical
    Poissonian-Gaussian Noise Modeling and Fitting for Single-Image
    Raw-Data", IEEE Trans. Image Processing, 2008 -- signal-dependent
    shot noise + additive read noise model used below.
  - J. W. Goodman, "Speckle Phenomena in Optics: Theory and Applications",
    SPIE Press, 2007 -- multiplicative speckle noise model.
  - L. Reimer, "Scanning Electron Microscopy: Physics of Image Formation
    and Microanalysis", 2nd ed., Springer, 1998 -- edge effect: secondary
    electron yield (and hence brightness) rises sharply at topographic
    edges/feature boundaries, which is what the edge-brightening step
    approximates via a gradient-magnitude-weighted intensity boost.
  - M. T. Postek & A. E. Vladar, "Does your SEM really tell the truth?
    How would you know?", Proc. SPIE 8036, 2011 -- noise/artifact
    characteristics specific to semiconductor-metrology SEM imaging.
  - R. Hartley & A. Zisserman, "Multiple View Geometry in Computer
    Vision", 2nd ed., Cambridge University Press, 2004 -- radial
    (barrel/pincushion) lens distortion model used in
    `barrel_distortion`, and the vignetting (radial illumination
    falloff) model used in `vignette`.
  - J. Cazaux, "Charging in scanning electron microscopy: from
    historical understanding to the `dynamic double layer'", Journal of
    Applied Physics 110, 2011 -- sample charging on insulating/oxide
    regions producing bright streaking artifacts, modeled in
    `charging_streaks`.
  - R. C. Gonzalez & R. E. Woods, "Digital Image Processing", 4th ed.,
    Pearson, 2018 -- impulse (salt-and-pepper) noise model for dead/hot
    detector pixels, used in `salt_and_pepper`.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, sobel


def edge_brighten(img: np.ndarray, strength: float, sigma: float = 1.0) -> np.ndarray:
    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    grad_mag = np.hypot(gx, gy)
    grad_mag = grad_mag / (grad_mag.max() + 1e-8)
    grad_mag = gaussian_filter(grad_mag, sigma=sigma)
    return np.clip(img + strength * grad_mag, 0.0, 1.0)


def poisson_gaussian_noise(img: np.ndarray, rng: np.random.Generator, a: float, b: float) -> np.ndarray:
    """Signal-dependent shot noise (Poisson, scaled by `a`) plus additive
    Gaussian read noise (std `b`). Foi et al. 2008.
    """
    safe = np.clip(img, 0.0, 1.0)
    shot = rng.poisson(safe / a) * a
    read = rng.normal(0.0, b, size=img.shape)
    return np.clip(shot + read, 0.0, 1.0)


def speckle_noise(img: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    n = rng.normal(0.0, sigma, size=img.shape)
    return np.clip(img + img * n, 0.0, 1.0)


def gamma_contrast_jitter(img: np.ndarray, rng: np.random.Generator,
                           gamma_range=(0.85, 1.18), contrast_range=(0.9, 1.12)) -> np.ndarray:
    gamma = rng.uniform(*gamma_range)
    contrast = rng.uniform(*contrast_range)
    out = np.clip(img, 0, 1) ** gamma
    out = np.clip((out - 0.5) * contrast + 0.5, 0.0, 1.0)
    return out


def vignette(img: np.ndarray, strength: float) -> np.ndarray:
    """Radial darkening toward the frame edges, from off-axis beam/
    detector collection efficiency falloff (Hartley & Zisserman 2004
    vignetting model: intensity falls off with normalized radius^2)."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    r2 = ((xx - cx) / (w / 2.0)) ** 2 + ((yy - cy) / (h / 2.0)) ** 2
    falloff = 1.0 - strength * np.clip(r2, 0, 1)
    return np.clip(img * falloff, 0.0, 1.0)


def barrel_distortion(img: np.ndarray, k: float) -> np.ndarray:
    """Radial lens-style distortion (barrel if k>0, pincushion if k<0)
    from imperfect beam-scan linearity/calibration (Hartley & Zisserman
    2004 radial distortion model)."""
    if k == 0:
        return img
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    nx, ny = (xx - cx) / (w / 2.0), (yy - cy) / (h / 2.0)
    r2 = nx ** 2 + ny ** 2
    factor = 1.0 + k * r2
    src_x = cx + nx * factor * (w / 2.0)
    src_y = cy + ny * factor * (h / 2.0)
    return map_coordinates(img, [src_y, src_x], order=1, mode="nearest").astype(np.float32)


def charging_streaks(img: np.ndarray, rng: np.random.Generator, *,
                      streak_prob: float, intensity: float) -> np.ndarray:
    """Occasional bright horizontal streaks from local sample charging,
    common on insulating/oxide regions (Cazaux 2011)."""
    out = img.copy()
    n_rows = out.shape[0]
    mask = rng.random(n_rows) < streak_prob
    for row in np.nonzero(mask)[0]:
        width = int(rng.integers(2, 6))
        r0, r1 = max(0, row - width // 2), min(n_rows, row + width // 2 + 1)
        out[r0:r1, :] = np.clip(out[r0:r1, :] + intensity, 0.0, 1.0)
    return out


def salt_and_pepper(img: np.ndarray, rng: np.random.Generator, prob: float) -> np.ndarray:
    """Impulse noise from dead/hot detector pixels or discharge events
    (Gonzalez & Woods 2018)."""
    out = img.copy()
    mask = rng.random(img.shape)
    out[mask < prob / 2] = 0.0
    out[mask > 1 - prob / 2] = 1.0
    return out


def degrade(img: np.ndarray, rng: np.random.Generator, *,
            blur_sigma: float, shot_a: float, read_b: float,
            speckle_sigma: float, edge_strength: float,
            vignette_strength: float = 0.0, barrel_k: float = 0.0,
            streak_prob: float = 0.0, streak_intensity: float = 0.3,
            salt_pepper_prob: float = 0.0) -> np.ndarray:
    """Full independent SEM degradation pipeline for one captured image."""
    out = edge_brighten(img, strength=edge_strength)
    out = gaussian_filter(out, sigma=blur_sigma)
    out = barrel_distortion(out, k=barrel_k)
    out = vignette(out, strength=vignette_strength)
    out = poisson_gaussian_noise(out, rng, a=shot_a, b=read_b)
    out = speckle_noise(out, rng, sigma=speckle_sigma)
    out = charging_streaks(out, rng, streak_prob=streak_prob, intensity=streak_intensity)
    out = salt_and_pepper(out, rng, prob=salt_pepper_prob)
    out = gamma_contrast_jitter(out, rng)
    return out.astype(np.float32)


def reference_degradation_params(rng: np.random.Generator) -> dict:
    # shot_a/read_b/speckle_sigma were originally calibrated ~3x higher
    # and, combined with the search-side noisier-scale multiplier below,
    # pushed even the *correct* match's NCC score down near the noise
    # floor of incorrect locations (measured directly: true-location NCC
    # ~0.15 vs. random-location NCC up to ~0.43 on 1.0x-scale search
    # noise) -- i.e. noise this strong makes the task nearly unsolvable
    # by *any* method, not just a meaningfully hard one. Reduced so the
    # true match stays clearly separable from noise on average while
    # still being a real, independently-drawn degradation per capture.
    return dict(
        blur_sigma=float(rng.uniform(0.2, 0.4)),
        shot_a=float(rng.uniform(0.006, 0.015)),
        read_b=float(rng.uniform(0.003, 0.008)),
        speckle_sigma=float(rng.uniform(0.01, 0.025)),
        edge_strength=float(rng.uniform(0.10, 0.20)),
        vignette_strength=float(rng.uniform(0.0, 0.08)),
        barrel_k=float(rng.uniform(-0.03, 0.03)),
        streak_prob=float(rng.uniform(0.0, 0.002)),
        streak_intensity=float(rng.uniform(0.15, 0.35)),
        salt_pepper_prob=float(rng.uniform(0.0, 0.001)),
    )


def search_degradation_params(rng: np.random.Generator, *, harder: bool = False) -> dict:
    """Search-image noise is drawn from a strictly noisier range than the
    reference, per the mandatory requirement that search images carry more
    noise than the reference on real test data. `harder=True` widens the
    range further to emulate the organizers' held-out (noisier) test set.
    """
    lo, hi = (1.3, 1.9) if not harder else (1.8, 2.6)
    scale = float(rng.uniform(lo, hi))
    base = reference_degradation_params(rng)
    return dict(
        blur_sigma=base["blur_sigma"] * scale,
        shot_a=base["shot_a"] * scale,
        read_b=base["read_b"] * scale,
        speckle_sigma=base["speckle_sigma"] * scale,
        edge_strength=base["edge_strength"],
        vignette_strength=float(np.clip(base["vignette_strength"] * scale, 0, 0.35)),
        barrel_k=base["barrel_k"],
        streak_prob=float(np.clip(base["streak_prob"] * scale, 0, 0.01)),
        streak_intensity=base["streak_intensity"],
        salt_pepper_prob=float(np.clip(base["salt_pepper_prob"] * scale, 0, 0.005)),
    )
