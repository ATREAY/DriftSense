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
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, sobel


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


def degrade(img: np.ndarray, rng: np.random.Generator, *,
            blur_sigma: float, shot_a: float, read_b: float,
            speckle_sigma: float, edge_strength: float) -> np.ndarray:
    """Full independent SEM degradation pipeline for one captured image."""
    out = edge_brighten(img, strength=edge_strength)
    out = gaussian_filter(out, sigma=blur_sigma)
    out = poisson_gaussian_noise(out, rng, a=shot_a, b=read_b)
    out = speckle_noise(out, rng, sigma=speckle_sigma)
    out = gamma_contrast_jitter(out, rng)
    return out.astype(np.float32)


def reference_degradation_params(rng: np.random.Generator) -> dict:
    return dict(
        blur_sigma=float(rng.uniform(0.3, 0.7)),
        shot_a=float(rng.uniform(0.02, 0.05)),
        read_b=float(rng.uniform(0.01, 0.025)),
        speckle_sigma=float(rng.uniform(0.03, 0.07)),
        edge_strength=float(rng.uniform(0.10, 0.20)),
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
    )
