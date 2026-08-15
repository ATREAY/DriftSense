"""Procedural DRAM / FinFET layout generators.

Both patterns are defined as pure coordinate functions f(x, y) -> intensity,
evaluated on a physical (master) coordinate grid rather than rendered once
into a fixed-size bitmap. This is what keeps the reference crop and its
embedded appearance inside the search image *pixel-consistent*: the search
image is produced by resampling the exact same function at a 10x coarser
step size, so the reference region is not pasted/composited into the search
canvas (which would need seam blending) -- it falls out of the sampling
grid automatically, the way a real SEM would image the same physical die
region at two different magnifications.

References for the structural parameters chosen here:
  - S. M. Sze & K. K. Ng, "Physics of Semiconductor Devices", 3rd ed.,
    Wiley, 2007 -- DRAM 1T1C cell array: word-line/bit-line orthogonal grid
    with a storage-node contact at each crossing.
  - C. Auth et al., "A 22nm high performance and low-power CMOS technology
    featuring fully-depleted tri-gate transistors", IEDM 2012 -- FinFET
    layout: dense parallel fins crossed by orthogonal gate lines.
"""
from __future__ import annotations

import numpy as np


def _periodic_lines(coord: np.ndarray, pitch: float, width_frac: float) -> np.ndarray:
    """1 inside a line of given fractional width within each pitch period, else 0."""
    phase = np.mod(coord, pitch) / pitch
    return (phase < width_frac).astype(np.float32)


def render_dram(x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
    """Word-lines (horizontal) x bit-lines (vertical) grid with a via dot
    at every intersection. High contrast, fine pitch, extremely regular.
    """
    pitch = params["pitch"]
    line_w = params["line_width_frac"]
    via_r = params["via_radius_frac"] * pitch

    word_lines = _periodic_lines(y, pitch, line_w)
    bit_lines = _periodic_lines(x, pitch, line_w)
    grid = np.clip(word_lines + bit_lines, 0, 1)

    cx = (np.round(x / pitch) + 0.5) * pitch
    cy = (np.round(y / pitch) + 0.5) * pitch
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    vias = (dist < via_r).astype(np.float32)

    img = params["background"] + grid * (params["line_level"] - params["background"])
    img = np.where(vias > 0, params["via_level"], img)
    return img.astype(np.float32)


def render_finfet(x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
    """Dense parallel vertical fins crossed by one or two horizontal gate
    bars near the intersection region. High-contrast vertical structure
    with distinctive gate crossings.
    """
    fin_pitch = params["fin_pitch"]
    fin_w = params["fin_width_frac"]
    gate_pitch = params["gate_pitch"]
    gate_w = params["gate_width_frac"]

    fins = _periodic_lines(x, fin_pitch, fin_w)
    gates = _periodic_lines(y, gate_pitch, gate_w)

    img = np.full_like(x, params["background"], dtype=np.float32)
    img = np.where(fins > 0, params["fin_level"], img)
    # gate bars sit on top of / brighten the crossing region
    crossing = np.clip(fins + gates, 0, 1) * gates
    img = np.where(crossing > 0, params["gate_level"], img)
    return img.astype(np.float32)


RENDERERS = {"dram": render_dram, "finfet": render_finfet}


def _sample_field(rng: np.random.Generator, n_components: int, wavelength_range: tuple,
                   amplitude: float) -> list[dict]:
    """Coefficients for a smooth, non-periodic large-scale shading field:
    a sum of a few random-orientation sinusoids with incommensurate
    wavelengths. This gives the search canvas a globally unique low-
    frequency "die shading" cue on top of the locally-periodic structure --
    real inspection images are not perfectly flat-field (stage tilt, dose
    drift, vignetting), and without *some* non-periodic cue the
    localization task inside a purely periodic array is mathematically
    ambiguous up to the lattice period. `amplitude` controls task
    difficulty: near 0 reproduces the fully ambiguous periodic-only case
    the organizers require at least one instance of; larger values give an
    easier, more uniquely-identifiable region.
    """
    comps = []
    for _ in range(n_components):
        theta = float(rng.uniform(0, 2 * np.pi))
        wavelength = float(rng.uniform(*wavelength_range))
        phase = float(rng.uniform(0, 2 * np.pi))
        comps.append({"theta": theta, "wavelength": wavelength, "phase": phase})
    return [{"amplitude": amplitude / n_components, **c} for c in comps]


def _apply_field(x: np.ndarray, y: np.ndarray, field: list[dict]) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    for c in field:
        proj = x * np.cos(c["theta"]) + y * np.sin(c["theta"])
        out += c["amplitude"] * np.sin(2 * np.pi * proj / c["wavelength"] + c["phase"])
    return out


def default_params(arch: str, rng: np.random.Generator, *, field_amplitude: float = 0.15) -> dict:
    """Randomize structural parameters per sample so no two layouts are
    pixel-identical, while keeping them within realistic device ranges.
    `field_amplitude` sets the strength of the non-periodic shading cue
    (see `_sample_field`); pass ~0 to generate a maximally-ambiguous,
    fully-periodic hard case.
    """
    # Pitches are picked so that after the MAG_RATIO=10x zoom-out to search
    # magnification the period is still a few pixels wide (weakly
    # resolvable, matching "the reference pattern appears shrunk ~10x
    # somewhere inside" rather than fully aliasing away into flat noise).
    if arch == "dram":
        params = dict(
            pitch=float(rng.uniform(70, 100)),
            line_width_frac=float(rng.uniform(0.16, 0.24)),
            via_radius_frac=float(rng.uniform(0.14, 0.20)),
            background=float(rng.uniform(0.12, 0.22)),
            line_level=float(rng.uniform(0.55, 0.70)),
            via_level=float(rng.uniform(0.85, 1.0)),
        )
    elif arch == "finfet":
        params = dict(
            fin_pitch=float(rng.uniform(70, 100)),
            fin_width_frac=float(rng.uniform(0.30, 0.42)),
            gate_pitch=float(rng.uniform(500, 700)),
            gate_width_frac=float(rng.uniform(0.10, 0.16)),
            background=float(rng.uniform(0.10, 0.20)),
            fin_level=float(rng.uniform(0.55, 0.68)),
            gate_level=float(rng.uniform(0.85, 1.0)),
        )
    else:
        raise ValueError(f"unknown arch {arch!r}")
    # Wavelengths are kept comparable to (not much larger than) the
    # reference's own 1000-unit physical footprint: if a period is far
    # wider than what the reference can see, the reference only samples a
    # near-linear sliver of the field and can't encode a genuinely
    # matchable positional "fingerprint" from it (empirically this made
    # v1/v2 models fail to generalize past training-set memorization even
    # though the underlying architecture could learn -- see experiments/
    # dram_finfet_v1 and _v2 history). Shorter wavelengths let the
    # reference capture enough of a cycle (peak/trough/slope) to be
    # locally distinctive.
    params["field"] = _sample_field(rng, n_components=4, wavelength_range=(700, 1900),
                                     amplitude=field_amplitude)
    return params


def render_region(arch: str, params: dict, ox: float, oy: float, size: int, step: float) -> np.ndarray:
    """Render a size x size image sampling the master pattern starting at
    physical origin (ox, oy) with a given physical-units-per-pixel `step`.
    step=1 -> reference (native) magnification; step=MAG_RATIO -> search
    (low) magnification.

    When step > 1 the pattern's sharp line/via edges contain harmonics
    well above the output Nyquist rate, so naive point-sampling produces
    Moire aliasing rather than the smooth blur a real lower-magnification
    SEM capture would show (beam spot + pixel-averaging act as an
    anti-alias filter). We approximate that filter with box-filter
    supersampling: each output pixel is the average of an S x S grid of
    sub-samples across its physical footprint.
    """
    supersample = 1 if step <= 1.5 else min(8, int(np.ceil(step)))
    sub = (np.arange(supersample, dtype=np.float32) + 0.5) / supersample - 0.5

    acc = np.zeros((size, size), dtype=np.float32)
    for dy in sub:
        for dx in sub:
            coords_x = (np.arange(size, dtype=np.float32) + 0.5 + dx) * step
            coords_y = (np.arange(size, dtype=np.float32) + 0.5 + dy) * step
            x = ox + coords_x[None, :]
            y = oy + coords_y[:, None]
            x = np.broadcast_to(x, (size, size))
            y = np.broadcast_to(y, (size, size))
            sample = RENDERERS[arch](x, y, params)
            if params.get("field"):
                sample = sample + _apply_field(x, y, params["field"])
            acc += sample
    acc /= supersample * supersample
    return np.clip(acc, 0.0, 1.0)
