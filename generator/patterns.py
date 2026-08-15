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

# Real chips are not one uniform repeating pattern across a whole field of
# view: a memory/logic device is built from discrete sub-array "mats"
# (blocks of the periodic cell array) separated by strips of visually
# distinct material -- peripheral circuitry, sense-amp/decoder rows,
# global routing. Diagnosed directly against the reference scaffold
# (aayushraina21/drift-sense-synthetic-data, src/patterns/zones.py): its
# classical ZNCC baseline reaches 60-75% accuracy at <=5px specifically
# because most crops straddle (or are deliberately biased to straddle) a
# mat/strip boundary, giving genuine non-periodic local structure to
# match against -- not because of noise calibration or a smooth global
# cue. An earlier version of this generator used one uniform periodic
# pattern per sample (plus a weak smooth shading field), which measured
# at true-match NCC scores statistically indistinguishable from
# random-offset scores (~0.15 vs. up to ~0.43, effectively tied) --
# mat/strip zoning is the actual fix, not a noise-level tweak.
MAT_SIZE_NM = 2600.0
STRIP_WIDTH_NM = 320.0
_ZONE_PERIOD = MAT_SIZE_NM + STRIP_WIDTH_NM


def _in_strip(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xm = np.mod(x, _ZONE_PERIOD)
    ym = np.mod(y, _ZONE_PERIOD)
    return (xm >= MAT_SIZE_NM) | (ym >= MAT_SIZE_NM)


def _mat_cell_parity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Checkerboard parity of which mat cell (x, y) falls in. A single
    fixed preset repeated identically in every mat cell is itself exactly
    periodic at the zone scale (every ~2920nm), which just moves the tie-
    breaking problem up a level instead of solving it -- real floorplans
    don't tile one identical block forever either. Alternating between two
    presets breaks that repeat the same way src/patterns/zones.py's
    per-mat random preset choice does, simplified to two presets on a
    checkerboard (deterministic, cheap to compute per-pixel) rather than
    a full per-cell random draw."""
    ci = np.floor(x / _ZONE_PERIOD)
    cj = np.floor(y / _ZONE_PERIOD)
    return np.mod(ci + cj, 2).astype(np.int32)


def render_strip(x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
    """Flat routing/peripheral-material fill with sparse orthogonal
    interconnect lines -- visually distinct from the dense periodic mats
    it separates, matching the reference scaffold's strip texture."""
    base = params["strip_base"]
    line = params["strip_line"]
    pitch = params["strip_line_pitch"]
    width_frac = params["strip_line_width_frac"]
    phase = params["strip_phase"]
    rows = _periodic_lines(y - phase, pitch, width_frac)
    cols = _periodic_lines(x - phase, pitch, width_frac)
    routing = np.clip(rows + cols, 0, 1)
    return base + routing * (line - base)


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


def _structural_params(arch: str, rng: np.random.Generator, family: str) -> dict:
    if arch == "dram":
        from generator.presets import get_preset
        params = get_preset("dram", family)
        params.update(
            background=float(rng.uniform(0.12, 0.22)),
            line_level=float(rng.uniform(0.55, 0.70)),
            via_level=float(rng.uniform(0.85, 1.0)),
        )
    elif arch == "finfet":
        from generator.presets import get_preset
        params = get_preset("finfet", family)
        params.update(
            background=float(rng.uniform(0.10, 0.20)),
            fin_level=float(rng.uniform(0.55, 0.68)),
            gate_level=float(rng.uniform(0.85, 1.0)),
        )
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return params


def default_params(arch: str, rng: np.random.Generator, *, family: str,
                    field_amplitude: float = 0.06) -> dict:
    """Structural geometry (pitch, width fractions) comes from a fixed
    preset `family` (see generator/presets.py) rather than being drawn
    fresh per sample -- earlier fully-randomized geometry made the
    localization task unlearnable from a few thousand pairs (see README
    "Current results", v1-v4). Only imaging-condition brightness levels
    are still randomized per sample: those represent detector gain /
    contrast-brightness settings that legitimately vary capture-to-
    capture, and a network needs to be invariant to them regardless.

    A second "neighbor" preset (`params["mat_b"]`) is also generated for
    the checkerboard mat-cell alternation in `render_region` -- a single
    preset tiled identically in every mat cell is itself exactly periodic
    at the zone scale, see `_mat_cell_parity`.

    `field_amplitude` sets the strength of an additional smooth
    non-periodic shading cue (see `_sample_field`); kept low by default
    now that fixed-family/zone structure is the primary learnable signal,
    not the sole one. Pass 0 for the fully-ambiguous periodic-only hard case.
    """
    from generator.presets import preset_names
    params = _structural_params(arch, rng, family)

    other_families = [f for f in preset_names(arch) if f != family]
    neighbor_family = str(rng.choice(other_families)) if other_families else family
    params["mat_b"] = _structural_params(arch, rng, neighbor_family)

    params["field"] = _sample_field(rng, n_components=4, wavelength_range=(700, 1900),
                                     amplitude=field_amplitude)
    params.update(
        strip_base=float(rng.uniform(0.32, 0.40)),
        strip_line=float(rng.uniform(0.48, 0.56)),
        strip_line_pitch=float(rng.uniform(180, 260)),
        strip_line_width_frac=float(rng.uniform(0.03, 0.06)),
        strip_phase=float(rng.uniform(0, 220)),
    )
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
            mat_sample_a = RENDERERS[arch](x, y, params)
            mat_sample_b = RENDERERS[arch](x, y, params["mat_b"])
            mat_sample = np.where(_mat_cell_parity(x, y) == 0, mat_sample_a, mat_sample_b)
            strip_sample = render_strip(x, y, params)
            sample = np.where(_in_strip(x, y), strip_sample, mat_sample)
            if params.get("field"):
                sample = sample + _apply_field(x, y, params["field"])
            acc += sample
    acc /= supersample * supersample
    return np.clip(acc, 0.0, 1.0)
