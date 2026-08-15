"""Fixed structural presets for DRAM and FinFET die families.

Earlier versions of this generator drew pitch/linewidth independently at
random for every training pair. Empirically (see README's "Current
results" section, v1-v4) that made the localization task nearly
impossible to learn from a few thousand pairs: the network had to
generalize over an unbounded space of texture statistics instead of
learning to recognize a bounded set of real patterns. The reference
starter scaffold (github.com/aayushraina21's drift-sense-synthetic-data
Hugging Face Space, src/presets.py) uses exactly this fix -- a small
fixed pool of "families" with constant geometry -- and its classical
ZNCC baseline (no training) reaches 60-75% accuracy at <=5px tolerance
on it, versus ~10% at <=100px for our earlier fully-randomized data.
This module follows the same approach.

Only the *structural geometry* (pitch, width fractions) is fixed per
family below; per-sample imaging conditions (background/foreground
brightness levels, representing detector gain / contrast-brightness
settings that legitimately vary capture-to-capture) are still randomized
in generator/patterns.py -- the network still needs to be invariant to
those, and randomizing them is a defensible augmentation, not a source of
unlearnable task variance.

Pitch/linewidth values below are illustrative of publicly documented
process-node scaling trends, not exact proprietary fab specifications
(same disclaimer as the reference scaffold).

References:
  - S. M. Sze & K. K. Ng, "Physics of Semiconductor Devices", 3rd ed.,
    Wiley, 2007 -- DRAM word-line/bit-line pitch scaling as a multiple of
    minimum feature size F.
  - C. Auth et al., "A 22nm high performance and low-power CMOS
    technology featuring fully-depleted tri-gate transistors", IEDM 2012
    -- FinFET fin/gate pitch reference point.
  - IEEE IRDS (International Roadmap for Devices and Systems), "More
    Moore" chapter -- process-node feature-size scaling trend used to
    interpolate the 45nm-7nm FinFET preset ladder below.
"""
from __future__ import annotations

# DRAM: word-line pitch ~2F, bit-line pitch ~3F (Sze & Ng 1T1C array).
DRAM_PRESETS = {
    "DRAM_1X":      dict(pitch=64.0,  line_width_frac=0.22, via_radius_frac=0.17),
    "DRAM_DENSE":   dict(pitch=48.0,  line_width_frac=0.20, via_radius_frac=0.16),
    "DRAM_LOOSE":   dict(pitch=96.0,  line_width_frac=0.22, via_radius_frac=0.17),
    "DRAM_WIDE":    dict(pitch=120.0, line_width_frac=0.24, via_radius_frac=0.18),
    "DRAM_COMPACT": dict(pitch=72.0,  line_width_frac=0.20, via_radius_frac=0.16),
    "DRAM_LEGACY":  dict(pitch=160.0, line_width_frac=0.24, via_radius_frac=0.19),
}

# FinFET: fin pitch 40-140nm / gate pitch 76-260nm / gate length 24-80nm
# across process nodes, per the reference scaffold's stated ranges.
FINFET_PRESETS = {
    "FINFET_45NM": dict(fin_pitch=140.0, gate_pitch=260.0, gate_width_frac=80.0 / 260.0),
    "FINFET_32NM": dict(fin_pitch=110.0, gate_pitch=200.0, gate_width_frac=60.0 / 200.0),
    "FINFET_22NM": dict(fin_pitch=90.0,  gate_pitch=160.0, gate_width_frac=45.0 / 160.0),
    "FINFET_14NM": dict(fin_pitch=70.0,  gate_pitch=120.0, gate_width_frac=35.0 / 120.0),
    "FINFET_10NM": dict(fin_pitch=55.0,  gate_pitch=95.0,  gate_width_frac=28.0 / 95.0),
    "FINFET_7NM":  dict(fin_pitch=40.0,  gate_pitch=76.0,  gate_width_frac=24.0 / 76.0),
}
FINFET_PRESETS = {
    name: dict(p, fin_width_frac=0.36) for name, p in FINFET_PRESETS.items()
}

PRESETS = {"dram": DRAM_PRESETS, "finfet": FINFET_PRESETS}


def preset_names(arch: str) -> list[str]:
    return list(PRESETS[arch].keys())


def get_preset(arch: str, name: str) -> dict:
    return dict(PRESETS[arch][name])
