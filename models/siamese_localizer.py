"""Full localization model: shared encoder -> depthwise cross-correlation
-> small refinement head -> heatmap -> soft-argmax coordinate -> sub-pixel
regression -> (post-hoc) local NCC snap.

Scale handling: the ~10x magnification ratio between reference and search
is a known, fixed dataset parameter (not something the network needs to
learn), so the reference is pre-resized down by that ratio before encoding
-- exactly matching the apparent size its content will have inside the
search image. This is what lets a single-scale (no scale-pyramid)
cross-correlation work, unlike generic trackers that must search over
scale because their target's size in-frame is unknown.

Coarse-to-fine: a first pass runs on the full 1000x1000 search image
(downsampled to a fixed network input size) to get an approximate peak,
then a second pass re-runs the same encoder on a small high-resolution
crop taken from the *original* search image around that peak.

Sub-pixel head + NCC snap (v6): diagnosed directly from v5's results --
the 25x25 response map has a 16px cell stride, which puts a hard floor on
precision no matter how well-trained the encoder is, and classical NCC
(exhaustive per-pixel search, no such floor) already outperformed the
learned model at tight tolerance. Two additions address this without a
full redesign: (1) `SubpixelHead` regresses a bounded continuous
correction from the local neighborhood of the response peak, trained
end-to-end; (2) `localize()` finishes with a small local NCC search
(classical, untrained) confined to a window around the model's own
prediction -- cheap and free of the periodicity-tie problem that a
whole-image NCC search has, since the model has already narrowed down
the right neighborhood.

Reference:
  - L. Bertinetto et al., "Fully-Convolutional Siamese Networks for Object
    Tracking", ECCV 2016 Workshops (SiamFC) -- template/search input-size
    convention and cross-correlation response-map design this follows.
  - Subpixel correction via local peak-neighborhood regression is a
    standard way to recover precision beyond a coarse correlation grid's
    native resolution, e.g. in patch-based subpixel alignment (see
    docs/CITATIONS.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Encoder
from .correlation import xcorr_depthwise

EXEMPLAR_INPUT = 128
SEARCH_INPUT = 512
ENCODER_STRIDE = 16


class RefineHead(nn.Module):
    def __init__(self, in_ch: int = 1, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def soft_argmax_2d(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """logits: (B, 1, H, W) -> (B, 2) fractional (x_idx, y_idx) in cell units."""
    b, _, h, w = logits.shape
    flat = (logits.view(b, -1) / temperature)
    probs = F.softmax(flat, dim=1).view(b, h, w)
    ys = torch.arange(h, device=logits.device, dtype=logits.dtype)
    xs = torch.arange(w, device=logits.device, dtype=logits.dtype)
    ex = (probs.sum(dim=1) * xs).sum(dim=1)
    ey = (probs.sum(dim=2) * ys).sum(dim=1)
    return torch.stack([ex, ey], dim=1), probs


class SubpixelHead(nn.Module):
    """Regresses a bounded (x, y) correction, in cell units, from the local
    window of response-map logits around the soft-argmax peak. Soft-argmax
    already gives a continuous coordinate, but it's a weighted average over
    a coarse (16px-stride) grid -- this looks at the raw local response
    shape directly, the way peak-neighborhood subpixel interpolation does
    in classical correlation-based alignment.
    """

    def __init__(self, window: int = 5, hidden: int = 32, max_offset_cells: float = 1.5):
        super().__init__()
        self.window = window
        self.max_offset = max_offset_cells
        self.mlp = nn.Sequential(
            nn.Linear(window * window, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )

    def forward(self, logits: torch.Tensor, coarse_coords: torch.Tensor) -> torch.Tensor:
        b, _, h, w = logits.shape
        pad = self.window // 2
        padded = F.pad(logits.squeeze(1), (pad, pad, pad, pad), mode="replicate")  # (B, H+2p, W+2p)
        cx = coarse_coords[:, 0].round().long().clamp(0, w - 1) + pad
        cy = coarse_coords[:, 1].round().long().clamp(0, h - 1) + pad
        patches = torch.empty(b, self.window * self.window, device=logits.device, dtype=logits.dtype)
        for i in range(b):
            patch = padded[i, cy[i] - pad:cy[i] + pad + 1, cx[i] - pad:cx[i] + pad + 1]
            patches[i] = patch.reshape(-1)
        delta = torch.tanh(self.mlp(patches)) * self.max_offset
        return coarse_coords + delta


class SiameseLocalizer(nn.Module):
    def __init__(self, feat_ch: int = 128, temperature: float = 1.0, subpixel_window: int = 5):
        super().__init__()
        self.encoder = Encoder(in_ch=1, feat_ch=feat_ch)
        self.head = RefineHead()
        self.subpixel = SubpixelHead(window=subpixel_window)
        self.temperature = temperature

    def response_map(self, exemplar: torch.Tensor, search: torch.Tensor) -> torch.Tensor:
        ef = self.encoder(exemplar)
        sf = self.encoder(search)
        corr = xcorr_depthwise(sf, ef)
        return self.head(corr)

    def forward(self, exemplar: torch.Tensor, search: torch.Tensor):
        """Returns (heatmap_logits, refined_coords_cellspace, probs).
        `refined_coords` is the soft-argmax coordinate plus the sub-pixel
        head's learned correction -- this is the coordinate that should be
        used everywhere (loss, inference), since it's what the model will
        actually predict at deployment.
        """
        logits = self.response_map(exemplar, search)
        coarse_coords, probs = soft_argmax_2d(logits, self.temperature)
        refined_coords = self.subpixel(logits, coarse_coords)
        return logits, refined_coords, probs

    def cellspace_to_pixels(self, coords: torch.Tensor, exemplar_feat_size: int,
                             stride: int = ENCODER_STRIDE) -> torch.Tensor:
        """(B,2) cell coords -> (B,2) pixel coords in the *search input*
        (SEARCH_INPUT-sized) space, i.e. the receptive-field center of
        each correlation output cell.
        """
        offset = (exemplar_feat_size * stride) / 2.0
        return coords * stride + offset

    @staticmethod
    def pixels_to_cellspace(pixels: torch.Tensor, exemplar_feat_size: int,
                             stride: int = ENCODER_STRIDE) -> torch.Tensor:
        """Inverse of cellspace_to_pixels -- used to build training targets
        directly in the same fractional cell-index space the soft-argmax
        head predicts in.
        """
        offset = (exemplar_feat_size * stride) / 2.0
        return (pixels - offset) / stride


@dataclass
class LocalizationResult:
    x: float
    y: float
    coarse_x: float
    coarse_y: float
    model_x: float  # prediction before the final local-NCC snap
    model_y: float
    coarse_heatmap: torch.Tensor  # for visualization / debugging


def _local_ncc_snap(reference_np, search_np, center_xy, mag_ratio: float, window: int = 60):
    """Classical, untrained local refinement: search only a small window
    around the model's own prediction with normalized cross-correlation.
    This is cheap and doesn't hit the whole-image periodicity-tie problem
    (see docs/CITATIONS.md) because the model has already narrowed down the
    right neighborhood -- NCC just needs to find the exact pixel within it,
    which is exactly the regime where NCC's exhaustive per-pixel search
    beats the model's coarse response grid (see README "Current results").
    """
    from skimage.feature import match_template
    from skimage.transform import resize

    h, w = search_np.shape
    cx, cy = center_xy
    template_size = max(8, round(reference_np.shape[0] / mag_ratio))
    if template_size >= 2 * window:
        return center_xy
    x0 = int(min(max(cx - window, 0), max(w - 2 * window, 0)))
    y0 = int(min(max(cy - window, 0), max(h - 2 * window, 0)))
    crop = search_np[y0:y0 + 2 * window, x0:x0 + 2 * window]
    if crop.shape[0] <= template_size or crop.shape[1] <= template_size:
        return center_xy
    template = resize(reference_np, (template_size, template_size), anti_aliasing=True)
    result = match_template(crop, template, pad_input=True)
    idx = np.unravel_index(np.argmax(result), result.shape)
    return float(x0 + idx[1]), float(y0 + idx[0])


@torch.no_grad()
def localize(model: SiameseLocalizer, reference_full: torch.Tensor, search_full: torch.Tensor, *,
             mag_ratio: float = 10.0, refine_window: int = 220, ncc_snap: bool = True,
             ncc_window: int = 60, device=None) -> LocalizationResult:
    """reference_full, search_full: (1, 1, 1000, 1000) tensors in [0, 1],
    original problem-statement resolution. Returns full-resolution (x, y)
    in the original search image's pixel coordinates.
    """
    device = device or next(model.parameters()).device
    reference_full = reference_full.to(device)
    search_full = search_full.to(device)
    _, _, ref_h, ref_w = reference_full.shape
    _, _, search_h, search_w = search_full.shape

    # ---- coarse pass: whole search image ----
    exemplar = F.interpolate(reference_full, size=(EXEMPLAR_INPUT, EXEMPLAR_INPUT),
                              mode="bilinear", align_corners=False)
    search_in = F.interpolate(search_full, size=(SEARCH_INPUT, SEARCH_INPUT),
                               mode="bilinear", align_corners=False)
    logits, coords, _ = model(exemplar, search_in)
    ef_size = EXEMPLAR_INPUT // ENCODER_STRIDE
    px = model.cellspace_to_pixels(coords, ef_size)[0]  # in SEARCH_INPUT space
    scale_x = search_w / SEARCH_INPUT
    scale_y = search_h / SEARCH_INPUT
    coarse_x = float(px[0]) * scale_x
    coarse_y = float(px[1]) * scale_y

    # ---- fine pass: crop a window around the coarse peak at native res ----
    half = refine_window // 2
    cx0 = int(min(max(coarse_x - half, 0), search_w - refine_window))
    cy0 = int(min(max(coarse_y - half, 0), search_h - refine_window))
    crop = search_full[:, :, cy0:cy0 + refine_window, cx0:cx0 + refine_window]
    crop_in = F.interpolate(crop, size=(SEARCH_INPUT, SEARCH_INPUT), mode="bilinear", align_corners=False)
    logits2, coords2, _ = model(exemplar, crop_in)
    px2 = model.cellspace_to_pixels(coords2, ef_size)[0]
    crop_scale = refine_window / SEARCH_INPUT
    model_x = cx0 + float(px2[0]) * crop_scale
    model_y = cy0 + float(px2[1]) * crop_scale

    final_x, final_y = model_x, model_y
    if ncc_snap:
        ref_np = reference_full[0, 0].detach().cpu().numpy()
        search_np = search_full[0, 0].detach().cpu().numpy()
        final_x, final_y = _local_ncc_snap(ref_np, search_np, (model_x, model_y),
                                            mag_ratio=mag_ratio, window=ncc_window)

    return LocalizationResult(x=final_x, y=final_y, coarse_x=coarse_x, coarse_y=coarse_y,
                               model_x=model_x, model_y=model_y,
                               coarse_heatmap=logits.detach().cpu())
