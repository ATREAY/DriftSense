"""Full localization model: shared encoder -> depthwise cross-correlation
-> small refinement head -> heatmap -> soft-argmax coordinate.

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
crop taken from the *original* search image around that peak, giving a
sub-pixel-ish refinement without ever having to run the encoder densely
over the full-resolution 1000x1000 image.

Reference:
  - L. Bertinetto et al., "Fully-Convolutional Siamese Networks for Object
    Tracking", ECCV 2016 Workshops (SiamFC) -- template/search input-size
    convention and cross-correlation response-map design this follows.
"""
from __future__ import annotations

from dataclasses import dataclass

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


class SiameseLocalizer(nn.Module):
    def __init__(self, feat_ch: int = 128, temperature: float = 1.0):
        super().__init__()
        self.encoder = Encoder(in_ch=1, feat_ch=feat_ch)
        self.head = RefineHead()
        self.temperature = temperature

    def response_map(self, exemplar: torch.Tensor, search: torch.Tensor) -> torch.Tensor:
        ef = self.encoder(exemplar)
        sf = self.encoder(search)
        corr = xcorr_depthwise(sf, ef)
        return self.head(corr)

    def forward(self, exemplar: torch.Tensor, search: torch.Tensor):
        """Returns (heatmap_logits, coords_cellspace, probs)."""
        logits = self.response_map(exemplar, search)
        coords, probs = soft_argmax_2d(logits, self.temperature)
        return logits, coords, probs

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
    coarse_heatmap: torch.Tensor  # for visualization / debugging


@torch.no_grad()
def localize(model: SiameseLocalizer, reference_full: torch.Tensor, search_full: torch.Tensor, *,
             mag_ratio: int = 10, refine_window: int = 220, device=None) -> LocalizationResult:
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
    fine_x = cx0 + float(px2[0]) * crop_scale
    fine_y = cy0 + float(px2[1]) * crop_scale

    return LocalizationResult(x=fine_x, y=fine_y, coarse_x=coarse_x, coarse_y=coarse_y,
                               coarse_heatmap=logits.detach().cpu())
