"""Heatmap + coordinate regression loss for the localization head.

Combines a per-cell Gaussian-target BCE (dense supervision, helps early
training when the soft-argmax expectation is still far from the target)
with a direct L1 on the soft-argmax coordinate (sharpens final precision).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_heatmap_target(coords: torch.Tensor, h: int, w: int, sigma: float = 1.0) -> torch.Tensor:
    """coords: (B, 2) fractional (x, y) cell-space target -> (B, h, w) Gaussian bump."""
    device = coords.device
    ys = torch.arange(h, device=device, dtype=coords.dtype).view(1, h, 1)
    xs = torch.arange(w, device=device, dtype=coords.dtype).view(1, 1, w)
    tx = coords[:, 0].view(-1, 1, 1)
    ty = coords[:, 1].view(-1, 1, 1)
    d2 = (xs - tx) ** 2 + (ys - ty) ** 2
    return torch.exp(-d2 / (2 * sigma ** 2))


def localization_loss(logits: torch.Tensor, pred_coords: torch.Tensor, target_coords: torch.Tensor, *,
                       heatmap_sigma: float = 1.0, coord_weight: float = 1.0,
                       heatmap_weight: float = 1.0) -> dict:
    """logits: (B,1,H,W) raw response map. pred_coords/target_coords: (B,2)
    in the same fractional cell-index space. Returns dict of loss terms.
    """
    b, _, h, w = logits.shape
    target_hm = gaussian_heatmap_target(target_coords, h, w, sigma=heatmap_sigma)
    hm_loss = F.binary_cross_entropy_with_logits(logits.view(b, h, w), target_hm)
    coord_loss = F.l1_loss(pred_coords, target_coords)
    total = heatmap_weight * hm_loss + coord_weight * coord_loss
    return {"total": total, "heatmap": hm_loss.detach(), "coord": coord_loss.detach()}
