"""Depthwise cross-correlation between an exemplar (template) feature map
and a larger search feature map, as in SiamFC / SiamRPN-style trackers.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def xcorr_depthwise(search_feat: torch.Tensor, exemplar_feat: torch.Tensor) -> torch.Tensor:
    """search_feat: (B, C, Hs, Ws); exemplar_feat: (B, C, He, We).
    Returns a single-channel response map (B, 1, Hs-He+1, Ws-We+1) per
    batch element, correlating each exemplar against its own search image
    (grouped conv trick avoids a python loop over the batch).
    """
    b, c, he, we = exemplar_feat.shape
    search_grouped = search_feat.view(1, b * c, search_feat.shape[2], search_feat.shape[3])
    kernel = exemplar_feat.reshape(b * c, 1, he, we)
    out = F.conv2d(search_grouped, kernel, groups=b * c)
    out = out.view(b, c, out.shape[2], out.shape[3])
    return out.mean(dim=1, keepdim=True)
