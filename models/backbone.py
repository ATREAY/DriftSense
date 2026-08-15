"""Lightweight shared CNN encoder for the reference/search Siamese pair.

A small 4-stage strided-conv stack (AlexNet/SiamFC-lite style) rather than
a full ResNet: template matching needs translation-equivariant local
features, not deep semantic abstraction, and a small backbone keeps
inference on a 1000x1000 search image fast on a single GPU.

Reference:
  - L. Bertinetto, J. Valmadre, J. F. Henriques, A. Vedaldi, P. H. S. Torr,
    "Fully-Convolutional Siamese Networks for Object Tracking", ECCV 2016
    Workshops -- the shared-encoder + cross-correlation design this model
    follows (SiamFC).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(cin, cout, k, stride, pad, pool=False):
    layers = [
        nn.Conv2d(cin, cout, k, stride=stride, padding=pad, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """Stride-16 encoder. Input HxW -> feature (H/16)x(W/16)."""

    def __init__(self, in_ch: int = 1, feat_ch: int = 128):
        super().__init__()
        self.stage1 = _conv_block(in_ch, 32, k=5, stride=2, pad=2)        # /2
        self.stage2 = _conv_block(32, 64, k=3, stride=1, pad=1, pool=True)  # /4
        self.stage3 = _conv_block(64, 96, k=3, stride=1, pad=1, pool=True)  # /8
        self.stage4 = _conv_block(96, feat_ch, k=3, stride=1, pad=1, pool=True)  # /16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x
