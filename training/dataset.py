"""PyTorch Dataset over a generator/generate_dataset.py manifest.

Each __getitem__ randomly picks one of two training regimes so the shared
encoder sees both usage patterns it will face at inference time (see
models/siamese_localizer.py:localize):
  - "coarse": full 1000x1000 search image downsampled to network input.
  - "refine": a native-resolution crop around the true target, offset by a
    random jitter (simulating "the coarse pass landed nearby but not
    exactly on target"), upsampled to network input.

Images are decoded once at dataset construction and cached in RAM as
uint8 arrays (~1MB/image; a 5000-pair train split is ~10GB, well within a
single Slurm job's memory request). This matters specifically on shared
multi-tenant DGX nodes on this cluster (e.g. dgx-a100-02), which have no
per-job CPU isolation -- a plain per-epoch disk-read + PNG-decode
DataLoader gets starved badly enough there to make a "faster" GPU tier
much slower wall-clock than an uncontended one (same fix used in the
sibling WaferRestore project's README, Incident 2). The cache is built
before DataLoader worker processes fork, so workers share it via
copy-on-write with no extra memory cost per worker.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.siamese_localizer import EXEMPLAR_INPUT, SEARCH_INPUT, ENCODER_STRIDE, SiameseLocalizer


def _load_uint8(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


class DriftSenseDataset(Dataset):
    def __init__(self, data_dir: str | Path, refine_prob: float = 0.5,
                 refine_window_range=(150, 300), refine_jitter_frac: float = 0.35,
                 cache_in_ram: bool = True):
        self.data_dir = Path(data_dir)
        manifest = json.loads((self.data_dir / "manifest.json").read_text())
        self.pairs = manifest["pairs"]
        self.search_size = manifest["search_size"]
        self.refine_prob = refine_prob
        self.refine_window_range = refine_window_range
        self.refine_jitter_frac = refine_jitter_frac
        self.ef_size = EXEMPLAR_INPUT // ENCODER_STRIDE

        self._cache: dict[str, np.ndarray] | None = None
        if cache_in_ram:
            self._cache = {}
            for rec in self.pairs:
                for key in ("reference_path", "search_path"):
                    p = str(self.data_dir / rec[key])
                    self._cache[p] = _load_uint8(self.data_dir / rec[key])

    def _load_gray(self, path: Path) -> torch.Tensor:
        if self._cache is not None:
            arr = self._cache[str(path)]
        else:
            arr = _load_uint8(path)
        return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)  # (1, H, W)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rec = self.pairs[idx]
        ref = self._load_gray(self.data_dir / rec["reference_path"])
        search = self._load_gray(self.data_dir / rec["search_path"])
        gx, gy = rec["gt_x"], rec["gt_y"]

        exemplar = resize(ref, [EXEMPLAR_INPUT, EXEMPLAR_INPUT], antialias=True)

        use_refine = random.random() < self.refine_prob
        if use_refine:
            win = random.randint(*self.refine_window_range)
            jitter = win * self.refine_jitter_frac
            cx = gx + random.uniform(-jitter, jitter)
            cy = gy + random.uniform(-jitter, jitter)
            x0 = int(min(max(cx - win / 2, 0), self.search_size - win))
            y0 = int(min(max(cy - win / 2, 0), self.search_size - win))
            crop = search[:, y0:y0 + win, x0:x0 + win]
            search_in = resize(crop, [SEARCH_INPUT, SEARCH_INPUT], antialias=True)
            target_px = (gx - x0) * (SEARCH_INPUT / win)
            target_py = (gy - y0) * (SEARCH_INPUT / win)
        else:
            search_in = resize(search, [SEARCH_INPUT, SEARCH_INPUT], antialias=True)
            scale = SEARCH_INPUT / self.search_size
            target_px = gx * scale
            target_py = gy * scale

        target_pixels = torch.tensor([target_px, target_py], dtype=torch.float32)
        target_cell = SiameseLocalizer.pixels_to_cellspace(target_pixels, self.ef_size)

        return {
            "exemplar": exemplar,
            "search": search_in,
            "target_cell": target_cell,
            "regime": "refine" if use_refine else "coarse",
        }
