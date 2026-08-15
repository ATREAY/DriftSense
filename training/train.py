#!/usr/bin/env python
"""Train the SiameseLocalizer on a generated DriftSense dataset.

Usage:
  python training/train.py --config configs/dram_finfet.yaml
  python training/train.py --config configs/dram_finfet.yaml --resume experiments/dram_finfet_v4/last.pt

GPU selection is handled entirely outside this script (see
training/select_gpu.sh + training/slurm_train.sh, which pick A100 > V100 >
P100 by live idle-GPU availability and set CUDA_VISIBLE_DEVICES before
this process starts) -- here we just take whatever `cuda` device Slurm
handed us, falling back to CPU if none.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.siamese_localizer import SiameseLocalizer
from training.dataset import DriftSenseDataset
from training.losses import localization_loss


def run_epoch(model, loader, device, optimizer, loss_cfg, train: bool):
    model.train(train)
    total_loss = total_hm = total_coord = 0.0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            exemplar = batch["exemplar"].to(device)
            search = batch["search"].to(device)
            target = batch["target_cell"].to(device)

            logits, coords, _ = model(exemplar, search)
            losses = localization_loss(logits, coords, target, **loss_cfg)

            if train:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()

            bs = exemplar.shape[0]
            total_loss += losses["total"].item() * bs
            total_hm += losses["heatmap"].item() * bs
            total_coord += losses["coord"].item() * bs
            n += bs
    return {"loss": total_loss / n, "heatmap": total_hm / n, "coord": total_coord / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None,
                     help="Checkpoint to continue training from (model + optimizer state). "
                          "--config's `epochs` is the number of *additional* epochs to run.")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train_ds = DriftSenseDataset(cfg["train_dir"], refine_prob=cfg.get("refine_prob", 0.5))
    val_ds = DriftSenseDataset(cfg["val_dir"], refine_prob=cfg.get("refine_prob", 0.5))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg.get("num_workers", 4), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg.get("num_workers", 4))

    model = SiameseLocalizer(feat_ch=cfg.get("feat_ch", 128),
                              temperature=cfg.get("temperature", 1.0)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    start_epoch = 0
    history = []
    best_val = float("inf")
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_val = ckpt.get("best_val", float("inf"))
        # Old checkpoints (pre-resume-support) don't carry best_val; fall
        # back to scanning this run's own history.json so a resume never
        # silently overwrites a genuinely-better best.pt with a worse one.
        hist_path = Path(cfg["output_dir"]) / "history.json"
        if best_val == float("inf") and hist_path.exists():
            prior_vals = [h["val"]["loss"] for h in json.loads(hist_path.read_text())]
            if prior_vals:
                best_val = min(prior_vals)
        print(f"Resumed from {args.resume} at epoch {start_epoch}, best_val={best_val:.4f}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    loss_cfg = dict(heatmap_sigma=cfg.get("heatmap_sigma", 1.0),
                     coord_weight=cfg.get("coord_weight", 1.0),
                     heatmap_weight=cfg.get("heatmap_weight", 1.0))

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, device, optimizer, loss_cfg, train=True)
        val_metrics = run_epoch(model, val_loader, device, optimizer, loss_cfg, train=False)
        scheduler.step()
        dt = time.time() - t0
        global_epoch = start_epoch + epoch + 1
        print(f"epoch {global_epoch} (+{epoch + 1}/{cfg['epochs']}) ({dt:.1f}s) "
              f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
              f"train_coord={train_metrics['coord']:.4f} val_coord={val_metrics['coord']:.4f}")
        history.append({"epoch": global_epoch, "train": train_metrics, "val": val_metrics, "seconds": dt})

        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "cfg": cfg, "epoch": global_epoch, "best_val": best_val}
        torch.save(ckpt, out_dir / "last.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            ckpt["best_val"] = best_val
            torch.save(ckpt, out_dir / "best.pt")

    history_path = out_dir / "history.json"
    prior = json.loads(history_path.read_text()) if history_path.exists() else []
    history_path.write_text(json.dumps(prior + history, indent=2))
    print(f"Done. Best val loss: {best_val:.4f}. Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
