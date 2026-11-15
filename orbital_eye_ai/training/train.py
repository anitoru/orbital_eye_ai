"""Train and evaluate the retinal-layer U-Net (run in Google Colab with a GPU).

    python training/train.py --data_dir /content/drive/MyDrive/orbital_eye/datasets/duke_dme \
                             --out_dir  /content/drive/MyDrive/orbital_eye/checkpoints

Outputs in out_dir:
    retinal_unet.pt        best checkpoint (by validation Dice) + held-out TEST metrics
    loss_curves.png        training / validation loss and validation Dice
    test_metrics.json      per-class Dice, IoU, precision, recall, boundary error (px)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TrainConfig  # noqa: E402
from dataset import OCTSegDataset, load_split  # noqa: E402
from orbital.ml import metrics  # noqa: E402
from orbital.ml.unet import UNet, count_parameters  # noqa: E402


def dice_loss(logits, target, num_classes, eps=1e-6):
    probs = logits.softmax(1)
    onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    inter = (probs * onehot).sum((0, 2, 3))
    union = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def evaluate(model, loader, cfg, device):
    model.eval()
    losses, scores = [], {"dice": [], "iou": [], "precision": [], "recall": [], "boundary_mae_px": []}
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses.append((F.cross_entropy(logits, y) + cfg.dice_weight * dice_loss(logits, y, cfg.num_classes)).item())
            pred = logits.argmax(1).cpu().numpy()
            for p, t in zip(pred, y.cpu().numpy()):
                s = metrics.per_class_scores(p, t, cfg.num_classes)
                for k in ("dice", "iou", "precision", "recall"):
                    scores[k].append(s[k])
                scores["boundary_mae_px"].append(metrics.boundary_mae(p, t, cfg.num_classes))

    def per_class_mean(rows):
        arr = np.array([[np.nan if v is None else v for v in r] for r in rows], dtype=float)
        return [None if np.all(np.isnan(c)) else round(float(np.nanmean(c)), 4) for c in arr.T]

    summary = {k: per_class_mean(v) for k, v in scores.items()}
    retinal = slice(1, cfg.num_classes - 1)  # layers only (exclude vitreous / below BM)
    summary["mean_dice_layers"] = round(float(np.nanmean([v for v in summary["dice"][retinal] if v is not None])), 4)
    summary["mean_iou_layers"] = round(float(np.nanmean([v for v in summary["iou"][retinal] if v is not None])), 4)
    summary["mean_boundary_mae_px"] = round(float(np.nanmean([v for v in summary["boundary_mae_px"] if v is not None])), 3)
    return float(np.mean(losses)), summary


def plot_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train_loss"], label="train"); ax[0].plot(history["val_loss"], label="validation")
    ax[0].set_title("Loss (CE + Dice)"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history["val_dice"], color="tab:green"); ax[1].set_title("Validation mean Dice (retinal layers)")
    ax[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    cfg = TrainConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=cfg.data_dir)
    p.add_argument("--out_dir", default=cfg.out_dir)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    args = p.parse_args()
    cfg.data_dir, cfg.out_dir, cfg.epochs, cfg.batch_size = args.data_dir, args.out_dir, args.epochs, args.batch_size
    os.makedirs(cfg.out_dir, exist_ok=True)

    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_s = load_split(cfg.data_dir, cfg.train_subjects, cfg.annotator)
    val_s = load_split(cfg.data_dir, cfg.val_subjects, cfg.annotator)
    test_s = load_split(cfg.data_dir, cfg.test_subjects, cfg.annotator)
    print(f"B-scans  train={len(train_s)}  val={len(val_s)}  test={len(test_s)} (patient-level split)")

    mk = lambda s, aug, shuffle: DataLoader(OCTSegDataset(s, cfg.input_height, cfg.input_width, aug, cfg.seed),
                                            batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers)
    train_dl, val_dl, test_dl = mk(train_s, True, True), mk(val_s, False, False), mk(test_s, False, False)

    model = UNet(1, cfg.num_classes, cfg.base, cfg.depth).to(device)
    print("Parameters:", count_parameters(model))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)

    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice, best_path = -1.0, os.path.join(cfg.out_dir, "retinal_unet.pt")
    for epoch in range(cfg.epochs):
        model.train(); t0 = time.time(); epoch_losses = []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y) + cfg.dice_weight * dice_loss(logits, y, cfg.num_classes)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_losses.append(loss.item())
        sched.step()
        val_loss, val = evaluate(model, val_dl, cfg, device)
        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val["mean_dice_layers"])
        print(f"epoch {epoch + 1:3d}  train {history['train_loss'][-1]:.4f}  val {val_loss:.4f}  "
              f"val Dice {val['mean_dice_layers']:.4f}  ({time.time() - t0:.0f}s)")
        if val["mean_dice_layers"] > best_dice:
            best_dice = val["mean_dice_layers"]
            torch.save({"model_state": model.state_dict(), "config": cfg.to_dict(), "best_val_dice": best_dice}, best_path)

    plot_curves(history, os.path.join(cfg.out_dir, "loss_curves.png"))

    # Final, single evaluation on the held-out TEST subjects with the best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, test = evaluate(model, test_dl, cfg, device)
    with open(os.path.join(cfg.out_dir, "test_metrics.json"), "w") as f:
        json.dump({"test": test, "history": history}, f, indent=2)

    ckpt.update({
        "version": cfg.version,
        "trained_on": f"Duke DME (Chiu et al. 2015), subjects {cfg.train_subjects}; test subjects {cfg.test_subjects}",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": {
            "test mean Dice (layers)": test["mean_dice_layers"],
            "test mean IoU (layers)": test["mean_iou_layers"],
            "test mean boundary error (px)": test["mean_boundary_mae_px"],
            "best validation Dice": round(best_dice, 4),
            "test B-scans": len(test_s),
        },
    })
    ckpt["config"] = {k: ckpt["config"][k] for k in ("num_classes", "input_height", "input_width", "base", "depth")}
    torch.save(ckpt, best_path)
    print("Saved", best_path)
    print("TEST:", json.dumps(ckpt["metrics"], indent=2))


if __name__ == "__main__":
    main()
