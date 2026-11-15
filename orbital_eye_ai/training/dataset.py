"""Duke DME OCT dataset loader (Chiu et al., Biomed. Opt. Express 2015).

Expected per-subject MATLAB file containing (verify after download!):
  images          uint8  (H, W, N)   B-scans (Spectralis, 496 x 768)
  manualLayers1   float  (8, W, N)   boundary row positions by expert 1, NaN = not annotated
  manualLayers2   float  (8, W, N)   expert 2
Only some B-scans and a central range of columns are annotated; this loader
keeps exactly those. Boundaries are converted to a label mask:
  0 = above boundary 1 (vitreous), k = between boundary k and k+1, 8 = below boundary 8.

Run `python training/dataset.py --inspect <data_dir>` first to print what each file
really contains before training.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from typing import List, Tuple

import numpy as np


def load_mat(path: str) -> dict:
    try:
        from scipy.io import loadmat
        return loadmat(path)
    except NotImplementedError:  # MATLAB v7.3 files are HDF5
        import h5py
        with h5py.File(path, "r") as f:
            return {k: np.array(f[k]).T for k in f.keys()}


def _layers_array(mat: dict, annotator: int, n_images: int, width: int) -> np.ndarray:
    arr = np.asarray(mat[f"manualLayers{annotator}"], dtype=np.float32)
    # normalise orientation to (8, W, N)
    axes = list(arr.shape)
    b_ax = axes.index(8)
    w_ax = [i for i, s in enumerate(axes) if s == width and i != b_ax][0]
    n_ax = [i for i in range(3) if i not in (b_ax, w_ax)][0]
    if axes[n_ax] != n_images:
        raise ValueError(f"Unexpected layer array shape {arr.shape}")
    return np.transpose(arr, (b_ax, w_ax, n_ax))


def boundaries_to_mask(bounds: np.ndarray, height: int) -> np.ndarray:
    """bounds: (8, W) row positions -> (H, W) uint8 label mask."""
    n_b, width = bounds.shape
    rows = np.arange(height)[:, None]
    mask = np.zeros((height, width), dtype=np.uint8)
    for k in range(n_b):
        mask += (rows >= np.round(bounds[k])[None, :]).astype(np.uint8)
    return mask


def subject_number(path: str) -> int:
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def load_subject(path: str, annotator: int = 1) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return [(image float32 [0,1] (H, w), mask uint8 (H, w))] for annotated B-scans, cropped to annotated columns."""
    mat = load_mat(path)
    images = np.asarray(mat["images"])
    h, w, n = images.shape
    layers = _layers_array(mat, annotator, n, w)
    samples = []
    for i in range(n):
        b = layers[:, :, i]
        cols = np.all(~np.isnan(b), axis=0)
        if cols.sum() < 32:
            continue  # this B-scan is not annotated
        idx = np.nonzero(cols)[0]
        lo, hi = idx.min(), idx.max() + 1
        b = b[:, lo:hi]
        b = np.maximum.accumulate(b, axis=0)  # enforce boundary order top->bottom
        img = images[:, lo:hi, i].astype(np.float32) / 255.0
        samples.append((img, boundaries_to_mask(b, h)))
    return samples


def load_split(data_dir: str, subjects: List[int], annotator: int = 1):
    files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
    chosen = [f for f in files if subject_number(f) in subjects]
    if not chosen:
        raise FileNotFoundError(f"No .mat files for subjects {subjects} in {data_dir}")
    data = []
    for f in chosen:
        data.extend(load_subject(f, annotator))
    return data


class OCTSegDataset:
    """PyTorch Dataset: resizes each sample to (H_in, W_in); light augmentation for training."""

    def __init__(self, samples, height: int, width: int, augment: bool = False, seed: int = 0):
        self.samples, self.h, self.w, self.augment = samples, height, width, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        import torch
        from PIL import Image

        img, mask = self.samples[i]
        img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((self.w, self.h), Image.BILINEAR),
                         dtype=np.float32) / 255.0
        mask = np.asarray(Image.fromarray(mask).resize((self.w, self.h), Image.NEAREST), dtype=np.int64)
        if self.augment:
            if self.rng.random() < 0.5:                       # left-right flip (anatomically plausible for B-scans)
                img, mask = img[:, ::-1], mask[:, ::-1]
            gain, bias = self.rng.uniform(0.8, 1.2), self.rng.uniform(-0.08, 0.08)
            img = np.clip(img * gain + bias, 0, 1)            # brightness / contrast
            shift = int(self.rng.integers(-20, 21))           # small vertical translation
            img, mask = np.roll(img, shift, 0), np.roll(mask, shift, 0)
            if shift > 0:
                img[:shift], mask[:shift] = 0, 0
            elif shift < 0:
                img[shift:], mask[shift:] = 0, mask.max()
            img = np.clip(img + self.rng.normal(0, 0.02, img.shape), 0, 1)  # mild noise
        return torch.from_numpy(np.ascontiguousarray(img))[None].float(), torch.from_numpy(np.ascontiguousarray(mask))


def inspect(data_dir: str):
    for f in sorted(glob.glob(os.path.join(data_dir, "*.mat"))):
        mat = load_mat(f)
        keys = {k: getattr(mat[k], "shape", None) for k in mat if not k.startswith("__")}
        n_ann = len(load_subject(f, 1)) if "manualLayers1" in mat else 0
        print(os.path.basename(f), keys, "annotated B-scans:", n_ann)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inspect", required=True, help="folder with .mat files")
    inspect(p.parse_args().inspect)
