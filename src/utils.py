"""
utils.py
========
Utilitas umum: seeding, RNG terisolasi, normalisasi input, augmentasi,
dan logging ke file.
"""

import json
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def fixed_rng(seed: int):
    """
    Jalankan blok kode dengan RNG (python, numpy, torch) ter-seed TETAP,
    lalu kembalikan state RNG sebelumnya. Dipakai untuk membuat daftar
    patch validasi yang identik di semua run & epoch, tanpa mengganggu
    RNG training.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


class Normalizer:
    """
    Normalisasi input per-band di GPU.
      mode="zscore" : (x - mean) / std, statistik dari scene TRAINING (fase 0)
      mode="div255" : x / 255 (konvensi data uint8; cek konvensi SSL4EO-L)
      mode="none"   : tanpa normalisasi (perilaku kode versi lama)
    """

    def __init__(self, mode: str, stats: dict | None, device: str):
        self.mode = mode
        if mode == "zscore":
            if stats is None or "band_mean" not in stats:
                raise FileNotFoundError(
                    "Statistik band belum ada. Jalankan scripts/phase0_scan_train.py dulu."
                )
            self.mean = torch.tensor(stats["band_mean"], device=device).view(1, -1, 1, 1)
            self.std = torch.tensor(stats["band_std"], device=device).view(1, -1, 1, 1).clamp_min(1e-6)
        elif mode not in ("div255", "none"):
            raise ValueError(f"Mode normalisasi tidak dikenal: {mode}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "zscore":
            return (x - self.mean) / self.std
        if self.mode == "div255":
            return x / 255.0
        return x


def augment_batch(images: torch.Tensor, masks: torch.Tensor):
    """
    Flip horizontal/vertikal + rotasi 90 derajat, diputuskan PER SAMPEL
    (versi lama per batch). images [B,C,H,W], masks [B,H,W] (harus 3D).
    """
    assert masks.dim() == 3, f"mask harus [B,H,W], dapat {tuple(masks.shape)}"
    out_img, out_msk = [], []
    for i in range(images.shape[0]):
        img, msk = images[i], masks[i]
        if torch.rand(1).item() < 0.5:
            img, msk = torch.flip(img, dims=[2]), torch.flip(msk, dims=[1])
        if torch.rand(1).item() < 0.5:
            img, msk = torch.flip(img, dims=[1]), torch.flip(msk, dims=[0])
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            img, msk = torch.rot90(img, k, dims=[1, 2]), torch.rot90(msk, k, dims=[0, 1])
        out_img.append(img)
        out_msk.append(msk)
    return torch.stack(out_img), torch.stack(out_msk)


def get_mask(batch) -> torch.Tensor:
    """Ambil mask [B,H,W] (long) dari batch TorchGeo."""
    m = batch["mask"]
    if m.dim() == 4:
        m = m.squeeze(1)
    return m.long()


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)  # tulis atomik: file tidak pernah setengah jadi


class Tee:
    """Duplikasi stdout ke file log (per run)."""

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *exc):
        sys.stdout = self.stdout
        self.file.close()
