"""
samplers.py
===========
Semua strategi sampling patch training.

Index patch (hasil scripts/phase0_scan_train.py) berisi SELURUH patch grid
dari scene TRAINING saja, masing-masing dengan jumlah piksel per kelas.
Karena itu definisi "patch ber-shadow" bisa diatur lewat ambang tau tanpa
scan ulang. Field "bbox" di record berisi key patch native (scene_id, row, col).

Strategi:
  random          : patch acak uniform (scene sebanding luas raster, posisi uniform;
                    padanan RandomGeoSampler di CRS asli)
  oversample      : location-aware oversampling (metode kita). Dengan peluang p
                    ambil patch dari pool shadow (shadow_frac >= tau) secara
                    UNIFORM, selain itu patch acak (random). Opsional jitter.
  ratio_weighted  : sama seperti oversample, tapi di dalam pool shadow patch
                    dipilih proporsional terhadap fraksi piksel shadow
                    (baseline "positive-pixel-ratio sampling").
  class_aware     : tiap draw pilih kelas (Shadow/Clear/Thin/Cloud) secara
                    uniform, lalu patch yang mengandung kelas itu
                    (baseline "class-aware sampling", Shen et al. 2016).
  nonempty        : patch uniform dari SEMUA patch grid yang berisi piksel valid
                    (bukan Fill seluruhnya), TANPA prioritas shadow. Kontrol untuk
                    memisahkan efek "menghindari Fill" dari efek "memprioritaskan
                    shadow" (patch acak tetap bisa jatuh di pojok Fill scene Landsat).
"""

import pickle
import random
from pathlib import Path

from . import config as C


def load_patch_index(path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Index patch tidak ditemukan: {p}. Jalankan scripts/phase0_scan_train.py dulu."
        )
    with open(p, "rb") as f:
        return pickle.load(f)


def shadow_frac(rec) -> float:
    return rec["counts"][C.SHADOW] / max(rec["valid"], 1)


def shadow_pool(records: list, tau: float) -> list:
    """
    tau = 0      -> patch dengan >= 1 piksel shadow (definisi versi lama)
    tau > 0      -> patch dengan fraksi shadow (atas piksel valid) >= tau
    """
    if tau <= 0:
        return [r for r in records if r["counts"][C.SHADOW] > 0]
    return [r for r in records if shadow_frac(r) >= tau]


class _BaseSampler:
    def __init__(self, view, size: int, length: int, seed: int, jitter: float = 0.0):
        self.view = view
        self.size = size
        self.length = length
        self.jitter = jitter
        # RNG milik sampler sendiri -> deterministik & independen dari RNG lain
        self.rng = random.Random(seed)

    def _maybe_jitter(self, key):
        """Geser patch acak maksimal `jitter` x lebar patch (0 = tanpa jitter)."""
        if self.jitter <= 0:
            return key
        sid, row, col = key
        d = int(round(self.jitter * self.size))
        return (sid, row + self.rng.randint(-d, d), col + self.rng.randint(-d, d))

    def _random_key(self):
        return self.view.random_key(self.rng)

    def __len__(self):
        return self.length


class RandomSampler(_BaseSampler):
    def __iter__(self):
        for _ in range(self.length):
            yield self._random_key()


class ShadowOversampler(_BaseSampler):
    """Mixture: p * Uniform(pool shadow) + (1-p) * Uniform(domain)."""

    def __init__(self, view, size, length, seed, records, p=0.6, tau=0.0,
                 jitter=0.0, weight_by_ratio=False):
        super().__init__(view, size, length, seed, jitter)
        self.p = p
        self.pool = shadow_pool(records, tau)
        if not self.pool:
            raise RuntimeError(f"Pool shadow kosong untuk tau={tau}. Turunkan tau.")
        self.weights = [shadow_frac(r) for r in self.pool] if weight_by_ratio else None
        mode = "ratio-weighted" if weight_by_ratio else "uniform"
        print(f"[Sampler] oversample ({mode}): pool={len(self.pool)} patch, "
              f"p={p}, tau={tau}, jitter={jitter}")

    def __iter__(self):
        for _ in range(self.length):
            if self.rng.random() < self.p:
                if self.weights is None:
                    rec = self.rng.choice(self.pool)
                else:
                    rec = self.rng.choices(self.pool, weights=self.weights, k=1)[0]
                yield self._maybe_jitter(rec["bbox"])
            else:
                yield self._random_key()


class ClassAwareSampler(_BaseSampler):
    """Pilih kelas uniform, lalu patch uniform yang mengandung kelas itu."""

    def __init__(self, view, size, length, seed, records, jitter=0.0):
        super().__init__(view, size, length, seed, jitter)
        self.pools = {c: [r for r in records if r["counts"][c] > 0] for c in C.EVAL_CLASSES}
        self.pools = {c: p for c, p in self.pools.items() if p}
        sizes = {C.CLASS_NAMES[c]: len(p) for c, p in self.pools.items()}
        print(f"[Sampler] class_aware: pool per kelas = {sizes}")

    def __iter__(self):
        classes = list(self.pools.keys())
        for _ in range(self.length):
            c = self.rng.choice(classes)
            yield self._maybe_jitter(self.rng.choice(self.pools[c])["bbox"])


class NonEmptySampler(_BaseSampler):
    """Uniform atas patch grid yang punya >= 1 piksel valid (tanpa prioritas kelas)."""

    def __init__(self, view, size, length, seed, records, jitter=0.0):
        super().__init__(view, size, length, seed, jitter)
        self.pool = [r for r in records if r["valid"] > 0]
        if not self.pool:
            raise RuntimeError("Tidak ada patch non-kosong di index.")
        print(f"[Sampler] nonempty: pool={len(self.pool)} dari {len(records)} patch grid")

    def __iter__(self):
        for _ in range(self.length):
            yield self._maybe_jitter(self.rng.choice(self.pool)["bbox"])


def build_sampler(cfg, view, records: list | None):
    """Bangun sampler sesuai RunConfig."""
    common = dict(view=view, size=cfg.patch_size, length=cfg.patches_per_epoch,
                  seed=cfg.seed * 1000 + 17)
    if cfg.sampling == "random":
        return RandomSampler(**common)
    if records is None:
        raise ValueError(f"Sampling '{cfg.sampling}' butuh index patch.")
    if cfg.sampling == "oversample":
        return ShadowOversampler(**common, records=records, p=cfg.p, tau=cfg.tau,
                                 jitter=cfg.jitter)
    if cfg.sampling == "ratio_weighted":
        return ShadowOversampler(**common, records=records, p=cfg.p, tau=cfg.tau,
                                 jitter=cfg.jitter, weight_by_ratio=True)
    if cfg.sampling == "class_aware":
        return ClassAwareSampler(**common, records=records, jitter=cfg.jitter)
    if cfg.sampling == "nonempty":
        return NonEmptySampler(**common, records=records, jitter=cfg.jitter)
    raise ValueError(f"Sampling tidak dikenal: {cfg.sampling}")
