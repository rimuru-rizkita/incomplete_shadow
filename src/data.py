"""
data.py
=======
DataModule tunggal untuk L7 Irish dan L8 Biome, membaca setiap scene di CRS ASLINYA (30 m).

Kenapa tidak lewat TorchGeo (versi sebelum 2026-11-02):
  TorchGeo 0.8.1 menyatukan semua scene ke satu CRS (EPSG:3857) dengan resolusi file pertama
  (105 m untuk L8 Biome, 52 m untuk L7 Irish). Web Mercator tidak mempertahankan luas, sehingga
  scene tropis dilatih/diuji di ~105 m (3,5x lebih kasar), scene Antartika di-upsample, dan
  bobot piksel antar scene berbeda sampai ~60x (3 scene Antartika = 73% piksel test L8).
  Di sini setiap patch dibaca langsung dari raster aslinya, sehingga:
    - semua patch 30 m, piksel berbobot sebanding luas sebenarnya;
    - patch tidak pernah melintasi dua scene (identitas scene ada di key patch).

Patch diidentifikasi dengan key (scene_id, row, col): posisi pojok kiri atas dalam piksel raster
scene. Bagian patch di luar raster dibaca sebagai Fill (0). Semua sampler, index patch (fase 0),
dan daftar patch val/test memakai key ini.

Perbedaan penting lain dari versi lama:
  1. Validasi per-epoch memakai daftar patch yang di-materialize SEKALI dengan seed tetap.
  2. Test grid di-materialize sebagai list (confusion matrix per scene / bootstrap).
  3. Split per lokasi WRS-2 dan DISTRATIFIKASI per grup dataset (biome L8 / zona lintang L7),
     sehingga setiap biome terwakili di train/val/test.
"""

import os
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path

_PROJ_GUARD = "_CLOUDSHADOW_PROJ_FIX_DONE"


def _fix_proj_data_path():
    """Fix konflik proj.db di Windows (dipertahankan dari kode lama)."""
    if os.environ.get(_PROJ_GUARD) == "1":
        return
    candidates = []
    try:
        import rasterio
        rdir = os.path.dirname(rasterio.__file__)
        for sub in ["proj_data", os.path.join("gdal_data", "..", "proj_data")]:
            c = os.path.join(rdir, sub)
            if os.path.isdir(c):
                candidates.append(c)
    except ImportError:
        pass
    try:
        import pyproj
        candidates.append(pyproj.datadir.get_data_dir())
    except ImportError:
        pass
    for path in candidates:
        if os.path.isfile(os.path.join(path, "proj.db")):
            os.environ["PROJ_DATA"] = path
            os.environ["PROJ_LIB"] = path
            break
    os.environ[_PROJ_GUARD] = "1"


_fix_proj_data_path()

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402
from rasterio.warp import transform_bounds  # noqa: E402
from rasterio.windows import Window  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from . import config as C  # noqa: E402
from .geo_compat import location_key  # noqa: E402
from .utils import get_mask  # noqa: E402

# nilai mask mentah -> kelas (sama dengan ordinal_map TorchGeo L7IrishMask / L8BiomeMask)
RAW_CLASSES = {64: C.SHADOW, 128: C.CLEAR, 192: C.THIN, 255: C.CLOUD}

# file citra per dataset (regex = filename_regex TorchGeo) dan mask pasangannya
_IMAGE_RE = {
    "l7irish": re.compile(r"^L71\d{6}_\d{11}\.TIF$"),
    "l8biome": re.compile(r"^LC8\d{13}[A-Z]{3}\d{2}\.TIF$"),
}


def _mask_path(dataset: str, image: Path) -> Path:
    if dataset == "l8biome":
        return image.with_name(f"{image.stem}_fixedmask.TIF")
    hits = sorted(image.parent.glob("L7_p*_r*_newmask2015.TIF"))
    if len(hits) != 1:
        raise FileNotFoundError(f"mask L7 untuk {image}: {len(hits)} kandidat")
    return hits[0]


@dataclass
class Scene:
    scene_id: str      # "<folder>/<file>", format sama dengan split versi sebelumnya
    path: str          # citra
    mask_path: str
    group: str         # biome (L8) / zona lintang (L7) = folder induk
    location: str      # WRS-2 path/row
    height: int
    width: int
    # footprint dalam derajat (EPSG:4326), untuk cek overlap antar split
    minx: float
    maxx: float
    miny: float
    maxy: float


def dataset_root(dataset: str, partial: str = "none") -> Path:
    return C.partial_root(dataset, partial) if partial != "none" else Path(C.DATASETS[dataset]["root"])


def discover_scenes(dataset: str, partial: str = "none") -> list:
    root = dataset_root(dataset, partial)
    if not root.exists():
        hint = "scripts/make_partial_masks.py" if partial != "none" else "unduh dataset dulu"
        raise FileNotFoundError(f"{root} belum ada ({hint})")
    scenes = []
    for img in sorted(root.rglob("*.TIF")):
        if not _IMAGE_RE[dataset].match(img.name):
            continue
        msk = _mask_path(dataset, img)
        with rasterio.open(img) as a, rasterio.open(msk) as m:
            if (a.height, a.width) != (m.height, m.width) or a.transform != m.transform:
                raise ValueError(f"grid citra dan mask berbeda: {img}")
            lon0, lat0, lon1, lat1 = transform_bounds(a.crs, "EPSG:4326", *a.bounds)
            h, w = a.height, a.width
        scenes.append(Scene(f"{img.parent.name}/{img.name}", str(img), str(msk),
                            img.parent.parent.name, location_key(str(img)), h, w,
                            lon0, lon1, lat0, lat1))
    if not scenes:
        raise FileNotFoundError(f"tidak ada citra {dataset} di {root}")
    return scenes


def location_split(scenes: list, fractions, seed: int) -> dict:
    """
    Split per lokasi WRS-2 (path/row), distratifikasi per grup (biome / zona lintang).
    Di tiap grup, lokasi diacak (seed tetap) lalu diisikan ke test -> val sampai jumlah scene
    mencapai round(fraksi x jumlah scene grup); sisanya train. Satu lokasi utuh selalu masuk
    satu split.
    """
    rng = random.Random(seed)
    by_group = {}
    for s in scenes:
        by_group.setdefault(s.group, {}).setdefault(s.location, []).append(s.scene_id)
    out = {"train": [], "val": [], "test": []}
    for group in sorted(by_group):
        locs = by_group[group]
        names = sorted(locs)
        rng.shuffle(names)
        n = sum(len(v) for v in locs.values())
        targets = {"test": round(fractions[2] * n), "val": round(fractions[1] * n)}
        filled = {"test": 0, "val": 0}
        for loc in names:
            split = next((k for k in ("test", "val") if filled[k] < targets[k]), "train")
            out[split].extend(locs[loc])
            if split != "train":
                filled[split] += len(locs[loc])
    return {k: sorted(v) for k, v in out.items()}


class NativePatchDataset(Dataset):
    """Patch 2D dari daftar scene; index = key (scene_id, row, col)."""

    def __init__(self, scenes: list, patch_size: int, partial: bool = False):
        self.scenes = {s.scene_id: s for s in scenes}
        self.patch_size = patch_size
        lut = torch.zeros(256, dtype=torch.long)
        for raw, cls in RAW_CLASSES.items():
            lut[raw] = cls
        if partial:
            lut[C.AMBIG_RAW] = C.AMBIG
        self.lut = lut
        self._handles, self._pid = {}, None

    def __len__(self):
        return len(self.scenes)

    def _open(self, path: str):
        if self._pid != os.getpid():  # handle rasterio tidak boleh dibagi antar proses worker
            self._handles, self._pid = {}, os.getpid()
        if path not in self._handles:
            self._handles[path] = rasterio.open(path)
        return self._handles[path]

    def _read(self, path: str, row: int, col: int) -> np.ndarray:
        src, ps = self._open(path), self.patch_size
        out = np.zeros((src.count, ps, ps), dtype=src.dtypes[0])
        r0, c0 = max(row, 0), max(col, 0)
        r1, c1 = min(row + ps, src.height), min(col + ps, src.width)
        if r1 > r0 and c1 > c0:
            out[:, r0 - row:r1 - row, c0 - col:c1 - col] = src.read(
                window=Window(c0, r0, c1 - c0, r1 - r0))
        return out

    def __getitem__(self, key) -> dict:
        sid, row, col = key
        s = self.scenes[sid]
        image = torch.from_numpy(self._read(s.path, row, col))
        mask = self.lut[torch.from_numpy(self._read(s.mask_path, row, col)[0]).long()]
        return {"image": image, "mask": mask}

    # ------------------------------------------------------------------ key
    def grid_keys(self, stride: int | None = None) -> list:
        """Grid tanpa overlap yang menutup seluruh raster (tepi kanan/bawah dipad Fill)."""
        st = stride or self.patch_size
        keys = []
        for sid, s in self.scenes.items():
            for r in range(0, s.height, st):
                for c in range(0, s.width, st):
                    keys.append((sid, r, c))
        return keys

    def random_key(self, rng: random.Random):
        """Scene dipilih sebanding luas raster, posisi uniform; patch selalu di dalam raster."""
        ids = list(self.scenes)
        weights = [self.scenes[i].height * self.scenes[i].width for i in ids]
        sid = rng.choices(ids, weights=weights, k=1)[0]
        s, ps = self.scenes[sid], self.patch_size
        return (sid, rng.randint(0, max(s.height - ps, 0)), rng.randint(0, max(s.width - ps, 0)))


def _collate(samples: list) -> dict:
    return {"image": torch.stack([s["image"] for s in samples]),
            "mask": torch.stack([s["mask"] for s in samples])}


class SceneLookup:
    """Scene asal sebuah patch; dengan key native cukup baca elemen pertama."""

    def __init__(self, scenes: list):
        self.scenes = scenes

    @staticmethod
    def find(key) -> str:
        return key[0]


class CloudDataModule:
    def __init__(
        self,
        dataset: str,
        patch_size: int = 256,
        batch_size: int = 8,
        num_workers: int = 4,
        download: bool = True,  # dipertahankan untuk kompatibilitas; dataset harus sudah ada
        partial: str = "none",
    ):
        if dataset not in C.DATASETS:
            raise ValueError(f"Dataset tidak dikenal: {dataset}")
        self.dataset = dataset
        self.partial = partial
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    # ------------------------------------------------------------------
    def setup(self, verbose: bool = True):
        all_scenes = discover_scenes(self.dataset, self.partial)
        if C.SPLIT_MODE != "location":
            raise ValueError(f"SPLIT_MODE {C.SPLIT_MODE} tidak didukung loader native")
        ids = location_split(all_scenes, C.SPLIT_FRACTIONS, C.SPLIT_SEED)
        by_id = {s.scene_id: s for s in all_scenes}
        self.scenes = {k: [by_id[i] for i in v] for k, v in ids.items()}
        partial = self.partial != "none"
        self.train_view, self.val_view, self.test_view = (
            NativePatchDataset(self.scenes[k], self.patch_size, partial) for k in ("train", "val", "test"))
        if verbose:
            print(f"[{self.dataset}] scene: train={len(self.scenes['train'])}, "
                  f"val={len(self.scenes['val'])}, test={len(self.scenes['test'])} (CRS asli, 30 m)")
        return self

    def view(self, split: str) -> NativePatchDataset:
        return {"train": self.train_view, "val": self.val_view, "test": self.test_view}[split]

    # ------------------------------------------------------------------
    def _loader(self, view, sampler, num_workers=None):
        return DataLoader(
            view,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers if num_workers is None else num_workers,
            collate_fn=_collate,
            persistent_workers=False,
        )

    def train_loader(self, sampler):
        return self._loader(self.train_view, sampler)

    def grid_bboxes(self, split: str, stride: int | None = None) -> list:
        return self.view(split).grid_keys(stride)

    def fixed_val_bboxes(self, n_patches: int) -> list:
        """
        Daftar patch validasi IDENTIK untuk semua run/epoch (seed tetap), hanya patch yang
        berisi piksel valid. Di-cache di data/ supaya semua run memakai daftar yang sama persis.
        """
        cache = C.DATA_DIR / (f"{self.dataset}_val_keys_native_n{n_patches}"
                              f"_ps{self.patch_size}_seed{C.VAL_SEED}.pkl")
        if cache.exists():
            with open(cache, "rb") as f:
                return pickle.load(f)
        rng = random.Random(C.VAL_SEED)
        candidates = [self.val_view.random_key(rng) for _ in range(n_patches * 3)]
        chosen, offset = [], 0
        for batch in self.bbox_loader("val", candidates):
            valid = (get_mask(batch) != C.FILL).flatten(1).any(1).tolist()
            for j, ok in enumerate(valid):
                if ok and len(chosen) < n_patches:
                    chosen.append(candidates[offset + j])
            offset += len(valid)
            if len(chosen) >= n_patches:
                break
        if len(chosen) < n_patches:
            print(f"[WARNING] hanya {len(chosen)} patch validasi non-kosong (diminta {n_patches}).")
        print(f"[Val] {len(chosen)} patch validasi non-kosong dari {offset} kandidat -> {cache.name}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(chosen, f)
        return chosen

    def bbox_loader(self, split: str, bboxes: list, num_workers=None):
        """DataLoader untuk list key patch (urutan dijaga, dipakai untuk val & test)."""
        return self._loader(self.view(split), bboxes, num_workers)

    def scene_lookup(self, split: str) -> SceneLookup:
        return SceneLookup(self.scenes[split])
