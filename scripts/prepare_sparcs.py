"""
prepare_sparcs.py
=================
Siapkan SPARCS (Hughes & Kennedy 2019; 80 subscene Landsat 8 1000x1000 px dengan label shadow
LENGKAP) sebagai test set eksternal untuk model L8 Biome.

Input  : dataset/sparcs/sparcs_data_L8.zip (https://emapr.ceoas.oregonstate.edu/sparcs/)
         Isi zip: 78 *_data.tif + *_labels.tif (2 subscene hanya berisi foto; *_mtl.txt yang
         disebut readme tidak ada, tetapi tidak dibutuhkan, lihat di bawah).
Output : dataset/sparcs/prepared/{scene}.npz  (image uint8 [11,H,W], mask uint8 [H,W])
         data/sparcs_prepared.json            (ringkasan, pemetaan kelas, cek overlap dengan Biome)

Radiometri: L8 Biome versi TorchGeo adalah DN Level-1 uint16 yang dipotong ke uint8 (DN // 256),
BUKAN reflektansi TOA. Bukti: median band cirrus B9 (hampir konstan secara fisik) = 20 di Biome
dan 5056 // 256 = 19 di SPARCS; dengan skala TOA SSL4EO-L ([0, 0.4] -> [0, 255]) median
reflektansi SPARCS jadi ~2x Biome. Jadi SPARCS dikonversi dengan DN // 256 yang sama.
SPARCS tidak menyertakan B8 (pankromatik). B8 diisi dari regresi linear B2-B4 yang di-fit
pada piksel scene TRAINING L8 Biome (R^2 = 0.999, RMSE 1.3 DN uint8; lihat PAN_COEF).

Pemetaan kelas SPARCS -> kelas proyek:
  0 Shadow, 1 Shadow di atas air        -> Cloud Shadow
  2 Air, 3 Salju/es, 4 Darat, 6 Banjir  -> Clear
  5 Awan                                -> Cloud (SPARCS tidak membedakan awan tipis/tebal)
  piksel tanpa data (DN 0 di semua band) -> Fill

    python scripts/prepare_sparcs.py
"""

import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402
from src.data import _fix_proj_data_path  # noqa: E402,F401  (set PROJ_DATA sebelum rasterio)
from src.utils import load_json, save_json  # noqa: E402

import rasterio  # noqa: E402

SPARCS_DIR = C.DATASET_DIR / "sparcs"
ZIP_PATH = SPARCS_DIR / "sparcs_data_L8.zip"
RAW_DIR = SPARCS_DIR / "raw"
OUT_DIR = SPARCS_DIR / "prepared"

# urutan band di *_data.tif (B8 pankromatik tidak ada); dicek dari rentang DN (B9 ~5000, B10/B11 >20000)
DATA_BANDS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11]
# B8 ~ a*B2 + b*B3 + c*B4 + d (uint8, scene train L8 Biome)
PAN_COEF = (0.3040, 0.0825, 0.5937, -0.1404)

LABEL_MAP = {0: C.SHADOW, 1: C.SHADOW, 2: C.CLEAR, 3: C.CLEAR, 4: C.CLEAR, 5: C.CLOUD, 6: C.CLEAR}


def convert(dn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """dn [10,H,W] uint16 -> (image uint8 [11,H,W] urutan B1..B11, valid [H,W])."""
    valid = (dn > 0).all(0)
    bands = {b: (dn[i] >> 8).astype(np.uint8) for i, b in enumerate(DATA_BANDS)}
    a, b_, c, d = PAN_COEF
    pan = a * bands[2].astype(np.float32) + b_ * bands[3] + c * bands[4] + d
    bands[8] = np.clip(np.rint(pan), 0, 255).astype(np.uint8)
    img = np.stack([bands[b] for b in range(1, 12)])
    img[:, ~valid] = 0
    return img, valid


def biome_scene_ids() -> dict:
    """scene id L8 Biome -> split (train/val/test)."""
    split = load_json(C.artefact_paths("l8biome")["split"])
    out = {}
    for name, scenes in split["scenes"].items():
        for s in scenes:
            out[Path(s).stem[:21]] = name
    return out


def main():
    if not RAW_DIR.exists():
        print(f"ekstrak {ZIP_PATH} ...")
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(RAW_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_files = sorted(RAW_DIR.rglob("*_data.tif"))
    print(f"{len(data_files)} subscene")
    biome = biome_scene_ids()
    biome_pr = {sid[3:9]: split for sid, split in biome.items()}

    scenes, totals = [], np.zeros(C.NUM_CLASSES, dtype=np.int64)
    for f in data_files:
        sid = f.name.replace("_data.tif", "")
        with rasterio.open(f) as r:
            dn = r.read()
        if dn.shape[0] != len(DATA_BANDS):
            raise ValueError(f"{sid}: {dn.shape[0]} band, diharapkan {len(DATA_BANDS)}")
        with rasterio.open(f.with_name(f"{sid}_labels.tif")) as r:
            lab = r.read(1)
        img, valid = convert(dn)
        mask = np.full(lab.shape, C.FILL, dtype=np.uint8)
        for raw, cls in LABEL_MAP.items():
            mask[lab == raw] = cls
        mask[~valid] = C.FILL
        np.savez_compressed(OUT_DIR / f"{sid}.npz", image=img, mask=mask)
        counts = np.bincount(mask.ravel(), minlength=C.NUM_CLASSES)[:C.NUM_CLASSES]
        totals += counts
        scene_id = sid[:21]
        scenes.append({
            "id": sid, "scene": scene_id,
            "pixel_counts": counts.tolist(),
            "band_mean_valid": img[:, valid].mean(1).round(2).tolist() if valid.any() else None,
            "biome_same_scene": biome.get(scene_id),
            "biome_same_pathrow": biome_pr.get(scene_id[3:9]),
        })
        print(f"  {sid}: shadow {counts[C.SHADOW] / max(counts[1:].sum(), 1) * 100:5.2f}%"
              f"  biome scene={scenes[-1]['biome_same_scene']} pathrow={scenes[-1]['biome_same_pathrow']}")

    valid_px = totals[1:].sum()
    summary = {
        "source": "https://emapr.ceoas.oregonstate.edu/sparcs/ (Hughes & Kennedy 2019)",
        "data_bands": DATA_BANDS, "radiometry": "DN // 256 (sama dengan L8 Biome TorchGeo)",
        "pan_coef": PAN_COEF, "label_map": {str(k): C.CLASS_NAMES[v] for k, v in LABEL_MAP.items()},
        "n_scenes": len(scenes), "pixel_counts": totals.tolist(),
        "pct_of_valid": {C.CLASS_NAMES[c]: float(totals[c] / valid_px * 100) for c in C.EVAL_CLASSES},
        "n_scene_with_shadow": sum(s["pixel_counts"][C.SHADOW] > 0 for s in scenes),
        "biome_same_scene": [s["id"] for s in scenes if s["biome_same_scene"]],
        "biome_same_pathrow_train": [s["id"] for s in scenes if s["biome_same_pathrow"] == "train"],
        "scenes": scenes,
    }
    save_json(summary, C.DATA_DIR / "sparcs_prepared.json")
    print(f"\nshadow {summary['pct_of_valid']['Cloud Shadow']:.2f}% piksel valid, "
          f"{summary['n_scene_with_shadow']}/{len(scenes)} scene ber-shadow")
    print(f"scene sama dengan Biome: {summary['biome_same_scene']}")
    print(f"path/row sama dengan scene train Biome: {summary['biome_same_pathrow_train']}")


if __name__ == "__main__":
    main()
