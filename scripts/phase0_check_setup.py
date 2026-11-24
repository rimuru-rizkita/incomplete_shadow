"""
FASE 0a: cek environment, split scene-level, dan lapisan kompatibilitas TorchGeo.

Memeriksa apakah split train/val/test terpisah di level scene/lokasi dan
apakah masih ada overlap spasial antar split.

Output: data/{dataset}_split.json (daftar scene per split + overlap footprint)

    python scripts/phase0_check_setup.py --dataset l7irish l8biome
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src import config as C  # noqa: E402
from src.data import CloudDataModule  # noqa: E402
from src.geo_compat import footprint_overlaps  # noqa: E402
from src.utils import save_json  # noqa: E402


def check_env():
    import rasterio
    print(f"Python  : {sys.version.split()[0]}")
    print(f"PyTorch : {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"Rasterio: {rasterio.__version__}")
    print(f"GPU     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'TIDAK ADA'}")


def check_dataset(name: str):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    dm = CloudDataModule(name, num_workers=0).setup()

    # 1) Baca patch native, termasuk patch yang sebagian di luar raster (dipad Fill)
    sid = dm.scenes["train"][0].scene_id
    for key in [(sid, 0, 0), (sid, -100, -100)]:
        sample = dm.train_view[key]
        print(f"Patch {key}: image {tuple(sample['image'].shape)} {sample['image'].dtype}, "
              f"mask {tuple(sample['mask'].shape)}, kelas {sorted(sample['mask'].unique().tolist())}")

    # 2) Scene per split
    groups = {s: Counter(sc.group for sc in dm.scenes[s]) for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        print(f"  {s:5s}: {len(dm.scenes[s])} scene | grup (folder induk): {dict(groups[s])}")

    # 3) Irisan scene antar split (harus kosong) + overlap footprint spasial
    ids = {s: {sc.scene_id for sc in dm.scenes[s]} for s in dm.scenes}
    dup = (ids["train"] & ids["test"]) | (ids["train"] & ids["val"]) | (ids["val"] & ids["test"])
    print(f"Scene yang muncul di >1 split: {len(dup)} (harus 0)")

    # 4) Lokasi WRS-2 (path/row) per split
    locs = {s: {sc.location for sc in dm.scenes[s]} for s in dm.scenes}
    loc_dup = (locs["train"] & locs["test"]) | (locs["train"] & locs["val"]) | (locs["val"] & locs["test"])
    print(f"Split mode: {C.SPLIT_MODE} | lokasi: train={len(locs['train'])}, "
          f"val={len(locs['val'])}, test={len(locs['test'])}")
    print(f"Lokasi path/row yang muncul di >1 split: {len(loc_dup)}"
          + (" (harus 0)" if C.SPLIT_MODE == "location" else ""))

    # 5) Overlap footprint (bbox lon/lat) sisa; untuk split lokasi ini hanya tepi scene bertetangga
    ov_test = footprint_overlaps(dm.scenes["test"], dm.scenes["train"])
    ov_val = footprint_overlaps(dm.scenes["val"], dm.scenes["train"])
    big = lambda ov: sum(1 for p in ov if p[2] >= 0.5)  # noqa: E731
    print(f"Pasangan footprint test-train yang beririsan: {len(ov_test)} "
          f"(>=50% luas scene test: {big(ov_test)})")
    print(f"Pasangan footprint val-train yang beririsan : {len(ov_val)} "
          f"(>=50%: {big(ov_val)})")
    if big(ov_test):
        print("  (ada scene test yang sebagian besar arealnya juga ada di train. Laporkan di paper.)")

    save_json({
        "dataset": name,
        "split_mode": C.SPLIT_MODE,
        "split_seed": C.SPLIT_SEED,
        "locations": {s: sorted(locs[s]) for s in locs},
        "location_in_multiple_splits": sorted(loc_dup),
        "fractions": C.SPLIT_FRACTIONS,
        "scenes": {s: sorted(ids[s]) for s in ids},
        "groups": {s: dict(groups[s]) for s in groups},
        "scene_in_multiple_splits": sorted(dup),
        "footprint_overlap_test_train": ov_test,
        "footprint_overlap_val_train": ov_val,
    }, C.artefact_paths(name)["split"])
    print(f"[Saved] {C.artefact_paths(name)['split']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=list(C.DATASETS))
    args = ap.parse_args()
    check_env()
    for d in args.dataset:
        check_dataset(d)
