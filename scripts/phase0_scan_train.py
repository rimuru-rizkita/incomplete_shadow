"""
FASE 0b: scan PENUH seluruh scene TRAINING (grid, stride = patch size).

Menggantikan build_shadow_index.py versi lama, yang:
  - men-scan full_dataset (termasuk scene val/test)  -> risiko leakage
  - berhenti setelah 3000 lokasi                      -> index hanya dari sebagian scene

Output:
  data/{ds}_patch_index.pkl   semua patch grid training + jumlah piksel per kelas
                              (dipakai semua sampler; tau bisa diatur tanpa scan ulang)
  data/{ds}_train_stats.json  statistik untuk paper & normalisasi:
     - jumlah piksel per kelas, prevalensi shadow DENGAN dan TANPA Fill
       (menjelaskan inkonsistensi 0.7% vs 0.033% di versi lama)
     - fraksi patch grid yang mengandung shadow, untuk beberapa tau
     - pi_shadow empiris di bawah sampler random (pengganti asumsi 0.01 di Eq. 7)
     - mean/std per band (piksel valid saja) untuk normalisasi z-score
     - waktu scan (biaya preprocessing)

Opsional --extra_splits val test: hitung statistik kelas val/test juga (TANPA index).

    python scripts/phase0_scan_train.py --dataset l7irish l8biome
"""

import argparse
import pickle
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src import config as C  # noqa: E402
from src.data import CloudDataModule  # noqa: E402
from src.utils import get_mask, load_json, save_json  # noqa: E402

TAUS = [0.0, 0.001, 0.01, 0.05, 0.1]


def scan_split(dm, split, with_index: bool, max_patches=None):
    bboxes = dm.grid_bboxes(split)
    if max_patches:
        bboxes = bboxes[:max_patches]
    lookup = dm.scene_lookup(split)
    loader = dm.bbox_loader(split, bboxes)
    nb = C.DATASETS[dm.dataset]["in_channels"]

    pixel_counts = np.zeros(C.NUM_CLASSES, dtype=np.int64)
    band_sum = np.zeros(nb, dtype=np.float64)
    band_sq = np.zeros(nb, dtype=np.float64)
    n_valid_total = 0
    records = []
    offset = 0
    t0 = time.time()
    print(f"[{dm.dataset}/{split}] scan {len(bboxes)} patch grid...")
    for i, batch in enumerate(loader):
        masks = get_mask(batch)
        images = batch["image"].double()
        for j in range(masks.shape[0]):
            m = masks[j]
            counts = torch.bincount(m.flatten(), minlength=C.NUM_CLASSES).numpy()
            pixel_counts += counts
            valid = m != C.FILL
            nv = int(valid.sum())
            if nv:
                px = images[j][:, valid]  # [C, nv]
                band_sum += px.sum(1).numpy()
                band_sq += (px ** 2).sum(1).numpy()
                n_valid_total += nv
            if with_index:
                b = bboxes[offset + j]
                records.append({
                    "bbox": b,
                    "scene": lookup.find(b) if lookup.scenes else "unknown",
                    "valid": nv,
                    "counts": counts.tolist(),
                })
        offset += masks.shape[0]
        if (i + 1) % 200 == 0:
            print(f"  {offset}/{len(bboxes)} patch ({time.time() - t0:.0f} dtk)")
    minutes = (time.time() - t0) / 60
    return records, pixel_counts, band_sum, band_sq, n_valid_total, minutes, len(bboxes)


def empirical_pi_random(dm, n=5000):
    """Fraksi patch ber-shadow di bawah sampler random (sampling baseline sebenarnya)."""
    rng = random.Random(12345)
    bboxes = [dm.train_view.random_key(rng) for _ in range(n)]
    loader = dm.bbox_loader("train", bboxes)
    n_shadow, n_nonempty, n_total = 0, 0, 0
    for batch in loader:
        m = get_mask(batch)
        n_total += m.shape[0]
        n_nonempty += int((m != C.FILL).flatten(1).any(1).sum())
        n_shadow += int((m == C.SHADOW).flatten(1).any(1).sum())
    return {"n_sampled": n_total, "n_with_valid_pixels": n_nonempty, "n_with_shadow": n_shadow,
            "pi_shadow": n_shadow / max(n_total, 1)}


def class_summary(pixel_counts):
    total = int(pixel_counts.sum())
    valid = int(pixel_counts[1:].sum())
    return {
        "pixel_counts": pixel_counts.tolist(),
        "pct_of_all_pixels": (pixel_counts / max(total, 1) * 100).tolist(),
        "pct_of_valid_pixels": [0.0] + (pixel_counts[1:] / max(valid, 1) * 100).tolist(),
    }


def run(dataset, num_workers, max_patches, n_random, extra_splits):
    dm = CloudDataModule(dataset, num_workers=num_workers, batch_size=16).setup()
    paths = C.artefact_paths(dataset)

    records, pc, bsum, bsq, nvalid, minutes, n_grid = scan_split(dm, "train", True, max_patches)
    mean = bsum / max(nvalid, 1)
    std = np.sqrt(np.maximum(bsq / max(nvalid, 1) - mean ** 2, 0))

    with open(paths["index"], "wb") as f:
        pickle.dump({"meta": {"dataset": dataset, "split": "train", "patch_size": dm.patch_size,
                              "stride": dm.patch_size, "n_records": len(records),
                              "scan_minutes": minutes}, "records": records}, f)
    print(f"[Saved] {paths['index']} ({len(records)} patch)")

    nonempty = [r for r in records if r["valid"] > 0]
    tau_table = {}
    for tau in TAUS:
        if tau == 0:
            pool = [r for r in nonempty if r["counts"][C.SHADOW] > 0]
        else:
            pool = [r for r in nonempty if r["counts"][C.SHADOW] / r["valid"] >= tau]
        tau_table[str(tau)] = {
            "n_patches": len(pool),
            "pct_of_nonempty_grid_patches": len(pool) / max(len(nonempty), 1) * 100,
            "n_scenes": len({r["scene"] for r in pool}),
        }

    stats = {
        "dataset": dataset,
        "split": "train",
        **class_summary(pc),
        "n_grid_patches": n_grid,
        "n_nonempty_grid_patches": len(nonempty),
        "shadow_pool_by_tau": tau_table,
        "band_mean": mean.tolist(),
        "band_std": std.tolist(),
        "scan_minutes": minutes,
    }
    if n_random:
        print(f"[{dataset}] estimasi pi_shadow di bawah sampler random (n={n_random})...")
        stats["random_sampler"] = empirical_pi_random(dm, n_random)

    for split in extra_splits:
        _, pc_s, *_ = scan_split(dm, split, False, max_patches)
        stats[f"{split}_classes"] = class_summary(pc_s)

    old = load_json(paths["stats"]) or {}
    old.update(stats)
    save_json(old, paths["stats"])
    print(f"[Saved] {paths['stats']}")

    s = stats
    print(f"\nRINGKASAN {dataset}:")
    print(f"  Shadow: {s['pct_of_all_pixels'][1]:.3f}% semua piksel | "
          f"{s['pct_of_valid_pixels'][1]:.3f}% piksel valid (tanpa Fill)")
    print(f"  Patch grid ber-shadow (tau=0): {tau_table['0.0']['pct_of_nonempty_grid_patches']:.2f}% "
          f"dari {len(nonempty)} patch non-kosong, {tau_table['0.0']['n_scenes']} scene")
    if "random_sampler" in s:
        print(f"  pi_shadow (sampler random): {s['random_sampler']['pi_shadow'] * 100:.2f}%")
    print(f"  Waktu scan: {minutes:.1f} menit")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=list(C.DATASETS))
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_patches", type=int, default=None, help="Untuk uji cepat saja")
    ap.add_argument("--n_random", type=int, default=5000)
    ap.add_argument("--extra_splits", nargs="*", default=[], choices=["val", "test"])
    args = ap.parse_args()
    for d in args.dataset:
        run(d, args.num_workers, args.max_patches, args.n_random, args.extra_splits)
