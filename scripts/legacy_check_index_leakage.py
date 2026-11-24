"""
Cek index shadow VERSI LAMA (data/l7irish_shadow_index.pkl dari build_shadow_index.py lama):
berapa persen lokasi di index berasal dari scene val/test?

Tidak dibutuhkan untuk eksperimen baru (index baru dibangun dari scene training saja),
tapi berguna untuk mendokumentasikan kebocoran pada versi lama.

    python scripts/legacy_check_index_leakage.py --dataset l7irish --index_path data/l7irish_shadow_index.pkl
"""

import argparse
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CloudDataModule  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="l7irish", choices=["l7irish", "l8biome"])
    ap.add_argument("--index_path", required=True)
    ap.add_argument("--max_check", type=int, default=None)
    args = ap.parse_args()

    dm = CloudDataModule(args.dataset, num_workers=0).setup()
    with open(args.index_path, "rb") as f:
        bboxes = pickle.load(f)
    if args.max_check and args.max_check < len(bboxes):
        random.seed(0)
        bboxes = random.sample(bboxes, args.max_check)

    counts = Counter()
    lookups = {s: dm.scene_lookup(s) for s in ("train", "val", "test")}
    for b in bboxes:
        hit = tuple(s for s, lk in lookups.items() if lk.find(b) != "unknown")
        counts[hit or ("NONE",)] += 1

    n = len(bboxes)
    print(f"Dicek: {n} bbox")
    for k, v in counts.most_common():
        print(f"  {'+'.join(k):<16} {v:>6} ({v / n * 100:.1f}%)")
    leaked = sum(v for k, v in counts.items() if "val" in k or "test" in k)
    print(f"\nBbox dari scene val/test: {leaked} ({leaked / n * 100:.1f}%)")


if __name__ == "__main__":
    main()
