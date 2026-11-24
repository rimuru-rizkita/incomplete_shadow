"""
make_partial_masks.py
=====================
Membuat salinan dataset dengan mask PARTIAL LABEL untuk scene yang shadow-nya tidak
dianotasi (L7 Irish: ~45/206 scene berlabel shadow; L8 Biome: ~32/96). Di scene seperti
itu bayangan awan ikut berlabel Clear, sehingga prediksi shadow yang benar dihukum.

Dua mode (piksel Clear di scene tanpa anotasi shadow -> nilai AMBIG_RAW = "Clear atau Shadow"):
  all  : SEMUA piksel Clear dibuat ambigu (setara partial label Francis 2024, SEnSeI-v2).
  geom : hanya piksel Clear di jalur proyeksi bayangan awan yang dibuat ambigu. Jalur =
         mask awan (thin + thick, dari ground truth) digeser menjauhi matahari sejauh
         h / tan(elevasi) untuk tinggi awan h dalam CLOUD_HEIGHT_RANGE_M (seperti Fmask),
         memakai SUN_AZIMUTH & SUN_ELEVATION dari file MTL. Di luar jalur tetap Clear.

Scene yang mask-nya berisi piksel shadow (>= SHADOW_ANNOT_MIN_PX) dianggap berlabel dan
disalin apa adanya. File citra & MTL di-hardlink (tanpa tambahan ruang disk).

Output: dataset/partial/{mode}/{dataset}/... (struktur folder sama) dan ringkasan
data/{dataset}_partial_{mode}.json.

    python scripts/make_partial_masks.py --dataset l7irish l8biome --mode all geom
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402
from src.data import _fix_proj_data_path  # noqa: E402,F401  (set PROJ_DATA sebelum rasterio)
from src.utils import save_json  # noqa: E402

import rasterio  # noqa: E402
from pyproj import Geod, Transformer  # noqa: E402

RAW_SHADOW, RAW_CLEAR, RAW_THIN, RAW_CLOUD = 64, 128, 192, 255
MASK_GLOB = {"l7irish": "L7_p*_r*_newmask2015.TIF", "l8biome": "LC8*_fixedmask.TIF"}


def read_mtl(scene_dir: Path) -> dict:
    mtl = next(scene_dir.glob("*_MTL.txt"))
    out = {}
    for line in mtl.read_text(errors="ignore").splitlines():
        if "=" in line:
            k, v = (s.strip() for s in line.split("=", 1))
            if k in ("SUN_AZIMUTH", "SUN_ELEVATION"):
                out[k] = float(v)
    return out


def shadow_direction(src, sun_azimuth: float) -> tuple[float, float]:
    """
    Vektor satuan arah bayangan di koordinat raster (kolom, baris). Azimuth matahari
    diukur dari utara sejati; arah bayangan = azimuth + 180. Dikonversi ke arah grid
    lewat titik tengah scene supaya benar juga untuk proyeksi polar (grid north != true north).
    """
    h, w = src.height, src.width
    cx, cy = src.xy(h // 2, w // 2)
    to_ll = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
    to_xy = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    lon, lat = to_ll.transform(cx, cy)
    lon2, lat2, _ = Geod(ellps="WGS84").fwd(lon, lat, (sun_azimuth + 180.0) % 360.0, 1000.0)
    x2, y2 = to_xy.transform(lon2, lat2)
    dcol = (x2 - cx) / src.res[0]
    drow = -(y2 - cy) / src.res[1]
    n = float(np.hypot(dcol, drow))
    return dcol / n, drow / n


def shadow_swath(cloud: np.ndarray, direction, elevation_deg: float, res_m: float) -> np.ndarray:
    """Union mask awan yang digeser sepanjang arah bayangan untuk semua tinggi awan (bool)."""
    ds = C.GEOM_DOWNSAMPLE
    H, W = cloud.shape
    hh, ww = -(-H // ds), -(-W // ds)
    pad = np.zeros((hh * ds, ww * ds), dtype=bool)
    pad[:H, :W] = cloud
    small = pad.reshape(hh, ds, ww, ds).any(axis=(1, 3))

    tan_e = np.tan(np.radians(max(elevation_deg, 1.0)))
    h_min, h_max = C.CLOUD_HEIGHT_RANGE_M
    step = res_m * ds
    k_min, k_max = int(np.floor(h_min / tan_e / step)), int(np.ceil(h_max / tan_e / step))
    ux, uy = direction
    offsets = {(int(round(k * uy)), int(round(k * ux))) for k in range(k_min, k_max + 1)}

    swath = np.zeros_like(small)
    for dr, dc in offsets:
        if abs(dr) >= hh or abs(dc) >= ww:
            continue
        src_r = slice(max(0, -dr), hh - max(0, dr))
        dst_r = slice(max(0, dr), hh - max(0, -dr))
        src_c = slice(max(0, -dc), ww - max(0, dc))
        dst_c = slice(max(0, dc), ww - max(0, -dc))
        swath[dst_r, dst_c] |= small[src_r, src_c]

    if C.GEOM_BUFFER_PX > 0:
        from scipy.ndimage import binary_dilation
        swath = binary_dilation(swath, iterations=C.GEOM_BUFFER_PX)
    big = np.repeat(np.repeat(swath, ds, axis=0), ds, axis=1)
    return big[:H, :W]


def mirror_scene_dir(src_dir: Path, dst_dir: Path, skip: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in src_dir.iterdir():
        if f.is_file() and f != skip:
            target = dst_dir / f.name
            if not target.exists():
                try:
                    os.link(f, target)
                except OSError:
                    shutil.copy2(f, target)


def process(dataset: str, mode: str):
    src_root = Path(C.DATASETS[dataset]["root"])
    dst_root = C.partial_root(dataset, mode)
    masks = sorted(src_root.rglob(MASK_GLOB[dataset]))
    print(f"\n[{dataset} | {mode}] {len(masks)} mask -> {dst_root}")
    summary, t0 = [], time.time()
    for i, mpath in enumerate(masks, 1):
        rel = mpath.relative_to(src_root)
        out_path = dst_root / rel
        mirror_scene_dir(mpath.parent, out_path.parent, skip=mpath)
        with rasterio.open(mpath) as src:
            mask = src.read(1)
            profile = src.profile
            n_shadow = int((mask == RAW_SHADOW).sum())
            clear = mask == RAW_CLEAR
            rec = {"scene": f"{rel.parent.name}/{mpath.name}", "shadow_px": n_shadow,
                   "clear_px": int(clear.sum()),
                   "cloud_px": int(((mask == RAW_THIN) | (mask == RAW_CLOUD)).sum())}
            annotated = n_shadow >= C.SHADOW_ANNOT_MIN_PX
            rec["shadow_annotated"] = annotated
            ambig = np.zeros_like(clear)
            if mode == "geom":
                # Jalur juga dihitung di scene berlabel: cakupan shadow asli oleh jalur
                # (swath_shadow_recall) memvalidasi geometri; mask scene berlabel tidak diubah.
                mtl = read_mtl(mpath.parent)
                cloud = (mask == RAW_THIN) | (mask == RAW_CLOUD)
                direction = shadow_direction(src, mtl["SUN_AZIMUTH"])
                swath = shadow_swath(cloud, direction, mtl["SUN_ELEVATION"], src.res[0])
                rec.update(sun_azimuth=mtl["SUN_AZIMUTH"], sun_elevation=mtl["SUN_ELEVATION"],
                           direction_colrow=[round(direction[0], 4), round(direction[1], 4)],
                           swath_frac_of_clear=int((swath & clear).sum()) / max(rec["clear_px"], 1))
                if annotated:
                    rec["swath_shadow_recall"] = int((swath & (mask == RAW_SHADOW)).sum()) / n_shadow
                else:
                    ambig = clear & swath
            elif not annotated:
                ambig = clear
        rec["ambig_px"] = int(ambig.sum())
        rec["ambig_frac_of_clear"] = rec["ambig_px"] / max(rec["clear_px"], 1)
        out = mask.copy()
        out[ambig] = C.AMBIG_RAW
        profile.update(compress="lzw")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out, 1)
        summary.append(rec)
        extra = (f", jalur menutup {100 * rec['swath_shadow_recall']:.1f}% shadow asli"
                 if "swath_shadow_recall" in rec else "")
        print(f"  [{i}/{len(masks)}] {rec['scene']}: "
              f"{'berlabel' if annotated else 'tanpa label'}, ambig {100 * rec['ambig_frac_of_clear']:.1f}% "
              f"dari Clear{extra} ({time.time() - t0:.0f} s)", flush=True)

    unl = [r for r in summary if not r["shadow_annotated"]]
    agg = {
        "dataset": dataset, "mode": mode, "n_scenes": len(summary),
        "n_shadow_annotated": len(summary) - len(unl),
        "ambig_frac_of_clear_in_unannotated": (sum(r["ambig_px"] for r in unl)
                                               / max(sum(r["clear_px"] for r in unl), 1)),
        "swath_shadow_recall_annotated": (
            sum(r["swath_shadow_recall"] * r["shadow_px"] for r in summary if "swath_shadow_recall" in r)
            / max(sum(r["shadow_px"] for r in summary if "swath_shadow_recall" in r), 1)
            if mode == "geom" else None),
        "swath_frac_of_clear_annotated": (
            sum(r["swath_frac_of_clear"] * r["clear_px"] for r in summary if r["shadow_annotated"])
            / max(sum(r["clear_px"] for r in summary if r["shadow_annotated"]), 1)
            if mode == "geom" else None),
        "cloud_height_range_m": C.CLOUD_HEIGHT_RANGE_M, "downsample": C.GEOM_DOWNSAMPLE,
        "buffer_px": C.GEOM_BUFFER_PX, "scenes": summary,
    }
    save_json(agg, C.DATA_DIR / f"{dataset}_partial_{mode}.json")
    print(f"[{dataset} | {mode}] scene berlabel shadow: {agg['n_shadow_annotated']}/{len(summary)}; "
          f"ambig = {100 * agg['ambig_frac_of_clear_in_unannotated']:.1f}% piksel Clear di scene tanpa label")
    if mode == "geom":
        print(f"  validasi geometri di scene berlabel: jalur menutup "
              f"{100 * agg['swath_shadow_recall_annotated']:.1f}% piksel shadow asli, "
              f"dengan luas {100 * agg['swath_frac_of_clear_annotated']:.1f}% piksel Clear")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=list(C.DATASETS), choices=list(C.DATASETS))
    ap.add_argument("--mode", nargs="+", default=list(C.PARTIAL_MODES), choices=list(C.PARTIAL_MODES))
    args = ap.parse_args()
    for ds in args.dataset:
        for mode in args.mode:
            process(ds, mode)


if __name__ == "__main__":
    main()
