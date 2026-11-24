"""
geom_filter.py
==============
Filter geometri pasca-inferensi (tanpa training ulang): prediksi Shadow hanya dipertahankan bila
jatuh di jalur proyeksi bayangan awan. Jalur = mask awan PREDIKSI model sendiri (Thin + Cloud)
yang digeser menjauhi matahari untuk tinggi awan dalam CLOUD_HEIGHT_RANGE_M, dengan sudut
matahari dari MTL (sama dengan make_partial_masks.shadow_swath). Shadow di luar jalur -> Clear.

Varian "oracle" memakai awan ground truth untuk jalur (batas atas bila awan diprediksi sempurna).

Varian "geomaware" = METRIK sadar-geometri (prediksi tidak difilter): di scene TANPA anotasi
shadow, piksel GT Clear di dalam jalur bayangan dari awan GT dianggap ambigu dan tidak dihitung
(sama dengan mask partial-geom). Prediksi shadow di luar jalur tetap dihitung sebagai FP; di
scene beranotasi semua piksel dihitung seperti biasa.

Metrik dilaporkan untuk semua scene test dan untuk scene yang shadow-nya dianotasi saja
(mask berisi >= SHADOW_ANNOT_MIN_PX piksel shadow); di scene tanpa anotasi shadow yang benar
dihitung sebagai false positive.

Hasil: runs/{dataset}/geom_filter/{nama}.json (confusion per scene untuk raw/pred/oracle).

    python scripts/geom_filter.py --pattern "*samp-random__arch-unet__loss-wce__s42"
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config as C  # noqa: E402
from src.data import CloudDataModule, NativePatchDataset  # noqa: E402
from src.experiment import RunConfig  # noqa: E402
from src.metrics import confusion_from_arrays, metrics_from_confusion  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import Normalizer, load_json, save_json  # noqa: E402
from make_partial_masks import read_mtl, shadow_direction, shadow_swath  # noqa: E402

import rasterio  # noqa: E402
import torch  # noqa: E402

VARIANTS = ("raw", "pred", "oracle", "geomaware")


@torch.no_grad()
def predict_scene(model, ds: NativePatchDataset, sid: str, normalizer, device, batch=16):
    """Prediksi seluruh scene dengan grid test (tile 256 tanpa overlap); kembalikan pred, gt."""
    s, ps = ds.scenes[sid], ds.patch_size
    nh, nw = -(-s.height // ps), -(-s.width // ps)
    pred = np.zeros((nh * ps, nw * ps), dtype=np.uint8)
    gt = np.zeros_like(pred)
    keys = [(sid, r * ps, c * ps) for r in range(nh) for c in range(nw)]
    for i in range(0, len(keys), batch):
        samples = [ds[k] for k in keys[i:i + batch]]
        x = normalizer(torch.stack([d["image"] for d in samples]).float().to(device))
        out = model(x).float().argmax(1).cpu().numpy().astype(np.uint8)
        for (_, r, c), p, d in zip(keys[i:i + batch], out, samples):
            pred[r:r + ps, c:c + ps] = p
            gt[r:r + ps, c:c + ps] = d["mask"].numpy()
    return pred[:s.height, :s.width], gt[:s.height, :s.width]


def swath_for(cloud: np.ndarray, scene) -> np.ndarray:
    mtl = read_mtl(Path(scene.path).parent)
    with rasterio.open(scene.mask_path) as src:
        direction = shadow_direction(src, mtl["SUN_AZIMUTH"])
        res = src.res[0]
    return shadow_swath(cloud, direction, mtl["SUN_ELEVATION"], res)


def apply_filter(pred: np.ndarray, swath: np.ndarray) -> np.ndarray:
    out = pred.copy()
    out[(pred == C.SHADOW) & ~swath] = C.CLEAR
    return out


def summarize(per_scene: dict, scenes: list) -> dict:
    out = {}
    for subset, ids in (("all", scenes), ("annotated", [s for s in scenes if per_scene[s]["annotated"]])):
        out[subset] = {"n_scenes": len(ids)}
        for v in VARIANTS:
            cm = sum(np.asarray(per_scene[s][v]) for s in ids)
            out[subset][v] = metrics_from_confusion(cm)
    return out


def process(result_path: Path, dm: CloudDataModule, device, force: bool):
    res = load_json(result_path)
    cfg = RunConfig.from_dict(res["config"])
    ckpt = result_path.parent / "checkpoints" / f"{cfg.name()}.pt"
    out_path = result_path.parent / "geom_filter" / f"{cfg.name()}.json"
    if out_path.exists() and not force:
        old = load_json(out_path)
        if "geomaware" in old["summary"]["all"]:
            print(f"[skip] {cfg.name()}")
            return old["summary"]
    stats = load_json(C.artefact_paths(cfg.dataset)["stats"])
    normalizer = Normalizer(cfg.norm, stats, device)
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    ds = NativePatchDataset(dm.scenes["test"], dm.patch_size)
    per_scene, t0 = {}, time.time()
    for scene in dm.scenes["test"]:
        pred, gt = predict_scene(model, ds, scene.scene_id, normalizer, device)
        sw_pred = swath_for((pred == C.THIN) | (pred == C.CLOUD), scene)
        sw_gt = swath_for((gt == C.THIN) | (gt == C.CLOUD), scene)
        valid = gt != C.FILL
        annotated = int((gt == C.SHADOW).sum()) >= C.SHADOW_ANNOT_MIN_PX
        gt_ga = gt.copy()
        if not annotated:
            gt_ga[(gt == C.CLEAR) & sw_gt] = C.FILL
        per_scene[scene.scene_id] = {
            "annotated": annotated,
            "ambig_frac_valid": float(((gt_ga == C.FILL) & valid).sum() / max(valid.sum(), 1)),
            "geomaware": confusion_from_arrays(pred, gt_ga).tolist(),
            "swath_frac_valid_pred": float((sw_pred & valid).sum() / max(valid.sum(), 1)),
            "raw": confusion_from_arrays(pred, gt).tolist(),
            "pred": confusion_from_arrays(apply_filter(pred, sw_pred), gt).tolist(),
            "oracle": confusion_from_arrays(apply_filter(pred, sw_gt), gt).tolist(),
        }
    ids = list(per_scene)
    summary = summarize(per_scene, ids)
    save_json({"name": cfg.name(), "config": cfg.to_dict(), "summary": summary,
               "cloud_height_range_m": C.CLOUD_HEIGHT_RANGE_M, "per_scene": per_scene}, out_path)
    print(f"{cfg.name()} ({time.time() - t0:.0f} s)")
    for subset in ("all", "annotated"):
        line = [f"  {subset:9s} (n={summary[subset]['n_scenes']:2d})"]
        for v in VARIANTS:
            s = summary[subset][v]["per_class"]["Cloud Shadow"]
            line.append(f"{v}: IoU {s['iou']:5.2f} P {s['precision']:5.1f} R {s['recall']:5.1f}")
        print(" | ".join(line))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="l8biome", choices=list(C.DATASETS))
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dm = CloudDataModule(args.dataset, num_workers=0).setup(verbose=False)
    files = sorted((C.RUNS_DIR / args.dataset).glob(f"{args.pattern}.json"))
    print(f"{len(files)} run, {len(dm.scenes['test'])} scene test")
    for f in files:
        process(f, dm, device, args.force)


if __name__ == "__main__":
    main()
