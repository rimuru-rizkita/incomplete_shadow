"""
eval_sparcs.py
==============
Validasi eksternal model L8 Biome pada SPARCS (label shadow lengkap; siapkan dulu dengan
scripts/prepare_sparcs.py). Test set Biome sendiri hanya berlabel shadow di ~1/3 scene, sehingga
precision shadow di sana bias ke bawah; SPARCS tidak punya masalah itu.

Inferensi per subscene: pad reflect ke kelipatan 256, tile 256x256 tanpa overlap (sama dengan
grid test), lalu crop. Thin Cloud dan Cloud digabung jadi "Cloud" karena SPARCS tidak
membedakannya; metrik dilaporkan untuk 3 kelas (Shadow, Clear, Cloud).

Hasil: runs/l8biome/sparcs/{nama}.json (confusion 5x5 global + per scene, metrik 3 kelas).
Scene SPARCS yang sama persis dengan scene Biome (lihat data/sparcs_prepared.json) dikeluarkan
dari metrik ringkasan; confusion per scene tetap disimpan.

    python scripts/eval_sparcs.py                       # semua run L8 Biome di runs/
    python scripts/eval_sparcs.py --pattern "*oversample*"
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402
from src.experiment import RunConfig  # noqa: E402
from src.metrics import confusion_from_arrays  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import Normalizer, load_json, save_json  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

PREP_DIR = C.DATASET_DIR / "sparcs" / "prepared"
TILE = 256
EVAL3 = {"Cloud Shadow": [C.SHADOW], "Clear": [C.CLEAR], "Cloud": [C.THIN, C.CLOUD]}


def metrics3(cm5) -> dict:
    """Metrik 3 kelas dari confusion 5x5 (baris GT, kolom prediksi; Thin+Cloud digabung)."""
    cm5 = np.asarray(cm5, dtype=np.float64)
    groups = list(EVAL3.values())
    cm = np.array([[cm5[np.ix_(g, h)].sum() for h in groups] for g in groups])
    pred_fill = cm5[:, C.FILL]  # prediksi Fill dihitung sebagai salah
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    support = cm.sum(1) + np.array([pred_fill[g].sum() for g in groups])
    fn = support - tp
    eps = 1e-12
    iou, prec, rec = tp / (tp + fp + fn + eps), tp / (tp + fp + eps), tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    oa = tp.sum() / (cm.sum() + pred_fill[[C.SHADOW, C.CLEAR, C.THIN, C.CLOUD]].sum() + eps)
    out = {"mIoU3": float(iou.mean() * 100), "overall_accuracy": float(oa * 100),
           "micro_iou": float(oa / (2 - oa) * 100), "per_class": {}}
    for k, name in enumerate(EVAL3):
        out["per_class"][name] = {"iou": float(iou[k] * 100), "precision": float(prec[k] * 100),
                                  "recall": float(rec[k] * 100), "f1": float(f1[k] * 100),
                                  "support_px": int(support[k])}
    return out


@torch.no_grad()
def predict(model, img: np.ndarray, normalizer, device, batch: int = 16) -> np.ndarray:
    _, H, W = img.shape
    ph, pw = -H % TILE, -W % TILE
    x = torch.from_numpy(img).float().unsqueeze(0).to(device)
    x = normalizer(x)
    x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    tiles = x.unfold(2, TILE, TILE).unfold(3, TILE, TILE)  # [1,C,nh,nw,T,T]
    nh, nw = tiles.shape[2], tiles.shape[3]
    tiles = tiles.permute(0, 2, 3, 1, 4, 5).reshape(nh * nw, -1, TILE, TILE)
    preds = torch.cat([model(tiles[i:i + batch]).float().argmax(1)
                       for i in range(0, tiles.shape[0], batch)])
    pred = preds.view(nh, nw, TILE, TILE).permute(0, 2, 1, 3).reshape(nh * TILE, nw * TILE)
    return pred[:H, :W].cpu().numpy()


def process(result_path: Path, scenes: list, exclude: set, args, device):
    res = load_json(result_path)
    cfg = RunConfig.from_dict(res["config"])
    ckpt = result_path.parent / "checkpoints" / f"{cfg.name()}.pt"
    out_path = result_path.parent / "sparcs" / f"{cfg.name()}.json"
    if out_path.exists() and not args.force:
        print(f"[skip] {cfg.name()}")
        return
    if not ckpt.exists():
        print(f"[skip] checkpoint tidak ada: {ckpt}")
        return
    stats = load_json(C.artefact_paths(cfg.dataset)["stats"])
    normalizer = Normalizer(cfg.norm, stats, device)
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    per_scene = {}
    for f in scenes:
        d = np.load(f)
        per_scene[f.stem] = confusion_from_arrays(predict(model, d["image"], normalizer, device),
                                                  d["mask"])
    total = sum(cm for sid, cm in per_scene.items() if sid not in exclude)
    m = metrics3(total)
    save_json({"name": cfg.name(), "key": cfg.key(), "config": cfg.to_dict(),
               "excluded_scenes": sorted(exclude), "metrics": m, "confusion": total.tolist(),
               "per_scene_confusion": {k: v.tolist() for k, v in per_scene.items()}}, out_path)
    s = m["per_class"]["Cloud Shadow"]
    print(f"{cfg.name():70s} mIoU3 {m['mIoU3']:5.1f} | shadow P {s['precision']:5.1f} "
          f"R {s['recall']:5.1f} F1 {s['f1']:5.1f} IoU {s['iou']:5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default=str(C.RUNS_DIR))
    ap.add_argument("--pattern", default="*", help="glob nama run, mis. '*oversample*'")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prep = load_json(C.DATA_DIR / "sparcs_prepared.json")
    exclude = set(prep["biome_same_scene"])
    scenes = sorted(PREP_DIR.glob("*.npz"))
    files = sorted(Path(args.runs_dir).glob(f"l8biome/{args.pattern}.json"))
    print(f"{len(scenes)} subscene SPARCS ({len(exclude)} dikeluarkan), {len(files)} run")
    for f in files:
        process(f, scenes, exclude, args, device)


if __name__ == "__main__":
    main()
