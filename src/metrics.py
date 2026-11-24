"""
metrics.py
==========
Semua metrik dihitung dari confusion matrix (baris = ground truth,
kolom = prediksi). Fill di-ignore. mIoU = rata-rata 4 kelas
(Shadow, Clear, Thin Cloud, Cloud) = macro, metrik utama. micro_iou = definisi
TorchGeo/SSL4EO-L, dilaporkan untuk perbandingan dengan paper tersebut.

Hanya bergantung pada numpy supaya juga bisa dipakai di analisis
(bootstrap) tanpa GPU.
"""

import numpy as np

from . import config as C


def confusion_from_arrays(pred, target, num_classes=C.NUM_CLASSES):
    """pred, target: array integer (numpy atau torch di CPU) bentuk sama."""
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)
    valid = target != C.FILL
    idx = target[valid].astype(np.int64) * num_classes + pred[valid].astype(np.int64)
    return np.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def metrics_from_confusion(cm) -> dict:
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    eps = 1e-12
    iou = tp / (tp + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    ev = C.EVAL_CLASSES
    oa = tp[ev].sum() / (cm[np.ix_(ev, range(C.NUM_CLASSES))].sum() + eps)
    out = {
        "overall_accuracy": float(oa * 100),
        "mIoU": float(iou[ev].mean() * 100),
        # micro-averaged Jaccard (TorchGeo JaccardIndex(average='micro'), dilaporkan sebagai
        # "mIoU" di SSL4EO-L): sum TP / sum (TP+FP+FN) = OA / (2 - OA); didominasi Clear.
        "micro_iou": float(oa / (2 - oa) * 100),
        "macro_f1": float(f1[ev].mean() * 100),
        "per_class": {},
    }
    for c in ev:
        out["per_class"][C.CLASS_NAMES[c]] = {
            "iou": float(iou[c] * 100), "precision": float(prec[c] * 100),
            "recall": float(rec[c] * 100), "f1": float(f1[c] * 100),
            "support_px": int(cm[c].sum()),
        }
    return out


def flat_metrics(m: dict) -> dict:
    """Ratakan dict metrik jadi kolom-kolom untuk DataFrame."""
    oa = m["overall_accuracy"] / 100
    row = {"mIoU": m["mIoU"], "OA": m["overall_accuracy"], "macroF1": m["macro_f1"],
           "microIoU": m.get("micro_iou", oa / (2 - oa) * 100)}
    for cname, vals in m["per_class"].items():
        short = cname.replace("Cloud Shadow", "shadow").replace("Thin Cloud", "thin") \
                     .replace("Cloud", "cloud").replace("Clear", "clear")
        for k in ("iou", "precision", "recall", "f1"):
            row[f"{short}_{k}"] = vals[k]
    return row


class ConfusionAccumulator:
    """Akumulasi confusion matrix global + per scene."""

    def __init__(self):
        self.total = np.zeros((C.NUM_CLASSES, C.NUM_CLASSES), dtype=np.int64)
        self.per_scene = {}

    def update(self, preds, targets, scene_ids=None):
        """preds/targets [B,H,W] (numpy); scene_ids list panjang B atau None."""
        for i in range(preds.shape[0]):
            cm = confusion_from_arrays(preds[i], targets[i])
            self.total += cm
            if scene_ids is not None:
                sid = scene_ids[i]
                if sid not in self.per_scene:
                    self.per_scene[sid] = np.zeros_like(self.total)
                self.per_scene[sid] += cm

    def compute(self) -> dict:
        return metrics_from_confusion(self.total)
