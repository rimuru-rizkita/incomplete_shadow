"""
make_figures.py  (FASE 5b)
==========================
  --cm           confusion matrix ter-normalisasi (agregat semua seed), tanpa GPU
  --curves       kurva validasi (shadow recall & loss) rata-rata antar seed, tanpa GPU
  --qualitative  prediksi pada patch TEST (bukan val/train) untuk beberapa konfigurasi,
                 + peta error shadow (TP hijau, FP merah, FN biru). Patch dipilih acak
                 ber-seed: setengah mengandung shadow, setengah tipikal (tidak dipilih
                 berdasarkan hasil model) -> contoh tidak dipilih secara manual.

    python make_figures.py --dataset l7irish --cm --curves
    python make_figures.py --dataset l7irish --qualitative --seed 42 --n_shadow 3 --n_typical 3
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import config as CONF  # noqa: E402
from src.experiment import RunConfig  # noqa: E402

COLORS = ["#000000", "#8B4513", "#4A90D9", "#B0C4DE", "#FFFFFF"]


def key_of(**kw):
    return RunConfig(seed=0, **kw).key()


def figure_configs(ds):
    return [
        ("Random | U-Net | WCE", dict(dataset=ds, sampling="random")),
        ("OS | U-Net | WCE", dict(dataset=ds, sampling="oversample")),
        ("OS | CBAM | WCE", dict(dataset=ds, sampling="oversample", cbam=True)),
        ("OS | U-Net | Hybrid", dict(dataset=ds, sampling="oversample", loss="hybrid")),
        ("OS | CBAM | Hybrid", dict(dataset=ds, sampling="oversample", cbam=True, loss="hybrid")),
    ]


def load_by_key(ds):
    out = {}
    for f in (CONF.RUNS_DIR / ds).glob("*.json"):
        r = json.loads(f.read_text())
        out.setdefault(r["key"], []).append(r)
    return out


def mask_rgb(mask):
    rgb = np.zeros((*mask.shape, 3))
    for c, h in enumerate(COLORS):
        rgb[mask == c] = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return rgb


def image_rgb(img, bands):
    rgb = img[list(bands)].transpose(1, 2, 0).astype(float)
    lo, hi = np.percentile(rgb, (2, 98))
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)


def error_rgb(pred, gt):
    rgb = np.ones((*gt.shape, 3)) * 0.85
    rgb[gt == CONF.FILL] = 0
    tp = (gt == CONF.SHADOW) & (pred == CONF.SHADOW)
    fp = (gt != CONF.SHADOW) & (gt != CONF.FILL) & (pred == CONF.SHADOW)
    fn = (gt == CONF.SHADOW) & (pred != CONF.SHADOW)
    rgb[tp], rgb[fp], rgb[fn] = (0.1, 0.7, 0.1), (0.9, 0.1, 0.1), (0.1, 0.3, 0.9)
    return rgb


# ------------------------------------------------------------------ CM & curves (tanpa GPU)
def plot_cm(ds, out):
    by_key = load_by_key(ds)
    cfgs = [(lbl, key_of(**kw)) for lbl, kw in figure_configs(ds)]
    cfgs = [(lbl, k) for lbl, k in cfgs if k in by_key]
    if not cfgs:
        print("Belum ada hasil untuk CM.")
        return
    ev = CONF.EVAL_CLASSES
    names = [CONF.CLASS_NAMES[c] for c in ev]
    fig, axes = plt.subplots(1, len(cfgs), figsize=(4.2 * len(cfgs), 4))
    axes = np.atleast_1d(axes)
    for ax, (lbl, k) in zip(axes, cfgs):
        cm = np.sum([np.array(r["confusion"]) for r in by_key[k]], 0)[np.ix_(ev, ev)].astype(float)
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
        for i in range(len(ev)):
            for j in range(len(ev)):
                ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 50 else "black")
        ax.set_xticks(range(len(ev)), names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(ev)), names, fontsize=8)
        ax.set_title(f"{lbl}\n(n={len(by_key[k])} seed)", fontsize=9)
        ax.set_xlabel("Predicted")
    axes[0].set_ylabel("Ground truth")
    plt.tight_layout()
    p = out / f"confusion_{ds}.png"
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {p}")


def plot_curves(ds, out):
    by_key = load_by_key(ds)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for lbl, kw in figure_configs(ds):
        rs = by_key.get(key_of(**kw))
        if not rs:
            continue
        n = min(len(r["history"]) for r in rs)
        for ax, field in zip(axes, ["val_shadow_recall", "val_loss"]):
            arr = np.array([[h[field] for h in r["history"][:n]] for r in rs])
            ax.plot(range(1, n + 1), arr.mean(0), label=lbl)
            ax.fill_between(range(1, n + 1), arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0), alpha=0.15)
    axes[0].set_ylabel("Val shadow recall (%)")
    axes[1].set_ylabel("Val loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    p = out / f"curves_{ds}.png"
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {p}")


# ------------------------------------------------------------------ qualitative (GPU/CPU)
def plot_qualitative(ds, seed, n_shadow, n_typical, out, pick_seed=0):
    import torch

    from src.data import CloudDataModule
    from src.models import build_model
    from src.utils import Normalizer, get_mask, load_json

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dm = CloudDataModule(ds, num_workers=0, batch_size=1).setup()
    stats = load_json(CONF.artefact_paths(ds)["stats"])

    models = []
    for lbl, kw in figure_configs(ds):
        cfg = RunConfig(seed=seed, **kw)
        if not cfg.checkpoint_path().exists():
            print(f"  [skip] checkpoint tidak ada: {cfg.name()}")
            continue
        m = build_model(cfg)
        m.load_state_dict(torch.load(cfg.checkpoint_path(), map_location=device, weights_only=True))
        models.append((lbl, m.to(device).eval(), Normalizer(cfg.norm, stats, device)))
    if not models:
        print("Tidak ada checkpoint.")
        return

    # Pilih patch test secara acak ber-seed (tidak bergantung hasil model)
    bboxes = dm.grid_bboxes("test")
    rng = random.Random(pick_seed)
    rng.shuffle(bboxes)
    chosen_shadow, chosen_typ = [], []
    for b in bboxes:
        s = dm.test_view[b]
        m = get_mask({"mask": s["mask"].unsqueeze(0)})[0].numpy()
        if (m != CONF.FILL).mean() < 0.5:
            continue
        has = (m == CONF.SHADOW).any()
        if has and len(chosen_shadow) < n_shadow:
            chosen_shadow.append((s["image"].float().numpy(), m, "shadow"))
        elif not has and len(chosen_typ) < n_typical:
            chosen_typ.append((s["image"].float().numpy(), m, "typical"))
        if len(chosen_shadow) >= n_shadow and len(chosen_typ) >= n_typical:
            break
    samples = chosen_shadow + chosen_typ
    bands = CONF.DATASETS[ds]["rgb_bands"]

    ncol = 2 + len(models) + 1
    fig, axes = plt.subplots(len(samples), ncol, figsize=(2.3 * ncol, 2.4 * len(samples)))
    axes = np.atleast_2d(axes)
    for r, (img, gt, tag) in enumerate(samples):
        x = torch.from_numpy(img).unsqueeze(0).to(device)
        preds = []
        with torch.no_grad():
            for lbl, m, norm in models:
                preds.append(m(norm(x)).argmax(1)[0].cpu().numpy())
        panels = [image_rgb(img, bands), mask_rgb(gt)] + [mask_rgb(p) for p in preds] \
            + [error_rgb(preds[min(1, len(preds) - 1)], gt)]
        titles = ["Image", "Ground truth"] + [lbl for lbl, _, _ in models] + \
                 [f"Shadow error\n({models[min(1, len(models) - 1)][0]})"]
        for c, (pan, t) in enumerate(zip(panels, titles)):
            axes[r, c].imshow(pan)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(t, fontsize=8)
        axes[r, 0].text(-10, 128, tag, rotation=90, va="center", ha="right", fontsize=8)
    handles = [mpatches.Patch(color=COLORS[c], label=CONF.CLASS_NAMES[c], ec="gray") for c in range(5)]
    handles += [mpatches.Patch(color=(0.1, 0.7, 0.1), label="TP"),
                mpatches.Patch(color=(0.9, 0.1, 0.1), label="FP"),
                mpatches.Patch(color=(0.1, 0.3, 0.9), label="FN")]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    p = out / f"qualitative_{ds}_seed{seed}.png"
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=list(CONF.DATASETS))
    ap.add_argument("--cm", action="store_true")
    ap.add_argument("--curves", action="store_true")
    ap.add_argument("--qualitative", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_shadow", type=int, default=3)
    ap.add_argument("--n_typical", type=int, default=2)
    args = ap.parse_args()
    out = CONF.FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    for ds in args.dataset:
        if args.cm:
            plot_cm(ds, out)
        if args.curves:
            plot_curves(ds, out)
        if args.qualitative:
            plot_qualitative(ds, args.seed, args.n_shadow, args.n_typical, out)
