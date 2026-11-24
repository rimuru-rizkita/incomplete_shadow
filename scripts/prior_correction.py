"""
prior_correction.py
===================
Koreksi prior post-hoc (tanpa training ulang) untuk run yang sudah selesai.

Oversampling membuat prior kelas saat training (pi_train, mis. ~60% patch ber-shadow) jauh
berbeda dari prior alami (pi_target), sehingga recall shadow naik tetapi precision jatuh.
Dua koreksi dievaluasi dari checkpoint yang sama:

  prior : logit_c += log(pi_target_c / pi_train_c)  (Saerens et al. 2002; Menon et al. 2021).
          pi_train diukur dari sampler training run itu sendiri, pi_target = frekuensi piksel
          valid di scene training (fase 0). Tanpa tuning.
  tuned : hanya logit shadow digeser bias b; b dipilih di VALIDASI (F1 shadow, scene val yang
          berisi shadow), lalu diterapkan ke test.

Hasil per run: runs/{ds}/prior/{nama}.json (confusion global + per scene untuk base/prior/tuned,
dan confusion global test untuk seluruh grid b sebagai kurva precision-recall).

    python scripts/prior_correction.py                         # semua run di runs/
    python scripts/prior_correction.py --pattern "*oversample*" --runs_dir runs/_backup_old_protocol_20261101
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402
from src.data import CloudDataModule  # noqa: E402
from src.experiment import RunConfig  # noqa: E402
from src.metrics import ConfusionAccumulator, confusion_from_arrays, metrics_from_confusion  # noqa: E402
from src.models import build_model  # noqa: E402
from src.samplers import build_sampler, load_patch_index  # noqa: E402
from src.utils import Normalizer, get_mask, load_json, save_json, set_seed  # noqa: E402

import torch  # noqa: E402

BIASES = np.round(np.arange(-8.0, 4.01, 0.5), 2)


def shadow_scores(cm):
    s = metrics_from_confusion(cm)["per_class"]["Cloud Shadow"]
    return s["precision"], s["recall"], s["f1"], s["iou"]


def train_prior(cfg, dm, records, n_patches, seed=0):
    """Frekuensi kelas piksel valid yang benar-benar dilihat sampler training run ini."""
    set_seed(seed)
    sampler = build_sampler(replace(cfg, patches_per_epoch=n_patches), dm.train_view, records)
    counts = np.zeros(C.NUM_CLASSES)
    for batch in dm.train_loader(sampler):
        m = get_mask(batch)
        m[m == C.AMBIG] = C.CLEAR
        counts += np.bincount(m.flatten().numpy(), minlength=C.NUM_CLASSES)[:C.NUM_CLASSES]
    counts[C.FILL] = 0
    return counts / counts.sum()


@torch.no_grad()
def run_eval(model, loader, normalizer, device, adjust, biases, scene_ids=None):
    """Confusion untuk: base, prior (vektor adjust), dan setiap bias shadow di `biases`."""
    model.eval()
    accs = {"base": ConfusionAccumulator(), "prior": ConfusionAccumulator()}
    grid = np.zeros((len(biases), C.NUM_CLASSES, C.NUM_CLASSES), dtype=np.int64)
    grid_scene = {}
    adj = torch.tensor(adjust, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    bvec = torch.tensor(biases, dtype=torch.float32, device=device)
    offset = 0
    for batch in loader:
        x = normalizer(batch["image"].float().to(device))
        masks = get_mask(batch)
        masks[masks == C.AMBIG] = C.CLEAR
        logits = model(x).float()
        bs = x.shape[0]
        sids = scene_ids[offset:offset + bs] if scene_ids is not None else None
        tgt = masks.numpy()
        accs["base"].update(logits.argmax(1).cpu().numpy(), tgt, sids)
        accs["prior"].update((logits + adj).argmax(1).cpu().numpy(), tgt, sids)
        for k, b in enumerate(bvec):
            lg = logits.clone()
            lg[:, C.SHADOW] += b
            pred = lg.argmax(1).cpu().numpy()
            for i in range(bs):
                cm = confusion_from_arrays(pred[i], tgt[i])
                grid[k] += cm
                if sids is not None:
                    grid_scene.setdefault(sids[i], np.zeros_like(grid))[k] += cm
        offset += bs
    return accs, grid, grid_scene


def process(result_path: Path, args, device):
    res = load_json(result_path)
    cfg = RunConfig.from_dict(res["config"])
    ckpt = result_path.parent / "checkpoints" / f"{cfg.name()}.pt"
    out_path = result_path.parent / "prior" / f"{cfg.name()}.json"
    if out_path.exists() and not args.force:
        print(f"[skip] {cfg.name()}")
        return
    if not ckpt.exists():
        print(f"[skip] checkpoint tidak ada: {ckpt}")
        return
    print(f"\n=== {cfg.name()}")

    paths = C.artefact_paths(cfg.dataset)
    stats = load_json(paths["stats"])
    records = load_patch_index(paths["index"])["records"] if cfg.uses_pool() else None
    dm = CloudDataModule(cfg.dataset, cfg.patch_size, cfg.batch_size, args.num_workers,
                         partial=cfg.partial).setup(verbose=False)

    pi_train = train_prior(cfg, dm, records, args.n_train_patches)
    pc = np.asarray(stats["pixel_counts"], dtype=float)
    pc[C.FILL] = 0
    pi_target = pc / pc.sum()
    adjust = np.zeros(C.NUM_CLASSES)
    ev = C.EVAL_CLASSES
    adjust[ev] = np.log(pi_target[ev] + 1e-12) - np.log(pi_train[ev] + 1e-12)
    print("  pi_train  :", np.round(pi_train, 4).tolist())
    print("  pi_target :", np.round(pi_target, 4).tolist())
    print("  adjust    :", np.round(adjust, 3).tolist())

    normalizer = Normalizer(cfg.norm, stats, device)
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    # --- validasi: pilih bias shadow
    val_bboxes = dm.fixed_val_bboxes(cfg.val_patches)
    val_lookup = dm.scene_lookup("val")
    val_sids = [val_lookup.find(b) for b in val_bboxes] if val_lookup.scenes else None
    v_accs, v_grid, v_grid_scene = run_eval(model, dm.bbox_loader("val", val_bboxes), normalizer,
                                            device, adjust, BIASES, val_sids)
    # scene val yang berisi shadow (shadow tidak dianotasi di semua scene)
    shadow_scenes = [s for s, g in v_grid_scene.items() if g[0][C.SHADOW].sum() > 0]
    sel_grid = sum(v_grid_scene[s] for s in shadow_scenes) if shadow_scenes else v_grid
    f1s = [shadow_scores(sel_grid[k])[2] for k in range(len(BIASES))]
    k_star = int(np.argmax(f1s))
    b_star = float(BIASES[k_star])
    print(f"  val: {len(shadow_scenes)} scene ber-shadow; b* = {b_star:+.1f} (F1 shadow val {f1s[k_star]:.2f})")

    # --- test
    test_bboxes = dm.grid_bboxes("test")
    lookup = dm.scene_lookup("test")
    test_sids = [lookup.find(b) for b in test_bboxes] if lookup.scenes else None
    t_accs, t_grid, t_grid_scene = run_eval(model, dm.bbox_loader("test", test_bboxes), normalizer,
                                            device, adjust, BIASES, test_sids)
    tuned = {"total": t_grid[k_star],
             "per_scene": {s: g[k_star] for s, g in t_grid_scene.items()}}
    out = {
        "name": cfg.name(), "key": cfg.key(), "config": cfg.to_dict(),
        "pi_train": pi_train.tolist(), "pi_target": pi_target.tolist(), "adjust": adjust.tolist(),
        "biases": BIASES.tolist(), "b_star": b_star, "val_select_scenes": len(shadow_scenes),
        "val_f1_by_bias": f1s,
        "test": {
            "base": {"confusion": t_accs["base"].total.tolist(),
                     "per_scene_confusion": {k: v.tolist() for k, v in t_accs["base"].per_scene.items()}},
            "prior": {"confusion": t_accs["prior"].total.tolist(),
                      "per_scene_confusion": {k: v.tolist() for k, v in t_accs["prior"].per_scene.items()}},
            "tuned": {"confusion": tuned["total"].tolist(),
                      "per_scene_confusion": {k: v.tolist() for k, v in tuned["per_scene"].items()}},
        },
        "test_grid_confusion": t_grid.tolist(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out, out_path)
    for name in ("base", "prior", "tuned"):
        P, R, F, I = shadow_scores(np.array(out["test"][name]["confusion"]))
        print(f"  test {name:5s}: shadow P={P:5.2f} R={R:5.2f} F1={F:5.2f} IoU={I:5.2f} | "
              f"mIoU={metrics_from_confusion(np.array(out['test'][name]['confusion']))['mIoU']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default=str(C.RUNS_DIR))
    ap.add_argument("--pattern", default="*", help="glob nama run, mis. '*oversample*'")
    ap.add_argument("--n_train_patches", type=int, default=800)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(Path(args.runs_dir).glob(f"*/{args.pattern}.json"))
    print(f"{len(files)} run ditemukan di {args.runs_dir}")
    for f in files:
        process(f, args, device)


if __name__ == "__main__":
    main()
