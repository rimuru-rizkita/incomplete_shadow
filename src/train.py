"""
train.py
========
Training + evaluasi SATU run (satu RunConfig).

Perubahan utama dari versi lama:
  - validasi per-epoch memakai patch yang SAMA di semua epoch & run
  - normalisasi input per-band, augmentasi per sampel
  - semua faktor eksperimen + waktu (train/test) + jumlah parameter
    disimpan di JSON hasil
  - test menyimpan confusion matrix global DAN per scene (untuk bootstrap)
  - checkpoint terbaik disimpan; run yang sudah selesai otomatis di-skip
"""

import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from . import config as C
from .data import CloudDataModule
from .experiment import RunConfig
from .losses import build_loss
from .metrics import ConfusionAccumulator
from .models import build_model
from .samplers import build_sampler, load_patch_index
from .utils import Normalizer, augment_batch, get_mask, load_json, save_json, set_seed


def _count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _count_gflops(model, cfg, device):
    """GFLOPs satu forward pass 1 patch (FLOPs = 2 x MAC, via FlopCounterMode). None bila gagal."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        x = torch.zeros(1, C.DATASETS[cfg.dataset]["in_channels"], cfg.patch_size,
                        cfg.patch_size, device=device)
        was_training = model.training
        model.eval()
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x)
        model.train(was_training)
        return fc.get_total_flops() / 1e9
    except Exception as e:  # jangan hentikan training karena penghitung FLOPs
        print(f"[warning] hitung FLOPs gagal: {e!r}")
        return None


def _select_score(vm: dict, val_loss: float) -> float:
    """Skor pemilihan checkpoint; makin besar makin baik (lihat C.SELECT_METRIC)."""
    mode = C.SELECT_METRIC
    if mode == "loss":
        return -val_loss
    if mode == "shadow_f1":
        return vm["per_class"]["Cloud Shadow"]["f1"]
    if mode == "mIoU":
        return vm["mIoU"]
    raise ValueError(f"SELECT_METRIC tidak dikenal: {mode}")


@torch.no_grad()
def evaluate(model, loader, normalizer, device, loss_fn=None, scene_ids=None, max_batches=None):
    """Evaluasi pada loader berurutan. scene_ids: list per patch (urutan sama dengan loader)."""
    model.eval()
    acc = ConfusionAccumulator()
    total_loss, n_loss, offset = 0.0, 0, 0
    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        images = normalizer(batch["image"].float().to(device))
        masks = get_mask(batch).to(device)
        masks[masks == C.AMBIG] = C.CLEAR  # partial label: evaluasi memakai label asli
        bs = images.shape[0]
        logits = model(images).float()
        if loss_fn is not None and (masks != C.FILL).any():
            loss = loss_fn(logits, masks)
            if torch.isfinite(loss):
                total_loss += loss.item()
                n_loss += 1
        preds = logits.argmax(1).cpu().numpy()
        sids = scene_ids[offset:offset + bs] if scene_ids is not None else None
        acc.update(preds, masks.cpu().numpy(), sids)
        offset += bs
    return acc, (total_loss / n_loss if n_loss else float("nan"))


def train_one_run(cfg: RunConfig, num_workers: int = 4, quick_test: bool = False,
                  force: bool = False) -> dict | None:
    out_path = cfg.result_path()
    if out_path.exists() and not force and not quick_test:
        print(f"[skip] {cfg.name()} sudah selesai.")
        return load_json(out_path)

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 70}\nRUN: {cfg.name()}\nDevice: {device}\n{'=' * 70}")

    paths = C.artefact_paths(cfg.dataset)
    stats = load_json(paths["stats"])
    records = None
    if cfg.uses_pool():
        records = load_patch_index(paths["index"])["records"]

    # ------------------------------------------------------------ data
    if cfg.partial != "none" and cfg.loss not in ("ce", "wce"):
        raise ValueError(f"partial={cfg.partial} hanya didukung loss ce/wce (marginal loss)")
    dm = CloudDataModule(cfg.dataset, cfg.patch_size, cfg.batch_size, num_workers,
                         partial=cfg.partial).setup()
    length = 16 if quick_test else cfg.patches_per_epoch
    qcfg = cfg if not quick_test else type(cfg)(**{**cfg.to_dict(), "patches_per_epoch": length})
    sampler = build_sampler(qcfg, dm.train_view, records)
    train_loader = dm.train_loader(sampler)

    val_bboxes = dm.fixed_val_bboxes(8 if quick_test else cfg.val_patches)
    val_loader = dm.bbox_loader("val", val_bboxes)

    test_bboxes = dm.grid_bboxes("test")
    if quick_test:
        test_bboxes = test_bboxes[:32]
    lookup = dm.scene_lookup("test")
    test_scene_ids = [lookup.find(b) for b in test_bboxes] if lookup.scenes else None
    test_loader = dm.bbox_loader("test", test_bboxes)

    # ------------------------------------------------------------ model & loss
    normalizer = Normalizer(cfg.norm, stats, device)
    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg.loss, stats).to(device)
    n_total, n_train = _count_params(model)
    gflops = _count_gflops(model, cfg, device)
    print(f"[Model] params total={n_total:,} trainable={n_train:,} GFLOPs={gflops}")

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=cfg.lr, weight_decay=C.TRAIN_DEFAULTS["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=C.TRAIN_DEFAULTS["plateau_patience"])

    epochs = 1 if quick_test else cfg.epochs
    best_loss, best_state, best_epoch, patience = float("inf"), None, 0, 0
    best_score = float("-inf")
    history = []
    t_train = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss, n_b, n_shadow_patches, n_patches = 0.0, 0, 0, 0
        for batch in train_loader:
            images = batch["image"].float().to(device)
            masks = get_mask(batch).to(device)
            n_patches += masks.shape[0]
            n_shadow_patches += int((masks == C.SHADOW).flatten(1).any(1).sum())
            if not (masks != C.FILL).any():
                continue
            if cfg.augment:
                images, masks = augment_batch(images, masks)
            images = normalizer(images)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images).float(), masks)
            if not torch.isfinite(loss):
                print("  [warning] loss non-finite, batch di-skip")
                continue
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            n_b += 1
        tr_loss /= max(n_b, 1)

        val_acc, val_loss = evaluate(model, val_loader, normalizer, device, loss_fn)
        vm = val_acc.compute()
        if np.isfinite(val_loss):
            scheduler.step(val_loss)
        shadow_frac_seen = n_shadow_patches / max(n_patches, 1)
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss,
            "val_mIoU": vm["mIoU"],
            "val_shadow_recall": vm["per_class"]["Cloud Shadow"]["recall"],
            "val_shadow_iou": vm["per_class"]["Cloud Shadow"]["iou"],
            "val_shadow_f1": vm["per_class"]["Cloud Shadow"]["f1"],
            "train_patches_with_shadow": shadow_frac_seen,
            "lr": optimizer.param_groups[0]["lr"],
        })
        print(f"Epoch {epoch:02d} | train {tr_loss:.4f} | val {val_loss:.4f} | "
              f"mIoU {vm['mIoU']:.2f} | shadowR {history[-1]['val_shadow_recall']:.2f} | "
              f"patch ber-shadow {shadow_frac_seen * 100:.1f}%")

        score = _select_score(vm, val_loss)
        eligible = epoch >= C.TRAIN_DEFAULTS["select_from_epoch"] or quick_test
        if eligible and np.isfinite(score) and score > best_score:
            best_score, best_loss, best_epoch, patience = score, val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif eligible:
            patience += 1
            if (patience >= C.TRAIN_DEFAULTS["early_stop_patience"]
                    and epoch >= C.TRAIN_DEFAULTS["min_epochs"]):
                print(f"Early stopping di epoch {epoch}")
                break

    train_minutes = (time.time() - t_train) / 60
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if not quick_test:
        cfg.checkpoint_path().parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, cfg.checkpoint_path())

    # ------------------------------------------------------------ test
    model.load_state_dict(best_state)
    t_test = time.time()
    test_acc, _ = evaluate(model, test_loader, normalizer, device, scene_ids=test_scene_ids)
    test_minutes = (time.time() - t_test) / 60
    test_metrics = test_acc.compute()

    result = {
        "config": cfg.to_dict(),
        "name": cfg.name(),
        "key": cfg.key(),
        "test": test_metrics,
        "confusion": test_acc.total.tolist(),
        "per_scene_confusion": {k: v.tolist() for k, v in test_acc.per_scene.items()},
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "select_metric": C.SELECT_METRIC,
        "select_from_epoch": C.TRAIN_DEFAULTS["select_from_epoch"],
        "min_epochs": C.TRAIN_DEFAULTS["min_epochs"],
        "best_val_score": best_score,
        "gflops": gflops,
        "params_total": n_total,
        "params_trainable": n_train,
        "train_minutes": train_minutes,
        "test_minutes": test_minutes,
        "peak_gpu_mem_gb": peak_mem_gb,
        "n_test_patches": len(test_bboxes),
        "model_load_info": getattr(getattr(model, "encoder", None), "load_info", None),
        "quick_test": quick_test,
    }
    s = test_metrics["per_class"]["Cloud Shadow"]
    print(f"[TEST] mIoU={test_metrics['mIoU']:.2f} | shadow R={s['recall']:.2f} "
          f"P={s['precision']:.2f} F1={s['f1']:.2f} IoU={s['iou']:.2f} | "
          f"train {train_minutes:.1f} mnt, test {test_minutes:.1f} mnt")

    if not quick_test:
        save_json(result, out_path)
    return result
