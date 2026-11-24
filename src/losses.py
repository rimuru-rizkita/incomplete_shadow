"""
losses.py
=========
Loss yang dipakai di semua fase. Fill (kelas 0) selalu di-ignore.

  ce      : cross-entropy tanpa bobot
  wce     : cross-entropy dengan bobot hand-picked versi lama [0,5,1,1.5,1]
            (= baseline versi lama)
  cb      : class-balanced CE, bobot "effective number" (Cui et al., CVPR 2019)
            dihitung dari jumlah piksel kelas di scene training (fase 0)
  ohem    : online hard example mining, CE pada top-k% piksel tersulit
  ohem_w  : OHEM dengan bobot kelas WCE (setara WCE kecuali seleksi piksel tersulit)
  hybrid  : 0.5 * Focal + 0.5 * Tversky (metode versi lama), dengan perbaikan:
              - pt dihitung dari CE TANPA bobot (versi lama: pt = exp(-w*CE)
                sehingga faktor fokus terdistorsi untuk kelas berbobot)
              - rata-rata hanya atas piksel valid (versi lama ikut membagi
                dengan piksel Fill)

Hyperparameter yang harus dilaporkan di paper: gamma=2.0, alpha_T=0.3,
beta_T=0.7, smoothing eps=1e-6, lambda1=lambda2=0.5, OHEM keep=25%,
CB beta=1-1/n_max (lihat class_balanced_weights).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


class WeightedCE(nn.Module):
    """
    (Weighted) CE. Piksel berlabel C.AMBIG ("Clear atau Shadow", dari mask partial label)
    memakai marginal loss: -log(p_clear + p_shadow), berbobot seperti Clear
    (Shi et al. 2021; Francis 2024). Tanpa piksel AMBIG hasilnya identik dengan F.cross_entropy.
    """

    def __init__(self, weight=None):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits, targets):
        ambig = targets == C.AMBIG
        if not ambig.any():
            return F.cross_entropy(logits, targets, weight=self.weight, ignore_index=C.FILL)
        logp = F.log_softmax(logits, dim=1)
        nll = -logp.gather(1, targets.masked_fill(ambig, C.CLEAR).unsqueeze(1)).squeeze(1)
        marginal = -torch.logsumexp(logp[:, [C.CLEAR, C.SHADOW]], dim=1)
        nll = torch.where(ambig, marginal, nll)
        w = (self.weight[targets.masked_fill(ambig, C.CLEAR)] if self.weight is not None
             else torch.ones_like(nll))
        w = w * (targets != C.FILL)
        return (nll * w).sum() / w.sum().clamp_min(1e-12)


class OHEMLoss(nn.Module):
    """CE per piksel, ambil keep_ratio piksel valid dengan loss terbesar."""

    def __init__(self, keep_ratio: float = 0.25, min_kept: int = 10000, weight=None):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.min_kept = min_kept
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits, targets):
        loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none", ignore_index=C.FILL)
        loss = loss[targets != C.FILL]
        if loss.numel() == 0:
            return logits.sum() * 0.0
        k = max(int(loss.numel() * self.keep_ratio), min(self.min_kept, loss.numel()))
        return torch.topk(loss, k, sorted=False).values.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits, targets):
        valid = targets != C.FILL
        if valid.sum() == 0:
            return logits.sum() * 0.0
        ce = F.cross_entropy(logits, targets, reduction="none", ignore_index=C.FILL)
        pt = torch.exp(-ce)  # probabilitas kelas benar (tanpa bobot)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss[valid].mean()


class TverskyLoss(nn.Module):
    """Tversky multi-kelas, dihitung atas seluruh batch (sama seperti versi lama)."""

    def __init__(self, alpha=0.3, beta=0.7, eps=1e-6):
        super().__init__()
        self.alpha, self.beta, self.eps = alpha, beta, eps

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        valid = (targets != C.FILL).unsqueeze(1).float()
        onehot = F.one_hot(targets.clamp_min(0), C.NUM_CLASSES).permute(0, 3, 1, 2).float()
        probs, onehot = probs * valid, onehot * valid
        dims = (0, 2, 3)
        tp = (probs * onehot).sum(dims)
        fp = (probs * (1 - onehot)).sum(dims)
        fn = ((1 - probs) * onehot).sum(dims)
        ti = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return (1 - ti)[1:].mean()  # skip Fill


class HybridLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, t_alpha=0.3, t_beta=0.7, lam1=0.5, lam2=0.5):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.tversky = TverskyLoss(alpha=t_alpha, beta=t_beta)
        self.lam1, self.lam2 = lam1, lam2

    def forward(self, logits, targets):
        return self.lam1 * self.focal(logits, targets) + self.lam2 * self.tversky(logits, targets)


def class_balanced_weights(pixel_counts: list, beta: float | None = None) -> torch.Tensor:
    """
    Bobot effective-number (Cui et al. 2019): w_c = (1-beta) / (1-beta^n_c),
    dinormalisasi rata-rata 1 di kelas valid.

    Catatan: untuk hitungan PIKSEL (ratusan juta), beta=0.9999 membuat semua
    bobot hampir sama (effective number jenuh di 1/(1-beta)=10^4). Karena itu
    default beta = 1 - 1/n_max (n_max = jumlah piksel kelas terbanyak).
    Laporkan nilai beta ini di paper.
    """
    counts = torch.tensor(pixel_counts, dtype=torch.float64)
    valid = [c for c in range(C.NUM_CLASSES) if c != C.FILL]
    if beta is None:
        beta = 1.0 - 1.0 / counts[valid].max().item()
    # 1 - beta^n dihitung stabil: -expm1(n * log(beta))
    eff = -torch.expm1(counts * torch.log1p(torch.tensor(-(1.0 - beta), dtype=torch.float64)))
    w = (1.0 - beta) / eff.clamp_min(1e-300)
    w[C.FILL] = 0.0
    w[valid] = w[valid] / w[valid].mean()
    return w.float()


def build_loss(name: str, train_stats: dict | None = None) -> nn.Module:
    legacy = torch.tensor(C.LEGACY_CLASS_WEIGHTS, dtype=torch.float32)
    if name == "ce":
        return WeightedCE(None)
    if name == "wce":
        return WeightedCE(legacy)
    if name == "cb":
        if not train_stats:
            raise FileNotFoundError("Loss 'cb' butuh statistik fase 0 (pixel_counts).")
        w = class_balanced_weights(train_stats["pixel_counts"])
        print(f"[Loss] class-balanced weights: {[round(x, 4) for x in w.tolist()]}")
        return WeightedCE(w)
    if name == "ohem":
        return OHEMLoss()
    if name == "ohem_w":
        return OHEMLoss(weight=legacy)
    if name == "hybrid":
        return HybridLoss(alpha=legacy)
    raise ValueError(f"Loss tidak dikenal: {name}")
