"""
phases.py
=========
Definisi semua run per fase. Setiap fase mengembalikan list RunConfig.
Konfigurasi identik antar fase (mis. sel oversample p=0.6 di fase 1 & 3)
punya nama sama, sehingga hanya dijalankan sekali.

Fase 0 (bukan training) ada di scripts/phase0_*.py.

  Fase 1  Factorial 2x2x2: sampling {random, oversample} x CBAM {off, on}
          x loss {wce, hybrid}
  Fase 2  Baseline imbalance lain (U-Net, tanpa CBAM):
          loss: ce, cb, ohem (sampling random)
          sampling: ratio_weighted, class_aware, nonempty (loss wce)
  Fase 3  Sensitivitas: p, tau, jitter (L7 Irish)
  Fase 4  Backbone pretrained x sampling:
          ResNet-18 SSL4EO-L (fine-tune, frozen, scratch)
          DeepLabV3+ ResNet-50 ImageNet {random, oversample}
  Fase 6  Pilot partial label (seed 42)
  Fase 7  Desain ringkas: U-Net WCE, random + {non-empty, oversample} x partial
          {none, all, geom}, 5 seed (menggantikan sisa fase 1)
  Fase 8  Backbone pretrained (ResNet-18 SSL4EO-L, DeepLabV3+ R50 ImageNet), non-empty x
          partial {none, geom}, 3 seed
"""

from itertools import product

from . import config as C
from .experiment import RunConfig


def phase1(datasets, seeds):
    runs = []
    for ds, seed, samp, cbam, loss in product(datasets, seeds, ["random", "oversample"],
                                              [False, True], ["wce", "hybrid"]):
        runs.append(RunConfig(dataset=ds, seed=seed, sampling=samp, cbam=cbam, loss=loss))
    return runs


def phase2(datasets, seeds):
    runs = []
    for ds, seed in product(datasets, seeds):
        for loss in ["ce", "cb", "ohem"]:
            runs.append(RunConfig(dataset=ds, seed=seed, sampling="random", loss=loss))
        runs.append(RunConfig(dataset=ds, seed=seed, sampling="ratio_weighted", loss="wce"))
        runs.append(RunConfig(dataset=ds, seed=seed, sampling="class_aware", loss="wce"))
        runs.append(RunConfig(dataset=ds, seed=seed, sampling="nonempty", loss="wce"))
    return runs


def phase3(datasets, seeds):
    runs = []
    for ds, seed in product(datasets, seeds):
        for p in [0.2, 0.4, 0.6, 0.8, 1.0]:
            runs.append(RunConfig(dataset=ds, seed=seed, sampling="oversample", p=p))
        for tau in [0.001, 0.01, 0.05]:
            runs.append(RunConfig(dataset=ds, seed=seed, sampling="oversample", tau=tau))
        runs.append(RunConfig(dataset=ds, seed=seed, sampling="oversample", jitter=0.25))
    return runs


def phase4(datasets, seeds):
    runs = []
    for ds, seed in product(datasets, seeds):
        for samp in ["random", "oversample"]:
            runs.append(RunConfig(dataset=ds, seed=seed, sampling=samp, arch="resnet18",
                                  pretrained=True))
        runs.append(RunConfig(dataset=ds, seed=seed, sampling="oversample", arch="resnet18",
                              pretrained=True, freeze_encoder=True))
        runs.append(RunConfig(dataset=ds, seed=seed, sampling="oversample", arch="resnet18",
                              pretrained=False))
        for samp in ["random", "oversample"]:
            runs.append(RunConfig(dataset=ds, seed=seed, sampling=samp, arch="deeplabv3plus",
                                  pretrained=True))
    return runs


def phase6(datasets, seeds):
    """Pilot partial label: mask asli vs semua-Clear-ambigu (Francis 2024) vs dibatasi geometri."""
    runs = []
    for ds, seed in product(datasets, seeds):
        for samp in ["nonempty", "oversample"]:
            for partial in ["none", "all", "geom"]:
                runs.append(RunConfig(dataset=ds, seed=seed, sampling=samp, loss="wce",
                                      partial=partial))
    return runs


def phase7(datasets, seeds):
    """
    Desain ringkas (2026-11-03): U-Net + WCE saja; baseline random dan partial label
    {none, all, geom} x sampling {non-empty, oversample}. Urutan seed-major supaya setiap seed
    lengkap untuk kedua dataset sebelum seed berikutnya. Nama run sama dengan fase 1/6.
    """
    runs = []
    for seed, ds in product(seeds, datasets):
        runs.append(RunConfig(dataset=ds, seed=seed, sampling="random", loss="wce"))
        for samp in ["nonempty", "oversample"]:
            for partial in ["none", "all", "geom"]:
                runs.append(RunConfig(dataset=ds, seed=seed, sampling=samp, loss="wce",
                                      partial=partial))
    return runs


def phase8(datasets, seeds):
    """
    Backbone pretrained untuk desain utama: ResNet-18 SSL4EO-L MoCo (+ decoder U-Net) dan
    DeepLabV3+ ResNet-50 ImageNet, sampling non-empty, partial label {none, geom}, loss WCE.
    """
    runs = []
    for seed, ds in product(seeds, datasets):
        for arch in ["resnet18", "deeplabv3plus"]:
            for partial in ["none", "geom"]:
                runs.append(RunConfig(dataset=ds, seed=seed, sampling="nonempty", loss="wce",
                                      arch=arch, pretrained=True, partial=partial))
    return runs


def phase9(datasets, seeds):
    """
    Sensitivitas p oversampling untuk desain utama: U-Net WCE, tanpa partial label,
    p in {0.2, 0.4, 0.8}; p = 0.6 sudah ada dari Fase 7. Nama run sama dengan Fase 3.
    """
    runs = []
    for seed, ds in product(seeds, datasets):
        for p in [0.2, 0.4, 0.8]:
            runs.append(RunConfig(dataset=ds, seed=seed, sampling="oversample", p=p, loss="wce"))
    return runs


def phase10(datasets, seeds):
    """
    Kontrol arsitektur/loss untuk desain utama: faktorial 2x2x2 sampling
    {random, oversample} x CBAM {off, on} x loss {wce, hybrid}, ditambah baseline imbalance
    lain (CB loss, OHEM dengan sampling random; class-aware sampling dengan WCE). Sel WCE tanpa
    CBAM sudah ada dari Fase 7 dan sel seed 42 dari Fase 1, sehingga dilewati otomatis.
    """
    runs = []
    for seed, ds in product(seeds, datasets):
        for samp, cbam, loss in product(["random", "oversample"], [False, True], ["wce", "hybrid"]):
            if not cbam and loss == "wce":
                continue
            runs.append(RunConfig(dataset=ds, seed=seed, sampling=samp, cbam=cbam, loss=loss))
        for loss in ["cb", "ohem"]:
            runs.append(RunConfig(dataset=ds, seed=seed, sampling="random", loss=loss))
        runs.append(RunConfig(dataset=ds, seed=seed, sampling="class_aware", loss="wce"))
    return runs


def phase11(datasets, seeds):
    """
    OHEM berbobot kelas WCE (2026-11-07). OHEM Fase 10 memakai CE tanpa bobot dan kolaps (shadow
    hampir tidak pernah diprediksi); ohem_w hanya berbeda dari random+WCE pada seleksi piksel tersulit.
    """
    return [RunConfig(dataset=ds, seed=seed, sampling="random", loss="ohem_w")
            for seed, ds in product(seeds, datasets)]


def phase12(datasets, seeds):
    """
    Baseline Transformer (2026-11-18): SegFormer MiT-B2 ImageNet, loss WCE,
    sampling {non-empty, oversample} x partial label {none, geom}. Menguji apakah pembalikan
    peringkat oversampling dan efek label geom juga muncul pada arsitektur Transformer.
    """
    runs = []
    # urut per dataset (L8 dulu) supaya hasil SPARCS bisa dianalisis sebelum run L7 selesai
    for ds, seed in product(datasets, seeds):
        for samp in ["nonempty", "oversample"]:
            for partial in ["none", "geom"]:
                runs.append(RunConfig(dataset=ds, seed=seed, sampling=samp, loss="wce",
                                      arch="segformer", pretrained=True, partial=partial))
    return runs


PHASES = {
    1: {"fn": phase1, "datasets": ["l7irish", "l8biome"], "seeds": C.SEEDS,
        "desc": "Factorial 2x2x2 (sampling x CBAM x loss)"},
    2: {"fn": phase2, "datasets": ["l7irish", "l8biome"], "seeds": C.SEEDS,
        "desc": "Baseline imbalance lain (CE, CB, OHEM, ratio-weighted, class-aware, non-empty)"},
    3: {"fn": phase3, "datasets": ["l7irish"], "seeds": C.SEEDS,
        "desc": "Sensitivitas p, tau, jitter"},
    4: {"fn": phase4, "datasets": ["l7irish", "l8biome"], "seeds": C.SEEDS,
        "desc": "Backbone pretrained (ResNet-18 SSL4EO-L, DeepLabV3+ ImageNet) x sampling"},
    6: {"fn": phase6, "datasets": ["l8biome", "l7irish"], "seeds": [42],
        "desc": "Pilot partial label (none / all / geom) x sampling (non-empty, oversample)"},
    7: {"fn": phase7, "datasets": ["l8biome", "l7irish"], "seeds": C.SEEDS,
        "desc": "ringkas: U-Net WCE, random + partial label x sampling, 5 seed"},
    8: {"fn": phase8, "datasets": ["l8biome", "l7irish"], "seeds": C.SEEDS[:3],
        "desc": "backbone pretrained (ResNet-18 SSL4EO-L, DeepLabV3+) x partial {none, geom}"},
    9: {"fn": phase9, "datasets": ["l8biome", "l7irish"], "seeds": C.SEEDS[:3],
        "desc": "sensitivitas p oversampling (0.2, 0.4, 0.8), U-Net WCE, 3 seed"},
    10: {"fn": phase10, "datasets": ["l8biome", "l7irish"], "seeds": C.SEEDS[:3],
         "desc": "faktorial CBAM x loss x sampling + CB, OHEM, class-aware, 3 seed"},
    11: {"fn": phase11, "datasets": ["l8biome", "l7irish"], "seeds": C.SEEDS[:3],
         "desc": "OHEM berbobot kelas WCE (random), 3 seed"},
    12: {"fn": phase12, "datasets": ["l8biome", "l7irish"], "seeds": C.SEEDS[:3],
         "desc": "SegFormer MiT-B2 ImageNet x sampling {non-empty, oversample} x partial {none, geom}, 3 seed"},
}


def get_runs(phase: int, datasets=None, seeds=None):
    spec = PHASES[phase]
    return spec["fn"](datasets or spec["datasets"], seeds or spec["seeds"])
