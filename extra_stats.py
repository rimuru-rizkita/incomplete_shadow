"""
extra_stats.py
==============
Uji tambahan yang memakai confusion per scene yang sudah ada (tanpa training baru):

  1. Pembalikan peringkat (Exp. 1): Kendall tau antara peringkat lima strategi pada view yang berbeda
     dan probabilitas pembalikan di bawah bootstrap scene (scene tiap view di-resample terpisah).
  2. Ekuivalensi (Exp. 2): CI bootstrap 90% (setara TOST alfa 0.05) untuk efek CBAM / hybrid / CBAM+hybrid,
     margin ekuivalensi terkecil yang masih lolos, dan P(|efek kontrol| < efek sampler) dengan
     bootstrap scene bersama.
  3. Uji permutasi berpasangan per scene (sign-flip): label A/B ditukar per scene (sama untuk semua seed),
     statistik = selisih shadow IoU gabungan (rata-rata antar seed). Holm atas semua perbandingan.

Shadow IoU dihitung tervektorisasi dari confusion dan dicek sama dengan analyze_grsl.metric_fn.
Output: analysis/grsl/extra_rank.csv, extra_equiv.csv, extra_perm.csv

    python extra_stats.py            # 2000 bootstrap, 10000 permutasi
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyze_grsl import holm, key_of, load_views, metric_fn
from src import config as C

OUT = Path(__file__).parent / "analysis" / "grsl"
SEEDS3 = [42, 123, 2024]
S = C.SHADOW
GT_EVAL = [C.SHADOW, C.CLEAR, C.THIN, C.CLOUD]


def shadow_iou(cm, sparcs):
    """cm: (..., 5, 5) baris GT, kolom prediksi. Sama dengan metrics_from_confusion / metrics3."""
    tp = cm[..., S, S]
    col = cm[..., GT_EVAL, S].sum(-1) if sparcs else cm[..., :, S].sum(-1)
    fn = cm[..., S, :].sum(-1) - tp
    return 100 * tp / (tp + (col - tp) + fn + 1e-12)


def stack(views, key, view, seeds, scenes=None):
    """Array (n_seed, n_scene, 5, 5) untuk scene bersama."""
    rs = [views[(key, s)][view] for s in seeds]
    if scenes is None:
        scenes = sorted(set.intersection(*[set(r) for r in rs]))
    return np.stack([[r[sc] for sc in scenes] for r in rs]).astype(np.float64), scenes


def common(views, keys, view, seeds):
    sets = [set(views[(k, s)][view]) for k in keys for s in seeds]
    return sorted(set.intersection(*sets))


def iou_pooled(X, idx, sparcs):
    """Rata-rata antar seed dari shadow IoU gabungan atas scene idx (boleh berulang)."""
    return shadow_iou(X[:, idx].sum(1), sparcs).mean()


# ----------------------------------------------------------------------------- 1. peringkat
EXP1 = [("random", dict(sampling="random")), ("CB loss", dict(sampling="random", loss="cb")),
        ("non-empty", dict(sampling="nonempty")), ("class-aware", dict(sampling="class_aware")),
        ("oversampling", dict(sampling="oversample"))]


def rank_test(views, n_boot, rng):
    rows, dist = [], {}
    pairs = [("l8biome", "all", "sparcs"), ("l8biome", "annotated", "sparcs"), ("l8biome", "all", "annotated"),
             ("l7irish", "all", "annotated"), ("l7irish", "all", "geomaware")]
    cache = {}
    for ds in ("l8biome", "l7irish"):
        keys = [key_of(dataset=ds, **kw) for _, kw in EXP1]
        for view in ("all", "annotated", "geomaware", "sparcs"):
            try:
                sc = common(views, keys, view, SEEDS3)
            except KeyError:
                continue
            X = [stack(views, k, view, SEEDS3, sc)[0] for k in keys]
            cache[(ds, view)] = (X, len(sc))
    names = [n for n, _ in EXP1]
    ios = names.index("oversampling")
    for ds, v1, v2 in pairs:
        (X1, n1), (X2, n2) = cache[(ds, v1)], cache[(ds, v2)]
        sp1, sp2 = v1 == "sparcs", v2 == "sparcs"
        a = np.array([iou_pooled(x, np.arange(n1), sp1) for x in X1])
        b = np.array([iou_pooled(x, np.arange(n2), sp2) for x in X2])
        tau = stats.kendalltau(a, b).statistic
        taus, os_rev, low_high = [], [], []
        for _ in range(n_boot):
            i1, i2 = rng.integers(0, n1, n1), rng.integers(0, n2, n2)
            aa = np.array([iou_pooled(x, i1, sp1) for x in X1])
            bb = np.array([iou_pooled(x, i2, sp2) for x in X2])
            taus.append(stats.kendalltau(aa, bb).statistic)
            r1, r2 = (-aa).argsort().argsort(), (-bb).argsort().argsort()  # 0 = terbaik
            os_rev.append(r1[ios] > r2[ios])
            low_high.append(r1[ios] == len(names) - 1 and r2[ios] == 0)
        taus = np.array(taus)
        rows.append(dict(dataset=ds, view_a=v1, view_b=v2, n_scenes_a=n1, n_scenes_b=n2,
                         ranks_a=" > ".join(np.array(names)[(-a).argsort()]),
                         ranks_b=" > ".join(np.array(names)[(-b).argsort()]),
                         tau=tau, tau_low=np.percentile(taus, 2.5), tau_high=np.percentile(taus, 97.5),
                         p_tau_neg=(taus < 0).mean(), p_os_better_in_b=np.mean(os_rev),
                         p_os_last_a_first_b=np.mean(low_high)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 2. ekuivalensi
def equiv_test(views, n_boot, rng):
    rows = []
    for ds, view in (("l8biome", "sparcs"), ("l7irish", "annotated"), ("l7irish", "geomaware")):
        sp = view == "sparcs"
        k = lambda **kw: key_of(dataset=ds, **kw)  # noqa: E731
        ctrl = {f"{s}: {name}": (k(sampling=s, **kw), k(sampling=s))
                for s in ("random", "oversample")
                for name, kw in (("+CBAM", dict(cbam=True)), ("+hybrid", dict(loss="hybrid")),
                                 ("+CBAM+hybrid", dict(cbam=True, loss="hybrid")))}
        samp = (k(sampling="oversample"), k(sampling="random"))
        allkeys = sorted({x for p in list(ctrl.values()) + [samp] for x in p})
        sc = common(views, allkeys, view, SEEDS3)
        X = {kk: stack(views, kk, view, SEEDS3, sc)[0] for kk in allkeys}
        n = len(sc)
        full = np.arange(n)
        eff = lambda pair, idx: iou_pooled(X[pair[0]], idx, sp) - iou_pooled(X[pair[1]], idx, sp)  # noqa: E731
        boots = {name: [] for name in ctrl}
        boots["sampler"] = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            boots["sampler"].append(eff(samp, idx))
            for name, pair in ctrl.items():
                boots[name].append(eff(pair, idx))
        sb = np.array(boots["sampler"])
        for name, pair in ctrl.items():
            bb = np.array(boots[name])
            lo90, hi90 = np.percentile(bb, [5, 95])
            rows.append(dict(dataset=ds, view=view, n_scenes=n, effect=name, diff=eff(pair, full),
                             ci90_low=lo90, ci90_high=hi90, min_equiv_margin=max(abs(lo90), abs(hi90)),
                             equiv_at_5pp=(lo90 > -5) and (hi90 < 5),
                             sampler_effect=eff(samp, full), p_smaller_than_sampler=np.mean(np.abs(bb) < sb)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 3. permutasi
def perm_pairs():
    out = []
    for ds in ("l8biome", "l7irish"):
        k = lambda **kw: key_of(dataset=ds, **kw)  # noqa: E731
        views_main = ["sparcs", "all"] if ds == "l8biome" else ["annotated", "geomaware"]
        for v in views_main:
            out += [(ds, v, "U-Net: oversample vs random", k(sampling="oversample"), k(sampling="random")),
                    (ds, v, "U-Net: oversample vs non-empty", k(sampling="oversample"), k(sampling="nonempty")),
                    (ds, v, "U-Net: oversample vs CB loss", k(sampling="oversample"), k(sampling="random", loss="cb"))]
        for v in (["sparcs"] if ds == "l8biome" else ["annotated", "geomaware"]):
            out += [(ds, v, "U-Net NE: geom vs none", k(sampling="nonempty", partial="geom"), k(sampling="nonempty")),
                    (ds, v, "U-Net OS: geom vs none", k(sampling="oversample", partial="geom"), k(sampling="oversample"))]
            for name, kw in (("R18-SSL4EO", dict(arch="resnet18", pretrained=True)),
                             ("DLV3+", dict(arch="deeplabv3plus", pretrained=True)),
                             ("SegFormer", dict(arch="segformer", pretrained=True))):
                out.append((ds, v, f"{name} NE: geom vs none", k(sampling="nonempty", partial="geom", **kw),
                            k(sampling="nonempty", **kw)))
            sf = dict(arch="segformer", pretrained=True)
            out.append((ds, v, "SegFormer: oversample vs non-empty", k(sampling="oversample", **sf),
                        k(sampling="nonempty", **sf)))
    return out


def perm_test(views, n_perm, rng):
    rows = []
    for ds, v, name, ka, kb in perm_pairs():
        seeds = sorted({s for (kk, s) in views if kk == ka} & {s for (kk, s) in views if kk == kb})
        sp = v == "sparcs"
        sc = common(views, [ka, kb], v, seeds)
        A, _ = stack(views, ka, v, seeds, sc)
        B, _ = stack(views, kb, v, seeds, sc)
        n = len(sc)
        obs = iou_pooled(A, np.arange(n), sp) - iou_pooled(B, np.arange(n), sp)
        sumA, sumB = A.sum(1), B.sum(1)  # (seed, 5, 5)
        D = (B - A).reshape(len(seeds), n, -1)  # tukar scene -> A dapat B
        cnt, done, chunk = 0, 0, 2000
        while done < n_perm:
            m = min(chunk, n_perm - done)
            F = rng.integers(0, 2, (m, n)).astype(np.float64)  # 1 = tukar
            dA = np.einsum("pn,snk->psk", F, D).reshape(m, len(seeds), 5, 5)
            pa = shadow_iou(sumA[None] + dA, sp).mean(1)
            pb = shadow_iou(sumB[None] - dA, sp).mean(1)
            cnt += np.sum(np.abs(pa - pb) >= abs(obs) - 1e-9)
            done += m
        rows.append(dict(dataset=ds, view=v, comparison=name, n_seeds=len(seeds), n_scenes=n, diff=obs,
                         p_perm=(cnt + 1) / (n_perm + 1)))
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df.p_perm.values)
    return df


def check_iou(views):
    """shadow_iou tervektorisasi == metric_fn untuk semua run dan view."""
    worst = 0.0
    for (key, seed), v in views.items():
        for view, pcs in v.items():
            cm = sum(pcs.values())
            worst = max(worst, abs(shadow_iou(np.asarray(cm, float), view == "sparcs")
                                   - metric_fn(view)(cm)["shadow_iou"]))
    assert worst < 1e-6, worst
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_perm", type=int, default=10000)
    args = ap.parse_args()
    views, _ = load_views(C.RUNS_DIR)
    print("cek shadow IoU vs metric_fn, selisih maks:", check_iou(views))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    r = rank_test(views, args.n_boot, np.random.default_rng(1))
    r.to_csv(OUT / "extra_rank.csv", index=False)
    print("\n== 1. Peringkat\n", r.round(3).to_string(index=False))
    e = equiv_test(views, args.n_boot, np.random.default_rng(2))
    e.to_csv(OUT / "extra_equiv.csv", index=False)
    print("\n== 2. Ekuivalensi\n", e.round(3).to_string(index=False))
    p = perm_test(views, args.n_perm, np.random.default_rng(3))
    p.to_csv(OUT / "extra_perm.csv", index=False)
    print("\n== 3. Permutasi\n", p.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
