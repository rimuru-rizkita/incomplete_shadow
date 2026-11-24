"""
analyze_grsl.py  (desain utama, Fase 7-12)
==========================================
Statistik untuk klaim "geometri menangani anotasi shadow yang tidak lengkap".

Tiap run dievaluasi dalam beberapa "view" (masing-masing = kumpulan confusion per scene):
  all        : semua scene test (metrik standar)
  annotated  : hanya scene test yang shadow-nya dianotasi (mask berisi piksel shadow)
  geomaware  : metrik sadar-geometri (scripts/geom_filter.py): di scene tanpa anotasi shadow,
               piksel Clear di jalur bayangan awan GT diabaikan; FP di luar jalur tetap dihitung
  sparcs     : validasi eksternal SPARCS (label shadow lengkap; L8 saja; scripts/eval_sparcs.py)

Uji: t berpasangan per SEED (keluarga perbandingan per dataset x view x metrik, koreksi Holm),
plus CI 95% bootstrap per SCENE (scene sama untuk A dan B; metrik dirata-rata antar seed).

Output: analysis/grsl/{table_by_view.csv, comparisons.csv, report.md}

    python analyze_grsl.py               # n_boot 2000
    python analyze_grsl.py --n_boot 0    # cepat, tanpa bootstrap
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from src import config as C  # noqa: E402
from src.experiment import RunConfig  # noqa: E402
from src.metrics import metrics_from_confusion  # noqa: E402
from eval_sparcs import metrics3  # noqa: E402

VIEWS = ("all", "annotated", "geomaware", "sparcs")
METRICS = ("shadow_iou", "shadow_f1", "shadow_precision", "shadow_recall", "mIoU")


def key_of(loss="wce", **kw):
    return RunConfig(seed=0, loss=loss, **kw).key()


def metric_fn(view):
    m = metrics3 if view == "sparcs" else metrics_from_confusion

    def f(cm):
        r = m(cm)
        s = r["per_class"]["Cloud Shadow"]
        return {"shadow_iou": s["iou"], "shadow_f1": s["f1"], "shadow_precision": s["precision"],
                "shadow_recall": s["recall"], "mIoU": r["mIoU3" if view == "sparcs" else "mIoU"]}
    return f


def load_views(runs_dir: Path):
    """{(key, seed): {view: {scene: cm}}} dan metadata konfigurasi per key."""
    out, meta = {}, {}
    for ds in C.DATASETS:
        d = runs_dir / ds
        for f in sorted(d.glob("*__loss-*.json")):
            r = json.loads(f.read_text())
            cfg = r["config"]
            if r.get("quick_test") or "per_scene_confusion" not in r:
                continue
            k = (r["key"], cfg["seed"])
            meta[r["key"]] = {c: cfg[c] for c in ("dataset", "sampling", "p", "arch", "pretrained", "partial",
                                                      "cbam", "loss")}
            pc = {s: np.asarray(v) for s, v in r["per_scene_confusion"].items()}
            v = {"all": pc, "annotated": {s: cm for s, cm in pc.items()
                                          if cm[C.SHADOW].sum() >= C.SHADOW_ANNOT_MIN_PX}}
            g = d / "geom_filter" / f.name
            if g.exists():
                gj = json.loads(g.read_text())
                if "geomaware" in gj["summary"]["all"]:
                    v["geomaware"] = {s: np.asarray(x["geomaware"]) for s, x in gj["per_scene"].items()}
            sp = d / "sparcs" / f.name
            if sp.exists():
                sj = json.loads(sp.read_text())
                ex = set(sj["excluded_scenes"])
                v["sparcs"] = {s: np.asarray(x) for s, x in sj["per_scene_confusion"].items() if s not in ex}
            out[k] = v
    return out, meta


def label_of(m):
    arch = {"unet": "U-Net", "resnet18": "ResNet18-SSL4EO", "deeplabv3plus": "DLV3+-IN1k",
            "segformer": "SegFormer-B2-IN1k"}[m["arch"]]
    if m.get("cbam"):
        arch += "+CBAM"
    samp = {"random": "random", "nonempty": "non-empty", "oversample": "oversample",
            "class_aware": "class-aware"}[m["sampling"]]
    if m["sampling"] == "oversample" and m["p"] != 0.6:  # Fase 9; label tanpa p = p 0.6
        samp += f" p={m['p']:g}"
    pl = {"none": "", "all": " + partial-all", "geom": " + partial-geom"}[m["partial"]]
    loss = "" if m.get("loss", "wce") == "wce" else f" | {m['loss']}"
    return f"{arch} | {samp}{pl}{loss}"


def table(views, meta):
    rows = []
    for (key, seed), v in views.items():
        for view, pcs in v.items():
            rows.append({"key": key, "seed": seed, "view": view, "n_scenes": len(pcs),
                         **meta[key], **metric_fn(view)(sum(pcs.values()))})
    return pd.DataFrame(rows)


def families(ds):
    pre = {"resnet18": dict(arch="resnet18", pretrained=True),
           "deeplabv3plus": dict(arch="deeplabv3plus", pretrained=True),
           "segformer": dict(arch="segformer", pretrained=True)}
    k = lambda **kw: key_of(dataset=ds, **kw)  # noqa: E731
    fam = {"F1 partial label": [], "F2 sampling": [], "F3 backbone": [], "F4 oversampling p": [],
           "F5 CBAM x loss": [], "F6 imbalance baselines": []}
    for samp in ("random", "oversample"):
        base = k(sampling=samp)
        fam["F5 CBAM x loss"] += [
            (f"{samp}: +CBAM vs base", k(sampling=samp, cbam=True), base),
            (f"{samp}: +hybrid vs base", k(sampling=samp, loss="hybrid"), base),
            (f"{samp}: +CBAM+hybrid vs base", k(sampling=samp, cbam=True, loss="hybrid"), base)]
    fam["F5 CBAM x loss"].append(("oversample vs random (CBAM+hybrid)",
                                  k(sampling="oversample", cbam=True, loss="hybrid"),
                                  k(sampling="random", cbam=True, loss="hybrid")))
    rnd = k(sampling="random")
    for name, kb in (("CB loss", k(sampling="random", loss="cb")),
                     ("OHEM", k(sampling="random", loss="ohem")),
                     ("weighted OHEM", k(sampling="random", loss="ohem_w")),
                     ("class-aware", k(sampling="class_aware"))):
        fam["F6 imbalance baselines"] += [(f"{name} vs random WCE", kb, rnd),
                                          (f"oversample vs {name}", k(sampling="oversample"), kb)]
    for p in (0.2, 0.4, 0.8):
        fam["F4 oversampling p"].append((f"oversample p={p:g} vs p=0.6", k(sampling="oversample", p=p),
                                         k(sampling="oversample")))
    fam["F4 oversampling p"].append(("oversample p=0.2 vs non-empty", k(sampling="oversample", p=0.2),
                                     k(sampling="nonempty")))
    for samp in ("nonempty", "oversample"):
        n, a, g = (k(sampling=samp, partial=p) for p in ("none", "all", "geom"))
        fam["F1 partial label"] += [(f"U-Net {samp}: geom vs none", g, n),
                                    (f"U-Net {samp}: all vs none", a, n),
                                    (f"U-Net {samp}: geom vs all", g, a)]
    for name, kw in pre.items():
        fam["F1 partial label"].append((f"{name}: geom vs none", k(sampling="nonempty", partial="geom", **kw),
                                        k(sampling="nonempty", **kw)))
    fam["F2 sampling"] = [("non-empty vs random", k(sampling="nonempty"), k(sampling="random")),
                          ("oversample vs non-empty", k(sampling="oversample"), k(sampling="nonempty")),
                          ("oversample vs random", k(sampling="oversample"), k(sampling="random"))]
    # Fase 12: hanya SegFormer yang punya run oversample di desain utama (backbone lain hanya non-empty)
    sf = pre["segformer"]
    for pl in ("none", "geom"):
        fam["F2 sampling"].append((f"segformer: oversample vs non-empty (pl-{pl})",
                                   k(sampling="oversample", partial=pl, **sf),
                                   k(sampling="nonempty", partial=pl, **sf)))
    fam["F1 partial label"].append(("segformer oversample: geom vs none",
                                    k(sampling="oversample", partial="geom", **sf),
                                    k(sampling="oversample", **sf)))
    for name, kw in pre.items():
        for pl in ("none", "geom"):
            fam["F3 backbone"].append((f"{name} vs U-Net (non-empty, pl-{pl})",
                                       k(sampling="nonempty", partial=pl, **kw), k(sampling="nonempty", partial=pl)))
    return fam


def paired(df, ka, kb, view, metric):
    a = df[(df.key == ka) & (df.view == view)][["seed", metric]].rename(columns={metric: "a"})
    b = df[(df.key == kb) & (df.view == view)][["seed", metric]].rename(columns={metric: "b"})
    m = a.merge(b, on="seed")
    n = len(m)
    res = {"n_pairs": n, "mean_a": m.a.mean() if n else np.nan, "mean_b": m.b.mean() if n else np.nan}
    if n < 2:
        return {**res, "diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "dz": np.nan}
    d = (m.a - m.b).to_numpy()
    sd = d.std(ddof=1)
    p = stats.ttest_rel(m.a, m.b).pvalue
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return {**res, "diff": d.mean(), "ci_low": d.mean() - half, "ci_high": d.mean() + half,
            "p": p, "dz": d.mean() / sd if sd > 0 else np.nan}


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(order) - r) * p[i]))
        out[i] = run
    return out


def boot(views, ka, kb, view, metric, n_boot, rng):
    seeds = sorted({s for (k, s) in views if k == ka} & {s for (k, s) in views if k == kb})
    ra = [views[(ka, s)].get(view) for s in seeds]
    rb = [views[(kb, s)].get(view) for s in seeds]
    if not seeds or any(x is None for x in ra + rb):
        return np.nan, np.nan
    scenes = sorted(set.intersection(*[set(x) for x in ra + rb]))
    if len(scenes) < 3:
        return np.nan, np.nan
    A = np.stack([[x[s] for s in scenes] for x in ra]).astype(np.float64)
    B = np.stack([[x[s] for s in scenes] for x in rb]).astype(np.float64)
    f = metric_fn(view)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(scenes), len(scenes))
        diffs[i] = (np.mean([f(A[j, idx].sum(0))[metric] for j in range(len(seeds))])
                    - np.mean([f(B[j, idx].sum(0))[metric] for j in range(len(seeds))]))
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--boot_metrics", nargs="+", default=["shadow_iou", "mIoU"])
    args = ap.parse_args()
    out = C.ANALYSIS_DIR / "grsl"
    out.mkdir(parents=True, exist_ok=True)
    views, meta = load_views(C.RUNS_DIR)
    df = table(views, meta)
    df["label"] = df.key.map(lambda k: label_of(meta[k]))
    df.sort_values(["dataset", "label", "view", "seed"]).to_csv(out / "per_seed.csv", index=False)

    g = df.groupby(["dataset", "label", "view"])
    summ = g[list(METRICS)].agg(["mean", lambda x: x.std(ddof=1)])
    summ.columns = [f"{m}_{'mean' if s == 'mean' else 'sd'}" for m, s in summ.columns]
    summ.insert(0, "n_seeds", g.size())
    summ.insert(1, "n_scenes", g.n_scenes.first())
    summ.reset_index().to_csv(out / "table_by_view.csv", index=False)

    rng = np.random.default_rng(0)
    rows = []
    for ds in C.DATASETS:
        for fam, comps in families(ds).items():
            comps = [c for c in comps if (df.key == c[1]).any() and (df.key == c[2]).any()]
            for view in VIEWS:
                for metric in METRICS:
                    block = []
                    for label, ka, kb in comps:
                        r = paired(df, ka, kb, view, metric)
                        if not r["n_pairs"]:
                            continue
                        lo, hi = (boot(views, ka, kb, view, metric, args.n_boot, rng)
                                  if args.n_boot and metric in args.boot_metrics else (np.nan, np.nan))
                        block.append({"dataset": ds, "family": fam, "comparison": label, "view": view,
                                      "metric": metric, **r, "boot_low": lo, "boot_high": hi})
                    for b, pa in zip(block, holm([b["p"] for b in block])):
                        b["p_holm"] = pa
                    rows += block
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "comparisons.csv", index=False)

    rep = ["# Analisis utama (Fase 7-12)\n",
           "mean ± SD antar seed. diff = A − B, t berpasangan per seed, p_holm per keluarga × view × "
           "metrik × dataset; boot = CI 95% bootstrap per scene.\n"]
    for ds in C.DATASETS:
        t = summ.reset_index()
        t = t[t.dataset == ds]
        for metric in ("shadow_iou", "mIoU"):
            piv = t.assign(v=[f"{m:.2f} ± {s:.2f}" if np.isfinite(s) else f"{m:.2f}"
                              for m, s in zip(t[f"{metric}_mean"], t[f"{metric}_sd"])])
            piv = piv.pivot_table(index=["label"], columns="view", values="v", aggfunc="first")
            n = t.groupby("label").n_seeds.max()
            piv.insert(0, "seeds", n)
            rep += [f"\n## {ds} — {metric}\n", piv.reset_index().to_markdown(index=False)]
        c = comp[(comp.dataset == ds) & (comp.metric == "shadow_iou")]
        if len(c):
            rep += [f"\n### Uji berpasangan {ds} — shadow IoU\n",
                    c[["family", "comparison", "view", "n_pairs", "mean_a", "mean_b", "diff", "p",
                       "p_holm", "boot_low", "boot_high"]].to_markdown(index=False, floatfmt=".3f")]
    (out / "report.md").write_text("\n".join(rep), encoding="utf-8")
    print(f"{len(df)} baris, {df.key.nunique()} konfigurasi -> {out}/report.md")


if __name__ == "__main__":
    main()
