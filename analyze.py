"""
analyze.py  (FASE 5a)
=====================
Membaca semua hasil di runs/ lalu menghasilkan tabel & statistik untuk paper:

  analysis/runs_raw.csv              semua run per seed (transparansi; mis. pola 0/0/15.6)
  analysis/summary_by_config.csv     mean ± SD (ddof=1) semua metrik, SEMUA kelas
  analysis/phase{1..4}_table.md      tabel per fase
  analysis/comparisons.csv           uji berpasangan (dipasangkan per SEED), koreksi Holm,
                                     + CI bootstrap per scene test
  analysis/anova_phase1.csv          ANOVA 3 arah + partial eta^2 (factorial)
  analysis/error_analysis.csv        asal false positive & tujuan false negative shadow
  analysis/cost_table.csv            parameter, waktu train/test, epoch terbaik
  analysis/report.md                 gabungan semuanya

    python analyze.py
    python analyze.py --n_boot 5000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import config as CONF
from src.experiment import RunConfig
from src.metrics import flat_metrics, metrics_from_confusion

KEY_METRICS = ["shadow_recall", "shadow_precision", "shadow_f1", "shadow_iou", "mIoU"]
CONFIG_COLS = ["dataset", "sampling", "p", "tau", "jitter", "arch", "cbam", "loss",
               "pretrained", "freeze_encoder", "augment", "norm", "lr"]


# ------------------------------------------------------------------ loading
def load_results(runs_dir=CONF.RUNS_DIR):
    results = []
    for f in sorted(Path(runs_dir).glob("*/*.json")):
        with open(f) as fh:
            r = json.load(fh)
        if r.get("quick_test"):
            continue
        results.append(r)
    return results


def to_frame(results):
    rows = []
    for r in results:
        cfg = r["config"]
        row = {"key": r["key"], "seed": cfg["seed"], **{c: cfg[c] for c in CONFIG_COLS}}
        row.update(flat_metrics(r["test"]))
        row.update({"best_epoch": r["best_epoch"], "train_minutes": r["train_minutes"],
                    "test_minutes": r["test_minutes"], "params_total": r["params_total"],
                    "params_trainable": r["params_trainable"],
                    "gflops": r.get("gflops") if r.get("gflops") is not None else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def key_of(**kw):
    return RunConfig(seed=0, **kw).key()


# ------------------------------------------------------------------ summaries
def summarize(df):
    metric_cols = [c for c in df.columns if c not in CONFIG_COLS + ["key", "seed"]]
    g = df.groupby("key")
    out = g[metric_cols].agg(["mean", lambda x: x.std(ddof=1)])
    out.columns = [f"{m}_{'mean' if s == 'mean' else 'sd'}" for m, s in out.columns]
    out.insert(0, "n_seeds", g.size())
    return out


def fmt(mean, sd):
    return f"{mean:.2f} ± {sd:.2f}" if np.isfinite(sd) else f"{mean:.2f}"


def phase_table(summary, keys_labels, metrics):
    rows = []
    for key, label in keys_labels:
        if key not in summary.index:
            rows.append({"Config": label, "n": 0, **{m: "belum ada" for m in metrics}})
            continue
        s = summary.loc[key]
        rows.append({"Config": label, "n": int(s["n_seeds"]),
                     **{m: fmt(s[f"{m}_mean"], s[f"{m}_sd"]) for m in metrics}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ paired tests
def paired_test(df, key_a, key_b, metric):
    a = df[df.key == key_a][["seed", metric]].rename(columns={metric: "a"})
    b = df[df.key == key_b][["seed", metric]].rename(columns={metric: "b"})
    m = a.merge(b, on="seed")  # DIPASANGKAN PER SEED (versi lama: urutan glob)
    n = len(m)
    res = {"n_pairs": n, "mean_a": m.a.mean() if n else np.nan,
           "mean_b": m.b.mean() if n else np.nan}
    if n < 2:
        return {**res, "diff": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "t": np.nan, "p": np.nan, "dz": np.nan}
    d = (m.a - m.b).to_numpy()
    sd = d.std(ddof=1)
    t, p = stats.ttest_rel(m.a, m.b)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return {**res, "diff": d.mean(), "ci_low": d.mean() - half, "ci_high": d.mean() + half,
            "t": t, "p": p, "dz": d.mean() / sd if sd > 0 else np.nan}


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


# ------------------------------------------------------------------ scene bootstrap
METRIC_FROM_CM = {
    "shadow_recall": lambda m: m["per_class"]["Cloud Shadow"]["recall"],
    "shadow_precision": lambda m: m["per_class"]["Cloud Shadow"]["precision"],
    "shadow_f1": lambda m: m["per_class"]["Cloud Shadow"]["f1"],
    "shadow_iou": lambda m: m["per_class"]["Cloud Shadow"]["iou"],
    "mIoU": lambda m: m["mIoU"],
}


def scene_bootstrap(results_by_key, key_a, key_b, metric, n_boot, rng):
    """
    CI 95% untuk selisih metrik (A - B) dengan me-resample SCENE test.
    Tiap iterasi: scene yang sama dipakai untuk A dan B; metrik tiap seed
    dihitung dari jumlah confusion matrix scene terpilih, lalu dirata-rata antar seed.
    """
    ra, rb = results_by_key.get(key_a, []), results_by_key.get(key_b, [])
    if not ra or not rb:
        return np.nan, np.nan
    scene_sets = [set(r["per_scene_confusion"]) for r in ra + rb]
    scenes = sorted(set.intersection(*scene_sets)) if scene_sets else []
    if len(scenes) < 3:
        return np.nan, np.nan
    stack = lambda rs: np.stack([[r["per_scene_confusion"][s] for s in scenes] for r in rs])  # noqa: E731
    A, B = stack(ra), stack(rb)  # [seeds, S, 5, 5]
    f = METRIC_FROM_CM[metric]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(scenes), len(scenes))
        ma = np.mean([f(metrics_from_confusion(A[s, idx].sum(0))) for s in range(A.shape[0])])
        mb = np.mean([f(metrics_from_confusion(B[s, idx].sum(0))) for s in range(B.shape[0])])
        diffs[i] = ma - mb
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


# ------------------------------------------------------------------ comparison families
def comparison_families(ds):
    """(nama keluarga, [(label, key_a, key_b), ...]) -- koreksi Holm per keluarga & metrik."""
    fam = []
    os_vs_rand = []
    for cbam in (False, True):
        for loss in ("wce", "hybrid"):
            lbl = f"OS vs random | {'CBAM' if cbam else 'U-Net'}, {loss}"
            os_vs_rand.append((lbl, key_of(dataset=ds, sampling="oversample", cbam=cbam, loss=loss),
                               key_of(dataset=ds, sampling="random", cbam=cbam, loss=loss)))
    fam.append(("P1: efek oversampling", os_vs_rand))

    comp = []
    for samp in ("random", "oversample"):
        for loss in ("wce", "hybrid"):
            comp.append((f"CBAM vs U-Net | {samp}, {loss}",
                         key_of(dataset=ds, sampling=samp, cbam=True, loss=loss),
                         key_of(dataset=ds, sampling=samp, cbam=False, loss=loss)))
        for cbam in (False, True):
            comp.append((f"hybrid vs wce | {samp}, {'CBAM' if cbam else 'U-Net'}",
                         key_of(dataset=ds, sampling=samp, cbam=cbam, loss="hybrid"),
                         key_of(dataset=ds, sampling=samp, cbam=cbam, loss="wce")))
    fam.append(("P1: efek CBAM & loss", comp))

    ref = key_of(dataset=ds, sampling="oversample")
    base = [(f"OS(wce) vs {lbl}", ref, k) for lbl, k in [
        ("random+CE", key_of(dataset=ds, loss="ce")),
        ("random+WCE", key_of(dataset=ds, loss="wce")),
        ("random+CB", key_of(dataset=ds, loss="cb")),
        ("random+OHEM", key_of(dataset=ds, loss="ohem")),
        ("ratio-weighted", key_of(dataset=ds, sampling="ratio_weighted")),
        ("class-aware", key_of(dataset=ds, sampling="class_aware")),
        ("random non-empty", key_of(dataset=ds, sampling="nonempty")),
    ]]
    # Dekomposisi efek oversampling: menghindari Fill (non-empty vs random) dan
    # memprioritaskan shadow (OS vs non-empty, sudah ada di atas)
    base.append(("non-empty vs random+WCE", key_of(dataset=ds, sampling="nonempty"),
                 key_of(dataset=ds, loss="wce")))
    fam.append(("P2: OS vs baseline lain", base))

    sens = [(f"p={p} vs p=0.6", key_of(dataset=ds, sampling="oversample", p=p), ref)
            for p in (0.2, 0.4, 0.8, 1.0)]
    sens += [(f"tau={t} vs tau=0", key_of(dataset=ds, sampling="oversample", tau=t), ref)
             for t in (0.001, 0.01, 0.05)]
    sens += [("jitter=0.25 vs 0", key_of(dataset=ds, sampling="oversample", jitter=0.25), ref)]
    fam.append(("P3: sensitivitas", sens))

    ssl = lambda **kw: key_of(dataset=ds, arch="resnet18", **kw)  # noqa: E731
    fam.append(("P4: pretraining", [
        ("SSL+OS vs SSL+random", ssl(sampling="oversample", pretrained=True),
         ssl(sampling="random", pretrained=True)),
        ("SSL+OS vs scratch ResNet+OS", ssl(sampling="oversample", pretrained=True),
         ssl(sampling="oversample", pretrained=False)),
        ("SSL fine-tune vs frozen (OS)", ssl(sampling="oversample", pretrained=True),
         ssl(sampling="oversample", pretrained=True, freeze_encoder=True)),
        ("SSL+OS vs U-Net+OS", ssl(sampling="oversample", pretrained=True), ref),
    ]))
    dlv = lambda **kw: key_of(dataset=ds, arch="deeplabv3plus", pretrained=True, **kw)  # noqa: E731
    fam.append(("P4: backbone kompetitif", [
        ("DLV3+ OS vs DLV3+ random", dlv(sampling="oversample"), dlv(sampling="random")),
        ("DLV3+ OS vs U-Net OS", dlv(sampling="oversample"), ref),
        ("DLV3+ random vs U-Net random", dlv(sampling="random"), key_of(dataset=ds, loss="wce")),
    ]))
    return fam


def run_comparisons(df, results_by_key, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ds in sorted(df.dataset.unique()):
        for fam_name, comps in comparison_families(ds):
            comps = [c for c in comps if (df.key == c[1]).any() and (df.key == c[2]).any()]
            if not comps:
                continue
            for metric in KEY_METRICS:
                block = []
                for label, ka, kb in comps:
                    r = paired_test(df, ka, kb, metric)
                    lo, hi = (scene_bootstrap(results_by_key, ka, kb, metric, n_boot, rng)
                              if n_boot else (np.nan, np.nan))
                    block.append({"dataset": ds, "family": fam_name, "comparison": label,
                                  "metric": metric, **r, "boot_ci_low": lo, "boot_ci_high": hi})
                adj = holm([b["p"] for b in block])
                for b, pa in zip(block, adj):
                    b["p_holm"] = pa
                rows += block
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ ANOVA
def anova_phase1(df):
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    rows = []
    for ds in sorted(df.dataset.unique()):
        keys = {key_of(dataset=ds, sampling=s, cbam=c, loss=l)
                for s in ("random", "oversample") for c in (False, True) for l in ("wce", "hybrid")}
        d = df[df.key.isin(keys)].copy()
        if d.key.nunique() < 8 or d.groupby("key").size().min() < 2:
            print(f"[ANOVA] {ds}: factorial belum lengkap, dilewati.")
            continue
        d["cbam"] = d["cbam"].astype(str)
        for metric in KEY_METRICS:
            model = ols(f"{metric} ~ C(sampling) * C(cbam) * C(loss)", data=d).fit()
            tab = sm.stats.anova_lm(model, typ=2)
            ss_res = tab.loc["Residual", "sum_sq"]
            for term, r in tab.drop(index="Residual").iterrows():
                rows.append({"dataset": ds, "metric": metric,
                             "term": term.replace("C(", "").replace(")", ""),
                             "F": r["F"], "p": r["PR(>F)"],
                             "partial_eta2": r["sum_sq"] / (r["sum_sq"] + ss_res)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ error analysis
def error_analysis(results_by_key, labels):
    rows = []
    for key, label in labels:
        rs = results_by_key.get(key)
        if not rs:
            continue
        cm = np.sum([np.array(r["confusion"]) for r in rs], axis=0).astype(float)
        sh = CONF.SHADOW
        fp_total = cm[:, sh].sum() - cm[sh, sh]
        fn_total = cm[sh, :].sum() - cm[sh, sh]
        row = {"Config": label, "TP_px": cm[sh, sh], "FP_px": fp_total, "FN_px": fn_total}
        for c in (CONF.CLEAR, CONF.THIN, CONF.CLOUD):
            name = CONF.CLASS_NAMES[c]
            row[f"FP from {name} (%)"] = cm[c, sh] / max(fp_total, 1) * 100
            row[f"FN to {name} (%)"] = cm[sh, c] / max(fn_total, 1) * 100
        row["FP/TP ratio"] = fp_total / max(cm[sh, sh], 1)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ labels per fase
def phase_labels(ds):
    p1 = [(key_of(dataset=ds, sampling=s, cbam=c, loss=l),
           f"{'OS' if s == 'oversample' else 'Random'} | {'U-Net+CBAM' if c else 'U-Net'} | {l}")
          for s in ("random", "oversample") for c in (False, True) for l in ("wce", "hybrid")]
    p2 = [(key_of(dataset=ds, loss="ce"), "Random | CE"),
          (key_of(dataset=ds, loss="wce"), "Random | WCE"),
          (key_of(dataset=ds, loss="cb"), "Random | Class-balanced CE"),
          (key_of(dataset=ds, loss="ohem"), "Random | OHEM"),
          (key_of(dataset=ds, sampling="ratio_weighted"), "Ratio-weighted | WCE"),
          (key_of(dataset=ds, sampling="class_aware"), "Class-aware | WCE"),
          (key_of(dataset=ds, sampling="nonempty"), "Random non-empty | WCE"),
          (key_of(dataset=ds, sampling="oversample"), "Location-aware OS | WCE (ours)")]
    p3 = [(key_of(dataset=ds, sampling="oversample", p=p), f"p={p}") for p in (0.2, 0.4, 0.6, 0.8, 1.0)]
    p3 += [(key_of(dataset=ds, sampling="oversample", tau=t), f"tau={t}") for t in (0.001, 0.01, 0.05)]
    p3 += [(key_of(dataset=ds, sampling="oversample", jitter=0.25), "jitter=0.25")]
    ssl = lambda **kw: key_of(dataset=ds, arch="resnet18", **kw)  # noqa: E731
    p4 = [(ssl(sampling="random", pretrained=True), "ResNet18-SSL fine-tune | Random"),
          (ssl(sampling="oversample", pretrained=True), "ResNet18-SSL fine-tune | OS"),
          (ssl(sampling="oversample", pretrained=True, freeze_encoder=True), "ResNet18-SSL frozen | OS"),
          (ssl(sampling="oversample", pretrained=False), "ResNet18 scratch | OS"),
          (key_of(dataset=ds, sampling="oversample"), "U-Net scratch | OS (ref)")]
    dlv = lambda **kw: key_of(dataset=ds, arch="deeplabv3plus", pretrained=True, **kw)  # noqa: E731
    p4 += [(dlv(sampling="random"), "DeepLabV3+ R50-ImageNet | Random"),
           (dlv(sampling="oversample"), "DeepLabV3+ R50-ImageNet | OS")]
    return {1: p1, 2: p2, 3: p3, 4: p4}


def md(df):
    return df.to_markdown(index=False, floatfmt=".3f") if len(df) else "_(belum ada data)_"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=2000, help="0 = tanpa bootstrap")
    ap.add_argument("--runs_dir", default=str(CONF.RUNS_DIR))
    ap.add_argument("--out_dir", default=str(CONF.ANALYSIS_DIR))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = load_results(args.runs_dir)
    if not results:
        print("Belum ada hasil di runs/.")
        return
    df = to_frame(results)
    by_key = {}
    for r in results:
        by_key.setdefault(r["key"], []).append(r)
    print(f"{len(df)} run, {df.key.nunique()} konfigurasi")

    df.sort_values(["dataset", "key", "seed"]).to_csv(out / "runs_raw.csv", index=False)
    summ = summarize(df)
    summ.to_csv(out / "summary_by_config.csv")

    report = ["# Laporan Analisis\n",
              "Semua nilai: mean ± SD sampel (ddof=1) antar seed. Uji berpasangan dipasangkan "
              "per seed; p_holm = koreksi Holm per keluarga & metrik; boot_ci = CI 95% "
              "bootstrap per scene test.\n"]
    # mIoU = macro (metrik utama); OA & microIoU = definisi TorchGeo/SSL4EO-L untuk perbandingan
    all_metrics = ["mIoU", "OA", "microIoU", "shadow_recall", "shadow_precision", "shadow_f1",
                   "shadow_iou", "clear_iou", "thin_iou", "cloud_iou"]
    err_labels = []
    for ds in sorted(df.dataset.unique()):
        labels = phase_labels(ds)
        for ph in (1, 2, 3, 4):
            t = phase_table(summ, labels[ph], all_metrics)
            t.to_markdown(out / f"phase{ph}_{ds}_table.md", index=False)
            report += [f"\n## Fase {ph} — {ds}\n", md(t), ""]
        err_labels += [(k, f"{ds} | {l}") for k, l in labels[1] + labels[2]]

        raw = df[df.dataset == ds].pivot_table(index="key", columns="seed", values="shadow_recall")
        report += [f"\n### Shadow recall per seed — {ds}\n", raw.round(2).reset_index().to_markdown(index=False)]

    comp = run_comparisons(df, by_key, args.n_boot)
    comp.to_csv(out / "comparisons.csv", index=False)
    if len(comp):
        show = comp[comp.metric.isin(["shadow_recall", "shadow_f1", "shadow_iou"])][
            ["dataset", "family", "comparison", "metric", "n_pairs", "diff", "ci_low", "ci_high",
             "p", "p_holm", "dz", "boot_ci_low", "boot_ci_high"]]
        report += ["\n## Uji berpasangan (metrik shadow)\n", md(show)]

    try:
        an = anova_phase1(df)
        an.to_csv(out / "anova_phase1.csv", index=False)
        report += ["\n## ANOVA 3 arah fase 1 (partial eta²)\n", md(an)]
    except ImportError:
        print("[ANOVA] statsmodels tidak terpasang, dilewati.")

    ea = error_analysis(by_key, err_labels)
    ea.to_csv(out / "error_analysis.csv", index=False)
    report += ["\n## Analisis error shadow (agregat semua seed)\n", md(ea)]

    cost = df.groupby("key")[["params_total", "params_trainable", "gflops", "train_minutes",
                              "test_minutes", "best_epoch"]].mean().reset_index()
    for ds in CONF.DATASETS:
        st = Path(CONF.artefact_paths(ds)["stats"])
        if st.exists():
            s = json.loads(st.read_text())
            report.append(f"\nPreprocessing {ds}: scan index {s.get('scan_minutes', float('nan')):.1f} menit "
                          f"(sekali, di luar waktu training).")
    cost.to_csv(out / "cost_table.csv", index=False)
    report += ["\n## Biaya komputasi (rata-rata per konfigurasi)\n", md(cost)]

    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[Saved] {out}/report.md (+ CSV)")


if __name__ == "__main__":
    main()
