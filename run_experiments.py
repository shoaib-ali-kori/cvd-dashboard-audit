"""
run_experiments.py - Phase 3 experiments on the controlled benchmark.

Protocol
  * DEV benchmark  (seed 42)  : used for development and for calibrating tau (5-fold cross-validation).
  * TEST benchmark (seed 2026): generated after the pipeline was frozen; used once for every reported number.
Outputs (in --out): metrics.json, tables/*.csv, figures/*.png, run_info.json

Usage:
  python make_benchmark.py --out benchmark      --seed 42
  python make_benchmark.py --out benchmark_test --seed 2026
  python run_experiments.py --dev benchmark --test benchmark_test --out results
"""
import argparse, json, platform, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr, binomtest
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cvdaudit import *
from make_benchmark import hex2rgb

OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "sky": "#56B4E9", "verm": "#D55E00",
         "pink": "#CC79A7", "yellow": "#F0E442", "grey": "#666666"}
TAU_GRID = [5, 8, 10, 12, 15]
WIDE_GRID = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50]   # for distance metrics with a different scale


# ------------------------------------------------------------------ data
def load_set(d):
    d = Path(d)
    manifest = json.load(open(d / "manifest.json"))
    recs = []
    for m in manifest:
        img = np.array(Image.open(d / "charts" / m["file"]).convert("RGB"))
        t0 = time.perf_counter()
        pal, sh, bg = extract_palette(img)
        t_ext = time.perf_counter() - t0
        ramp = np.array([hex2rgb(c) for c in m["ramp"]]) if m["ramp"] else None
        recs.append({**m, "pal": pal, "bg_est": bg, "ramp_arr": ramp,
                     "kind": infer_ramp_kind(ramp) if ramp is not None else None, "t_ext": t_ext})
    return recs


BASE = dict(tau=10.0, model="machado", severity=1.0, cvds=CVDS, use_structure=True, use_gradient=True,
            use_contrast=True, metric="ciede2000")


def predict(r, cfg):
    return hybrid_audit(r["pal"], r["bg_est"], tau=cfg["tau"], cues=r["cues"], ramp=r["ramp_arr"],
                        ramp_kind=r["kind"], legend_readable=r["legend_readable"], model=cfg["model"],
                        severity=cfg["severity"], cvds=cfg["cvds"], use_structure=cfg["use_structure"],
                        use_gradient=cfg["use_gradient"], use_contrast=cfg["use_contrast"], metric=cfg["metric"])[0]


# ------------------------------------------------------------------ metrics
def fail_prf(truth, pred):
    t = np.array([x == "Fail" for x in truth]); p = np.array([x == "Fail" for x in pred])
    tp, fp, fn = (t & p).sum(), (~t & p).sum(), (t & ~p).sum()
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)


def summarize(truth, pred):
    pr, rc, f1 = fail_prf(truth, pred)
    truth, pred = list(truth), list(pred)
    ct = [i for i, x in enumerate(truth) if x == "Cannot Tell"]
    return {"fail_precision": pr, "fail_recall": rc, "fail_f1": f1,
            "macro_f1": f1_score(truth, pred, labels=list(LABELS), average="macro", zero_division=0),
            "accuracy": float(np.mean([a == b for a, b in zip(truth, pred)])),
            "abstention_rate": float(np.mean([p == "Cannot Tell" for p in pred])),
            "cannot_tell_accuracy": float(np.mean([pred[i] == "Cannot Tell" for i in ct])) if ct else float("nan")}


def bootstrap_f1(truth, pred, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.array([x == "Fail" for x in truth]); p = np.array([x == "Fail" for x in pred])
    idx = rng.integers(0, len(t), (n, len(t)))
    tt, pp = t[idx], p[idx]
    tp = (tt & pp).sum(1); fp = (~tt & pp).sum(1); fn = (tt & ~pp).sum(1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))


def mcnemar_fail(truth, pred_a, pred_b):
    """Exact McNemar on correctness of the binary Fail / not-Fail decision."""
    t = np.array([x == "Fail" for x in truth])
    ca = np.array([(x == "Fail") == y for x, y in zip(pred_a, t)])
    cb = np.array([(x == "Fail") == y for x, y in zip(pred_b, t)])
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------------------------ calibration
def calibrate(recs, cfg, grid, seed=0):
    truth = [r["truth"] for r in recs]
    preds = {t: [predict(r, {**cfg, "tau": t}) for r in recs] for t in grid}
    f1_all = {t: fail_prf(truth, preds[t])[2] for t in grid}
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    cv = []
    for tr, te in skf.split(np.zeros(len(truth)), truth):
        best = max(grid, key=lambda t: (fail_prf([truth[i] for i in tr], [preds[t][i] for i in tr])[2], -t))
        cv.append(fail_prf([truth[i] for i in te], [preds[best][i] for i in te])[2])
    final = max(grid, key=lambda t: (f1_all[t], -t))
    return final, f1_all, float(np.mean(cv)), float(np.std(cv))


# ------------------------------------------------------------------ scores for ROC-AUC / sensitivity
def collapse_score(r, model="machado", severity=1.0, ref=5.0):
    """Largest relative loss of distance among pairs distinguishable under normal vision (0 = none, 1 = total)."""
    pal = r["pal"]
    if len(pal) < 2:
        return np.nan
    f = palette_features(pal, r["bg_est"], model, severity)
    dn = f["d_norm"]; s = 0.0
    for c in CVDS:
        m = dn >= ref
        if m.any():
            s = max(s, float(np.max((dn[m] - f[c][m]) / dn[m])))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="benchmark")
    ap.add_argument("--test", default="benchmark_test")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out); (out / "tables").mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(parents=True, exist_ok=True)
    res = {}

    print("extracting palettes ...")
    dev, test = load_set(a.dev), load_set(a.test)
    print(f"dev={len(dev)} charts, test={len(test)} charts")
    dev_truth, test_truth = [r["truth"] for r in dev], [r["truth"] for r in test]

    # ---------------- 1. calibration on DEV
    tau, f1_dev, cv_mean, cv_sd = calibrate(dev, BASE, TAU_GRID)
    cfg = {**BASE, "tau": float(tau)}
    res["calibration"] = {"tau_grid": TAU_GRID, "dev_f1_by_tau": f1_dev, "selected_tau": tau,
                          "cv_f1_mean": cv_mean, "cv_f1_sd": cv_sd}
    print(f"selected tau={tau}  (5-fold CV F1 {cv_mean:.3f} +/- {cv_sd:.3f})")

    # ---------------- 2. main comparison on TEST
    P = {"Hybrid (ours)": [predict(r, cfg) for r in test],
         "B1 contrast-only": [baseline_b1(r["pal"], r["bg_est"], r["ramp_arr"]) for r in test],
         "B2 simulation-only": [baseline_b2(r["pal"], r["bg_est"], r["ramp_arr"]) for r in test]}
    main_rows = []
    for name, pr in P.items():
        s = summarize(test_truth, pr); lo, hi = bootstrap_f1(test_truth, pr)
        main_rows.append({"method": name, **s, "f1_ci_low": lo, "f1_ci_high": hi})
    main_df = pd.DataFrame(main_rows); main_df.to_csv(out / "tables/main_comparison.csv", index=False)
    print(main_df.round(3).to_string(index=False))

    # McNemar with Holm
    comps = []
    for other in ("B1 contrast-only", "B2 simulation-only"):
        b, c, p = mcnemar_fail(test_truth, P["Hybrid (ours)"], P[other]); comps.append((other, b, c, p))
    adj = holm([c[3] for c in comps])
    mc_df = pd.DataFrame([{"comparison": f"Hybrid vs {o}", "hybrid_only_correct": b, "baseline_only_correct": c,
                           "p_raw": p, "p_holm": adj[i]} for i, (o, b, c, p) in enumerate(comps)])
    mc_df.to_csv(out / "tables/mcnemar.csv", index=False); print(mc_df.round(4).to_string(index=False))

    # confusion matrices
    conf = {}
    for name, pr in P.items():
        cm = pd.crosstab(pd.Categorical(test_truth, LABELS), pd.Categorical(pr, LABELS), dropna=False)
        cm.to_csv(out / f"tables/confusion_{name.split()[0]}.csv"); conf[name] = cm.values

    # ROC-AUC of the continuous scores (categorical charts, truth Pass (no redundancy) vs Fail(cvd))
    sel = [i for i, r in enumerate(test) if r["type"] != "heatmap" and len(r["pal"]) >= 2 and
           ((r["truth"] == "Pass" and r["redundancy"] == "none") or r["issue"] == "cvd")]
    y = np.array([test[i]["truth"] == "Fail" for i in sel])
    sc_h = np.array([collapse_score(test[i]) for i in sel])
    sc_b1 = np.array([3.0 / float(contrast_ratio(test[i]["pal"], test[i]["bg_est"][None, :]).min()) for i in sel])
    sc_b2 = np.array([-float(min(palette_features(test[i]["pal"], test[i]["bg_est"])[c].min() for c in CVDS)) for i in sel])
    res["roc_auc_cvd_score"] = {"n_charts": int(len(sel)), "hybrid_collapse_score": roc_auc_score(y, sc_h),
                                "b1_contrast_score": roc_auc_score(y, sc_b1), "b2_min_sim_distance": roc_auc_score(y, sc_b2)}
    print("ROC-AUC", res["roc_auc_cvd_score"])

    # cost
    t_aud = []
    for r in test:
        t0 = time.perf_counter(); predict(r, cfg); t_aud.append(time.perf_counter() - t0)
    res["cost_ms_per_chart"] = {"palette_extraction": 1000 * float(np.mean([r["t_ext"] for r in test])),
                                "audit_decision": 1000 * float(np.mean(t_aud))}

    # ---------------- 3. tau sensitivity (dev and test)
    f1_test = {t: fail_prf(test_truth, [predict(r, {**cfg, "tau": t}) for r in test])[2] for t in TAU_GRID}
    pd.DataFrame({"tau": TAU_GRID, "f1_dev": [f1_dev[t] for t in TAU_GRID], "f1_test": [f1_test[t] for t in TAU_GRID]}
                 ).to_csv(out / "tables/tau_sensitivity.csv", index=False)

    # ---------------- 4. ablations (each variant recalibrated on DEV over its own grid, evaluated on TEST)
    variants = [
        ("Full model", {}, TAU_GRID),
        ("- protan removed", {"cvds": ("deutan", "tritan")}, TAU_GRID),
        ("- deutan removed", {"cvds": ("protan", "tritan")}, TAU_GRID),
        ("- tritan removed", {"cvds": ("protan", "deutan")}, TAU_GRID),
        ("CIEDE2000 -> CIE76", {"metric": "cie76"}, WIDE_GRID),
        ("CIEDE2000 -> sRGB Euclid", {"metric": "rgb"}, WIDE_GRID),
        ("Machado -> Vienot", {"model": "vienot"}, TAU_GRID),
        ("Severity 1.0 -> 0.6", {"severity": 0.6}, TAU_GRID),
        ("- structure module", {"use_structure": False}, TAU_GRID),
        ("- gradient module", {"use_gradient": False}, TAU_GRID),
        ("- contrast check", {"use_contrast": False}, TAU_GRID),
        ("Calibrated tau -> fixed 10", {"fixed": True}, [10]),
    ]
    abl = []
    for name, over, grid in variants:
        c0 = {**BASE, **{k: v for k, v in over.items() if k != "fixed"}}
        t_v = 10.0 if over.get("fixed") else calibrate(dev, c0, grid)[0]
        c1 = {**c0, "tau": float(t_v)}
        pr = [predict(r, c1) for r in test]
        s = summarize(test_truth, pr); lo, hi = bootstrap_f1(test_truth, pr, n=2000)
        abl.append({"variant": name, "tau": t_v, "fail_f1": s["fail_f1"], "f1_ci_low": lo, "f1_ci_high": hi,
                    "fail_precision": s["fail_precision"], "fail_recall": s["fail_recall"], "macro_f1": s["macro_f1"],
                    "abstention_rate": s["abstention_rate"]})
    abl_df = pd.DataFrame(abl)
    abl_df["delta_f1_vs_full"] = abl_df["fail_f1"] - abl_df.loc[0, "fail_f1"]
    abl_df.to_csv(out / "tables/ablation.csv", index=False)
    print(abl_df.round(3).to_string(index=False))

    # ---------------- 5. sensitivity to simulation model / severity (RQ2 proxy)
    settings = {"Machado 0.3": ("machado", 0.3), "Machado 0.6": ("machado", 0.6), "Machado 1.0": ("machado", 1.0),
                "Vienot 1.0": ("vienot", 1.0)}
    cat = [r for r in test if r["type"] != "heatmap" and len(r["pal"]) >= 2]
    sc = {k: np.array([collapse_score(r, m, s) for r in cat]) for k, (m, s) in settings.items()}
    rho = pd.DataFrame([[spearmanr(sc[a1], sc[b1])[0] for b1 in settings] for a1 in settings],
                       index=list(settings), columns=list(settings))
    rho.to_csv(out / "tables/sensitivity_spearman.csv")
    flips = {}
    base_pred = [predict(r, cfg) for r in test]
    for k, (m, s) in settings.items():
        alt = [predict(r, {**cfg, "model": m, "severity": s}) for r in test]
        flips[k] = float(np.mean([x != y_ for x, y_ in zip(base_pred, alt)]))
    res["label_flip_rate_vs_default"] = flips
    print("flip rates", flips)

    # ---------------- 6. extraction quality and error analysis (TEST)
    ext_rows = []
    for r in test:
        if r["type"] == "heatmap":
            continue
        tr = np.array([hex2rgb(c) for c in r["palette"]])
        d = [float(de2000(to_lab(t[None]), to_lab(r["pal"])).min()) if len(r["pal"]) else 99.0 for t in tr]
        ext_rows.append({"type": r["type"], "jpeg": bool(r["jpeg_q"]), "theme": r["theme"], "k_true": r["k"],
                         "k_found": len(r["pal"]), "exact": r["k"] == len(r["pal"]), "mean_de": float(np.mean(d))})
    ext = pd.DataFrame(ext_rows); ext.to_csv(out / "tables/extraction_quality.csv", index=False)
    res["extraction"] = {"exact_k_rate": float(ext["exact"].mean()), "mean_colour_error_de2000": float(ext["mean_de"].mean()),
                         "overcount_rate": float((ext.k_found > ext.k_true).mean()),
                         "undercount_rate": float((ext.k_found < ext.k_true).mean())}

    errs = []
    for r, p in zip(test, P["Hybrid (ours)"]):
        if p == r["truth"]:
            continue
        k_f = len(r["pal"])
        if r["truth"] == "Pass" and p == "Cannot Tell" and r["redundancy"] == "full":
            cause = "abstained on unsafe palette with full redundancy (by design)"
        elif r["type"] != "heatmap" and k_f > r["k"]:
            cause = "extraction: spurious extra colour"
        elif r["type"] != "heatmap" and k_f < r["k"]:
            cause = "extraction: missed colour"
        elif r["type"] == "heatmap":
            cause = "gradient decision"
        elif r["redundancy"] != "none":
            cause = "redundancy handling"
        else:
            cause = "decision threshold / other"
        errs.append({"file": r["file"], "truth": r["truth"], "pred": p, "type": r["type"], "theme": r["theme"],
                     "jpeg": bool(r["jpeg_q"]), "cause": cause})
    err_df = pd.DataFrame(errs, columns=["file", "truth", "pred", "type", "theme", "jpeg", "cause"])
    err_df.to_csv(out / "tables/errors_hybrid_test.csv", index=False)
    err_counts = err_df["cause"].value_counts()
    err_counts.to_csv(out / "tables/error_causes.csv", header=["count"])
    print(err_counts.to_string())

    res["main"] = main_rows; res["dataset"] = {"dev": dict(Counter(dev_truth)), "test": dict(Counter(test_truth))}
    json.dump(res, open(out / "metrics.json", "w"), indent=2, default=float)
    json.dump({"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
               "tau_grid": TAU_GRID, "dev_dir": a.dev, "test_dir": a.test}, open(out / "run_info.json", "w"), indent=2)

    # ---------------- figures
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
    # Fig 1 main comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.25; x = np.arange(3)
    for i, (m, col) in enumerate([("fail_precision", OKABE["sky"]), ("fail_recall", OKABE["orange"]), ("fail_f1", OKABE["blue"])]):
        ax.bar(x + (i - 1) * w, main_df[m], w, label=m.replace("fail_", "").replace("f1", "F1").capitalize(), color=col)
    err = np.array([main_df.fail_f1 - main_df.f1_ci_low, main_df.f1_ci_high - main_df.fail_f1])
    ax.errorbar(x + w, main_df.fail_f1, yerr=err, fmt="none", ecolor="black", capsize=3, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(main_df.method); ax.set_ylim(0, 1.05); ax.set_ylabel("Fail class score")
    ax.set_title("Fail-class precision, recall and F1 on the test benchmark\n(error bars: bootstrap 95% CI of F1)", fontsize=10)
    ax.legend(frameon=False, ncol=3, loc="upper right"); fig.tight_layout(); fig.savefig(out / "figures/fig1_main_comparison.png", dpi=200); plt.close(fig)
    # Fig 2 ablation
    d = abl_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cols = [OKABE["blue"] if v == "Full model" else (OKABE["verm"] if dl < -0.005 else OKABE["grey"]) for v, dl in zip(d.variant, d.delta_f1_vs_full)]
    ax.barh(d.variant, d.fail_f1, color=cols)
    ax.errorbar(d.fail_f1, range(len(d)), xerr=[d.fail_f1 - d.f1_ci_low, d.f1_ci_high - d.fail_f1], fmt="none", ecolor="black", capsize=2, lw=0.8)
    ax.set_xlim(0, 1.05); ax.set_xlabel("Fail-class F1 (test)"); ax.set_title("Ablation: one component changed at a time", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "figures/fig2_ablation.png", dpi=200); plt.close(fig)
    # Fig 3 tau curve
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(TAU_GRID, [f1_dev[t] for t in TAU_GRID], "o-", color=OKABE["blue"], label="dev")
    ax.plot(TAU_GRID, [f1_test[t] for t in TAU_GRID], "s--", color=OKABE["verm"], label="test")
    ax.axvline(tau, color=OKABE["grey"], ls=":", label=f"selected tau = {tau:g}")
    ax.set_xlabel("tau (CIEDE2000)"); ax.set_ylabel("Fail-class F1"); ax.set_ylim(0, 1.05); ax.legend(frameon=False)
    ax.set_title("Sensitivity to the confusability threshold", fontsize=10); fig.tight_layout(); fig.savefig(out / "figures/fig3_tau_sensitivity.png", dpi=200); plt.close(fig)
    # Fig 4 confusion matrices
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, cm) in zip(axs, conf.items()):
        ax.imshow(cm, cmap="Blues")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(4)); ax.set_xticklabels(LABELS, rotation=30, ha="right"); ax.set_yticks(range(4)); ax.set_yticklabels(LABELS)
        ax.set_xlabel("predicted"); ax.set_ylabel("truth"); ax.set_title(name, fontsize=10)
    fig.tight_layout(); fig.savefig(out / "figures/fig4_confusion.png", dpi=200); plt.close(fig)
    # Fig 5 sensitivity
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    im = axs[0].imshow(rho.values, vmin=0.5, vmax=1, cmap="Blues")
    for i in range(len(rho)):
        for j in range(len(rho)):
            axs[0].text(j, i, f"{rho.values[i, j]:.2f}", ha="center", va="center", color="white" if rho.values[i, j] > 0.8 else "black")
    axs[0].set_xticks(range(len(rho))); axs[0].set_xticklabels(rho.columns, rotation=30, ha="right"); axs[0].set_yticks(range(len(rho))); axs[0].set_yticklabels(rho.index)
    axs[0].set_title("Spearman correlation of chart scores\nbetween simulation settings", fontsize=10)
    axs[1].bar(list(flips), [100 * v for v in flips.values()], color=OKABE["orange"])
    axs[1].set_ylabel("labels that change vs Machado 1.0 (%)"); axs[1].set_title("Decision stability", fontsize=10)
    plt.setp(axs[1].get_xticklabels(), rotation=30, ha="right"); fig.tight_layout(); fig.savefig(out / "figures/fig5_sensitivity.png", dpi=200); plt.close(fig)
    # Fig 6 error causes
    if len(err_counts):
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.barh(err_counts.index[::-1], err_counts.values[::-1], color=OKABE["verm"]); ax.set_xlabel("charts")
        ax.set_title("Hybrid audit: where the test errors come from", fontsize=10); fig.tight_layout()
        fig.savefig(out / "figures/fig6_error_causes.png", dpi=200); plt.close(fig)
    # Fig 7 extraction quality
    g = ext.assign(fmt=np.where(ext.jpeg, "JPEG", "PNG")).groupby(["type", "fmt"]).agg(exact=("exact", "mean"), de=("mean_de", "mean")).reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, lab in zip(axs, ["exact", "de"], ["exact colour-count recovery", "mean colour error (CIEDE2000)"]):
        piv = g.pivot(index="type", columns="fmt", values=col)
        piv.plot.bar(ax=ax, color=[OKABE["blue"], OKABE["orange"]], rot=0); ax.set_ylabel(lab); ax.set_xlabel(""); ax.legend(frameon=False)
    fig.suptitle("Palette extraction quality on the test benchmark", fontsize=10); fig.tight_layout()
    fig.savefig(out / "figures/fig7_extraction_quality.png", dpi=200); plt.close(fig)
    print("done ->", out)


if __name__ == "__main__":
    main()
