"""
audit_real.py - run the hybrid audit on the dashboards captured by capture.py (pilot set).

For each dashboard with status ok it
  1. reads data/raw/<id>/default.png and regions_default.json,
  2. selects chart regions (svg, canvas, and iframes whose inside could not be read; photos are skipped),
  3. extracts each chart's palette and audits it (hybrid audit, B1, B2),
  4. writes CVD-simulated previews, a per-chart table, a per-dashboard table and prevalence with Wilson 95% CIs.

There are no human labels for these dashboards, so no accuracy is reported here: the output is a descriptive
audit and a visual check, not a validation.

Usage: python audit_real.py --csv urls.csv --raw data/raw --out results_real [--tau 10]
"""
import argparse, csv, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cvdaudit import *

MIN_W, MIN_H = 150, 100


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def inside(a, b, frac=0.9):
    """True if box a lies at least `frac` inside box b."""
    ix = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    return ix * iy >= frac * a["w"] * a["h"]


def pick_regions(regions, W, H, include_img=False):
    non_top = [r for r in regions if r.get("frame", "top") != "top"]
    cands = []
    for r in regions:
        if r["tag"] == "img" and not include_img:
            continue
        if r["tag"] == "iframe" and any(inside(q, r) for q in non_top):
            continue                                   # its inner charts are audited instead
        x0, y0 = max(0, int(r["x"])), max(0, int(r["y"]))
        x1, y1 = min(W, int(r["x"] + r["w"])), min(H, int(r["y"] + r["h"]))
        if x1 - x0 < MIN_W or y1 - y0 < MIN_H:
            continue
        cands.append({**r, "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
    cands.sort(key=lambda r: -r["w"] * r["h"])
    kept = []
    for r in cands:                                    # drop regions contained in a larger kept region
        if not any(inside(r, k) for k in kept):
            kept.append(r)
    return kept


def audit_crop(crop, tau, n_patterns):
    """Returns dict with label, reason, palette and baselines."""
    pal, sh, bg = extract_palette(crop)
    out = {"n_colours": len(pal), "palette": ["#%02x%02x%02x" % tuple((c * 255).round().astype(int)) for c in pal],
           "bg": "#%02x%02x%02x" % tuple((bg * 255).round().astype(int))}
    if is_continuous(crop):                            # photo, map or continuous scale without a readable legend
        out.update(label="Cannot Tell", reason="continuous colours / no readable legend")
    else:
        lab, why = hybrid_audit(pal, bg, tau=tau, cues={"patterns": n_patterns}, legend_readable=True)
        out.update(label=lab, reason=why)
    out["b1"] = baseline_b1(pal, bg)
    out["b2"] = baseline_b2(pal, bg)
    if len(pal) >= 2:
        f = palette_features(pal, bg)
        out["min_contrast"] = float(f["contrast"].min())
        for c in CVDS:
            m = f["d_norm"] >= tau
            out[f"min_dist_{c}"] = float(f[c][m].min()) if m.any() else float("nan")
    return out


def simulate_image(img_u8, cvd):
    return (np.clip(simulate(img_u8.astype(float) / 255.0, cvd), 0, 1) * 255).astype(np.uint8)


def preview(did, crops, labels, out_path, max_rows=4, tw=300):
    crops, labels = crops[:max_rows], labels[:max_rows]
    if not crops:
        return
    fig, axs = plt.subplots(len(crops), 4, figsize=(13, 2.6 * len(crops)), squeeze=False)
    for i, (cr, lab) in enumerate(zip(crops, labels)):
        h, w = cr.shape[:2]
        s = min(1.0, tw / w)
        im = np.array(Image.fromarray(cr).resize((max(1, int(w * s)), max(1, int(h * s)))))
        for j, cvd in enumerate(("normal",) + CVDS):
            axs[i, j].imshow(im if cvd == "normal" else simulate_image(im, cvd))
            axs[i, j].axis("off")
            if i == 0:
                axs[i, j].set_title(cvd, fontsize=10)
        axs[i, 0].text(0, -0.04, lab, transform=axs[i, 0].transAxes, fontsize=8, va="bottom")
    fig.suptitle(f"Dashboard {did}: chart regions under simulated colour-vision deficiency (Machado, severity 1.0)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="urls.csv")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results_real")
    ap.add_argument("--tau", type=float, default=None, help="default: tau selected in results/metrics.json, else 10")
    ap.add_argument("--include-img", action="store_true")
    a = ap.parse_args()
    tau = a.tau
    if tau is None:
        try:
            tau = float(json.load(open("results/metrics.json"))["calibration"]["selected_tau"])
        except Exception:
            tau = 10.0
    out = Path(a.out); (out / "previews").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.csv, newline="", encoding="utf-8-sig")))
    chart_rows, dash_rows, excluded = [], [], []
    for row in rows:
        d = Path(a.raw) / row["id"]
        mp = d / "meta_default.json"
        if not mp.exists():
            excluded.append((row["id"], "not captured")); continue
        meta = json.loads(mp.read_text())
        if meta.get("status") != "ok":
            excluded.append((row["id"], meta.get("status", "?"))); continue
        img = np.array(Image.open(d / "default.png").convert("RGB"))
        H, W = img.shape[:2]
        regions = json.loads((d / "regions_default.json").read_text())
        picked = pick_regions(regions, W, H, a.include_img)
        crops, labs, n_fail, n_pass = [], [], 0, 0
        for k, r in enumerate(picked):
            crop = img[r["y"]:r["y"] + r["h"], r["x"]:r["x"] + r["w"]]
            res = audit_crop(crop, tau, int(r.get("n_pattern", 0) or 0))
            chart_rows.append({"id": row["id"], "platform": row.get("platform", ""), "region": k, "tag": r["tag"],
                               "x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"], **res})
            n_fail += res["label"] == "Fail"; n_pass += res["label"] == "Pass"
            crops.append(crop); labs.append(f"region {k} ({r['tag']}): {res['label']} - {res['reason']}")
        assess = n_fail + n_pass
        dash_rows.append({"id": row["id"], "platform": row.get("platform", ""), "domain": row.get("domain", ""),
                          "title": meta.get("title", ""), "regions_found": len(picked), "assessable": assess,
                          "failing": n_fail,
                          "label": "No assessable chart" if assess == 0 else ("Fail" if n_fail else "Pass"),
                          "cvd_score": (100.0 * (1 - n_fail / assess)) if assess else float("nan")})
        # preview: the charts that were actually assessable first
        order = sorted(range(len(crops)), key=lambda i: (labs[i].split(": ")[1].split(" - ")[0] not in ("Fail", "Pass"), i))
        preview(row["id"], [crops[i] for i in order], [labs[i] for i in order], out / "previews" / f"dashboard_{row['id']}.png")
    ch = pd.DataFrame(chart_rows); ds = pd.DataFrame(dash_rows)
    ch.to_csv(out / "charts.csv", index=False); ds.to_csv(out / "dashboards.csv", index=False)
    # prevalence (assessable dashboards only)
    prev = []
    ass = ds[ds.label != "No assessable chart"]
    groups = [("ALL", ass)] + [(p, g) for p, g in ass.groupby("platform")]
    for name, g in groups:
        k, n = int((g.label == "Fail").sum()), len(g)
        lo, hi = wilson(k, n)
        prev.append({"group": name, "dashboards": n, "with_failing_chart": k, "share": k / n if n else float("nan"),
                     "wilson95_low": lo, "wilson95_high": hi})
    pd.DataFrame(prev).to_csv(out / "prevalence.csv", index=False)
    print(f"tau = {tau:g}; dashboards captured: {len(ds)}; excluded: {len(excluded)}")
    print(ds[["id", "platform", "regions_found", "assessable", "failing", "label", "cvd_score"]].to_string(index=False))
    print("\nPrevalence (assessable dashboards):"); print(pd.DataFrame(prev).round(3).to_string(index=False))
    if excluded:
        print("\nExcluded:", excluded)
    if len(ch):
        print("\nChart labels:", ch.label.value_counts().to_dict())
        print("B1 / B2 labels:", ch.b1.value_counts().to_dict(), ch.b2.value_counts().to_dict())


if __name__ == "__main__":
    main()
