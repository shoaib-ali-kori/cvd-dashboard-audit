"""
review.py - human spot-check of the real-dashboard audit.

Step A  (make):  python review.py make  --charts results_real/charts.csv --raw data/raw --out review --exclude 1,2,3,4,5
   -> review/sheet_1.png, sheet_2.png, ...   numbered chart crops, normal vision next to deuteranopia simulation
      (the audit's label is NOT shown, so the reviewer is not anchored)
   -> review/review_labels.csv               fill the human_label column

   Labels:  Pass     every colour/category that is distinct in normal vision stays distinguishable in the simulations
            Fail     at least two colours/categories that are distinct in normal vision become hard to tell apart
            CannotTell  you cannot judge (no legend, too small, cropped)
            NotChart  the region is not a data chart (photo, banner, icon, page chrome, blank)

Step B  (score): python review.py score --labels review/review_labels.csv --charts results_real/charts.csv --out review
   [--second review/review_labels_2.csv]  second annotator -> Cohen's kappa
   -> review/review_scores.csv and a printed summary
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cvdaudit import simulate

PER_SHEET = 8


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def make(a):
    ch = pd.read_csv(a.charts, dtype={"id": str})
    excl = {s.strip() for s in a.exclude.split(",") if s.strip()}
    ch = ch[~ch["id"].isin(excl) & ch["label"].isin(["Pass", "Fail", "Cannot Tell"])].copy()
    ch["key"] = ch["id"].astype(str) + "-" + ch["region"].astype(str)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for _, r in ch.iterrows():
        img = np.array(Image.open(Path(a.raw) / str(r["id"]) / "default.png").convert("RGB"))
        crop = img[int(r["y"]):int(r["y"] + r["h"]), int(r["x"]):int(r["x"] + r["w"])]
        w = 420 / crop.shape[1] if crop.shape[1] > 420 else 1.0
        crop = np.array(Image.fromarray(crop).resize((max(1, int(crop.shape[1] * w)), max(1, int(crop.shape[0] * w)))))
        sim = (np.clip(simulate(crop / 255.0, "deutan"), 0, 1) * 255).astype(np.uint8)
        tiles.append((r["key"], crop, sim))
    for s in range(0, len(tiles), PER_SHEET):
        chunk = tiles[s:s + PER_SHEET]
        fig, axs = plt.subplots(len(chunk), 2, figsize=(10, 2.7 * len(chunk)), squeeze=False)
        for i, (key, c, sm) in enumerate(chunk):
            axs[i, 0].imshow(c); axs[i, 1].imshow(sm)
            axs[i, 0].set_title(f"[{key}]  normal", fontsize=9, loc="left"); axs[i, 1].set_title("deuteranopia simulation", fontsize=9, loc="left")
            axs[i, 0].axis("off"); axs[i, 1].axis("off")
        fig.tight_layout(); fig.savefig(out / f"sheet_{s // PER_SHEET + 1}.png", dpi=100); plt.close(fig)
    pd.DataFrame({"key": ch["key"], "human_label": "", "note": ""}).to_csv(out / "review_labels.csv", index=False)
    print(f"{len(tiles)} charts -> {out}/sheet_*.png and review_labels.csv (fill human_label)")


def kappa(a, b):
    labs = sorted(set(a) | set(b)); n = len(a)
    po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labs)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score(a):
    ch = pd.read_csv(a.charts, dtype={"id": str})
    ch["key"] = ch["id"].astype(str) + "-" + ch["region"].astype(str)
    h = pd.read_csv(a.labels, dtype=str).fillna("")
    h = h[h["human_label"].str.strip() != ""]
    if h.empty:
        raise SystemExit(f"No filled human_label values found in {a.labels}. Fill the column, save as CSV, and run again.")
    unknown = sorted(set(h["human_label"]) - {"Pass", "Fail", "CannotTell", "NotChart"})
    if unknown:
        raise SystemExit(f"Unexpected labels {unknown}. Use exactly: Pass, Fail, CannotTell, NotChart.")
    if not set(h["key"]) & set(ch["key"]):
        raise SystemExit(f"None of the keys in {a.labels} match the audit table. Keys in your file start with {list(h['key'][:3])} "
                         f"but the audit has {list(ch['key'][:3])}. Excel often turns keys like 9-1 into dates: edit the CSV in Notepad or VS Code.")
    rep = {"labelled_rows": len(h)}
    missing = sorted(set(h["key"]) - set(ch["key"]))
    rep["labelled_keys_not_in_audit_table"] = len(missing)
    if missing:
        print(f"WARNING: {len(missing)} of {len(h)} labelled keys are not in {a.charts}: {missing[:12]}"
              + (" ..." if len(missing) > 12 else "") +
              "\nThe audit table no longer matches the sheets you labelled (it was regenerated). Scores below cover only matching keys "
              "and are NOT reliable unless the matching crops are unchanged.")
    if a.second:
        h2 = pd.read_csv(a.second, dtype=str).fillna("")
        j = h.merge(h2, on="key", suffixes=("", "_2"))
        j = j[j["human_label_2"].str.strip() != ""]
        if j.empty:
            raise SystemExit(f"{a.second} has no filled human_label values that match {a.labels}. Leave out --second until the second annotator is done.")
        rep["annotators_n"] = len(j); rep["cohen_kappa"] = kappa(list(j.human_label), list(j.human_label_2))
        rep["raw_agreement"] = float(np.mean(j.human_label == j.human_label_2))
        h = h.merge(h2[["key", "human_label"]], on="key", how="left", suffixes=("", "_2"))
        h["human_label"] = np.where(h["human_label_2"].fillna("") != "", h["human_label_2"], h["human_label"]) if a.use_second_as_final else h["human_label"]
    d = h.merge(ch, on="key")
    rep["reviewed_regions"] = len(d)
    rep["not_a_chart_share"] = float((d.human_label == "NotChart").mean())    # region-selection precision = 1 - this
    rep["human_cannot_tell"] = int((d.human_label == "CannotTell").sum())
    g = d[d.human_label.isin(["Pass", "Fail"])].copy()
    rep["judgeable_charts"] = len(g)
    rows = []
    for name, col in (("Hybrid", "label"), ("B1 contrast-only", "b1"), ("B2 simulation-only", "b2")):
        p = g[col] == "Fail"; t = g.human_label == "Fail"
        tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        lo, hi = wilson(tp, tp + fp)
        rows.append({"method": name, "n": len(g), "fail_precision": pr, "precision_wilson_low": lo, "precision_wilson_high": hi,
                     "fail_recall": rc, "fail_f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0,
                     "agreement": float(((p) == (t)).mean()) if len(g) else float("nan"), "tp": tp, "fp": fp, "fn": fn})
    res = pd.DataFrame(rows)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "review_scores.csv", index=False)
    json.dump(rep, open(out / "review_summary.json", "w"), indent=2)
    print(json.dumps(rep, indent=2)); print(res.round(3).to_string(index=False))
    d[d.human_label.isin(["Pass", "Fail"]) & ((d.label == "Fail") != (d.human_label == "Fail"))][["key", "human_label", "label", "reason", "note"]].to_csv(out / "disagreements.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("--charts", default="results_real/charts.csv"); m.add_argument("--raw", default="data/raw")
    m.add_argument("--out", default="review"); m.add_argument("--exclude", default="")
    s = sub.add_parser("score"); s.add_argument("--labels", default="review/review_labels.csv"); s.add_argument("--charts", default="results_real/charts.csv")
    s.add_argument("--out", default="review"); s.add_argument("--second", default=""); s.add_argument("--use-second-as-final", action="store_true")
    a = ap.parse_args()
    make(a) if a.cmd == "make" else score(a)
