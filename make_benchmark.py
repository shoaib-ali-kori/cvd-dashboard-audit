"""
make_benchmark.py - controlled synthetic chart benchmark (Phase 3).

Every chart has a ground-truth label known by construction, never produced by the audit itself:
  * palette safety comes from provenance: the published CVD-safe Okabe-Ito set, and lightness-stepped palettes
    (CVD preserves lightness, so colours separated by >= 18 L* units stay distinguishable), versus textbook
    red/green and blue/purple confusions (matplotlib tab10, ColorBrewer Set1, dark red/green)
  * Tol Bright and viridis samples were left out: Tol's palette is documented for protan/deutan only (the audit
    also covers tritan) and viridis is a continuous map, not a categorical set
  * non-text contrast failure is the exact WCAG 3:1 rule applied to the true hex colours and true background
  * Not Present  = single-colour chart;  Cannot Tell = legend withheld (as in the E3 design)
  * unsafe palette + complete redundant encoding (distinct shapes/patterns AND direct labels) = Pass
Usage: python make_benchmark.py --out benchmark --seed 42
Outputs: benchmark/charts/*.png|jpg and benchmark/manifest.json
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.ndimage import gaussian_filter
from PIL import Image
from cvdaudit import contrast_ratio, de2000, to_lab

plt.rcParams["hatch.linewidth"] = 1.2

OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]   # black omitted
TOL = ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377"]                # grey omitted
TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
SET1 = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999"]
FAIL_PAIRS = [("#d62728", "#2ca02c"), ("#e41a1c", "#4daf4a"), ("#cc0000", "#007700"), ("#1f77b4", "#9467bd")]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
HATCHES = ["//", "\\\\", "xx", "..", "++", "oo", "**"]
THEMES = {"white": ("#ffffff", "#222222", "#d9d9d9"), "lightgray": ("#f0f0f0", "#222222", "#d0d0d0"),
          "dark": ("#1e1e1e", "#dddddd", "#444444")}
SAFE_CMAPS = {"viridis": "sequential", "cividis": "sequential", "magma": "sequential", "Blues": "sequential",
              "RdBu": "diverging"}
UNSAFE_CMAPS = {"jet": "sequential", "gist_rainbow": "sequential", "nipy_spectral": "sequential",
                "RdYlGn": "diverging"}


def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


def ok_contrast(c, bg, margin=3.0):
    return bool(contrast_ratio(hex2rgb(c)[None, :], hex2rgb(bg)[None, :])[0] >= margin)


HUES = {"blue": 260, "orange": 55, "purple": 310, "teal": 190}


def lightness_palette(rng, k, bg):
    """Single-hue colours stepped in lightness (>= 18 L* apart), each passing 3.3:1 contrast on bg."""
    from skimage import color as skc
    hue = np.deg2rad(HUES[str(rng.choice(list(HUES)))])
    cols = []
    for L in range(20, 92, 2):
        rgb = np.clip(skc.lab2rgb(np.array([[[L, 45 * np.cos(hue), 45 * np.sin(hue)]]]))[0, 0], 0, 1)
        h = matplotlib.colors.to_hex(rgb)
        if ok_contrast(h, bg, 3.3):
            cols.append((L, h))
    if len(cols) < 2:
        return None
    lo, hi = cols[0][0], cols[-1][0]
    kk = int(min(k, (hi - lo) // 18 + 1))
    if kk < 2:
        return None
    targets = np.linspace(lo, hi, kk)
    return [min(cols, key=lambda t: abs(t[0] - tv))[1] for tv in targets]


def safe_palette(rng, k, bg):
    """CVD-safe palette whose every colour also passes 3.3:1 contrast on bg (k shrinks if the family has fewer)."""
    for fam in rng.permutation(["okabe", "lightness"]):
        if fam == "lightness":
            cols = lightness_palette(rng, k, bg)
            if cols:
                return cols, "lightness"
            continue
        cands = [c.lower() for c in OKABE if ok_contrast(c, bg, 3.3)]
        if len(cands) >= 2:
            kk = min(k, len(cands))
            return list(rng.choice(cands, kk, replace=False)), "okabe"
    raise RuntimeError("no safe palette for " + bg)


def unsafe_palette(rng, k, bg):
    """Contains a textbook confusable pair (both colours pass 3.3:1 on bg) plus fillers."""
    pairs = [p for p in FAIL_PAIRS if ok_contrast(p[0], bg, 3.3) and ok_contrast(p[1], bg, 3.3)]
    a, b = pairs[rng.integers(len(pairs))]
    pool = [c for c in TAB10 + SET1 if c not in (a, b) and ok_contrast(c, bg, 3.3)]
    cols = [a, b]
    for c in rng.permutation(pool):               # fillers must be clearly different under normal vision (dE2000 >= 15)
        if len(cols) >= k:
            break
        if all(de2000(to_lab(hex2rgb(c)[None]), to_lab(hex2rgb(o)[None]))[0] >= 15 for o in cols):
            cols.append(str(c))
    cols = [c.lower() for c in cols]
    rng.shuffle(cols)
    return cols, "unsafe_pair"


def trap_palette(rng, k, bg, theme):
    """Safe palette with one deliberately low-contrast colour (truth: contrast failure)."""
    cols, fam = safe_palette(rng, max(2, k - 1), bg)
    bad = "#f0e442" if theme != "dark" else "#2b2b6b"
    cols = [c for c in cols if c != bad][: max(2, k - 1)] + [bad]
    rng.shuffle(cols)
    return cols, fam + "+lowcontrast"


def save_img(fig, path, jpeg_q):
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    img = Image.fromarray(arr)
    if jpeg_q:
        path = path.with_suffix(".jpg")
        img.save(path, quality=int(jpeg_q))
    else:
        img.save(path)
    return path


def style_axes(ax, bg, fg, grid_col, grid):
    ax.set_facecolor(bg)
    for s in ax.spines.values():
        s.set_color(fg)
    ax.tick_params(colors=fg)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    if grid:
        ax.grid(True, color=grid_col, linewidth=0.8)
        ax.set_axisbelow(True)


def render_categorical(rng, kind, cols, theme, redundancy, legend, size, grid, path, jpeg_q):
    bg, fg, gc = THEMES[theme]
    k = len(cols)
    fig, ax = plt.subplots(figsize=size, dpi=100, facecolor=bg)
    style_axes(ax, bg, fg, gc, grid)
    n_mark = k if redundancy == "full" else (k // 2 if redundancy == "partial" else 0)
    cues = {"marker_shapes": 0, "patterns": 0, "direct_labels": 0}
    names = [f"Series {chr(65 + i)}" for i in range(k)]
    if kind == "bar":
        groups = 4
        w = 0.8 / k
        for i, c in enumerate(cols):
            vals = rng.uniform(3, 10, groups)
            hatch = HATCHES[i] if i < n_mark else None
            ax.bar(np.arange(groups) + i * w, vals, w, color=c, label=names[i], hatch=hatch,
                   edgecolor=fg if hatch else c, linewidth=0.6)
        cues["patterns"] = n_mark
    elif kind == "line":
        x = np.arange(24)
        for i, c in enumerate(cols):
            y = np.cumsum(rng.normal(0, 1, 24)) + i * 2
            mk = MARKERS[i] if i < n_mark else None
            ax.plot(x, y, color=c, lw=2.5, marker=mk, markersize=6, markevery=3, label=names[i])
            if redundancy == "full":
                ax.text(x[-1] + 0.3, y[-1], names[i], color=fg, fontsize=8, va="center")
        cues["marker_shapes"] = n_mark
    else:
        for i, c in enumerate(cols):
            cx, cy = rng.uniform(0, 10, 2)
            pts = rng.normal([cx, cy], 1.1, (25, 2))
            mk = MARKERS[i] if i < n_mark else "o"
            ax.scatter(pts[:, 0], pts[:, 1], s=45, color=c, marker=mk, label=names[i])
            if redundancy == "full":
                ax.text(cx, cy + 2.2, names[i], color=fg, fontsize=8, ha="center")
        cues["marker_shapes"] = n_mark
    if redundancy == "full":
        cues["direct_labels"] = k
    if legend:
        ax.legend(frameon=False, labelcolor=fg, fontsize=8, loc="upper left")
    ax.set_xlabel("x"); ax.set_ylabel("value")
    fig.tight_layout()
    return save_img(fig, path, jpeg_q), cues


def render_single(rng, kind, col, theme, size, grid, path, jpeg_q):
    bg, fg, gc = THEMES[theme]
    fig, ax = plt.subplots(figsize=size, dpi=100, facecolor=bg)
    style_axes(ax, bg, fg, gc, grid)
    if kind == "bar":
        ax.bar(np.arange(6), rng.uniform(3, 10, 6), color=col)
    elif kind == "line":
        ax.plot(np.arange(24), np.cumsum(rng.normal(0, 1, 24)), color=col, lw=2.5)
    else:
        ax.scatter(*rng.normal(5, 1.5, (2, 40)), s=45, color=col)
    fig.tight_layout()
    return save_img(fig, path, jpeg_q)


def render_heatmap(rng, cmap, theme, legend, size, path, jpeg_q):
    bg, fg, gc = THEMES[theme]
    field = gaussian_filter(rng.normal(size=(40, 60)), 4)
    fig, ax = plt.subplots(figsize=size, dpi=100, facecolor=bg)
    ax.set_facecolor(bg)
    im = ax.imshow(field, cmap=cmap, interpolation="nearest", aspect="auto")
    ax.tick_params(colors=fg)
    if legend:
        cb = fig.colorbar(im, ax=ax)
        cb.ax.tick_params(colors=fg)
    fig.tight_layout()
    return save_img(fig, path, jpeg_q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    (out / "charts").mkdir(parents=True, exist_ok=True)
    manifest = []

    def common():
        theme = str(rng.choice(["white", "lightgray", "dark"], p=[0.5, 0.2, 0.3]))
        size = [(5, 3.4), (6, 4), (7, 4.5)][rng.integers(3)]
        grid = bool(rng.random() < 0.5)
        jpeg_q = int(rng.integers(60, 91)) if rng.random() < 0.3 else 0
        kind = str(rng.choice(["bar", "line", "scatter"]))
        return theme, size, grid, jpeg_q, kind

    def add(entry):
        entry["id"] = f"c{len(manifest):04d}"
        manifest.append(entry)
        return entry["id"]

    plan = ([("safe", None)] * 90 + [("unsafe", "none")] * 90 + [("unsafe", "full")] * 35 +
            [("unsafe", "partial")] * 30 + [("trap", None)] * 40)
    for fam_kind, red in plan:
        theme, size, grid, jpeg_q, kind = common()
        bg = THEMES[theme][0]
        k = int(rng.integers(2, 7)) if fam_kind == "unsafe" else int(rng.integers(3, 7))
        if fam_kind == "safe":
            cols, fam = safe_palette(rng, k, bg)
        elif fam_kind == "unsafe":
            cols, fam = unsafe_palette(rng, k, bg)
        else:
            cols, fam = trap_palette(rng, k, bg, theme)
        k = len(cols)
        red = red or "none"
        cid = f"c{len(manifest):04d}"
        path, cues = render_categorical(rng, kind, cols, theme, red, True, size, grid, out / "charts" / f"{cid}.png", jpeg_q)
        contrast_fail = bool(np.any(contrast_ratio(np.array([hex2rgb(c) for c in cols]), hex2rgb(bg)[None, :]) < 3.0))
        unsafe = fam_kind == "unsafe"
        if contrast_fail:
            truth, issue = "Fail", "contrast"
        elif unsafe and red != "full":
            truth, issue = "Fail", "cvd"
        else:
            truth, issue = "Pass", "ok"
        add({"file": path.name, "type": kind, "family": fam, "palette": cols, "bg": bg, "theme": theme,
             "redundancy": red, "cues": cues, "legend_readable": True, "ramp": None, "ramp_kind": None,
             "truth": truth, "issue": issue, "jpeg_q": jpeg_q, "k": k})

    for _ in range(30):                     # single colour -> Not Present
        theme, size, grid, jpeg_q, kind = common()
        col = str(rng.choice(["#1f4e79", "#8c2d04"] if theme != "dark" else ["#9ecae1", "#fdd0a2"]))
        cid = f"c{len(manifest):04d}"
        path = render_single(rng, kind, col, theme, size, grid, out / "charts" / f"{cid}.png", jpeg_q)
        add({"file": path.name, "type": kind, "family": "single", "palette": [col], "bg": THEMES[theme][0],
             "theme": theme, "redundancy": "none", "cues": {}, "legend_readable": True, "ramp": None,
             "ramp_kind": None, "truth": "Not Present", "issue": "single colour", "jpeg_q": jpeg_q, "k": 1})

    for _ in range(20):                     # legend withheld -> Cannot Tell
        theme, size, grid, jpeg_q, kind = common()
        k = int(rng.integers(3, 6))
        bg0 = THEMES[theme][0]
        cols, fam = (safe_palette(rng, k, bg0) if rng.random() < 0.5 else unsafe_palette(rng, k, bg0))
        k = len(cols)
        cid = f"c{len(manifest):04d}"
        path, cues = render_categorical(rng, kind, cols, theme, "none", False, size, grid, out / "charts" / f"{cid}.png", jpeg_q)
        add({"file": path.name, "type": kind, "family": fam, "palette": cols, "bg": THEMES[theme][0],
             "theme": theme, "redundancy": "none", "cues": cues, "legend_readable": False, "ramp": None,
             "ramp_kind": None, "truth": "Cannot Tell", "issue": "legend withheld", "jpeg_q": jpeg_q, "k": k})

    heat = ([(c, k, True, "Pass") for c, k in SAFE_CMAPS.items() for _ in range(12)] +
            [(c, k, True, "Fail") for c, k in UNSAFE_CMAPS.items() for _ in range(10)] +
            [(c, k, False, "Cannot Tell") for c, k in {**SAFE_CMAPS, **UNSAFE_CMAPS}.items() for _ in range(4)])
    for cmap, kind_g, legend, truth in heat:
        theme, size, grid, jpeg_q, _ = common()
        cid = f"c{len(manifest):04d}"
        path = render_heatmap(rng, cmap, theme, legend, size, out / "charts" / f"{cid}.png", jpeg_q)
        ramp = [matplotlib.colors.to_hex(colormaps[cmap](v)) for v in np.linspace(0, 1, 32)] if legend else None
        add({"file": path.name, "type": "heatmap", "family": cmap, "palette": [], "bg": THEMES[theme][0],
             "theme": theme, "redundancy": "none", "cues": {}, "legend_readable": legend, "ramp": ramp,
             "ramp_kind": None, "truth": truth, "issue": "gradient" if legend else "legend withheld",
             "jpeg_q": jpeg_q, "k": 0})

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    from collections import Counter
    print(f"wrote {len(manifest)} charts to {out}/")
    print("truth labels:", dict(Counter(m["truth"] for m in manifest)))
    print("issues:", dict(Counter(m["issue"] for m in manifest)))


if __name__ == "__main__":
    main()
