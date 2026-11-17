"""
cvdaudit.py - core library for the CVD dashboard audit (Phase 3).

Contents
  1. CVD simulation: Machado et al. (2009) for protan/deutan/tritan (severity 0-1),
     Vienot et al. (1999) for protan/deutan (severity 1.0).
  2. Palette extraction from a rendered chart (histogram + leader clustering in CIELAB).
  3. Features: pairwise CIEDE2000 under normal and simulated vision, WCAG contrast, gradient checks.
  4. Decisions: hybrid audit (perception + structure modules) and baselines B1, B2.

All colours are sRGB floats in [0, 1] unless a function says uint8.
"""
import numpy as np
from itertools import combinations
from scipy import ndimage as ndi
from skimage import color

CVDS = ("protan", "deutan", "tritan")

# --- 1. Simulation ---------------------------------------------------------------------------
# Machado, Oliveira & Fernandes (2009), severity 1.0, applied to LINEAR RGB.
MACHADO_100 = {
    "protan": np.array([[0.152286, 1.052583, -0.204868],
                        [0.114503, 0.786281, 0.099216],
                        [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968],
                        [0.280085, 0.672501, 0.047413],
                        [-0.011820, 0.042940, 0.968881]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779],
                        [-0.078411, 0.930809, 0.147602],
                        [0.004733, 0.691367, 0.303900]]),
}

# Vienot, Brettel & Mollon (1999): project in LMS, return to linear RGB.
_RGB2LMS = np.array([[17.8824, 43.5161, 4.11935],
                     [3.45565, 27.1554, 3.86714],
                     [0.0299566, 0.184309, 1.46709]])
_P = {"protan": np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]]),
      "deutan": np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]])}
VIENOT = {k: np.linalg.inv(_RGB2LMS) @ p @ _RGB2LMS for k, p in _P.items()}


def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def simulate(rgb, cvd, model="machado", severity=1.0):
    """Simulate colour-vision deficiency. rgb: (..., 3) sRGB in [0,1]."""
    if cvd == "normal":
        return np.asarray(rgb, float)
    if model == "machado":
        M = (1 - severity) * np.eye(3) + severity * MACHADO_100[cvd]
    elif model == "vienot":
        if cvd not in VIENOT:
            raise ValueError("Vienot model is defined for protan/deutan only")
        M = VIENOT[cvd]
    else:
        raise ValueError(model)
    lin = srgb_to_linear(rgb)
    return linear_to_srgb(lin @ M.T)


def to_lab(rgb):
    return color.rgb2lab(np.asarray(rgb, float).reshape(-1, 1, 3)).reshape(-1, 3)


def de2000(lab1, lab2):
    return color.deltaE_ciede2000(lab1, lab2)


def de76(lab1, lab2):
    return np.linalg.norm(lab1 - lab2, axis=-1)


def pair_distances(rgb, cvd="normal", model="machado", severity=1.0, metric="ciede2000"):
    """Distances for every colour pair (i<j) of a palette, in order of itertools.combinations."""
    sim = simulate(rgb, cvd, model, severity)
    idx = list(combinations(range(len(sim)), 2))
    if not idx:
        return np.array([])
    ii, jj = [i for i, _ in idx], [j for _, j in idx]
    if metric == "rgb":                      # Euclidean distance in sRGB (0-100 scale), used only for the ablation
        return np.linalg.norm((sim[ii] - sim[jj]) * 100, axis=-1)
    lab = to_lab(sim)
    return de2000(lab[ii], lab[jj]) if metric == "ciede2000" else de76(lab[ii], lab[jj])


# --- 2. Palette extraction -------------------------------------------------------------------
def extract_palette(img_u8, min_share=0.01, merge_de=6.0, kmax=12, tint_ratio=0.5, tint_de=5.0,
                    bg_de=14.0, neutral_min_share=0.12, denoise=True, bin_shift=3, core=True, core_tol=12):
    """
    Palette of the data marks in a rendered chart image (H x W x 3 uint8).
    1) background = most frequent colour; pixels within dE2000 < bg_de of it are dropped (also absorbs JPEG ringing)
    2) dark text and light gridlines (low chroma at the lightness extremes) are dropped
    3) after a 3x3 median filter and keeping only flat-interior (core) pixels, remaining pixels are binned (32 levels per channel); bins holding >= min_share of the
       remaining pixels are kept (flat fills concentrate in few bins, anti-aliased edge tints
       spread over many small bins)
    4) kept bins are merged greedily when closer than merge_de (CIEDE2000)
    5) anti-aliasing tints are dropped: a colour that lies (within tint_de) on the blend line between the
       background and a larger palette colour, and holds less than tint_ratio of that colour's pixels,
       is an edge blend and not a category colour
    Returns (palette_rgb [k,3] floats, shares [k], background_rgb [3]).
    """
    img = np.asarray(img_u8)[..., :3]
    if denoise:                                    # 3x3 median removes JPEG speckle, keeps flat fills and 2px+ lines
        img = ndi.median_filter(img, size=(3, 3, 1))
    if core:                                       # keep flat-interior pixels only: drops text strokes, hairlines, edge blends
        mx = ndi.maximum_filter(img, size=(2, 2, 1)).astype(int)
        mn = ndi.minimum_filter(img, size=(2, 2, 1)).astype(int)
        mask = ((mx - mn) <= core_tol).all(axis=-1)
        img = img[mask] if mask.sum() >= 200 else img.reshape(-1, 3)
    else:
        img = img.reshape(-1, 3)
    q = (img >> bin_shift).astype(np.int32)
    keys = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    ukeys, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    # exact mean colour per bin
    sums = np.zeros((len(ukeys), 3))
    np.add.at(sums, inv, img.astype(float))
    centers = sums / counts[:, None] / 255.0
    lab = to_lab(centers)

    bg_i = int(np.argmax(counts))
    bg_lab = lab[bg_i]
    d_bg = de2000(lab, np.tile(bg_lab, (len(lab), 1)))
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    neutral_text = (chroma < 8) & ((lab[:, 0] < 30) | (lab[:, 0] > 82))
    keep = (d_bg >= bg_de) & ~neutral_text
    total = counts[keep].sum()
    if total == 0:
        return np.zeros((0, 3)), np.array([]), centers[bg_i]
    keep &= counts >= max(20, min_share * total)

    order = np.argsort(-counts)
    cl_lab, cl_w, cl_rgb = [], [], []
    for i in order:
        if not keep[i]:
            continue
        for c in range(len(cl_lab)):
            if de2000(lab[i][None], cl_lab[c][None])[0] < merge_de:
                w = cl_w[c] + counts[i]
                cl_rgb[c] = (cl_rgb[c] * cl_w[c] + centers[i] * counts[i]) / w
                cl_lab[c] = to_lab(cl_rgb[c][None])[0]
                cl_w[c] = w
                break
        else:
            cl_lab.append(lab[i].copy())
            cl_rgb.append(centers[i].copy())
            cl_w.append(float(counts[i]))
    cl_rgb, cl_w = np.array(cl_rgb), np.array(cl_w)
    bg_rgb = centers[bg_i]
    alphas = np.linspace(0.05, 0.95, 19)[:, None]
    drop = np.zeros(len(cl_rgb), bool)
    for i in np.argsort(cl_w):                      # smallest first
        for j in range(len(cl_rgb)):
            if i == j or drop[j] or cl_w[i] > tint_ratio * cl_w[j]:
                continue
            hit = False
            for e in (bg_rgb,):                      # blend of cluster j with the background
                blend = alphas * cl_rgb[j] + (1 - alphas) * e
                if de2000(to_lab(blend), np.tile(to_lab(cl_rgb[i][None]), (len(blend), 1))).min() < tint_de:
                    hit = True
                    break
            if hit:
                drop[i] = True
                break
    cl_rgb, cl_w = cl_rgb[~drop], cl_w[~drop]
    # grey clusters with a small share are text anti-aliasing, not a grey data series
    cl_chroma = np.hypot(*to_lab(cl_rgb)[:, 1:].T) if len(cl_rgb) else np.array([])
    keep_c = ~((cl_chroma < 6) & (cl_w / max(cl_w.sum(), 1) < neutral_min_share))
    cl_rgb, cl_w = cl_rgb[keep_c], cl_w[keep_c]
    top = np.argsort(-cl_w)[:kmax]
    return cl_rgb[top], cl_w[top] / cl_w.sum(), centers[bg_i]


# --- 3. Features -----------------------------------------------------------------------------
def rel_luminance(rgb):
    lin = srgb_to_linear(rgb)
    return lin @ np.array([0.2126, 0.7152, 0.0722])


def contrast_ratio(rgb1, rgb2):
    l1, l2 = rel_luminance(rgb1), rel_luminance(rgb2)
    hi, lo = np.maximum(l1, l2), np.minimum(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def palette_features(pal, bg, model="machado", severity=1.0, cvds=CVDS, metric="ciede2000"):
    """Per-pair distances under normal vision and each simulated condition."""
    f = {"d_norm": pair_distances(pal, "normal", metric=metric)}
    for c in cvds:
        m, s = model, severity
        if model == "vienot" and c == "tritan":      # Vienot has no tritan model: fall back to Machado
            m, s = "machado", 1.0
        f[c] = pair_distances(pal, c, m, s, metric=metric)
    f["contrast"] = contrast_ratio(pal, np.asarray(bg)[None, :]) if len(pal) else np.array([])
    return f


def categorical_fail(feat, tau, cvds=CVDS):
    """Confusable pair: distinguishable under normal vision (>= tau) but below tau after simulation."""
    for c in cvds:
        if np.any((feat["d_norm"] >= tau) & (feat[c] < tau)):
            return True
    return False


def gradient_fail(ramp, kind, tau, model="machado", severity=1.0, cvds=CVDS, l_tol=2.0):
    """
    ramp: [n,3] colours along a legend. 'sequential': lightness must be monotonic after simulation
    (violations larger than l_tol are failures). 'diverging': the two end colours must stay
    distinguishable (CIEDE2000 >= tau) after simulation.
    """
    ramp = np.asarray(ramp, float)
    for c in ("normal",) + tuple(cvds):
        sim = simulate(ramp, c, model if not (model == "vienot" and c == "tritan") else "machado", severity)
        lab = to_lab(sim)
        if kind == "sequential":
            dl = np.diff(lab[:, 0])
            if not (np.all(dl >= -l_tol) or np.all(dl <= l_tol)):
                return True
        else:
            if de2000(lab[:1], lab[-1:])[0] < tau:
                return True
    return False


# --- 4. Decisions ----------------------------------------------------------------------------
LABELS = ("Pass", "Fail", "Not Present", "Cannot Tell")


def strong_redundancy(cues, n_cat):
    """cues from DOM-like manifest: distinct marker shapes / patterns / direct labels cover every category."""
    if not cues:
        return False
    return bool(cues.get("marker_shapes", 0) >= n_cat or cues.get("patterns", 0) >= n_cat
                or cues.get("direct_labels", 0) >= n_cat)


def hybrid_audit(pal, bg, tau=10.0, cues=None, ramp=None, ramp_kind=None, legend_readable=True,
                 model="machado", severity=1.0, cvds=CVDS, use_structure=True, use_gradient=True,
                 use_contrast=True, metric="ciede2000"):
    """Two-module hybrid audit. Returns (label, reason)."""
    # structure module: unreadable input -> Cannot Tell
    if use_structure and not legend_readable:
        return "Cannot Tell", "legend/DOM unreadable"
    if ramp is not None and use_gradient:
        return ("Fail", "gradient") if gradient_fail(ramp, ramp_kind, tau, model, severity, cvds) else ("Pass", "gradient ok")
    if len(pal) < 2:
        return "Not Present", "fewer than 2 colour categories"
    feat = palette_features(pal, bg, model, severity, cvds, metric)
    cvd_fail = categorical_fail(feat, tau, cvds)
    low_contrast = use_contrast and bool(np.any(feat["contrast"] < 3.0))
    if cvd_fail and use_structure and strong_redundancy(cues, len(pal)) and not low_contrast:
        return "Cannot Tell", "colour-only failure, strong redundancy: human review"
    if cvd_fail or low_contrast:
        return "Fail", "cvd" if cvd_fail else "contrast"
    return "Pass", "ok"


def baseline_b1(pal, bg, ramp=None):
    """B1 contrast-only: WCAG 1.4.11 3:1 check of each mark colour against the background."""
    if ramp is not None:
        return "Pass"                       # no contrast rule applies to a legend-less hue ramp
    if len(pal) < 2:
        return "Not Present"
    return "Fail" if np.any(contrast_ratio(pal, np.asarray(bg)[None, :]) < 3.0) else "Pass"


def baseline_b2(pal, bg, ramp=None, thr=10.0, model="machado"):
    """B2 simulation-only: fixed threshold, any simulated pair below it fails; no calibration or structure."""
    if ramp is not None:
        pal = np.asarray(ramp)[:: max(1, len(ramp) // 8)]
    if len(pal) < 2:
        return "Not Present"
    feat = palette_features(pal, bg, model, 1.0)
    return "Fail" if any(np.any(feat[c] < thr) for c in CVDS) else "Pass"


def infer_ramp_kind(ramp, tol=2.0):
    """Sequential vs diverging, inferred from the lightness profile of a legend ramp (no manifest hint).
    Diverging: both halves are monotonic towards a light/dark centre, and the two ends have similar lightness."""
    L = to_lab(np.asarray(ramp, float))[:, 0]
    n = len(L)
    mid = n // 2
    left, right = L[: mid + 1], L[mid:]
    centre_gap = abs(L[mid] - 0.5 * (L[0] + L[-1]))
    ends_similar = abs(L[0] - L[-1]) < 15
    mono_left = np.all(np.diff(left) >= -tol) or np.all(np.diff(left) <= tol)
    mono_right = np.all(np.diff(right) >= -tol) or np.all(np.diff(right) <= tol)
    opposite = (L[mid] - L[0]) * (L[-1] - L[mid]) < 0
    ext = int(np.argmax(L)) if L[mid] > 0.5 * (L[0] + L[-1]) else int(np.argmin(L))
    centred = abs(ext - mid) <= max(2, 0.1 * n)         # the lightness extreme sits at the middle of the ramp
    return "diverging" if (centre_gap > 25 and ends_similar and mono_left and mono_right and opposite and centred) else "sequential"


def is_continuous(img_u8, min_colours=8, nn_de=12.0):
    """
    True for images dominated by a continuous colour field (heatmap, choropleth, photo): many distinct colours
    (min_share 0.2%) whose median nearest-neighbour CIEDE2000 distance is small. Categorical palettes have
    well-separated colours, so their nearest-neighbour distances are large.
    """
    fine, _, _ = extract_palette(img_u8, min_share=0.002, kmax=40)
    if len(fine) < min_colours:
        return False
    lab = to_lab(fine)
    d = np.array([[de2000(lab[i][None], lab[j][None])[0] if i != j else np.inf for j in range(len(lab))]
                  for i in range(len(lab))])
    return bool(np.median(d.min(axis=1)) < nn_de)
