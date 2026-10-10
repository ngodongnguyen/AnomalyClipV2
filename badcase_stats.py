"""
EXP-032 / H022: pure numpy/scipy statistics for the per-image bad-case characterisation of the deployed ECP (no torch).
Rules are fixed in research/EXPERIMENTS.md EXP-032; constants below mirror them and must not be tuned after seeing numbers.
"""
import math
import numpy as np
from scipy import ndimage as ndi

TPR_TARGET = 0.80
FAR_PX = 28.0               # far-FP: distance to the lesion > 28 px (as EXP-025)
MIN_COMP_AREA = 100         # far-FP components smaller than this are ignored (speckle)
SPEC_FRAC, SPEC_V, SPEC_S = 0.20, 0.90, 0.25
DARK_V = 0.15
BORDER_FRAC = 0.08
TOP_DECILE = 0.10
MULTI_N = 3
BIGFP_RATIO = 2.0
COLD_PCT = 0.60
WORST_FRAC = 0.10
RATIO_MIN, LO_MIN, SETS_MIN = 2.0, 1.0, 3
GATED_TAGS = ("SPEC", "DARK", "BORDER", "EDGEY", "REDDISH", "MULTI", "BIGFP", "COLDLESION")
DESCRIPTIVE_TAGS = ("SMALL", "LARGE")        # strata, excluded from the gate
ALL_TAGS = GATED_TAGS + DESCRIPTIVE_TAGS
NEAR_DEFINITIONAL = ("COLDLESION", "BIGFP")  # flagged in the printout


# ---------------------------------------------------------------- operating point and components
def threshold_at_tpr(pos_scores, tpr=TPR_TARGET):
    """Largest t such that mean(pos_scores >= t) >= tpr (predicted positive = score >= t)."""
    p = np.asarray(pos_scores, float).ravel()
    n = p.size
    if n == 0:
        raise ValueError("no positive scores")
    k = int(math.ceil(tpr * n - 1e-9))        # need at least k positives at or above t
    k = min(max(k, 1), n)
    return float(np.partition(p, n - k)[n - k])


def distance_to_lesion(gt):
    """Euclidean distance (px) of every pixel to the nearest lesion pixel (0 inside the lesion)."""
    return ndi.distance_transform_edt(~np.asarray(gt, bool))


def far_fp_components(score, thr, dist, far=FAR_PX, min_area=MIN_COMP_AREA):
    """Far-FP pixels = score >= thr and dist > far. Returns (far_fp_mask, [components largest first]); each component is a dict
    with mask index (label), area, mean_score, cy, cx.  4-connectivity; components under min_area are dropped from the list
    (the mask still contains their pixels, only the component list is filtered)."""
    fp = (np.asarray(score) >= thr) & (np.asarray(dist) > far)
    lab, n = ndi.label(fp)
    comps = []
    if n:
        idx = np.arange(1, n + 1)
        areas = ndi.sum(fp, lab, idx)
        for i, a in zip(idx, areas):
            if a >= min_area:
                ys, xs = np.nonzero(lab == i)
                comps.append({"label": int(i), "area": int(a), "mean_score": float(np.asarray(score)[ys, xs].mean()),
                              "cy": float(ys.mean()), "cx": float(xs.mean())})
        comps.sort(key=lambda c: (-c["area"], c["label"]))
    return fp, lab, comps


def border_dist_frac(cy, cx, h, w):
    """Distance of a centroid to the nearest image border as a fraction of the shorter side."""
    return float(min(cy, h - 1 - cy, cx, w - 1 - cx) / min(h, w))


def lesion_pct_median(score, gt):
    """Median over lesion pixels of their rank percentile among all pixel scores of the image (searchsorted side right / N)."""
    s = np.asarray(score, float).ravel()
    srt = np.sort(s)
    lz = np.asarray(score, float)[np.asarray(gt, bool)]
    return float(np.median(np.searchsorted(srt, lz, side="right") / float(s.size)))


# ---------------------------------------------------------------- image statistics (RGB in [0,1])
def value_saturation(rgb):
    rgb = np.asarray(rgb, float)
    v = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    s = np.where(v > 0, (v - mn) / np.maximum(v, 1e-12), 0.0)
    return v, s


def sobel_magnitude(rgb):
    g = np.asarray(rgb, float).mean(axis=-1)
    return np.hypot(ndi.sobel(g, axis=0), ndi.sobel(g, axis=1))


def redness(rgb):
    rgb = np.asarray(rgb, float)
    return rgb[..., 0] - 0.5 * (rgb[..., 1] + rgb[..., 2])


def spec_fraction(v, s, comp_mask):
    m = np.asarray(comp_mask, bool)
    return float(((v[m] > SPEC_V) & (s[m] < SPEC_S)).mean())


NAN = float("nan")
REC_COLS = ["id", "area_frac", "quartile", "auroc", "hit_rate", "fp_frac", "n_comp", "comp_area", "comp_mean_score", "comp_cy", "comp_cx",
            "comp_border", "comp_spec", "comp_v", "comp_sobel", "comp_red", "lesion_pct", "lesion_px"]


def image_record(rgb, score, gt, thr, far=FAR_PX, min_area=MIN_COMP_AREA, dist=None):
    """Raw per-image numbers (no tags; EDGEY/REDDISH need the dataset). NaN component fields when no far-FP component exists."""
    gt = np.asarray(gt, bool)
    h, w = gt.shape
    if dist is None:
        dist = distance_to_lesion(gt)
    fp, lab, comps = far_fp_components(score, thr, dist, far, min_area)
    pred = np.asarray(score) >= thr
    rec = {"area_frac": float(gt.mean()), "hit_rate": float((pred & gt).sum() / max(gt.sum(), 1)), "fp_frac": float(fp.mean()),
           "n_comp": len(comps), "lesion_pct": lesion_pct_median(score, gt), "lesion_px": int(gt.sum())}
    keys = ("comp_area", "comp_mean_score", "comp_cy", "comp_cx", "comp_border", "comp_spec", "comp_v", "comp_sobel", "comp_red")
    for k in keys:
        rec[k] = NAN
    if comps:
        c = comps[0]
        m = lab == c["label"]
        v, s = value_saturation(rgb)
        rec.update(comp_area=float(c["area"]), comp_mean_score=c["mean_score"], comp_cy=c["cy"], comp_cx=c["cx"],
                   comp_border=border_dist_frac(c["cy"], c["cx"], h, w), comp_spec=spec_fraction(v, s, m), comp_v=float(v[m].mean()),
                   comp_sobel=float(sobel_magnitude(rgb)[m].mean()), comp_red=float(redness(rgb)[m].mean()))
    return rec, fp, lab, comps


# ---------------------------------------------------------------- tags
def top_decile_mask(values, frac=TOP_DECILE):
    """k = ceil(frac * n_finite) largest finite values (ties by earlier index); NaN never selected."""
    v = np.asarray(values, float)
    fin = np.nonzero(np.isfinite(v))[0]
    out = np.zeros(v.size, bool)
    if fin.size == 0:
        return out
    k = int(math.ceil(frac * fin.size - 1e-9))
    order = fin[np.argsort(-v[fin], kind="stable")]
    out[order[:k]] = True
    return out


def assign_tags(cols):
    """cols: dict of arrays with the REC_COLS names. Returns {tag: bool array}. Component tags are False without a component."""
    n_comp = np.asarray(cols["n_comp"], float)
    has = n_comp >= 1
    f = lambda k: np.asarray(cols[k], float)
    with np.errstate(invalid="ignore"):
        tags = {
            "SPEC": has & (f("comp_spec") >= SPEC_FRAC),
            "DARK": has & (f("comp_v") < DARK_V),
            "BORDER": has & (f("comp_border") < BORDER_FRAC),
            "EDGEY": has & top_decile_mask(np.where(has, f("comp_sobel"), NAN)),
            "REDDISH": has & top_decile_mask(np.where(has, f("comp_red"), NAN)),
            "MULTI": n_comp >= MULTI_N,
            "BIGFP": has & (f("comp_area") >= BIGFP_RATIO * f("lesion_px")),
            "COLDLESION": f("lesion_pct") < COLD_PCT,
        }
    q = np.asarray(cols["quartile"])
    tags["SMALL"] = q == 0
    tags["LARGE"] = q == 3
    return tags


# ---------------------------------------------------------------- prevalence ratio with image bootstrap
def worst_set(values, frac=WORST_FRAC, largest=False):
    """Boolean mask of the k = ceil(frac n) worst images: lowest values (default) or largest; ties by earlier index."""
    v = np.asarray(values, float)
    k = int(math.ceil(frac * v.size - 1e-9))
    order = np.argsort(-v if largest else v, kind="stable")
    m = np.zeros(v.size, bool)
    m[order[:k]] = True
    return m


def ratio_from_counts(a, nw, b, nr):
    """Prevalence ratio (a/nw)/(b/nr). Both counts 0 -> 1.0; exactly one 0 -> add 0.5 to all four cells of the 2x2 table."""
    if a == 0 and b == 0:
        return 1.0
    if a == 0 or b == 0:
        return float(((a + 0.5) / (nw + 1.0)) / ((b + 0.5) / (nr + 1.0)))
    return float((a / nw) / (b / nr))


def prevalence_table(values, tagmat, largest=False, frac=WORST_FRAC):
    """tagmat [N, K] bool. Returns (counts_worst, n_worst, counts_rest, n_rest, ratios[K])."""
    w = worst_set(values, frac, largest)
    a = tagmat[w].sum(0)
    b = tagmat[~w].sum(0)
    nw, nr = int(w.sum()), int((~w).sum())
    return a, nw, b, nr, np.array([ratio_from_counts(int(x), nw, int(y), nr) for x, y in zip(a, b)])


def bootstrap_ratio(values, tagmat, largest=False, frac=WORST_FRAC, B=2000, seed=0):
    """Image bootstrap; the worst decile is re-selected inside each resample. Returns (lo[K], hi[K]) 2.5/97.5 percentiles."""
    v = np.asarray(values, float)
    T = np.asarray(tagmat, bool)
    n = v.size
    rng = np.random.RandomState(seed)
    K = T.shape[1]
    out = np.empty((B, K))
    for b in range(B):
        idx = rng.randint(0, n, n)
        a, nw, c, nr, r = prevalence_table(v[idx], T[idx], largest, frac)
        out[b] = r
    return np.percentile(out, 2.5, axis=0), np.percentile(out, 97.5, axis=0)


def analyse_set(auroc, tagmat, largest=False, B=2000, seed=0):
    a, nw, b, nr, r = prevalence_table(auroc, tagmat, largest)
    lo, hi = bootstrap_ratio(auroc, tagmat, largest, B=B, seed=seed)
    return {"a": a, "nw": nw, "b": b, "nr": nr, "ratio": r, "lo": lo, "hi": hi}


def passes(res):
    return (res["ratio"] >= RATIO_MIN) & (res["lo"] > LO_MIN)


def candidate_verdict(per_set_pass, tags):
    """per_set_pass: {set: bool array over `tags`}. Returns ({tag: n_sets_passing}, [candidate tags])."""
    cnt = {t: int(sum(bool(p[i]) for p in per_set_pass.values())) for i, t in enumerate(tags)}
    return cnt, [t for t in tags if t in GATED_TAGS and cnt[t] >= SETS_MIN]


# ---------------------------------------------------------------- image-level sets
def deterministic_rank(ids, scores):
    """1-based rank, score descending, ties by id ascending."""
    order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), str(ids[i])))
    rank = np.empty(len(ids), int)
    for r, i in enumerate(order, 1):
        rank[i] = r
    return rank


def youden_counts(labels, scores):
    """TP, FP, FN, TN at the threshold (score >= t positive) maximising Youden J = TPR - FPR; first maximiser (largest t) on ties."""
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, float)
    P, N = max(int(y.sum()), 1), max(int((1 - y).sum()), 1)
    best = (-2.0, None)
    for t in np.unique(s)[::-1]:
        pos = s >= t
        j = (pos & (y == 1)).sum() / P - (pos & (y == 0)).sum() / N
        if j > best[0] + 1e-12:
            best = (j, t)
    t = best[1]
    pos = s >= t
    return {"thr": float(t), "TP": int((pos & (y == 1)).sum()), "FP": int((pos & (y == 0)).sum()),
            "FN": int((~pos & (y == 1)).sum()), "TN": int((~pos & (y == 0)).sum())}


def score_summary(labels, scores):
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, float)
    a, n = s[y == 1], s[y == 0]
    lo, hi = float(np.percentile(a, 5)), float(np.percentile(n, 95))
    inside = ((s >= lo) & (s <= hi)).mean() if lo <= hi else 0.0
    return {"median_normal": float(np.median(n)), "median_anomalous": float(np.median(a)), "p5_anomalous": lo, "p95_normal": hi,
            "overlap_interval": [lo, hi] if lo <= hi else None, "overlap_fraction_of_images": float(inside), **youden_counts(y, s)}
