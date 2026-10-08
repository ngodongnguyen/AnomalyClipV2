"""Torch-free helpers for analyze_headroom.py (EXP-025 / H016: oracle error decomposition of ECP's pixel maps).
numpy/scipy only; unit-tested in test_headroom_stats.py.  GT is used ONLY to build diagnostic oracle edits; no method is implied.

Zones (signed distance d to the GT boundary, output pixels; d > 0 outside the lesion, d <= 0 inside):
  FAR  : d > far          RIM : 0 < d <= far          EDGE : -edge <= d <= 0          CORE : d < -edge
Outside pixels use the Euclidean distance to the nearest lesion pixel; inside pixels use -(Euclidean distance to the nearest
background pixel) (>= 1 pixel, so the innermost boundary layer has d = -1).
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

FAR, RIM, EDGE, CORE = 0, 1, 2, 3
ZONE_NAMES = ("FAR", "RIM", "EDGE", "CORE")
EDITS = ("FAR_FIX", "RIM_FIX", "CORE_FIX", "EDGE_FIX", "PERFECT")   # EDGE_FIX is report-only (not in the decision rule)
RULE_MODES = ("FAR_FIX", "RIM_FIX", "CORE_FIX")
MIN_CLOSURE = 0.30
MIN_DATASETS = 3


def signed_distance(gt):
    gt = np.asarray(gt, bool)
    if not gt.any() or gt.all():
        raise ValueError("mask needs both lesion and background pixels")
    return np.where(gt, -distance_transform_edt(gt), distance_transform_edt(~gt))


def zone_map(gt, far=28, edge=7):
    """int8 [H, W] with values FAR/RIM/EDGE/CORE; the four zones partition the image."""
    d = signed_distance(gt)
    z = np.empty(d.shape, np.int8)
    z[d > far] = FAR
    z[(d > 0) & (d <= far)] = RIM
    z[(d <= 0) & (d >= -edge)] = EDGE
    z[d < -edge] = CORE
    return z


def edit_values(edit, gmin, gmax):
    """list of (zone, value) assignments of one oracle edit (each edit is applied alone to the baseline map)."""
    if edit == "FAR_FIX":
        return [(FAR, gmin)]
    if edit == "RIM_FIX":
        return [(RIM, gmin)]
    if edit == "CORE_FIX":
        return [(CORE, gmax)]
    if edit == "EDGE_FIX":
        return [(EDGE, gmax)]
    if edit == "PERFECT":
        return [(FAR, gmin), (RIM, gmin), (EDGE, gmax), (CORE, gmax)]
    raise ValueError(edit)


def apply_edit(buf, base, zones, edit, gmin, gmax):
    """buf <- base with the edit applied (in place into a preallocated buffer; no extra dataset-sized allocation but one bool temp)."""
    np.copyto(buf, base)
    for z, v in edit_values(edit, gmin, gmax):
        buf[zones == z] = v
    return buf


def gap_closure(metric_edit, metric_base):
    """(edit - base) / (1 - base); NaN when base is already 1 or either value is NaN."""
    if metric_base is None or metric_edit is None or not np.isfinite(metric_base) or not np.isfinite(metric_edit) or metric_base >= 1.0:
        return float("nan")
    return float((metric_edit - metric_base) / (1.0 - metric_base))


def quartile_index(areas):
    """0..3 (small -> large lesion) from quantile bins of the area within the dataset."""
    a = np.asarray(areas, float)
    return np.digitize(a, np.quantile(a, [0.25, 0.5, 0.75]))


def dominant_mode(closures, min_closure=MIN_CLOSURE):
    """closures: {mode: closure} for RULE_MODES. The mode with the largest closure, counted only if >= min_closure; else None."""
    c = {m: closures[m] for m in RULE_MODES if m in closures and np.isfinite(closures[m])}
    if not c:
        return None
    m = max(c, key=c.get)
    return m if c[m] >= min_closure else None


def decide_targets(dom_all, dom_q1, min_datasets=MIN_DATASETS):
    """dom_all / dom_q1: {dataset: mode or None}. A mode is a TARGET if dominant on >= min_datasets datasets overall,
    or dominant in the smallest-area quartile on >= min_datasets datasets. Returns (targets_all, targets_q1) as sorted lists."""
    def count(d):
        out = {}
        for v in d.values():
            if v is not None:
                out[v] = out.get(v, 0) + 1
        return sorted(m for m, n in out.items() if n >= min_datasets)
    return count(dom_all), count(dom_q1)


def pooled_metrics(gt, base, zones, auroc_fn, edits=EDITS, idx=None):
    """Pooled AUROC of the baseline and of every edit on (a subset of) the images. Test helper and small-data path: the driver
    streams its own loop to keep memory low. gt/base/zones: [N, H, W]."""
    gmin, gmax = float(base.min()), float(base.max())
    sel = slice(None) if idx is None else idx
    out = {"base": float(auroc_fn(gt[sel], base[sel]))}
    buf = np.empty_like(base)
    for e in edits:
        apply_edit(buf, base, zones, e, gmin, gmax)
        out[e] = float(auroc_fn(gt[sel], buf[sel]))
    return out
