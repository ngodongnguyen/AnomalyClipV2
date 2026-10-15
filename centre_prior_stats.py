"""
EXP-050 statistics (numpy/scipy only, no torch). Rule and thresholds: notes/EXP-050_registration.md (copy into research/EXPERIMENTS.md before any run).

Part A (source geometry, H039a): does the zoom augmentation of training (dataset.Dataset._sample_crop_box) move MVTec anomalies toward the frame centre
  (A1, centralisation) or tie large extent z to central position (A2, size-centre coupling) relative to the native frames?
Part B (H039b): on the frozen ECP, is the PE position response smaller when the pixel prompt is conditioned on a small extent (z_pix = -0.69) than at the
  deployed z_pix = 1.6?  Uses the EXP-048 / EXP-049 per-image CSVs (z = 1.6) and new CSVs from analyze_pe_shift.py --z_pix -0.69.
"""
import math
import numpy as np
from scipy.ndimage import center_of_mass
from scipy.stats import spearmanr

# mirror of extent_prompt.AREA_MIN / Z_REF / Z_SCALE (extent_prompt imports torch; test_centre_prior_stats checks equality when torch is present)
AREA_MIN = 0.005
Z_REF = math.log(0.02)
Z_SCALE = 2.0

# Part A thresholds (registered)
A1_SUPPORT, A1_NOT = -0.05, -0.02          # paired mean change of centroid offset c (zoomed minus native), per image
A2_SUPPORT, A2_NOT = -0.10, -0.03          # rho(z, c) of the training mixture minus rho(z, c) of the native frames
# Part B thresholds (registered)
Q_COUPLED, Q_NOT = 0.50, 0.80              # Q = G(z=-0.69) / G(z=1.6)
DECISION = ("ClinicDB", "ColonDB", "Endo")


# ------------------------------------------------------------------ geometry
def area_to_z(area):
    """numpy twin of extent_prompt.area_to_z."""
    a = np.clip(np.asarray(area, float), AREA_MIN, 1.0)
    return (np.log(a) - Z_REF) / Z_SCALE


def mask_offset(mask):
    """Centroid distance from the frame centre, each axis divided by its own half-side (the frame is resized to a square at training and test),
    so a square frame gives exactly zpos_stats.centroid_offset (0 centre, sqrt(2) corner)."""
    m = np.asarray(mask, bool)
    h, w = m.shape
    cy, cx = center_of_mass(m)
    return float(np.hypot((cy - (h - 1) / 2) / ((h - 1) / 2), (cx - (w - 1) / 2) / ((w - 1) / 2)))


def mask_z(mask):
    m = np.asarray(mask, bool)
    return float(area_to_z(m.mean()))


def crop(mask, box):
    """box = (x0, y0, x1, y1) as PIL / Dataset._sample_crop_box."""
    x0, y0, x1, y1 = box
    return np.asarray(mask, bool)[y0:y1, x0:x1]


def simulate_visits(masks, sample_box, epochs=15, p=0.5, rand=None):
    """Training visits of the anomalous source images, in the order of the EXP-040 audit (image outer, epoch inner): per image and epoch, with
    probability p (`rand.random() < p`) a crop box is drawn by `sample_box(mask)` (None = failed attempt -> the native frame is used, as
    Dataset.__getitem__ does). `masks` may be any iterable (loaded lazily). `rand` = a random.Random-like object (default: the `random` module,
    which Dataset._sample_crop_box itself draws from). Returns a dict of arrays: img (index), zoomed (bool), z, c, z_native, c_native,
    and the counts n_attempts, n_success (comparable with the EXP-040 audit: 9,407 attempts, 8,768 successful crops)."""
    import random as _random
    rand = rand or _random
    img, zoomed, zs, cs, zn, cn = [], [], [], [], [], []
    n_att = n_ok = 0
    for k, m in enumerate(masks):
        m = np.asarray(m, bool)
        z0, c0 = mask_z(m), mask_offset(m)
        zn.append(z0); cn.append(c0)
        for _ in range(epochs):
            box = None
            if rand.random() < p:
                n_att += 1
                box = sample_box(m)
            if box is None:
                z, c, zm = z0, c0, False
            else:
                n_ok += 1
                cm = crop(m, box)
                z, c, zm = mask_z(cm), mask_offset(cm), True
            img.append(k); zoomed.append(zm); zs.append(z); cs.append(c)
    return dict(img=np.array(img, int), zoomed=np.array(zoomed, bool), z=np.array(zs), c=np.array(cs),
                z_native=np.array(zn), c_native=np.array(cn), n_attempts=n_att, n_success=n_ok)


def load_crop_box(dataset_py="dataset.py", place="uniform", beta=1.0):
    """Dataset._sample_crop_box, compiled from the source of dataset.py itself (so the exact training code runs without importing torch).
    Returns (f, sha256 of the function source); f(mask, target_range) -> box or None, drawing from the global `random` module as in training.
    place / beta = the EXP-051 placement attributes of the Dataset instance ('uniform' = original behaviour)."""
    import ast
    import hashlib
    import random as _random
    from types import SimpleNamespace
    src = open(dataset_py).read()
    tree = ast.parse(src)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_sample_crop_box":
            fn = node
    if fn is None:
        raise RuntimeError("_sample_crop_box not found in " + dataset_py)
    code = ast.get_source_segment(src, fn)
    ns = {"np": np, "random": _random}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), dataset_py, "exec"), ns)
    raw = ns["_sample_crop_box"]
    owner = SimpleNamespace(zoom_place=place, zoom_beta=beta)

    def f(mask, target_range):
        m = np.asarray(mask, bool)
        return raw(owner, SimpleNamespace(size=(m.shape[1], m.shape[0])), m, target_range)
    return f, hashlib.sha256(code.encode()).hexdigest()


# ------------------------------------------------------------------ Part A statistics
def _rho(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y)[0])


def part_a(v, n_boot=2000, seed=0):
    """v = simulate_visits output. Image-cluster bootstrap (an image is resampled with all its visits).
    A1: per image d_i = mean c over its zoomed visits - its native c (images with no zoomed visit are excluded); statistic = mean d_i.
    A2: rho(z, c) over all visits (the training mixture) minus rho(z, c) over the native frames (one row per image)."""
    n_img = len(v["c_native"])
    by_img = [np.flatnonzero(v["img"] == k) for k in range(n_img)]
    d = np.full(n_img, np.nan)
    for k, ix in enumerate(by_img):
        zz = ix[v["zoomed"][ix]]
        if len(zz):
            d[k] = v["c"][zz].mean() - v["c_native"][k]
    has = np.isfinite(d)

    def a2(sel):
        ix = np.concatenate([by_img[k] for k in sel]) if len(sel) else np.array([], int)
        return _rho(v["z"][ix], v["c"][ix]) - _rho(v["z_native"][sel], v["c_native"][sel])

    allk = np.arange(n_img)
    out = dict(n_img=n_img, n_img_zoomed=int(has.sum()), n_visits=len(v["img"]), frac_zoomed=float(v["zoomed"].mean()),
               A1=float(np.nanmean(d)) if has.any() else float("nan"),
               rho_mix=_rho(v["z"], v["c"]), rho_native=_rho(v["z_native"], v["c_native"]),
               rho_zoomed=_rho(v["z"][v["zoomed"]], v["c"][v["zoomed"]]),
               c_native_mean=float(v["c_native"].mean()),
               c_zoomed_mean=float(v["c"][v["zoomed"]].mean()) if v["zoomed"].any() else float("nan"))
    out["A2"] = a2(allk)
    rng = np.random.RandomState(seed)
    b1, b2 = [], []
    for _ in range(n_boot):
        s = rng.randint(0, n_img, n_img)
        ds = d[s]
        b1.append(np.nanmean(ds) if np.isfinite(ds).any() else np.nan)
        b2.append(a2(s))
    b1, b2 = np.array(b1), np.array(b2)
    q = lambda b: (float(np.nanquantile(b, 0.025)), float(np.nanquantile(b, 0.975))) if np.isfinite(b).any() else (float("nan"),) * 2
    out["A1_ci"], out["A2_ci"] = q(b1), q(b2)
    return out


def a1_label(st):
    m, hi = st["A1"], st["A1_ci"][1]
    if not np.isfinite(m):
        return "INVALID"
    if round(m, 10) <= A1_SUPPORT and hi < 0:
        return "CENTRALISES"
    if round(m, 10) >= A1_NOT:
        return "NO_CENTRALISATION"
    return "UNCLEAR"


def a2_label(st):
    m, hi = st["A2"], st["A2_ci"][1]
    if not np.isfinite(m):
        return "INVALID"
    if round(m, 10) <= A2_SUPPORT and hi < 0:
        return "COUPLES"
    if round(m, 10) >= A2_NOT:
        return "NO_COUPLING"
    return "UNCLEAR"


def verdict_a(st):
    """SUPPORTED if A1 CENTRALISES or A2 COUPLES; NOT SUPPORTED if A1 NO_CENTRALISATION and A2 NO_COUPLING; otherwise INCONCLUSIVE."""
    l1, l2 = a1_label(st), a2_label(st)
    if "INVALID" in (l1, l2):
        return "INCOMPLETE", l1, l2
    if l1 == "CENTRALISES" or l2 == "COUPLES":
        return "SUPPORTED", l1, l2
    if l1 == "NO_CENTRALISATION" and l2 == "NO_COUPLING":
        return "NOT SUPPORTED", l1, l2
    return "INCONCLUSIVE", l1, l2


# ------------------------------------------------------------------ Part B statistics
def paired_g(rows_hi, rows_lo, n_boot=2000, seed=0):
    """rows_* = analyze_pe_shift.load_rows output (z = 1.6 and z = -0.69) of ONE set and ONE checkpoint. Peripheral images kept in BOTH.
    G = PE_REC - PE_RAND per image. Returns dict: n, G_hi, G_lo, Q = G_lo / G_hi, dG = G_hi - G_lo with paired-bootstrap CI, sham means."""
    def keyed(rows):
        return {r["id"]: r for r in rows if r["group"] == "peripheral" and r["status"] == "kept"}
    hi, lo = keyed(rows_hi), keyed(rows_lo)
    ids = sorted(set(hi) & set(lo))
    out = dict(n=len(ids), n_hi_only=len(set(hi) - set(lo)), n_lo_only=len(set(lo) - set(hi)))
    geo = ("sy_rec", "sx_rec", "sy_rand", "sx_rand")
    bad = [i for i in ids if any(hi[i].get(k) != lo[i].get(k) for k in geo)]
    if bad:     # same image must get the same recentre and random PE shifts in both runs (they depend only on the mask and the id)
        raise ValueError(f"PE shift geometry differs between the z runs for {len(bad)} images, e.g. {bad[0]}")
    if len(ids) < 3:
        return out
    g = lambda d, i: d[i]["auroc_pe_rec"] - d[i]["auroc_pe_rand"]
    gh = np.array([g(hi, i) for i in ids]); gl = np.array([g(lo, i) for i in ids])
    out.update(G_hi=float(gh.mean()), G_lo=float(gl.mean()),
               sham_hi=float(np.mean([hi[i]["auroc_sham"] for i in ids])), sham_lo=float(np.mean([lo[i]["auroc_sham"] for i in ids])),
               dpe_hi=float(np.mean([hi[i]["auroc_pe_rec"] - hi[i]["auroc_sham"] for i in ids])),
               dpe_lo=float(np.mean([lo[i]["auroc_pe_rec"] - lo[i]["auroc_sham"] for i in ids])))
    out["Q"] = float(out["G_lo"] / out["G_hi"]) if out["G_hi"] > 0 else float("nan")
    dg = gh - gl
    rng = np.random.RandomState(seed)
    b = dg[rng.randint(0, len(ids), (n_boot, len(ids)))].mean(1)
    out["dG"] = float(dg.mean()); out["dG_ci"] = (float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975)))
    return out


def set_label_b(st):
    """COUPLED: G_hi > 0, Q <= 0.50 and dG CI lower bound > 0.  NOT: G_hi > 0 and Q >= 0.80.  INVALID: < 3 paired images or G_hi <= 0.  Otherwise MIXED."""
    if st is None or st.get("n", 0) < 3 or not (st.get("G_hi", 0) > 0) or not np.isfinite(st.get("Q", np.nan)):
        return "INVALID"
    q = round(st["Q"], 10)
    if q <= Q_COUPLED and st["dG_ci"][0] > 0:
        return "COUPLED"
    if q >= Q_NOT:
        return "NOT"
    return "MIXED"


def seed_verdict_b(labels):
    """labels: {set: label} over the decision sets. < 2 valid sets -> INCONCLUSIVE; >= 2 COUPLED -> Z-COUPLED; >= 2 NOT -> NOT Z-COUPLED; else MIXED."""
    ls = [labels.get(s) for s in DECISION]
    if any(x is None for x in ls):
        return "INCOMPLETE"
    if sum(x != "INVALID" for x in ls) < 2:
        return "INCONCLUSIVE"
    if sum(x == "COUPLED" for x in ls) >= 2:
        return "Z-COUPLED"
    if sum(x == "NOT" for x in ls) >= 2:
        return "NOT Z-COUPLED"
    return "MIXED"


def verdict_b(seed_labels):
    """seed_labels: {seed tag: seed verdict} for the three ECP seeds. Any INCOMPLETE -> INCOMPLETE; >= 2 Z-COUPLED -> Z-COUPLED;
    >= 2 NOT Z-COUPLED -> NOT Z-COUPLED; otherwise MIXED."""
    ls = list(seed_labels.values())
    if len(ls) < 3 or any(x == "INCOMPLETE" for x in ls):
        return "INCOMPLETE"
    if sum(x == "Z-COUPLED" for x in ls) >= 2:
        return "Z-COUPLED"
    if sum(x == "NOT Z-COUPLED" for x in ls) >= 2:
        return "NOT Z-COUPLED"
    return "MIXED"
