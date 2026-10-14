"""EXP-048 statistics (numpy/scipy only). Patch-unit geometry, arms, common valid region, ratio statistics and the registered rule.
Image-translation convention is pos_shift_stats's: shifted(y, x) = I(y - dy, x - dx); a token at patch (i, j) moves to (i + sy, j + sx) when (dy, dx) = 14 * (sy, sx)."""
import hashlib
import numpy as np
from scipy.ndimage import center_of_mass
import pos_shift_stats as P

PATCH, SIDE = 14, 37
CENTRE = (SIDE - 1) // 2                  # 18
R_PE, R_NOT, D_IMG_MIN = 0.50, 0.20, 0.020
DECISION = ("ClinicDB", "ColonDB", "Endo")


def patch_centroid(mask, patch=PATCH):
    """Lesion centroid in patch units (patch k has its centre at pixel patch*k + (patch-1)/2), rounded to integers -> (ci, cj)."""
    cy, cx = center_of_mass(np.asarray(mask, bool))
    o = (patch - 1) / 2
    return int(np.rint((cy - o) / patch)), int(np.rint((cx - o) / patch))


def rec_shift(mask):
    """(sy, sx) = (CENTRE - ci, CENTRE - cj): token (ci, cj) receives the PE of the grid centre (PE arm) / the image moves it there (IMG arm)."""
    ci, cj = patch_centroid(mask)
    return CENTRE - ci, CENTRE - cj


def rand_shift(sample_id, sy, sx):
    """Same |sy|, |sx|, random signs drawn from RandomState(SHA-256 of 'EXP-048:'+id). The sign pattern of the recentre shift is excluded, so RAND != REC
    whenever the shift is not (0, 0) (with k nonzero components there are 2^k - 1 admissible patterns, k = 1 gives the opposite direction). (0, 0) -> (0, 0)."""
    comps = [(0, abs(sy)), (1, abs(sx))]
    nz = [k for k, a in comps if a > 0]
    if not nz:
        return 0, 0
    rec_sign = {0: int(np.sign(sy)), 1: int(np.sign(sx))}
    seed = int(hashlib.sha256(("EXP-048:" + sample_id).encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    alts = [p for p in range(2 ** len(nz)) if any(((1 if (p >> b) & 1 else -1) != rec_sign[k]) for b, k in enumerate(nz))]
    p = alts[rng.randint(len(alts))]
    sg = {k: (1 if (p >> b) & 1 else -1) for b, k in enumerate(nz)}
    return sg.get(0, 1) * abs(sy), sg.get(1, 1) * abs(sx)


def img_shift_px(sy, sx, patch=PATCH):
    """(dx, dy) pixel displacement for pos_shift_stats conventions."""
    return int(sx) * patch, int(sy) * patch


def pe_valid_pixels(valid_grid, patch=PATCH):
    """(side, side) bool grid of cells with unclamped source -> (side*patch)^2 bool pixel mask."""
    return np.kron(np.asarray(valid_grid, bool), np.ones((patch, patch), bool))


def common_region(shape, pe_valid_grids, img_shifts_px):
    """Pixels valid under every PE shift (unclamped source cell) AND inside the canvas after every image translation."""
    reg = P.common_valid(shape, list(img_shifts_px))
    for g in pe_valid_grids:
        reg &= pe_valid_pixels(g)
    return reg


def boot_ratio_g(sham, pe_rec, pe_rand, img_rec, n_boot=2000, seed=0):
    """Paired image bootstrap. Returns dict: D_IMG, D_PE, D_PErand (means), R = D_PE / D_IMG with percentile CI, G = mean(PE_REC - PE_RAND) with CI."""
    a = [np.asarray(v, float) for v in (sham, pe_rec, pe_rand, img_rec)]
    s, pr, pn, ir = a
    n = len(s)
    d_img, d_pe, d_pn = ir - s, pr - s, pn - s
    g = pr - pn
    out = dict(n=n, D_IMG=float(d_img.mean()), D_PE=float(d_pe.mean()), D_PErand=float(d_pn.mean()))
    out["R"] = float(out["D_PE"] / out["D_IMG"]) if out["D_IMG"] != 0 else float("nan")
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, (n_boot, n))
    bi, bp = d_img[idx].mean(1), d_pe[idx].mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(bi != 0, bp / bi, np.nan)
    r = r[np.isfinite(r)]
    out["R_ci"] = (float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))) if len(r) else (float("nan"), float("nan"))
    bg = g[idx].mean(1)
    out["G"] = float(g.mean()); out["G_ci"] = (float(np.quantile(bg, 0.025)), float(np.quantile(bg, 0.975)))
    return out


def is_valid(d_img):
    return bool(np.isfinite(d_img) and round(d_img, 10) >= D_IMG_MIN)


def set_pass_pe(st):
    """R >= 0.50 and CI lower bound of G > 0 (values rounded to 1e-10)."""
    return bool(is_valid(st["D_IMG"]) and np.isfinite(st["R"]) and round(st["R"], 10) >= R_PE and round(st["G_ci"][0], 10) > 0)


def set_pass_not(st):
    return bool(is_valid(st["D_IMG"]) and np.isfinite(st["R"]) and round(st["R"], 10) <= R_NOT)


def verdict(sets, decision=DECISION):
    """sets[name] = dict from boot_ratio_g (or None). -> (label, n_valid, n_pe, n_not).  Fewer than 2 VALID decision sets -> INCONCLUSIVE;
    PE-MEDIATED if >= 2 sets pass_pe; NOT PE-MEDIATED if >= 2 sets pass_not; otherwise MIXED (the two cannot both hold: 2 + 2 > 3 sets)."""
    st = [sets.get(s) for s in decision]
    nv = sum(1 for v in st if v is not None and is_valid(v["D_IMG"]))
    if nv < 2:
        return "INCONCLUSIVE", nv, 0, 0
    npe = sum(1 for v in st if v is not None and set_pass_pe(v))
    nno = sum(1 for v in st if v is not None and set_pass_not(v))
    if npe >= 2:
        return "PE-MEDIATED", nv, npe, nno
    if nno >= 2:
        return "NOT PE-MEDIATED", nv, npe, nno
    return "MIXED", nv, npe, nno
