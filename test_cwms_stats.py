"""Known-answer tests for cwms_stats.py (EXP-036). Plain script: python test_cwms_stats.py"""
import numpy as np
from scipy.ndimage import gaussian_filter

import pos_shift_stats as P
from cwms_stats import *

rng = np.random.RandomState(0)

# 1) view lists and constants
assert CWMS_VIEWS == [(0, 0), (130, 0), (-130, 0), (0, 130), (0, -130)] and SMALL_VIEWS == [(0, 0), (14, 0), (-14, 0), (0, 14), (0, -14)]
assert abs(SIGMA_EXTRA - np.sqrt(32 ** 2 - 4 ** 2)) < 1e-12 and SIGMA_W == 130.0

# 2) weights: exactly zero on padded pixels, positive elsewhere; the denominator is positive at the real geometry (518 px, 130 px, sigma_w 130)
S = 518
den = np.zeros((S, S))
for dx, dy in CWMS_VIEWS:
    w = centre_weight((S, S), dx, dy)
    v = valid_region((S, S), dx, dy)
    assert (w[~v] == 0).all() and (w[v] > 0).all() and w.max() <= 1.0
    den += w
    u = uniform_weight((S, S), dx, dy)
    assert (u[~v] == 0).all() and (u[v] == 1).all()
assert den.min() >= np.exp(-3.95) * 0.99 and (den > 0).all()                       # corner of the identity view: exp(-(2 * 258.5^2) / (2 * 130^2)) = 0.0192
w0 = centre_weight((S, S), 0, 0)
assert abs(w0[259, 259] - np.exp(-(0.5 ** 2 + 0.5 ** 2) / (2 * 130.0 ** 2))) < 1e-12 and abs(w0[0, 0] - np.exp(-(2 * 258.5 ** 2) / (2 * 130.0 ** 2))) < 1e-12
# padded pixels of a shifted view are exactly the complement of the valid mask used for inverse_align
for dx, dy in CWMS_VIEWS[1:]:
    al = P.inverse_align(np.ones((S, S)), dx, dy)
    assert np.array_equal(np.isnan(al), ~valid_region((S, S), dx, dy))

# 3) a hand-computed weight: S = 5, view (+1, 0), sigma_w = 1: pixel (y=2, x=0) sits at (2, 1), centre (2, 2): d^2 = 1
w = centre_weight((5, 5), 1, 0, sigma_w=1.0)
assert abs(w[2, 0] - np.exp(-0.5)) < 1e-12 and abs(w[2, 1] - 1.0) < 1e-12 and w[2, 4] == 0.0     # x = 4 -> 5 is outside the canvas
w = centre_weight((5, 5), 0, -1, sigma_w=1.0)
assert abs(w[3, 2] - 1.0) < 1e-12 and w[0, 2] == 0.0 and abs(w[4, 2] - np.exp(-0.5)) < 1e-12

# 4) fusion of identical maps returns the map; single-view fusion is the view itself; uniform fusion of two valid maps is their mean
m = rng.rand(S, S)
shifted = [m if (dx, dy) == (0, 0) else P.shift_image_fill(m, dx, dy, 0.0) for dx, dy in CWMS_VIEWS]
assert np.allclose(cwms_map(shifted), m, atol=1e-12) and np.allclose(uniform_map(shifted, CWMS_VIEWS), m, atol=1e-12)
assert np.allclose(cwms_map([m], [(0, 0)]), m, rtol=0, atol=1e-15)
a, b = rng.rand(8, 8), rng.rand(8, 8)
assert np.allclose(fuse([a, b], [np.ones((8, 8)), np.ones((8, 8))]), (a + b) / 2)
assert np.allclose(fuse([a, np.full((8, 8), np.nan)], [np.ones((8, 8)), np.zeros((8, 8))]), a)    # NaN with zero weight contributes nothing
try:
    fuse([a], [np.zeros((8, 8))]); raise AssertionError("zero denominator must raise")
except ValueError:
    pass

# 5) SIGMA32 construction equals gaussian_filter with sqrt(32^2 - 4^2); two-stage variance addition matches a single sigma 32 in the interior
x = rng.rand(200, 200).astype(np.float32)
assert np.array_equal(sigma32_map(x), gaussian_filter(x, sigma=np.sqrt(32 ** 2 - 4 ** 2)))
big = rng.rand(700, 700)
two = gaussian_filter(gaussian_filter(big, 4.0, mode="constant"), SIGMA_EXTRA, mode="constant")
one = gaussian_filter(big, 32.0, mode="constant")
assert np.abs(two - one)[150:-150, 150:-150].max() < 2e-3

# 6) planted centre-biased scorer: score = content * g(distance to the VIEW centre); lesion near the left edge, distractors in the background.
#    CWMS must beat FIXED and UNI5 (same views, uniform weights), since it trusts each pixel where it is near a view centre.
H, SH, TAU = 96, 24, 24.0
yy, xx = np.mgrid[0:H, 0:H]
g = np.exp(-((yy - (H - 1) / 2) ** 2 + (xx - (H - 1) / 2) ** 2) / (2 * TAU ** 2))
views = view_list(SH)


def auc(gt, s):
    return P.region_auroc(gt, s, np.ones_like(gt, bool))


res = {"FIXED": [], "CWMS": [], "UNI5": []}
for seed in range(8):
    r = np.random.RandomState(seed)
    gt = np.zeros((H, H), bool)
    y0 = int(r.randint(38, 50)); gt[y0:y0 + 12, 4:16] = True
    img = np.where(gt, 1.0, r.uniform(0, 0.8, (H, H)))
    maps = [P.shift_image_fill(img, dx, dy, 0.0) * g for dx, dy in views]            # the scorer sees the translated content; padding scores 0 and is invalid
    res["FIXED"].append(auc(gt, maps[0]))
    res["CWMS"].append(auc(gt, cwms_map(maps, views, sigma_w=TAU)))
    res["UNI5"].append(auc(gt, uniform_map(maps, views)))
mf, mc, mu = (float(np.mean(res[k])) for k in ("FIXED", "CWMS", "UNI5"))
assert mc > mf + 0.03 and mc > mu + 0.03, (mf, mc, mu)

# 7) position-invariant (equivariant) scorer: CWMS == FIXED == UNI5 exactly on every pixel (no fusion smoothing arises for equivariant maps)
img = rng.rand(H, H)
maps = [P.shift_image_fill(img, dx, dy, 0.0) for dx, dy in views]
assert np.allclose(cwms_map(maps, views, sigma_w=TAU), img, atol=1e-12) and np.allclose(uniform_map(maps, views), img, atol=1e-12)

# 8) bootstrap and per-set statistics
mm, lo, hi = boot_mean(np.full(50, 0.01))
assert mm == lo == hi == 0.01
mm, lo, hi = boot_mean(np.abs(rng.randn(200)) + 0.1)
assert lo > 0 and lo < mm < hi
assert all(np.isnan(v) for v in boot_mean([np.nan]))
au = {k: np.full(30, 0.8) for k in ARMS}
au["CWMS"] = np.full(30, 0.81); au["UNI5"] = np.full(30, 0.805); au["SMALL5"] = np.full(30, 0.79); au["SIGMA32"] = np.full(30, 0.808)
st = set_stats(au)
assert st["best_control"] == "SIGMA32" and abs(st["b"] - 0.002) < 1e-12 and abs(st["d"][0] - 0.01) < 1e-12 and st["d"][1] == st["d"][2]
try:
    set_stats({**au, "UNI5": np.full(29, 0.8)}); raise AssertionError("unpaired arms must raise")
except ValueError:
    pass
# groups: gain only on the 10 largest-c images
c = np.arange(30.0); diff = np.where(c >= 20, 0.1, 0.0)
gg = group_gain(diff, c, np.arange(30) % 4)
assert abs(gg["peripheral"] - 0.1) < 1e-12 and gg["central"] == 0.0 and gg["n_tercile"] == 10 and len(gg["quartile"]) == 4

# 9) verdict boundaries
def mk(d=0.006, lo=0.001, b=0.004, p=0.0):
    return dict(d=(d, lo, d + 0.01), b=b, p=p)


good = {s: mk() for s in SETS}
assert verdict(good) == ("ADVANCE", 6, 0, 0)
# set-level boundaries (pass is >= for d, b, p and > 0 for lo)
assert set_pass((0.005, 0.001, 0.01), 0.003, -1.0)                                  # exact boundaries pass
assert not set_pass((0.0049, 0.001, 0.01), 0.003, 0.0) and not set_pass((0.005, 0.0, 0.01), 0.003, 0.0)
assert not set_pass((0.005, 0.001, 0.01), 0.0029, 0.0) and not set_pass((0.005, 0.001, 0.01), 0.003, -1.0001)
assert not set_pass((0.005, 0.001, 0.01), np.nan, 0.0) and not set_pass((0.005, 0.001, 0.01), 0.003, np.nan)
assert set_loss((-0.005, -0.01, 0.0)) and not set_loss((-0.0049, -0.01, 0.0)) and not set_loss((0.0, -0.01, 0.01))
assert set_below_fail((0.0029, 0, 0)) and not set_below_fail((0.003, 0, 0))
# verdict: exactly 4 passes and the others neutral -> ADVANCE; 3 passes -> INCONCLUSIVE
four = {s: (mk() if i < 4 else mk(d=0.004, lo=-0.001, b=0.0, p=0.0)) for i, s in enumerate(SETS)}
assert verdict(four) == ("ADVANCE", 4, 0, 0)
three = {s: (mk() if i < 3 else mk(d=0.004, lo=-0.001, b=0.0, p=0.0)) for i, s in enumerate(SETS)}
assert verdict(three)[0] == "INCONCLUSIVE"
# a loss on any set blocks ADVANCE even with 5 passes; one loss alone is INCONCLUSIVE, two losses are FAIL
one_loss = {s: (mk() if i < 5 else mk(d=-0.005, lo=-0.01, b=-0.01, p=0.0)) for i, s in enumerate(SETS)}
assert verdict(one_loss)[0] == "INCONCLUSIVE" and verdict(one_loss)[3] == 1
two_loss = {s: (mk() if i < 4 else mk(d=-0.005, lo=-0.01, b=-0.01, p=0.0)) for i, s in enumerate(SETS)}
assert verdict(two_loss)[0] == "FAIL" and verdict(two_loss)[3] == 2
# FAIL when D < +0.003 on >= 3 sets (exactly +0.003 does not count)
f3 = {s: (mk() if i < 3 else mk(d=0.0029, lo=-0.001, b=0.0, p=0.0)) for i, s in enumerate(SETS)}
assert verdict(f3)[0] == "FAIL" and verdict(f3)[2] == 3
f3b = {s: (mk() if i < 3 else mk(d=0.003, lo=-0.001, b=0.0, p=0.0)) for i, s in enumerate(SETS)}
assert verdict(f3b)[0] == "INCONCLUSIVE"
f2 = {s: (mk() if i < 4 else mk(d=0.0, lo=-0.001, b=0.0, p=0.0)) for i, s in enumerate(SETS)}
assert verdict(f2)[0] == "ADVANCE"                                                  # two sets below +0.003 do not trigger FAIL
# AUPRO floor decides a pass, not a loss: 4 passes with P = -1.0001 on one of them -> only 3 passes
pf = dict(good); pf["Kvasir"] = mk(p=-1.0001); pf["ColonDB"] = mk(p=-1.0001); pf["ISIC"] = mk(p=-1.0001)
assert verdict(pf)[0:2] == ("INCONCLUSIVE", 3)
# missing set / non-finite entry
miss = dict(good); del miss["TN3K"]
assert verdict(miss)[0] == "INCOMPLETE"
nan = dict(good); nan["Endo"] = mk(p=np.nan)
assert verdict(nan)[0] == "INCOMPLETE"
assert verdict({**good, "Endo": None})[0] == "INCOMPLETE"
print("test_cwms_stats OK")
