import numpy as np
from posart_stats import auroc
from zens_stats import *

rng = np.random.RandomState(0)

# 1) best-of-K and win histogram on a hand table (ties -> lowest index)
T = np.array([[0.6, 0.9, 0.7, 0.9, 0.5], [0.8, 0.7, 0.6, 0.5, 0.4], [0.5, 0.5, 0.5, 0.5, 0.9], [0.1, 0.2, 0.3, 0.4, 0.5]])
b, ix = best_of_k(T)
assert b.tolist() == [0.9, 0.8, 0.9, 0.5] and ix.tolist() == [1, 0, 4, 4]
assert win_hist(ix, 5).tolist() == [1, 1, 0, 0, 2]
wh = win_hist(ix, 5, groups=[0, 0, 1, 3])
assert wh.shape == (4, 5) and wh.sum() == 4 and wh[0].tolist() == [1, 1, 0, 0, 0] and wh[1, 4] == 1 and wh[3, 4] == 1 and wh[2].sum() == 0
# ORACLE_BEST >= FIXED always (random tables, FIXED = a column)
for _ in range(20):
    tab = rng.rand(30, 5)
    assert (best_of_k(tab)[0] >= tab[:, 3] - 1e-15).all()

# 2) uniform ensemble of identical maps equals the map; a single map is itself
m = rng.rand(16, 16).astype(np.float32)
assert np.array_equal(mean_maps([m]), m)
assert np.allclose(mean_maps([m, m, m, m, m]), m, atol=1e-7)

# 3) alignment: a pure per-image offset error is fully closed by BGALIGN and MEDALIGN; a within-image error is not
H = 40
def sq(y0, y1, x0, x1):
    g = np.zeros((H, H), bool); g[y0:y1, x0:x1] = True; return g
imgs, gts = [], []
for k in range(12):
    g = sq(10, 22, 10, 22) if k % 2 == 0 else sq(14, 20, 14, 20)
    clean = np.where(g, 0.9, 0.1) + 0.01 * rng.rand(H, H)          # perfectly separable within image
    off = rng.uniform(-1.0, 1.0) * 3                                # big image-specific level shift
    gts.append(g); imgs.append((clean + off).astype(np.float32))
G, M = np.stack(gts), np.stack(imgs)
pooled = auroc(G, M)
assert pooled < 0.85                                                # offsets destroy the pooled AUROC
med = np.stack([m_ - median_offset(m_) for m_ in M])
bgo = [bg_offset(g_, m_, far=3) for g_, m_ in zip(gts, M)]
bga = np.stack([m_ - o for (o, _), m_ in zip(bgo, M)])
assert not any(f for _, f in bgo)
assert auroc(G, bga) > 0.999                                        # BGALIGN fully closes an offset-only error (bg is flat up to 0.01 noise)
assert auroc(G, med) > 0.999                                        # lesions occupy < 50% of pixels, so the median is background
assert abs(closure(auroc(G, bga), pooled) - 1.0) < 1e-2
# within-image error: lesion core cold in half the images (not an offset) -> not closed
M2 = M.copy()
for k in range(0, 12, 2):
    M2[k][G[k]] = M2[k][~G[k]].mean() - 0.5 + 0.01 * rng.rand(int(G[k].sum()))
base2 = auroc(G, M2)
bga2 = np.stack([m_ - bg_offset(g_, m_, far=3)[0] for g_, m_ in zip(gts, M2)])
cl2 = closure(auroc(G, bga2), base2)
assert cl2 < 0.8 and auroc(G, bga2) < 0.9
# fallback when no far pixel exists
o, fb = bg_offset(sq(0, 38, 0, 38), np.ones((H, H), np.float32), far=28)
assert fb and o == 1.0
# 4) closure arithmetic
assert abs(closure(0.95, 0.90) - 0.5) < 1e-12 and closure(0.9, 0.9) == 0 and np.isnan(closure(1.0, 1.0)) and np.isnan(closure(np.nan, 0.5))
assert closure(0.85, 0.90) < 0
# 5) paired bootstrap on known shifts
d = 0.02 + 0.01 * rng.randn(400)
mu, lo, hi = paired_bootstrap(d)
assert lo > 0.018 and hi < 0.022 and lo < mu < hi
mu0, lo0, hi0 = paired_bootstrap(0.05 * rng.randn(400))
assert lo0 < 0 < hi0
assert paired_bootstrap(d) == paired_bootstrap(d)                   # deterministic (seed 0)
mu1, lo1, hi1 = paired_bootstrap(np.full(10, 0.3)); assert (mu1, lo1, hi1) == (0.3, 0.3, 0.3) or abs(mu1 - 0.3) < 1e-12
# 6) verdicts incl. boundaries
mk = lambda *v: dict(zip("abcdef", v))
assert verdict_rz(mk(.0, .0, .0, .0, .05, .05)) == "CLOSE"
assert verdict_rz(mk(.0099, .0099, .0099, .0099, .05, .05)) == "CLOSE"
assert verdict_rz(mk(.010, .010, .010, .010, .0, .0)) == "INCONCLUSIVE"        # 0.010 is NOT < 0.010
assert verdict_rz(mk(.02, .02, .02, .02, .0, .0)) == "PURSUE"                  # 0.020 counts
assert verdict_rz(mk(.0199, .02, .02, .02, .03, .0)) == "PURSUE"
assert verdict_rz(mk(.0199, .0199, .02, .02, .0199, .0)) == "INCONCLUSIVE"
assert verdict_rz(mk(.0, .0, .0, np.nan, .05, .05)) == "INCONCLUSIVE"          # NaN never counts
assert verdict_rz(mk(.0, .0, .0, .015, .05, .05)) == "INCONCLUSIVE"
assert verdict_cal(mk(.5, .5, .5, .5, .0, .0)) == "BETWEEN-IMAGE"
assert verdict_cal(mk(.49, .5, .5, .4, .9, .0)) == "MIXED"
assert verdict_cal(mk(.0, .0, .0, .0, .9, .9)) == "WITHIN-IMAGE"
assert verdict_cal(mk(.2499, .2, .1, .0, .9, .9)) == "WITHIN-IMAGE"
assert verdict_cal(mk(.25, .2, .1, .0, .9, .9)) == "MIXED"                     # 0.25 is NOT < 0.25
assert verdict_cal(mk(.1, .1, .1, np.nan, .9, .9)) == "MIXED"
assert gain_loss(0.005, 0.001, 0.009) == "GAIN" and gain_loss(0.005, -0.001, 0.009) == "NONE" and gain_loss(0.0049, 0.001, 0.009) == "NONE"
assert gain_loss(-0.005, -0.009, -0.001) == "LOSS" and gain_loss(-0.005, -0.009, 0.001) == "NONE" and gain_loss(0.0, -0.1, 0.1) == "NONE"
print("test_zens_stats OK")
