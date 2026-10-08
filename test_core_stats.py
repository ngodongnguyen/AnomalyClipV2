import numpy as np
from core_stats import *

G = 37


def mask_from_patches(rect_list, grid=G, px=14):
    """518x518 binary mask: union of patch-aligned rectangles (y0, y1, x0, x1 in patch units)."""
    m = np.zeros((grid * px, grid * px), bool)
    for y0, y1, x0, x1 in rect_list:
        m[y0 * px:y1 * px, x0 * px:x1 * px] = True
    return m


# 1) square lesion 10x10 patches at the centre: core 8x8=64, rim_in 36, bg 1369-100
z = patch_zones(patch_fraction(mask_from_patches([(10, 20, 10, 20)]), G))
assert z["lesion"].sum() == 100 and z["core"].sum() == 64 and z["rim_in"].sum() == 36 and z["bg"].sum() == 1269
# near = 2-patch Chebyshev ring around the square = 14x14 - 10x10 = 96 ; bg_far = 1269 - 96
assert z["bg_near"].sum() == 96 and z["bg_far"].sum() == 1269 - 96
assert zone_counts(z) == {"core": 64, "rim_in": 36, "bg_far": 1173} and is_eligible(z)
# the zones are disjoint
assert not (z["core"] & z["rim_in"]).any() and not (z["bg_near"] & z["bg_far"]).any() and not (z["lesion"] & z["bg"]).any()

# 2) lesion touching the image border: rows 0..9, cols 0..9 -> core excludes the border patches: rows 1..8, cols 1..8 = 64
z = patch_zones(patch_fraction(mask_from_patches([(0, 10, 0, 10)]), G))
assert z["core"].sum() == 64 and z["rim_in"].sum() == 36
assert z["bg_near"].sum() == 12 * 12 - 100  # out-of-grid side clipped: rows/cols 10,11 only -> 144-100 = 44

# 3) two lesions: 6x6 and 5x5 -> cores 16 and 9
z = patch_zones(patch_fraction(mask_from_patches([(5, 11, 5, 11), (25, 30, 25, 30)]), G))
assert z["core"].sum() == 16 + 9 and z["lesion"].sum() == 36 + 25

# 4) tiny lesion 2x2: no CORE -> not eligible; 3x3 has a single core patch -> still not eligible (< 4)
z = patch_zones(patch_fraction(mask_from_patches([(10, 12, 10, 12)]), G))
assert z["core"].sum() == 0 and not is_eligible(z)
z = patch_zones(patch_fraction(mask_from_patches([(10, 13, 10, 13)]), G))
assert z["core"].sum() == 1 and not is_eligible(z)

# 5) fractional patches: a lesion edge cutting patches in half is neither lesion (< 0.9) nor bg
m = np.zeros((518, 518), bool); m[140:290, 140:290] = True   # 140 = 10*14 ; 290 = 20.7*14
fr = patch_fraction(m, G)
assert abs(fr[20, 15] - (290 - 280) / 14) < 1e-12 and fr[20, 15] < 0.9
zz = patch_zones(fr)
assert not zz["lesion"][20, 15] and not zz["bg"][20, 15]
assert zz["lesion"].sum() == 100          # rows/cols 10..19 are full; row/col 20 holds 10/14 = 0.71 -> not lesion

# 6) prototypes, V sign, F_bg on synthetic tokens (C=16)
rng = np.random.RandomState(0)
C = 16
e = np.eye(C)
z = patch_zones(patch_fraction(mask_from_patches([(10, 20, 10, 20)]), G))


def make_tokens(core_dir, noise=0.1):
    t = np.zeros((G * G, C))
    t[z["rim_in"].reshape(-1)] = e[0]
    t[z["bg"].reshape(-1)] = e[1]
    t[z["core"].reshape(-1)] = core_dir
    t = t + noise * rng.randn(G * G, C)
    return t / np.linalg.norm(t, axis=1, keepdims=True)


t_rim = make_tokens(e[0])
p = prototypes(t_rim, z)
V, cr, cb = visual_ambiguity(p)
assert V > 0.8 and cr > 0.9 and abs(np.linalg.norm(p["core"]) - 1) < 1e-9
assert f_bg(t_rim, z, p) < 0.01
t_bg = make_tokens(e[1])
p = prototypes(t_bg, z)
V, cr, cb = visual_ambiguity(p)
assert V < -0.8 and f_bg(t_bg, z, p) > 0.99
# core half rim half bg direction -> V near 0 and F_bg near 0.5
t_mix = make_tokens((e[0] + e[1]) / np.sqrt(2), noise=0.0)
p = prototypes(t_mix, z)
assert abs(visual_ambiguity(p)[0]) < 1e-9

# 7) coldness and zone means (hand-computed)
s = np.zeros((G, G)); s[z["rim_in"]] = 0.8; s[z["core"]] = 0.3; s[z["bg_far"]] = 0.1
assert abs(coldness(s, z) - 0.5) < 1e-12
zm = zone_means(s, z)
assert abs(zm["core"] - 0.3) < 1e-12 and abs(zm["rim_in"] - 0.8) < 1e-12 and abs(zm["bg_far"] - 0.1) < 1e-12
assert coldness(np.where(z["core"], 0.9, s), z) < 0           # warm core -> negative delta

# 8) spearman: known monotone and independent noise
x = rng.rand(300)
assert abs(spearman(x, np.exp(3 * x)) - 1) < 1e-12 and abs(spearman(x, -x ** 3) + 1) < 1e-12
assert abs(spearman(x, rng.rand(300))) < 0.2
assert np.isnan(spearman(x, np.ones(300)))
# partial spearman: delta and V both driven by the same size variable, otherwise independent -> raw rho large, partial near 0
n = rng.rand(400)
a = n + 0.1 * rng.randn(400); b = n + 0.1 * rng.randn(400)
assert spearman(a, b) > 0.8 and abs(partial_spearman(a, b, n)) < 0.15
# partial keeps a real relation that is not explained by size
b2 = a + 0.1 * rng.randn(400)
assert partial_spearman(a, b2, n) > 0.7

# 9) tercile selection
d = np.arange(10.0)
assert sorted(top_tercile(d)) == [6, 7, 8, 9]                  # ceil(10/3) = 4
assert list(top_tercile(np.array([3.0, 1.0, 2.0]))) == [0]      # ceil(3/3) = 1

# 10) permutation null: independent data -> rho small, p large, quantiles ~ +-1.96/sqrt(n-1) ; dependent data -> p minimal
a = rng.randn(100); b = rng.randn(100)
rho, p, lo, hi = perm_null(a, b, 1000, 0)
assert abs(rho) < 0.3 and p > 0.01
assert abs(lo + 1.96 / np.sqrt(99)) < 0.06 and abs(hi - 1.96 / np.sqrt(99)) < 0.06
rho, p, lo, hi = perm_null(a, a + 0.1 * rng.randn(100), 1000, 0)
assert rho > 0.9 and abs(p - 1 / 1001) < 1e-12
assert perm_null(a, b, 1000, 0) == perm_null(a, b, 1000, 0)    # deterministic

# 11) dataset_summary: cold images are background-like -> VISUAL-type numbers
n = 60
v = rng.randn(n)
delta = -v + 0.2 * rng.randn(n)                                 # colder (larger delta) = lower V
ncore = np.exp(rng.randn(n) + 3)
sm = dataset_summary(delta, v, ncore)
assert sm["eligible"] and sm["rho"] < -0.9 and sm["vmed_cold"] < 0 and sm["p"] < 0.01
assert not dataset_summary(delta[:29], v[:29], ncore[:29])["eligible"]


# 12) verdict on synthetic dataset summaries (boundaries included)
def S(rho, vm, el=True):
    return {"eligible": el, "rho": rho, "vmed_cold": vm}


vis = {k: S(-0.5, -0.1) for k in "ABCDEF"}
assert verdict(vis)[0] == "VISUAL"
rea = {k: S(0.05, 0.2) for k in "ABCDEF"}
assert verdict(rea)[0] == "READOUT"
mixed = {"A": S(-0.5, -0.1), "B": S(-0.5, -0.1), "C": S(0.1, 0.2), "D": S(0.1, 0.2), "E": S(-0.2, 0.1), "F": S(-0.4, 0.1)}
assert verdict(mixed)[0] == "MIXED/INCONCLUSIVE"
few = {"A": S(-0.5, -0.1), "B": S(-0.5, -0.1), "C": S(-0.5, -0.1), "D": S(-0.5, -0.1, el=False), "E": S(-0.5, -0.1, el=False)}
assert verdict(few)[0] == "INCONCLUSIVE" and "only 3" in verdict(few)[1]
# exactly 4 eligible datasets, all passing: allowed
four = {"A": S(-0.5, -0.1), "B": S(-0.5, -0.1), "C": S(-0.5, -0.1), "D": S(-0.5, -0.1), "E": S(0, 0.1, el=False)}
assert verdict(four)[0] == "VISUAL"
# boundary: rho = -0.30 passes visual (<=); -0.29999 and -0.3000001 sides
assert verdict({k: S(-0.30, -0.1) for k in "ABCD"})[0] == "VISUAL"
assert verdict({k: S(-0.29, -0.1) for k in "ABCD"})[0] == "MIXED/INCONCLUSIVE"
# boundary: rho = -0.15 does NOT pass readout (strict >); just above passes
assert verdict({k: S(-0.15, 0.1) for k in "ABCD"})[0] == "MIXED/INCONCLUSIVE"
assert verdict({k: S(-0.149, 0.1) for k in "ABCD"})[0] == "READOUT"
# vmed exactly 0 passes neither; NaN passes neither
assert verdict({k: S(-0.5, 0.0) for k in "ABCD"})[0] == "MIXED/INCONCLUSIVE"
assert verdict({k: S(float("nan"), float("nan")) for k in "ABCD"})[0] == "MIXED/INCONCLUSIVE"
# 3 visual + 1 readout cannot make either verdict
assert verdict({"A": S(-0.5, -0.1), "B": S(-0.5, -0.1), "C": S(-0.5, -0.1), "D": S(0.0, 0.1)})[0] == "MIXED/INCONCLUSIVE"

print("test_core_stats: all passed")
