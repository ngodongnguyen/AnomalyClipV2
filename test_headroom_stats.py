import numpy as np
from posart_stats import auroc
from headroom_stats import *

FA, ED = 3, 2   # toy far / edge widths


def square(h, w, y0, y1, x0, x1):
    g = np.zeros((h, w), bool); g[y0:y1, x0:x1] = True
    return g


# 1) zone partition on a 16x16 square lesion in a 40x40 canvas (hand-computed counts)
gt = square(40, 40, 12, 28, 12, 28)
z = zone_map(gt, far=FA, edge=7)
assert z.shape == gt.shape and set(np.unique(z)) <= {FAR, RIM, EDGE, CORE}
counts = [int((z == k).sum()) for k in range(4)]
assert sum(counts) == 1600                                  # disjoint and covering
assert counts[CORE] == 4 and counts[EDGE] == 252            # edt==8 only in the central 2x2; edt 1..7 elsewhere
assert counts[RIM] == 192 + 16 and counts[FAR] == 1600 - 256 - 208   # 4 strips 16x3 + 4 corners with dx^2+dy^2<=9, dx,dy>=1 (4 pts)
assert (z[gt] >= EDGE).all() and (z[~gt] <= RIM).all()
d = signed_distance(gt)
assert d[~gt].min() > 0 and d[gt].max() <= 0
try:
    zone_map(np.zeros((5, 5), bool)); raise SystemExit("empty mask must raise")
except ValueError:
    pass


def stack(maps):
    return np.stack(maps).astype(np.float32)


def run(gts, maps, far=FA, edge=2):
    zones = np.stack([zone_map(g, far, edge) for g in gts])
    gt = np.stack(gts).astype(np.uint8)
    base = stack(maps)
    return pooled_metrics(gt, base, zones, auroc), zones, gt, base


# 2) far false-positive blob only (thin lesion = all EDGE, no core): FAR_FIX -> 1.0; RIM_FIX and CORE_FIX unchanged
g = square(40, 40, 18, 22, 18, 22)
m = np.full((40, 40), 0.1); m[g] = 0.9; m[2:6, 2:6] = 0.9
r, zones, gt, base = run([g, g], [m, m])
assert r["base"] < 0.999 and abs(r["FAR_FIX"] - 1.0) < 1e-12
assert (zones == CORE).sum() == 0
assert r["RIM_FIX"] == r["base"] and r["CORE_FIX"] == r["base"]
assert abs(r["PERFECT"] - 1.0) < 1e-12

# 3) cold core only: CORE_FIX fixes it; FAR_FIX and RIM_FIX do nothing
g = square(40, 40, 10, 30, 10, 30)
m = np.full((40, 40), 0.1); m[g] = 0.9
zz = zone_map(g, FA, 2); m[zz == CORE] = 0.1
r, *_ = run([g], [m])
assert r["base"] < 0.999 and abs(r["CORE_FIX"] - 1.0) < 1e-12
assert r["FAR_FIX"] == r["base"] and r["RIM_FIX"] == r["base"] and abs(r["PERFECT"] - 1.0) < 1e-12

# 4) bleed only: bright rim just outside the lesion
m2 = np.full((40, 40), 0.1)
m2[zz == RIM] = 0.5
# dim lesion edge (0.4) below the bleed (0.5); bright core sets gmax
m2[g] = 0.4; m2[zz == CORE] = 0.9
r, *_ = run([g], [m2])
assert r["base"] < 0.999 and r["RIM_FIX"] > r["base"]
assert r["CORE_FIX"] == r["base"] and r["FAR_FIX"] == r["base"]
# pure bleed: lesion constant, rim above part of the lesion
m3 = np.full((40, 40), 0.1); m3[g] = 0.9; m3[zz == EDGE] = 0.3; m3[zz == RIM] = 0.5
r, *_ = run([g], [m3])
assert abs(r["RIM_FIX"] - r["base"]) > 1e-9 and r["CORE_FIX"] == r["base"] and r["FAR_FIX"] == r["base"]
m4 = m3.copy(); m4[zz == EDGE] = 0.9    # edge bright, so only rim vs nothing: base 1.0
r, *_ = run([g], [m4]); assert abs(r["base"] - 1.0) < 1e-12 and abs(r["RIM_FIX"] - 1.0) < 1e-12

# 5) PERFECT is 1.0 for random maps and random masks; edits never lower AUROC in the helpful direction
rng = np.random.RandomState(0)
gts, maps = [], []
for _ in range(4):
    gg = square(48, 48, *sorted(rng.randint(4, 44, 2)), *sorted(rng.randint(4, 44, 2)))
    if gg.sum() < 20: gg = square(48, 48, 10, 30, 10, 30)
    gts.append(gg); maps.append(rng.rand(48, 48))
r, *_ = run(gts, maps)
assert abs(r["PERFECT"] - 1.0) < 1e-12
for e in ("FAR_FIX", "RIM_FIX", "CORE_FIX", "EDGE_FIX"):
    assert r[e] >= r["base"] - 1e-12, (e, r)    # pushing negatives down / positives up cannot reduce pooled AUROC

# 6) gap-closure arithmetic
assert abs(gap_closure(0.95, 0.90) - 0.5) < 1e-12
assert abs(gap_closure(1.0, 0.8) - 1.0) < 1e-12 and abs(gap_closure(0.8, 0.8)) < 1e-12 and gap_closure(0.7, 0.8) < 0
assert np.isnan(gap_closure(1.0, 1.0)) and np.isnan(gap_closure(np.nan, 0.5))

# 7) quartile assignment: 0..3, balanced, monotone in area
a = np.arange(1, 101) / 100.0
q = quartile_index(a)
assert set(q) == {0, 1, 2, 3} and all(abs((q == k).sum() - 25) <= 1 for k in range(4)) and (np.diff(q) >= 0).all()
assert quartile_index(np.random.RandomState(1).rand(200)).max() == 3

# 8) dominance and target rule
assert dominant_mode({"FAR_FIX": 0.5, "RIM_FIX": 0.2, "CORE_FIX": 0.1}) == "FAR_FIX"
assert dominant_mode({"FAR_FIX": 0.29, "RIM_FIX": 0.2, "CORE_FIX": 0.1}) is None     # largest but < 30%
assert dominant_mode({"FAR_FIX": 0.1, "RIM_FIX": 0.31, "CORE_FIX": 0.3}) == "RIM_FIX"
assert dominant_mode({"FAR_FIX": float("nan"), "RIM_FIX": 0.4, "CORE_FIX": 0.1}) == "RIM_FIX"
da = {f"d{i}": m for i, m in enumerate(["FAR_FIX", "FAR_FIX", "FAR_FIX", "RIM_FIX", None, "CORE_FIX"])}
dq = {f"d{i}": m for i, m in enumerate(["CORE_FIX", "CORE_FIX", "RIM_FIX", "RIM_FIX", "CORE_FIX", None])}
assert decide_targets(da, dq) == (["FAR_FIX"], ["CORE_FIX"])
assert decide_targets({k: None for k in da}, {k: None for k in da}) == ([], [])
# EDGE_FIX / PERFECT never count as a rule mode
assert dominant_mode({"EDGE_FIX": 0.9, "PERFECT": 1.0}) is None
print("test_headroom_stats: all passed")
