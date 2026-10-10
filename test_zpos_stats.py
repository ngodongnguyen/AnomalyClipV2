import numpy as np
import zpos_stats as z

# centroid offset: centred, shifted, corner
S = 101
m = np.zeros((S, S), bool); m[45:56, 45:56] = True
assert abs(z.centroid_offset(m)) < 1e-9
m2 = np.zeros((S, S), bool); m2[0:3, 0:3] = True
assert abs(z.centroid_offset(m2) - np.hypot(49, 49) / 50) < 0.03
m3 = np.zeros((S, S), bool); m3[45:56, 95:101] = True
assert 0.9 < z.centroid_offset(m3) < 1.0

# ring contrast: dark lesion on a bright background, bright lesion, and empty ring
v = np.full((S, S), 0.8); mm = np.zeros((S, S), bool); mm[40:60, 40:60] = True
v[mm] = 0.2
assert abs(z.ring_contrast(v, mm, width=10) - (0.2 - 0.8)) < 1e-9
v2 = np.full((S, S), 0.2); v2[mm] = 0.9
assert abs(z.ring_contrast(v2, mm, width=10) - 0.7) < 1e-9
full = np.ones((20, 20), bool)
assert np.isnan(z.ring_contrast(np.ones((20, 20)), full, width=5))

# partial Spearman removes a pure size confound
rng = np.random.RandomState(0)
n = 800
logA = rng.randn(n)
c = 0.7 * logA + 0.7 * rng.randn(n)            # offset depends on size
auc_conf = 0.9 * logA + 0.3 * rng.randn(n)       # AUROC depends on size only
assert abs(z.partial_spearman(c, auc_conf, logA)) < 0.1
assert z.partial_spearman(c, auc_conf, rng.randn(n)) > 0.3         # without the right control it looks like an effect
auc_true = auc_conf - 1.0 * c                    # a real periphery effect on top
rho, lo, hi = z.boot_stat(z.partial_spearman, [c, auc_true, logA], n_boot=300)
assert rho < -0.4 and hi < 0, (rho, lo, hi)

# group difference and bootstrap
auc = rng.randn(400) * 0.02 + 0.9; flag = rng.rand(400) < 0.3
auc[flag] -= 0.05
d, l, h = z.boot_stat(lambda a, f: z.group_diff(a, f > 0.5), [auc, flag.astype(float)], n_boot=300)
assert d < -0.04 and h < 0, (d, l, h)
assert np.isnan(z.group_diff([0.9, 0.8], [True, True]))

# verdicts at the boundaries (4 decision sets: need 3, low 1)
S4 = lambda rs: [(r, r - 0.1, r + 0.05 if r < -0.15 else 0.05) for r in rs]
assert z.verdict_periphery(S4([-0.3, -0.25, -0.21, -0.05])) == "SUPPORTED"
assert z.verdict_periphery(S4([-0.3, -0.25, -0.05, -0.05])) == "INCONCLUSIVE"
assert z.verdict_periphery(S4([-0.12, -0.05, 0.0, 0.1])) == "NOT SUPPORTED"
assert z.verdict_periphery(S4([-0.12, -0.11, 0.0, 0.1])) == "INCONCLUSIVE"
assert z.verdict_periphery([(-0.5, -0.6, 0.01)] * 3 + [(-0.5, -0.6, -0.2)]) == "INCONCLUSIVE"       # CI touches zero in 3 sets
D4 = lambda ds: [(d, d - 0.02, d + 0.01 if d < -0.02 else 0.02) for d in ds]
G = [(30, 100)] * 4
assert z.verdict_dark(D4([-0.05, -0.04, -0.031, 0.0]), G) == "SUPPORTED"
assert z.verdict_dark(D4([-0.05, -0.04, -0.031, 0.0]), [(10, 100)] * 4) != "SUPPORTED"                  # small groups never count as support
assert z.verdict_dark(D4([-0.02, 0.0, 0.01, 0.0]), G) == "NOT SUPPORTED"
assert z.verdict_dark(D4([-0.02, -0.02, 0.01, 0.0]), G) == "INCONCLUSIVE"
print("test_zpos_stats OK")
