import numpy as np
import zstar_stats as z

rng = np.random.RandomState(0)
zs = [-0.69, 0.0, 0.8, 1.6, 2.4]
n = 600
ids = [f"img{i}" for i in range(n)]
area = np.exp(rng.uniform(np.log(0.005), np.log(0.5), n))
q = np.digitize(area, np.quantile(area, [0.25, 0.5, 0.75]))
fixed = 3
# (1) real dependence: the best z moves with area
best = np.clip(np.round((np.log(area) - np.log(0.005)) / (np.log(0.5) - np.log(0.005)) * 4).astype(int), 0, 4)
a1 = 0.85 - 0.04 * np.abs(np.arange(5)[None, :] - best[:, None]) + 0.01 * rng.randn(n, 5)
s1 = z.slope_stat(a1, zs)
rho, lo, hi = z.rho_boot(s1, np.log(area), n_boot=200)
assert rho > 0.5 and lo > 0.3 and z.association(rho, lo, hi) == "POS", (rho, lo, hi)
g1 = z.crossfit_gain(a1, q, fixed, ids)
m, l, h = z.boot_mean(g1, 500)
assert m > 0.01 and l > 0.005 and z.useful(m, l), (m, l)
# (2) no dependence: pure noise around a flat curve, fixed z is as good as any
a2 = 0.85 + 0.03 * rng.randn(n, 5)
rho, lo, hi = z.rho_boot(z.slope_stat(a2, zs), np.log(area), n_boot=200)
assert z.association(rho, lo, hi) == "NONE", (rho, lo, hi)
g2 = z.crossfit_gain(a2, q, fixed, ids)
m2, l2, _ = z.boot_mean(g2, 500)
assert m2 < 0.005 and not z.useful(m2, l2), (m2, l2)       # cross-fitting must not manufacture a gain from noise
# (3) shuffled quartile labels remove a real gain
g3 = z.crossfit_gain(a1, q, fixed, ids, groups=rng.permutation(q))
assert z.boot_mean(g3, 500)[0] < m / 2
# (4) fold assignment is deterministic and roughly balanced
f = z.fold_of(ids); assert np.array_equal(f, z.fold_of(ids)) and 0.4 < f.mean() < 0.6
# (5) verdict thresholds
assert z.verdict([True] * 4 + [False] * 2, ["POS"] * 4 + ["NONE"] * 2) == "WORTH A TRAINING STUDY"
assert z.verdict([True] * 4 + [False] * 2, ["POS"] * 2 + ["NEG"] * 2 + ["NONE"] * 2) == "INCONCLUSIVE"     # mixed signs do not count
assert z.verdict([True] * 2 + [False] * 4, ["POS"] * 6) == "CLOSE"
assert z.verdict([True] * 3 + [False] * 3, ["POS"] * 6) == "INCONCLUSIVE"
assert z.useful(0.005, 0.0001) and not z.useful(0.0049, 0.001) and not z.useful(0.01, 0.0)
assert z.association(0.20, 0.01, 0.4) == "POS" and z.association(0.19, 0.01, 0.4) == "NONE" and z.association(-0.3, -0.5, -0.1) == "NEG"
print("test_zstar_stats OK")
