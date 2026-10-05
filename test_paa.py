import numpy as np, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paa import paa_np
rng = np.random.default_rng(0)
G, C = 9, 4
# 1) s=1 is the identity; CLS untouched for every s
t = rng.normal(size=(1 + G * G, C))
assert np.allclose(paa_np(t, 1), t)
for s in (3, 5): assert np.allclose(paa_np(t, s)[0], t[0])
# 2) a constant field is unchanged, borders included (truncated window, not zero padded)
c = np.concatenate([rng.normal(size=(1, C)), np.tile(rng.normal(size=(1, C)), (G * G, 1))], 0)
for s in (3, 5): assert np.allclose(paa_np(c, s), c)
# 3) known answer: a single 1 in the interior spreads to the s x s window with weight 1/s^2
d = np.zeros((1 + G * G, 1)); d[1 + 4 * G + 4, 0] = 1.0
for s in (3, 5):
    o = paa_np(d, s)[1:, 0].reshape(G, G); r = s // 2
    assert np.allclose(o[4 - r:4 + r + 1, 4 - r:4 + r + 1], 1.0 / s ** 2); assert np.isclose(o.sum(), s * s / s ** 2)
# 4) known answer at a corner: window truncated to (r+1)^2 patches, so the corner patch itself averages over them
d = np.zeros((1 + G * G, 1)); d[1, 0] = 1.0
o3 = paa_np(d, 3)[1:, 0].reshape(G, G)
assert np.isclose(o3[0, 0], 1 / 4) and np.isclose(o3[0, 1], 1 / 6) and np.isclose(o3[1, 1], 1 / 9) and np.isclose(o3[2, 2], 0.0)
# 5) smoothing reduces variance of white noise by about 1/s^2 in the interior
n = rng.normal(size=(1 + 41 * 41, 1)); n[0] = 0
v1, v3, v5 = [paa_np(n, s)[1:, 0].reshape(41, 41)[3:-3, 3:-3].var() for s in (1, 3, 5)]
assert 0.09 < v3 / v1 < 0.14 and 0.03 < v5 / v1 < 0.055, (v3 / v1, v5 / v1)
# 6) even windows are rejected
try: paa_np(t, 2); raise SystemExit("even window accepted")
except AssertionError: pass
print("ALL PAA TESTS PASSED")
