import numpy as np
from locality_stats import *

rng = np.random.default_rng(0)
# 1. bias: symmetric, zero CLS row/col, zero diagonal, exact value for a known pair, inf -> zeros
G = 5; B = gauss_bias(G, 2.0)
assert B.shape == (26, 26) and np.allclose(B, B.T) and np.all(B[0] == 0) and np.all(B[:, 0] == 0)
assert np.allclose(np.diag(B), 0)
# token 1 = (0,0), token 1+G+1 = (1,1): d2=2 -> -2/(2*4) = -0.25
assert abs(B[1, 1 + G + 1] + 0.25) < 1e-6
assert np.all(gauss_bias(G, np.inf) == 0)

# 2. locality: with identical v-v logits, attention mass within radius 2 of the query grows monotonically as b shrinks
G = 9; n = 1 + G * G
logits = np.zeros((n, n), np.float32)  # uniform content similarity
q = 1 + 4 * G + 4                      # centre token
yy, xx = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
near = ((yy - 4) ** 2 + (xx - 4) ** 2 <= 4).ravel()
masses = []
for b in (np.inf, 8, 4, 2, 1):
    a = softmax(logits + gauss_bias(G, b))[q, 1:]
    assert abs(a.sum() + softmax(logits + gauss_bias(G, b))[q, 0] - 1) < 1e-5
    masses.append(a[near].sum())
assert all(masses[i] < masses[i + 1] for i in range(len(masses) - 1)), masses
assert abs(masses[0] - near.sum() / n) < 1e-5  # baseline = uniform

# 3. token smoothing: constant field unchanged, delta spreads and keeps its sum, s=0 identity
G = 11
const = np.ones((G * G, 3)); assert np.allclose(smooth_tokens(const, G, 2.0), 1.0)
d = np.zeros((G, G, 1)); d[5, 5, 0] = 1.0
sm = smooth_tokens(d.reshape(-1, 1), G, 1.0).reshape(G, G)
assert abs(sm.sum() - 1) < 1e-6 and sm[5, 5] < 1 and sm[5, 6] > 0
assert np.array_equal(smooth_tokens(d.reshape(-1, 1), G, 0), d.reshape(-1, 1))

# 4. auroc known answers
assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
assert abs(auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) - 0.5) < 1e-9
assert np.isnan(auroc([0, 0, 0], [1, 2, 3]))

# 5. quartile ids
ids = quartile_ids(np.arange(100.0)); assert [int((ids == k).sum()) for k in range(4)] == [25, 25, 25, 25]

# 6. verdict logic on synthetic tables with known answer
dss = ["A", "B", "C"]; arms = ["l1", "l2", "l3"]
mk = lambda v: {a: dict(zip(dss, v)) for a in arms}
z = {k: 0.0 for k in dss}
# strong, locality-specific gain, feature smoothing gives nothing -> CONFIRM
assert verdict(mk([0.02, 0.02, 0.02]), mk([0, 0, 0]), z, arms)[0] == "CONFIRM"
# gain exists but feature smoothing reproduces it -> AGGREGATION_ONLY
assert verdict(mk([0.02, 0.02, 0.02]), mk([0, 0, 0]), {k: 0.02 for k in dss}, arms)[0] == "AGGREGATION_ONLY"
# nothing above 0.005 -> FALSIFY
assert verdict(mk([0.001, -0.002, 0.003]), mk([0, 0, 0]), z, arms)[0] == "FALSIFY"
# gain but hurts small lesions badly at sigma4 -> arm fails -> INCONCLUSIVE
assert verdict(mk([0.02, 0.02, 0.02]), mk([-0.05, 0, 0]), z, arms)[0] == "INCONCLUSIVE"
# one dataset drops below tol -> arm fails
assert verdict(mk([0.02, 0.02, -0.01]), mk([0, 0, 0]), z, arms)[0] == "INCONCLUSIVE"

# 7. end-to-end sanity of the evaluation pipeline: noisy-but-informative scores on a blob gain AUROC from smoothing
G = 37; yy, xx = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
gt = ((yy - 18) ** 2 + (xx - 18) ** 2 <= 100)
score = gt * 0.6 + rng.normal(0, 1.0, (G, G))
raw = auroc(gt, score)
sm = smooth_tokens(score.reshape(-1, 1), G, 2.0).reshape(G, G)
assert auroc(gt, sm) > raw + 0.05, (raw, auroc(gt, sm))
print("ALL SYNTHETIC TESTS PASSED")
