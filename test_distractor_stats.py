import numpy as np, sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distractor_stats import *
rng = np.random.default_rng(0); C = 32
def unit(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)
T = unit(rng.normal(size=(6, C))); isd = np.array([0, 0, 0, 1, 1, 1], bool)   # 3 lesion, 3 distractor concepts
# known answer 1: FP patches = distractor direction + noise, TP patches = lesion direction + noise -> AUROC ~1
fp = unit(T[3:].mean(0) + 0.3 * rng.normal(size=(200, C))); tp = unit(T[:3].mean(0) + 0.3 * rng.normal(size=(200, C)))
X = np.vstack([fp, tp]); y = np.r_[np.ones(200, bool), np.zeros(200, bool)]
a = auroc(y, concept_margin(X, T, isd)); print("separable case AUROC", a); assert a > 0.95
# known answer 2: both groups drawn from the same distribution -> AUROC ~0.5
X2 = unit(rng.normal(size=(2000, C))); y2 = rng.random(2000) < 0.5
a2 = auroc(y2, concept_margin(X2, T, isd)); print("null case AUROC", a2); assert abs(a2 - 0.5) < 0.05
# known answer 3: null-control relabel on the separable case should drop toward 0.5 on average
cs = [auroc(y, concept_margin(X, T, split_control(6, isd, rng))) for _ in range(200)]
print("control mean AUROC", np.mean(cs)); assert abs(np.mean(cs) - 0.5) < 0.15
# known answer 4: auroc exact on tiny case
assert auroc(np.array([1,1,0,0],bool), np.array([.9,.8,.3,.1])) == 1.0
assert auroc(np.array([1,0],bool), np.array([.1,.9])) == 0.0
# known answer 5: FP patches point only at distractor #4 -> per-concept AUROC of concept 4 ~1, other distractors < 0.6
fp5 = unit(T[4] + 0.3 * rng.normal(size=(200, C))); X5 = np.vstack([fp5, tp])
P5 = concept_probs(X5, T); pc = [auroc(y, P5[:, k]) for k in range(6)]; print("per-concept AUROC", np.round(pc, 2))
assert pc[4] > 0.95 and pc[3] < 0.6 and pc[5] < 0.6
# known answer 6: module C re-ranking
H = 40; sc = np.random.default_rng(1).random((H, H)) * 0.2; sc[5:10, 5:10] = 0.9; sc[25:30, 25:30] = 0.8   # two blobs
dd = np.zeros((H, H)); dd[25:30, 25:30] = 0.9                                                       # 2nd blob is a "distractor"
r = rerank_pool(sc, dd, top_frac=0.05); fl = np.quantile(sc, 0.95)
assert np.allclose(rerank_pool(sc, np.zeros((H, H)), 0.05), sc)              # d=0 -> identity
out = np.ones((H, H), bool); out[5:10, 5:10] = out[25:30, 25:30] = False
assert np.allclose(r[out & (sc < fl)], sc[out & (sc < fl)])                  # outside pool untouched
assert (r[sc >= fl] >= fl - 1e-9).all()                                       # pool never falls below floor
assert r[7, 7] == sc[7, 7] and r[27, 27] < sc[27, 27]                         # clean blob kept, distractor blob pulled down
assert r[7, 7] > r[27, 27]                                                    # ranking flipped relative to the plain 0.9 vs 0.8? (clean stays top)
print("module C ok", r[7, 7], r[27, 27], fl)
print("ALL UNIT TESTS PASSED")
