import numpy as np
from posart_stats import *

rng = np.random.RandomState(0)
G_grid, C, P = 12, 16, 144
yy, xx = np.meshgrid(np.arange(G_grid), np.arange(G_grid), indexing="ij")
w = rng.randn(C); w /= np.linalg.norm(w)            # lesion direction


def make(n, art_scale, seed, uniform=True):
    r = np.random.RandomState(seed)
    # known position artifact, partly along the lesion direction (makes some positions look anomalous)
    G_true = art_scale * (np.sin(xx.ravel() / 2.0)[:, None] * w[None] * 2 + 0.5 * r.randn(P, C) * 0 +
                          np.cos(yy.ravel() / 3.0)[:, None] * np.random.RandomState(1).randn(C)[None])
    G_true -= G_true.mean(0, keepdims=True)
    imgs = []
    for _ in range(n):
        content = r.randn(P, C) * 1.0 + 3.0 * np.ones(C)    # random content + big global offset
        mask = np.zeros(P, bool)
        cy, cx = (r.randint(0, G_grid, 2) if uniform else r.randint(3, 9, 2)); mask[((yy.ravel() - cy) ** 2 + (xx.ravel() - cx) ** 2) < 6] = True
        content[mask] += 2.0 * w                            # lesion signal
        imgs.append((content + G_true, mask))
    return G_true, imgs

# 1) estimator recovers a known artifact, keeps global offset
G_true, imgs = make(600, 1.0, 5)
S = sum(t for t, _ in imgs); G_hat = estimate_artifact(S, len(imgs))
cos = split_half_cosine(G_hat, G_true)
assert cos > 0.95, cos
assert abs(G_hat.mean(0)).max() < 1e-9               # centered over positions
# 2) split-half reliability high when artifact exists, ~0 when it does not
_, a = make(300, 1.0, 11); _, b = make(300, 1.0, 12)
Ga = estimate_artifact(sum(t for t, _ in a), 300); Gb = estimate_artifact(sum(t for t, _ in b), 300)
assert split_half_cosine(Ga, Gb) > 0.9
_, a0 = make(300, 0.0, 11); _, b0 = make(300, 0.0, 12)
Ga0 = estimate_artifact(sum(t for t, _ in a0), 300); Gb0 = estimate_artifact(sum(t for t, _ in b0), 300)
c0 = split_half_cosine(Ga0, Gb0); assert abs(c0) < 0.2, c0
# 2b) documented confound: a centre-biased lesion/content layout creates a position pattern even with NO encoder artifact
_, ca = make(300, 0.0, 11, uniform=False); _, cb = make(300, 0.0, 12, uniform=False)
cc = split_half_cosine(estimate_artifact(sum(t for t, _ in ca), 300), estimate_artifact(sum(t for t, _ in cb), 300))
print("layout-confound cosine (no artifact, centre-biased lesions):", round(cc, 2))
assert cc > 0.4   # => estimate G from position-randomised SOURCE content (random crop + dihedral), not from raw target layout
# 3) subtraction improves per-image AUROC; permuted pattern does not
_, test = make(200, 1.0, 21)
def mean_auc(f):
    return np.nanmean([auroc(m, f(t) @ w) for t, m in test])
base = mean_auc(lambda t: t); art = mean_auc(lambda t: subtract_artifact(t, G_hat))
perm = mean_auc(lambda t: subtract_artifact(t, permute_positions(G_hat, np.random.RandomState(3))))
print(f"base {base:.3f}  art {art:.3f}  perm {perm:.3f}")
assert art > base + 0.03 and perm < art - 0.03
# 4) no artifact => subtraction of estimated G changes nothing material
_, test0 = make(200, 0.0, 22)
b0m = np.nanmean([auroc(m, t @ w) for t, m in test0])
a0m = np.nanmean([auroc(m, subtract_artifact(t, Ga0) @ w) for t, m in test0])
assert abs(a0m - b0m) < 0.01, (a0m, b0m)
# 5) energy ratio sanity
allt = np.stack([t for t, _ in imgs]); var = allt.reshape(-1, C).var(0).sum()
assert 0 < energy_ratio(G_hat, var) < 1
# 6) bootstrap CI covers a known mean shift; verdict logic
d = np.random.RandomState(0).randn(500) * 0.1 + 0.02
m, lo, hi = paired_bootstrap_ci(d); assert lo < 0.02 < hi and lo > 0
ds = ["a", "b", "c"]
assert verdict(0.8, dict(zip(ds, [.01, .008, .0])), dict(zip(ds, [.01, .01, .0])), dict(zip(ds, [.002, .001, -.01]))) == "CONFIRM"
assert verdict(0.1, dict(zip(ds, [.01] * 3)), dict(zip(ds, [.01] * 3)), dict(zip(ds, [.001] * 3))).startswith("FALSIFY")
assert verdict(0.8, dict(zip(ds, [.001, .002, 0.])), dict(zip(ds, [.0] * 3)), dict(zip(ds, [-.001] * 3))).startswith("FALSIFY")
assert verdict(0.8, dict(zip(ds, [.01, .01, .01])), dict(zip(ds, [.001, .0, .001])), dict(zip(ds, [.002] * 3))) == "INCONCLUSIVE"
assert verdict(0.8, dict(zip(ds, [.01, .01, -.02])), dict(zip(ds, [.01] * 3)), dict(zip(ds, [.002] * 3))) == "INCONCLUSIVE"
print("ALL SYNTHETIC TESTS PASSED")

# 7) new helpers
assert abs(cosine([1, 0], [1, 0]) - 1) < 1e-9 and abs(cosine([1, 0], [0, 1])) < 1e-9
A, B = split_halves(11, 3)
assert len(A) == 6 and len(B) == 5 and set(A) | set(B) == set(range(11)) and not set(A) & set(B)
r = np.random.RandomState(0)
fr = []
for _ in range(500):
    x0, y0, cw, ch = sample_crop(r, 400, 300)
    assert 0 <= x0 and x0 + cw <= 400 and 0 <= y0 and y0 + ch <= 300
    fr.append(cw * ch / (400 * 300))
assert 0.28 < min(fr) and max(fr) <= 1.0 and 0.5 < np.mean(fr) < 0.8, (min(fr), np.mean(fr))
# 8) EXP-018 stage-1 confound re-check end to end: position-randomised content (random dihedral on the PATTERN-FREE content)
#    gives split-half ~0 when no artifact, high when artifact exists (artifact is added AFTER the content randomisation)
def stage1(art):
    Gt = make(1, art, 5)[0]
    def half(seed):
        r = np.random.RandomState(seed); S = np.zeros((P, C))
        for _ in range(300):
            ct = (r.randn(P, C) + 3.0).reshape(G_grid, G_grid, C)
            ct = np.rot90(ct, r.randint(4))
            ct = ct[:, ::-1] if r.rand() < .5 else ct
            S += ct.reshape(P, C) + Gt
        return estimate_artifact(S, 300)
    return split_half_cosine(half(1), half(2))
assert stage1(1.0) > 0.8 and abs(stage1(0.0)) < 0.2
print("EXP-018 EXTRA TESTS PASSED")
