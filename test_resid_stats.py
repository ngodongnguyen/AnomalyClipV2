"""Local known-answer tests for resid_stats.py.  Run: /Users/nguyen.ngo.1/miniconda3/bin/python test_resid_stats.py"""
import numpy as np
from resid_stats import (swap_pairing, recon_max_abs_diff, norm_ratio, quartile_means, arm_passes, verdict,
                         auroc, paired_bootstrap_ci)

DS = ("A", "B", "C")


def V(a, b, c):
    return dict(zip(DS, (a, b, c)))


# 1) swap pairing: derangement + bijection + deterministic
for n in (2, 3, 10, 612):
    p = swap_pairing(n, 7)
    assert (p != np.arange(n)).all() and sorted(p.tolist()) == list(range(n))
assert (swap_pairing(50, 1) == swap_pairing(50, 1)).all() and (swap_pairing(50, 1) != swap_pairing(50, 2)).any()

# 2) decomposition check: exact case 0, perturbed case detected
r = np.random.RandomState(0)
x5 = r.randn(30, 8); at = [r.randn(30, 8) for _ in range(19)]
xf = x5 + sum(at)
assert recon_max_abs_diff(x5, at, xf) < 1e-12
xf2 = xf.copy(); xf2[3, 2] += 0.5
assert abs(recon_max_abs_diff(x5, at, xf2) - 0.5) < 1e-9

# 3) norm ratio known: x5 = 3 * unit, attn = 1 * unit -> 3
u = r.randn(30, 8); u /= np.linalg.norm(u, axis=1, keepdims=True)
assert abs(norm_ratio(3 * u, u) - 3.0) < 1e-9

# 4) quartiles: delta = 1 in smallest quartile only
la = np.arange(100.0); d = np.zeros(100); d[:25] = 1.0
q = quartile_means(la, d); assert q[0] == 1.0 and q[1] == q[2] == q[3] == 0.0

# 5) bootstrap reuse: null contains 0, shifted excludes it
nul = np.random.RandomState(1).randn(500) * 0.02
m, lo, hi = paired_bootstrap_ci(nul); assert lo < 0 < hi
m, lo, hi = paired_bootstrap_ci(nul + 0.01); assert lo > 0

# 6) control logic on synthetic tokens: big image-specific residual x5 swamps a small lesion signal in attn.
#    removing x5 must help; swapping x5 for another image's (same norm) must NOT help.
C, P, n_img = 32, 200, 40
w = np.random.RandomState(3).randn(C); w /= np.linalg.norm(w)
rs = np.random.RandomState(4)


def img():
    mask = np.zeros(P, bool); s0 = rs.randint(0, P - 30); mask[s0:s0 + 30] = True
    attn = rs.randn(P, C) * 0.3 + 1.0 * mask[:, None] * w[None]
    x5_ = rs.randn(P, C) * 3.0
    return x5_, attn, mask


def score(tok):
    tok = tok / np.linalg.norm(tok, axis=1, keepdims=True)
    return tok @ w


data = [img() for _ in range(n_img)]
partner = swap_pairing(n_img, 0)
base = np.array([auroc(m_, score(a + x)) for x, a, m_ in data])
nores = np.array([auroc(m_, score(a)) for x, a, m_ in data])
swap = np.array([auroc(m_, score(a + data[partner[i]][0])) for i, (x, a, m_) in enumerate(data)])
assert (nores - base).mean() > 0.2, (nores - base).mean()
assert abs((swap - base).mean()) < 0.03, (swap - base).mean()
# and a case where the signal lives IN the residual (removal hurts): arm < base
data2 = []
for _ in range(n_img):
    x5_, a_, m_ = img(); x5_ = x5_ * 0.1 + 1.0 * m_[:, None] * w[None]; data2.append((x5_, rs.randn(P, C) * 0.3, m_))
b2 = np.array([auroc(m_, score(a + x)) for x, a, m_ in data2]); n2 = np.array([auroc(m_, score(a)) for x, a, m_ in data2])
assert (n2 - b2).mean() < -0.1

# 7) verdict rule, known answers
z = V(0.0, 0.0, 0.0)
good = dict(d=V(0.010, 0.012, 0.008), lo=V(0.004, 0.006, 0.002), g=V(0.009, 0.010, 0.007))
nop = dict(d=V(0.0, 0.001, -0.001), lo=V(-0.003, -0.002, -0.004), g=V(0.0, 0.001, -0.001))


def mk(a, b):
    return ({"no_res": a["d"], "no_res_last": b["d"]}, {"no_res": a["lo"], "no_res_last": b["lo"]},
            {"no_res": a["g"], "no_res_last": b["g"]})


assert verdict(*mk(good, nop)).startswith("CONFIRM via no_res")
assert verdict(*mk(nop, good)).startswith("CONFIRM via no_res_last")
assert "no_res_last" not in verdict(*mk(good, nop))
assert verdict(*mk(nop, nop)).startswith("FALSIFY")
# gains reproduced by swap_res -> gap small -> INCONCLUSIVE (not CONFIRM, not FALSIFY)
swaprep = dict(d=V(0.010, 0.012, 0.008), lo=V(0.004, 0.006, 0.002), g=V(0.001, 0.0, 0.002))
assert verdict(*mk(swaprep, nop)) == "INCONCLUSIVE"
# 2/3 sets good, third set slightly worse than base but > -0.005 -> still passes
two = dict(d=V(0.010, 0.012, -0.002), lo=V(0.004, 0.006, -0.01), g=V(0.009, 0.010, -0.002))
assert verdict(*mk(two, nop)).startswith("CONFIRM")
# 2/3 good but one set <= -0.005 -> INCONCLUSIVE
neg = dict(d=V(0.010, 0.012, -0.006), lo=V(0.004, 0.006, -0.01), g=V(0.009, 0.010, -0.006))
assert verdict(*mk(neg, nop)) == "INCONCLUSIVE"
# only 1/3 sets good -> not confirm; d>=0.003 on <2 sets -> falsify
one = dict(d=V(0.010, 0.001, 0.0), lo=V(0.004, -0.003, -0.003), g=V(0.009, 0.0, 0.0))
assert verdict(*mk(one, nop)).startswith("FALSIFY")
# 0.004 everywhere: above kill (0.003) but below gain (0.005) -> INCONCLUSIVE
mid = dict(d=V(0.004, 0.004, 0.004), lo=V(0.001, 0.001, 0.001), g=V(0.004, 0.004, 0.004))
assert verdict(*mk(mid, nop)) == "INCONCLUSIVE"
# CI lower bound <= 0 on all sets blocks CONFIRM even if mean >= 0.005
ci = dict(d=V(0.006, 0.006, 0.006), lo=V(-0.001, -0.001, -0.001), g=V(0.006, 0.006, 0.006))
assert verdict(*mk(ci, nop)) == "INCONCLUSIVE"
# sign flip (strong loss on all sets) -> FALSIFY
loss = dict(d=V(-0.02, -0.03, -0.01), lo=V(-0.03, -0.04, -0.02), g=V(-0.02, -0.03, -0.01))
assert verdict(*mk(loss, loss)).startswith("FALSIFY")
print("ALL resid_stats TESTS PASSED")
