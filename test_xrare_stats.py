import sys, numpy as np
sys.path.insert(0, "/Users/nguyen.ngo.1/Nguyen/AnomalyCLIP"); sys.path.insert(0, ".")
from xrare_stats import allow_matrix, knn_mean_np
from within_stats import auroc
rng = np.random.default_rng(0)
def unit(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)

# T1: brute-force equality + same-image exclusion (patch duplicated inside one image must NOT count as neighbour)
n_img, npch, D = 6, 20, 8
F = unit(rng.normal(size=(n_img * npch, D))); img = np.repeat(np.arange(n_img), npch)
F[1] = F[0]                                   # exact duplicate inside image 0
G = unit(rng.normal(size=(n_img, D))); A = allow_matrix(G, 0)
c = knn_mean_np(F, img, F, img, A, k=3, chunk=7)
bf = []
for q in range(len(F)):
    s = sorted([F[q] @ F[j] for j in range(len(F)) if img[j] != img[q]])[-3:]; bf.append(np.mean(s))
assert np.allclose(c, bf), "knn mismatch"
# T2: near-image exclusion
G2 = unit(np.eye(n_img, D) + 0.01 * rng.normal(size=(n_img, D)))
A2 = allow_matrix(G2, 2); assert A2.sum(1).tolist() == [n_img - 1 - 2] * n_img and not A2.diagonal().any()
# T3: known answer. Normal patches come from 5 shared tissue clusters; 'lesion-like' patches = random unique directions
# per image. Commonness must separate: AUROC(unique vs shared | -c) ~ 1; shuffled labels ~ 0.5.
cent = unit(rng.normal(size=(5, 64))); n_img, per = 30, 40
P, lab, im = [], [], []
for i in range(n_img):
    for j in range(per):
        if j < 4: v = rng.normal(size=64); lab.append(1)
        else: v = cent[rng.integers(5)] * 6 + rng.normal(size=64); lab.append(0)
        P.append(v); im.append(i)
P = unit(np.array(P)); lab = np.array(lab, bool); im = np.array(im)
A3 = allow_matrix(unit(rng.normal(size=(n_img, 64))), 0)
c3 = knn_mean_np(P, im, P, im, A3, k=5)
a = auroc(lab, -c3); ctl = auroc(rng.permutation(lab), -c3)
print("T3 rare-vs-common AUROC", round(a, 3), "shuffled control", round(ctl, 3)); assert a > 0.95 and 0.4 < ctl < 0.6
# T4: if the lesion class is itself a recurring cluster (polyps in 100%-abnormal sets) the rarity signal must collapse toward 0.5
lesion = unit(rng.normal(size=64))
P4 = P.copy(); P4[lab] = unit(lesion * 6 + rng.normal(size=(lab.sum(), 64)))
c4 = knn_mean_np(P4, im, P4, im, A3, k=5); a4 = auroc(lab, -c4); print("T4 replicated-lesion AUROC", round(a4, 3)); assert a4 < 0.75 and a4 < a - 0.2
print("ALL OK")
