"""Pure-numpy core for analyze_cross_image_rarity.py (unit-testable without torch)."""
import numpy as np

def allow_matrix(G, m):
    """G: (n_img,D) L2-normalised global feats. allow[i,j]=True iff image j may serve as a key for queries from image i:
    j != i and j is NOT among the m images most similar to i (kills near-duplicate video frames)."""
    n = len(G); S = G @ G.T; np.fill_diagonal(S, -np.inf)
    allow = np.ones((n, n), bool); np.fill_diagonal(allow, False)
    if m > 0:
        for i in range(n):
            allow[i, np.argsort(-S[i])[:m]] = False
    return allow

def knn_mean_np(Q, q_img, K, k_img, allow, k=5, chunk=512):
    """mean over the k largest cosine sims between each query and keys from allowed images. Q,K L2-normalised."""
    out = np.empty(len(Q), np.float64)
    for a in range(0, len(Q), chunk):
        S = Q[a:a + chunk] @ K.T
        S = np.where(allow[q_img[a:a + chunk][:, None], k_img[None, :]], S, -np.inf)
        top = np.partition(S, -k, axis=1)[:, -k:]
        out[a:a + chunk] = top.mean(1)
    return out
