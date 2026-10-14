"""
TokenCLIP-style per-patch prompt subspaces with optimal-transport (OT) routing, on top of ECP (EXP-044, H033).

Reference: TokenCLIP, arXiv 2510.21171. The module below is rebuilt from a scout's summary of the arXiv page only; the
paper's code and several details were NOT available. Every choice that the summary did not fix is listed here and must
be treated as FIXED before the first run (no tuning on the medical sets afterwards).

What the module does (default off; nothing in train.py/test.py touches it unless --tok_q > 0):
  1. The ECP-conditioned text features T [B, 2, C] (index 0 = normal, 1 = anomalous, L2-normalised, ECP shift already
     applied: z stays the outermost conditioning) are mapped by Q heads into Q "subspace" embeddings
     S [B, Q, 2, C] = normalize(W_q T).
  2. The patch tokens (frozen encoder, L2-normalised) are routed to the Q subspaces by entropic optimal transport
     (Sinkhorn), cost = 1 - cosine(patch, subspace), uniform marginals (1/N per patch, 1/Q per subspace).
  3. The plan is sparsified (top-k, threshold) and renormalised per patch -> weights w [B, N, Q].
  4. Per-patch score = sum_q w_nq * softmax_k(cos(patch_n, S_qk) / 0.07)  -> a [B, N, 2] normal/anomalous
     probability pair, i.e. exactly the format of the existing patch read-out, so the OT map reuses the repo's own
     get_similarity_map and focal+Dice losses.
  5. Training: OT map gets the same focal + Dice losses as the base map (same weight lam); plus a hinge loss on the
     per-patch cosine margin and a Gram orthogonality penalty on the subspace embeddings.
     Test: pixel map = mean(OT map, base map), then the usual sigma smoothing. The image head is unchanged.

DESIGN CHOICES NOT FIXED BY THE SOURCE (all are builder decisions; see the report that accompanies this file):
  C1 Heads: Q bias-free Linear(C, C) layers, no activation ("single-layer MLP" read as one linear layer), NOT residual.
     Initialisation (the paper's is unknown): W_q = diag(mask_q), with the C coordinates split into Q contiguous,
     disjoint blocks. Hence at init the Q subspace embeddings are exactly orthogonal (Gram penalty = 0) and each keeps
     the original text direction restricted to its block (a literal coordinate subspace). For Q = 1 the mask is all
     ones, W = I and the OT map equals the base map at init (the TOKQ1 control starts as plain ECP).
  C2 Cost matrix: cost_nq = 1 - cos(patch_n, c_q), with c_q = normalize(S_q,normal + S_q,anomalous) (the subspace
     centroid of its two class embeddings). The paper's cost-matrix input is not specified in the summary.
  C3 The OT plan is computed in log domain (eps = 0.01 on costs in [0, 2] under-flows in the plain kernel form) and is
     DETACHED (no gradient through the assignment); gradients reach the heads through the per-patch scores.
     Plan row-normalised so that every patch's weights sum to 1 (after 100 iterations the row sums are ~1/N).
  C4 Sparsification: keep the top-k (k = 2) entries that are >= 0.2 (of the row-normalised plan), always keep the
     arg-max, renormalise the row. (Q = 1: the single weight is 1.)
  C5 Hinge loss: per-patch margin m_n = hinge_scale * sum_q w_nq (cos(patch_n, S_q,anom) - cos(patch_n, S_q,norm));
     patch label = average-pooled GT mask (to the 37 x 37 grid) > 0.5. Loss = mean_{normal patches} relu(eta + m)
     + mean_{anomalous patches} relu(eta - m), empty group skipped; eta = 5. hinge_scale = 100 (CLIP's logit
     scale) is a GUESS: with the repo's 1/0.07 the margin eta = 5 would need a cosine gap of 0.35 (unreachable) while
     scale 100 needs 0.05. Its weight is the --tok_hinge flag (1.0 in the registered TOK/TOKQ1 runs).
  C6 Orthogonality: L_ortho = mean over images and the two classes of ||G - I||_F^2, G = S_k S_k^T over the Q
     L2-normalised subspace embeddings of class k (sum of squared entries, not divided by Q^2); weight xi = 100
     (--tok_ortho).
  C7 Loss weights: OT-map focal+Dice terms are added to the pixel loss before the existing factor lam = 4 (same
     treatment as the base map); hinge and ortho are added outside lam.
  C8 Heads are trained jointly with the prompts and the extent conditioner (Adam lr 0.001, as everything else).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TEMP = 0.07                  # the repo's text-patch temperature
TOK_EPS = 0.01
TOK_ITERS = 100
TOK_TOPK = 2
TOK_THR = 0.2
TOK_ETA = 5.0
TOK_HINGE_SCALE = 100.0


def default_cfg(q):
    return dict(q=int(q), eps=TOK_EPS, iters=TOK_ITERS, topk=TOK_TOPK, thr=TOK_THR, eta=TOK_ETA, hinge_scale=TOK_HINGE_SCALE)


class TokenHeads(nn.Module):
    """Q single-layer linear heads: text features [B, 2, C] -> unit subspace embeddings [B, Q, 2, C]."""

    def __init__(self, q, dim=768):
        super().__init__()
        assert q >= 1
        self.q, self.dim = q, dim
        self.heads = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(q)])
        with torch.no_grad():
            for h, block in zip(self.heads, np.array_split(np.arange(dim), q)):
                h.weight.zero_()
                h.weight[block, block] = 1.0   # diag(mask_q): disjoint coordinate blocks (C1)

    def forward(self, text):
        out = torch.stack([h(text.float()) for h in self.heads], dim=1)   # [B, Q, 2, C]
        return F.normalize(out, dim=-1)


@torch.no_grad()
def sinkhorn_plan(cost, eps=TOK_EPS, iters=TOK_ITERS):
    """
    Log-domain entropic OT. cost [B, N, Q]; marginals 1/N (patches) and 1/Q (subspaces).
    Returns the plan row-normalised so that each patch's weights sum to 1: [B, N, Q].
    """
    B, N, Q = cost.shape
    logk = -cost.float() / eps
    log_a, log_b = -float(np.log(N)), -float(np.log(Q))
    u = torch.zeros(B, N, device=cost.device, dtype=logk.dtype)
    v = torch.zeros(B, Q, device=cost.device, dtype=logk.dtype)
    for _ in range(iters):
        u = log_a - torch.logsumexp(logk + v[:, None, :], dim=2)
        v = log_b - torch.logsumexp(logk + u[:, :, None], dim=1)
    plan = torch.exp(logk + u[:, :, None] + v[:, None, :])
    return plan / plan.sum(-1, keepdim=True).clamp_min(1e-30)


def sparsify(w, k=TOK_TOPK, thr=TOK_THR):
    """Keep the top-k entries of each row that are >= thr (the arg-max is always kept), renormalise rows."""
    vals, idx = w.topk(min(k, w.shape[-1]), dim=-1)
    keep = (vals >= thr)
    keep[..., 0] = True
    mask = torch.zeros_like(w).scatter(-1, idx, keep.to(w.dtype))
    out = w * mask
    return out / out.sum(-1, keepdim=True).clamp_min(1e-30)


@torch.no_grad()
def ot_weights(patch, sub, cfg):
    """patch [B, N, C] unit tokens (no CLS), sub [B, Q, 2, C] unit embeddings -> sparse routing weights [B, N, Q]."""
    centroid = F.normalize(sub.float().sum(2), dim=-1)                       # [B, Q, C]   (C2)
    cost = 1.0 - torch.einsum("bnc,bqc->bnq", patch.float(), centroid)
    return sparsify(sinkhorn_plan(cost, cfg["eps"], cfg["iters"]), cfg["topk"], cfg["thr"])


def token_probs(patch, sub, w):
    """Per-patch (normal, anomalous) probability pair [B, N, 2]: routing-weighted softmax over classes per subspace."""
    logits = torch.einsum("bnc,bqkc->bnqk", patch.float(), sub) / TEMP
    return (w.unsqueeze(-1) * logits.softmax(-1)).sum(2)


def token_margin(patch, sub, w, scale=TOK_HINGE_SCALE):
    """Routing-weighted cosine margin (anomalous - normal), scaled: [B, N]."""
    cos = torch.einsum("bnc,bqkc->bnqk", patch.float(), sub)
    return scale * (w * (cos[..., 1] - cos[..., 0])).sum(-1)


def hinge_loss(margin, abn, eta=TOK_ETA):
    """margin [B, N]; abn [B, N] bool (patch is anomalous). Normal patches want m <= -eta, anomalous m >= eta."""
    loss = margin.sum() * 0.0
    if (~abn).any():
        loss = loss + F.relu(eta + margin[~abn]).mean()
    if abn.any():
        loss = loss + F.relu(eta - margin[abn]).mean()
    return loss


def ortho_loss(sub):
    """sub [B, Q, 2, C] unit embeddings. Mean over images and classes of ||G - I||_F^2 (Gram over the Q subspaces)."""
    g = torch.einsum("bqkc,brkc->bkqr", sub, sub)
    eye = torch.eye(sub.shape[1], device=sub.device, dtype=g.dtype)
    return ((g - eye) ** 2).sum((-1, -2)).mean()


def patch_labels(gt, n_patches):
    """gt [B, H, W] (or [H, W]) binary mask -> bool [B, N]: average-pooled to the patch grid, > 0.5 is anomalous."""
    side = int(round(n_patches ** 0.5))
    gt = gt.float()
    gt = gt.reshape(-1, 1, gt.shape[-2], gt.shape[-1])
    return (F.adaptive_avg_pool2d(gt, side) > 0.5).reshape(gt.shape[0], -1)
