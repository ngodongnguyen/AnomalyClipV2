"""Known-answer tests of the EXP-044 token-prompt module (pure torch, CPU). Run: python test_token_prompt.py"""
import math
import torch
import torch.nn.functional as F
import token_prompt as T

torch.manual_seed(0)
C = 768

# --- heads: Q=3 starts as three disjoint coordinate subspaces (orthogonal, Gram penalty 0); Q=1 is the identity
text = F.normalize(torch.randn(4, 2, C), dim=-1)
h3 = T.TokenHeads(3)
sub = h3(text)
assert sub.shape == (4, 3, 2, C)
assert torch.allclose(sub.norm(dim=-1), torch.ones(4, 3, 2), atol=1e-5)
assert float(T.ortho_loss(sub)) < 1e-10
w = torch.stack([h.weight for h in h3.heads])
assert torch.equal((w != 0).sum(0).max(), torch.tensor(1)) and int((w != 0).sum()) == C   # disjoint blocks, each coordinate once
h1 = T.TokenHeads(1)
assert torch.allclose(h1(text)[:, 0], text, atol=1e-6)
assert int((torch.stack([h.weight for h in T.TokenHeads(5).heads]) != 0).sum()) == C

# --- Q=1: the OT map equals the repo's batched_similarity (softmax over the two classes of cos / 0.07)
patch = F.normalize(torch.randn(4, 50, C), dim=-1)
cfg = T.default_cfg(1)
wt = T.ot_weights(patch, h1(text), cfg)
assert torch.equal(wt, torch.ones(4, 50, 1))
ref = (torch.einsum("bnc,btc->bnt", patch, text) / 0.07).softmax(-1)      # extent_prompt.batched_similarity
assert torch.allclose(T.token_probs(patch, h1(text), wt), ref, atol=1e-5)
assert torch.allclose(T.token_probs(patch, h3(text), T.ot_weights(patch, h3(text), T.default_cfg(3))).sum(-1), torch.ones(4, 50), atol=1e-5)

torch.manual_seed(1)
# --- Sinkhorn: three well separated clusters of equal size are routed to their own subspace, balanced columns, no NaN
cent = F.normalize(torch.randn(3, C), dim=-1)
tok = torch.cat([F.normalize(cent[q] + 0.05 * torch.randn(40, C), dim=-1) for q in range(3)])[None]       # [1, 120, C]
cost = 1 - torch.einsum("bnc,qc->bnq", tok, cent)
plan = T.sinkhorn_plan(cost, 0.01, 100)
assert torch.isfinite(plan).all() and torch.allclose(plan.sum(-1), torch.ones(1, 120), atol=1e-5)
assert (plan.argmax(-1)[0] == torch.arange(3).repeat_interleave(40)).all()
assert float(plan.max(-1).values.min()) > 0.99
assert torch.allclose(plan.mean(1), torch.full((1, 3), 1 / 3), atol=1e-3)     # uniform subspace usage
# unequal clusters: the OT marginal forces balanced usage anyway (documented behaviour of the uniform column marginal)
cost_u = cost[:, :100]            # 40 / 40 / 20 (100 iterations do not fully converge: loose tolerance)
plan_u = T.sinkhorn_plan(cost_u, 0.01, 100)
assert torch.allclose(plan_u.mean(1), torch.full((1, 3), 1 / 3), atol=0.02)
# large costs / tiny eps do not under-flow (log domain)
assert torch.isfinite(T.sinkhorn_plan(torch.rand(2, 200, 4) * 2, 1e-3, 100)).all()

# --- sparsify: top-k, threshold, argmax always kept, rows renormalised
w0 = torch.tensor([[0.5, 0.3, 0.2], [0.34, 0.33, 0.33], [0.9, 0.06, 0.04], [0.45, 0.45, 0.10]])
sp = T.sparsify(w0, 2, 0.2)
assert torch.allclose(sp.sum(-1), torch.ones(4)) and ((sp > 0).sum(-1) <= 2).all()
assert torch.allclose(sp[0], torch.tensor([0.5 / 0.8, 0.3 / 0.8, 0.0]))
assert torch.allclose(sp[2], torch.tensor([1.0, 0.0, 0.0]))                      # others below 0.2
assert torch.allclose(sp[3], torch.tensor([0.5, 0.5, 0.0]))
assert (sp[1] > 0).sum() == 2
assert torch.equal(T.sparsify(torch.tensor([[0.15, 0.15, 0.7]]), 2, 0.8).argmax(-1), torch.tensor([2]))   # nothing passes: argmax kept

# --- ortho: identical subspaces -> ||G - I||_F^2 = 2 (Q=2: two unit off-diagonal entries); orthogonal -> 0
e = F.normalize(torch.randn(1, 1, 2, C), dim=-1)
same = e.expand(1, 2, 2, C).contiguous()
assert abs(float(T.ortho_loss(same)) - 2.0) < 1e-5
q = torch.linalg.qr(torch.randn(C, 2))[0].T                        # two orthonormal vectors
orth = q[None, :, None, :].expand(1, 2, 2, C).contiguous()
assert float(T.ortho_loss(orth)) < 1e-9
assert float(T.ortho_loss(F.normalize(torch.randn(3, 1, 2, C), dim=-1))) < 1e-9   # Q=1: G = [[1]]

# --- hinge: known values and empty groups
m = torch.tensor([[-6.0, -4.0, 6.0, 4.0]]); ab = torch.tensor([[False, False, True, True]])
# normal: relu(5 + [-6, -4]) = [0, 1] -> mean 0.5 ; anomalous: relu(5 - [6, 4]) = [0, 1] -> mean 0.5 ; total 1.0
assert abs(float(T.hinge_loss(m, ab, 5.0)) - 1.0) < 1e-6
assert float(T.hinge_loss(torch.zeros(1, 3), torch.zeros(1, 3, dtype=torch.bool), 5.0)) == 5.0     # only normal group: relu(5 + 0)
assert float(T.hinge_loss(torch.zeros(1, 3), torch.ones(1, 3, dtype=torch.bool), 5.0)) == 5.0
mm = torch.zeros(1, 2, requires_grad=True)
T.hinge_loss(mm, torch.tensor([[True, False]]), 5.0).backward()
assert torch.allclose(mm.grad, torch.tensor([[-1.0, 1.0]]))

# --- margin: scale and routing weights
ptc = F.normalize(torch.randn(1, 5, C), dim=-1)
s1 = F.normalize(torch.randn(1, 1, 2, C), dim=-1)
mg = T.token_margin(ptc, s1, torch.ones(1, 5, 1), 100.0)
cos = torch.einsum("bnc,bkc->bnk", ptc, s1[:, 0])
assert torch.allclose(mg, 100.0 * (cos[..., 1] - cos[..., 0]), atol=1e-4)

# --- patch labels: left half of the mask anomalous -> left half of the 37x37 grid (rounded by pooling)
gt = torch.zeros(2, 518, 518); gt[0, :, :259] = 1
lab = T.patch_labels(gt, 37 * 37)
assert lab.shape == (2, 1369) and not lab[1].any()
g = lab[0].reshape(37, 37)
assert g[:, :18].all() and not g[:, 19:].any()
assert T.patch_labels(torch.zeros(518, 518), 1369).shape == (1, 1369)

# --- gradients: the OT-map loss reaches the heads and the text features; ortho pushes identical heads apart
tx = text.clone().requires_grad_(True)
hh = T.TokenHeads(3)
sb = hh(tx)
ww = T.ot_weights(patch, sb.detach(), T.default_cfg(3))
pr = T.token_probs(patch, sb, ww)
y = (torch.rand(4, 50) > 0.5).float()
(F.binary_cross_entropy(pr[..., 1].clamp(1e-6, 1 - 1e-6), y) + T.hinge_loss(T.token_margin(patch, sb, ww), y.bool())).backward()
assert tx.grad is not None and torch.isfinite(tx.grad).all() and all(float(hd.weight.grad.abs().sum()) > 0 for hd in hh.heads)
hs = T.TokenHeads(2)
with torch.no_grad():
    hs.heads[1].weight.copy_(hs.heads[0].weight)
hs.heads[1].weight.data += 1e-2 * torch.randn(C, C)        # nearly identical subspaces
o0 = float(T.ortho_loss(hs(text)))
opt = torch.optim.SGD(hs.parameters(), lr=0.5)
for _ in range(20):
    opt.zero_grad(); T.ortho_loss(hs(text)).backward(); opt.step()
assert float(T.ortho_loss(hs(text))) < 0.5 * o0

print("test_token_prompt OK")
