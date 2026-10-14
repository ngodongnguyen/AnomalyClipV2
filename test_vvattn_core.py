"""Tests for vvattn_core.py (CPU torch, no CLIP weights). Run: python test_vvattn_core.py"""
import numpy as np
import torch
import vvattn_core as C
import vvattn_stats as S


def brute_entmax15(z):
    """float64 bisection on tau for sum max(z/2 - tau, 0)^2 = 1."""
    z = np.asarray(z, np.float64) / 2
    lo, hi = z.max() - 1.0, z.max()                      # f(lo) >= 1, f(hi) = 0
    for _ in range(200):
        mid = (lo + hi) / 2
        f = (np.maximum(z - mid, 0) ** 2).sum()
        lo, hi = (mid, hi) if f >= 1 else (lo, mid)
    return np.maximum(z - (lo + hi) / 2, 0) ** 2


def test_entmax_known_answers():
    p = C.entmax15(torch.tensor([0.0, 0.0]))
    assert torch.allclose(p, torch.tensor([0.5, 0.5]), atol=1e-6)
    p = C.entmax15(torch.tensor([10.0, 0.0]))
    assert torch.allclose(p, torch.tensor([1.0, 0.0]), atol=1e-6)               # dominant logit -> one-hot
    p = C.entmax15(torch.tensor([1.0, 0.0]))
    ref = brute_entmax15([1.0, 0.0])
    assert abs(float(p.sum()) - 1) < 1e-6 and np.allclose(p.numpy(), ref, atol=1e-6)
    assert ref[0] > ref[1] > 0   # both on the support, order kept


def test_entmax_matches_numeric_solution_and_is_sparse():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(6, 5, 200, generator=g) * 4
    p = C.entmax15(x, dim=-1)
    assert (p >= 0).all() and torch.allclose(p.sum(-1), torch.ones(6, 5), atol=1e-5)
    for i in range(6):
        for j in range(5):
            assert np.allclose(p[i, j].numpy(), brute_entmax15(x[i, j].numpy()), atol=2e-5)
    # sparse: with logit spread ~4 a large share of entries is exactly zero, softmax has none
    assert (p == 0).float().mean() > 0.3 and (torch.softmax(x, -1) == 0).float().mean() == 0
    # ordering as softmax: argsort of probabilities equals argsort of logits over the support
    for r in range(3):
        sup = p[0, r] > 0
        a = torch.argsort(x[0, r][sup]); b = torch.argsort(p[0, r][sup])
        assert torch.equal(a, b)
    # invariant to a constant shift of the logits, dim argument works
    assert torch.allclose(C.entmax15(x + 7.0), p, atol=1e-5)
    assert torch.allclose(C.entmax15(x.transpose(1, 2), dim=1), p.transpose(1, 2), atol=1e-6)
    # 1370-key rows as in the model (N = 1 + 37*37), float32 with large logits
    big = torch.randn(2, 4, 1370, generator=g) * 12
    pb = C.entmax15(big)
    assert torch.allclose(pb.sum(-1), torch.ones(2, 4), atol=1e-4) and np.allclose(pb[1, 3].numpy(), brute_entmax15(big[1, 3].numpy()), atol=1e-4)


def make_attn(dim=32, heads=4, seed=0):
    from AnomalyCLIP_lib.AnomalyCLIP import Attention
    torch.manual_seed(seed)
    a = Attention(dim, dim, num_heads=heads, qkv_bias=True).eval()
    a.layer_idx = 10
    return a


def run_patched(a, x, state):
    from AnomalyCLIP_lib.AnomalyCLIP import Attention
    orig = Attention.forward
    Attention.forward = C.make_vv_forward(state)
    try:
        with torch.no_grad():
            return a(x)
    finally:
        Attention.forward = orig


def test_patched_forward_default_equals_original_and_masks():
    a = make_attn(); x = torch.randn(1, 17, 32)
    with torch.no_grad():
        ref_x, ref_ori = a(x)
    st = C.new_state()
    out_x, out_ori = run_patched(a, x, st)
    assert torch.equal(out_x, ref_x) and torch.equal(out_ori, ref_ori)                 # default state == original, bitwise
    st["hmask"] = {10: torch.ones(4)}                                                    # all-ones mask == baseline exactly
    m_x, m_ori = run_patched(a, x, st)
    assert torch.equal(m_x, ref_x) and torch.equal(m_ori, ref_ori)
    for h in range(4):                                                                   # single-head masks == manually zeroing that slice
        keep = torch.ones(4); keep[h] = 0
        st["hmask"] = {10: keep}
        s_x, s_ori = run_patched(a, x, st)
        B, N, Cc = x.shape
        qkv = a.qkv(x).reshape(B, N, 3, 4, Cc // 4).permute(2, 0, 3, 1, 4)
        v = qkv[2]
        o = (((v @ v.transpose(-2, -1)) * a.scale).softmax(-1) @ v).transpose(1, 2).reshape(B, N, Cc).clone()
        o[:, :, h * (Cc // 4):(h + 1) * (Cc // 4)] = 0
        with torch.no_grad():
            man = a.proj(o)
        assert torch.allclose(s_x, man, atol=1e-6) and torch.equal(s_ori, ref_ori)       # Q-K stream unchanged
        assert not torch.allclose(s_x, ref_x, atol=1e-4)
    st["hmask"] = {7: torch.zeros(4)}                                                    # a mask for another layer does nothing here
    assert torch.equal(run_patched(a, x, st)[0], ref_x)


def test_modes_change_only_vv_stream_and_respect_first_layer():
    a = make_attn(); x = torch.randn(1, 17, 32) * 3
    with torch.no_grad():
        ref_x, ref_ori = a(x)
    for kind in ("ent_all", "sharp_all"):
        st = C.new_state(); C.set_arm(st, kind)
        o_x, o_ori = run_patched(a, x, st)
        assert torch.equal(o_ori, ref_ori) and not torch.allclose(o_x, ref_x, atol=1e-4)
    st = C.new_state(); C.set_arm(st, "ent_last6")                                       # layer 10 < 19: unchanged
    assert torch.equal(run_patched(a, x, st)[0], ref_x)
    a.layer_idx = 20
    st = C.new_state(); C.set_arm(st, "ent_last6")
    with torch.no_grad():
        ref20 = a(x)[0]
    assert not torch.allclose(run_patched(a, x, st)[0], ref20, atol=1e-4)
    # sharp == softmax(2*logits)
    st = C.new_state(); C.set_arm(st, "sharp_all")
    B, N, Cc = x.shape
    v = a.qkv(x).reshape(B, N, 3, 4, Cc // 4).permute(2, 0, 3, 1, 4)[2]
    man = a.proj((((v @ v.transpose(-2, -1)) * a.scale * 2).softmax(-1) @ v).transpose(1, 2).reshape(B, N, Cc))
    assert torch.allclose(run_patched(a, x, st)[0], man, atol=1e-5)


def test_set_arm_mask_dict():
    keep = S.random_mask(152, 0)
    st = C.new_state(); C.set_arm(st, "head", keep)
    assert set(st["hmask"]) == set(range(6, 25)) and float(sum(v.sum() for v in st["hmask"].values())) == 152
    C.set_arm(st, "base")
    assert st["hmask"] is None and st["mode"] == "softmax"


def test_head_score_accumulator_planted():
    G = 6; n = G * G
    lab = torch.zeros(n, dtype=torch.bool); lab[:12] = True                # 12 lesion patches (first two rows), others normal
    kl, kn = lab, ~lab
    q = torch.nonzero(lab)[:, 0]
    acc = C.HeadScoreAccumulator()
    acc.set_image(kl, kn, q)
    torch.manual_seed(0)
    attn = torch.zeros(1, S.N_HEADS, 1 + n, 1 + n)
    base = torch.rand(S.N_HEADS, 1 + n, 1 + n) + 0.01
    base[0, 1:, 1:][:, :12] *= 50                                           # head 0: lesion keys get 50x the weight (planted signal)
    attn[0] = base / base.sum(-1, keepdim=True)
    for layer in range(S.FIRST_LAYER, 25):
        acc(layer, attn)
    acc(3, attn)                                                            # layers < 6 are ignored
    pos, neg = acc.numpy()
    au = S.hist_auroc(pos, neg)
    assert au.shape == (19, 16)
    assert (au[:, 0] > 0.99).all() and (np.abs(au[:, 1:] - 0.5) < 0.1).all(), (au[0, :3])
    assert pos[0, 0].sum() == 12 * 11 and neg[0, 0].sum() == 12 * 24         # self pair excluded; keys: 11 other lesion, 24 normal


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
    print("test_vvattn_core: ALL OK")
