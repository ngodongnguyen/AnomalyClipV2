"""Tests for pe_shift_core.py (python test_pe_shift_core.py). Needs torch (CPU)."""
import sys
import numpy as np
import torch
import pe_shift_core as C


def synth(side=5, w=3):
    pe = torch.zeros(side * side + 1, w)
    pe[0] = -1.0
    for i in range(side):
        for j in range(side):
            pe[1 + i * side + j] = torch.tensor([i, j, 10 * i + j], dtype=torch.float32)
    return pe


def test_zero_shift_bitwise():
    pe = synth()
    new, valid = C.shift_pe(pe, 0, 0)
    assert torch.equal(new, pe) and valid.all()


def test_known_answer_shift_clamp_valid():
    pe = synth(5)
    new, valid = C.shift_pe(pe, 2, -1)               # token (i,j) <- cell (clip(i+2), clip(j-1))
    for i in range(5):
        for j in range(5):
            ei, ej = min(max(i + 2, 0), 4), min(max(j - 1, 0), 4)
            assert new[1 + i * 5 + j].tolist() == [ei, ej, 10 * ei + ej]
            assert valid[i, j] == (0 <= i + 2 < 5 and 0 <= j - 1 < 5)
    assert torch.equal(new[0], pe[0])                # CLS row untouched
    assert valid.sum() == 3 * 4
    # lesion-to-centre convention: token at (ci,cj) receives the PE of the centre when s = centre - c
    ci, cj, c = 0, 4, 2
    new, _ = C.shift_pe(pe, c - ci, c - cj)
    assert new[1 + ci * 5 + cj].tolist()[:2] == [c, c]
    # shift larger than the grid: everything clamped, nothing valid
    new, valid = C.shift_pe(pe, 7, 0)
    assert not valid.any() and new[1:, 0].eq(4).all()


def test_flat():
    pe = synth()
    f = C.flat_pe(pe)
    assert torch.equal(f[0], pe[0])
    assert torch.allclose(f[1:], pe[1:].mean(0, keepdim=True).expand(25, 3))


class Holder(torch.nn.Module):
    def __init__(self, pe):
        super().__init__(); self.positional_embedding = torch.nn.Parameter(pe.clone())


def test_restore_even_on_exception():
    h = Holder(synth()); before = h.positional_embedding.data
    try:
        with C.patched_pe(h, "shift", 1, 1):
            assert not torch.equal(h.positional_embedding.data, before)
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert h.positional_embedding.data.data_ptr() == before.data_ptr() and torch.equal(h.positional_embedding.data, before)
    with C.patched_pe(h, "flat"):
        pass
    assert h.positional_embedding.data.data_ptr() == before.data_ptr() and torch.equal(h.positional_embedding.data, before)


def test_ensure_grid():
    h = Holder(synth(5))
    C.ensure_grid(h, 5)
    try:
        C.ensure_grid(h, 37); raise AssertionError("should stop")
    except RuntimeError:
        pass


def test_real_vit_integration():
    from AnomalyCLIP_lib.AnomalyCLIP import VisionTransformer
    torch.manual_seed(0)
    # 56 px input, patch 14 -> 4x4 grid; native PE grid 2x2 -> first forward interpolates (as 24 -> 37 on the server)
    v = VisionTransformer(input_resolution=28, patch_size=14, width=64, layers=3, heads=2, output_dim=16).eval()
    v.DAPM_replace(DPAM_layer=3)                      # as load_model does (DPAM_layer=30 on the real model)
    x = torch.randn(1, 3, 56, 56)
    with torch.no_grad():
        v(x, [3], DPAM_layer=3)                       # warm-up: interpolates and stores PE
        C.ensure_grid(v, 4)
        base_cls, base_p = v(x, [3], DPAM_layer=3)
        with C.patched_pe(v, "shift", 0, 0):
            c0, p0 = v(x, [3], DPAM_layer=3)
        assert torch.equal(c0, base_cls) and torch.equal(p0[0], base_p[0])      # (0,0) shift = original, bitwise
        with C.patched_pe(v, "shift", 1, -1) as valid:
            c1, p1 = v(x, [3], DPAM_layer=3)
        assert valid.shape == (4, 4) and valid.sum() == 9
        assert not torch.equal(p1[0], base_p[0])                                # a real shift changes the patch tokens
        c2, p2 = v(x, [3], DPAM_layer=3)
        assert torch.equal(p2[0], base_p[0]) and torch.equal(c2, base_cls)      # restored


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
    print("ALL PASSED")
