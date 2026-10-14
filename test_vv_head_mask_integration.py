"""EXP-046 CPU integration test with the REAL VisionTransformer / Attention classes (small width, 24 layers, 16 heads; no CLIP weights).
Checks: all-ones head mask via vvattn_core.install_head_mask == unmasked forward (bitwise, patch tokens and CLS); a real mask changes the V-V patch tokens
only (CLS from the original stream is bitwise unchanged); layer numbering 6..24; restore; train.py accepts the flag. Run: python test_vv_head_mask_integration.py"""
import subprocess, sys, os
import numpy as np
import torch
from AnomalyCLIP_lib.AnomalyCLIP import VisionTransformer, Attention
import vvattn_core as C
import vvattn_stats as S


def small_visual():
    torch.manual_seed(0)
    v = VisionTransformer(input_resolution=28, patch_size=14, width=64, layers=24, heads=16, output_dim=8).eval()
    v.DAPM_replace(DPAM_layer=20)
    return v


class Holder(torch.nn.Module):          # install_head_mask only needs model.visual.transformer.resblocks and parameters()
    def __init__(self, visual):
        super().__init__(); self.visual = visual


def run(v, x):
    with torch.no_grad():
        return v(x, [24], DPAM_layer=20)


def test_all_ones_equals_unmasked_and_restore():
    v = small_visual(); m = Holder(v); x = torch.randn(2, 3, 28, 28)
    orig = Attention.forward
    cls0, pt0 = run(v, x)
    restore = C.install_head_mask(m, np.ones((19, 16), bool))
    assert Attention.forward is not orig
    cls1, pt1 = run(v, x)
    assert torch.equal(cls0, cls1) and all(torch.equal(a, b) for a, b in zip(pt0, pt1))
    nums = [b.attn.layer_idx for b in v.transformer.resblocks if isinstance(b.attn, Attention)]
    assert nums == list(range(6, 25)), nums                                   # DAPM_replace(20) replaces blocks 6..24
    restore(); restore()
    assert Attention.forward is orig
    cls2, pt2 = run(v, x)
    assert torch.equal(cls0, cls2) and torch.equal(pt0[0], pt2[0])


def test_real_mask_changes_only_vv_stream():
    v = small_visual(); m = Holder(v); x = torch.randn(2, 3, 28, 28)
    cls0, pt0 = run(v, x)
    keep = S.random_mask(152, 0)
    restore = C.install_head_mask(m, keep)
    try:
        cls1, pt1 = run(v, x)
    finally:
        restore()
    assert torch.equal(cls0, cls1)                                            # CLS comes from the original Q-K stream: unchanged
    assert not torch.allclose(pt0[0], pt1[0], atol=1e-5)                      # V-V patch tokens change
    # all-zero mask on the last layer only == manual: same as a mask dict with zeros there; sanity that layer 24 alone matters
    keep2 = np.ones((19, 16), bool); keep2[18, :] = False                      # layer 24 (row 18) all heads dropped
    restore = C.install_head_mask(m, keep2)
    try:
        _, pt2 = run(v, x)
    finally:
        restore()
    assert not torch.allclose(pt0[0], pt2[0], atol=1e-5)
    keep3 = np.ones((19, 16), bool); keep3[0, 3] = False                      # layer 6 head 3 only
    restore = C.install_head_mask(m, keep3)
    try:
        _, pt3 = run(v, x)
    finally:
        restore()
    assert not torch.allclose(pt0[0], pt3[0], atol=1e-6) and not torch.allclose(pt3[0], pt1[0], atol=1e-6)


def test_cli_flags_present():
    root = os.path.dirname(os.path.abspath(__file__))
    out = subprocess.run([sys.executable, "train.py", "--help"], cwd=root, capture_output=True, text=True)
    assert "--vv_head_mask" in out.stdout, out.stderr[-500:]
    src = open(os.path.join(root, "test.py")).read()
    import ast; ast.parse(src)
    assert '"--vv_head_mask"' in src and "resolve_test_mask(args.vv_head_mask" in src and "'vvattn_core.py'" in src


def test_checkpoint_record_roundtrip():
    import tempfile
    keep = S.random_mask(152, 0)
    with tempfile.TemporaryDirectory() as d:
        np.save(os.path.join(d, "m.npy"), keep)
        k, sha = S.load_keep(os.path.join(d, "m.npy"))
        rec = S.mask_record(os.path.join(d, "m.npy"), k, sha)
        torch.save({"prompt_learner": {}, "vv_head_mask": rec}, os.path.join(d, "c.pth"))
        try:
            c = torch.load(os.path.join(d, "c.pth"), weights_only=True)      # plain python record: also loads under the strict loader
        except TypeError:
            c = torch.load(os.path.join(d, "c.pth"))
        got, r = S.resolve_test_mask("", c["vv_head_mask"])
        assert np.array_equal(got, keep) and r["sha256"] == sha


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
    print("test_vv_head_mask_integration: ALL OK")
