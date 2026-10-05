"""Patch Average Aggregation (PAA), the parameter-free multi-scale neighbourhood average used by MoECLIP
(arXiv 2603.03101, Sec. 4.5): each patch feature is replaced by the mean of the s x s window of patch features centred on it
(s odd, stride 1). s = 1 is the identity. Borders use the truncated window (mean over the in-grid patches only).
The class token (index 0) is never touched. The numpy reference is the unit-tested ground truth; the torch version is
checked against it in `selftest()` (run it on the server)."""
import numpy as np


def paa_np(tokens, s):
    """tokens: [N+1, C] (CLS first), N a perfect square. Returns the same shape."""
    if s == 1:
        return tokens.copy()
    assert s % 2 == 1 and s > 0, "window must be odd"
    cls, patches = tokens[:1], tokens[1:]
    g = int(round(patches.shape[0] ** 0.5)); assert g * g == patches.shape[0]
    x = patches.reshape(g, g, -1)
    r = s // 2
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(g):
        for j in range(g):
            win = x[max(0, i - r): i + r + 1, max(0, j - r): j + r + 1]
            out[i, j] = win.reshape(-1, x.shape[-1]).mean(0)
    return np.concatenate([cls, out.reshape(g * g, -1).astype(tokens.dtype)], 0)


def paa_torch(tokens, s):
    """tokens: [B, N+1, C] (CLS first). Same semantics as paa_np; s == 1 returns the input tensor unchanged."""
    if s == 1:
        return tokens
    import torch.nn.functional as F
    assert s % 2 == 1 and s > 0, "window must be odd"
    cls, patches = tokens[:, :1], tokens[:, 1:]
    b, n, c = patches.shape
    g = int(round(n ** 0.5)); assert g * g == n
    x = patches.transpose(1, 2).reshape(b, c, g, g)
    x = F.avg_pool2d(x, kernel_size=s, stride=1, padding=s // 2, count_include_pad=False)
    import torch
    return torch.cat([cls, x.reshape(b, c, n).transpose(1, 2)], dim=1)


def selftest(device="cpu"):
    import torch
    g = torch.Generator().manual_seed(0)
    t = torch.randn(2, 1 + 7 * 7, 5, generator=g)
    for s in (1, 3, 5):
        got = paa_torch(t.to(device), s).cpu().numpy()
        ref = np.stack([paa_np(t[b].numpy(), s) for b in range(t.shape[0])])
        assert np.allclose(got, ref, atol=1e-5), f"paa torch/numpy mismatch at s={s}"
    print("paa selftest OK")
