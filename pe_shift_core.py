"""EXP-048 / H037: run the frozen ViT with a modified PATCH positional embedding (CLS row untouched), restore afterwards.
Layout (AnomalyCLIP_lib VisionTransformer.forward): positional_embedding rows = [CLS, patch (0,0), (0,1), ..., (side-1, side-1)] (row-major).
The 24x24 -> 37x37 interpolation happens inside the first forward and is stored in .data, so call ensure_grid() after a warm-up forward."""
from contextlib import contextmanager
import numpy as np
import torch


def grid_side(pe):
    n = pe.shape[0] - 1
    side = int(round(n ** 0.5))
    if side * side != n:
        raise ValueError(f"positional embedding has {pe.shape[0]} rows, not 1 + side^2")
    return side


def ensure_grid(visual, side):
    """STOP if the first-call interpolation has not happened yet (PE still 24x24)."""
    got = grid_side(visual.positional_embedding.data)
    if got != side:
        raise RuntimeError(f"positional embedding is {got}x{got}, expected {side}x{side}: run one forward at the target size first")


def shift_pe(pe, sy, sx):
    """pe: (1 + side^2, W). Token (i, j) gets the PE of cell (clip(i + sy), clip(j + sx)); CLS row unchanged.
    Returns (new_pe, valid) with valid[i, j] True iff 0 <= i + sy < side and 0 <= j + sx < side (unclamped source inside the grid)."""
    side = grid_side(pe)
    ii = torch.arange(side)
    si, sj = ii + int(sy), ii + int(sx)
    ci, cj = si.clamp(0, side - 1), sj.clamp(0, side - 1)
    src = (ci[:, None] * side + cj[None, :]).reshape(-1) + 1
    new = pe.clone()
    new[1:] = pe[src.to(pe.device)]
    valid = ((si >= 0) & (si < side))[:, None] & ((sj >= 0) & (sj < side))[None, :]
    return new, valid.numpy()


def flat_pe(pe):
    """Every patch row replaced by the mean of the patch rows; CLS row unchanged."""
    new = pe.clone()
    new[1:] = pe[1:].mean(0, keepdim=True)
    return new


@contextmanager
def patched_pe(visual, mode="shift", sy=0, sx=0):
    """Temporarily replace visual.positional_embedding.data. Yields the (side, side) bool validity grid (all True except for mode 'shift' clamping).
    The original tensor object is put back in a finally block, so an exception inside the forward cannot leave a modified PE."""
    orig = visual.positional_embedding.data
    side = grid_side(orig)
    if mode == "shift":
        new, valid = shift_pe(orig, sy, sx)
    elif mode == "flat":
        new, valid = flat_pe(orig), np.ones((side, side), bool)
    else:
        raise ValueError(mode)
    visual.positional_embedding.data = new
    try:
        yield valid
    finally:
        visual.positional_embedding.data = orig
