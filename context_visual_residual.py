"""Frozen-backbone visual residual and source-only context-pair construction.

The residual consumes the projected visual tokens returned by ``encode_image``.
The context intervention keeps a 5 x 5 native-patch neighborhood unchanged;
only its central 3 x 3 patches are valid for paired supervision.
"""

import torch
from torch import nn
import torch.nn.functional as F


_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ContextVisualResidual(nn.Module):
    """Zero-initialized, 49,152-parameter residual on projected patch tokens.

    Input and output are [B, 1 + grid**2, dim]. The CLS token is copied exactly.
    The caller owns feature normalization and the unchanged text/map readout.
    """

    def __init__(self, dim=768, rank=32):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.down = nn.Linear(dim, rank, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1] < 2 or tokens.shape[2] != self.down.in_features:
            raise ValueError("tokens must have shape [B, 1 + patches, dim]")
        patches = tokens[:, 1:, :]
        residual = self.up(self.act(self.down(self.norm(patches))))
        return torch.cat((tokens[:, :1, :], patches + residual), dim=1)


@torch.no_grad()
def make_context_pair(images, masks, generator):
    """Alter distant context while preserving a native 5 x 5-patch neighborhood.

    ``images`` are CLIP-normalized [B, 3, 518, 518] tensors and ``masks`` are
    [B, 1, 518, 518] source masks. ``generator`` must be a CPU torch.Generator;
    all random choices use it, so replaying its seed reproduces the pair.

    Each sample chooses an interior native-patch center. With probability 1/2,
    a positive patch is selected if one exists; otherwise selection is uniform.
    The candidate outside the protected region is a gamma-adjusted image
    (gamma 0.85 or 1.15) or the horizontally flipped whole-image content.
    A smoothstep blend ramps from zero outside the protected 5 x 5 square to
    full strength by the outside edge of its enclosing 7 x 7 square. The
    protected pixels are copied with torch.where, guaranteeing exact equality.

    Returns edited images, a Boolean [B,1,H,W] central-3x3 pixel mask, and a
    Boolean [B,37**2] central-3x3 token mask *excluding* CLS. The edited view's
    labels outside this central core are deliberately unspecified: apply any
    supervised/paired loss on the edited view only where these masks are true.
    """
    if images.ndim != 4 or images.shape[1:] != (3, 518, 518):
        raise ValueError("images must have shape [B,3,518,518]")
    if masks.shape != (images.shape[0], 1, 518, 518):
        raise ValueError("masks must have shape [B,1,518,518]")
    if generator is None or generator.device.type != "cpu":
        raise ValueError("generator must be a CPU torch.Generator")
    if not torch.isfinite(images).all() or not torch.isfinite(masks).all():
        raise ValueError("images and masks must be finite")

    batch, _, height, width = images.shape
    patch = 14
    grid = height // patch
    edited = images.clone()
    core_pixels = torch.zeros((batch, 1, height, width), dtype=torch.bool, device=images.device)
    core_tokens = torch.zeros((batch, grid * grid), dtype=torch.bool, device=images.device)
    mean = images.new_tensor(_CLIP_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(_CLIP_STD).view(1, 3, 1, 1)
    ys = torch.arange(height, device=images.device)
    xs = torch.arange(width, device=images.device)

    for b in range(batch):
        # Three native patches of margin are needed for the 7 x 7 feather box.
        positive_grid = F.max_pool2d(masks[b:b + 1].float().cpu(), patch).flatten() > 0.5
        positive_grid = positive_grid.reshape(grid, grid)
        interior = torch.zeros((grid, grid), dtype=torch.bool)
        interior[3:grid - 3, 3:grid - 3] = True
        use_positive = bool(torch.randint(2, (1,), generator=generator).item())
        candidates = (positive_grid & interior) if use_positive else interior
        if not candidates.any():
            candidates = interior
        coords = candidates.nonzero(as_tuple=False)
        chosen = int(torch.randint(len(coords), (1,), generator=generator).item())
        row, col = map(int, coords[chosen].tolist())

        core_pixels[b, :, (row - 1) * patch:(row + 2) * patch,
                    (col - 1) * patch:(col + 2) * patch] = True
        for rr in range(row - 1, row + 2):
            core_tokens[b, rr * grid + col - 1:rr * grid + col + 2] = True

        y0, y1 = (row - 2) * patch, (row + 3) * patch
        x0, x1 = (col - 2) * patch, (col + 3) * patch
        dy = torch.maximum(y0 - ys, ys - (y1 - 1)).clamp_min(0)
        dx = torch.maximum(x0 - xs, xs - (x1 - 1)).clamp_min(0)
        distance = torch.maximum(dy[:, None], dx[None, :]).to(torch.float32) / patch
        t = distance.clamp(0, 1)
        alpha = (t * t * (3 - 2 * t)).to(images.dtype).view(1, 1, height, width)

        if bool(torch.randint(2, (1,), generator=generator).item()):
            candidate = torch.flip(images[b:b + 1], dims=(-1,))
        else:
            gamma = (0.85, 1.15)[int(torch.randint(2, (1,), generator=generator).item())]
            rgb = (images[b:b + 1].float() * std.float() + mean.float()).clamp(0, 1)
            candidate = ((rgb.pow(gamma) - mean.float()) / std.float()).to(images.dtype)
        blended = images[b:b + 1] + alpha * (candidate - images[b:b + 1])
        edited[b:b + 1] = torch.where(alpha == 0, images[b:b + 1], blended)

    return edited, core_pixels, core_tokens
