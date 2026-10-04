"""Deterministic region selection and image interventions for EXP-015."""

import hashlib
import numpy as np


GRID = 37
PATCH = 14
SIZE = 518
SEED = 111
PROTECTED_RADIUS = 3  # 7 x 7 native patches; required minimum was 5 x 5.
PHOTO_GAMMAS = (0.9, 1.1)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def stable_key(value):
    return hashlib.sha256(("EXP-015:111:" + str(value)).encode()).hexdigest()


def select_centers(margins, high_count=3, middle_count=3):
    """Select separated high and middle centers using model scores, never masks."""
    grid = np.asarray(margins, dtype=np.float64).reshape(GRID, GRID)
    if not np.isfinite(grid).all():
        raise ValueError("nonfinite native margins")
    allowed = [(r, c) for r in range(4, GRID - 4)
               for c in range(4, GRID - 4)]
    ranked = sorted(allowed, key=lambda rc: (-grid[rc], rc[0], rc[1]))
    high_pool = ranked[:max(high_count, int(np.ceil(0.05 * len(ranked))))]
    mid_start = int(0.40 * len(ranked))
    mid_end = int(0.60 * len(ranked))
    middle_pool = ranked[mid_start:mid_end]
    # The middle-band order is fixed before looking at GT or intervention effects.
    middle_pool = sorted(middle_pool,
                         key=lambda rc: (abs(ranked.index(rc) - len(ranked) / 2),
                                         rc[0], rc[1]))
    chosen = []
    def add(pool, label, count):
        added = 0
        for r, c in pool:
            if all(max(abs(r - a), abs(c - b)) >= 7 for a, b, _ in chosen):
                chosen.append((r, c, label))
                added += 1
                if added == count:
                    break
    add(high_pool, "high", high_count)
    add(middle_pool, "middle", middle_count)
    return chosen


def center_label(mask, row, col, far_distance):
    """Use GT only after center/view choice; classify the queried native patch."""
    y0, x0 = row * PATCH, col * PATCH
    sub = np.asarray(mask, dtype=bool)[y0:y0 + PATCH, x0:x0 + PATCH]
    if sub.shape != (PATCH, PATCH):
        raise ValueError("invalid queried patch")
    fraction = float(sub.mean())
    if fraction >= 0.9:
        return "TP", fraction
    far = np.asarray(far_distance)[y0:y0 + PATCH, x0:x0 + PATCH]
    if not sub.any() and float(far.min()) > 28:
        return "far_FP", fraction
    return "other", fraction


def paired_auc(fp_values, tp_values):
    """Exact small-sample FP-above-TP AUROC with half credit for ties."""
    if not fp_values or not tp_values:
        return None
    f = np.asarray(fp_values, dtype=float)[:, None]
    t = np.asarray(tp_values, dtype=float)[None, :]
    return float(((f > t) + 0.5 * (f == t)).mean())


def intervention(image, row, col, kind, sample_id):
    """Edit only beyond a 7 x 7 patch square; input is normalized CLIP RGB."""
    import torch
    if tuple(image.shape) != (1, 3, SIZE, SIZE):
        raise ValueError("expected normalized [1,3,518,518] input")
    if not (4 <= row < GRID - 4 and 4 <= col < GRID - 4):
        raise ValueError("center lacks the full protected/feather region")
    if kind not in ("sham", "photo", "rearrange", "near_photo"):
        raise ValueError(kind)
    yy = torch.arange(SIZE, device=image.device)
    xx = torch.arange(SIZE, device=image.device)
    y0, y1 = (row - PROTECTED_RADIUS) * PATCH, (row + PROTECTED_RADIUS + 1) * PATCH
    x0, x1 = (col - PROTECTED_RADIUS) * PATCH, (col + PROTECTED_RADIUS + 1) * PATCH
    dy = torch.maximum(y0 - yy, yy - (y1 - 1)).clamp_min(0)
    dx = torch.maximum(x0 - xx, xx - (x1 - 1)).clamp_min(0)
    distance = torch.maximum(dy[:, None], dx[None, :]).float() / PATCH
    t = distance.clamp(0, 1)
    alpha = (t * t * (3 - 2 * t)).to(image.dtype)
    if kind == "near_photo":
        # One to three patches outside the protected square, then fade to zero.
        outer = (1 - ((distance - 1).clamp(0, 3) / 3))
        alpha = alpha * outer.to(image.dtype)
    alpha = alpha.view(1, 1, SIZE, SIZE)
    if kind == "sham":
        candidate = image
    elif kind == "rearrange":
        candidate = torch.flip(image, dims=(-1,))
    else:
        gamma = PHOTO_GAMMAS[int(stable_key((sample_id, row, col))[-1], 16) % 2]
        mean = image.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = image.new_tensor(CLIP_STD).view(1, 3, 1, 1)
        rgb = (image.float() * std.float() + mean.float()).clamp(0, 1)
        candidate = ((rgb.pow(gamma) - mean.float()) / std.float()).to(image.dtype)
    edited = torch.where(alpha == 0, image, image + alpha * (candidate - image))
    protected = edited[..., y0:y1, x0:x1]
    if not torch.equal(protected, image[..., y0:y1, x0:x1]):
        raise RuntimeError("protected pixels changed")
    if kind == "sham" and not torch.equal(edited, image):
        raise RuntimeError("sham differs from identity")
    return edited
