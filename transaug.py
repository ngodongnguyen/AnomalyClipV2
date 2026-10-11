"""EXP-037: label-preserving translation augmentation with content-free (CLIP-mean colour) fill. Pure PIL/numpy; no torch.
Anomalous images: the shift keeps the lesion bounding box inside the canvas (so the lesion area, hence the extent label z, is unchanged).
Normal images: shifts uniform in +-NORMAL_FRAC of the side, so that filled borders are not a shortcut for 'anomalous'."""
import numpy as np
from PIL import Image

CLIP_MEAN_RGB = (123, 117, 104)      # OPENAI_DATASET_MEAN (0.4815, 0.4578, 0.4082) * 255, rounded: 0 after the CLIP normalisation (up to rounding)
NORMAL_FRAC = 0.30


def lesion_bbox(mask_np):
    ys, xs = np.nonzero(mask_np)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def sample_shift_anomalous(mask_np, rng):
    """Uniform integer (dx, dy) such that the lesion bounding box stays inside the canvas."""
    h, w = mask_np.shape
    y0, y1, x0, x1 = lesion_bbox(mask_np)
    dx = rng.randint(-x0, w - x1)
    dy = rng.randint(-y0, h - y1)
    return int(dx), int(dy)


def sample_shift_normal(w, h, rng, frac=NORMAL_FRAC):
    mx, my = int(round(frac * w)), int(round(frac * h))
    return rng.randint(-mx, mx), rng.randint(-my, my)


def translate_fill(img, dx, dy, fill):
    """Return a new PIL image of the same size with the content translated by (dx, dy) and the uncovered area filled with `fill`."""
    out = Image.new(img.mode, img.size, fill)
    out.paste(img, (int(dx), int(dy)))               # PIL clips the part that leaves the canvas
    return out


def translate_pair(img, mask, rng, anomalous):
    """img: PIL RGB(A)/L image, mask: PIL 'L' mask of the same size (all zero for normal images). Returns (img', mask', (dx, dy))."""
    w, h = img.size
    if anomalous:
        m = np.asarray(mask) > 127
        if not m.any():
            dx, dy = sample_shift_normal(w, h, rng)
        else:
            dx, dy = sample_shift_anomalous(m, rng)
    else:
        dx, dy = sample_shift_normal(w, h, rng)
    fill = CLIP_MEAN_RGB if img.mode == "RGB" else tuple(CLIP_MEAN_RGB[:len(img.getbands())])
    return translate_fill(img, dx, dy, fill), translate_fill(mask, dx, dy, 0), (dx, dy)
