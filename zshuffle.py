"""EXP-028 control: break the link between an image and its extent label while keeping the label distribution.

derangement_among(mask, rng) returns an index vector `src` (len = len(mask)) such that z_shuffled = z[src]:
the positions where mask is True are cyclically rotated among themselves (every selected position receives the
value of a DIFFERENT selected position, so the marginal distribution of z is unchanged but the pairing with the
image is destroyed); other positions keep their own value. With fewer than 2 selected positions nothing can be
shuffled and the identity is returned.
"""
import numpy as np


def derangement_among(mask, rng):
    mask = np.asarray(mask, dtype=bool)
    src = np.arange(mask.shape[0])
    idx = np.flatnonzero(mask)
    if idx.size >= 2:
        p = rng.permutation(idx)
        src[p] = np.roll(p, 1)
    return src
