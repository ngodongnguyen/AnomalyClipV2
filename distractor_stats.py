"""Pure-numpy core statistics for analyze_distractor_semantics.py (unit-testable without torch)."""
import numpy as np
from scipy.stats import rankdata


def auroc(y, s):
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def concept_margin(patch_feat, text_feat, is_distractor, temp=100.0):
    """patch_feat [N,C], text_feat [K,C] (both L2-normalised), is_distractor [K] bool.
    Softmax over ALL K concepts per patch; margin = log(mass on distractors) - log(mass on lesion concepts)."""
    logits = temp * patch_feat @ text_feat.T
    logits = logits - logits.max(1, keepdims=True)
    p = np.exp(logits); p /= p.sum(1, keepdims=True)
    d = p[:, is_distractor].sum(1); l = p[:, ~is_distractor].sum(1)
    return np.log(d + 1e-9) - np.log(l + 1e-9)


def split_control(K, is_distractor, rng):
    """Random re-labelling of the concept set with the same group sizes (null control)."""
    perm = rng.permutation(K)
    out = np.zeros(K, bool); out[perm[: int(is_distractor.sum())]] = True
    return out


def concept_probs(patch_feat, text_feat, temp=100.0):
    """Per-patch softmax over all K concepts -> [N,K] (same softmax as concept_margin)."""
    logits = temp * patch_feat @ text_feat.T
    logits = logits - logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)
