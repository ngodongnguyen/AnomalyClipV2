"""
Extent-conditioned prompt learning (ECP).

The abnormal/normal text prompts of AnomalyCLIP are shared by all images although the extent of the anomaly
(fraction of the image it covers) varies from ~1% (industrial defects) to 5-45% (polyps). ECP estimates the extent
of each image from frozen visual features, supervises that estimate with the ground-truth lesion area, and shifts
the learnable context tokens of the normal / abnormal prompts by a vector generated from the estimated extent.

Modes (the last two are controls for the ablation "is it the extent signal that matters?"):
  extent : extent estimator -> log-area z -> prompt shift          (the method)
  global : prompt shift generated directly from the visual feature (Crane-style image conditioning, no extent supervision)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

AREA_MIN = 0.005
Z_REF = math.log(0.02)
Z_SCALE = 2.0
N_BINS = 16


def area_to_z(area):
    """Lesion area fraction -> roughly unit-scale log-extent (0.5% -> -0.69, 2% -> 0, 15% -> 1.0, 45% -> 1.55)."""
    return (torch.log(area.clamp(min=AREA_MIN, max=1.0)) - Z_REF) / Z_SCALE


def structure_hist(patch_feature):
    """
    Scale-free description of how the tokens of one image split into groups: histogram (N_BINS) of the cosine
    between every patch token and the image's mean token. A large lesion makes a large minority cluster far from
    the mean token; a small one does not. patch_feature: [B, N+1, C] (CLS first). Returns [B, N_BINS].
    """
    p = F.normalize(patch_feature[:, 1:, :].float(), dim=-1)
    m = F.normalize(p.mean(1, keepdim=True), dim=-1)
    cos = (p * m).sum(-1).clamp(0, 1 - 1e-6)
    idx = (cos * N_BINS).long()
    return F.one_hot(idx, N_BINS).float().mean(1)


def visual_descriptor(image_features, patch_features):
    """[B, 768 + N_BINS] from the CLS embedding and the token-structure histogram of the last feature layer."""
    return torch.cat([image_features.float(), structure_hist(patch_features[-1])], dim=-1)


def fourier(z, n_freq=4):
    feats = [z.unsqueeze(-1)]
    for k in range(n_freq):
        feats += [torch.sin((2 ** k) * math.pi * z).unsqueeze(-1), torch.cos((2 ** k) * math.pi * z).unsqueeze(-1)]
    return torch.cat(feats, dim=-1)


class ExtentConditioner(nn.Module):
    def __init__(self, mode="extent", in_dim=768 + N_BINS, ctx_dim=768, hidden=256, n_freq=4):
        super().__init__()
        assert mode in ("extent", "global")
        self.mode, self.n_freq, self.ctx_dim = mode, n_freq, ctx_dim
        if mode == "extent":
            self.estimator = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
            cond_in = 1 + 2 * n_freq
        else:
            cond_in = in_dim
        self.cond = nn.Sequential(nn.Linear(cond_in, hidden), nn.GELU(), nn.Linear(hidden, 2 * ctx_dim))
        nn.init.zeros_(self.cond[-1].weight)
        nn.init.zeros_(self.cond[-1].bias)

    def forward(self, desc, z_override=None):
        """desc: [B, in_dim]. Returns (z_pred [B] or None, shift_normal [B, ctx_dim], shift_anomaly [B, ctx_dim])."""
        if self.mode == "extent":
            z_pred = self.estimator(desc).squeeze(-1)
            z = z_pred if z_override is None else z_override
            c = self.cond(fourier(z, self.n_freq))
        else:
            z_pred = None
            c = self.cond(desc)
        c_pos, c_neg = c.chunk(2, dim=-1)
        return z_pred, c_pos, c_neg


def conditioned_text_features(model, pl, c_pos, c_neg):
    """
    Per-image text features [B, 2, C] (index 0 = normal, 1 = anomalous) from the prompt learner's tokens shifted by
    c_pos / c_neg (added to every learnable context token, as in CoCoOp).
    """
    B = c_pos.shape[0]
    dt = pl.ctx_pos.dtype

    def build(prefix, ctx, suffix, c):
        ctx = ctx[0, 0].unsqueeze(0) + c.to(dt).unsqueeze(1)
        return torch.cat([prefix[0, 0].unsqueeze(0).expand(B, -1, -1), ctx, suffix[0, 0].unsqueeze(0).expand(B, -1, -1)], dim=1)

    prompts = torch.cat([build(pl.token_prefix_pos, pl.ctx_pos, pl.token_suffix_pos, c_pos),
                         build(pl.token_prefix_neg, pl.ctx_neg, pl.token_suffix_neg, c_neg)], dim=0)
    tokenized = torch.cat([pl.tokenized_prompts_pos.reshape(1, -1).expand(B, -1),
                           pl.tokenized_prompts_neg.reshape(1, -1).expand(B, -1)], dim=0)
    tf = model.encode_text_learn(prompts, tokenized, pl.compound_prompts_text).float()
    tf = torch.stack(tf.chunk(2, dim=0), dim=1)
    return tf / tf.norm(dim=-1, keepdim=True)


def batched_similarity(patch_feature, text_features):
    """Same as AnomalyCLIP_lib.compute_similarity but with a different text feature pair per image."""
    sim = torch.einsum("bnc,btc->bnt", patch_feature, text_features) / 0.07
    return sim.softmax(-1)
