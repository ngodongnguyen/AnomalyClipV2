"""EXP-045 torch helpers: exact sort-based entmax-1.5, a patched V-V Attention.forward (driver-side monkey patch, restored afterwards), per-head masking
and the MVTec head-score accumulator. No model code is edited; with the default state the patched forward reproduces Attention.forward exactly."""
import atexit
import contextlib
import numpy as np
import torch

from vvattn_stats import FIRST_LAYER, N_HEADS, N_LAYERS

NB = 4096                      # histogram bins for the head score (log10 attention weight in [-12, 0])
LOG_LO = -12.0


def entmax15(logits, dim=-1):
    """Exact entmax-1.5 (Peters et al. 2019): p = max(z/2 - tau, 0)^2 with sum p = 1, tau by the sorted closed form. Returns float32 or the input float dtype."""
    z = logits if logits.dtype in (torch.float32, torch.float64) else logits.float()
    z = z.movedim(dim, -1)
    z = (z - z.max(-1, keepdim=True).values) / 2
    zs, _ = torch.sort(z, dim=-1, descending=True)
    n = z.shape[-1]
    k = torch.arange(1, n + 1, device=z.device, dtype=z.dtype)
    mean = zs.cumsum(-1) / k
    mean_sq = (zs * zs).cumsum(-1) / k
    delta = (1 - k * (mean_sq - mean * mean)) / k
    tau = mean - torch.sqrt(delta.clamp_min(0))
    support = (tau <= zs).sum(-1, keepdim=True)            # >= 1 always (k=1: tau = z_max - 1 <= z_max)
    tau_star = tau.gather(-1, support - 1)
    p = (z - tau_star).clamp_min(0) ** 2
    return p.movedim(-1, dim)


def new_state():
    """mode: softmax (baseline) | entmax | sharp (softmax on 2x logits); first: first layer (1-based block index) where mode applies;
    hmask: None or {layer: float tensor [H]} multiplied onto the per-head V-V output before proj; capture: None or fn(layer, attn[B,H,N,N])."""
    return {"mode": "softmax", "first": FIRST_LAYER, "hmask": None, "capture": None, "temp": 2.0}


def make_vv_forward(state):
    """Same ops as AnomalyCLIP_lib.AnomalyCLIP.Attention.forward (eval mode), plus the three opt-in changes of `state`. Only the V-V stream changes;
    attn_ori / x_ori (original Q-K stream, CLS path) are untouched."""
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_ori = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        attn_ori = self.attn_drop(attn_ori)
        logits = (v @ v.transpose(-2, -1)) * self.scale
        layer = getattr(self, "layer_idx", None)
        mode = state["mode"] if (layer is not None and layer >= state["first"]) else "softmax"
        if mode == "entmax":
            attn = entmax15(logits.float(), dim=-1).to(logits.dtype)
        elif mode == "sharp":
            attn = (logits * state["temp"]).softmax(dim=-1)
        else:
            attn = logits.softmax(dim=-1)
        if state["capture"] is not None and mode == "softmax":
            state["capture"](layer, attn)
        attn = self.attn_drop(attn)
        x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
        o = attn @ v                                                         # [B, H, N, hd]
        hm = state["hmask"]
        if hm is not None and layer in hm:
            o = o * hm[layer].to(device=o.device, dtype=o.dtype).view(1, -1, 1, 1)   # zero whole heads before self.proj
        x = o.transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        x_ori = self.proj_drop(self.proj(x_ori))
        return [x, x_ori]
    return forward


@contextlib.contextmanager
def patched_attention(model, state):
    """Install the patched forward on Attention (class level), number the V-V blocks 1..24 (layer_idx = block index + 1), restore on exit."""
    from AnomalyCLIP_lib.AnomalyCLIP import Attention
    orig = Attention.forward
    for l, blk in enumerate(model.visual.transformer.resblocks):
        if isinstance(blk.attn, Attention):
            blk.attn.layer_idx = l + 1
    Attention.forward = make_vv_forward(state)
    try:
        yield
    finally:
        Attention.forward = orig


def head_mask_dict(keep, device):
    """keep: bool np [19,16] (True = head kept) -> {layer: float tensor [16]}"""
    return {FIRST_LAYER + i: torch.from_numpy(keep[i].astype(np.float32)).to(device) for i in range(N_LAYERS)}


def set_arm(state, kind, keep=None, device="cpu"):
    """Reset `state` to one arm. kind in base | ent_all | ent_last6 | sharp_all | head (needs keep [19,16])."""
    state.update(mode="softmax", first=FIRST_LAYER, hmask=None)
    if kind == "ent_all":
        state.update(mode="entmax")
    elif kind == "ent_last6":
        state.update(mode="entmax", first=19)
    elif kind == "sharp_all":
        state.update(mode="sharp")
    elif kind == "head":
        state["hmask"] = head_mask_dict(keep, device)
    elif kind != "base":
        raise ValueError(kind)


class HeadScoreAccumulator:
    """Pooled histograms of the V-V attention weight for (lesion query -> lesion key) [pos] versus (lesion query -> normal key) [neg], per (layer, head).
    Weights are binned in log10 space (NB bins over [-12, 0]) so a pooled AUROC over hundreds of millions of pairs needs only counts."""
    def __init__(self, device="cpu"):
        self.pos = torch.zeros(N_LAYERS, N_HEADS, NB, dtype=torch.int64, device=device)
        self.neg = torch.zeros_like(self.pos)

    def set_image(self, key_lesion, key_normal, q_idx):
        """key_lesion/key_normal: bool [G*G] patch labels (ambiguous in neither); q_idx: LongTensor of lesion query patches (0-based patch index)."""
        self.kl, self.kn, self.q = key_lesion, key_normal, q_idx

    def __call__(self, layer, attn):
        if layer is None or layer < FIRST_LAYER:
            return
        A = attn[0].float()[:, 1 + self.q][:, :, 1:]                         # [H, Lq, G*G] patch->patch weights
        n_key = A.shape[-1]
        selfm = self.q[:, None] == torch.arange(n_key, device=A.device)[None, :]
        pos_m = (self.kl[None, :] & ~selfm)[None].expand_as(A)
        neg_m = self.kn[None, None, :].expand_as(A)
        b = ((torch.log10(A.clamp_min(1e-12)) - LOG_LO) / (-LOG_LO) * NB).long().clamp_(0, NB - 1)
        off = (torch.arange(N_HEADS, device=A.device) * NB).view(-1, 1, 1)
        for hist, m in ((self.pos, pos_m), (self.neg, neg_m)):
            c = torch.bincount((b + off)[m], minlength=N_HEADS * NB).view(N_HEADS, NB)
            hist[layer - FIRST_LAYER] += c

    def numpy(self):
        return self.pos.cpu().numpy(), self.neg.cpu().numpy()


def install_head_mask(model, keep, device=None):
    """EXP-046: install the patched V-V forward with the head mask `keep` (bool [19,16]) for the rest of the process (train.py / test.py).
    Must be called after model.visual.DAPM_replace and BEFORE the first encode_image. Class-level patch, restored at interpreter exit
    (or earlier by calling the returned function; idempotent). Only the V-V stream changes, the original Q-K / CLS stream is untouched."""
    from vvattn_stats import validate_keep
    keep = validate_keep(keep)
    if device is None:
        device = next(model.parameters()).device
    state = new_state()
    state["hmask"] = head_mask_dict(keep, device)
    cm = patched_attention(model, state)
    cm.__enter__()
    done = []

    def restore():
        if not done:
            done.append(1)
            cm.__exit__(None, None, None)
    atexit.register(restore)
    return restore
