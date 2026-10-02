"""
Does the GLOBAL scope of the V-V attention (DPAM) limit dense-feature quality on large smooth lesions?
Visual-side diagnostic, forward passes only, no training, labels used only to SCORE.

Arms (text features are computed ONCE per image from the baseline forward and reused, so only the visual path changes):
  base          unmodified V-V attention
  loc{b}_all    Gaussian locality prior on the V-V attention logits, -d^2/(2 b^2) (d in patch units), all 19 V-V layers
  loc{b}_last6  same, only layers 19..24
  fsm{s}        CONTROL: no attention change; Gaussian-smooth the final patch-token grid (sigma s patches) before readout
Every arm is scored at post-hoc map smoothing sigma = 4 (protocol) and sigma = 32 (the strong baseline: it is
+2..+6 per-image AUROC here), per-image pixel AUROC at 518x518, plus by lesion-size quartile.
The question is NOT "does locality help at sigma 4" (it would, by smoothing) but "does it add anything on top of sigma 32,
and anything that plain feature-grid smoothing does not already give".

Run (from the repo root; copy locality_stats.py next to this file):
  python analyze_attn_locality.py --dataset colon --data_path $CVC/CVC-ClinicDB --checkpoint_path checkpoints/ecp_extent/epoch_15.pth \
         --ec_z_pix 1.6 --out_csv attnloc_ClinicDB.csv
then
  python analyze_attn_locality.py --aggregate attnloc_ClinicDB.csv attnloc_Kvasir.csv attnloc_ColonDB.csv
"""
import os
import csv
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter

from locality_stats import auroc, gauss_bias, smooth_tokens, quartile_ids, verdict

ARMS = {  # name -> (kind, param, first_layer)
    "base": ("none", None, None),
    "loc8_all": ("loc", 8.0, 6), "loc4_all": ("loc", 4.0, 6), "loc2_all": ("loc", 2.0, 6),
    "loc4_last6": ("loc", 4.0, 19), "loc2_last6": ("loc", 2.0, 19),
    "fsm1": ("fsm", 1.0, None), "fsm2": ("fsm", 2.0, None),
}
LOC_ARMS = [a for a, v in ARMS.items() if v[0] == "loc"]
FSM_ARMS = [a for a, v in ARMS.items() if v[0] == "fsm"]
SIGMAS = (4, 32)


def run(args):
    import torch
    import AnomalyCLIP_lib
    from AnomalyCLIP_lib.AnomalyCLIP import Attention
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from dataset import Dataset
    from utils import get_transform
    from extent_prompt import ExtentConditioner, visual_descriptor, conditioned_text_features

    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth,
              "learnabel_text_embedding_length": args.t_n_ctx}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params)
    model.eval()
    preprocess, target_transform = get_transform(args)
    data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    idx = np.random.RandomState(0).permutation(len(data))[: args.limit] if args.limit else range(len(data))

    pl = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    pl.load_state_dict(ckpt["prompt_learner"])
    pl.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)
    cond = ExtentConditioner(mode=ckpt["extent_cond"]).to(device)
    cond.load_state_dict(ckpt["conditioner"])
    cond.eval()

    for l, blk in enumerate(model.visual.transformer.resblocks):
        if isinstance(blk.attn, Attention):
            blk.attn.layer_idx = l + 1

    state = {"b": None, "first": None, "cache": {}}

    def vv_forward(self, x):  # identical to Attention.forward except for the optional additive locality bias on the V-V logits
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_ori = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        logits = (v @ v.transpose(-2, -1)) * self.scale
        if state["b"] is not None and self.layer_idx >= state["first"]:
            G = int(round((N - 1) ** 0.5))
            key = (G, state["b"])
            if key not in state["cache"]:
                state["cache"][key] = torch.from_numpy(gauss_bias(G, state["b"])).to(x.device)
            logits = logits + state["cache"][key].to(logits.dtype)
        attn = logits.softmax(dim=-1)
        x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return [self.proj_drop(self.proj(x)), self.proj_drop(self.proj(x_ori))]

    prompts, tokenized, compound = pl(cls_id=None)

    def encode(img, arm):
        kind, p, first = ARMS[arm]
        state["b"], state["first"] = (p, first) if kind == "loc" else (None, None)
        with torch.no_grad():
            imf, pf = model.encode_image(img, [24], DPAM_layer=20)
        return imf, pf

    def readout(pf, tf, arm):
        kind, p, _ = ARMS[arm]
        feats = pf[-1]  # [1, N, C]
        if kind == "fsm":
            tok = feats[0, 1:].float().cpu().numpy()
            G = int(round(tok.shape[0] ** 0.5))
            sm = torch.from_numpy(smooth_tokens(tok, G, p)).to(feats.device).to(feats.dtype)
            feats = torch.cat([feats[:, :1], sm[None]], dim=1)
        f = feats / feats.norm(dim=-1, keepdim=True)
        sim, _ = AnomalyCLIP_lib.compute_similarity(f, tf[0])
        sm_ = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], args.image_size)
        return ((sm_[..., 1] + 1 - sm_[..., 0]) / 2.0)[0].float().cpu().numpy()

    # sanity: patched forward with no bias must reproduce the original forward on one image
    first = data[int(idx[0])]
    img0 = first["img"].unsqueeze(0).to(device)
    _, pf_patched_off = encode(img0, "base")
    orig_forward = Attention.forward
    Attention.forward = vv_forward
    _, pf_new = encode(img0, "base")
    diff = (pf_new[-1].float() - pf_patched_off[-1].float()).abs().max().item()
    print(f"sanity: max|patched(no bias) - original| = {diff:.2e} (must be < 1e-2)")
    assert diff < 1e-2

    rows = []
    for i in idx:
        it = data[int(i)]
        gt = it["img_mask"][0].numpy() > 0.5 if it["img_mask"].ndim == 3 else it["img_mask"].numpy() > 0.5
        if gt.sum() < 20 or (~gt).sum() < 20:
            continue
        img = it["img"].unsqueeze(0).to(device)
        imf, pf = encode(img, "base")
        imf = imf / imf.norm(dim=-1, keepdim=True)
        z = torch.full((1,), args.ec_z_pix, device=device)
        _, c_pos, c_neg = cond(visual_descriptor(imf, pf), z_override=z)
        tf = conditioned_text_features(model, pl, c_pos, c_neg)
        row = {"image": os.path.basename(it["img_path"]), "log_area": float(np.log(gt.mean()))}
        for arm in ARMS:
            pf_a = pf if arm == "base" else encode(img, arm)[1]
            m = readout(pf_a, tf, arm)
            for s in SIGMAS:
                row[f"{arm}@{s}"] = auroc(gt, gaussian_filter(m, sigma=s))
        rows.append(row)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_csv} ({len(rows)} images)")
    report(args.out_csv)


def load(path):
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) if k != "image" else 0.0 for r in rows]) for k in rows[0] if k != "image"}


def report(path):
    d = load(path)
    q = quartile_ids(d["log_area"])
    print(f"\n=== {os.path.basename(path)} (n={len(q)}); per-image AUROC, delta vs base at the SAME post-sigma ===")
    print(f"{'arm':12s} " + " ".join(f"{'AUC@' + str(s):>8s} {'d@' + str(s):>8s} {'95%CI':>7s}" for s in SIGMAS) + "   dQ0..Q3 @32")
    for arm in ARMS:
        parts = []
        for s in SIGMAS:
            dd = d[f"{arm}@{s}"] - d[f"base@{s}"]
            parts.append(f"{d[f'{arm}@{s}'].mean():8.4f} {dd.mean():+8.4f} {1.96 * dd.std() / np.sqrt(len(dd)):7.4f}")
        qd = [float((d[f"{arm}@32"] - d["base@32"])[q == k].mean()) for k in range(4)]
        print(f"{arm:12s} " + " ".join(parts) + "   " + " ".join(f"{x:+.3f}" for x in qd))


def aggregate(paths):
    names = [os.path.basename(p).replace("attnloc_", "").replace(".csv", "") for p in paths]
    delta32, delta4_q0, fsm = {a: {} for a in LOC_ARMS}, {a: {} for a in LOC_ARMS}, {}
    for nm, p in zip(names, paths):
        d = load(p)
        q = quartile_ids(d["log_area"])
        for a in LOC_ARMS:
            delta32[a][nm] = float((d[f"{a}@32"] - d["base@32"]).mean())
            delta4_q0[a][nm] = float((d[f"{a}@4"] - d["base@4"])[q == 0].mean())
        fsm[nm] = max(float((d[f"{a}@32"] - d["base@32"]).mean()) for a in FSM_ARMS)
    print("Delta32 per locality arm:", {a: {k: round(v, 4) for k, v in delta32[a].items()} for a in LOC_ARMS})
    print("Delta4(Q0) per locality arm:", {a: {k: round(v, 4) for k, v in delta4_q0[a].items()} for a in LOC_ARMS})
    print("best feature-smoothing control Delta32:", {k: round(v, 4) for k, v in fsm.items()})
    v, passed = verdict(delta32, delta4_q0, fsm, LOC_ARMS)
    print(f"\nVERDICT: {v}   (arms passing: {passed})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", nargs="+", default=None)
    ap.add_argument("--data_path")
    ap.add_argument("--checkpoint_path")
    ap.add_argument("--dataset")
    ap.add_argument("--ec_z_pix", type=float, default=1.6)
    ap.add_argument("--out_csv", default="attnloc.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    a = ap.parse_args()
    aggregate(a.aggregate) if a.aggregate else run(a)
