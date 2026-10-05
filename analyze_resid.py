"""
EXP-019 / H010: ClearCLIP-style residual removal on the V-V stream.  Frozen model, forward passes only; labels only SCORE.

With DPAM_layer=20 the last 19 blocks (layers 6..24) are dual-path.  The x stream is x_5 (residual stream after layer 5) plus
the V-V attention output of every layer 6..24 (no FFN on this stream), and the patch tokens are ln_post(x_24) @ proj.
Arms (ONE set of ECP text features per image from the UNMODIFIED forward, z_pix = 1.6 constant; only the token readout changes):
  base         unmodified tokens
  no_res       x_24 - x_5 = sum_{l=6..24} VVattn_l
  no_res_last  VVattn_24 only
  swap_res     sum_{l=6..24} VVattn_l + x_5 of a different image (fixed-seed derangement within the dataset): control
Readout = test.py --features_list 24 path (normalise, compute_similarity, get_similarity_map, (s1+1-s0)/2), post-sigma 4 and 32.
`base@4` must equal the EXP-012 per-image AUROCs (tol 1e-4) via --reference_csv, else the run stops.

One command per dataset:
  python analyze_resid.py eval --data_path $CVC/CVC-ClinicDB --checkpoint_path checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth \
         --reference_csv <EXP-012 .../ClinicDB/ecp_extent/pixel_per_image_predictions.csv> --out_csv resid_ClinicDB.csv
Aggregate:
  python analyze_resid.py --aggregate resid_ClinicDB.csv resid_Kvasir.csv resid_ColonDB.csv
"""
import os
import csv
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter

from resid_stats import (auroc, paired_bootstrap_ci, swap_pairing, norm_ratio, quartile_means, verdict, TEST_ARMS)

ARMS = ("base", "no_res", "no_res_last", "swap_res")
SIGMAS = (4, 32)
PAIR_SEED = 20261005
N_ATTN = 19  # layers 6..24


def load_model(args):
    import torch
    import AnomalyCLIP_lib
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from extent_prompt import ExtentConditioner
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth,
              "learnabel_text_embedding_length": args.t_n_ctx}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params)
    model.eval()
    pl = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    pl.load_state_dict(ckpt["prompt_learner"])
    pl.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)
    cond = ExtentConditioner(mode=ckpt["extent_cond"]).to(device)
    cond.load_state_dict(ckpt["conditioner"])
    cond.eval()
    return device, model, pl, cond


class Capture:
    """Forward hooks (model untouched): x_5, per-layer V-V attention outputs (layers 6..24), final x-stream tokens. All [1, 1370, 1024]."""

    def __init__(self, model):
        from AnomalyCLIP_lib.AnomalyCLIP import Attention
        blocks = model.visual.transformer.resblocks
        assert len(blocks) == 24
        assert all(isinstance(blocks[i].attn, Attention) for i in range(5, 24)) and not isinstance(blocks[4].attn, Attention)
        blocks[4].register_forward_hook(self._h5)
        for i in range(5, 24):
            blocks[i].attn.register_forward_hook(self._hattn)
        blocks[23].register_forward_hook(self._hfinal)
        self.reset()

    def reset(self):
        self.x5 = self.seq = self.sum32 = self.last = self.final = None
        self.n = 0

    def _h5(self, m, inp, out):  # out is the LND tensor that block 6 later mutates in place -> clone now
        self.x5 = out.permute(1, 0, 2).clone()

    def _hattn(self, m, inp, out):  # out[0] = x_res [B, N, C], the term added by `x += x_res`
        xr = out[0].detach()
        self.seq = (self.x5 + xr) if self.seq is None else (self.seq + xr)  # model-dtype sequential accumulation (mirrors x += x_res)
        self.sum32 = xr.float() if self.sum32 is None else self.sum32 + xr.float()
        self.last = xr.clone()
        self.n += 1

    def _hfinal(self, m, inp, out):
        self.final = out[0].permute(1, 0, 2).clone()

    def take(self):
        d = dict(x5=self.x5, seq=self.seq, sum32=self.sum32, last=self.last, final=self.final, n=self.n)
        self.reset()
        return d


def stage_eval(args):
    import torch
    import AnomalyCLIP_lib
    from dataset import Dataset
    from utils import get_transform
    from extent_prompt import visual_descriptor, conditioned_text_features

    device, model, pl, cond = load_model(args)
    cap = Capture(model)
    preprocess, target_transform = get_transform(args)
    data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    partner = swap_pairing(len(data), PAIR_SEED)
    idx = np.random.RandomState(0).permutation(len(data))[: args.limit] if args.limit else range(len(data))
    ref = None
    if args.reference_csv:
        ref = {r["sample_id"]: float(r["per_image_pixel_auroc"]) for r in csv.DictReader(open(args.reference_csv))}
    else:
        print("WARNING: no --reference_csv, EXP-012 parity NOT checked")
    vis = model.visual

    def fwd(img):
        with torch.no_grad():
            imf, pf = model.encode_image(img, [24], DPAM_layer=20)
        return imf, pf, cap.take()

    def tokens_to_map(tok, tf):
        tok = tok / tok.norm(dim=-1, keepdim=True)
        sim, _ = AnomalyCLIP_lib.compute_similarity(tok, tf[0])
        sm = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], args.image_size)
        return ((sm[..., 1] + 1 - sm[..., 0]) / 2.0)[0].float().cpu().numpy()

    @torch.no_grad()
    def readtok(x_pre):  # pre-ln_post x-stream tokens [1, N, C] -> ln_post @ proj, exactly VisionTransformer.forward
        t = vis.ln_post(x_pre.float()).to(vis.proj.dtype)
        return t @ vis.proj

    @torch.no_grad()
    def inline_testpy(pf, tf):  # literal copy of the test.py loop body, single layer (sanity)
        pfeat = pf[0] / pf[0].norm(dim=-1, keepdim=True)
        similarity, _ = AnomalyCLIP_lib.compute_similarity(pfeat, tf[0])
        sm = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
        return ((sm[..., 1] + 1 - sm[..., 0]) / 2.0)[0].float().cpu().numpy()

    rows, max_err, checked = [], 0.0, False
    max_seq, max_f32, max_tok = 0.0, 0.0, 0.0
    for i in idx:
        it = data[int(i)]
        gt = it["img_mask"][0].numpy() > 0.5 if it["img_mask"].ndim == 3 else it["img_mask"].numpy() > 0.5
        if gt.sum() < 20 or (~gt).sum() < 20:
            continue
        img = it["img"].unsqueeze(0).to(device)
        with torch.no_grad():
            imf, pf, c = fwd(img)
            assert c["n"] == N_ATTN, f"expected {N_ATTN} V-V attention hooks, got {c['n']}"
            imf = imf / imf.norm(dim=-1, keepdim=True)
            zt = torch.full((1,), args.ec_z_pix, device=device)
            _, c_pos, c_neg = cond(visual_descriptor(imf, pf), z_override=zt)
            tf = conditioned_text_features(model, pl, c_pos, c_neg)
            # --- decomposition assert: x_5 + sum VVattn_l reproduces the model's own final x-stream tokens (pre-ln_post)
            d_seq = (c["seq"].float() - c["final"].float()).abs().max().item()       # model-dtype sequential sum (same rounding as x += x_res)
            d_f32 = (c["x5"].float() + c["sum32"] - c["final"].float()).abs().max().item()  # fp32 sum (report-only; differs by fp16 rounding)
            d_tok = (readtok(c["seq"]).float() - pf[0].float()).abs().max().item()  # readout of the reconstruction vs model's own tokens
            max_seq, max_f32, max_tok = max(max_seq, d_seq), max(max_f32, d_f32), max(max_tok, d_tok)
            if d_seq >= 1e-2 or d_tok >= 1e-2:
                raise SystemExit(f"STOP: decomposition fails on {it['img_path']}: |seq-final|={d_seq:.3e} |tok-pf|={d_tok:.3e} (must be < 1e-2)")
            if not checked:
                d0 = np.abs(tokens_to_map(pf[0], tf) - inline_testpy(pf, tf)).max()
                print(f"sanity: decomposition max|seq-final|={d_seq:.2e}, fp32-sum diff={d_f32:.2e} (report), readout max|tok-pf|={d_tok:.2e}; "
                      f"max|readout - test.py loop|={d0:.2e} (all must be < 1e-2)")
                assert d0 < 1e-2
                checked = True
            # partner x_5 (swap control): a plain forward on a different, fixed-seed-paired image of the same dataset
            pimg = data[int(partner[int(i)])]["img"].unsqueeze(0).to(device)
            x5p = fwd(pimg)[2]["x5"].float()
            S, x5 = c["sum32"], c["x5"].float()
            toks = {"base": pf[0], "no_res": readtok(S), "no_res_last": readtok(c["last"]), "swap_res": readtok(S + x5p)}
            nr = float((x5[0, 1:].norm(dim=-1) / S[0, 1:].norm(dim=-1)).mean())
            nrl = float((x5[0, 1:].norm(dim=-1) / c["last"][0, 1:].float().norm(dim=-1)).mean())
            nsw = float((x5p[0, 1:].norm(dim=-1) / x5[0, 1:].norm(dim=-1)).mean())
        sid = os.path.relpath(it["img_path"], args.data_path)
        row = {"image": sid, "log_area": float(np.log(gt.mean())), "norm_ratio": nr, "norm_ratio_last": nrl, "swap_norm_ratio": nsw}
        for arm in ARMS:
            m = tokens_to_map(toks[arm], tf)
            for s in SIGMAS:
                row[f"{arm}@{s}"] = auroc(gt, gaussian_filter(m, sigma=s))
        if ref is not None:
            if sid not in ref:
                raise SystemExit(f"STOP: {sid} missing from EXP-012 reference")
            err = abs(row["base@4"] - ref[sid])
            max_err = max(max_err, err)
            if err > 1e-4:
                raise SystemExit(f"STOP: parity failure on {sid}: {row['base@4']:.6f} vs EXP-012 {ref[sid]:.6f} (|d|={err:.2e} > 1e-4)")
        rows.append(row)
        if len(rows) % 50 == 0:
            print(f"{len(rows)} images", flush=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_csv} ({len(rows)} images); max|seq-final|={max_seq:.2e}, max|fp32sum-final|={max_f32:.2e}, max|tok-pf|={max_tok:.2e}; "
          f"max |base@4 - EXP-012| = {max_err:.2e}" + ("" if ref else " (NOT CHECKED)"))


# ------------------------------------------------------------------ aggregate
def load(path):
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "image"}


def aggregate(paths):
    names = [os.path.basename(p).replace("resid_", "").replace(".csv", "") for p in paths]
    d32 = {a: {} for a in TEST_ARMS}; lo = {a: {} for a in TEST_ARMS}; gap = {a: {} for a in TEST_ARMS}
    for nm, p in zip(names, paths):
        d = load(p)
        print(f"\n=== {nm} (n={len(d['log_area'])}) per-image pixel AUROC, paired bootstrap 95% CI over images ===")
        print("mean AUROC   " + "  ".join(f"{a}@{s}={np.nanmean(d[f'{a}@{s}']):.4f}" for s in SIGMAS for a in ARMS))
        for s in SIGMAS:
            for a in ARMS[1:]:
                m, l, h = paired_bootstrap_ci(d[f"{a}@{s}"] - d[f"base@{s}"])
                print(f"delta {a} - base @sigma{s}: {m:+.4f}  [{l:+.4f}, {h:+.4f}]")
                if s == 32 and a in TEST_ARMS:
                    d32[a][nm], lo[a][nm] = m, l
        for a in TEST_ARMS:
            m, l, h = paired_bootstrap_ci(d[f"{a}@32"] - d["swap_res@32"])
            gap[a][nm] = m
            print(f"{a} - swap_res @32: {m:+.4f}  [{l:+.4f}, {h:+.4f}]")
        for a in TEST_ARMS:  # report-only
            m, l, h = paired_bootstrap_ci(d[f"{a}@4"] - d["base@32"])
            print(f"(report-only) {a}@4 - base@32: {m:+.4f}  [{l:+.4f}, {h:+.4f}]")
        print(f"(report-only) token-norm ratio ||x_5||/||sum attn|| (mean over images of token-mean): {d['norm_ratio'].mean():.3f}; "
              f"||x_5||/||attn_24|| {d['norm_ratio_last'].mean():.3f}; swap ||x_5'||/||x_5|| {d['swap_norm_ratio'].mean():.3f}")
        for a in TEST_ARMS:
            q = quartile_means(d["log_area"], d[f"{a}@32"] - d["base@32"])
            print(f"(report-only) delta32 {a} by area quartile (small->large): " + " ".join(f"{x:+.4f}" for x in q))
    print("\ndelta32:", {a: {k: round(v, 4) for k, v in d32[a].items()} for a in TEST_ARMS})
    print("CI lower:", {a: {k: round(v, 4) for k, v in lo[a].items()} for a in TEST_ARMS})
    print("arm - swap_res @32:", {a: {k: round(v, 4) for k, v in gap[a].items()} for a in TEST_ARMS})
    print(f"\nVERDICT: {verdict(d32, lo, gap)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=("eval",))
    ap.add_argument("--aggregate", nargs="+", default=None)
    ap.add_argument("--data_path")
    ap.add_argument("--checkpoint_path")
    ap.add_argument("--dataset", default="colon")
    ap.add_argument("--reference_csv", default=None, help="EXP-012 pixel_per_image_predictions.csv for this dataset (parity, tol 1e-4)")
    ap.add_argument("--ec_z_pix", type=float, default=1.6)
    ap.add_argument("--out_csv", default="resid.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    a = ap.parse_args()
    if a.aggregate:
        aggregate(a.aggregate)
    elif a.mode == "eval":
        stage_eval(a)
    else:
        ap.error("give eval | --aggregate")
