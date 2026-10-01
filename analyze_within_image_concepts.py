"""
Is the 'FP hotspot is text-nameable' signal (section 12) a WITHIN-image signal, or a between-image confound?
No training. Same model/score/concepts as analyze_distractor_semantics.py. Run from repo root (copy this file,
within_stats.py and distractor_stats.py there). Core stats unit-tested in test_within_stats.py (synthetic known answers).
Reports: (1) pooled patch AUROC (reproduces section 12), (2) within-image patch AUROC (margin demeaned by construction),
(3) component-level paired stats: for (TP comp, FP comp) pairs in the SAME image, how often margin ranks FP above TP,
and among pairs the plain score gets wrong (FP>=TP) how many margin fixes. + random-concept-relabel control for each.
"""
import argparse, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner, tokenize
from dataset import Dataset
from utils import get_transform
from extent_prompt import ExtentConditioner, visual_descriptor, conditioned_text_features
from distractor_stats import LESION, DISTRACT, concept_margin, split_control
from within_stats import auroc, within_image_auroc, pool_components, pair_stats

def run(a):
    dev = "cuda"
    params = {"Prompt_length": a.n_ctx, "learnabel_text_embedding_depth": a.depth, "learnabel_text_embedding_length": a.t_n_ctx}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=dev, design_details=params); model.eval()
    pre, tt = get_transform(a)
    loader = torch.utils.data.DataLoader(Dataset(root=a.data_path, transform=pre, target_transform=tt, dataset_name=a.dataset), batch_size=1)
    pl = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
    ck = torch.load(a.checkpoint_path, map_location="cpu"); pl.load_state_dict(ck["prompt_learner"]); pl.to(dev); model.to(dev)
    model.visual.DAPM_replace(DPAM_layer=20)
    cond = ExtentConditioner(mode=ck["extent_cond"]).to(dev); cond.load_state_dict(ck["conditioner"]); cond.eval()
    names = LESION + DISTRACT
    with torch.no_grad():
        tok = tokenize([f"a photo of {n}" for n in names]).to(dev)
        T = F.normalize(model.encode_text_learn(model.token_embedding(tok).type(model.dtype), tok, []).float(), dim=-1).cpu().numpy()
    isd = np.array([False] * len(LESION) + [True] * len(DISTRACT)); rng = np.random.default_rng(0)
    Y, M, Mc, IM = [], [], [], []
    pair = {"m": [], "f": [], "c": [], "fc": []}; n_img_pairs = 0
    for ii, it in enumerate(tqdm(loader)):
        gt = it["img_mask"][0, 0].numpy() > 0.5
        if gt.sum() < 20 or (~gt).sum() < 20: continue
        with torch.no_grad():
            imf, pfl = model.encode_image(it["img"].to(dev), [24], DPAM_layer=20); pf = pfl[-1]
            imf = imf / imf.norm(dim=-1, keepdim=True)
            _, cp, cn = cond(visual_descriptor(imf, pfl), z_override=torch.full((1,), a.ec_z_pix, device=dev))
            tf = conditioned_text_features(model, pl, cp, cn)
            pfn = F.normalize(pf, dim=-1)
            sim, _ = AnomalyCLIP_lib.compute_similarity(pfn, tf[0])
            side = int((sim.shape[1] - 1) ** 0.5)
            s = ((sim[0, 1:, 1] + 1 - sim[0, 1:, 0]) / 2).cpu().numpy(); P = pfn[0, 1:].float().cpu().numpy()
        gray = np.array(Image.open(it["img_path"][0]).convert("L").resize((side, side), Image.BILINEAR)); valid = (gray > 10).ravel()
        g = np.array(Image.fromarray(gt.astype(np.uint8) * 255).resize((side, side), Image.NEAREST)) > 127
        dout = distance_transform_edt(~g); gf = g.ravel()
        if gf[valid].sum() < 4: continue
        m = concept_margin(P, T, isd); mc = concept_margin(P, T, split_control(len(names), isd, rng))
        thr = np.quantile(s[valid], 1 - a.top_frac); pool = valid & (s >= thr)
        tp = pool & gf; fp = pool & ~gf & (dout.ravel() > 2)
        idx = np.where(tp | fp)[0]
        Y += list(fp[idx]); M += list(m[idx]); Mc += list(mc[idx]); IM += [ii] * len(idx)
        comps = pool_components(pool.reshape(side, side), g, dout)
        if comps:
            cs = [s[c].mean() for c, _ in comps]; cf = [f for _, f in comps]
            r = pair_stats(cs, [m[c].mean() for c, _ in comps], cf); rc = pair_stats(cs, [mc[c].mean() for c, _ in comps], cf)
            if r is not None:
                n_img_pairs += 1; pair["m"].append(r[0]); pair["c"].append(rc[0])
                if r[3] > 0: pair["f"].append(r[2]); pair["fc"].append(rc[2])
    Y = np.array(Y, bool); IM = np.array(IM)
    wm, nw = within_image_auroc(Y, M, IM); wc, _ = within_image_auroc(Y, Mc, IM)
    print(f"\n=== {a.dataset} {a.data_path}  pool top {a.top_frac:.0%}  TP={int((~Y).sum())} FP={int(Y.sum())}")
    print(f"(1) POOLED patch AUROC(FP vs TP | margin)        = {auroc(Y, M):.3f}   control {auroc(Y, Mc):.3f}   [section 12: ~0.74-0.80]")
    print(f"(2) WITHIN-IMAGE patch AUROC (mean over {nw} imgs)  = {wm:.3f}   control {wc:.3f}")
    print(f"(3) COMPONENT pairs: images with >=1 TP and >=1 FP comp = {n_img_pairs}")
    print(f"    P(margin ranks FP comp above TP comp), mean over imgs = {np.nanmean(pair['m']):.3f}   control {np.nanmean(pair['c']):.3f}")
    print(f"    among pairs where plain score is WRONG (FP>=TP) [{len(pair['f'])} imgs]: margin fixes = {np.nanmean(pair['f']):.3f}   control {np.nanmean(pair['fc']):.3f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True); p.add_argument("--checkpoint_path", required=True); p.add_argument("--dataset", default="colon")
    p.add_argument("--ec_z_pix", type=float, default=1.6); p.add_argument("--top_frac", type=float, default=0.10)
    p.add_argument("--image_size", type=int, default=518); p.add_argument("--depth", type=int, default=9)
    p.add_argument("--n_ctx", type=int, default=12); p.add_argument("--t_n_ctx", type=int, default=4)
    run(p.parse_args())
