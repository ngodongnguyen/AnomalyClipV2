"""
Is the false-positive hotspot a NAMEABLE normal-anatomy concept? (no training, no labels used by any method)
Pre-registered decision rule is in the report; core stat unit-tested in distractor_stats.py.
Within the top-`top_frac` scoring patches of each image (ECP z_pix score), compare patches OUTSIDE the GT
(far from boundary, = FP hotspot) vs INSIDE the GT (= TP) by a raw-CLIP concept margin
(log mass on normal-anatomy distractor prompts - log mass on lesion prompts), plus a random-relabel null control.
Run from repo root (copy this file + distractor_stats.py there).
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
from distractor_stats import auroc, concept_margin, split_control

LESION = ["a polyp", "a tumor", "a lesion", "a protruding growth", "a mass", "an ulcer"]
DISTRACT = ["a mucosal fold", "a blood vessel", "an air bubble", "stool residue", "a surgical instrument",
            "specular reflection", "normal healthy mucosa", "a dark lumen", "an endoscope border"]


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
        # text transformer here only accepts [x, deep_prompts, counter]; an empty deep list = plain frozen CLIP text encoder
        T = F.normalize(model.encode_text_learn(model.token_embedding(tok).type(model.dtype), tok, []).float(), dim=-1).cpu().numpy()
    isd = np.array([False] * len(LESION) + [True] * len(DISTRACT)); rng = np.random.default_rng(0)
    Y, M, Mc, lesion_vs_bg = [], [], [], []
    for it in tqdm(loader):
        gt = it["img_mask"][0, 0].numpy() > 0.5
        if gt.sum() < 20 or (~gt).sum() < 20: continue
        with torch.no_grad():
            imf, pf = model.encode_image(it["img"].to(dev), [24], DPAM_layer=20); pf = pf[-1]
            imf = imf / imf.norm(dim=-1, keepdim=True)
            _, cp, cn = cond(visual_descriptor(imf, pf), z_override=torch.full((1,), a.ec_z_pix, device=dev))
            tf = conditioned_text_features(model, pl, cp, cn)
            pfn = F.normalize(pf, dim=-1)
            sim, _ = AnomalyCLIP_lib.compute_similarity(pfn, tf[0])
            side = int((sim.shape[1] - 1) ** 0.5)
            s = ((sim[0, 1:, 1] + 1 - sim[0, 1:, 0]) / 2).cpu().numpy(); P = pfn[0, 1:].float().cpu().numpy()
        gray = np.array(Image.open(it["img_path"][0]).convert("L").resize((side, side), Image.BILINEAR)); valid = (gray > 10).ravel()
        g = np.array(Image.fromarray(gt.astype(np.uint8) * 255).resize((side, side), Image.NEAREST)) > 127
        dout = distance_transform_edt(~g).ravel(); gf = g.ravel()
        if gf[valid].sum() < 4: continue
        m = concept_margin(P, T, isd); mc = concept_margin(P, T, split_control(len(names), isd, rng))
        lesion_vs_bg.append(auroc(gf[valid], -m[valid]))                       # precondition: lesion concepts align with real lesions
        thr = np.quantile(s[valid], 1 - a.top_frac); pool = valid & (s >= thr)
        tp = pool & gf; fp = pool & ~gf & (dout > 2)
        if tp.sum() == 0 or fp.sum() == 0: continue
        idx = np.where(tp | fp)[0]; Y += list(fp[idx]); M += list(m[idx]); Mc += list(mc[idx])
    Y = np.array(Y, bool)
    print(f"\n=== {a.dataset} {a.data_path}: pool patches TP={int((~Y).sum())} FP={int(Y.sum())}")
    print(f"precondition  lesion-vs-bg AUROC of -margin (mean per image) = {np.nanmean(lesion_vs_bg):.3f}  (need >= 0.60)")
    print(f"MAIN   AUROC(FP vs TP | margin)       = {auroc(Y, np.array(M)):.3f}")
    print(f"CONTROL AUROC (random concept relabel) = {auroc(Y, np.array(Mc)):.3f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True); p.add_argument("--checkpoint_path", required=True); p.add_argument("--dataset", default="colon")
    p.add_argument("--ec_z_pix", type=float, default=1.6); p.add_argument("--top_frac", type=float, default=0.05)
    p.add_argument("--image_size", type=int, default=518); p.add_argument("--depth", type=int, default=9)
    p.add_argument("--n_ctx", type=int, default=12); p.add_argument("--t_n_ctx", type=int, default=4)
    run(p.parse_args())
