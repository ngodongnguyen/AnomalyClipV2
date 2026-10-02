"""
Cross-image (transductive) commonness of top-scoring patches. No training, no labels used by the signal.
Signal c(p) = mean of top-k cosine sims between patch p and patches of OTHER test images (excluding the --excl_near
most-similar images, to kill near-duplicate video frames). Hypothesis H (pre-registered): false-positive hotspots are
recurring generic structures, so FP patches have HIGHER c than TP lesion patches.
Copy xrare_stats.py + this file into repo root (needs within_stats.py, already in repo).  Pre-registered rules: see bottom.
  python analyze_cross_image_rarity.py --dataset colon --data_path <root> --checkpoint_path checkpoints/ecp_extent/epoch_15.pth
"""
import argparse, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from dataset import Dataset
from utils import get_transform
from extent_prompt import ExtentConditioner, visual_descriptor, conditioned_text_features
from within_stats import auroc, within_image_auroc
from xrare_stats import allow_matrix, knn_mean_np

def knn_mean_torch(Q, q_img, K, k_img, allow, k=5, chunk=2048):
    out = []
    for a in range(0, len(Q), chunk):
        S = Q[a:a + chunk] @ K.T
        S = S.masked_fill(~allow[q_img[a:a + chunk][:, None], k_img[None, :]], float("-inf"))
        out.append(S.topk(k, dim=1).values.float().mean(1))
    return torch.cat(out)

def selftest(dev):
    g = torch.Generator().manual_seed(0); n, p, d = 5, 30, 16
    Fm = F.normalize(torch.randn(n * p, d, generator=g), dim=-1); im = torch.arange(n).repeat_interleave(p)
    A = torch.tensor(allow_matrix(F.normalize(torch.randn(n, d, generator=g), dim=-1).numpy(), 1))
    ref = knn_mean_np(Fm.numpy(), im.numpy(), Fm.numpy(), im.numpy(), A.numpy(), k=4, chunk=11)
    got = knn_mean_torch(Fm.to(dev), im.to(dev), Fm.to(dev), im.to(dev), A.to(dev), k=4, chunk=17).cpu().numpy()
    assert np.allclose(ref, got, atol=1e-4), "torch/numpy knn mismatch"; print("selftest OK")

def run(a):
    dev = "cuda"; selftest(dev)
    params = {"Prompt_length": a.n_ctx, "learnabel_text_embedding_depth": a.depth, "learnabel_text_embedding_length": a.t_n_ctx}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=dev, design_details=params); model.eval()
    pre, tt = get_transform(a)
    ds = Dataset(root=a.data_path, transform=pre, target_transform=tt, dataset_name=a.dataset)
    rng = np.random.default_rng(0); sel = set(rng.permutation(len(ds))[:a.n_img].tolist())
    pl = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
    ck = torch.load(a.checkpoint_path, map_location="cpu"); pl.load_state_dict(ck["prompt_learner"]); pl.to(dev); model.to(dev)
    model.visual.DAPM_replace(DPAM_layer=20)
    cond = ExtentConditioner(mode=ck["extent_cond"]).to(dev); cond.load_state_dict(ck["conditioner"]); cond.eval()
    recs = []; G = []; Pall = []
    for ii in tqdm(sorted(sel)):
        it = ds[ii]; img = it["img"].unsqueeze(0).to(dev); gt = it["img_mask"][0].numpy() > 0.5 if it["img_mask"].dim() == 3 else it["img_mask"].numpy() > 0.5
        with torch.no_grad():
            imf, pfl = model.encode_image(img, [24], DPAM_layer=20); pf = pfl[-1]; imf = imf / imf.norm(dim=-1, keepdim=True)
            _, cp, cn = cond(visual_descriptor(imf, pfl), z_override=torch.full((1,), a.ec_z_pix, device=dev))
            tf = conditioned_text_features(model, pl, cp, cn); pfn = F.normalize(pf, dim=-1)
            sim, _ = AnomalyCLIP_lib.compute_similarity(pfn, tf[0]); side = int((sim.shape[1] - 1) ** 0.5)
            s = ((sim[0, 1:, 1] + 1 - sim[0, 1:, 0]) / 2).cpu().numpy(); P = pfn[0, 1:].half()
        if gt.sum() < 20 or (~gt).sum() < 20: continue
        gray = np.array(Image.open(it["img_path"]).convert("L").resize((side, side), Image.BILINEAR)); valid = (gray > 10).ravel()
        g = np.array(Image.fromarray(gt.astype(np.uint8) * 255).resize((side, side), Image.NEAREST)) > 127
        if g.ravel()[valid].sum() < 4: continue
        recs.append(dict(s=s, valid=valid, gf=g.ravel(), dout=distance_transform_edt(~g).ravel(), n=len(Pall)))
        Pall.append(P); G.append(F.normalize(imf.float(), dim=-1)[0].cpu().numpy())
    n = len(recs); print(f"{n} images kept")
    allow = torch.tensor(allow_matrix(np.array(G), a.excl_near)).to(dev)
    # keys/queries: valid patches of all kept images
    Kt = torch.cat([Pall[i][torch.tensor(recs[i]["valid"]).to(dev)] for i in range(n)]); kimg = torch.cat([torch.full((int(recs[i]["valid"].sum()),), i) for i in range(n)]).to(dev)
    c_all = knn_mean_torch(Kt, kimg, Kt, kimg, allow, k=a.k, chunk=a.chunk).cpu().numpy()  # chunk x N_keys matrix: keep small (OOM at 2048 on ClinicDB)
    gmean = F.normalize(Kt.float().mean(0, keepdim=True), dim=-1); b_all = (Kt.float() @ gmean.T)[:, 0].cpu().numpy()   # trivial baseline
    off = 0; Y, C, B, IM, S_pool = [], [], [], [], []; S_all, C_all, L_all, I_all = [], [], [], []
    for i, r in enumerate(recs):
        m = int(r["valid"].sum()); c = c_all[off:off + m]; b = b_all[off:off + m]; off += m
        v = np.flatnonzero(r["valid"]); s = r["s"][v]; gf = r["gf"][v]; dout = r["dout"][v]
        S_all.append(s); C_all.append(c); L_all.append(gf); I_all.append(np.full(m, i))
        thr = np.quantile(s, 1 - a.top_frac); pool = s >= thr; tp = pool & gf; fp = pool & ~gf & (dout > 2); idx = np.flatnonzero(tp | fp)
        Y += list(fp[idx]); C += list(c[idx]); B += list(b[idx]); IM += [i] * len(idx)
    Y = np.array(Y, bool); IM = np.array(IM); rs = np.random.default_rng(1)
    ac, ab, ctl = auroc(Y, C), auroc(Y, B), auroc(rs.permutation(Y), C); wc, nw = within_image_auroc(Y, C, IM)
    print(f"\n=== {a.dataset}  k={a.k} excl_near={a.excl_near} pool top {a.top_frac:.0%}  TP={int((~Y).sum())} FP={int(Y.sum())}")
    print(f"(1) POOLED AUROC(FP vs TP | commonness c)   = {ac:.3f}   [H: >=0.65]   control(shuffled labels) {ctl:.3f}")
    print(f"(2) WITHIN-IMAGE AUROC (over {nw} imgs)      = {wc:.3f}   [H: >=0.60]")
    print(f"(3) trivial baseline AUROC(cos to dataset-mean patch) = {ab:.3f}   |c-0.5| - |b-0.5| = {abs(ac-.5)-abs(ab-.5):+.3f}  [H: >=+0.05]")
    # Stage 2 proxy (patch grid, all valid patches): s' = z(s) - lam*z(c), global z-scores (no per-image normalisation)
    S = np.concatenate(S_all); Cc = np.concatenate(C_all); L = np.concatenate(L_all); I = np.concatenate(I_all)
    z = lambda x: (x - x.mean()) / x.std(); sp = z(S) - a.lam * z(Cc)
    pa0, pa1 = auroc(L, S), auroc(L, sp); wa0, _ = within_image_auroc(L, S, I, 3); wa1, _ = within_image_auroc(L, sp, I, 3)
    print(f"(4) STAGE-2 PROXY patch-grid AUROC all valid patches: pooled {pa0:.4f} -> {pa1:.4f} ({100*(pa1-pa0):+.2f} pt) | within-image {wa0:.4f} -> {wa1:.4f}  (lam={a.lam}, fixed)")
    print("    (patch-grid proxy only; the formal exploitation test would be a test.py flag under the 12c rule)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True); p.add_argument("--checkpoint_path", required=True); p.add_argument("--dataset", default="colon")
    p.add_argument("--ec_z_pix", type=float, default=1.6); p.add_argument("--top_frac", type=float, default=0.10)
    p.add_argument("--k", type=int, default=5); p.add_argument("--excl_near", type=int, default=10); p.add_argument("--n_img", type=int, default=250)
    p.add_argument("--lam", type=float, default=0.5); p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--image_size", type=int, default=518); p.add_argument("--depth", type=int, default=9)
    p.add_argument("--n_ctx", type=int, default=12); p.add_argument("--t_n_ctx", type=int, default=4)
    run(p.parse_args())
