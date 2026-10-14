"""EXP-045: source score of every (layer, head) V-V attention head, from MVTec only (no medical label is used for selection).

select  : 300 anomalous MVTec test images (ascending SHA-256 of the sample id), patch labels = mask pooled to the 37x37 grid
          (lesion >= 0.5, normal <= 0.1, in-between skipped). Score of a head = AUROC separating the V-V attention weight
          (lesion query -> lesion key, self pair excluded) from (lesion query -> normal key), POOLED over images.
          Saves results/EXP-045/head_scores_mvtec.npy ([19 layers 6..24, 16 heads]) and head_ranking_mvtec.json.
transfer: report only. The same score on a medical set (its masks), printed with the Spearman against the MVTec scores. NEVER used to select.

Builder choices (fixed in code, written in the EXP-045 report): <= 128 lesion queries per image (deterministic, seeded by the image hash) to bound cost;
pooled AUROC from 4096 log10-weight bins over [1e-12, 1] (resolution 0.7% in weight; same-bin pairs count 1/2); weights are the softmax V-V
attention over ALL keys (CLS key included in the softmax, only patch keys are scored).

  python scripts/select_heads_mvtec.py select --data_path $A/mvtec
  python scripts/select_heads_mvtec.py transfer --name ClinicDB --dataset colon --data_path $CVC/CVC-ClinicDB
"""
import os
import sys
import json
import time
import hashlib
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import vvattn_stats as S


def sid_hash(s):
    return hashlib.sha256(s.encode()).hexdigest()


def patch_labels(mask01, grid=37):
    """mask [H,W] in {0,1} (H = W = 37 * k) -> (lesion bool [G*G], normal bool [G*G]) from the mean mask value per patch."""
    H = mask01.shape[0]
    k = H // grid
    frac = mask01[: grid * k, : grid * k].reshape(grid, k, grid, k).mean((1, 3)).reshape(-1)
    return frac >= 0.5, frac <= 0.1


def run(args):
    import torch
    import AnomalyCLIP_lib
    from dataset import Dataset
    from utils import get_transform
    import vvattn_core as C

    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth,
              "learnabel_text_embedding_length": args.t_n_ctx}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=params)
    model.eval()
    model.visual.DAPM_replace(DPAM_layer=20)
    preprocess, target_transform = get_transform(args)
    data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)

    cand = []
    for i, d in enumerate(data.data_all):
        if args.mode == "select":
            if d["anomaly"] == 1 and os.path.isfile(os.path.join(args.data_path, d["mask_path"])):
                cand.append((sid_hash(d["img_path"]), d["img_path"], i))
        else:
            cand.append((sid_hash(d["img_path"]), d["img_path"], i))
    cand.sort()
    if args.mode == "transfer":                                  # medical: keep images with mask >= 20 px and background >= 20 px, first n_images by hash
        keep = []
        for h, p, i in cand:
            m = data[i]["img_mask"]
            m = (m[0] if m.ndim == 3 else m).numpy() > 0.5
            if m.sum() >= 20 and (~m).sum() >= 20:
                keep.append((h, p, i))
            if len(keep) >= args.n_images:
                break
        cand = keep
    n = args.limit if args.limit else args.n_images
    cand = cand[:n]
    ids = [p for _, p, _ in cand]
    list_hash = S.image_list_sha256(ids)
    print(f"{args.mode}: {len(cand)} images, image-list sha256 {list_hash[:16]}...", flush=True)

    state = C.new_state()
    acc = C.HeadScoreAccumulator(device=device)
    state["capture"] = acc
    used = skipped = 0
    t0 = time.time()
    with C.patched_attention(model, state):
        for n_i, (h, p, i) in enumerate(cand):
            it = data[i]
            m = it["img_mask"]
            m = ((m[0] if m.ndim == 3 else m).numpy() > 0.5).astype(np.float32)
            kl, kn = patch_labels(m)
            if kl.sum() < 2 or kn.sum() < 1:
                skipped += 1
                continue
            lesion_idx = np.nonzero(kl)[0]
            rs = np.random.RandomState(int(h[:8], 16))
            q = np.sort(rs.permutation(lesion_idx)[: args.max_queries])
            acc.set_image(torch.from_numpy(kl).to(device), torch.from_numpy(kn).to(device), torch.from_numpy(q).long().to(device))
            with torch.no_grad():
                model.encode_image(it["img"].unsqueeze(0).to(device), [24], DPAM_layer=20)
            used += 1
            if (n_i + 1) % 25 == 0:
                print(f"{n_i + 1}/{len(cand)} images ({time.time() - t0:.0f}s)", flush=True)
    pos, neg = acc.numpy()
    scores = S.hist_auroc(pos, neg)                              # [19, 16]
    print(f"used {used} images, skipped {skipped} (no >=2 lesion / >=1 normal patch); pooled pairs per head: "
          f"pos median {int(np.median(pos.sum(-1)))}, neg median {int(np.median(neg.sum(-1)))}; NaN heads {int(np.isnan(scores).sum())}; {time.time() - t0:.0f}s")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "select":
        np.save(os.path.join(args.out_dir, "head_scores_mvtec.npy"), scores)
        rank = S.ranking_from_scores(scores)
        out = {"layers": list(range(S.FIRST_LAYER, S.FIRST_LAYER + S.N_LAYERS)), "scores": np.where(np.isnan(scores), None, scores).tolist(),
               "ranking": [[int(l), int(h)] for l, h in rank], "image_list_sha256": list_hash, "n_images_requested": len(cand),
               "n_images_used": used, "max_queries": args.max_queries, "bins": 4096, "nan_to": 0.5, "data_path": args.data_path}
        with open(os.path.join(args.out_dir, "head_ranking_mvtec.json"), "w") as f:
            json.dump(out, f)
        print("saved head_scores_mvtec.npy, head_ranking_mvtec.json")
        flat = np.where(np.isnan(scores), 0.5, scores)
        print(f"score range {flat.min():.3f}..{flat.max():.3f}, mean {flat.mean():.3f}; share of heads with score > 0.5: {(flat > 0.5).mean():.2f}")
        print("top-10 (layer,head,score):", [(l, h, round(float(flat[l - S.FIRST_LAYER, h]), 3)) for l, h in rank[:10]])
        print("bottom-10:", [(l, h, round(float(flat[l - S.FIRST_LAYER, h]), 3)) for l, h in rank[-10:]])
        print("mean score per layer 6..24:", np.round(flat.mean(1), 3).tolist())
    else:
        np.save(os.path.join(args.out_dir, f"head_scores_{args.name}.npy"), scores)
        ref_path = os.path.join(args.out_dir, "head_scores_mvtec.npy")
        if os.path.exists(ref_path):
            from scipy.stats import spearmanr
            a = np.load(ref_path).reshape(-1); b = scores.reshape(-1)
            ok = ~np.isnan(a) & ~np.isnan(b)
            rho = spearmanr(a[ok], b[ok])[0]
            ra = np.argsort(np.argsort(-a[ok])); rb = np.argsort(np.argsort(-b[ok]))
            half = ok.sum() // 2
            print(f"[REPORT-ONLY, never used to select] {args.name}: Spearman(MVTec head score, {args.name} head score) = {rho:+.3f} over {ok.sum()} heads; "
                  f"overlap of the two top-50% sets = {((ra < half) & (rb < half)).sum()}/{half}")
        else:
            print("no head_scores_mvtec.npy yet: run `select` first")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("select", "transfer"))
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--dataset", default="mvtec")
    ap.add_argument("--name", default="mvtec")
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "results", "EXP-045"))
    ap.add_argument("--n_images", type=int, default=300)
    ap.add_argument("--max_queries", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="smoke: use only the first N images of the hash order")
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    a = ap.parse_args()
    if a.mode == "transfer" and a.n_images == 300:
        a.n_images = 100
    run(a)
