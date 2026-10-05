"""
EXP-018 / H009: is there a fixed-pattern field g(pos), independent of image content, in the final V-V patch tokens
(positional-embedding interpolation 24x24 -> 37x37), and does subtracting a SOURCE-estimated g improve CVC per-image pixel AUROC
beyond post-hoc sigma32, position-specifically?  Frozen model, forward passes only; labels are used only to SCORE.

Stage 1 (existence, no CVC labels; MVTec source with random dihedral + random-resized-crop so content is uniform over positions):
  python analyze_posart.py g --mvtec_path $MVTEC --checkpoint_path checkpoints/ecp_extent/epoch_15.pth --out_npz posart_G.npz \
         --cvc_info $CVC/CVC-ClinicDB $CVC/Kvasir $CVC/CVC-ColonDB
Stage 2 (effect, one command per dataset):
  python analyze_posart.py eval --data_path $CVC/CVC-ClinicDB --checkpoint_path checkpoints/ecp_extent/epoch_15.pth \
         --g_npz posart_G.npz --reference_csv <EXP-012 .../ClinicDB/ecp_extent/pixel_per_image_predictions.csv> --out_csv posart_ClinicDB.csv
Aggregate:
  python analyze_posart.py --aggregate posart_ClinicDB.csv posart_Kvasir.csv posart_ColonDB.csv --g_npz posart_G.npz

Arms (all share ONE set of ECP text features per image, computed from the unmodified forward; only the token readout changes):
  base      raw tokens                         -> normalise -> test.py readout (sum over --layers)
  art_src   token - G_l (per layer, patch tokens only; CLS untouched) -> normalise -> same readout
  perm_src  token - G_l[perm]  (position-PERMUTED G, fixed seed): same energy, wrong positions -> specificity control
Each arm is scored after post-sigma 4 and 32. Decision metric: per-image pixel AUROC at 518x518, paired by image.
`base24@4` (layer 24 only, sigma 4) is the EXP-012 identity map and is checked against the EXP-012 per-image AUROCs (tol 1e-4).
"""
import os
import csv
import json
import hashlib
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter

from posart_stats import (auroc, estimate_artifact, split_half_cosine, energy_ratio, permute_positions, subtract_artifact,
                          paired_bootstrap_ci, verdict, cosine, split_halves, sample_crop)

ALL_LAYERS = (6, 12, 18, 24)
ARMS = ("base", "art_src", "perm_src")
SIGMAS = (4, 32)
PERM_SEED = 20261005


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def load_model(args, with_ecp):
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


def encode_tokens(model, img):
    """Raw patch tokens (after ln_post @ proj, before any normalisation) of layers 6/12/18/24 + CLS image feature."""
    import torch
    with torch.no_grad():
        imf, pf = model.encode_image(img, list(ALL_LAYERS), DPAM_layer=20)
    return imf, pf


# ------------------------------------------------------------------ stage 1
def load_pil(path):
    from PIL import Image
    return Image.open(path).convert("RGB")


def augment(img, rng, size):
    """Random dihedral (8) + random-resized-crop (area 0.3-1) + resize to size x size (bicubic)."""
    from PIL import Image
    k = rng.randint(8)
    if k >= 4:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    for _ in range(k % 4):
        img = img.transpose(Image.ROTATE_90)
    W, H = img.size
    x0, y0, w, h = sample_crop(rng, W, H)
    return img.crop((x0, y0, x0 + w, y0 + h)).resize((size, size), Image.BICUBIC)


def stage_g(args):
    import torch
    from dataset import Dataset
    from utils import get_transform
    device, model, pl, cond = load_model(args, True)
    preprocess, target_transform = get_transform(args)
    # same split the ECP was trained on: train.py builds Dataset(..., mode default 'test') on the MVTec root
    src = Dataset(root=args.mvtec_path, transform=preprocess, target_transform=target_transform, dataset_name="mvtec")
    n_all = len(src)
    ids = np.random.RandomState(args.seed).permutation(n_all)[: args.n_source]
    A, B = split_halves(len(ids), args.seed)
    half_of = {int(j): 0 for j in A}
    half_of.update({int(j): 1 for j in B})
    L = len(ALL_LAYERS)
    S = [np.zeros((L, 37 * 37, 768)) for _ in range(2)]
    cnt = [0, 0]
    s1 = np.zeros((L, 768)); s2 = np.zeros((L, 768)); ntok = 0
    rng = np.random.RandomState(args.seed + 1)
    paths = []
    for j, i in enumerate(ids):
        path = os.path.join(src.root, src.data_all[int(i)]["img_path"])
        paths.append(os.path.relpath(path, src.root))
        img = preprocess(augment(load_pil(path), rng, args.image_size)).unsqueeze(0).to(device)
        _, pf = encode_tokens(model, img)
        toks = np.stack([t[0, 1:].double().cpu().numpy() for t in pf])  # [L, 1369, 768]
        S[half_of[j]] += toks
        cnt[half_of[j]] += 1
        s1 += toks.sum(1); s2 += (toks ** 2).sum(1); ntok += toks.shape[1]
        if (j + 1) % 100 == 0:
            print(f"stage1 {j + 1}/{len(ids)}", flush=True)
    GA = [estimate_artifact(S[0][l], cnt[0]) for l in range(L)]
    GB = [estimate_artifact(S[1][l], cnt[1]) for l in range(L)]
    G = np.stack([estimate_artifact(S[0][l] + S[1][l], cnt[0] + cnt[1]) for l in range(L)]).astype(np.float32)
    cos = np.array([split_half_cosine(GA[l], GB[l]) for l in range(L)])
    var = (s2 / ntok - (s1 / ntok) ** 2).sum(1)
    energy = np.array([energy_ratio(G[l].astype(np.float64), var[l]) for l in range(L)])
    dec = [ALL_LAYERS.index(l) for l in args.layers]
    cos_dec = float(cos[dec].mean())
    print("\n=== STAGE 1: existence of a fixed position pattern (MVTec source, randomised content positions) ===")
    print(f"n_source={len(ids)} (halves {cnt[0]}/{cnt[1]})")
    for l, c, e in zip(ALL_LAYERS, cos, energy):
        print(f"  layer {l:2d}: split-half cosine {c:+.4f}   energy ratio (report-only) {e:.4f}")
    print(f"split-half cosine, mean over 4 layers (report): {cos.mean():+.4f}")
    print(f"split-half cosine, mean over DECISION layers {list(args.layers)}: {cos_dec:+.4f}   (gate: >= 0.30)")
    cos_cvc = {}
    if args.cvc_info:  # information only: CVC layout can create a pattern with no artifact (known confound)
        for root in args.cvc_info:
            ds = Dataset(root=root, transform=preprocess, target_transform=target_transform, dataset_name="colon")
            sel = np.random.RandomState(0).permutation(len(ds))[: args.cvc_info_n]
            T = np.zeros((L, 37 * 37, 768))
            for i in sel:  # unlabeled: only the image tensor is used
                img = preprocess(load_pil(os.path.join(ds.root, ds.data_all[int(i)]["img_path"]))).unsqueeze(0).to(device)
                _, pf = encode_tokens(model, img)
                T += np.stack([t[0, 1:].double().cpu().numpy() for t in pf])
            Gc = [estimate_artifact(T[l], len(sel)) for l in range(L)]
            cos_cvc[os.path.basename(root)] = [split_half_cosine(G[l], Gc[l]) for l in range(L)]
            print(f"  [info only] cos(source G, G from {len(sel)} unlabeled {os.path.basename(root)}) per layer: "
                  + " ".join(f"{x:+.3f}" for x in cos_cvc[os.path.basename(root)]))
    digest = sha_array(G)
    np.savez(args.out_npz, G=G, layers=np.array(ALL_LAYERS), split_half_cos=cos, cos_decision=cos_dec,
             decision_layers=np.array(args.layers), energy=energy, sha256=digest,
             source_ids_sha256=hashlib.sha256("\n".join(paths).encode()).hexdigest(), n_source=len(ids), seed=args.seed,
             cos_cvc_json=json.dumps(cos_cvc))
    print(f"\nsaved {args.out_npz}  G sha256={digest}")
    print("STAGE 1 " + ("PASS (>=0.30) -> run stage 2" if cos_dec >= 0.30 else "FAIL (<0.30) -> FALSIFY (a); skip stage 2"))


# ------------------------------------------------------------------ stage 2
def stage_eval(args):
    import torch
    import AnomalyCLIP_lib
    from dataset import Dataset
    from utils import get_transform
    from extent_prompt import visual_descriptor, conditioned_text_features

    z = np.load(args.g_npz, allow_pickle=False)
    G_all = z["G"]
    assert sha_array(G_all) == str(z["sha256"]), "G file hash mismatch"
    print(f"G sha256={str(z['sha256'])}  split-half cos (decision)={float(z['cos_decision']):+.4f}")
    device, model, pl, cond = load_model(args, True)
    preprocess, target_transform = get_transform(args)
    data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    idx = np.random.RandomState(0).permutation(len(data))[: args.limit] if args.limit else range(len(data))
    ref = None
    if args.reference_csv:
        ref = {r["sample_id"]: float(r["per_image_pixel_auroc"]) for r in csv.DictReader(open(args.reference_csv))}
    else:
        print("WARNING: no --reference_csv, EXP-012 parity NOT checked")
    layer_pos = [ALL_LAYERS.index(l) for l in args.layers]
    Gt = {}
    perms = {}
    for p in layer_pos:
        Gt[p] = torch.from_numpy(G_all[p]).to(device)
        perms[p] = np.random.RandomState(PERM_SEED + p).permutation(Gt[p].shape[0])
    Gperm = {p: Gt[p][torch.from_numpy(perms[p]).to(device)] for p in layer_pos}

    def tokens_to_map(tok, tf):
        tok = tok / tok.norm(dim=-1, keepdim=True)
        sim, _ = AnomalyCLIP_lib.compute_similarity(tok, tf[0])
        sm = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], args.image_size)
        return ((sm[..., 1] + 1 - sm[..., 0]) / 2.0)

    @torch.no_grad()
    def readout(pf, tf, Gd, positions):
        """sum over the chosen layers of the test.py per-layer map, tokens optionally shifted by -Gd[p] (patch tokens only)."""
        maps = []
        for p in positions:
            tok = pf[p]
            if Gd is not None:
                tok = torch.cat([tok[:, :1], tok[:, 1:] - Gd[p].to(tok.dtype)[None]], dim=1)
            maps.append(tokens_to_map(tok, tf))
        return torch.stack(maps).sum(0)[0].float().cpu().numpy()

    @torch.no_grad()
    def inline_testpy(pf, tf, positions):  # literal copy of the test.py loop body (no distractor), for the sanity assert
        out = []
        for p in positions:
            patch_feature = pf[p] / pf[p].norm(dim=-1, keepdim=True)
            similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, tf[0])
            similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
            out.append((similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0)
        return torch.stack(out).sum(dim=0)[0].float().cpu().numpy()

    rows, max_err, checked = [], 0.0, False
    for i in idx:
        it = data[int(i)]
        gt = it["img_mask"][0].numpy() > 0.5 if it["img_mask"].ndim == 3 else it["img_mask"].numpy() > 0.5
        if gt.sum() < 20 or (~gt).sum() < 20:
            continue
        img = it["img"].unsqueeze(0).to(device)
        imf, pf = encode_tokens(model, img)
        imf = imf / imf.norm(dim=-1, keepdim=True)
        zt = torch.full((1,), args.ec_z_pix, device=device)
        with torch.no_grad():
            _, c_pos, c_neg = cond(visual_descriptor(imf, pf), z_override=zt)  # uses pf[-1] (layer 24) only, as with --features_list 24
            tf = conditioned_text_features(model, pl, c_pos, c_neg)
        if not checked:  # no-subtraction path must equal the original test.py readout
            zero = {p: torch.zeros_like(Gt[p]) for p in layer_pos}
            d1 = np.abs(readout(pf, tf, zero, layer_pos) - inline_testpy(pf, tf, layer_pos)).max()
            d2 = np.abs(readout(pf, tf, None, layer_pos) - inline_testpy(pf, tf, layer_pos)).max()
            print(f"sanity: max|readout(tok-0) - test.py loop| = {d1:.2e}, max|readout(None) - test.py loop| = {d2:.2e} (must be < 1e-2)")
            assert d1 < 1e-2 and d2 < 1e-2
            checked = True
        sid = os.path.relpath(it["img_path"], args.data_path)
        row = {"image": sid, "log_area": float(np.log(gt.mean()))}
        # EXP-012 identity map: layer 24 only, sigma 4
        m24 = readout(pf, tf, None, [ALL_LAYERS.index(24)])
        row["base24@4"] = auroc(gt, gaussian_filter(m24, sigma=4))
        if ref is not None:
            if sid not in ref:
                raise SystemExit(f"STOP: {sid} missing from EXP-012 reference")
            err = abs(row["base24@4"] - ref[sid])
            max_err = max(max_err, err)
            if err > 1e-4:
                raise SystemExit(f"STOP: parity failure on {sid}: {row['base24@4']:.6f} vs EXP-012 {ref[sid]:.6f} (|d|={err:.2e} > 1e-4)")
        for arm, Gd in (("base", None), ("art_src", Gt), ("perm_src", Gperm)):
            m = readout(pf, tf, Gd, layer_pos)
            for s in SIGMAS:
                row[f"{arm}@{s}"] = auroc(gt, gaussian_filter(m, sigma=s))
        rows.append(row)
        if len(rows) % 50 == 0:
            print(f"{len(rows)} images", flush=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_csv} ({len(rows)} images); max |base24@4 - EXP-012| = {max_err:.2e}" + ("" if ref else " (NOT CHECKED)"))


# ------------------------------------------------------------------ aggregate
def load(path):
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != "image"}


def summarise(d):
    """Per-dataset numbers used by the decision rule (pure; tested locally)."""
    out = {"n": len(d["log_area"])}
    for s in SIGMAS:
        for a in ARMS:
            out[f"{a}@{s}"] = float(np.nanmean(d[f"{a}@{s}"]))
        out[f"d_art@{s}"] = paired_bootstrap_ci(d[f"art_src@{s}"] - d[f"base@{s}"])
        out[f"d_perm@{s}"] = paired_bootstrap_ci(d[f"perm_src@{s}"] - d[f"base@{s}"])
    out["gap32"] = paired_bootstrap_ci(d["art_src@32"] - d["perm_src@32"])
    out["art4_vs_base32"] = paired_bootstrap_ci(d["art_src@4"] - d["base@32"])
    return out


def aggregate(paths, g_npz):
    names = [os.path.basename(p).replace("posart_", "").replace(".csv", "") for p in paths]
    d32, gap, lo = {}, {}, {}
    for nm, p in zip(names, paths):
        r = summarise(load(p))
        print(f"\n=== {nm} (n={r['n']}) per-image pixel AUROC, paired bootstrap 95% CI over images ===")
        print("mean AUROC   " + "  ".join(f"{a}@{s}={r[f'{a}@{s}']:.4f}" for s in SIGMAS for a in ARMS))
        for s in SIGMAS:
            for a in ("art", "perm"):
                m, l, h = r[f"d_{a}@{s}"]
                print(f"delta {a}_src - base @sigma{s}: {m:+.4f}  [{l:+.4f}, {h:+.4f}]")
        m, l, h = r["gap32"]
        print(f"art_src - perm_src @32: {m:+.4f}  [{l:+.4f}, {h:+.4f}]")
        m, l, h = r["art4_vs_base32"]
        print(f"(report-only) art_src@4 - base@32: {m:+.4f}  [{l:+.4f}, {h:+.4f}]")
        d32[nm], lo[nm], gap[nm] = r["d_art@32"][0], r["d_art@32"][1], r["gap32"][0]
    z = np.load(g_npz, allow_pickle=False)
    cos = float(z["cos_decision"])
    print(f"\nstage-1 split-half cosine (decision layers {list(z['decision_layers'])}): {cos:+.4f}")
    print("delta32 (art_src - base):", {k: round(v, 4) for k, v in d32.items()}, " CI lower:", {k: round(v, 4) for k, v in lo.items()},
          " art-perm gap32:", {k: round(v, 4) for k, v in gap.items()})
    print(f"\nVERDICT: {verdict(cos, d32, gap, lo)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=("g", "eval"))
    ap.add_argument("--aggregate", nargs="+", default=None)
    ap.add_argument("--g_npz", default="posart_G.npz")
    ap.add_argument("--out_npz", default="posart_G.npz")
    ap.add_argument("--mvtec_path")
    ap.add_argument("--n_source", type=int, default=600)
    ap.add_argument("--cvc_info", nargs="*", default=[])
    ap.add_argument("--cvc_info_n", type=int, default=200)
    ap.add_argument("--layers", type=int, nargs="+", default=[24], choices=ALL_LAYERS,
                    help="decision layers (24 = EXP-012/--features_list 24 pipeline; 6 12 18 24 = test.py default sum)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_path")
    ap.add_argument("--checkpoint_path")
    ap.add_argument("--dataset", default="colon")
    ap.add_argument("--reference_csv", default=None, help="EXP-012 pixel_per_image_predictions.csv for this dataset (parity, tol 1e-4)")
    ap.add_argument("--ec_z_pix", type=float, default=1.6)
    ap.add_argument("--out_csv", default="posart.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    a = ap.parse_args()
    if a.aggregate:
        aggregate(a.aggregate, a.g_npz)
    elif a.mode == "g":
        stage_g(a)
    elif a.mode == "eval":
        stage_eval(a)
    else:
        ap.error("give g | eval | --aggregate")
