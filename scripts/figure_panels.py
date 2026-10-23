"""Export clean figure panels (image, ground-truth mask, anomaly / similarity map, overlay) for ONE image with a trained (ECP) checkpoint.
Server, needs torch.  Writes PNGs under --out_dir, no other side effects.
  python scripts/figure_panels.py --image <img.png> [--mask <mask.png>] --checkpoint_path checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth \
         --out_dir figure_panels --z_list 1.6            # several z: --z_list -0.69 0.0 0.8 1.6 2.4  (shows the effect of the extent coordinate)
Files: <name>_image.png (518x518 RGB), <name>_gt.png (binary mask, if --mask), <name>_map_z<z>.png (jet heat map alone),
<name>_overlay_z<z>.png (jet map blended 50% on the image, the style of the repo's visualizer), plus <name>_map_z<z>.npy (raw float map).
Maps use the pixel head exactly as test.py (sigma 4 Gaussian, per-image min-max normalisation for display only). For a dataset without ECP use --z_list with any value; it is ignored by non-ECP checkpoints."""
import os, sys, argparse
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def jet(m01):
    import cv2
    u8 = (np.clip(m01, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", default=None)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--out_dir", default="figure_panels")
    ap.add_argument("--name", default=None)
    ap.add_argument("--z_list", type=float, nargs="+", default=[1.6])
    ap.add_argument("--sigma", type=int, default=4)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    a = ap.parse_args()

    import torch
    import AnomalyCLIP_lib
    from utils import get_transform
    from extent_prompt import visual_descriptor, conditioned_text_features
    from analyze_badcases import load_model

    S = a.image_size
    name = a.name or os.path.splitext(os.path.basename(a.image))[0]
    os.makedirs(a.out_dir, exist_ok=True)
    device, model, pl, cond = load_model(a)
    preprocess, _ = get_transform(a)

    pil = Image.open(a.image).convert("RGB")
    rgb = np.asarray(pil.resize((S, S), Image.BICUBIC))
    Image.fromarray(rgb).save(os.path.join(a.out_dir, f"{name}_image.png"))
    if a.mask:
        m = np.asarray(Image.open(a.mask).convert("L").resize((S, S), Image.NEAREST)) > 127
        Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(a.out_dir, f"{name}_gt.png"))

    img = preprocess(pil).unsqueeze(0).to(device)
    for z in a.z_list:
        with torch.no_grad():
            imf, pf = model.encode_image(img, [24], DPAM_layer=20)
            imf = imf / imf.norm(dim=-1, keepdim=True)
            pfeat = pf[0] / pf[0].norm(dim=-1, keepdim=True)
            if cond is not None:
                desc = visual_descriptor(imf, pf)
                zt = torch.full((imf.shape[0],), float(z), device=device)
                _, c_pos, c_neg = cond(desc, z_override=zt)
                tf = conditioned_text_features(model, pl, c_pos, c_neg)[0]
            else:                                                     # plain AnomalyCLIP prompts
                prompts, tok, comp = pl(cls_id=None)
                tf = model.encode_text_learn(prompts, tok, comp).float()
                tf = torch.stack(torch.chunk(tf, dim=0, chunks=2), dim=1)
                tf = (tf / tf.norm(dim=-1, keepdim=True))[0]
            sim, _ = AnomalyCLIP_lib.compute_similarity(pfeat, tf)
            sm = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:, :], S)
            amap = ((sm[..., 1] + 1 - sm[..., 0]) / 2.0)[0].float().cpu().numpy()
        amap = gaussian_filter(amap, sigma=a.sigma).astype(np.float32)
        np.save(os.path.join(a.out_dir, f"{name}_map_z{z}.npy"), amap)
        m01 = (amap - amap.min()) / (amap.max() - amap.min() + 1e-12)
        heat = jet(m01)
        Image.fromarray(heat).save(os.path.join(a.out_dir, f"{name}_map_z{z}.png"))
        over = (0.5 * rgb.astype(float) + 0.5 * heat.astype(float)).astype(np.uint8)
        Image.fromarray(over).save(os.path.join(a.out_dir, f"{name}_overlay_z{z}.png"))
        print(f"z={z}: map range {amap.min():.4f}..{amap.max():.4f} -> {a.out_dir}/{name}_overlay_z{z}.png")


if __name__ == "__main__":
    main()
