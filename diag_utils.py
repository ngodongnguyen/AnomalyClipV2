"""Torch-free helpers for analyze_diagnostic_suite.py (kept separate so they can be unit-tested anywhere)."""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, distance_transform_edt
from sklearn.metrics import roc_auc_score
from skimage.color import rgb2lab


def gauss(m, sigma):
    return gaussian_filter(m, sigma=sigma) if sigma > 0 else m


def norm01(m):
    return (m - m.min()) / (m.max() - m.min() + 1e-8)


def safe_auc(gt_flat, m):
    return float(roc_auc_score(gt_flat, m.ravel()))


def raw_descriptors(rgb, gt):
    """Image/lesion descriptors computed from the raw RGB image only (no CLIP)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    valid = gray > 10
    lesion = gt & valid
    outside = (~gt) & valid
    if lesion.sum() < 10 or outside.sum() < 10:
        return None
    sc = rgb.shape[0] / 518.0
    lab = rgb2lab(rgb)
    blur = cv2.GaussianBlur(gray, (0, 0), max(0.5, 1.5 * sc))
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    n_valid = valid.sum()
    return dict(
        log_area=float(np.log(lesion.sum() / n_valid)),
        lab_contrast=float(np.linalg.norm(lab[lesion].mean(0) - lab[outside].mean(0))),
        tex_ratio=float(np.log((grad[lesion].mean() + 1e-6) / (grad[outside].mean() + 1e-6))),
        specular_frac=float(((gray > 235) & valid).sum() / n_valid),
        dark_frac=float(((gray < 50) & valid).sum() / n_valid),
        log_sharpness=float(np.log(cv2.Laplacian(gray, cv2.CV_32F)[valid].var() + 1e-6)),
        mean_gray=float(gray[valid].mean()),
        mean_sat=float(hsv[..., 1][valid].mean()),
    )


def model_free_maps(rgb, sigma):
    """Trivial cue maps (no model). Orientation fixed a priori: higher = 'more lesion-like'."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    valid = gray > 10
    E = gray.shape[0]
    sc = E / 518.0
    lab = rgb2lab(rgb)
    ys, xs = np.nonzero(valid)
    cy, cx = ys.mean(), xs.mean()
    yy, xx = np.mgrid[0:E, 0:E]
    prior = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (0.25 * E) ** 2)))
    blur = cv2.GaussianBlur(gray, (0, 0), max(0.5, 1.5 * sc))
    grad = np.sqrt(cv2.Sobel(blur, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(blur, cv2.CV_32F, 0, 1) ** 2)
    maps = {
        "free_center": prior.astype(np.float32),
        "free_redness": gauss(lab[..., 1].astype(np.float32), sigma),
        "free_smooth": gauss(-grad, sigma),
        "free_bright": gauss(gray, sigma),
        "free_sat": gauss(hsv[..., 1], sigma),
    }
    return maps, gray, valid


def failure_typing(b, gt, gray, valid):
    """Where does the (baseline) map fire, and how well does it cover the lesion?"""
    E = b.shape[0]
    sc = E / 518.0
    top = b >= np.quantile(b, 0.99)
    spec = binary_dilation(gray > 235, iterations=max(1, int(round(5 * sc))))
    dark = (gray < 50) & valid
    border = valid & (distance_transform_edt(valid) < 20 * sc)
    k = int(gt.sum())
    thr_k = np.partition(b.ravel(), -k)[-k]
    pred = b >= thr_k
    dice = 2.0 * (pred & gt).sum() / (pred.sum() + gt.sum())
    nb = norm01(b)
    return dict(
        peak_in=float(gt.ravel()[np.argmax(b)]),
        top_in=float(gt[top].mean()),
        recall10=float((b[gt] >= np.quantile(b, 0.9)).mean()),
        dice_oracle=float(dice),
        top_spec=float(spec[top].mean()),
        top_dark=float(dark[top].mean()),
        top_border=float(border[top].mean()),
        top_outfov=float((~valid)[top].mean()),
        dyn_range=float(np.quantile(nb, 0.99) - np.quantile(nb, 0.5)),
    )


TYPING_COLS = ["peak_in", "top_in", "recall10", "dice_oracle", "top_spec", "top_dark", "top_border",
               "top_outfov", "dyn_range", "mean_gray", "mean_sat"]


def print_report(rows, model_variants, free_variants, base="scale518"):
    n = len(rows)
    la = np.array([r["log_area"] for r in rows])
    qid = np.digitize(la, np.quantile(la, [.25, .5, .75]))

    def arr(k):
        return np.array([r[k] for r in rows], dtype=float)

    table = {v: arr(v) for v in model_variants + free_variants}
    scale_names = [v for v in model_variants if v.startswith("scale")]
    table["ORACLE best-scale"] = np.max([table[v] for v in scale_names], 0)
    table["ORACLE best-of-all"] = np.max([table[v] for v in model_variants], 0)
    b = table[base]
    q_n = [int((qid == k).sum()) for k in range(4)]

    print(f"\n=== A. Mean per-image pixel-AUROC by lesion-area quartile (n={n}; Q1 smallest ... Q4 largest) ===")
    print(f"quartile sizes: {q_n}   baseline = {base}")
    print(f"{'variant':<22}{'ALL':>7}{'Q1':>7}{'Q2':>7}{'Q3':>7}{'Q4':>7}{'dALL':>8}{'dQ4':>8}")

    def line(name):
        a = table[name]
        cells = [a.mean()] + [a[qid == k].mean() for k in range(4)]
        d_all = a.mean() - b.mean()
        d_q4 = a[qid == 3].mean() - b[qid == 3].mean()
        print(f"{name:<22}" + "".join(f"{c:>7.3f}" for c in cells) + f"{d_all:>+8.3f}{d_q4:>+8.3f}")

    for v in model_variants:
        line(v)
    print("-" * 80)
    line("ORACLE best-scale")
    line("ORACLE best-of-all")
    print("-" * 80)
    for v in free_variants:
        line(v)

    cand = [v for v in model_variants if v != base]
    d_all = {v: table[v].mean() - b.mean() for v in cand}
    d_q4 = {v: table[v][qid == 3].mean() - b[qid == 3].mean() for v in cand}
    print("\nTop-5 variants by dALL :", ", ".join(f"{v} ({d_all[v]:+.3f})" for v in sorted(cand, key=lambda v: -d_all[v])[:5]))
    print("Top-5 variants by dQ4  :", ", ".join(f"{v} ({d_q4[v]:+.3f})" for v in sorted(cand, key=lambda v: -d_q4[v])[:5]))

    print(f"\n=== B. Where does the baseline map fire? (top-1% pixels; fractions may overlap) ===")
    print(f"{'group':<10}{'n':>5}" + "".join(f"{c:>13}" for c in TYPING_COLS))
    order = np.argsort(b)
    k20 = max(1, int(0.2 * n))
    groups = [("ALL", np.ones(n, bool))] + [(f"Q{k + 1}", qid == k) for k in range(4)]
    worst = np.zeros(n, bool)
    worst[order[:k20]] = True
    best = np.zeros(n, bool)
    best[order[-k20:]] = True
    groups += [("WORST20%", worst), ("BEST20%", best)]
    for name, sel in groups:
        vals = [arr(c)[sel].mean() for c in TYPING_COLS]
        print(f"{name:<10}{int(sel.sum()):>5}" + "".join(f"{v:>13.3f}" for v in vals))
    print("(peak_in: argmax inside lesion; top_in: share of top-1% pixels inside lesion; recall10: lesion pixels "
          "above the 90th pct; dice_oracle: Dice when thresholding at the true lesion area; top_spec/dark/border/"
          "outfov: share of top-1% pixels on glare / dark lumen / near FOV edge / outside FOV)")
