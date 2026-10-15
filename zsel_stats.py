"""
EXP-052 statistics (numpy only, no torch): source-only operating points of the extent axis, and the rules that judge them.
Rule text: notes/EXP-052_registration.md (copy into research/EXPERIMENTS.md before any run).

  z_pix rule: the 95th percentile of the extent labels z that the conditioner saw during training (the training mixture of native frames and zoom crops of
              the SOURCE anomalies, simulated with the exact training crop code; no model, no target data).
  z_img rule: the z of a fixed grid with the highest macro (per-category mean) image AUROC on the OTHER industrial dataset of the AnomalyCLIP protocol
              (VisA for an MVTec-trained checkpoint, MVTec for a VisA-trained one); ties go to the smaller z. No medical data.
"""
import numpy as np

Z_GRID = (-0.69, -0.3, 0.0, 0.4, 0.8, 1.2, 1.6)      # fixed before any scan
Q_PIX = 95                                           # percentile of the training-mixture z


def z_pix_rule(z_mixture, q=Q_PIX):
    z = np.asarray(z_mixture, float)
    if z.size == 0 or not np.isfinite(z).all():
        raise ValueError("empty or non-finite z sample")
    return float(np.percentile(z, q))


def auroc(labels, scores):
    """Exact rank AUROC (ties averaged). NaN when one class is absent."""
    y = np.asarray(labels, int); s = np.asarray(scores, float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ss = s[order]
    i = 0
    while i < len(ss):                      # average ranks over ties
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def macro_auroc(categories, labels, scores):
    """Mean over categories of the per-category image AUROC (test.py's 'mean' row); categories with one class only are skipped."""
    c = np.asarray(categories); y = np.asarray(labels); s = np.asarray(scores, float)
    vals = [auroc(y[c == k], s[c == k]) for k in sorted(set(c.tolist()))]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def select_z_img(macro_by_z, grid=Z_GRID):
    """macro_by_z: {z: macro image AUROC}. Highest value; ties (equal after rounding to 1e-9) go to the smaller z. Every grid z must be present."""
    if any(z not in macro_by_z or not np.isfinite(macro_by_z[z]) for z in grid):
        raise ValueError("incomplete scan")
    best = max(round(macro_by_z[z], 9) for z in grid)
    return min(z for z in grid if round(macro_by_z[z], 9) == best)


# ------------------------------------------------------------------ rules on the medical evaluation (differences in points)
def win(diffs, thr=1.0):
    """EXP-024 'win': mean over seeds >= thr AND positive in every seed."""
    d = np.asarray(diffs, float)
    return bool(round(d.mean(), 6) >= thr and (d > 0).all())


def r1_ecp_vs_zo(pro_diffs, auc_diffs):
    """EXP-024 R1 with the source-rule ECP. pro_diffs/auc_diffs: {set: [per-seed ECP_rule - ZO]} over the six pixel sets.
    CONFIRMED if wins on >= 4/6 for BOTH PRO and AUROC; NOT CONFIRMED if wins on <= 2/6 for either; otherwise INCONCLUSIVE."""
    wp = sum(win(v) for v in pro_diffs.values()); wa = sum(win(v) for v in auc_diffs.values())
    if wp >= 4 and wa >= 4:
        return "CONFIRMED", wp, wa
    if wp <= 2 or wa <= 2:
        return "NOT CONFIRMED", wp, wa
    return "INCONCLUSIVE", wp, wa


def r3_image(img_diffs):
    """EXP-024 R3 with the source-rule ECP. img_diffs: {image set: [per-seed ECP_rule - ZO AUROC]}. CONFIRMED if wins on >= 2/3, otherwise NOT CONFIRMED."""
    w = sum(win(v) for v in img_diffs.values())
    return ("CONFIRMED" if w >= 2 else "NOT CONFIRMED"), w


def r2_cost(pix_auc, pix_pro, img_auc):
    """Seed-mean (rule - deployed) differences in points. COSTS if any pixel AUROC or PRO <= -1.0 or any image AUROC <= -1.0 (a material loss
    anywhere); otherwise COMPARABLE (gains are allowed and reported)."""
    vals = list(pix_auc) + list(pix_pro) + list(img_auc)
    if any(v is None for v in vals) or len(pix_auc) != 6 or len(pix_pro) != 6 or len(img_auc) != 3:
        return "INCOMPLETE"
    return "COSTS" if any(round(v, 6) <= -1.0 for v in vals) else "COMPARABLE"


def visa_image_recovered(rule_img, ctrl_img):
    """VisA-trained ECP at the source-rule z_img vs the VisA-trained no-zoom control: RECOVERED if rule >= ctrl - 1.0 on >= 2/3 image sets."""
    n = sum(round(r - c, 6) >= -1.0 for r, c in zip(rule_img, ctrl_img))
    return ("RECOVERED" if n >= 2 else "NOT RECOVERED"), n
