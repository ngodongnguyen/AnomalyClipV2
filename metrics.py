from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
import numpy as np
from skimage import measure

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    # Same result as the original loop, but the connected regions of every mask (they do not depend on the threshold) are
    # labelled once instead of at all 200 thresholds, and 1 - masks is built once: faster and far less memory churn.
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    regions = [[(r.coords[:, 0], r.coords[:, 1], r.area) for r in measure.regionprops(measure.label(mask))] for mask in masks]
    inverse_masks = 1 - masks
    inverse_sum = inverse_masks.sum()
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, image_regions in zip(binary_amaps, regions):
            for rows, cols, area in image_regions:
                tp_pixels = binary_amap[rows, cols].sum()
                pro.append(tp_pixels / area)
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_sum
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc


def roc_auc_lowmem(y_true, y_score):
    """Exact binary ROC AUC (ties count one half), equal to sklearn's roc_auc_score, computed by sorting only the negative
    scores and counting with searchsorted. On a Kvasir-sized set (270 M pixels) sklearn's argsort-based version needs several GB
    more RAM, which is what got the evaluation killed by systemd-oomd."""
    y = np.asarray(y_true).ravel()
    sc = np.asarray(y_score).ravel()
    pos_mask = y > 0.5
    n_pos = int(pos_mask.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Only one class present in y_true. ROC AUC score is not defined in that case.")
    pos = sc[pos_mask]
    neg = sc[~pos_mask]
    neg.sort()
    left = np.searchsorted(neg, pos, side="left")
    right = np.searchsorted(neg, pos, side="right")
    return (left.sum(dtype=np.float64) + 0.5 * (right - left).sum(dtype=np.float64)) / (float(n_pos) * float(n_neg))


def image_level_metrics(results, obj, metric):
    gt = results[obj]['gt_sp']
    pr = results[obj]['pr_sp']
    gt = np.array(gt)
    pr = np.array(pr)
    if metric == 'image-auroc':
        performance = roc_auc_score(gt, pr)
    elif metric == 'image-ap':
        performance = average_precision_score(gt, pr)

    return performance
    # table.append(str(np.round(performance * 100, decimals=1)))


def pixel_level_metrics(results, obj, metric):
    gt = results[obj]['imgs_masks']
    pr = results[obj]['anomaly_maps']
    gt = np.asarray(gt)   # no copies of the (N, 518, 518) arrays
    pr = np.asarray(pr)
    if metric == 'pixel-auroc':
        performance = roc_auc_lowmem(gt.ravel(), pr.ravel())
    elif metric == 'pixel-aupro':
        if len(gt.shape) == 4:
            gt = gt.squeeze(1)
        if len(pr.shape) == 4:
            pr = pr.squeeze(1)
        performance = cal_pro_score(gt, pr)
    return performance
    