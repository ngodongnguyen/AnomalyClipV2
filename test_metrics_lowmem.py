"""Checks that the memory-light metrics give the same numbers as the original implementations (known answers + the old code).
sklearn is not installed locally, so a tiny stand-in with the same trapezoid `auc` is injected; run with the miniconda python."""
import sys, types, importlib.util, os
import numpy as np
from scipy.stats import rankdata
sk = types.ModuleType("sklearn"); skm = types.ModuleType("sklearn.metrics")
skm.auc = lambda x, y: float(np.trapezoid(y, x))
for n in ("roc_auc_score", "average_precision_score", "f1_score", "precision_recall_curve", "pairwise"): setattr(skm, n, None)
sk.metrics = skm; sys.modules["sklearn"] = sk; sys.modules["sklearn.metrics"] = skm
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
here = os.path.dirname(os.path.abspath(__file__))
new = load(os.path.join(here, "metrics.py"), "metrics_new")
from skimage import measure
from sklearn.metrics import auc

# --- the original cal_pro_score, copied verbatim from metrics.py before the change ---
def cal_pro_score_original(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc
# ---
class old: cal_pro_score = staticmethod(cal_pro_score_original)
rng = np.random.default_rng(0)
# 1) ROC AUC: exact tiny case, ties, and a rank-based reference
assert new.roc_auc_lowmem(np.array([1, 1, 0, 0]), np.array([.9, .8, .3, .1])) == 1.0
assert new.roc_auc_lowmem(np.array([1, 0]), np.array([.1, .9])) == 0.0
assert new.roc_auc_lowmem(np.array([1, 0]), np.array([.5, .5])) == 0.5
for trial in range(20):
    n = int(rng.integers(50, 4000)); y = (rng.random(n) < rng.uniform(0.05, 0.6)).astype(np.float32)
    if y.sum() in (0, n): continue
    s = np.round(rng.normal(size=n) + 1.2 * y, int(rng.integers(0, 3))).astype(np.float32)        # rounding creates many ties
    y64 = y.astype(np.float64); r = rankdata(s); ref = (r[y == 1].sum() - y64.sum() * (y64.sum() + 1) / 2) / (y64.sum() * (n - y64.sum()))
    assert abs(new.roc_auc_lowmem(y, s) - ref) < 1e-12, (trial, new.roc_auc_lowmem(y, s), ref)
try: new.roc_auc_lowmem(np.zeros(5), rng.random(5)); raise SystemExit("one-class input accepted")
except ValueError: pass
# 2) PRO: identical to the original on synthetic masks with several regions per image and some empty masks
for trial in range(6):
    N, H = 7, 36
    masks = np.zeros((N, H, H), dtype=np.float32)
    for i in range(N - 1):
        for _ in range(int(rng.integers(1, 4))):
            y0, x0 = rng.integers(0, H - 8, 2); h, w = rng.integers(3, 8, 2); masks[i, y0:y0 + h, x0:x0 + w] = 1
    amaps = (rng.random((N, H, H)) * 0.4 + 0.5 * masks + rng.normal(0, .05, (N, H, H))).astype(np.float32)
    amaps = np.round(amaps, 2) if trial % 2 else amaps
    a, b = old.cal_pro_score(masks, amaps), new.cal_pro_score(masks, amaps)
    assert a == b, (trial, a, b)
print("ALL METRICS-LOWMEM TESTS PASSED")
