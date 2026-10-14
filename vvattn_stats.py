"""Torch-free statistics for EXP-045 (sparse V-V attention and source-selected head subsets).
numpy/scipy only; unit-tested in test_vvattn_stats.py. The decision rule below is the PRE-REGISTERED one (research/EXPERIMENTS.md EXP-045); do not change it."""
import hashlib
import os
import numpy as np
from scipy.stats import rankdata

N_LAYERS, N_HEADS = 19, 16            # dual-path V-V layers 6..24, 16 heads each
FIRST_LAYER = 6
N_PAIRS = N_LAYERS * N_HEADS          # 304
RAND_SEEDS = (0, 1, 2)
SETS = ("ClinicDB", "Kvasir", "ColonDB")

# ---- pre-registered thresholds (per-image AUROC on the 0..1 scale, sigma 4 decisive)
PASS_DELTA = 0.003      # delta4 = AUROC(arm) - AUROC(base) >= +0.003 with CI lower bound > 0
LOSS_DELTA = -0.005     # no set may have delta4 <= -0.005
CTRL_MARGIN = 0.003     # (arm - control) >= +0.003
MIN_SETS = 2            # on >= 2 of 3 sets
ARM_CONTROL = {"ent_all": "sharp_all", "ent_last6": "sharp_all", "head_top50": "head_rand50", "head_top25": "head_rand25"}
FAMILIES = {"entmax": ("ent_all", "ent_last6"), "head": ("head_top50", "head_top25")}
ARM_FRACTION = {"head_top50": 0.50, "head_top25": 0.25, "head_bot50": 0.50, "head_rand50": 0.50, "head_rand25": 0.25}


def r10(x):
    """Round to 1e-10 before every threshold comparison (the rule says so; avoids 0.0029999999999 style misses)."""
    return round(float(x), 10)


def auroc(y, s):
    y = np.asarray(y, bool).ravel(); s = np.asarray(s).ravel()
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def hist_auroc(pos_hist, neg_hist):
    """AUROC from two aligned histograms (bins ascending in score; same-bin pairs count 1/2). Vectorised over leading dims."""
    P = np.asarray(pos_hist, np.float64); N = np.asarray(neg_hist, np.float64)
    below = np.cumsum(N, axis=-1) - N
    num = (P * (below + 0.5 * N)).sum(-1)
    den = P.sum(-1) * N.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def image_list_sha256(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def ranking_from_scores(scores):
    """scores [19,16] (NaN -> 0.5, uninformative) -> list of (layer, head) sorted by descending score, ties by (layer, head)."""
    s = np.where(np.isnan(scores), 0.5, scores).reshape(-1)
    order = sorted(range(len(s)), key=lambda i: (-s[i], i))
    return [(FIRST_LAYER + i // N_HEADS, i % N_HEADS) for i in order]


def mask_from_ranking(ranking, which, n_keep):
    """bool [19,16] keep-mask. which: 'top' = the n_keep best, 'bottom' = the n_keep worst."""
    sel = ranking[:n_keep] if which == "top" else ranking[len(ranking) - n_keep:]
    m = np.zeros((N_LAYERS, N_HEADS), bool)
    for l, h in sel:
        m[l - FIRST_LAYER, h] = True
    return m


def random_mask(n_keep, seed):
    idx = np.random.RandomState(seed).permutation(N_PAIRS)[:n_keep]
    m = np.zeros(N_PAIRS, bool); m[idx] = True
    return m.reshape(N_LAYERS, N_HEADS)


def paired_ci(d, n_boot=2000, seed=0):
    d = np.asarray(d, np.float64)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), (n_boot, len(d)))
    means = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def column(table, arm, sigma):
    """per-image AUROC array for an arm; 'head_rand50'/'head_rand25' are the per-image mean over seeds 0,1,2."""
    if arm in ("head_rand50", "head_rand25"):
        return np.mean([table[f"{arm}_s{s}@{sigma}"] for s in RAND_SEEDS], axis=0)
    return table[f"{arm}@{sigma}"]


def evaluate_arm_set(table, arm, sigma=4, n_boot=2000):
    base = column(table, "base", sigma); a = column(table, arm, sigma)
    out = {}
    out["delta"], out["lo"], out["hi"] = paired_ci(a - base, n_boot)
    ctrl = ARM_CONTROL.get(arm)
    if ctrl is not None:
        out["dctrl"] = float(np.mean(a - column(table, ctrl, sigma)))
    else:
        out["dctrl"] = float("nan")
    return out


def decide_arm(per_set):
    """per_set: {set: {'delta','lo','dctrl'}} -> (PASS bool, detail). Rule (fixed, EXP-045; STRICT reading of "on those same sets"):
    D = sets with delta >= +0.003 AND CI lower > 0; the arm PASSES if |D| >= 2 AND (arm - control) >= +0.003 on EVERY set in D
    AND no set has delta <= -0.005."""
    in_d, ctrl_ok, loss = {}, {}, {}
    for ds, r in per_set.items():
        in_d[ds] = bool(r10(r["delta"]) >= PASS_DELTA and r10(r["lo"]) > 0)
        ctrl_ok[ds] = bool((not np.isnan(r["dctrl"])) and r10(r["dctrl"]) >= CTRL_MARGIN)
        loss[ds] = bool(r10(r["delta"]) <= LOSS_DELTA)
    d_sets = [ds for ds, v in in_d.items() if v]
    passed = len(d_sets) >= MIN_SETS and all(ctrl_ok[ds] for ds in d_sets) and not any(loss.values())
    counts = {ds: bool(in_d[ds] and ctrl_ok[ds]) for ds in per_set}
    return bool(passed), {"counts": counts, "in_delta_set": in_d, "ctrl_ok": ctrl_ok, "loss": loss}


def family_verdict(tables, sigma=4, n_boot=2000):
    """tables: {set: {col: array}}; returns dict with per-arm results, PASS flags, per-family and overall verdict."""
    missing = [s for s in SETS if s not in tables]
    if missing:
        return {"verdict": "INCOMPLETE", "missing": missing}
    res = {}
    for fam, arms in FAMILIES.items():
        for arm in arms:
            per_set = {ds: evaluate_arm_set(tables[ds], arm, sigma, n_boot) for ds in SETS}
            ok, det = decide_arm(per_set)
            res[arm] = {"per_set": per_set, "pass": ok, **det}
    fam_out = {fam: ("ADVANCE" if any(res[a]["pass"] for a in arms) else "CLOSE") for fam, arms in FAMILIES.items()}
    return {"verdict": fam_out, "arms": res}


# ---------------------------------------------------------------------------------------------------------------------------------------
# EXP-046: head-mask files for training/testing (torch-free; unit-tested in test_vvattn_stats.py)
# Layout (same as vvattn_core.head_mask_dict and the EXP-045 masks): keep[i, h], i = 0..18 <-> V-V block (layer) 6+i, h = 0..15, True = head kept.
MASK_LAYOUT = "keep[i,h]: i=0..18 -> V-V layer 6+i (1-based block index), h=0..15 head, 1 = kept, 0 = V-V head output zeroed before proj"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_keep(keep):
    """-> bool array [19,16]; raises ValueError unless the shape is exactly (19, 16) and the values are 0/1 (304 entries)."""
    a = np.asarray(keep)
    if a.shape != (N_LAYERS, N_HEADS):
        raise ValueError(f"head mask must have shape ({N_LAYERS}, {N_HEADS}) = {N_PAIRS} entries, got {a.shape}")
    if not np.isin(a, (0, 1)).all():
        raise ValueError("head mask values must be 0/1 or bool")
    return a.astype(bool)


def keep_sha256(keep):
    return hashlib.sha256(validate_keep(keep).astype(np.uint8).tobytes()).hexdigest()


def load_keep(path):
    """-> (keep bool [19,16], sha256 of the file bytes)."""
    return validate_keep(np.load(path, allow_pickle=False)), sha256_file(path)


def mask_record(path, keep, file_sha):
    """Plain-python record stored in the checkpoint (lists/ints/str only, loadable with torch.load(weights_only=True))."""
    keep = validate_keep(keep)
    return {"path": os.path.abspath(path), "sha256": file_sha, "keep_sha256": keep_sha256(keep), "n_keep": int(keep.sum()),
            "layout": MASK_LAYOUT, "keep": keep.astype(int).tolist()}


def resolve_test_mask(flag_path, ckpt_record):
    """Mask used at evaluation. Returns (keep or None, record or None).
    no flag, no record -> (None, None): original behaviour. record only -> the checkpoint's own mask (a trained checkpoint can never silently run unmasked).
    flag only -> ValueError (the checkpoint was trained without a mask). both -> the file must equal the recorded mask, else ValueError."""
    if ckpt_record is None:
        if flag_path:
            raise ValueError("--vv_head_mask was given but the checkpoint was trained without a V-V head mask")
        return None, None
    keep = validate_keep(ckpt_record["keep"])
    if keep_sha256(keep) != ckpt_record["keep_sha256"]:
        raise ValueError("checkpoint head-mask record is internally inconsistent (keep_sha256)")
    if flag_path:
        fkeep, fsha = load_keep(flag_path)
        if not np.array_equal(fkeep, keep):
            raise ValueError(f"--vv_head_mask {flag_path} (sha256 {fsha[:12]}) differs from the mask stored in the checkpoint "
                             f"(sha256 {ckpt_record['sha256'][:12]}, keep_sha256 {ckpt_record['keep_sha256'][:12]})")
    return keep, ckpt_record
