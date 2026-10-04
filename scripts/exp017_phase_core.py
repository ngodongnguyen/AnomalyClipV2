"""Pure statistics and alignment for EXP-017's frozen-model phase diagnostic."""

import hashlib

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import rankdata
from scripts.exp014_readout_core import auc


def stable_key(sample_id):
    return hashlib.sha256(("EXP-017:111:" + sample_id).encode()).hexdigest()


def inverse_align(shifted_map, dx, dy):
    """Original (y,x) reads shifted-map (y+dy,x+dx); invalid edge is NaN."""
    h, w = shifted_map.shape
    x0, x1 = max(0, -dx), min(w, w - dx)
    y0, y1 = max(0, -dy), min(h, h - dy)
    out = np.full((h, w), np.nan, dtype=np.float64)
    out[y0:y1, x0:x1] = shifted_map[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return out


def common_interior(shape, border=14):
    mask = np.zeros(shape, dtype=bool)
    mask[border:shape[0]-border, border:shape[1]-border] = True
    return mask


def rank_percentile(values, valid):
    x = np.asarray(values)[valid]
    if not np.all(np.isfinite(x)):
        raise ValueError("nonfinite score in common interior")
    return rankdata(x, method="average") / len(x)


def phase_statistics(mask, maps, border=14):
    """Score fixed candidate maps and rank instability without GT-based tuning."""
    valid = common_interior(mask.shape, border)
    label = mask[valid]
    if not (label.any() and (~label).any()):
        raise ValueError("common interior lacks one mask class")
    original = rank_percentile(maps["identity"], valid)
    in_dist = distance_transform_edt(mask)
    out_dist = distance_transform_edt(~mask)
    boundary = ((mask & (in_dist <= 14)) | ((~mask) & (out_dist <= 14)))[valid]
    far = ((~mask) & (out_dist > 28))[valid]
    shifts = {}
    for group, scale in (("half", 7), ("full", 14)):
        aligned = [maps[f"{group}_{axis}{sign}"] for axis in ("x", "y")
                   for sign in ("pos", "neg")]
        diffs = np.stack([np.abs(rank_percentile(m, valid) - original) for m in aligned])
        shifts[group] = float(diffs.mean())
        shifts[f"{group}_boundary"] = float(diffs[:, boundary].mean()) if boundary.any() else None
        shifts[f"{group}_far"] = float(diffs[:, far].mean()) if far.any() else None
        maps[f"ensemble_{group}"] = np.mean([maps["identity"], *aligned], axis=0)
    scores = {}
    for name in ("identity", "ensemble_half", "ensemble_full", "smooth8", "smooth16", "smooth32"):
        scores[f"auc_{name}"] = float(auc(label, maps[name][valid]))
    return {**scores, **shifts, "mask_area_fraction": float(mask.mean()),
            "interior_mask_area_fraction": float(label.mean()),
            "boundary_pixels": int(boundary.sum()), "far_healthy_pixels": int(far.sum())}


def pilot_gate(rows_by_dataset, repetitions=2000, seed=111):
    """Registered three-set decision; CIs are bootstrapped over images."""
    rng = np.random.default_rng(seed)
    details = {}
    wins = []
    safe = []
    for name, rows in rows_by_dataset.items():
        if not rows:
            raise ValueError(f"{name}: no eligible rows")
        phase = np.array([r["half"] - r["full"] for r in rows])
        versus_full = np.array([r["auc_ensemble_half"] - r["auc_ensemble_full"] for r in rows])
        versus_smooth = np.array([r["auc_ensemble_half"] - r["auc_smooth32"] for r in rows])
        bootstrap = rng.integers(0, len(rows), (repetitions, len(rows)))
        lower = float(np.quantile(phase[bootstrap].mean(axis=1), 0.025))
        p, f, s = float(phase.mean()), float(versus_full.mean()), float(versus_smooth.mean())
        phase_pass = p >= 0.01 and lower > 0
        performance_pass = f >= 0.005 and s >= 0.005
        wins.append(phase_pass and performance_pass)
        safe.append(f >= -0.005 and s >= -0.005)
        details[name] = {"images": len(rows), "half_minus_full_rank_change": p,
                         "phase_bootstrap_95_lower": lower,
                         "half_minus_full_ensemble_auc": f,
                         "half_minus_sigma32_auc": s,
                         "phase_pass": phase_pass, "performance_pass": performance_pass}
    return {"pass": sum(wins) >= 2 and all(safe), "dataset_results": details,
            "rule": "phase>=.01 and bootstrap lower>0; both AUC gains>=.005 on >=2/3, no loss<-.005"}
