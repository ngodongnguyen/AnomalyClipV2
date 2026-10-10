"""Unit tests (known answers, numpy/scipy only) for badcase_stats.py.  Run: python test_badcase_stats.py"""
import numpy as np
import badcase_stats as B


def test_threshold_tpr():
    pos = np.arange(1, 101, dtype=float)             # 1..100
    t = B.threshold_at_tpr(pos, 0.8)
    assert t == 21.0 and (pos >= t).mean() >= 0.8 and (pos >= t + 1).mean() < 0.8
    assert B.threshold_at_tpr(np.array([5.0]), 0.8) == 5.0
    ties = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2.0])  # 0.8 of 10 = 8 needed -> t = 1 (all ten >= 1)
    assert B.threshold_at_tpr(ties, 0.8) == 1.0


def test_components_blob():
    h = w = 100
    gt = np.zeros((h, w), bool); gt[10:20, 10:20] = True
    dist = B.distance_to_lesion(gt)
    score = np.zeros((h, w), np.float32)
    score[12:18, 12:18] = 1.0                        # hot lesion
    score[60:80, 60:80] = 0.9                        # far blob 20x20 = 400 px
    score[40:42, 80:82] = 0.9                        # small far blob 4 px (below min area)
    score[25:35, 12:22] = 0.9                        # near blob: within 28 px of the lesion -> not far
    fp, lab, comps = B.far_fp_components(score, 0.5, dist)
    assert len(comps) == 1 and comps[0]["area"] == 400
    assert abs(comps[0]["cy"] - 69.5) < 1e-9 and abs(comps[0]["cx"] - 69.5) < 1e-9
    assert fp[60:80, 60:80].all() and not fp[25:35, 12:22].any()
    assert abs(B.border_dist_frac(69.5, 69.5, h, w) - 0.295) < 1e-9
    assert abs(B.border_dist_frac(3.0, 50.0, h, w) - 0.03) < 1e-9


def test_pct_and_hit():
    gt = np.zeros((10, 10), bool); gt[:2, :] = True        # 20 lesion px
    sc = np.arange(100, dtype=float).reshape(10, 10)       # lesion has the lowest scores 0..19
    assert abs(B.lesion_pct_median(sc, gt) - 0.105) < 1e-9  # median rank (10 or 11)/100
    sc2 = sc[::-1].copy()                                    # lesion has highest scores
    assert B.lesion_pct_median(sc2, gt) > 0.85


def test_spec_rule_and_hsv():
    rgb = np.full((20, 20, 3), 0.3)
    rgb[:5, :10] = 1.0                                       # 50 white px: V=1, S=0
    v, s = B.value_saturation(rgb)
    assert v[0, 0] == 1.0 and s[0, 0] == 0.0 and abs(s[10, 10] - 0.0) < 1e-12
    m = np.zeros((20, 20), bool); m[:10, :10] = True         # 100 px, 50 are specular -> 0.5
    assert abs(B.spec_fraction(v, s, m) - 0.5) < 1e-12
    m2 = np.zeros((20, 20), bool); m2[10:, 10:] = True       # none specular
    assert B.spec_fraction(v, s, m2) == 0.0
    red = np.zeros((4, 4, 3)); red[..., 0] = 0.8; red[..., 1] = 0.2; red[..., 2] = 0.4
    assert abs(B.redness(red)[0, 0] - 0.5) < 1e-12
    sat = np.zeros((4, 4, 3)); sat[..., 0] = 1.0             # pure red: S=1 -> not specular
    v, s = B.value_saturation(sat)
    assert s[0, 0] == 1.0


def test_sobel_edge():
    img = np.zeros((20, 20, 3)); img[:, 10:] = 1.0
    g = B.sobel_magnitude(img)
    assert g[:, 9:11].mean() > 1.0 and g[:, :5].max() == 0.0


def test_top_decile():
    v = np.arange(20, dtype=float); v[3] = np.nan
    m = B.top_decile_mask(v)                                 # 19 finite -> k = 2 -> values 19, 18
    assert m.sum() == 2 and m[19] and m[18]
    assert B.top_decile_mask(np.full(10, 1.0)).sum() == 1    # ties: exactly k, earliest index
    assert B.top_decile_mask(np.full(5, np.nan)).sum() == 0


def test_ratio_counts():
    assert B.ratio_from_counts(0, 10, 0, 90) == 1.0
    assert abs(B.ratio_from_counts(5, 10, 9, 90) - 5.0) < 1e-12
    exp = ((0 + .5) / 11) / ((9 + .5) / 91)                  # one zero cell -> +0.5 on all four cells
    assert abs(B.ratio_from_counts(0, 10, 9, 90) - exp) < 1e-12
    exp2 = ((5 + .5) / 11) / ((0 + .5) / 91)
    assert abs(B.ratio_from_counts(5, 10, 0, 90) - exp2) < 1e-12


def test_planted_effect_and_null():
    rng = np.random.RandomState(1)
    n = 400
    auroc = rng.uniform(0.5, 1.0, n)
    worst = B.worst_set(auroc)
    assert worst.sum() == 40
    planted = np.where(worst, rng.rand(n) < 0.7, rng.rand(n) < 0.1)     # 70% vs 10% -> ratio ~7
    null = rng.rand(n) < 0.2
    T = np.stack([planted, null], 1)
    res = B.analyse_set(auroc, T, B=500)
    p = B.passes(res)
    assert res["ratio"][0] > 4 and res["lo"][0] > 2.0 and p[0]
    assert not p[1] and res["lo"][1] < 1.5
    # permutation control: shuffled AUROC removes the planted effect
    perm = np.random.RandomState(0).permutation(auroc)
    res2 = B.analyse_set(perm, T, B=500)
    assert not B.passes(res2)[0]


def test_gate_counts():
    tags = ("SPEC", "DARK", "SMALL")
    ps = {"a": np.array([1, 0, 1], bool), "b": np.array([1, 0, 1], bool), "c": np.array([1, 1, 1], bool), "d": np.array([0, 0, 0], bool)}
    cnt, cand = B.candidate_verdict(ps, tags)
    assert cnt == {"SPEC": 3, "DARK": 1, "SMALL": 3} and cand == ["SPEC"]       # SMALL is descriptive, never a candidate


def test_worst_largest_and_ties():
    v = np.array([0.5, 0.5, 0.5, 0.9, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5])
    w = B.worst_set(v)
    assert w.sum() == 1 and w[4]
    w2 = B.worst_set(v, largest=True)
    assert w2.sum() == 1 and w2[3]


def test_image_record_and_tags():
    h = w = 100
    rng = np.random.RandomState(0)
    gt = np.zeros((h, w), bool); gt[10:20, 10:20] = True
    rgb = np.full((h, w, 3), 0.4)
    rgb[60:80, 60:80] = 1.0                                   # specular far blob
    score = rng.rand(h, w).astype(np.float32) * 0.1
    score[10:20, 10:20] = 0.8
    score[60:80, 60:80] = 0.9
    rec, fp, lab, comps = B.image_record(rgb, score, gt, 0.5)
    assert rec["n_comp"] == 1 and rec["comp_area"] == 400 and rec["hit_rate"] == 1.0
    assert rec["comp_spec"] == 1.0 and abs(rec["fp_frac"] - 0.04) < 1e-12 and rec["lesion_pct"] > 0.9
    # clean image: no far-FP
    score2 = score.copy(); score2[60:80, 60:80] = 0.05
    rec2, *_ = B.image_record(rgb, score2, gt, 0.5)
    assert rec2["n_comp"] == 0 and np.isnan(rec2["comp_area"])
    cols = {k: np.array([rec[k], rec2[k]]) for k in B.REC_COLS if k not in ("id", "quartile", "auroc")}
    cols["quartile"] = np.array([0, 3])
    t = B.assign_tags(cols)
    assert t["SPEC"].tolist() == [True, False] and t["MULTI"].tolist() == [False, False]
    assert t["BIGFP"].tolist() == [True, False]                # 400 >= 2 * 100
    assert t["COLDLESION"].tolist() == [False, False]
    assert t["BORDER"].tolist() == [False, False] and t["SMALL"].tolist() == [True, False] and t["LARGE"].tolist() == [False, True]
    assert t["EDGEY"].tolist() == [True, False]                 # one component -> it is the top decile; none without a component
    dark = dict(cols); dark["comp_v"] = np.array([0.1, np.nan])
    assert B.assign_tags(dark)["DARK"].tolist() == [True, False]
    cold = dict(cols); cold["lesion_pct"] = np.array([0.4, 0.5])
    assert B.assign_tags(cold)["COLDLESION"].tolist() == [True, True]   # image-level tag, also for the no-component image


def test_image_level():
    ids = ["b", "a", "c", "d"]
    sc = np.array([0.9, 0.9, 0.1, 0.5])
    assert B.deterministic_rank(ids, sc).tolist() == [2, 1, 4, 3]          # tie between a and b resolved by id
    lab = np.array([1, 1, 0, 0])
    c = B.youden_counts(lab, sc)
    assert (c["TP"], c["FP"], c["FN"], c["TN"]) == (2, 0, 0, 2)
    # overlapping distributions
    y = np.array([1] * 5 + [0] * 5)
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.3, 0.7, 0.4, 0.3, 0.2, 0.1])
    r = B.score_summary(y, s)
    assert abs(r["median_anomalous"] - 0.7) < 1e-12 and abs(r["median_normal"] - 0.3) < 1e-12
    assert r["TP"] + r["FN"] == 5 and r["FP"] + r["TN"] == 5


if __name__ == "__main__":
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f(); n += 1
    print(f"OK ({n} tests)")
