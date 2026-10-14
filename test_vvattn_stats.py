"""Known-answer tests for vvattn_stats.py (numpy/scipy only). Run: python test_vvattn_stats.py"""
import numpy as np
import vvattn_stats as S


def const_table(n=50, base=0.80, **deltas):
    """table with constant per-image AUROCs: arm = base + delta (no variance -> CI collapses on the delta)."""
    t = {}
    for sg in (4, 32):
        t[f"base@{sg}"] = np.full(n, base)
    for arm, d in deltas.items():
        for sg in (4, 32):
            t[f"{arm}@{sg}"] = np.full(n, base + d)
    return t


def tables_for(delta_by_set, extra):
    out = {}
    for ds in S.SETS:
        d = delta_by_set[ds]
        out[ds] = const_table(**{**extra, "ent_all": d, "ent_last6": d, "sharp_all": 0.0,
                                 "head_top50": d, "head_top25": d,
                                 **{f"head_rand50_s{s}": 0.0 for s in S.RAND_SEEDS},
                                 **{f"head_rand25_s{s}": 0.0 for s in S.RAND_SEEDS}})
    return out


def test_hist_auroc():
    rng = np.random.RandomState(0)
    pos = rng.normal(1.0, 1, 4000); neg = rng.normal(0, 1, 6000)
    bins = np.linspace(-6, 7, 2001)
    P = np.histogram(pos, bins)[0]; N = np.histogram(neg, bins)[0]
    exact = S.auroc(np.r_[np.ones(4000), np.zeros(6000)], np.r_[pos, neg])
    assert abs(S.hist_auroc(P, N) - exact) < 2e-3
    assert S.hist_auroc([0, 0, 5], [4, 0, 0]) == 1.0           # separable
    assert S.hist_auroc([0, 5, 0], [0, 4, 0]) == 0.5           # same bin -> tie
    assert S.hist_auroc([5, 0], [0, 4]) == 0.0
    assert np.isnan(S.hist_auroc([0, 0], [1, 1]))              # no positives
    both = S.hist_auroc(np.stack([P, P[::-1]]), np.stack([N, N[::-1]]))   # vectorised over leading dim
    assert both.shape == (2,) and abs(both[0] - S.hist_auroc(P, N)) < 1e-12


def test_masks_and_ranking():
    sc = np.arange(S.N_PAIRS, dtype=float).reshape(S.N_LAYERS, S.N_HEADS)    # larger index = higher score
    sc[0, 0] = np.nan                                                          # NaN is ranked as 0.5 (uninformative)
    rk = S.ranking_from_scores(sc)
    assert len(rk) == 304 and len(set(rk)) == 304
    assert rk[0] == (24, 15) and rk[1] == (24, 14)
    top50 = S.mask_from_ranking(rk, "top", 152); bot50 = S.mask_from_ranking(rk, "bottom", 152)
    assert top50.sum() == 152 and bot50.sum() == 152 and not (top50 & bot50).any()
    assert S.mask_from_ranking(rk, "top", 76).sum() == 76
    assert (S.mask_from_ranking(rk, "top", 76) & ~top50).sum() == 0            # top25 is nested in top50
    assert top50[-1, -1] and not top50[0, 1]
    r0, r0b, r1 = S.random_mask(152, 0), S.random_mask(152, 0), S.random_mask(152, 1)
    assert r0.sum() == 152 and np.array_equal(r0, r0b) and not np.array_equal(r0, r1)
    assert S.random_mask(76, 2).sum() == 76
    assert S.image_list_sha256(["a", "b"]) == S.image_list_sha256(["a", "b"]) != S.image_list_sha256(["b", "a"])


def test_paired_ci():
    m, lo, hi = S.paired_ci(np.full(30, 0.01))
    assert abs(m - 0.01) < 1e-12 and abs(lo - 0.01) < 1e-12 and abs(hi - 0.01) < 1e-12
    rng = np.random.RandomState(1)
    m, lo, hi = S.paired_ci(rng.normal(0, 0.05, 400))
    assert lo < 0 < hi


def test_arm_boundaries():
    ok = {"delta": 0.003, "lo": 1e-6, "dctrl": 0.003}
    assert S.decide_arm({"A": ok, "B": ok, "C": {"delta": 0.0, "lo": -1, "dctrl": 0}})[0]          # exactly the thresholds, 2/3 sets
    assert S.decide_arm({"A": {**ok, "delta": 0.0029999}, "B": ok, "C": {"delta": 0.0, "lo": -1, "dctrl": 0}})[0] is False
    assert S.decide_arm({"A": {**ok, "delta": 0.00300000000004}, "B": ok, "C": ok})[0]            # rounding to 1e-10
    assert S.decide_arm({"A": {**ok, "lo": 0.0}, "B": ok, "C": {"delta": 0.0, "lo": -1, "dctrl": 0}})[0] is False   # CI lower must be > 0
    assert S.decide_arm({"A": {**ok, "dctrl": 0.0029}, "B": ok, "C": {"delta": 0.0, "lo": -1, "dctrl": 0}})[0] is False  # control margin
    assert S.decide_arm({"A": ok, "B": ok, "C": {"delta": -0.005, "lo": -1, "dctrl": 0}})[0] is False  # loss exactly -0.005 breaks
    assert S.decide_arm({"A": ok, "B": ok, "C": {"delta": -0.00499, "lo": -1, "dctrl": 0}})[0]
    assert S.decide_arm({"A": ok, "B": {"delta": 0.0, "lo": -1, "dctrl": 0}, "C": {"delta": 0.0, "lo": -1, "dctrl": 0}})[0] is False  # 1/3 only
    assert S.decide_arm({"A": {**ok, "dctrl": float("nan")}, "B": ok, "C": {"delta": 0.0, "lo": -1, "dctrl": 0}})[0] is False  # no control -> no pass


def test_family_advance_close():
    # head family passes (delta +0.01 on all sets vs random 0.0), entmax family is exactly equal to its control -> CLOSE
    dd = {ds: 0.01 for ds in S.SETS}
    t = tables_for(dd, {})
    for ds in S.SETS:                                     # make entmax arms equal to the sharp control
        for sg in (4, 32):
            for a in ("ent_all", "ent_last6"):
                t[ds][f"{a}@{sg}"] = t[ds][f"sharp_all@{sg}"]
    v = S.family_verdict(t, n_boot=200)
    assert v["verdict"] == {"entmax": "CLOSE", "head": "ADVANCE"}, v["verdict"]
    assert v["arms"]["head_top50"]["pass"] and not v["arms"]["ent_all"]["pass"]
    # everything at +0.01 over base but control equals the arm (random masks/sharp as good) -> both CLOSE (the control is the point of the rule)
    t2 = tables_for(dd, {})
    for ds in S.SETS:
        for sg in (4, 32):
            for s in S.RAND_SEEDS:
                t2[ds][f"head_rand50_s{s}@{sg}"] = t2[ds][f"head_top50@{sg}"]
                t2[ds][f"head_rand25_s{s}@{sg}"] = t2[ds][f"head_top25@{sg}"]
            t2[ds][f"sharp_all@{sg}"] = t2[ds][f"ent_all@{sg}"]
    v2 = S.family_verdict(t2, n_boot=200)
    assert v2["verdict"]["head"] == "CLOSE" and v2["verdict"]["entmax"] == "CLOSE"
    # ADVANCE needs only one arm: entmax ent_last6 good on 2/3 sets, no loss; ent_all bad
    dd3 = {"ClinicDB": 0.01, "Kvasir": 0.01, "ColonDB": 0.0}
    t3 = tables_for(dd3, {})
    for ds in S.SETS:
        for sg in (4, 32):
            t3[ds][f"ent_all@{sg}"] = t3[ds][f"base@{sg}"] - 0.01     # ent_all loses everywhere
    v3 = S.family_verdict(t3, n_boot=200)
    assert v3["verdict"]["entmax"] == "ADVANCE" and not v3["arms"]["ent_all"]["pass"] and v3["arms"]["ent_last6"]["pass"]
    # a loss on any set (<= -0.005) vetoes an otherwise passing arm
    dd4 = {"ClinicDB": 0.01, "Kvasir": 0.01, "ColonDB": -0.006}
    v4 = S.family_verdict(tables_for(dd4, {}), n_boot=200)
    assert v4["verdict"]["head"] == "CLOSE"
    # missing set
    assert S.family_verdict({"ClinicDB": t["ClinicDB"]})["verdict"] == "INCOMPLETE"


def test_random_mean_control():
    # head_rand* control is the per-image mean over the three seeds
    t = const_table(n=10, head_rand50_s0=0.0, head_rand50_s1=0.03, head_rand50_s2=0.0)
    assert abs(S.column(t, "head_rand50", 4).mean() - (0.8 + 0.01)) < 1e-12


def test_strict_control_reading():
    import vvattn_stats as V
    good = {"delta": 0.004, "lo": 0.001, "dctrl": 0.004}
    nctrl = {"delta": 0.004, "lo": 0.001, "dctrl": 0.001}
    # delta qualifies on 3 sets but the control margin only on 2 -> strict rule: FAIL
    ok, _ = V.decide_arm({"A": good, "B": good, "C": nctrl})
    assert not ok
    # delta qualifies on 2 sets and the control on both of them; third set neutral -> PASS
    neutral = {"delta": 0.001, "lo": -0.001, "dctrl": 0.0}
    ok, _ = V.decide_arm({"A": good, "B": good, "C": neutral})
    assert ok
    print("ok test_strict_control_reading")


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
    print("test_vvattn_stats: ALL OK")
