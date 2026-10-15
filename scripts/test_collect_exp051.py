"""Known-answer tests for scripts/collect_exp051.py rules.  python scripts/test_collect_exp051.py"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_exp051 as C

D = C.DECISION


def per_cen(per_means, cen_means, lo_pos=True):
    per = {s: (m, (m - 0.005) if lo_pos else -0.001, m + 0.01) for s, m in zip(D, per_means)}
    cen = {s: (m, m - 0.005, m + 0.005) for s, m in zip(D, cen_means)}
    return per, cen


def test_primary():
    ok6, ok3 = [0.0] * 6, [0.0] * 3
    per, cen = per_cen([0.02, 0.015, 0.001], [0.0, 0.0, 0.0])
    assert C.primary_verdict(per, cen, ok6, ok6, ok3)[0] == "ADVANCE"
    per, cen = per_cen([0.010, 0.010, 0.0], [0.0, 0.0, 0.0])            # exactly at the threshold counts
    assert C.primary_verdict(per, cen, ok6, ok6, ok3)[0] == "ADVANCE"
    per, cen = per_cen([0.02, 0.02, 0.02], [0.0, 0.0, 0.0], lo_pos=False)  # CI lower bound not > 0 -> no pass; not low either
    assert C.primary_verdict(per, cen, ok6, ok6, ok3)[0] == "INCONCLUSIVE"
    per, cen = per_cen([0.02, 0.02, 0.0], [0.0, -0.006, 0.0])           # central guard broken
    assert C.primary_verdict(per, cen, ok6, ok6, ok3)[0] == "INCONCLUSIVE"
    per, cen = per_cen([0.02, 0.02, 0.0], [0.0, -0.005, 0.0])           # exactly -0.005 is allowed
    assert C.primary_verdict(per, cen, ok6, ok6, ok3)[0] == "ADVANCE"
    per, cen = per_cen([0.02, 0.02, 0.0], [0.0, 0.0, 0.0])
    assert C.primary_verdict(per, cen, [0, 0, 0, 0, 0, -1.1], ok6, ok3)[0] == "INCONCLUSIVE"     # pooled guard
    assert C.primary_verdict(per, cen, ok6, ok6, [0, -0.99, 0])[0] == "ADVANCE"
    assert C.primary_verdict(per, cen, ok6, ok6, [0, -1.0, 0])[0] == "INCONCLUSIVE"             # image guard: -1.0 itself breaks it
    per, cen = per_cen([0.002, 0.0, 0.05], [0.0, 0.0, 0.0])
    assert C.primary_verdict(per, cen, ok6, ok6, ok3)[0] == "FAIL"                              # low on 2/3
    per, cen = per_cen([0.02, 0.02, 0.02], [0.0, 0.0, 0.0])
    assert C.primary_verdict(per, cen, [-1.0, -1.0, -1.0, 0, 0, 0], ok6, ok3)[0] == "FAIL"       # PRO <= -1 on 3/6
    assert C.primary_verdict(per, cen, [None] + ok6[1:], ok6, ok3)[0] == "INCOMPLETE"
    assert C.primary_verdict({D[0]: per[D[0]]}, cen, ok6, ok6, ok3)[0] == "INCOMPLETE"


def test_position_stats():
    ids = [f"i{k:03d}" for k in range(90)]
    c = {i: k / 90 for k, i in enumerate(ids)}                     # offset grows with k
    ecp = {i: 0.8 for i in ids}
    eb = {i: 0.8 + (0.05 if c[i] >= 60 / 90 else 0.0) for i in ids}   # gain only in the outer third
    st = C.position_stats(eb, ecp, c, n_boot=200)
    assert st["n"] == 90 and abs(st["per"][0] - 0.05) < 1e-12 and abs(st["cen"][0]) < 1e-12
    st = C.position_stats({i: eb[i] for i in ids[:3]}, ecp, c)
    assert st["per"] is None


def test_mech():
    mk = lambda q, lo: dict(n=50, G_hi=0.1, G_lo=0.1 * q, Q=q, dG=0.1 * (1 - q), dG_ci=(lo, 0.1))
    assert C.mech_set_label(mk(0.70, 0.01)) == "REDUCED"
    assert C.mech_set_label(mk(0.70, -0.01)) == "MIXED"
    assert C.mech_set_label(mk(0.90, 0.0)) == "UNCHANGED"
    assert C.mech_set_label(dict(n=50, G_hi=-0.01, Q=np.nan)) == "INVALID"
    L = lambda a, b, c: dict(zip(D, (a, b, c)))
    assert C.mech_verdict(L("REDUCED", "REDUCED", "UNCHANGED")) == "MECHANISM SUPPORTED"
    assert C.mech_verdict(L("UNCHANGED", "MIXED", "UNCHANGED")) == "MECHANISM NOT SUPPORTED"
    assert C.mech_verdict(L("REDUCED", "MIXED", "UNCHANGED")) == "MIXED"
    assert C.mech_verdict(L("REDUCED", "INVALID", "INVALID")) == "INCONCLUSIVE"
    assert C.mech_verdict({D[0]: "REDUCED", D[1]: None, D[2]: "REDUCED"}) == "INCOMPLETE"


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f):
            f(); print("ok", n)
    print("all collect_exp051 tests passed")
