"""python test_pe_shift_stats.py  (numpy/scipy only; imports pos_shift_stats)"""
import numpy as np
import pe_shift_stats as S
import pos_shift_stats as P


def blob(cy, cx, r=20, n=518):
    y, x = np.mgrid[:n, :n]
    return (y - cy) ** 2 + (x - cx) ** 2 <= r * r


def test_patch_centroid_and_rec():
    # a lesion centred on the middle of patch (18, 18) -> centre of canvas -> zero shift
    assert S.patch_centroid(blob(258.5, 258.5)) == (18, 18) and S.rec_shift(blob(258.5, 258.5)) == (0, 0)
    m = blob(14 * 3 + 6.5, 14 * 30 + 6.5)             # patch (3, 30)
    assert S.patch_centroid(m) == (3, 30) and S.rec_shift(m) == (15, -12)


def test_direction_img_moves_lesion_to_centre_and_pe_matches():
    m = blob(14 * 3 + 6.5, 14 * 30 + 6.5, r=14)
    sy, sx = S.rec_shift(m)
    dx, dy = S.img_shift_px(sy, sx)
    assert (dx, dy) == (-12 * 14, 15 * 14)
    img = m.astype(float)[None]
    moved = P.shift_image_fill(img, dx, dy, 0.0)[0] > 0.5
    assert S.patch_centroid(moved) == (S.CENTRE, S.CENTRE)
    # PE arm: token (ci, cj) takes the PE of cell (ci + sy, cj + sx) = centre; same displacement as the image arm, same sign
    assert (3 + sy, 30 + sx) == (S.CENTRE, S.CENTRE)


def test_rand_shift():
    for sid in ("a", "b", "img_17.png", "zz"):
        for sy, sx in [(15, -12), (0, 7), (-4, 0), (3, 3), (0, 0)]:
            r = S.rand_shift(sid, sy, sx)
            assert (abs(r[0]), abs(r[1])) == (abs(sy), abs(sx))
            assert r == S.rand_shift(sid, sy, sx)                         # deterministic
            if (sy, sx) != (0, 0):
                assert r != (sy, sx)                                      # never equal to the recentre shift
    assert S.rand_shift("q", 0, 7)[1] == -7 and S.rand_shift("q", -4, 0)[0] == 4   # one nonzero axis: opposite direction
    pats = {S.rand_shift(str(k), 5, 5) for k in range(200)}
    assert pats == {(-5, 5), (5, -5), (-5, -5)}                           # the three admissible patterns all occur


def test_region():
    g_ok = np.ones((37, 37), bool)
    g = g_ok.copy(); g[:10] = False
    reg = S.common_region((518, 518), [g_ok, g], [(0, 0), (28, -56)])
    assert not reg[:140].any()                                            # PE-invalid rows
    assert not reg[:56].any() and not reg[:, 490:].any()                  # image shift (dy=-56 loses the top 56 rows? valid y in [56, 518)) and dx=28 loses the left
    ref = P.common_valid((518, 518), [(0, 0), (28, -56)]) & np.kron(g, np.ones((14, 14), bool))
    assert np.array_equal(reg, ref)


def test_stats_known_answer():
    rng = np.random.RandomState(1)
    n = 300
    s = rng.rand(n) * 0.5 + 0.2
    ir = s + 0.05                                    # D_IMG = 0.05
    pr = s + 0.04; pn = s - 0.01                     # D_PE = 0.04 (R = 0.8), G = 0.05
    st = S.boot_ratio_g(s, pr, pn, ir, 500)
    assert abs(st["D_IMG"] - 0.05) < 1e-12 and abs(st["R"] - 0.8) < 1e-12 and abs(st["G"] - 0.05) < 1e-12
    assert st["G_ci"][0] > 0 and abs(st["R_ci"][0] - 0.8) < 1e-9
    assert S.set_pass_pe(st) and not S.set_pass_not(st)
    # null: PE arms equal the sham with noise only -> R near 0, G CI covers 0
    nz = rng.randn(n) * 0.02
    st0 = S.boot_ratio_g(s, s + nz, s + rng.randn(n) * 0.02, s + 0.05 + rng.randn(n) * 0.01, 500)
    assert abs(st0["R"]) < 0.2 and st0["G_ci"][0] < 0 < st0["G_ci"][1]


def mk(d_img, r, g_lo=0.01):
    return dict(D_IMG=d_img, D_PE=r * d_img, D_PErand=0.0, R=r, R_ci=(0, 0), G=g_lo + 0.01, G_ci=(g_lo, g_lo + 0.02), n=10)


def test_rule_boundaries():
    V = S.verdict
    assert V({"ClinicDB": mk(0.05, 0.50), "ColonDB": mk(0.05, 0.50), "Endo": mk(0.05, 0.1)})[0] == "PE-MEDIATED"        # R exactly 0.50
    assert V({"ClinicDB": mk(0.05, 0.4999), "ColonDB": mk(0.05, 0.50), "Endo": mk(0.05, 0.1)})[0] == "MIXED"
    assert V({"ClinicDB": mk(0.05, 0.9, g_lo=0.0), "ColonDB": mk(0.05, 0.9, g_lo=-0.01), "Endo": mk(0.05, 0.9)})[0] == "MIXED"  # G lower bound must be > 0
    assert V({"ClinicDB": mk(0.05, 0.20), "ColonDB": mk(0.05, 0.20), "Endo": mk(0.05, 0.9)})[0] == "NOT PE-MEDIATED"   # R exactly 0.20
    assert V({"ClinicDB": mk(0.05, 0.2001), "ColonDB": mk(0.05, 0.20), "Endo": mk(0.05, 0.9)})[0] == "MIXED"
    # D_IMG exactly 0.020 is VALID, 0.0199999 is not
    assert V({"ClinicDB": mk(0.020, 0.0), "ColonDB": mk(0.020, 0.0), "Endo": mk(0.05, 0.9)})[0] == "NOT PE-MEDIATED"
    assert V({"ClinicDB": mk(0.0199999, 0.0), "ColonDB": mk(0.020, 0.0), "Endo": mk(0.0199999, 0.9)})[0] == "INCONCLUSIVE"
    assert V({"ClinicDB": mk(0.0199999, 0.0), "ColonDB": mk(0.020, 0.0), "Endo": mk(0.05, 0.9)})[0] == "MIXED"           # 2 valid sets, one each way
    # fewer than 2 valid decision sets, missing sets
    assert V({"ClinicDB": mk(0.05, 0.9), "ColonDB": mk(0.01, 0.9), "Endo": None})[0] == "INCONCLUSIVE"
    assert V({"ClinicDB": mk(0.05, 0.9)})[0] == "INCONCLUSIVE"
    # an invalid set never counts for either side; report-only sets are ignored
    assert V({"ClinicDB": mk(0.05, 0.9), "ColonDB": mk(0.05, 0.9), "Endo": mk(0.001, 0.0), "Kvasir": mk(0.05, 0.0)})[0] == "PE-MEDIATED"


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
    print("ALL PASSED")
