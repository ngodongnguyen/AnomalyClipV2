"""Unit tests for attr_stats.py and the report stage of analyze_attr_gate.py (synthetic data). Run: python test_attr_stats.py"""
import os, csv, tempfile, subprocess, sys
import numpy as np
import attr_stats as S
import zpos_stats as Z

yy, xx = np.mgrid[:518, :518]


def disk(cy, cx, r):
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def test_solidity():
    d = S.solidity(disk(259, 259, 80))
    assert 0.97 <= d <= 1.0, d
    sq = np.zeros((518, 518), bool); sq[100:200, 100:300] = True
    assert abs(S.solidity(sq) - 1.0) < 1e-9
    cres = disk(259, 259, 100) & ~disk(259, 300, 90)         # crescent
    assert S.solidity(cres) < 0.75, S.solidity(cres)
    ring = disk(259, 259, 100) & ~disk(259, 259, 70)
    assert S.solidity(ring) < 0.6                              # ring: hull is the full disk, ~ 1 - (70/100)^2 = 0.51
    two = disk(100, 100, 30) | disk(400, 400, 30)              # two blobs
    assert S.solidity(two) < 0.2
    assert np.isnan(S.solidity(np.zeros((518, 518), bool)))
    line = np.zeros((518, 518), bool); line[100, 100:200] = True      # 1-px line: corners give a positive area, solidity ~1
    assert 0.99 <= S.solidity(line) <= 1.0
    print("solidity ok")


def test_inhom_bcm():
    m = disk(259, 259, 60)
    v = np.full((518, 518), 0.3); v[m] = 0.7
    assert S.inhomogeneity(v, m) < 1e-6                         # constant inside
    assert abs(S.boundary_contrast_mag(v, m) - 0.4) < 1e-9      # planted step
    v2 = np.full((518, 518), 0.7); v2[m] = 0.3
    assert abs(S.boundary_contrast_mag(v2, m) - 0.4) < 1e-9     # magnitude: sign ignored
    rng = np.random.RandomState(0)
    v3 = np.full((518, 518), 0.5); v3[m] = rng.rand(m.sum())
    assert S.inhomogeneity(v3, m) > 0.5
    full = np.ones((518, 518), bool)
    assert np.isnan(S.boundary_contrast_mag(v, full))           # empty ring
    print("inhom/bcm ok")


def test_partial_multi():
    rng = np.random.RandomState(1); n = 3000
    area, c = rng.rand(n), rng.rand(n)
    # (1) attribute and AUROC both driven only by the controls (area, c) plus noise -> ~0 (a near-noiseless nonlinear function of the controls can leak through a rank-linear partial: known limitation)
    attr = area + 0.3 * rng.randn(n); auc = -2 * area + c + 0.3 * rng.randn(n)
    assert abs(S.partial_spearman_multi(attr, auc, area, c)) < 0.1
    # (1b) attribute an exact monotone function of area: nothing left after control -> NaN (never passes)
    assert np.isnan(S.partial_spearman_multi(np.exp(area), -area, area, c))
    # (2) pure noise -> ~0
    assert abs(S.partial_spearman_multi(rng.randn(n), rng.randn(n), area, c)) < 0.06
    # (3) planted dependence beyond controls -> clearly positive
    a = rng.randn(n); y = a + area + c + 0.5 * rng.randn(n)
    assert S.partial_spearman_multi(a, y, area, c) > 0.5
    # (4) one control equals zpos_stats.partial_spearman
    x, yv, z = rng.randn(n), rng.randn(n) + 0.3 * rng.randn(n), rng.randn(n)
    assert abs(S.partial_spearman_multi(x, yv, z) - Z.partial_spearman(x, yv, z)) < 1e-9
    # (5) bootstrap CI contains the point estimate and excludes 0 for the planted case
    r, lo, hi = S.boot_partial(a[:400], y[:400], area[:400] + 0.01, c[:400], n_boot=200)
    assert lo <= r <= hi and lo > 0
    print("partial ok")


def test_rules():
    # A: exactly 0.80 passes; 0.7999999 does not; 4/6 boundary
    assert S.check_A([0.8, 0.8, 0.8, 0.8, 0.1, 0.1])[0]
    assert not S.check_A([0.8, 0.8, 0.8, 0.79999, 0.9, 0.1])[0] or S.check_A([0.8, 0.8, 0.8, 0.79999, 0.9, 0.1])[1] == 4
    assert not S.check_A([0.8, 0.8, 0.8, 0.7, 0.1, 0.1])[0]
    # B: exactly 0.80 passes, 0.81 fails; needs MVTec too
    assert S.check_B(0.80, [0.8, -0.8, 0.8, -0.8, 0.9, 0.9])[0]
    assert not S.check_B(0.81, [0.1] * 6)[0]
    assert not S.check_B(0.5, [0.1, 0.1, 0.1, 0.9, 0.9, 0.9])[0]
    assert S.check_B(-0.8, [0.1] * 4 + [0.95] * 2)[0]
    # C: |rho| = 0.20 passes, CI must exclude 0, same sign on >= 4 sets
    P = (0.20, 0.05, 0.35)
    assert S.set_pass_C(P) == 1 and S.set_pass_C((-0.20, -0.35, -0.05)) == -1
    assert S.set_pass_C((0.19, 0.05, 0.35)) == 0 and S.set_pass_C((0.30, -0.01, 0.5)) == 0
    assert S.check_C([P] * 4 + [(0.0, -0.1, 0.1)] * 2)[0]
    assert not S.check_C([P] * 3 + [(-0.3, -0.4, -0.2)] * 3)[0]       # mixed sign, 3 + 3
    assert S.check_C([(-0.3, -0.4, -0.2)] * 4 + [P] * 2)[0]
    assert S.verdict_attribute(True, True, True) == "CANDIDATE" and S.verdict_attribute(True, False, True) == "DROPPED"
    assert S.verdict_direction({"a": "DROPPED", "b": "DROPPED"}) == "CLOSE"
    assert S.verdict_direction({"a": "CANDIDATE", "b": "DROPPED"}).startswith("CANDIDATES")
    # support fraction
    mv = np.arange(101.0)
    assert S.support_fraction(mv, [5, 50, 95, 96, 4]) == 0.6
    print("rules ok")


def test_report_end_to_end():
    """Synthetic EXP-032/EXP-047 directory: one planted attribute (inhom drives AUROC), run `report`, check verdict lines."""
    rng = np.random.RandomState(0)
    d = tempfile.mkdtemp()
    sets = ["ClinicDB", "ColonDB", "ISIC", "Endo", "Kvasir", "TN3K"]
    mv = d + "/mv"; os.makedirs(mv + "/MVTec")
    n = 400
    with open(mv + "/MVTec/attr_MVTec.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "category", "area_frac", "solidity", "bcm", "inhom"])
        for i in range(n):
            w.writerow([f"m{i}", "cat%d" % (i % 3), rng.uniform(0.01, 0.3), rng.uniform(0.5, 1), rng.uniform(0, 0.5), rng.uniform(0.2, 1.5)])
    for s in sets:
        os.makedirs(f"{d}/{s}")
        m = 500
        area = rng.uniform(0.01, 0.3, m); c = rng.rand(m); sol = rng.uniform(0.5, 1, m); bcm = rng.uniform(0, 0.5, m); inh = rng.uniform(0.2, 1.5, m)
        auc = 0.9 - 0.1 * inh + 0.02 * rng.randn(m)
        with open(f"{d}/{s}/badcase_{s}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["id", "auroc", "area_frac", "hit_rate", "comp_cy", "comp_cx"])
            for i in range(m): w.writerow([f"{s}{i}", auc[i], area[i], 0.5, 259, 259])
        with open(f"{d}/{s}/position_{s}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["id", "c", "contrast"])
            for i in range(m): w.writerow([f"{s}{i}", c[i], 0.0])
        with open(f"{d}/{s}/attr_{s}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["id", "area_frac", "solidity", "bcm", "inhom"])
            for i in range(m): w.writerow([f"{s}{i}", area[i], sol[i], bcm[i], inh[i]])
    out = subprocess.run([sys.executable, "analyze_attr_gate.py", "report", "--attr_dir", d, "--mvtec_dir", mv, "--pos_dir", d, "--boot", "200"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    txt = out.stdout
    assert "inhom: CANDIDATE" in txt and "solidity: DROPPED" in txt and "bcm: DROPPED" in txt, txt
    assert "DIRECTION: CANDIDATES: inhom" in txt, txt
    print("report end-to-end ok")


if __name__ == "__main__":
    test_solidity(); test_inhom_bcm(); test_partial_multi(); test_rules(); test_report_end_to_end()
    print("ALL OK")
