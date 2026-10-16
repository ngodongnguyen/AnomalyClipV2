"""Known-answer tests for zsel_stats.py (EXP-052).  python test_zsel_stats.py"""
import numpy as np
import zsel_stats as S


def test_quantile():
    z = np.arange(0, 101, dtype=float)          # 0..100
    assert abs(S.z_pix_rule(z) - 95.0) < 1e-12
    assert abs(S.z_pix_rule([1.6] * 10) - 1.6) < 1e-12
    for bad in ([], [1.0, np.nan]):
        try:
            S.z_pix_rule(bad); raise AssertionError("bad input accepted")
        except ValueError:
            pass


def test_auroc():
    assert S.auroc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert S.auroc([0, 0, 1, 1], [0.4, 0.3, 0.2, 0.1]) == 0.0
    assert S.auroc([0, 1], [0.5, 0.5]) == 0.5                       # ties averaged
    assert S.auroc([0, 0, 1], [0.1, 0.5, 0.5]) == 0.75
    assert np.isnan(S.auroc([1, 1], [0.1, 0.2]))
    rng = np.random.RandomState(0)                                    # compare with the pairwise definition
    y = rng.randint(0, 2, 200); s = np.round(rng.rand(200), 2)
    pos, neg = s[y == 1], s[y == 0]
    ref = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg))
    assert abs(S.auroc(y, s) - ref) < 1e-12


def test_macro():
    c = ["a"] * 4 + ["b"] * 4 + ["c"] * 2
    y = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1]                                 # category c has one class only -> skipped
    s = [0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1, 0.5, 0.6]
    assert abs(S.macro_auroc(c, y, s) - 0.5) < 1e-12


def test_select():
    m = {z: 0.80 for z in S.Z_GRID}
    m[0.4] = 0.85; m[1.2] = 0.85
    assert S.select_z_img(m) == 0.4                                   # tie -> smaller z
    m[1.2] = 0.86
    assert S.select_z_img(m) == 1.2
    m2 = dict(m); del m2[-0.3]
    try:
        S.select_z_img(m2); raise AssertionError("incomplete scan accepted")
    except ValueError:
        pass


def test_rules():
    six = lambda v: {k: v for k in "abcdef"}
    assert S.win([1.0, 1.2, 0.8]) and not S.win([3.0, 3.0, -0.1]) and not S.win([0.9, 0.9, 0.9])
    pro = six([2.0, 2.0, 2.0]); auc = six([2.0, 2.0, 2.0])
    assert S.r1_ecp_vs_zo(pro, auc)[0] == "CONFIRMED"
    pro2 = {**pro, "a": [0.0] * 3, "b": [0.0] * 3, "c": [0.0] * 3}
    assert S.r1_ecp_vs_zo(pro2, auc)[0] == "INCONCLUSIVE"            # PRO wins 3/6
    pro3 = {**pro2, "d": [0.0] * 3}
    assert S.r1_ecp_vs_zo(pro3, auc)[0] == "NOT CONFIRMED"           # PRO wins 2/6
    assert S.r3_image({"x": [5, 5, 5], "y": [5, 5, 5], "z": [0, 0, 0]})[0] == "CONFIRMED"
    assert S.r3_image({"x": [5, 5, 5], "y": [0, 0, 0], "z": [0, 0, 0]})[0] == "NOT CONFIRMED"
    assert S.r2_cost([0.5] * 6, [-0.99] * 6, [0, -0.99, 0]) == "COMPARABLE"
    assert S.r2_cost([0.5] * 5 + [-1.5], [0.0] * 6, [0, 0, 0]) == "COSTS"           # one material loss is enough
    assert S.r2_cost([0.0] * 6, [0.0] * 6, [0, -1.0, 0]) == "COSTS"                 # -1.0 itself counts as a loss
    assert S.r2_cost([0.0] * 4 + [1.5, 1.5], [0.0] * 6, [0, 0, 0]) == "COMPARABLE"  # gains allowed
    assert S.r2_cost([None] + [0.0] * 5, [0.0] * 6, [0, 0, 0]) == "INCOMPLETE"
    assert S.visa_image_recovered([95.0, 90.0, 97.0], [95.7, 96.4, 97.5]) == ("RECOVERED", 2)
    assert S.visa_image_recovered([86.8, 75.8, 73.5], [95.7, 96.4, 97.5]) == ("NOT RECOVERED", 0)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f):
            f(); print("ok", n)
    print("all zsel_stats tests passed")
