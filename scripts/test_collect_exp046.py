import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_exp046 as m

OK_P = [1.0, 1.5, 2.0, 1.0, 0.0, 0.0]; OK_A = [0.3, 0.3, 0.5, 0.4, 0.0, 0.0]; OK_I = [0.0, -0.99, 1.0]; OK_R = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
d = m.decide
assert d(OK_P, OK_A, OK_I, OK_R)[0] == "ADVANCE"                                    # every threshold exactly at its boundary
assert d(OK_P, OK_A, OK_I, [0.99, 1.0, 1.0, 1.0, 0.0, 0.0])[0] == "INCONCLUSIVE"    # only 3/6 control wins
assert d(OK_P, OK_A, OK_I, [-5.0] * 6)[0] == "INCONCLUSIVE"                        # RAND50 matches/beats HEAD50: never FAIL by itself
assert d([0.99] + OK_P[1:], OK_A, OK_I, OK_R)[0] == "INCONCLUSIVE"                  # 3/6 PRO wins
assert d(OK_P, [0.29] + OK_A[1:], OK_I, OK_R)[0] == "INCONCLUSIVE"                  # 3/6 AUROC wins
assert d(OK_P[:5] + [-1.0], OK_A, OK_I, OK_R)[0] == "INCONCLUSIVE"                  # PRO change exactly -1.0 blocks ADVANCE
assert d(OK_P, OK_A[:5] + [-1.0], OK_I, OK_R)[0] == "INCONCLUSIVE"                  # AUROC change exactly -1.0 blocks ADVANCE
assert d(OK_P, OK_A, [0.0, 0.0, -1.0], OK_R)[0] == "INCONCLUSIVE"                   # image AUROC exactly -1.0 blocks ADVANCE
assert d(OK_P, OK_A, [0.0, 0.0, -0.99], OK_R)[0] == "ADVANCE"
assert d([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.5] * 6, OK_I, OK_R)[0] == "FAIL"        # PRO wins <= 2
assert d([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.5] * 6, OK_I, OK_R)[0] == "INCONCLUSIVE"
assert d([2.0] * 6, [0.5, 0.5, 0.5, -0.5, -0.5, -0.5], OK_I, OK_R)[0] == "FAIL"     # AUROC <= -0.5 on exactly 3 sets
assert d([2.0] * 6, [0.5, 0.5, 0.5, 0.5, -0.5, -0.5], OK_I, OK_R)[0] == "ADVANCE"
assert d([2.0] * 6, [0.5, 0.5, 0.5, -0.49, -0.49, -0.49], OK_I, OK_R)[0] == "INCONCLUSIVE"
assert d([2.0] * 6, [0.29, 0.29, 0.29, 0.29, 0.5, 0.5], OK_I, OK_R)[0] == "INCONCLUSIVE"
assert d(OK_P[:5] + [None], OK_A, OK_I, OK_R)[0] == "INCOMPLETE"
assert d(OK_P, OK_A, [0.0, None, 0.0], OK_R)[0] == "INCOMPLETE"
assert d(OK_P, OK_A, OK_I, OK_R[:5] + [None])[0] == "INCOMPLETE"                    # ADVANCE needs the control
assert d([0.0] * 6, [0.5] * 6, OK_I, [None] * 6)[0] == "FAIL"                       # FAIL clause does not need the control
assert d(OK_P, OK_A, OK_I, OK_R)[1]["pro_wins_vs_rand50"] == 4
assert d([1.0 - 1e-9] + OK_P[1:], OK_A, OK_I, OK_R)[0] == "ADVANCE"                 # rounding to 1e-6
assert d(OK_P, OK_A, OK_I, [1.0 - 1e-9] + OK_R[1:])[0] == "ADVANCE"
assert d(OK_P, OK_A, [0.0, 0.0, -1.0 + 1e-9], OK_R)[0] == "INCONCLUSIVE"            # -0.999999999 rounds to -1.0
assert m.sub((1.0, 2.0), (0.5, 0.5), 1) == 1.5 and m.sub(None, (1, 1), 0) is None
assert set(m.ECP_PIX) == {n for n, _ in m.PIX} and set(m.ECP_IMG) == {n for n, _ in m.IMG}
assert set(m.SEEN) | set(m.UNSEEN) == set(m.ECP_PIX) and not set(m.SEEN) & set(m.UNSEEN)
print("test_collect_exp046 OK")
