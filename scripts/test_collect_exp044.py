import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_exp044 as m

OK_P = [1.0, 1.5, 2.0, 1.0, 0.0, 0.0]; OK_A = [0.3, 0.3, 0.5, 0.4, 0.0, 0.0]; OK_I = [0.0, -0.99, 1.0]
assert m.decide(OK_P, OK_A, OK_I)[0] == "ADVANCE"                                  # every threshold exactly at its boundary
assert m.decide([0.99] + OK_P[1:], OK_A, OK_I)[0] == "INCONCLUSIVE"                # 3/6 PRO wins
assert m.decide(OK_P, [0.29] + OK_A[1:], OK_I)[0] == "INCONCLUSIVE"                # 3/6 AUROC wins (>= +0.3 needed)
assert m.decide(OK_P[:5] + [-1.0], OK_A, OK_I)[0] == "INCONCLUSIVE"                # a PRO change of exactly -1.0 blocks ADVANCE, 4 wins so not FAIL
assert m.decide(OK_P, OK_A[:5] + [-1.0], OK_I)[0] == "INCONCLUSIVE"                # AUROC -1.0 blocks ADVANCE; only 1 set <= -0.5 so not FAIL
assert m.decide(OK_P, OK_A, [0.0, 0.0, -1.0])[0] == "INCONCLUSIVE"                 # image AUROC change exactly -1.0 blocks ADVANCE, never FAIL
assert m.decide(OK_P, OK_A, [0.0, 0.0, -0.99])[0] == "ADVANCE"
assert m.decide([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.5] * 6, OK_I)[0] == "FAIL"      # PRO wins <= 2
assert m.decide([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.5] * 6, OK_I)[0] == "INCONCLUSIVE"   # 3 PRO wins is not FAIL
assert m.decide([2.0] * 6, [0.5, 0.5, 0.5, -0.5, -0.5, -0.5], OK_I)[0] == "FAIL"   # AUROC <= -0.5 on exactly 3 sets
assert m.decide([2.0] * 6, [0.5, 0.5, 0.5, 0.5, -0.5, -0.5], OK_I)[0] == "ADVANCE"  # only 2 sets <= -0.5 (and > -1.0): ADVANCE
assert m.decide([2.0] * 6, [0.5, 0.5, 0.5, 0.5, -0.49, -0.49], OK_I)[0] == "ADVANCE"
assert m.decide([2.0] * 6, [0.5, 0.5, 0.5, -0.49, -0.49, -0.49], OK_I)[0] == "INCONCLUSIVE"   # 3 AUROC wins only, none <= -0.5
assert m.decide([2.0] * 6, [0.29, 0.29, 0.29, 0.29, 0.5, 0.5], OK_I)[0] == "INCONCLUSIVE"
assert m.decide(OK_P[:5] + [None], OK_A, OK_I)[0] == "INCOMPLETE"
assert m.decide(OK_P, OK_A, [0.0, None, 0.0])[0] == "INCOMPLETE"
# floating noise at a threshold does not change the outcome
assert m.decide([1.0 - 1e-9] + OK_P[1:], OK_A, OK_I)[0] == "ADVANCE"
assert m.sub((1.0, 2.0), (0.5, 0.5), 1) == 1.5 and m.sub(None, (1, 1), 0) is None
assert set(m.ECP_PIX_FALLBACK) == {n for n, _ in m.PIX} and set(m.ECP_IMG) == {n for n, _ in m.IMG}
print("test_collect_exp044 OK")
