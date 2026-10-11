import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import collect_exp037 as m

# primary rule at the boundaries
ok_pro = [1.0, 1.2, 2.0, 1.5, 0.0, 0.5]; ok_auc = [0.3, 0.4, 0.5, 0.3, 0.0, 0.0]
assert m.primary_verdict(ok_pro, ok_auc)[0] == "PASS"
assert m.primary_verdict([0.99, 1.2, 2.0, 1.5, 0.0, 0.5], ok_auc)[0] == "INCONCLUSIVE"        # 3/6 PRO wins: neither PASS nor FAIL
assert m.primary_verdict(ok_pro, [0.29, 0.4, 0.5, 0.3, 0.0, 0.0])[0] == "INCONCLUSIVE"          # 3/6 AUROC wins
assert m.primary_verdict([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.5] * 6)[0] == "FAIL"                # PRO wins <= 2
assert m.primary_verdict([2.0] * 6, [0.5, 0.5, 0.5, -0.5, -0.5, -0.5])[0] == "FAIL"            # AUROC <= -0.5 on 3 sets
assert m.primary_verdict([2.0] * 5 + [-1.0], [0.5] * 6)[0] == "INCONCLUSIVE"                    # one set at -1.0 blocks PASS, not FAIL
assert m.primary_verdict([2.0] * 6, [0.5] * 5 + [-1.0])[0] == "INCONCLUSIVE"
assert m.primary_verdict([1.0] * 5 + [None], [0.5] * 6)[0] == "INCOMPLETE"
# mechanism
good = (0.01, 0.002, 0.02)
assert m.concentrated({s: good for s in m.DECISION}) == "CONCENTRATED"
assert m.concentrated({**{s: good for s in m.DECISION[:3]}, m.DECISION[3]: (0.0, -0.01, 0.01)}) == "CONCENTRATED"
assert m.concentrated({**{s: good for s in m.DECISION[:2]}, **{s: (0.0, -0.01, 0.01) for s in m.DECISION[2:]}}) == "NOT CONCENTRATED"
assert m.concentrated({**{s: good for s in m.DECISION}, "ISIC": (0.004999, 0.002, 0.02)}) == "CONCENTRATED"  # 3 of 4 still pass
assert m.concentrated({s: (0.005, 0.0, 0.02) for s in m.DECISION}) == "NOT CONCENTRATED"                  # lower bound must be > 0
assert m.concentrated({"ClinicDB": good}) == "INCOMPLETE"
# terciles (largest / smallest c, id tie-break) and the paired bootstrap
cv = [0.9, 0.1, 0.5, 0.5, 0.3, 0.7]; ids = list("abcdef")
per, cen = m.terciles(cv, ids)
assert sorted(per.tolist()) == [0, 5] and sorted(cen.tolist()) == [1, 4]
mean, lo, hi = m.boot_diff(np.full(50, 0.02), np.zeros(50))
assert abs(mean - 0.02) < 1e-12 and lo == hi == mean
print("test_collect_exp037 OK")
