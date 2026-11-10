import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_exp058 as m


def table(fix0=0.0, pred=0.0, task_gain=3.0, drop=None, no_oracle=False):
    """Synthetic tables: TASK = base + task_gain, FIX0 = base + fix0 and PRED = base + pred (pixel AUPRO and image AUROC), ORACLE = base + 2."""
    t = {"pix": {}, "img": {}}
    for kind, sets in (("pix", m.PIX), ("img", m.IMG)):
        for n, _ in sets:
            base = 60.0 if kind == "pix" else 90.0
            g = {"TASK": task_gain, "FIX0": fix0, "PRED": pred, "ORACLE": 2.0}
            arms = ["TASK", "FIX0", "PRED"] + (["ORACLE"] if kind == "pix" else [])
            t[kind][n] = {a: {s: (80.0, base + g[a]) if kind == "pix" else (base + g[a], 90.0) for s in m.SEEDS} for a in arms}
            if kind == "pix" and no_oracle:
                t[kind][n]["ORACLE"] = {s: None for s in m.SEEDS}
    if drop:
        kind, n, arm, s = drop
        t[kind][n][arm][s] = None
    return t


d = m.decide(table(fix0=0.0, pred=2.0, task_gain=3.0)); assert d["VERDICT"][0] == "FIXED COORDINATES SUPPORTED", d      # TASK 3 > FIX0 0; TASK - PRED = +1
d = m.decide(table(fix0=0.0, pred=5.0, task_gain=3.0)); assert d["VERDICT"][0] == "NOT SUPPORTED", d                   # PRED 2 points better everywhere
d = m.decide(table(fix0=2.5, pred=2.0, task_gain=3.0)); assert d["VERDICT"][0] == "INCONCLUSIVE", d                    # TASK - FIX0 = +0.5 < +1.0
d = m.decide(table(drop=("pix", "Endo", "PRED", 333))); assert d["VERDICT"][0] == "INCOMPLETE", d
d = m.decide(table(fix0=0.0, pred=2.0, no_oracle=True)); assert d["VERDICT"][0] == "FIXED COORDINATES SUPPORTED", d     # ORACLE never blocks
# a win needs the difference positive in every seed (mean >= 1 with one seed <= 0 is not a win)
assert not m.win([2.0, 2.0, -0.1]) and m.win([1.0, 1.0, 1.0]) and not m.win([0.99, 1.0, 1.0])
assert m.loss([-1.0, -1.0, -1.0]) and not m.loss([-0.9, -1.0, -1.0])
# set-mean over seeds helper
t = table(fix0=0.0, pred=2.0)
assert abs(m.seed_means(t, "pix", "TASK", 1)[0] - 63.0) < 1e-9
assert m.seed_means(t, "img", "ORACLE", 0) == [None] * 3
print("test_collect_exp058 OK")
