import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_exp028 as m

def table(ecp_gap_pro, ecp_gap_auc=1.0, miss=False):
    t = {"pix": {}, "img": {}}
    for kind, sets in (("pix", m.PIX), ("img", m.IMG)):
        for n, _ in sets:
            t[kind][n] = {}
            for arm in m.ARMS:
                base = (80.0, 60.0) if kind == "pix" else (90.0, 90.0)
                if arm == "ECP":
                    v = (base[0] + ecp_gap_auc, base[1] + ecp_gap_pro)
                else:
                    v = base
                t[kind][n][arm] = {s: v for s in m.SEEDS}
    if miss:
        t["pix"]["Kvasir"]["SHUF"][222] = None
    return t

d = m.decide(table(3.0)); assert d["R4"][0] == "EXTENT INFORMATION CONFIRMED" and d["R5"][0] == "SUPERVISION CONFIRMED" and d["R6"][0] == "EXTENT SUPERVISION ADDS BENEFIT", d
d = m.decide(table(0.2, 0.1)); assert d["R4"][0] == "EXTENT INFORMATION NOT CONFIRMED" and d["R6"][0] == "BENEFIT NOT FROM EXTENT INFORMATION", d
d = m.decide(table(3.0, -1.0)); assert d["R4"][0] == "INCONCLUSIVE", d      # PRO wins but AUROC below zero on all sets
d = m.decide(table(3.0, miss=True)); assert d["R4"][0] == "INCOMPLETE" and d["R5"][0] == "SUPERVISION CONFIRMED" and d["R6"][0] == "INCOMPLETE", d
# one seed not positive -> not a win
t = table(3.0)
for n, _ in m.PIX[:3]:
    t["pix"][n]["SHUF"][333] = (80.0 + 1.0, 60.0 + 3.5)
d = m.decide(t); assert d["R4"][0] == "INCONCLUSIVE", d
print("test_collect_exp028 OK")
