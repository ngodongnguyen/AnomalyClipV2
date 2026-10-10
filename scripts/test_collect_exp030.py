import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_exp030 as m

def table(zo_gain, ecp_gain, img_gain=3.0, drop=False):
    t = {"pix": {}, "img": {}}
    for kind, sets in (("pix", m.PIX), ("img", m.IMG)):
        for n, _ in sets:
            base = 60.0 if kind == "pix" else 90.0
            g = {"CTRL": 0.0, "ZO": zo_gain if kind == "pix" else 0.0, "ECP": ecp_gain if kind == "pix" else img_gain}
            t[kind][n] = {a: {s: (80.0, base + g[a]) if kind == "pix" else (base + g[a], 90.0) for s in m.SEEDS} for a in m.ARMS}
    if drop:
        t["pix"]["Endo"]["CTRL"][333] = None
    return t

d = m.decide(table(1.0, 5.0)); assert d["N1"][0] == "CONFIRMED" and d["N2"][0] == "EXTENT-DOMINATED", d
d = m.decide(table(5.0, 6.0)); assert d["N2"][0] == "ZOOM-DOMINATED", d          # ECP-ZO = 1 < ZO-CTRL = 5
d = m.decide(table(0.5, 0.6)); assert d["N1"][0] == "NOT CONFIRMED", d
d = m.decide(table(1.0, 5.0, drop=True)); assert d["N1"][0] == "INCOMPLETE", d
print("test_collect_exp030 OK")
