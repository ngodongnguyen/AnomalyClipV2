import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_exp024 as c
random.seed(0)
def fake(gap_zo, gap_dual, img_gap_zo, img_gap_dual):
    t = {"pix": {}, "img": {}}
    for n, _ in c.PIX:
        t["pix"][n] = {}
        base = 80.0
        for arm, g in (("ZO", 0.0), ("DUAL", gap_zo - gap_dual), ("ECP", gap_zo)):
            t["pix"][n][arm] = {s: (base + g + random.uniform(-0.2, 0.2), 60 + g + random.uniform(-0.2, 0.2)) for s in c.SEEDS}
    for n, _ in c.IMG:
        t["img"][n] = {}
        for arm, g in (("ZO", 0.0), ("DUAL", img_gap_zo - img_gap_dual), ("ECP", img_gap_zo)):
            t["img"][n][arm] = {s: (90 + g + random.uniform(-0.2, 0.2), 90 + g) for s in c.SEEDS}
    return t
r = c.decide(fake(3.0, 2.5, 4.0, 1.0)); assert r["R1"][0] == "CONFIRMED" and r["R2"][0] == "AXIS CONFIRMED" and r["R3"][0] == "CONFIRMED", r
r = c.decide(fake(3.0, 0.2, 4.0, 3.0)); assert r["R1"][0] == "CONFIRMED" and r["R2"][0] == "AXIS NOT CONFIRMED" and r["R3"][0] == "CONFIRMED", r   # DUAL gets nearly all the gain
r = c.decide(fake(0.3, 0.1, 0.2, 0.1)); assert r["R1"][0] == "NOT CONFIRMED" and r["R2"][0] == "AXIS NOT CONFIRMED" and r["R3"][0] == "NOT CONFIRMED", r
t = fake(3.0, 2.5, 4.0, 1.0); t["pix"]["ColonDB"]["ECP"][222] = (79.0, 55.0)       # one seed negative -> that set is not a win
for n in ("ClinicDB", "Kvasir", "ISIC"): t["pix"][n]["ECP"][333] = (70.0, 50.0)
r = c.decide(t); assert r["R1"][0] != "CONFIRMED", r
# parsing
import tempfile
d = tempfile.mkdtemp(); p = os.path.join(d, "log.txt")
open(p, "w").write("junk\n| objects   |   pixel_auroc |   pixel_aupro |\n|:---|---:|---:|\n| colon     |          88.6 |          76.1 |\n| mean      |          88.6 |          76.1 |\n")
assert c.last_table_row(p) == (88.6, 76.1) and c.last_table_row(p + "x") is None
open(p, "w").write("| objects   |   pixel_auroc |   pixel_aupro |\n|:---|---:|---:|\n| colon     |            90 |          52.1 |\n| mean      |            90 |          52.1 |\n")
assert c.last_table_row(p) == (90.0, 52.1)
print("ALL COLLECT TESTS PASSED")
