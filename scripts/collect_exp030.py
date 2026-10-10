"""Collect EXP-030 (matched no-zoom control CTRL vs ZO vs ECP, seeds 111/222/333) and apply the pre-registered rules N1-N2.
ECP and ZO values are read exactly as in collect_exp024.py; CTRL from results/matched_ctrl_s{seed}_fixed/<set>/log.txt. Pure python."""
import os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_exp024 as c

SEEDS, PIX, IMG = c.SEEDS, c.PIX, c.IMG
ARMS = ("ECP", "ZO", "CTRL")


def load(root="."):
    t = c.load(root)
    for kind, sets in (("pix", PIX), ("img", IMG)):
        for name, key in sets:
            t[kind][name]["CTRL"] = {s: c.last_table_row(os.path.join(root, f"results/matched_ctrl_s{s}_fixed/{key}/log.txt")) for s in SEEDS}
    return t


def diffs(t, kind, name, a, b, idx):
    out = []
    for s in SEEDS:
        x, y = t[kind][name][a][s], t[kind][name][b][s]
        out.append(None if x is None or y is None else x[idx] - y[idx])
    return out


def missing(t):
    return [(k, n, a, s) for k in ("pix", "img") for n in t[k] for a in ARMS for s in SEEDS if t[k][n][a][s] is None]


def decide(t):
    if missing(t):
        return {k: ("INCOMPLETE", f"{len(missing(t))} results missing") for k in ("N1", "N2")}
    out = {}
    wp = sum(c.win(diffs(t, "pix", n, "ECP", "CTRL", 1), 1.0) for n, _ in PIX)
    wi = sum(c.win(diffs(t, "img", n, "ECP", "CTRL", 0), 1.0) for n, _ in IMG)
    out["N1"] = ("CONFIRMED" if wp >= 4 and wi >= 2 else "NOT CONFIRMED" if wp <= 2 else "INCONCLUSIVE", f"ECP vs CTRL: pixel PRO wins {wp}/6, image AUROC wins {wi}/3")
    zoom, ext = [], []
    for n, _ in PIX:
        zoom.append(statistics.mean(diffs(t, "pix", n, "ZO", "CTRL", 1)))
        ext.append(statistics.mean(diffs(t, "pix", n, "ECP", "ZO", 1)))
    z_dom = sum(z >= e for z, e in zip(zoom, ext))
    out["N2"] = ("ZOOM-DOMINATED" if z_dom >= 4 else "EXTENT-DOMINATED" if z_dom <= 2 else "MIXED",
                 f"mean PRO gain from zoom (ZO-CTRL) >= from extent (ECP-ZO) on {z_dom}/6 sets; zoom " + " ".join(f"{x:+.1f}" for x in zoom) + " | extent " + " ".join(f"{x:+.1f}" for x in ext))
    return out


def report(t):
    f = lambda d: "n/a" if None in d else f"{statistics.mean(d):+5.1f} ({' '.join(f'{x:+.1f}' for x in d)})"
    for kind, sets, head, m in (("pix", PIX, "pixel", "PRO"), ("img", IMG, "image", "AP")):
        print(f"\n== {head}: per seed 111 | 222 | 333")
        for name, _ in sets:
            for a in ARMS:
                v = [t[kind][name][a][s] for s in SEEDS]
                print(f"{name:9s} {a:5s} " + " | ".join("  n/a  " if x is None else f"{x[0]:5.1f}/{x[1]:5.1f}" for x in v))
            for a, b in (("ECP", "CTRL"), ("ZO", "CTRL"), ("ECP", "ZO")):
                print(f"{name:9s} {a}-{b:4s} AUROC {f(diffs(t, kind, name, a, b, 0)):28s} {m} {f(diffs(t, kind, name, a, b, 1))}")
    print("\n== pre-registered rules (EXP-030)")
    for k, (v, why) in decide(t).items():
        print(f"{k}: {v}   [{why}]")


if __name__ == "__main__":
    report(load(sys.argv[1] if len(sys.argv) > 1 else "."))
