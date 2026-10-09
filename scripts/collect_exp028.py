"""Collect EXP-028 (does extent SUPERVISION add anything beyond a conditioner?) and apply the pre-registered rules R4-R6.
Arms: ECP (true extent labels), SHUF (labels rotated among the anomalous images of a batch), LATENT (no extent supervision, no teacher
forcing; evaluated with its own per-image estimate), ZO and DUAL as references. Seeds 111/222/333. Pure python; run from the repo root.
ECP/ZO/DUAL values are read exactly as in collect_exp024.py; SHUF/LATENT from results/matched_{shuf,latent}_s{seed}_fixed/<set>/log.txt."""
import os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_exp024 as c

SEEDS, PIX, IMG = c.SEEDS, c.PIX, c.IMG
NEW = {"SHUF": "matched_shuf", "LATENT": "matched_latent"}
ARMS = ("ECP", "SHUF", "LATENT", "ZO", "DUAL")


def load(root="."):
    base = c.load(root)
    for kind, sets in (("pix", PIX), ("img", IMG)):
        for name, key in sets:
            for arm, ck in NEW.items():
                base[kind][name][arm] = {s: c.last_table_row(os.path.join(root, f"results/{ck}_s{s}_fixed/{key}/log.txt")) for s in SEEDS}
    return base


def diffs(t, kind, name, a, b, idx):
    out = []
    for s in SEEDS:
        x, y = t[kind][name][a][s], t[kind][name][b][s]
        out.append(None if x is None or y is None else x[idx] - y[idx])
    return out


def missing(t, arms):
    return [(k, n, a, s) for k in ("pix", "img") for n in t[k] for a in arms for s in SEEDS if t[k][n][a][s] is None]


def rule(t, other):
    """ECP versus `other` on pixel sets: wins = mean PRO diff >= +1.0 and positive in all seeds."""
    wp = sum(c.win(diffs(t, "pix", n, "ECP", other, 1), 1.0) for n, _ in PIX)
    pos = sum(None not in (d := diffs(t, "pix", n, "ECP", other, 0)) and statistics.mean(d) >= 0 for n, _ in PIX)
    wi = sum(c.win(diffs(t, "img", n, "ECP", other, 0), 1.0) for n, _ in IMG)
    return wp, pos, wi


def verdict(wp, pos, label):
    if wp >= 4 and pos >= 5:
        return f"{label} CONFIRMED"
    if wp <= 2:
        return f"{label} NOT CONFIRMED"
    return "INCONCLUSIVE"


def decide(t):
    out = {}
    for rid, other, label in (("R4", "SHUF", "EXTENT INFORMATION"), ("R5", "LATENT", "SUPERVISION")):
        miss = missing(t, ("ECP", other))
        if miss:
            out[rid] = ("INCOMPLETE", f"{len(miss)} results missing for ECP/{other}")
            continue
        wp, pos, wi = rule(t, other)
        out[rid] = (verdict(wp, pos, label), f"ECP vs {other}: PRO wins {wp}/6, mean pixel AUROC diff >= 0 on {pos}/6, image AUROC wins {wi}/3 (report-only)")
    if all(v[0].endswith("CONFIRMED") and "NOT" not in v[0] for v in out.values()):
        out["R6"] = ("EXTENT SUPERVISION ADDS BENEFIT", "R4 and R5 both confirmed (together with EXP-024 R2 against the learned-constant control)")
    elif any(v[0] == "INCOMPLETE" for v in out.values()):
        out["R6"] = ("INCOMPLETE", "needs R4 and R5")
    elif out["R4"][0].startswith("EXTENT INFORMATION NOT"):
        out["R6"] = ("BENEFIT NOT FROM EXTENT INFORMATION", "SHUF matches ECP on PRO: the gain is label-independent (stochastic prompt shifts / regularisation), not extent")
    else:
        out["R6"] = ("NOT ESTABLISHED", "R4/R5 not both confirmed")
    return out


def fmt(v):
    return "  n/a  " if v is None else f"{v[0]:5.1f}/{v[1]:5.1f}"


def report(t):
    for kind, sets, head in (("pix", PIX, "pixel AUROC/PRO (sigma 4)"), ("img", IMG, "image AUROC/AP")):
        print(f"\n== {head}: per seed 111 | 222 | 333, mean of the first number")
        for name, _ in sets:
            for arm in ARMS:
                vals = [t[kind][name][arm][s] for s in SEEDS]
                a = [v[0] for v in vals if v is not None]
                print(f"{name:9s} {arm:6s} " + " | ".join(fmt(v) for v in vals) + (f"   mean {statistics.mean(a):5.2f}" if a else "   n/a"))
    print("\n== paired differences, mean over seeds (per seed in brackets): AUROC / PRO-or-AP")
    for a, b in (("ECP", "SHUF"), ("ECP", "LATENT"), ("SHUF", "ZO"), ("LATENT", "ZO")):
        for kind, sets in (("pix", PIX), ("img", IMG)):
            for name, _ in sets:
                f = lambda d: "n/a" if None in d else f"{statistics.mean(d):+5.1f} ({' '.join(f'{x:+.1f}' for x in d)})"
                print(f"{a}-{b:6s} {name:9s} {f(diffs(t, kind, name, a, b, 0)):28s} {f(diffs(t, kind, name, a, b, 1))}")
    miss = missing(t, ARMS)
    if miss:
        by = {}
        for k, n, a, sd in miss:
            by.setdefault((a, sd), []).append(n)
        print("\n== MISSING results (rules not applied for incomplete arms):")
        for (a, sd), ns in sorted(by.items()):
            print(f"   {a:6s} seed {sd}: {len(ns)} sets ({', '.join(ns)})")
    print("\n== pre-registered rules (EXP-028)")
    for k, (v, why) in decide(t).items():
        print(f"{k}: {v}   [{why}]")


if __name__ == "__main__":
    report(load(sys.argv[1] if len(sys.argv) > 1 else "."))
