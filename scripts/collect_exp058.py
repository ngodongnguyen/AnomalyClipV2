"""Collect EXP-058 (inference strategy of the extent coordinate: TASK vs FIX0 vs PRED, ORACLE report-only; matched ECP seeds 111/222/333)
and apply the pre-registered rules S1, S2 and the verdict.  TASK (the deployed z_img=-0.69 / z_pix=1.6) is read exactly like the ECP arm of
collect_exp024.py; FIX0 / PRED / ORACLE from results/EXP-058/<strategy>_s<seed>/<set>/log.txt.  Pure python.  ORACLE is report-only (pixel sets
only, uses test masks) and never blocks or changes the verdict.  Rule text: research/EXPERIMENTS.md, EXP-058."""
import csv, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_exp024 as c

SEEDS, PIX, IMG = c.SEEDS, c.PIX, c.IMG
RULE_ARMS = ("TASK", "FIX0", "PRED")
STRATS = {"FIX0": "fix0", "PRED": "pred", "ORACLE": "oracle"}


def load(root="."):
    base = c.load(root)
    t = {"pix": {}, "img": {}}
    for kind, sets in (("pix", PIX), ("img", IMG)):
        for name, key in sets:
            t[kind][name] = {"TASK": dict(base[kind][name]["ECP"])}
            for arm, folder in STRATS.items():
                if arm == "ORACLE" and kind == "img":
                    continue   # no masks: the oracle is undefined on the image sets and must not be run there
                t[kind][name][arm] = {s: c.last_table_row(os.path.join(root, f"results/EXP-058/{folder}_s{s}/{key}/log.txt")) for s in SEEDS}
    return t


def diffs(t, kind, name, a, b, idx):
    out = []
    for s in SEEDS:
        x, y = t[kind][name][a][s], t[kind][name][b][s]
        out.append(None if x is None or y is None else x[idx] - y[idx])
    return out


def win(d, thr=1.0):
    return None not in d and round(statistics.mean(d), 10) >= thr and all(x > 0 for x in d)


def loss(d, thr=-1.0):
    return None not in d and round(statistics.mean(d), 10) <= thr


def missing(t):
    return [(k, n, a, s) for k in ("pix", "img") for n in t[k] for a in RULE_ARMS for s in SEEDS if t[k][n][a][s] is None]


def decide(t):
    miss = missing(t)
    if miss:
        return {"VERDICT": ("INCOMPLETE", f"{len(miss)} results missing for TASK / FIX0 / PRED")}
    w_pix = sum(win(diffs(t, "pix", n, "TASK", "FIX0", 1)) for n, _ in PIX)
    w_img = sum(win(diffs(t, "img", n, "TASK", "FIX0", 0)) for n, _ in IMG)
    l_pix = sum(loss(diffs(t, "pix", n, "TASK", "PRED", 1)) for n, _ in PIX)
    l_img = sum(loss(diffs(t, "img", n, "TASK", "PRED", 0)) for n, _ in IMG)
    s1 = w_pix >= 4 and w_img >= 2
    s2 = l_pix <= 2 and l_img <= 1
    out = {"S1": ("HOLDS" if s1 else "FAILS", f"TASK vs FIX0: AUPRO wins {w_pix}/6 (need >= 4), image AUROC wins {w_img}/3 (need >= 2)"),
           "S2": ("HOLDS" if s2 else "FAILS", f"TASK vs PRED: AUPRO losses {l_pix}/6 (need <= 2), image AUROC losses {l_img}/3 (need <= 1)")}
    v = "FIXED COORDINATES SUPPORTED" if s1 and s2 else "NOT SUPPORTED" if (l_pix >= 3 or l_img >= 2) else "INCONCLUSIVE"
    out["VERDICT"] = (v, "S1 and S2 " + ("both hold" if s1 and s2 else "do not both hold"))
    return out


def seed_means(t, kind, arm, idx):
    """Per seed: mean over the sets of one metric (None when any set is missing)."""
    out = []
    for s in SEEDS:
        vals = [t[kind][n].get(arm, {}).get(s) for n, _ in (PIX if kind == "pix" else IMG)]
        out.append(None if arm not in t[kind][PIX[0][0] if kind == "pix" else IMG[0][0]] or None in vals else statistics.mean(v[idx] for v in vals))
    return out


def cell(vals):
    if None in vals:
        return "n/a"
    return f"{statistics.mean(vals):5.1f} +- {statistics.pstdev(vals):3.1f}"


def z_pred_summary(root, folder, seed, key):
    path = os.path.join(root, f"results/EXP-058/{folder}_s{seed}/{key}/per_image.csv")
    if not os.path.isfile(path):
        return None
    z = []
    for r in csv.DictReader(open(path)):
        try:
            v = float(r.get("z_pred", "nan"))
        except ValueError:
            continue
        if v == v:
            z.append(v)
    return None if not z else (min(z), statistics.median(z), max(z))


def report(t, root="."):
    print("== per set, per seed 111 | 222 | 333 (pixel: AUROC/AUPRO, image: AUROC/AP)")
    for kind, sets in (("pix", PIX), ("img", IMG)):
        for name, _ in sets:
            for arm in ("TASK", "FIX0", "PRED", "ORACLE"):
                if arm not in t[kind][name]:
                    continue
                v = [t[kind][name][arm][s] for s in SEEDS]
                print(f"{name:9s} {arm:6s} " + " | ".join("  n/a  " if x is None else f"{x[0]:5.1f}/{x[1]:5.1f}" for x in v))
    print("\n== paper table: mean +- population sd over seeds of the per-seed set-mean (image over 3 sets, pixel over 6 sets)")
    print(f"{'strategy':28s} {'image AUROC':>14s} {'pixel AUROC':>14s} {'AUPRO':>14s}")
    names = {"FIX0": "Fixed z = 0", "PRED": "Predicted z-hat", "TASK": "Fixed task-specific z (Ours)", "ORACLE": "Oracle z_gt (diagnostic)"}
    for arm in ("FIX0", "PRED", "TASK", "ORACLE"):
        img = "n/a (undefined)" if arm == "ORACLE" else cell(seed_means(t, "img", arm, 0))
        print(f"{names[arm]:28s} {img:>14s} {cell(seed_means(t, 'pix', arm, 0)):>14s} {cell(seed_means(t, 'pix', arm, 1)):>14s}")
    print("\n== TASK minus the other strategies, mean over seeds (seeds in brackets): pixel AUPRO, image AUROC")
    for a, b in (("TASK", "FIX0"), ("TASK", "PRED"), ("ORACLE", "TASK")):
        for kind, sets, idx in (("pix", PIX, 1), ("img", IMG, 0)):
            if "ORACLE" in (a, b) and kind == "img":
                continue
            for name, _ in sets:
                d = diffs(t, kind, name, a, b, idx)
                s = "n/a" if None in d else f"{statistics.mean(d):+5.1f} ({' '.join(f'{x:+.1f}' for x in d)})"
                print(f"{a}-{b:5s} {name:9s} {'AUPRO' if kind == 'pix' else 'AUROC'} {s}")
    print("\n== predicted z-hat (min / median / max) per set, seed 111 (collapsed estimates show here)")
    for kind, sets in (("pix", PIX), ("img", IMG)):
        for name, key in sets:
            z = z_pred_summary(root, "pred", 111, key)
            print(f"{name:9s} " + ("per_image.csv not found" if z is None else f"{z[0]:+.2f} / {z[1]:+.2f} / {z[2]:+.2f}"))
    miss = missing(t)
    if miss:
        print(f"\nMISSING (TASK/FIX0/PRED): {len(miss)} of {len(RULE_ARMS) * len(SEEDS) * (len(PIX) + len(IMG))}; first 5: {miss[:5]}")
    om = [(n, s) for n, _ in PIX for s in SEEDS if t["pix"][n]["ORACLE"][s] is None]
    if om:
        print(f"ORACLE (report-only) missing for {len(om)} set-seed cells")
    print("\n== pre-registered rules (EXP-058)")
    for k, (v, why) in decide(t).items():
        print(f"{k}: {v}   [{why}]")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    report(load(root), root)
