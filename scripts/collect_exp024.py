"""Collect the EXP-024 three-seed results (ECP / ZO / DUAL, seeds 111, 222, 333) and apply the pre-registered rules R1-R3.
Pure python, no dependencies. Run from the repo root on the server:  python scripts/collect_exp024.py
Seed 111 comes from the existing EXP-012 / EXP-022 runs, seeds 222/333 from results/matched_*_s{222,333}_fixed."""
import json, os, re, statistics, sys

ARMS = ("ECP", "ZO", "DUAL")
SEEDS = (111, 222, 333)
PIX = [("ClinicDB", "CVC-ClinicDB"), ("Kvasir", "Kvasir"), ("ColonDB", "CVC-ColonDB"), ("ISIC", "isic"), ("Endo", "endo"), ("TN3K", "tn3k")]
IMG = [("HeadCT", "HeadCT_anomaly_detection"), ("BrainMRI", "BrainMRI"), ("Br35H", "br35")]
REF = "results/EXP-012/matched-retry-20261002"
EXP012_DIR = {"ClinicDB": "ClinicDB", "Kvasir": "Kvasir", "ColonDB": "ColonDB", "ISIC": "ISIC", "Endo": "Endo", "TN3K": "TN3K"}


def last_table_row(path):
    """(a, b) of the result row just above the final '| mean |' row of a test.py log, or None."""
    if not os.path.isfile(path):
        return None
    rows = [l for l in open(path, errors="ignore").read().splitlines() if l.startswith("|")]
    for i in range(len(rows) - 1, 0, -1):
        if re.match(r"\|\s*mean\s*\|", rows[i]):
            nums = re.findall(r"[-+]?\d+\.?\d*", rows[i - 1].split("|", 2)[2])
            return (float(nums[0]), float(nums[1])) if len(nums) >= 2 else None
    return None


def json_pixel(path):
    if not os.path.isfile(path):
        return None
    j = json.load(open(path))
    m = j.get("mean") if isinstance(j.get("mean"), dict) else j
    return (100 * m["pixel_auroc"], 100 * m["pixel_aupro"]) if "pixel_auroc" in m else None


def path_for(arm, seed, kind, key, short):
    """Where the result of (arm, seed) for one dataset lives."""
    if seed != 111:
        ck = {"ECP": "matched_ecp", "ZO": "matched_zoom", "DUAL": "matched_dual"}[arm] + f"_s{seed}"
        return f"results/{ck}_fixed/{key}/log.txt"
    if arm == "DUAL":
        return f"results/ecp_dual_fixed/{key}/log.txt"
    folder = {"ECP": "ecp_extent", "ZO": "zoom_only"}[arm]
    if kind == "img":
        return f"results/EXP-012-matched/{folder}/checkpoints/{key}/log.txt"
    return f"results/EXP-012-matched/{folder}/checkpoints_A_s4_fixed/{key}/log.txt"   # only exists for ECP (EXP-021 arm A)


def load(root="."):
    t = {"pix": {}, "img": {}}
    for kind, sets in (("pix", PIX), ("img", IMG)):
        for name, key in sets:
            t[kind][name] = {a: {} for a in ARMS}
            for arm in ARMS:
                for seed in SEEDS:
                    v = None
                    if kind == "pix" and seed == 111 and arm in ("ECP", "ZO"):
                        folder = {"ECP": "ecp_extent", "ZO": "zoom_only"}[arm]
                        v = json_pixel(os.path.join(root, REF, EXP012_DIR[name], folder, "pixel_level_metrics.json"))
                    if v is None:
                        v = last_table_row(os.path.join(root, path_for(arm, seed, kind, key, name)))
                    t[kind][name][arm][seed] = v
    return t


def diffs(t, kind, name, a, b, idx):
    out = []
    for s in SEEDS:
        x, y = t[kind][name][a][s], t[kind][name][b][s]
        out.append(None if x is None or y is None else x[idx] - y[idx])
    return out


def win(d, thr):
    return None not in d and statistics.mean(d) >= thr and all(x > 0 for x in d)


def missing(t):
    """(kind, set, arm, seed) of every result that could not be read."""
    return [(k, n, a, s) for k in ("pix", "img") for n in t[k] for a in ARMS for s in SEEDS if t[k][n][a][s] is None]


def decide(t):
    miss = missing(t)
    if miss:   # never apply the rules to partial data: a missing value would silently count as "not a win"
        return {k: ("INCOMPLETE", f"{len(miss)} of {2 * 0 + sum(len(t[kk]) for kk in t) * len(ARMS) * len(SEEDS)} results missing") for k in ("R1", "R2", "R3")}
    res = {}
    # R1: ECP vs ZO on pixel, both PRO and AUROC >= +1.0 as wins
    wp = sum(win(diffs(t, "pix", n, "ECP", "ZO", 1), 1.0) for n, _ in PIX)
    wa = sum(win(diffs(t, "pix", n, "ECP", "ZO", 0), 1.0) for n, _ in PIX)
    res["R1"] = ("CONFIRMED" if wp >= 4 and wa >= 4 else "NOT CONFIRMED" if wp <= 2 or wa <= 2 else "INCONCLUSIVE", f"PRO wins {wp}/6, AUROC wins {wa}/6")
    # R2: ECP vs DUAL, PRO wins on pixel and image AUROC difference >= -1.0 on all image sets
    w2 = sum(win(diffs(t, "pix", n, "ECP", "DUAL", 1), 1.0) for n, _ in PIX)
    img_ok = all((lambda d: None not in d and statistics.mean(d) >= -1.0)(diffs(t, "img", n, "ECP", "DUAL", 0)) for n, _ in IMG)
    res["R2"] = ("AXIS CONFIRMED" if w2 >= 4 and img_ok else "AXIS NOT CONFIRMED" if w2 <= 2 else "INCONCLUSIVE", f"PRO wins {w2}/6, image AUROC (ECP-DUAL) >= -1.0 on all sets: {img_ok}")
    # R3: image head ECP vs ZO, AUROC wins >= +1.0
    w3 = sum(win(diffs(t, "img", n, "ECP", "ZO", 0), 1.0) for n, _ in IMG)
    res["R3"] = ("CONFIRMED" if w3 >= 2 else "NOT CONFIRMED" if w3 == 0 else "INCONCLUSIVE", f"image AUROC wins {w3}/3")
    return res


def fmt(v):
    return "  n/a  " if v is None else f"{v[0]:5.1f}/{v[1]:5.1f}"


def report(t):
    for kind, sets, head in (("pix", PIX, "pixel AUROC/PRO (sigma 4)"), ("img", IMG, "image AUROC/AP")):
        print(f"\n== {head}: per seed 111 | 222 | 333, then mean +- sd of the first number (AUROC)")
        for name, _ in sets:
            for arm in ARMS:
                vals = [t[kind][name][arm][s] for s in SEEDS]
                a = [v[0] for v in vals if v is not None]
                ms = f"{statistics.mean(a):5.2f} +- {statistics.pstdev(a):4.2f}" if a else "n/a"
                print(f"{name:9s} {arm:5s} " + " | ".join(fmt(v) for v in vals) + f"   mean {ms}")
    print("\n== paired differences per seed (AUROC / PRO or AP), mean over seeds")
    for a, b in (("ECP", "ZO"), ("ECP", "DUAL"), ("DUAL", "ZO")):
        for kind, sets in (("pix", PIX), ("img", IMG)):
            for name, _ in sets:
                da, db = diffs(t, kind, name, a, b, 0), diffs(t, kind, name, a, b, 1)
                f = lambda d: "n/a" if None in d else f"{statistics.mean(d):+5.1f} ({' '.join(f'{x:+.1f}' for x in d)})"
                print(f"{a}-{b:4s} {name:9s} AUROC {f(da):28s} {'PRO' if kind == 'pix' else 'AP '} {f(db)}")
    miss = missing(t)
    if miss:
        by = {}
        for k, n, a, sd in miss:
            by.setdefault((a, sd), []).append(n)
        print("\n== MISSING results (the rules are NOT applied until all are present):")
        for (a, sd), ns in sorted(by.items()):
            print(f"   {a:5s} seed {sd}: {len(ns)} sets ({', '.join(ns)})")
    print("\n== pre-registered rules")
    for k, (verdict, why) in decide(t).items():
        print(f"{k}: {verdict}   [{why}]")


if __name__ == "__main__":
    report(load(sys.argv[1] if len(sys.argv) > 1 else "."))
