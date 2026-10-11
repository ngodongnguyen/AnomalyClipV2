"""Collect EXP-037 (position-jittered prompt training, pilot seed 111 by default) and apply the pre-registered rules.
Pure python/numpy/scipy; run from the repo root on the server:  python scripts/collect_exp037.py [seed]
Pooled TR values from results/matched_tr_s<seed>_fixed/<set>/log.txt; the matched ECP (same seed) from EXP-012 / EXP-024 as in collect_exp024.
Position check: per-image AUROC of TR (pixel_per_image_predictions.csv) versus ECP (results/EXP-032/<Set>/badcase_<Set>.csv), lesion offset c from
results/EXP-032/<Set>/position_<Set>.csv."""
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import collect_exp024 as c
import zpos_stats as Z

PIX, IMG = c.PIX, c.IMG
DECISION = ("ClinicDB", "ColonDB", "ISIC", "Endo")


def terciles(cvals, ids):
    """(peripheral idx, central idx): the ceil(n/3) largest / smallest c, ties by id order (as EXP-034)."""
    n = len(cvals); k = int(np.ceil(n / 3))
    order = sorted(range(n), key=lambda i: (-cvals[i], ids[i]))
    per = order[:k]
    cen = sorted(range(n), key=lambda i: (cvals[i], ids[i]))[:k]
    return np.array(per), np.array(cen)


def boot_diff(a, b, n_boot=2000, seed=0):
    """mean of (a - b) with an image-bootstrap 95% CI -> (mean, lo, hi); a, b equal-length arrays of per-image differences."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.RandomState(seed)
    m = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def primary_verdict(pro, auc):
    """pro, auc: lists of TR - ECP differences in points over the six pixel sets (None = missing)."""
    if any(v is None for v in pro + auc):
        return "INCOMPLETE", {}
    wp = sum(round(v, 6) >= 1.0 for v in pro)
    wa = sum(round(v, 6) >= 0.3 for v in auc)
    bad = sum(round(v, 6) <= -1.0 for v in pro) + sum(round(v, 6) <= -1.0 for v in auc)
    fail_a = sum(round(v, 6) <= -0.5 for v in auc)
    info = dict(pro_wins=wp, auc_wins=wa, bad=bad, auc_le_m05=fail_a)
    if wp >= 4 and wa >= 4 and bad == 0:
        return "PASS", info
    if wp <= 2 or fail_a >= 3:
        return "FAIL", info
    return "INCONCLUSIVE", info


def concentrated(stats):
    """stats[set] = (mean, lo, hi) of (peripheral gain - central gain); CONCENTRATED if >= +0.005 with lo > 0 on >= 3 of the 4 decision sets."""
    if any(stats.get(s) is None for s in DECISION):
        return "INCOMPLETE"
    n = sum(round(stats[s][0], 10) >= 0.005 and round(stats[s][1], 10) > 0 for s in DECISION)
    return "CONCENTRATED" if n >= 3 else "NOT CONCENTRATED"


def read_tr_pixel(seed, key):
    return c.last_table_row(f"results/matched_tr_s{seed}_fixed/{key}/log.txt")


def position_stats(seed, name, key):
    tr = {r["sample_id"]: float(r["per_image_pixel_auroc"]) for r in csv.DictReader(open(f"results/matched_tr_s{seed}_fixed/{key}/pixel_per_image_predictions.csv"))}
    ecp = {r["id"]: (float(r["auroc"]), float(r["area_frac"])) for r in csv.DictReader(open(f"results/EXP-032/{name}/badcase_{name}.csv"))}
    pos = {r["id"]: float(r["c"]) for r in csv.DictReader(open(f"results/EXP-032/{name}/position_{name}.csv"))}
    ids = sorted(i for i in ecp if i in tr and i in pos)
    a_tr = np.array([tr[i] for i in ids]); a_ec = np.array([ecp[i][0] for i in ids]); cv = np.array([pos[i] for i in ids]); area = np.array([ecp[i][1] for i in ids])
    per, cen = terciles(list(cv), ids)
    g = a_tr - a_ec
    # peripheral minus central gain: bootstrap the two terciles independently
    rng = np.random.RandomState(0)
    gp, gc = g[per], g[cen]
    bs = np.array([gp[rng.randint(0, len(gp), len(gp))].mean() - gc[rng.randint(0, len(gc), len(gc))].mean() for _ in range(2000)])
    pc = (float(gp.mean() - gc.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)))
    return dict(n=len(ids), matched=f"{len(ids)}/{len(ecp)}", per_gain=float(gp.mean()), cen_gain=float(gc.mean()), diff=pc,
                rho_ecp=Z.partial_spearman(cv, a_ec, np.log(area)), rho_tr=Z.partial_spearman(cv, a_tr, np.log(area)), mean_ecp=float(a_ec.mean()), mean_tr=float(a_tr.mean()))


def main(seed=111):
    base = c.load(".")
    pro, auc = [], []
    print(f"EXP-037 pilot, seed {seed}: pooled pixel AUROC / PRO (sigma 4, fixed z), TR - matched ECP")
    for name, key in PIX:
        e = base["pix"][name]["ECP"][seed]; t = read_tr_pixel(seed, key)
        if e is None or t is None:
            print(f"  {name:9s} missing (ECP {e}, TR {t})"); pro.append(None); auc.append(None); continue
        pro.append(t[1] - e[1]); auc.append(t[0] - e[0])
        print(f"  {name:9s} ECP {e[0]:5.1f}/{e[1]:5.1f}  TR {t[0]:5.1f}/{t[1]:5.1f}  diff AUROC {t[0]-e[0]:+.2f}  PRO {t[1]-e[1]:+.2f}")
    verdict, info = primary_verdict(pro, auc)
    print(f"PRIMARY: {verdict}   {info}")
    print("\nimage head (report-only guard; flag if AUROC change <= -1.0):")
    for name, key in IMG:
        e = base["img"][name]["ECP"][seed]; t = c.last_table_row(f"results/matched_tr_s{seed}_fixed/{key}/log.txt")
        if e is None or t is None:
            print(f"  {name:9s} missing"); continue
        d = t[0] - e[0]
        print(f"  {name:9s} ECP {e[0]:5.1f}/{e[1]:5.1f}  TR {t[0]:5.1f}/{t[1]:5.1f}  AUROC {d:+.2f} {'FLAG' if d <= -1.0 else ''}")
    print("\nposition check (per-image AUROC; terciles of lesion offset c as EXP-034):")
    cs = {}
    for name, key in PIX:
        try:
            st = position_stats(seed, name, key)
        except FileNotFoundError as ex:
            print(f"  {name:9s} missing file: {ex}"); cs[name] = None; continue
        cs[name] = st["diff"]
        print(f"  {name:9s} ({'DECISION' if name in DECISION else 'report'}; matched {st['matched']}) peripheral gain {st['per_gain']:+.4f}, central gain {st['cen_gain']:+.4f}, "
              f"difference {st['diff'][0]:+.4f} [{st['diff'][1]:+.4f}, {st['diff'][2]:+.4f}];  partial Spearman(c, AUROC | log area): ECP {st['rho_ecp']:+.3f}  TR {st['rho_tr']:+.3f}")
    print(f"MECHANISM: {concentrated(cs)}")
    for name in ("ColonDB", "Kvasir"):
        e = base["pix"][name]["ECP"][seed]; t = read_tr_pixel(seed, dict(PIX)[name])
        if e and t:
            print(f"  AUPRO {name}: ECP {e[1]:.1f}  TR {t[1]:.1f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 111)
