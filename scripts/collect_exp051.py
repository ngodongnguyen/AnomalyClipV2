"""Collect EXP-051 (position-debiased zoom: ECP trained with the zoom crop corner drawn from Beta(a, a), pilot seed 111) and apply the pre-registered rules
(notes/EXP-051_registration.md). Pure python/numpy/scipy; run from the repo root on the server:  python scripts/collect_exp051.py [seed]
  EB pooled values:   results/matched_eb_s<seed>_fixed/<set>/log.txt          (test.py, fixed z_img -0.69 / z_pix 1.6, sigma 4)
  EB per-image AUROC: results/matched_eb_s<seed>_fixed/<set>/pixel_per_image_predictions.csv
  ECP references:     pooled from EXP-012 / EXP-024 (collect_exp024.load), per-image results/EXP-032/<Set>/badcase_<Set>.csv, offset c position_<Set>.csv
  PE mechanism:       results/EXP-051/pe/<Set>/pe_<Set>.csv (analyze_pe_shift.py, z_pix 1.6) against results/EXP-048/<Set>/pe_<Set>.csv (ECP seed 111)."""
import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DECISION = ("ClinicDB", "ColonDB", "Endo")      # the EXP-048 decision sets (ISIC has no position response, EXP-035)
PER_PASS, PER_LOW, CEN_GUARD, POOLED_GUARD, PRO_FAIL = 0.010, 0.003, -0.005, -1.0, -1.0
Q_MECH, Q_NOT = 0.70, 0.90


def terciles(cvals, ids):
    """(peripheral idx, central idx): the ceil(n/3) largest / smallest c, ties by id order (as EXP-034 / EXP-037)."""
    n = len(cvals); k = int(np.ceil(n / 3))
    per = sorted(range(n), key=lambda i: (-cvals[i], ids[i]))[:k]
    cen = sorted(range(n), key=lambda i: (cvals[i], ids[i]))[:k]
    return np.array(per, int), np.array(cen, int)


def boot_mean(d, n_boot=2000, seed=0):
    """paired per-image differences -> (mean, lo, hi) with an image-bootstrap 95% percentile CI"""
    d = np.asarray(d, float)
    rng = np.random.RandomState(seed)
    m = d[rng.randint(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def position_stats(eb, ecp, pos, n_boot=2000):
    """eb, ecp: {id: per-image AUROC}; pos: {id: c}. Paired over ids present in all three. -> dict(n, per=(mean, lo, hi), cen=(mean, lo, hi))."""
    ids = sorted(i for i in ecp if i in eb and i in pos)
    if len(ids) < 6:
        return dict(n=len(ids), per=None, cen=None)
    g = np.array([eb[i] - ecp[i] for i in ids])
    per, cen = terciles([pos[i] for i in ids], ids)
    return dict(n=len(ids), per=boot_mean(g[per], n_boot), cen=boot_mean(g[cen], n_boot), mean_all=float(g.mean()))


def primary_verdict(per, cen, pooled_pro, pooled_auc, img_auc):
    """per, cen: {decision set: (mean, lo, hi) of the EB - ECP per-image AUROC change in the peripheral / central tercile};
    pooled_pro, pooled_auc: six EB - ECP differences in points; img_auc: three image-AUROC differences in points. Missing (None) anywhere -> INCOMPLETE.
    ADVANCE: peripheral mean >= +0.010 with CI lower bound > 0 on >= 2/3 decision sets AND every guard holds (central mean >= -0.005 on all decision sets,
             no pixel set with pooled PRO or AUROC <= -1.0, no image set with AUROC <= -1.0).
    FAIL:    peripheral mean < +0.003 on >= 2/3 decision sets OR pooled PRO <= -1.0 on >= 3/6 pixel sets.  Otherwise INCONCLUSIVE."""
    vals = [per.get(s) for s in DECISION] + [cen.get(s) for s in DECISION] + list(pooled_pro) + list(pooled_auc) + list(img_auc)
    if any(v is None for v in vals) or len(pooled_pro) != 6 or len(pooled_auc) != 6 or len(img_auc) != 3:
        return "INCOMPLETE", {}
    r = lambda x: round(x, 10)
    n_pass = sum(r(per[s][0]) >= PER_PASS and per[s][1] > 0 for s in DECISION)
    n_low = sum(r(per[s][0]) < PER_LOW for s in DECISION)
    guard_cen = all(r(cen[s][0]) >= CEN_GUARD for s in DECISION)
    guard_pool = all(r(v) > POOLED_GUARD for v in list(pooled_pro) + list(pooled_auc))     # no pixel set with PRO or AUROC <= -1.0 (EXP-024 / EXP-037 convention)
    guard_img = all(r(v) > POOLED_GUARD for v in img_auc)
    n_pro_bad = sum(r(v) <= PRO_FAIL for v in pooled_pro)
    info = dict(peripheral_pass=n_pass, peripheral_low=n_low, central_guard=guard_cen, pooled_guard=guard_pool, image_guard=guard_img, pro_le_m1=n_pro_bad)
    if n_pass >= 2 and guard_cen and guard_pool and guard_img:
        return "ADVANCE", info
    if n_low >= 2 or n_pro_bad >= 3:
        return "FAIL", info
    return "INCONCLUSIVE", info


def mech_set_label(st):
    """st = centre_prior_stats.paired_g(rows ECP, rows EB): Q = G_EB / G_ECP and dG = G_ECP - G_EB.
    REDUCED: Q <= 0.70 and dG CI lower bound > 0; UNCHANGED: Q >= 0.90; INVALID: < 3 paired images or G_ECP <= 0; otherwise MIXED."""
    if st is None or st.get("n", 0) < 3 or not (st.get("G_hi", 0) > 0) or not np.isfinite(st.get("Q", np.nan)):
        return "INVALID"
    q = round(st["Q"], 10)
    if q <= Q_MECH and st["dG_ci"][0] > 0:
        return "REDUCED"
    if q >= Q_NOT:
        return "UNCHANGED"
    return "MIXED"


def mech_verdict(labels):
    """>= 2/3 decision sets REDUCED -> MECHANISM SUPPORTED; >= 2/3 UNCHANGED -> MECHANISM NOT SUPPORTED; < 2 valid -> INCONCLUSIVE; else MIXED."""
    ls = [labels.get(s) for s in DECISION]
    if any(x is None for x in ls):
        return "INCOMPLETE"
    if sum(x != "INVALID" for x in ls) < 2:
        return "INCONCLUSIVE"
    if sum(x == "REDUCED" for x in ls) >= 2:
        return "MECHANISM SUPPORTED"
    if sum(x == "UNCHANGED" for x in ls) >= 2:
        return "MECHANISM NOT SUPPORTED"
    return "MIXED"


def main(seed=111):
    import collect_exp024 as c
    import centre_prior_stats as S
    import analyze_pe_shift as A
    base = c.load(".")
    run = f"results/matched_eb_s{seed}_fixed"
    print(f"EXP-051 pilot, seed {seed}: EB (Beta zoom placement) minus matched ECP; pooled pixel AUROC / PRO at sigma 4, fixed z")
    pro, auc = [], []
    for name, key in c.PIX:
        e = base["pix"][name]["ECP"][seed]; t = c.last_table_row(f"{run}/{key}/log.txt")
        if e is None or t is None:
            print(f"  {name:9s} missing (ECP {e}, EB {t})"); pro.append(None); auc.append(None); continue
        pro.append(t[1] - e[1]); auc.append(t[0] - e[0])
        print(f"  {name:9s} ECP {e[0]:5.1f}/{e[1]:5.1f}  EB {t[0]:5.1f}/{t[1]:5.1f}  AUROC {t[0] - e[0]:+.2f}  PRO {t[1] - e[1]:+.2f}")
    img = []
    print("image head (AUROC / AP, z_img -0.69):")
    for name, key in c.IMG:
        e = base["img"][name]["ECP"][seed]; t = c.last_table_row(f"{run}/{key}/log.txt")
        if e is None or t is None:
            print(f"  {name:9s} missing"); img.append(None); continue
        img.append(t[0] - e[0])
        print(f"  {name:9s} ECP {e[0]:5.1f}/{e[1]:5.1f}  EB {t[0]:5.1f}/{t[1]:5.1f}  AUROC {t[0] - e[0]:+.2f}")
    print("position (per-image pixel AUROC, EB - ECP, terciles of lesion offset c as EXP-034):")
    per, cen = {}, {}
    for name, key in c.PIX:
        try:
            eb = {r["sample_id"]: float(r["per_image_pixel_auroc"]) for r in csv.DictReader(open(f"{run}/{key}/pixel_per_image_predictions.csv"))}
            ecp = {r["id"]: float(r["auroc"]) for r in csv.DictReader(open(f"results/EXP-032/{name}/badcase_{name}.csv"))}
            pos = {r["id"]: float(r["c"]) for r in csv.DictReader(open(f"results/EXP-032/{name}/position_{name}.csv"))}
        except FileNotFoundError as ex:
            print(f"  {name:9s} missing file: {ex}"); continue
        st = position_stats(eb, ecp, pos)
        if st["per"] is None:
            print(f"  {name:9s} too few paired images ({st['n']})"); continue
        per[name], cen[name] = st["per"], st["cen"]
        f = lambda t: f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"
        print(f"  {name:9s} ({'DECISION' if name in DECISION else 'report'}; n {st['n']}/{len(ecp)})  peripheral {f(st['per'])}   central {f(st['cen'])}   all {st['mean_all']:+.4f}")
    v, info = primary_verdict(per, cen, pro, auc, img)
    print(f"PRIMARY: {v}   {info}")
    print("mechanism (PE diagnostic at z_pix 1.6, peripheral tercile; Q = G_EB / G_ECP, G = PE_REC - PE_RAND):")
    labels = {}
    for s in DECISION:
        pe, pb = f"results/EXP-048/{s}/pe_{s}.csv", f"results/EXP-051/pe/{s}/pe_{s}.csv"
        if not (os.path.isfile(pe) and os.path.isfile(pb)):
            print(f"  {s:9s} missing {pb if os.path.isfile(pe) else pe}"); labels[s] = None; continue
        st = S.paired_g(A.load_rows(pe), A.load_rows(pb))
        labels[s] = mech_set_label(st)
        if st.get("n", 0) >= 3:
            print(f"  {s:9s} n {st['n']}  G ECP {st['G_hi']:+.4f}  G EB {st['G_lo']:+.4f}  Q {st['Q']:+.3f}  G_ECP - G_EB {st['dG']:+.4f} "
                  f"[{st['dG_ci'][0]:+.4f}, {st['dG_ci'][1]:+.4f}]  sham ECP {st['sham_hi']:.4f} EB {st['sham_lo']:.4f}  {labels[s]}")
    print(f"MECHANISM: {mech_verdict(labels)}   (report-only reference: CTRL G about +0.04 to +0.06, EXP-049)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 111)
