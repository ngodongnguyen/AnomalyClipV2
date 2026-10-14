"""Collect EXP-046 (ECP trained AND tested with a V-V head mask, one seed 111) and apply the PRE-REGISTERED rule (research/EXPERIMENTS.md, EXP-046).
Pure python; run from the repo root on the server:  python scripts/collect_exp046.py
HEAD50 = results/matched_head50_s111_fixed, RAND50 = results/matched_rand50_s111_fixed (sigma 4, fixed z_img=-0.69, z_pix=1.6). A = matched ECP seed 111
(SHA-256 7278af95...): pixel numbers from the existing result files when present (as collect_exp024), else the two-decimal registry table below; image numbers are
the same-code reruns (EXP-043 / EXP-039, full precision).

RULE (points; every difference is rounded to 1e-6 before a threshold comparison; primary comparison HEAD50 - A):
  ADVANCE      if pooled pixel PRO >= +1.0 on >= 4/6 sets AND pooled pixel AUROC >= +0.3 on >= 4/6 sets AND no pixel PRO or AUROC change <= -1.0 on any set
               AND no image AUROC change <= -1.0 on any of the three image sets AND (HEAD50 - RAND50) PRO >= +1.0 on >= 4/6 sets;
  FAIL         if PRO >= +1.0 on <= 2/6 sets OR pixel AUROC <= -0.5 on >= 3/6 sets (the FAIL clauses do not involve RAND50, so FAIL holds even if RAND50 is missing);
  INCONCLUSIVE otherwise; INCOMPLETE if a needed value is missing.
Report-only: the six pixel sets split into 'seen in EXP-045' (ClinicDB, Kvasir, ColonDB) and 'unseen' (ISIC, Endo, TN3K); image AP; AUPRO is the PRO column."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_exp024 as c
import collect_exp044 as m44

PIX, IMG = c.PIX, c.IMG
ECP_PIX = {"ClinicDB": (88.33, 76.43), "Kvasir": (88.93, 51.81), "ColonDB": (84.97, 72.94), "ISIC": (93.87, 88.67), "Endo": (91.62, 79.59), "TN3K": (82.45, 51.99)}
ECP_IMG = {"HeadCT": (94.16, 93.05), "BrainMRI": (94.80, 96.47), "Br35H": (97.05, 96.89)}
SEEN = ("ClinicDB", "Kvasir", "ColonDB")
UNSEEN = ("ISIC", "Endo", "TN3K")
HEAD, RAND = "matched_head50_s111", "matched_rand50_s111"


def _ge(v, t): return round(v, 6) >= t
def _le(v, t): return round(v, 6) <= t


def decide(pro, auc, img, pro_vs_rand):
    """pro, auc: HEAD50 - A on the six pixel sets; img: image AUROC differences on three sets; pro_vs_rand: HEAD50 - RAND50 PRO on the six sets. None = missing."""
    if any(v is None for v in list(pro) + list(auc) + list(img)):
        return "INCOMPLETE", {}
    wp = sum(_ge(v, 1.0) for v in pro)
    wa = sum(_ge(v, 0.3) for v in auc)
    bad_pix = sum(_le(v, -1.0) for v in list(pro) + list(auc))
    bad_img = sum(_le(v, -1.0) for v in img)
    auc_loss = sum(_le(v, -0.5) for v in auc)
    info = dict(pro_wins=wp, auc_wins=wa, pixel_le_m1=bad_pix, image_le_m1=bad_img, auc_le_m05=auc_loss)
    if wp <= 2 or auc_loss >= 3:
        return "FAIL", info
    if any(v is None for v in pro_vs_rand):
        return "INCOMPLETE", info
    wr = sum(_ge(v, 1.0) for v in pro_vs_rand)
    info["pro_wins_vs_rand50"] = wr
    if wp >= 4 and wa >= 4 and bad_pix == 0 and bad_img == 0 and wr >= 4:
        return "ADVANCE", info
    return "INCONCLUSIVE", info


def ecp_pixel(root="."):
    base = c.load(root)["pix"]
    out = {}
    for name, _ in PIX:
        v = base[name]["ECP"][111]
        out[name] = (v, "result file") if v is not None else (ECP_PIX[name], "registry table (2 decimals)")
    return out


def fmt(v): return "  n/a     " if v is None else f"{v[0]:6.2f}/{v[1]:6.2f}"
def r2(x): return None if x is None else round(x, 2)
def sub(a, b, i): return None if a is None or b is None else a[i] - b[i]


def main():
    ecp = ecp_pixel(".")
    pro, auc, img, prr = [], [], [], []
    rows = {}
    print("EXP-046 (seed 111, sigma 4, fixed z_img=-0.69 z_pix=1.6): pooled pixel AUROC/PRO, HEAD50 (top-50% V-V heads) and RAND50 (random 50%, seed 0) versus the matched ECP (A)")
    for name, key in PIX:
        e, src = ecp[name]; h = m44.read_pixel(HEAD, key); r = m44.read_pixel(RAND, key)
        dp, da, dr, dra = sub(h, e, 1), sub(h, e, 0), sub(h, r, 1), sub(h, r, 0)
        pro.append(dp); auc.append(da); prr.append(dr); rows[name] = (dp, da)
        grp = "seen in EXP-045" if name in SEEN else "unseen"
        print(f"  {name:9s} [{grp:15s}] A {fmt(e)} [{src}]  HEAD50 {fmt(h)}  RAND50 {fmt(r)} | HEAD50-A AUROC {r2(da)} PRO {r2(dp)} | HEAD50-RAND50 AUROC {r2(dra)} PRO {r2(dr)}")
    print("image head (AUROC/AP; the CLS path is unmasked but trained jointly):")
    for name, key in IMG:
        e = ECP_IMG[name]; h = m44.read_image(HEAD, key); r = m44.read_image(RAND, key)
        d = sub(h, e, 0); img.append(d)
        flag = "FLAG" if d is not None and _le(d, -1.0) else ""
        print(f"  {name:9s} A {fmt(e)}  HEAD50 {fmt(h)}  RAND50 {fmt(r)} | HEAD50-A AUROC {r2(d)} AP {r2(sub(h, e, 1))} {flag} | HEAD50-RAND50 AUROC {r2(sub(h, r, 0))}")
    verdict, info = decide(pro, auc, img, prr)
    print(f"\nPRE-REGISTERED VERDICT (HEAD50 - matched ECP, with HEAD50 - RAND50): {verdict}   {info}")
    for title, names in (("seen in EXP-045 (ClinicDB, Kvasir, ColonDB)", SEEN), ("unseen (ISIC, Endo, TN3K)", UNSEEN)):
        d = [rows[n] for n in names]
        if any(x[0] is None for x in d):
            print(f"report-only {title}: n/a"); continue
        print(f"report-only {title}: PRO >= +1.0 on {sum(_ge(x[0], 1.0) for x in d)}/3, AUROC >= +0.3 on {sum(_ge(x[1], 0.3) for x in d)}/3, "
              f"mean dPRO {sum(x[0] for x in d) / 3:+.2f}, mean dAUROC {sum(x[1] for x in d) / 3:+.2f}")
    print("report-only ColonDB/Kvasir AUPRO: " + "; ".join(
        f"{n} A {ecp[n][0][1]:.1f} HEAD50 {(m44.read_pixel(HEAD, k) or (0, float('nan')))[1]:.1f} RAND50 {(m44.read_pixel(RAND, k) or (0, float('nan')))[1]:.1f}"
        for n, k in PIX if n in ("ColonDB", "Kvasir")))


if __name__ == "__main__":
    main()
