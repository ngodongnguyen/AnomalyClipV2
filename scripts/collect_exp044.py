"""Collect EXP-044 (TokenCLIP-style OT-routed prompt subspaces on ECP, one seed 111) and apply the pre-registered rule.
Pure python/numpy; run from the repo root on the server:  python scripts/collect_exp044.py
TOK = results/matched_tok_s111_fixed, TOKQ1 = results/matched_tokq1_s111_fixed (sigma 4, fixed z_img=-0.69, z_pix=1.6); report-only sigma 32 from
results/<ckpt>_sigma32_fixed. Matched ECP seed 111 (SHA-256 7278af95...): pixel numbers from the EXP-012 per-dataset pixel_level_metrics.json when present,
else the one-decimal table values of the registry; image numbers are the same-code reruns of EXP-043 / EXP-039 (full precision).

PRE-REGISTERED RULE (EXPERIMENTS.md, EXP-044; TOK - matched ECP at sigma 4, points):
  ADVANCE      if pixel PRO >= +1.0 on >= 4/6 sets AND pixel AUROC >= +0.3 on >= 4/6 sets AND no pixel PRO or AUROC change <= -1.0 on any set
               AND no image AUROC change <= -1.0 on any of the three image sets;
  FAIL         if pixel PRO >= +1.0 on <= 2/6 sets OR pixel AUROC <= -0.5 on >= 3/6 sets;
  INCONCLUSIVE otherwise; INCOMPLETE if any needed value is missing."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_exp024 as c

PIX, IMG = c.PIX, c.IMG
ECP_PIX_FALLBACK = {"ClinicDB": (88.3, 76.4), "Kvasir": (88.9, 51.8), "ColonDB": (85.0, 72.9), "ISIC": (93.9, 88.7), "Endo": (91.6, 79.6), "TN3K": (82.5, 52.0)}
ECP_IMG = {"HeadCT": (94.16, 93.05), "BrainMRI": (94.80, 96.47), "Br35H": (97.05, 96.89)}   # EXP-043 same-code rerun / EXP-039 paired CLS


def _ge(v, t): return round(v, 6) >= t
def _le(v, t): return round(v, 6) <= t


def decide(pro, auc, img):
    """pro, auc: TOK - ECP differences (points) on the six pixel sets; img: image AUROC differences on three sets. None = missing."""
    if any(v is None for v in list(pro) + list(auc) + list(img)):
        return "INCOMPLETE", {}
    wp = sum(_ge(v, 1.0) for v in pro)
    wa = sum(_ge(v, 0.3) for v in auc)
    bad_pix = sum(_le(v, -1.0) for v in list(pro) + list(auc))
    bad_img = sum(_le(v, -1.0) for v in img)
    auc_loss = sum(_le(v, -0.5) for v in auc)
    info = dict(pro_wins=wp, auc_wins=wa, pixel_le_m1=bad_pix, image_le_m1=bad_img, auc_le_m05=auc_loss)
    if wp >= 4 and wa >= 4 and bad_pix == 0 and bad_img == 0:
        return "ADVANCE", info
    if wp <= 2 or auc_loss >= 3:
        return "FAIL", info
    return "INCONCLUSIVE", info


def read_pixel(ckpt, key, tag="_fixed"):
    d = f"results/{ckpt}{tag}/{key}"
    return c.json_pixel(f"{d}/pixel_level_metrics.json") or c.last_table_row(f"{d}/log.txt")


def read_image(ckpt, key, tag="_fixed"):
    d = f"results/{ckpt}{tag}/{key}"
    p = f"{d}/image_level_metrics.json"
    if os.path.isfile(p):
        import json
        m = json.load(open(p))["mean"]
        return (100 * m["image_auroc"], 100 * m["image_ap"])
    return c.last_table_row(f"{d}/log.txt")


def ecp_pixel(root="."):
    base = c.load(root)["pix"]
    out = {}
    for name, _ in PIX:
        v = base[name]["ECP"][111]
        out[name] = (v, "result file") if v is not None else (ECP_PIX_FALLBACK[name], "registry fallback (1 decimal)")
    return out


def fmt(v): return "  n/a     " if v is None else f"{v[0]:6.2f}/{v[1]:6.2f}"
def sub(a, b, i): return None if a is None or b is None else a[i] - b[i]


def main():
    tok, q1 = "matched_tok_s111", "matched_tokq1_s111"
    ecp = ecp_pixel(".")
    pro, auc, img = [], [], []
    print("EXP-044 (seed 111, sigma 4, fixed z_img=-0.69 z_pix=1.6): pixel AUROC/PRO, TOK (Q=3) and TOKQ1 (Q=1) versus the matched ECP")
    for name, key in PIX:
        e, src = ecp[name]; t = read_pixel(tok, key); q = read_pixel(q1, key)
        dp, da = sub(t, e, 1), sub(t, e, 0)
        pro.append(dp); auc.append(da)
        print(f"  {name:9s} ECP {fmt(e)} [{src}]  TOK {fmt(t)}  TOKQ1 {fmt(q)} | TOK-ECP AUROC {da if da is None else round(da, 2)} PRO {dp if dp is None else round(dp, 2)}"
              f" | TOK-TOKQ1 AUROC {sub(t, q, 0) if sub(t, q, 0) is None else round(sub(t, q, 0), 2)} PRO {sub(t, q, 1) if sub(t, q, 1) is None else round(sub(t, q, 1), 2)}")
    print("image head (AUROC/AP; the image head is unchanged by the module, so differences are training effects only):")
    for name, key in IMG:
        e = ECP_IMG[name]; t = read_image(tok, key); q = read_image(q1, key)
        d = sub(t, e, 0); img.append(d)
        flag = "FLAG" if d is not None and _le(d, -1.0) else ""
        print(f"  {name:9s} ECP {fmt(e)}  TOK {fmt(t)}  TOKQ1 {fmt(q)} | TOK-ECP AUROC {d if d is None else round(d, 2)} AP {sub(t, e, 1) if sub(t, e, 1) is None else round(sub(t, e, 1), 2)} {flag}"
              f" | TOK-TOKQ1 AUROC {sub(t, q, 0) if sub(t, q, 0) is None else round(sub(t, q, 0), 2)}")
    verdict, info = decide(pro, auc, img)
    print(f"\nPRE-REGISTERED VERDICT (TOK - matched ECP): {verdict}   {info}")
    print("report-only AUPRO: " + "; ".join(f"{n} ECP {ecp[n][0][1]:.1f} TOK {(read_pixel(tok, k) or (0, float('nan')))[1]:.1f} TOKQ1 {(read_pixel(q1, k) or (0, float('nan')))[1]:.1f}" for n, k in PIX if n in ("ColonDB", "Kvasir")))
    dq = [sub(read_pixel(tok, k), read_pixel(q1, k), 1) for _, k in PIX]
    wq = sum(d is not None and _ge(d, 1.0) for d in dq)
    print(f"report-only: TOK - TOKQ1 PRO >= +1.0 on {wq}/6 sets (is it the OT routing or just the extra capacity / OT-map loss / hinge)")
    print("\nreport-only sigma 32 (pixel AUROC/PRO; results/<ckpt>_sigma32_fixed, only if eval32 was run):")
    for name, key in PIX:
        t = read_pixel(tok, key, "_sigma32_fixed"); q = read_pixel(q1, key, "_sigma32_fixed")
        print(f"  {name:9s} TOK {fmt(t)}  TOKQ1 {fmt(q)}")


if __name__ == "__main__":
    main()
