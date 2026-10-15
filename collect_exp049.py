"""
EXP-049 / H038 collector (numpy/scipy only, no torch).  Reads results/EXP-049/<tag>/<set>/pe_<set>.csv (and results/EXP-048/<set>/pe_<set>.csv for ECP seed 111),
applies the EXP-048 verdict function UNCHANGED (pe_shift_stats.verdict via analyze_pe_shift.set_stats) to every checkpoint, then the rules fixed in research/EXPERIMENTS.md EXP-049:
  R1 REPLICATION (ECP): both s222 and s333 PE-MEDIATED -> REPLICATED; exactly one -> MIXED; neither -> NOT REPLICATED.
  R2 GENERALITY: per method (CTRL, ZO) count seeds (of 3) that are PE-MEDIATED (MIXED / NOT PE-MEDIATED / INCONCLUSIVE all count as not mediated); a method PASSES with >= 2/3.
     both pass -> GENERAL; exactly one -> PARTIAL; none -> NOT GENERAL.
A checkpoint with a missing decision-set CSV (or fewer than 3 kept images in it) is INCOMPLETE; a rule that needs an INCOMPLETE checkpoint is INCOMPLETE (provisional counts are printed).
  python collect_exp049.py [--root results/EXP-049] [--exp048 results/EXP-048] [--boot 2000]
"""
import os
import json
import argparse
import numpy as np

import pe_shift_stats as S
import analyze_pe_shift as A

ECP_TAGS = ("ecp_s111", "ecp_s222", "ecp_s333")
METHODS = {"ECP": ECP_TAGS, "CTRL": ("ctrl_s111", "ctrl_s222", "ctrl_s333"), "ZO": ("zo_s111", "zo_s222", "zo_s333")}
ORDER = [t for ts in METHODS.values() for t in ts]
LABELS = ("PE-MEDIATED", "NOT PE-MEDIATED", "MIXED", "INCONCLUSIVE")


def r1_verdict(labels):
    """labels: dict tag -> checkpoint verdict (or 'INCOMPLETE'/None). Uses ecp_s222 and ecp_s333 only. -> (label, n_mediated_of_2)."""
    v = [labels.get("ecp_s222"), labels.get("ecp_s333")]
    n = sum(1 for x in v if x == "PE-MEDIATED")
    if any(x is None or x == "INCOMPLETE" for x in v):
        return "INCOMPLETE", n
    return ("REPLICATED" if n == 2 else "MIXED" if n == 1 else "NOT REPLICATED"), n


def r2_verdict(labels):
    """-> (label, {method: n_mediated_of_3}). INCOMPLETE if any CTRL / ZO checkpoint is missing or INCOMPLETE. INCONCLUSIVE counts as not mediated."""
    counts, incomplete = {}, False
    for m in ("CTRL", "ZO"):
        vs = [labels.get(t) for t in METHODS[m]]
        incomplete |= any(x is None or x == "INCOMPLETE" for x in vs)
        counts[m] = sum(1 for x in vs if x == "PE-MEDIATED")
    if incomplete:
        return "INCOMPLETE", counts
    npass = sum(1 for m in ("CTRL", "ZO") if counts[m] >= 2)
    return ("GENERAL" if npass == 2 else "PARTIAL" if npass == 1 else "NOT GENERAL"), counts


def checkpoint_dir(tag, root, exp048):
    return exp048 if tag == "ecp_s111" else os.path.join(root, tag)


def load_checkpoint(tag, root, exp048, boot):
    """-> dict(label, per_set={set: (st, info) or None}, missing=[...]). label INCOMPLETE if a decision-set CSV is absent or has < 3 kept peripheral images."""
    d = checkpoint_dir(tag, root, exp048)
    per_set, missing = {}, []
    for s in A.SETS:
        p = os.path.join(d, s, f"pe_{s}.csv")
        if not os.path.isfile(p):
            if s in S.DECISION:
                missing.append(p)
            continue
        per_set[s] = A.set_stats(A.load_rows(p), boot) + (A.load_rows(p),)
    res = {s: (per_set[s][0] if s in per_set else None) for s in S.DECISION}
    short = [s for s in S.DECISION if s in per_set and per_set[s][0] is None]
    if missing or short:
        return dict(label="INCOMPLETE", per_set=per_set, missing=missing + [f"{s}: < 3 kept images" for s in short], counts=None)
    label, nv, npe, nno = S.verdict(res)
    return dict(label=label, per_set=per_set, missing=[], counts=(nv, npe, nno))


def pooled_parity(tag, root):
    p = os.path.join(root, tag, "ClinicDB", "pooled_parity.json")
    return json.load(open(p)) if os.path.isfile(p) else None


def sham_level(rows):
    k = [r["auroc_sham"] for r in rows if r["group"] == "peripheral" and r["status"] == "kept"]
    return float(np.mean(k)) if k else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/EXP-049")
    ap.add_argument("--exp048", default="results/EXP-048")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    ck = {t: load_checkpoint(t, a.root, a.exp048, a.boot) for t in ORDER}
    labels = {t: ck[t]["label"] for t in ORDER}

    print("================ PER-CHECKPOINT VERDICTS (EXP-048 rule applied unchanged; decision sets ClinicDB, ColonDB, Endo; peripheral tercile) ================")
    for t in ORDER:
        c = ck[t]
        extra = f"VALID {c['counts'][0]}/3, PE-pass {c['counts'][1]}, NOT-pass {c['counts'][2]}" if c["counts"] else "missing: " + "; ".join(c["missing"])
        pj = pooled_parity(t, a.root) if t != "ecp_s111" else None
        par = f"  pooled ClinicDB parity {pj['pooled_auroc_pct']:.2f} vs {pj['recorded_pct']} ({'OK' if pj['ok'] else 'EXCEEDED'}{', limited' if pj['limited'] else ''})" if pj else ""
        print(f"{t:10s} {c['label']:16s} {extra}{par}")

    print("\n================ R, G, D_IMG, D_PE PER DECISION SET (peripheral tercile; R = D_PE / D_IMG) ================")
    print(f"{'checkpoint':10s} {'set':9s} {'kept':>5s} {'D_IMG':>8s} {'D_PE':>8s} {'R':>7s} {'R 95% CI':>17s} {'G':>8s} {'G_lo':>8s} {'VALID':>5s} {'pass_PE':>7s} {'pass_NOT':>8s}")
    rvals = {m: {s: [] for s in S.DECISION} for m in METHODS}
    for t in ORDER:
        m = next(k for k, v in METHODS.items() if t in v)
        for s in S.DECISION:
            e = ck[t]["per_set"].get(s)
            st = e[0] if e else None
            if st is None:
                print(f"{t:10s} {s:9s} missing"); continue
            rvals[m][s].append(st["R"])
            print(f"{t:10s} {s:9s} {st['n']:5d} {st['D_IMG']:+8.4f} {st['D_PE']:+8.4f} {st['R']:+7.3f} [{st['R_ci'][0]:+6.2f},{st['R_ci'][1]:+6.2f}] {st['G']:+8.4f} {st['G_ci'][0]:+8.4f} "
                  f"{str(S.is_valid(st['D_IMG'])):>5s} {str(S.set_pass_pe(st)):>7s} {str(S.set_pass_not(st)):>8s}")

    print("\n================ R ACROSS SEEDS PER METHOD (mean [min, max] over the available seeds) ================")
    for m in METHODS:
        parts = []
        for s in S.DECISION:
            v = [x for x in rvals[m][s] if np.isfinite(x)]
            parts.append(f"{s} {np.mean(v):+.3f} [{min(v):+.3f}, {max(v):+.3f}] (n={len(v)})" if v else f"{s} n/a")
        print(f"{m:5s} " + "   ".join(parts))

    print("\n================ REPORT-ONLY (not part of any rule) ================")
    print(f"{'checkpoint':10s} {'set':9s} {'SHAM periph. AUROC':>19s} {'PE_FLAT-SHAM':>13s} {'IMG_RAND-SHAM':>14s} {'central PE_REC-SHAM [CI]':>34s}   dropped(region/zero)")
    for t in ORDER:
        for s in A.SETS:
            e = ck[t]["per_set"].get(s)
            if not e:
                continue
            st, info, rows = e
            if st is None:
                print(f"{t:10s} {s:9s} too few kept images"); continue
            cen = info.get("central")
            cs = f"{cen[1][0]:+.4f} [{cen[1][1]:+.4f},{cen[1][2]:+.4f}] n={cen[0]}" if cen else "n/a"
            tag = "" if s in S.DECISION else " (report-only set)"
            print(f"{t:10s} {s:9s} {sham_level(rows):19.4f} {info['flat_minus_sham'][0]:+13.4f} {info['img_rand_minus_sham'][0]:+14.4f} {cs:>34s}   {info['n_drop']}/{info['n_zero']}{tag}")
    print("report-only sets (Kvasir, TN3K, ISIC) R / G where present:")
    for t in ORDER:
        for s in A.SETS:
            if s in S.DECISION or s not in ck[t]["per_set"] or ck[t]["per_set"][s][0] is None:
                continue
            st = ck[t]["per_set"][s][0]
            print(f"  {t:10s} {s:9s} D_IMG {st['D_IMG']:+.4f} D_PE {st['D_PE']:+.4f} R {st['R']:+.3f} G {st['G']:+.4f} [{st['G_ci'][0]:+.4f}] VALID {S.is_valid(st['D_IMG'])}")

    print("\n================ REGISTERED RULES (research/EXPERIMENTS.md EXP-049) ================")
    r1, n1 = r1_verdict(labels)
    r2, c2 = r2_verdict(labels)
    print(f"R1 REPLICATION (ECP s222 + s333; s111 from EXP-048 is {labels['ecp_s111']}): PE-MEDIATED seeds among s222/s333 = {n1}/2  ->  {r1}")
    print(f"R2 GENERALITY: PE-MEDIATED seeds CTRL {c2['CTRL']}/3, ZO {c2['ZO']}/3 (>= 2/3 passes; INCONCLUSIVE counts as not mediated)  ->  {r2}")
    print("Claim limits (fixed before the run): frozen CLIP ViT-L/14@336 at 518 px with learned prompts on MVTec (AnomalyCLIP family as implemented here); oracle lesion positions; development sets (D027); per-image AUROC.")
    print("A GENERAL verdict supports 'the position response of AnomalyCLIP-style ZSAD maps is carried by the absolute positional embedding of the shared frozen encoder', not 'all CLIP ZSAD methods'.")


if __name__ == "__main__":
    main()
