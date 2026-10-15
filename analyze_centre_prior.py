"""
EXP-050 / H039: does the zoom augmentation teach a centre prior, and does the ECP pixel prompt carry it along the extent axis?  Diagnostic, NOT a method.
Rule and thresholds: notes/EXP-050_registration.md (to be copied into research/EXPERIMENTS.md before any run).  Statistics: centre_prior_stats.py (unit-tested).

  python analyze_centre_prior.py source --data_root $MVTEC [--limit N]     # Part A, CPU only, no torch, no model: exact training crop code on MVTec masks
  python analyze_centre_prior.py zreport                                     # Part B report, no torch: EXP-048/049 (z_pix 1.6) vs results/EXP-050/zlow (z_pix -0.69)
  python analyze_centre_prior.py report                                      # both verdicts
  python analyze_centre_prior.py sweep --data_root $MVTEC                    # EXP-051: source-only choice of the Beta placement parameter (CPU)
Part B per-image CSVs are produced by analyze_pe_shift.py itself (unchanged) with --z_pix -0.69; see scripts/run_exp050.sh.
"""
import os
import csv
import json
import time
import hashlib
import random
import argparse
import numpy as np

import centre_prior_stats as S

CATEGORIES = ('carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid',
              'pill', 'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood')   # EXP-040 order (scripts/exp040_core.py)
EXP040_ATTEMPTS, EXP040_SUCCESS = 9407, 8768
Z_BINS = (-np.inf, 0.0, 0.5, 0.8, 1.2, np.inf)
ECP_SEEDS = ("ecp_s111", "ecp_s222", "ecp_s333")
REPORT_SETS = ("ClinicDB", "ColonDB", "Endo", "Kvasir", "TN3K")


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ------------------------------------------------------------------ Part A
def stage_source(args):
    from PIL import Image
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    js = os.path.join(out, "part_a.json")
    if os.path.exists(js) and not args.redo:
        print(f"{js} exists, skipped (use --redo)"); print_part_a(json.load(open(js))); return
    meta_path = os.path.join(args.data_root, "meta.json")
    meta = json.load(open(meta_path))
    if set(meta["test"]) != set(CATEGORIES):
        raise SystemExit("STOP: MVTec category list differs from the EXP-040 list")
    f_box, box_sha = S.load_crop_box(args.dataset_py)
    rows, invalid = [], 0
    for cat in CATEGORIES:
        for r in meta["test"][cat]:
            if int(r["anomaly"]) == 1:
                rows.append((cat, r))
    if args.limit:
        rows = rows[: args.limit]

    def masks():
        nonlocal invalid
        for cat, r in rows:
            ip = os.path.join(args.data_root, r["img_path"]); mp = os.path.join(args.data_root, r["mask_path"])
            with Image.open(ip) as im:
                size = im.size
            with Image.open(mp) as mi:
                if mi.size != size:          # EXP-040 skips these BEFORE any random draw; the training code would also leave them unzoomed
                    invalid += 1
                    kept_flags.append(False)
                    continue
                kept_flags.append(True)
                yield np.asarray(mi.convert("L")) > 0      # Dataset.__getitem__ thresholds the source mask at > 0

    kept_flags = []
    t0 = time.time()
    random.seed(args.seed)                    # as the EXP-040 audit: one global seed, image outer, epoch inner, categories in CATEGORIES order
    v = S.simulate_visits(masks(), lambda m: f_box(m, tuple(args.target)), epochs=args.epochs, p=args.p)
    kept_rows = [rw for rw, k in zip(rows, kept_flags) if k]
    print(f"[source] {len(kept_rows)} anomalous MVTec test images ({invalid} with mask/image size mismatch skipped), {len(v['img'])} visits, "
          f"{v['n_attempts']} crop attempts, {v['n_success']} successful ({time.time() - t0:.0f}s)")
    parity = dict(attempts=v["n_attempts"], success=v["n_success"], exp040_attempts=EXP040_ATTEMPTS, exp040_success=EXP040_SUCCESS,
                  match=(v["n_attempts"], v["n_success"]) == (EXP040_ATTEMPTS, EXP040_SUCCESS), checked=not args.limit)
    print(f"[parity] attempts / successes {v['n_attempts']} / {v['n_success']} vs EXP-040 audit {EXP040_ATTEMPTS} / {EXP040_SUCCESS}: "
          + ("MATCH" if parity["match"] else ("not compared (--limit)" if args.limit else "MISMATCH")))
    if not args.limit and not parity["match"] and not args.no_exp040_parity:
        raise SystemExit("STOP: the simulated crop stream differs from the EXP-040 audit (meta.json, dataset.py, seed or order changed); "
                         "inspect, or pass --no_exp040_parity to continue and report the mismatch")
    with open(os.path.join(out, "visits.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["img_index", "category", "img_path", "zoomed", "z", "c"])
        for i, zm, z, c in zip(v["img"], v["zoomed"], v["z"], v["c"]):
            cat, r = kept_rows[i]
            w.writerow([int(i), cat, r["img_path"], int(zm), f"{z:.6f}", f"{c:.6f}"])
    st = S.part_a(v, n_boot=args.boot)
    label, l1, l2 = S.verdict_a(st)
    # report-only: mean c by z bin, native frames vs zoomed visits; per category A1
    zb = []
    for lo, hi in zip(Z_BINS[:-1], Z_BINS[1:]):
        nat = (v["z_native"] >= lo) & (v["z_native"] < hi)
        zom = v["zoomed"] & (v["z"] >= lo) & (v["z"] < hi)
        zb.append(dict(lo=float(lo), hi=float(hi), n_native=int(nat.sum()), c_native=float(v["c_native"][nat].mean()) if nat.any() else None,
                       n_zoomed=int(zom.sum()), c_zoomed=float(v["c"][zom].mean()) if zom.any() else None))
    cats = {}
    for cat in CATEGORIES:
        ks = [k for k, (c_, _) in enumerate(kept_rows) if c_ == cat]
        if not ks:
            continue
        d = []
        for k in ks:
            ix = (v["img"] == k) & v["zoomed"]
            if ix.any():
                d.append(v["c"][ix].mean() - v["c_native"][k])
        cats[cat] = dict(n=len(ks), A1=float(np.mean(d)) if d else None, c_native=float(v["c_native"][ks].mean()))
    res = dict(experiment="EXP-050 Part A", data_root=os.path.abspath(args.data_root), meta_json_sha256=sha256_file(meta_path),
               dataset_py_sha256=sha256_file(args.dataset_py), crop_box_source_sha256=box_sha, seed=args.seed, epochs=args.epochs, p=args.p,
               target=list(args.target), limit=args.limit, n_invalid_mask=invalid, parity=parity, stats=st, a1_label=l1, a2_label=l2, verdict=label,
               z_bins=zb, categories=cats)
    if args.limit:
        res["verdict_note"] = "LIMITED subset (smoke): not a registered result"
    json.dump(res, open(js, "w"), indent=1)
    print_part_a(res)


def print_part_a(res):
    st = res["stats"]
    f = lambda t: f"[{t[0]:+.4f}, {t[1]:+.4f}]"
    print(f"\n=== EXP-050 Part A (source geometry; {st['n_img']} images, {st['n_visits']} visits, zoomed fraction {st['frac_zoomed']:.3f}) ===")
    print(f"  mean c: native {st['c_native_mean']:.4f}   zoomed visits {st['c_zoomed_mean']:.4f}")
    print(f"  A1 = mean_i(c zoomed - c native) = {st['A1']:+.4f} {f(st['A1_ci'])}  -> {res['a1_label']}   (CENTRALISES <= {S.A1_SUPPORT} with CI upper < 0; none >= {S.A1_NOT})")
    print(f"  rho(z, c): mixture {st['rho_mix']:+.4f}, native {st['rho_native']:+.4f}, zoomed only {st['rho_zoomed']:+.4f}")
    print(f"  A2 = rho_mix - rho_native = {st['A2']:+.4f} {f(st['A2_ci'])}  -> {res['a2_label']}   (COUPLES <= {S.A2_SUPPORT} with CI upper < 0; none >= {S.A2_NOT})")
    print("  report-only, mean c by z bin (native frames | zoomed visits):")
    for b in res["z_bins"]:
        cn = "   -  " if b["c_native"] is None else f"{b['c_native']:.3f}"
        cz = "   -  " if b["c_zoomed"] is None else f"{b['c_zoomed']:.3f}"
        print(f"    z in [{b['lo']:+.2f}, {b['hi']:+.2f}):  native n={b['n_native']:4d} c={cn}   zoomed n={b['n_zoomed']:5d} c={cz}")
    print("  report-only, A1 per category: " + ", ".join(f"{k} {d['A1']:+.3f}" if d["A1"] is not None else f"{k} -" for k, d in res["categories"].items()))
    print(f"PART A VERDICT: {res['verdict']}" + (f"   ({res['verdict_note']})" if res.get("verdict_note") else ""))


# ------------------------------------------------------------------ EXP-051 source sweep of the zoom placement
BETA_GRID = (0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05)        # fixed before the sweep (notes/EXP-051_registration.md); largest first


def select_beta(results, grid=BETA_GRID):
    """results: {a: part_a stats}. Registered rule: the LARGEST a of the grid (mildest change from uniform) whose zoomed training mixture is no longer
    centralised or size-centre coupled relative to the native frames: A1 >= -0.02 and A2 >= -0.03 (point estimates, rounded to 1e-10).
    Returns a or None (no admissible a in the grid -> no training)."""
    for a in sorted(grid, reverse=True):
        st = results.get(a)
        if st is not None and round(st["A1"], 10) >= S.A1_NOT and round(st["A2"], 10) >= S.A2_NOT:
            return a
    return None


def source_masks(args):
    """-> (rows, generator factory). Same selection and order as Part A / the EXP-040 audit."""
    from PIL import Image
    meta = json.load(open(os.path.join(args.data_root, "meta.json")))
    if set(meta["test"]) != set(CATEGORIES):
        raise SystemExit("STOP: MVTec category list differs from the EXP-040 list")
    rows = [(cat, r) for cat in CATEGORIES for r in meta["test"][cat] if int(r["anomaly"]) == 1]
    if args.limit:
        rows = rows[: args.limit]

    def gen():
        for cat, r in rows:
            with Image.open(os.path.join(args.data_root, r["img_path"])) as im:
                size = im.size
            with Image.open(os.path.join(args.data_root, r["mask_path"])) as mi:
                if mi.size != size:
                    continue
                yield np.asarray(mi.convert("L")) > 0
    return rows, gen


def stage_sweep(args):
    out = args.sweep_dir
    os.makedirs(out, exist_ok=True)
    js = os.path.join(out, "selection.json")
    if os.path.exists(js) and not args.redo:
        print(f"{js} exists, skipped (use --redo)"); print(open(js).read()); return
    rows, gen = source_masks(args)
    res, tab = {}, []
    for place, a in [("uniform", 1.0)] + [("beta", b) for b in BETA_GRID]:
        f_box, box_sha = S.load_crop_box(args.dataset_py, place=place, beta=a)
        t0 = time.time()
        random.seed(args.seed)
        v = S.simulate_visits(gen(), lambda m: f_box(m, tuple(args.target)), epochs=args.epochs, p=args.p)
        st = S.part_a(v, n_boot=args.boot)
        zz = v["z"][v["zoomed"]]
        row = dict(place=place, a=a, attempts=v["n_attempts"], success=v["n_success"], A1=st["A1"], A1_ci=st["A1_ci"], A2=st["A2"], A2_ci=st["A2_ci"],
                   a1_label=S.a1_label(st), a2_label=S.a2_label(st), c_zoomed=st["c_zoomed_mean"], c_native=st["c_native_mean"],
                   rho_mix=st["rho_mix"], rho_native=st["rho_native"], z_zoomed_mean=float(zz.mean()) if len(zz) else None, seconds=time.time() - t0)
        tab.append(row)
        if place == "uniform":
            ok = (v["n_attempts"], v["n_success"]) == (EXP040_ATTEMPTS, EXP040_SUCCESS)
            print(f"[parity] uniform placement: attempts / successes {v['n_attempts']} / {v['n_success']} vs EXP-040 {EXP040_ATTEMPTS} / {EXP040_SUCCESS}: "
                  + ("MATCH" if ok else ("not compared (--limit)" if args.limit else "MISMATCH")))
            if not ok and not args.limit and not args.no_exp040_parity:
                raise SystemExit("STOP: the uniform crop stream differs from the EXP-040 audit")
        else:
            res[a] = st
        print(f"  {place:7s} a={a:<5}  A1 {st['A1']:+.4f} [{st['A1_ci'][0]:+.4f}, {st['A1_ci'][1]:+.4f}] {row['a1_label']:18s}  "
              f"A2 {st['A2']:+.4f} [{st['A2_ci'][0]:+.4f}, {st['A2_ci'][1]:+.4f}] {row['a2_label']:12s}  mean c zoomed {st['c_zoomed_mean']:.3f} (native {st['c_native_mean']:.3f})  "
              f"mean z zoomed {row['z_zoomed_mean']:.3f}  ({row['seconds']:.0f}s)", flush=True)
    a = select_beta(res)
    sel = dict(experiment="EXP-051 source sweep", rule="largest a in the grid with A1 >= -0.02 and A2 >= -0.03", grid=list(BETA_GRID), selected_beta=a,
               limit=args.limit, seed=args.seed, epochs=args.epochs, p=args.p, target=list(args.target),
               dataset_py_sha256=sha256_file(args.dataset_py), crop_box_source_sha256=box_sha, table=tab)
    if args.limit:
        sel["note"] = "LIMITED subset (smoke): not the registered selection"
    json.dump(sel, open(js, "w"), indent=1)
    print(f"SELECTED zoom_beta = {a}" if a is not None else "NO ADMISSIBLE a in the grid: Step 2 training is NOT authorised (report and stop)")
    print(f"wrote {js}")


# ------------------------------------------------------------------ Part B
def hi_dir(tag, args):
    return args.exp048 if tag == "ecp_s111" else os.path.join(args.exp049, tag)


def stage_zreport(args):
    import analyze_pe_shift as A        # numpy-only at import (torch is imported inside stage_pixel)
    seed_labels, table = {}, {}
    for tag in ECP_SEEDS:
        labels = {}
        for s in REPORT_SETS:
            ph = os.path.join(hi_dir(tag, args), s, f"pe_{s}.csv")
            pl = os.path.join(args.zlow, tag, s, f"pe_{s}.csv")
            if not (os.path.isfile(ph) and os.path.isfile(pl)):
                if s in S.DECISION:
                    labels[s] = None
                    print(f"MISSING {ph if not os.path.isfile(ph) else pl}")
                continue
            st = S.paired_g(A.load_rows(ph), A.load_rows(pl), n_boot=args.boot)
            lab = S.set_label_b(st)
            table[(tag, s)] = (st, lab)
            if s in S.DECISION:
                labels[s] = lab
        sv = S.seed_verdict_b(labels) if all(labels.get(s) is not None for s in S.DECISION) else "INCOMPLETE"
        seed_labels[tag] = sv
    print("\n=== EXP-050 Part B (frozen ECP, peripheral tercile; G = PE_REC - PE_RAND per image; hi = z_pix 1.6 [EXP-048/049], lo = z_pix -0.69) ===")
    print(f"{'ckpt':9s} {'set':9s} {'n':>4s} {'sham hi':>8s} {'sham lo':>8s} {'D_PE hi':>8s} {'D_PE lo':>8s} {'G hi':>8s} {'G lo':>8s} {'Q':>7s}  {'dG [95% CI]':24s} label")
    for (tag, s), (st, lab) in table.items():
        if st.get("n", 0) < 3:
            print(f"{tag:9s} {s:9s} {st.get('n', 0):4d}  too few paired images  {lab}"); continue
        rep = "" if s in S.DECISION else "  (report-only)"
        print(f"{tag:9s} {s:9s} {st['n']:4d} {st['sham_hi']:8.4f} {st['sham_lo']:8.4f} {st['dpe_hi']:+8.4f} {st['dpe_lo']:+8.4f} {st['G_hi']:+8.4f} {st['G_lo']:+8.4f} "
              f"{st['Q']:+7.3f}  {st['dG']:+.4f} [{st['dG_ci'][0]:+.4f}, {st['dG_ci'][1]:+.4f}]  {lab}{rep}")
    # report-only reference: CTRL G from EXP-049 (no conditioner, so no z)
    ref = []
    for t in ("ctrl_s111", "ctrl_s222", "ctrl_s333"):
        for s in S.DECISION:
            p = os.path.join(args.exp049, t, s, f"pe_{s}.csv")
            if os.path.isfile(p):
                k = [r for r in A.load_rows(p) if r["group"] == "peripheral" and r["status"] == "kept"]
                if k:
                    ref.append(f"{t}/{s} {np.mean([r['auroc_pe_rec'] - r['auroc_pe_rand'] for r in k]):+.4f}")
    if ref:
        print("  report-only, CTRL G (EXP-049, no z): " + ", ".join(ref))
    for tag, sv in seed_labels.items():
        print(f"  seed verdict {tag}: {sv}")
    v = S.verdict_b(seed_labels)
    print(f"PART B VERDICT: {v}")
    os.makedirs(args.zlow, exist_ok=True)
    json.dump(dict(seed_verdicts=seed_labels, verdict=v,
                   table={f"{t}/{s}": dict(label=lab, **{k: (list(x) if isinstance(x, tuple) else x) for k, x in st.items()}) for (t, s), (st, lab) in table.items()}),
              open(os.path.join(args.zlow, "part_b.json"), "w"), indent=1)
    return v


def stage_report(args):
    js = os.path.join(args.out_dir, "part_a.json")
    va = None
    if os.path.isfile(js):
        res = json.load(open(js)); print_part_a(res); va = res["verdict"] if not res.get("limit") else "INCOMPLETE (limited run)"
    else:
        print(f"MISSING {js}")
    vb = stage_zreport(args)
    print("\n================ EXP-050 (rule fixed in notes/EXP-050_registration.md) ================")
    print(f"Part A (zoom geometry): {va}")
    print(f"Part B (PE response along the extent axis): {vb}")
    print("Step 2 (position-debiased zoom training pilot) is authorised only if Part A is SUPPORTED.  Part B decides the paper wording only:"
          " Z-COUPLED allows 'the extent-conditioned pixel prompt carries the position prior'; otherwise do not say so.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("source", "zreport", "report", "sweep"))
    ap.add_argument("--data_root", help="MVTec root holding meta.json (Part A)")
    ap.add_argument("--dataset_py", default="dataset.py")
    ap.add_argument("--out_dir", default="results/EXP-050/source")
    ap.add_argument("--zlow", default="results/EXP-050/zlow", help="analyze_pe_shift.py outputs at z_pix -0.69: <zlow>/<ecp tag>/<set>/pe_<set>.csv")
    ap.add_argument("--exp048", default="results/EXP-048")
    ap.add_argument("--exp049", default="results/EXP-049")
    ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--target", type=float, nargs=2, default=(0.10, 0.45))
    ap.add_argument("--limit", type=int, default=0, help="first N anomalous images only (smoke; parity and verdict not registered)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--sweep_dir", default="results/EXP-051/sweep")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--no_exp040_parity", action="store_true")
    a = ap.parse_args()
    if a.mode in ("source", "sweep") and not a.data_root:
        ap.error(f"{a.mode} needs --data_root")
    {"source": stage_source, "zreport": stage_zreport, "report": stage_report, "sweep": stage_sweep}[a.mode](a)
