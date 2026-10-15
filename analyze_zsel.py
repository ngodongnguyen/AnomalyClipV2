"""
EXP-052 / H041: source-only operating points for the extent axis (z_pix from the training extent labels, z_img from the other industrial dataset).
Rule and thresholds: notes/EXP-052_registration.md.  Statistics: zsel_stats.py (unit-tested).

  python analyze_zsel.py quantiles --dataset mvtec --data_root $MVTEC      # CPU, no model: training-mixture z of the source -> z_pix rule
  python analyze_zsel.py imgscan --tag ecp_s111 --checkpoint_path <ckpt> --dataset visa --data_path $VISA      # GPU: image AUROC per z on the other industrial set
  python analyze_zsel.py imgscan ... --dataset brain --data_path $HEADCT --parity_csv <test.py image_level_predictions.csv>   # parity smoke
  python analyze_zsel.py rule --tag ecp_s111 --source mvtec                  # prints "z_img z_pix" for the runner
  python analyze_zsel.py report                                              # rules on the medical evaluation (no torch)

In ECP 'extent' mode the prompt shift depends only on z when z is overridden (ExtentConditioner.forward), so the text features of each z are computed
once and the image score is exactly test.py's softmax(image_features . text(z_img) / 0.07)[anomalous] (checked by --parity_csv).
"""
import os
import csv
import json
import time
import random
import argparse
import numpy as np

import zsel_stats as Z
import centre_prior_stats as S

CKPTS = {   # tag -> (checkpoint folder under checkpoints/, source dataset it was trained on, the other industrial dataset)
    "ecp_s111": ("EXP-012-matched/ecp_extent/checkpoints", "mvtec", "visa"),
    "ecp_s222": ("matched_ecp_s222", "mvtec", "visa"),
    "ecp_s333": ("matched_ecp_s333", "mvtec", "visa"),
    "ecp_visa": ("ecp_visa", "visa", "mvtec"),
}
BASE = "results/EXP-052"


# ------------------------------------------------------------------ quantiles (CPU)
def stage_quantiles(args):
    from PIL import Image
    import analyze_centre_prior as ACP
    out = os.path.join(BASE, f"quantiles_{args.dataset}.json")
    if os.path.exists(out) and not args.redo:
        print(open(out).read()); return
    meta = json.load(open(os.path.join(args.data_root, "meta.json")))
    cats = list(ACP.CATEGORIES) if args.dataset == "mvtec" else list(meta["test"].keys())
    if args.dataset == "mvtec" and set(meta["test"]) != set(cats):
        raise SystemExit("STOP: MVTec category list differs from the EXP-040 list")
    rows = [(c, r) for c in cats for r in meta["test"][c] if int(r["anomaly"]) == 1]
    if args.limit:
        rows = rows[: args.limit]
    skipped = []

    def gen():
        for c, r in rows:
            with Image.open(os.path.join(args.data_root, r["img_path"])) as im:
                size = im.size
            mp = os.path.join(args.data_root, r["mask_path"]) if r.get("mask_path") else ""
            if not mp or not os.path.isfile(mp):
                skipped.append(r["img_path"]); continue
            with Image.open(mp) as mi:
                if mi.size != size:
                    skipped.append(r["img_path"]); continue
                yield np.asarray(mi.convert("L")) > 0
    f_box, sha = S.load_crop_box(args.dataset_py, place="uniform")
    random.seed(args.seed)
    t0 = time.time()
    v = S.simulate_visits(gen(), lambda m: f_box(m, (0.10, 0.45)), epochs=15, p=0.5)
    par = None
    if args.dataset == "mvtec":
        par = (v["n_attempts"], v["n_success"]) == (ACP.EXP040_ATTEMPTS, ACP.EXP040_SUCCESS)
        print(f"[parity] {v['n_attempts']} / {v['n_success']} vs EXP-040 {ACP.EXP040_ATTEMPTS} / {ACP.EXP040_SUCCESS}: "
              + ("MATCH" if par else ("not compared (--limit)" if args.limit else "MISMATCH")))
        if not par and not args.limit:
            raise SystemExit("STOP: the simulated crop stream differs from the EXP-040 audit")
    q = lambda a, p: float(np.percentile(a, p))
    res = dict(dataset=args.dataset, n_images=len(v["c_native"]), n_skipped=len(skipped), n_visits=len(v["z"]), attempts=v["n_attempts"],
               success=v["n_success"], exp040_parity=par, limit=args.limit, crop_box_sha256=sha, seed=args.seed,
               z_mix={str(p): q(v["z"], p) for p in (5, 25, 50, 75, 95, 99)}, z_native={str(p): q(v["z_native"], p) for p in (5, 25, 50, 75, 95, 99)},
               z_max=float(v["z"].max()), z_pix_rule=Z.z_pix_rule(v["z"]))
    os.makedirs(BASE, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"[quantiles] {args.dataset}: {res['n_images']} anomalous images ({res['n_skipped']} skipped: no usable mask), {res['n_visits']} visits ({time.time() - t0:.0f}s)")
    print(f"  training-mixture z percentiles {res['z_mix']}  max {res['z_max']:.3f}")
    print(f"  native-frame z percentiles     {res['z_native']}")
    print(f"  z_pix rule (P{Z.Q_PIX} of the training mixture) = {res['z_pix_rule']:.4f}   (deployed so far: 1.6)")


# ------------------------------------------------------------------ image scan over the z grid (GPU)
def stage_imgscan(args):
    import torch
    from dataset import Dataset
    from utils import get_transform
    from extent_prompt import visual_descriptor, conditioned_text_features
    from analyze_badcases import load_model
    out_dir = os.path.join(BASE, "imgscan", args.tag + (f"_{args.suffix}" if args.suffix else ""))
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "scores.csv")
    if os.path.exists(out_csv) and not args.redo:
        print(f"{out_csv} exists, skipped (use --redo)")
    else:
        device, model, pl, cond = load_model(args)
        if cond is None or cond.mode != "extent":
            raise SystemExit("STOP: imgscan needs an ECP 'extent' checkpoint")
        preprocess, target_transform = get_transform(args)
        data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
        idx = list(range(len(data)))
        if args.limit:
            idx = idx[: args.limit]
        with torch.no_grad():
            first = data[0]["img"].to(device).unsqueeze(0)
            imf, pf = model.encode_image(first, [24], DPAM_layer=20)
            imf = imf / imf.norm(dim=-1, keepdim=True)
            desc = visual_descriptor(imf, pf)
            tfs = {}
            for z in Z.Z_GRID:
                _, c_pos, c_neg = cond(desc, z_override=torch.full((1,), z, device=device))
                tfs[z] = conditioned_text_features(model, pl, c_pos, c_neg)[0]          # [2, C]; independent of the image (z overridden)
            # guard: the shift really is image-independent (a second image must give the same text features)
            if len(data) > 1:
                imf2, pf2 = model.encode_image(data[1]["img"].to(device).unsqueeze(0), [24], DPAM_layer=20)
                imf2 = imf2 / imf2.norm(dim=-1, keepdim=True)
                _, a, b = cond(visual_descriptor(imf2, pf2), z_override=torch.full((1,), Z.Z_GRID[0], device=device))
                d = (conditioned_text_features(model, pl, a, b)[0] - tfs[Z.Z_GRID[0]]).abs().max().item()
                print(f"[guard] text features at a fixed z differ between two images by {d:.2e}")
                if d > 1e-5:
                    raise SystemExit("STOP: the conditioned text features depend on the image; the precomputation is not valid")
            T = torch.stack([tfs[z] for z in Z.Z_GRID])                                     # [Zn, 2, C]
            rows, t0 = [], time.time()
            for n, i in enumerate(idx):
                it = data[i]
                f, _ = model.encode_image(it["img"].to(device).unsqueeze(0), [24], DPAM_layer=20)
                f = f / f.norm(dim=-1, keepdim=True)
                probs = ((torch.einsum("c,ztc->zt", f[0].float(), T.float()) / 0.07).softmax(-1)[:, 1]).cpu().numpy()
                rows.append([os.path.relpath(it["img_path"], args.data_path), it["cls_name"], int(it["anomaly"])] + [float(p) for p in probs])
                if (n + 1) % 250 == 0:
                    print(f"{n + 1}/{len(idx)} ({time.time() - t0:.0f}s)", flush=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["sample_id", "category", "label"] + [f"z{z:+.2f}" for z in Z.Z_GRID]); w.writerows(rows)
        print(f"wrote {out_csv} ({len(rows)} images)")
    rows = list(csv.DictReader(open(out_csv)))
    cats = [r["category"] for r in rows]; y = [int(r["label"]) for r in rows]
    macro = {z: Z.macro_auroc(cats, y, [float(r[f"z{z:+.2f}"]) for r in rows]) for z in Z.Z_GRID}
    for z in Z.Z_GRID:
        print(f"  z {z:+.2f}: macro image AUROC {macro[z] * 100:.2f}")
    if args.parity_csv:
        ref = {r["sample_id"]: float(r["image_anomaly_score"]) for r in csv.DictReader(open(args.parity_csv))}
        zc = f"z{args.parity_z:+.2f}"
        diffs = [abs(float(r[zc]) - ref[r["sample_id"]]) for r in rows if r["sample_id"] in ref]
        print(f"[parity] {len(diffs)}/{len(rows)} images joined with {args.parity_csv}; max |score - test.py| at z {args.parity_z} = {max(diffs) if diffs else float('nan'):.2e}")
        if not diffs or max(diffs) > 1e-4:
            raise SystemExit("STOP: image scores differ from test.py")
    if args.limit or args.parity_csv:
        print("(smoke / parity run: no selection written)"); return
    sel = dict(tag=args.tag, dataset=args.dataset, data_path=args.data_path, macro={str(z): v for z, v in macro.items()}, z_img_rule=Z.select_z_img(macro), grid=list(Z.Z_GRID))
    json.dump(sel, open(os.path.join(out_dir, "selection.json"), "w"), indent=1)
    print(f"SELECTED z_img = {sel['z_img_rule']}  (macro image AUROC {macro[sel['z_img_rule']] * 100:.2f} on {args.dataset})")


def rule_values(tag):
    ck, src, other = CKPTS[tag]
    q = json.load(open(os.path.join(BASE, f"quantiles_{src}.json")))
    s = json.load(open(os.path.join(BASE, "imgscan", tag, "selection.json")))
    if q.get("limit") or s.get("dataset") != other:
        raise SystemExit(f"STOP: invalid rule inputs for {tag}")
    return float(s["z_img_rule"]), float(q["z_pix_rule"])


def stage_rule(args):
    zi, zp = rule_values(args.tag)
    print(f"{zi:.4f} {zp:.4f}")


# ------------------------------------------------------------------ report (no torch)
def stage_report(args):
    import sys
    sys.path.insert(0, "scripts")
    import collect_exp024 as c
    base = c.load(".")
    print("================ EXP-052: source-only operating points ================")
    for t in CKPTS:
        try:
            zi, zp = rule_values(t); print(f"  {t:9s} z_img {zi:+.2f} (scan on {CKPTS[t][2]})   z_pix {zp:+.4f} (P{Z.Q_PIX} of {CKPTS[t][1]} training z)   deployed -0.69 / 1.6")
        except (FileNotFoundError, SystemExit) as e:
            print(f"  {t:9s} rule inputs missing ({e})")
    run = lambda t, key: c.last_table_row(f"results/{CKPTS[t][0]}_srcz_fixed/{key}/log.txt")
    seeds = (111, 222, 333)
    tags = {111: "ecp_s111", 222: "ecp_s222", 333: "ecp_s333"}
    pro_d, auc_d, img_d, cost = {}, {}, {}, dict(auc=[], pro=[], img=[])
    print("\nMVTec-trained ECP at the source rule vs zoom-only (ZO) and vs ECP at the deployed point; pooled AUROC / PRO, seed means")
    complete = True
    for name, key in c.PIX:
        r = [run(tags[s], key) for s in seeds]; zo = [base["pix"][name]["ZO"][s] for s in seeds]; dp = [base["pix"][name]["ECP"][s] for s in seeds]
        if any(v is None for v in r + zo + dp):
            print(f"  {name:9s} missing"); complete = False; cost["auc"].append(None); cost["pro"].append(None); continue
        pro_d[name] = [a[1] - b[1] for a, b in zip(r, zo)]; auc_d[name] = [a[0] - b[0] for a, b in zip(r, zo)]
        ca = float(np.mean([a[0] - b[0] for a, b in zip(r, dp)])); cp = float(np.mean([a[1] - b[1] for a, b in zip(r, dp)]))
        cost["auc"].append(ca); cost["pro"].append(cp)
        m = lambda vs, k: np.mean([v[k] for v in vs])
        print(f"  {name:9s} rule {m(r, 0):5.1f}/{m(r, 1):5.1f}  deployed {m(dp, 0):5.1f}/{m(dp, 1):5.1f}  ZO {m(zo, 0):5.1f}/{m(zo, 1):5.1f}   "
              f"rule-ZO {np.mean(auc_d[name]):+.2f}/{np.mean(pro_d[name]):+.2f}   rule-deployed {ca:+.2f}/{cp:+.2f}")
    for name, key in c.IMG:
        r = [run(tags[s], key) for s in seeds]; zo = [base["img"][name]["ZO"][s] for s in seeds]; dp = [base["img"][name]["ECP"][s] for s in seeds]
        if any(v is None for v in r + zo + dp):
            print(f"  {name:9s} missing"); complete = False; cost["img"].append(None); continue
        img_d[name] = [a[0] - b[0] for a, b in zip(r, zo)]
        ci = float(np.mean([a[0] - b[0] for a, b in zip(r, dp)])); cost["img"].append(ci)
        print(f"  {name:9s} image AUROC rule {np.mean([v[0] for v in r]):5.1f}  deployed {np.mean([v[0] for v in dp]):5.1f}  ZO {np.mean([v[0] for v in zo]):5.1f}   rule-ZO {np.mean(img_d[name]):+.2f}   rule-deployed {ci:+.2f}")
    if complete:
        print(f"R1 (rule ECP vs ZO, pixel; EXP-024 thresholds): {Z.r1_ecp_vs_zo(pro_d, auc_d)}")
        print(f"R3 (rule ECP vs ZO, image): {Z.r3_image(img_d)}")
        print(f"R2 (rule vs deployed, seed means): {Z.r2_cost(cost['auc'], cost['pro'], cost['img'])}")
    else:
        print("R1 / R2 / R3: INCOMPLETE")
    print("\nVisA-trained ECP (one seed) at the source rule vs EXP-020 references (ecp_visa at -0.69/1.6, zoom_visa, ctrl_visa)")
    ref = lambda ck, key: c.last_table_row(f"results/{ck}_fixed/{key}/log.txt")
    rule_img, ctrl_img = [], []
    for name, key in c.PIX:
        r, e, zv = run("ecp_visa", key), ref("ecp_visa", key), ref("zoom_visa", key)
        if r is None:
            print(f"  {name:9s} missing"); continue
        f = lambda v: "  -  /  -  " if v is None else f"{v[0]:5.1f}/{v[1]:5.1f}"
        print(f"  {name:9s} rule {f(r)}  deployed {f(e)}  zoom_visa {f(zv)}" + (f"   rule-zoom_visa {r[0] - zv[0]:+.1f}/{r[1] - zv[1]:+.1f}" if zv else ""))
    for name, key in c.IMG:
        r, e, ct = run("ecp_visa", key), ref("ecp_visa", key), ref("ctrl_visa", key)
        if r is None or ct is None:
            print(f"  {name:9s} missing"); rule_img.append(None); continue
        rule_img.append(r[0]); ctrl_img.append(ct[0])
        print(f"  {name:9s} image AUROC rule {r[0]:5.1f}  deployed {e[0] if e else float('nan'):5.1f}  ctrl_visa {ct[0]:5.1f}")
    if None not in rule_img and len(ctrl_img) == 3:
        print(f"VISA IMAGE HEAD: {Z.visa_image_recovered(rule_img, ctrl_img)}")
    else:
        print("VISA IMAGE HEAD: INCOMPLETE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("quantiles", "imgscan", "rule", "report"))
    ap.add_argument("--dataset", default="mvtec")
    ap.add_argument("--data_root"); ap.add_argument("--data_path")
    ap.add_argument("--dataset_py", default="dataset.py")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--suffix", default="", help="imgscan output sub-folder suffix (parity runs)")
    ap.add_argument("--checkpoint_path", default=None)
    ap.add_argument("--parity_csv", default=None, help="test.py image_level_predictions.csv to compare with (smoke)")
    ap.add_argument("--parity_z", type=float, default=-0.69)
    ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    a = ap.parse_args()
    if a.mode == "quantiles" and not a.data_root:
        ap.error("quantiles needs --data_root")
    if a.mode == "imgscan" and not (a.tag and a.checkpoint_path and a.data_path):
        ap.error("imgscan needs --tag, --checkpoint_path, --data_path")
    if a.mode == "rule" and a.tag not in CKPTS:
        ap.error(f"--tag must be one of {sorted(CKPTS)}")
    {"quantiles": stage_quantiles, "imgscan": stage_imgscan, "rule": stage_rule, "report": stage_report}[a.mode](a)
