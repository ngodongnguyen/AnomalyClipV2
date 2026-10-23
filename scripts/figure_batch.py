"""SERVER: make the panels for the qualitative figures from figures/selection.csv (needs torch, GPU; calls scripts/figure_panels.py once per image/checkpoint).
  python scripts/figure_batch.py                      # all images of the selection, CTRL / ZO / ECP (ECP also at 5 z values)
  python scripts/figure_batch.py --tags typical       # only the 'typical' rows
Output: figure_panels/<set>/<set>_<tag>_<ck>_{image,gt}.png and _overlay_z<z>.png, ready for  python3 scripts/make_figures.py qual|zsweep
The mask path of each image is looked up in the dataset's meta.json (img_path == id).  Nothing here changes any model or result."""
import os
import csv
import sys
import json
import argparse
import subprocess

A = "/home/ai3/NguyenND/AnomalyClipV2/data"
CVC = f"{A}/CVC"
ROOTS = {"ClinicDB": f"{CVC}/CVC-ClinicDB", "Kvasir": f"{CVC}/Kvasir", "ColonDB": f"{CVC}/CVC-ColonDB", "ISIC": f"{A}/ISIC",
         "Endo": f"{A}/EndoTect_2020_Segmentation_Test_Dataset", "TN3K": f"{A}/TN3K/Thyroid Dataset/tn3k"}
CKPT = {"ctrl": "checkpoints/matched_ctrl_s111/epoch_15.pth",
        "zo": "checkpoints/EXP-012-matched/zoom_only/checkpoints/epoch_15.pth",
        "ecp": "checkpoints/EXP-012-matched/ecp_extent/checkpoints/epoch_15.pth"}
ZS = {"ctrl": ["1.6"], "zo": ["1.6"], "ecp": ["-0.69", "0.0", "0.8", "1.6", "2.4"]}


def find_mask(root, img_id):
    with open(os.path.join(root, "meta.json")) as f:
        meta = json.load(f)["test"]
    for recs in meta.values():
        for r in recs:
            if r["img_path"] == img_id:
                return r["mask_path"]
    raise SystemExit(f"{img_id} not found in {root}/meta.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="figures/selection.csv")
    ap.add_argument("--tags", nargs="+", default=["typical", "hard"])
    ap.add_argument("--out_dir", default="figure_panels")
    a = ap.parse_args()
    with open(a.selection) as f:
        sel = [r for r in csv.DictReader(f) if r["tag"] in a.tags]
    for r in sel:
        root = ROOTS[r["set"]]
        img = os.path.join(root, r["id"])
        mask = os.path.join(root, find_mask(root, r["id"]))
        for ck, ckpt in CKPT.items():
            name = f"{r['set']}_{r['tag']}_{ck}"
            cmd = [sys.executable, "scripts/figure_panels.py", "--image", img, "--mask", mask, "--checkpoint_path", ckpt,
                   "--out_dir", os.path.join(a.out_dir, r["set"]), "--name", name, "--z_list", *ZS[ck]]
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
