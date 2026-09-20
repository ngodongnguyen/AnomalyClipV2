"""Area-fraction distribution of anomalies in a dataset's meta.json (e.g. the auxiliary training sets MVTec / VisA)."""
import os
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


def main(args):
    root = args.data_path
    meta = json.load(open(os.path.join(root, "meta.json")))
    fracs = []
    for cls, items in meta[args.mode].items():
        for it in items:
            if it["anomaly"] != 1:
                continue
            mp = os.path.join(root, it["mask_path"])
            if not os.path.isfile(mp):
                continue
            m = np.array(Image.open(mp).convert("L")) > 127
            if m.any():
                fracs.append(m.mean())
    f = np.array(fracs)
    print(f"\n=== Anomaly area fraction: {os.path.basename(root.rstrip('/'))} (n={len(f)}) ===")
    qs = [10, 25, 50, 75, 90, 95, 99]
    print("percentiles  :", "  ".join(f"p{q}={np.percentile(f, q):.3f}" for q in qs))
    print("share above  :", "  ".join(f">{t:.2f}: {100 * (f > t).mean():.1f}%" for t in (0.05, 0.10, 0.20, 0.30, 0.40)))
    print("reference (mean lesion area fraction of polyp quartiles Q1..Q4):")
    print("  ClinicDB 0.04 0.07 0.12 0.23 | Kvasir 0.06 0.12 0.21 0.42 | ColonDB 0.05 0.07 0.12 0.34")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--mode", default="test")
    main(ap.parse_args())
