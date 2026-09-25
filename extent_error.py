"""Extent-estimator error per dataset from per_image.csv files written by test.py (columns z_pred, z_gt).

    python extent_error.py results/ecp_extent/isic_s4/per_image.csv results/ecp_extent_pi/Kvasir/per_image.csv ...
Reports mean signed error (bias), mean absolute error, the range of GT extents and the correlation pred-vs-GT.
z = (ln(area) - ln 0.02) / 2  ->  |error| 0.5 means a factor e^1 = 2.7 in area.
"""
import csv
import sys
import numpy as np

print(f"{'file':<58}{'n':>5}{'bias':>8}{'MAE':>7}{'corr':>7}{'z_gt range':>16}")
for path in sys.argv[1:]:
    with open(path) as f:
        rows = [(float(r["z_pred"]), float(r["z_gt"])) for r in csv.DictReader(f) if r.get("z_pred") not in (None, "nan")]
    if not rows:
        print(f"{path:<58}  (no z_pred: not an ECP-extent run)")
        continue
    p, g = np.array(rows).T
    print(f"{path[-56:]:<58}{len(p):>5}{(p - g).mean():>+8.3f}{np.abs(p - g).mean():>7.3f}{np.corrcoef(p, g)[0, 1]:>+7.2f}"
          f"   [{g.min():+.2f},{g.max():+.2f}]")
