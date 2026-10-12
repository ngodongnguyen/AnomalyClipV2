"""Apply the registered EXP-038 one-seed pilot gate; run from repository root.

Reads results/EXP-038/exp038_static_aux_s111 and existing EXP-024 reference artifacts.
No result is imputed when an artifact is missing.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import collect_exp024 as ref

ROOT = "results/EXP-038/exp038_static_aux_s111"


def new_value(kind, key):
    folder = os.path.join(ROOT, key)
    if kind == "pix":
        value = ref.json_pixel(os.path.join(folder, "pixel_level_metrics.json"))
        if value is not None:
            return value
    return ref.last_table_row(os.path.join(folder, "log.txt"))


def decide(pixel_deltas, image_deltas):
    if len(pixel_deltas) != 6 or len(image_deltas) != 3:
        return "INCOMPLETE"
    pro_wins = sum(pro >= 1.0 for auc, pro in pixel_deltas)
    auc_wins = sum(auc >= 0.3 for auc, pro in pixel_deltas)
    any_big_pixel_loss = any(auc < -1.0 or pro < -1.0 for auc, pro in pixel_deltas)
    any_big_image_loss = any(auc < -1.0 for auc, ap in image_deltas)
    if pro_wins >= 4 and auc_wins >= 4 and not any_big_pixel_loss and not any_big_image_loss:
        return "ADVANCE"
    if pro_wins <= 2 or sum(auc <= -0.5 for auc, pro in pixel_deltas) >= 3:
        return "FAIL"
    return "INCONCLUSIVE"


def main():
    baseline = ref.load(".")
    pixel_deltas, image_deltas = [], []
    missing = []
    for kind, sets in (("pix", ref.PIX), ("img", ref.IMG)):
        print("\n" + ("pixel AUROC / PRO" if kind == "pix" else "image AUROC / AP"))
        for name, key in sets:
            value = new_value(kind, key)
            comparison = {arm: baseline[kind][name][arm][111] for arm in ref.ARMS}
            if value is None or any(v is None for v in comparison.values()):
                missing.append(f"{kind}/{name}")
                print(f"{name}: MISSING new or reference result")
                continue
            differences = {arm: (value[0] - v[0], value[1] - v[1]) for arm, v in comparison.items()}
            (pixel_deltas if kind == "pix" else image_deltas).append(differences["ECP"])
            print(f"{name:9s} static {value[0]:.2f}/{value[1]:.2f}  "
                  + "  ".join(f"vs {arm} {d[0]:+.2f}/{d[1]:+.2f}" for arm, d in differences.items()))
    print("\nEXP-038 pilot:", decide(pixel_deltas, image_deltas))
    if missing:
        print("Missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
