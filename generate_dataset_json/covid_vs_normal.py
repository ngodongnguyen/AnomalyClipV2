"""
Build a COVID-vs-Normal test root for the COVID-19 Radiography Database (EXP-056).

The original `covid.py` marks every folder except NORMAL as abnormal, so Lung_Opacity and Viral Pneumonia count as
anomalies too (about 52% anomalous). AnomalyCLIP's COVID-19 numbers match only COVID vs Normal (about 26% anomalous).
This script keeps exactly the folders named on the command line and writes a separate root with symlinks and a
meta.json, so the source dataset and any existing meta.json are never modified.

    python generate_dataset_json/covid_vs_normal.py --src <dataset root> --dst <new root>
        [--abnormal COVID] [--normal Normal] [--force]

If a species folder contains an `images` subfolder (the Kaggle layout), that subfolder is used.
"""
import argparse
import json
import os


def find_dir(src, name):
    """Case-insensitive match of a species folder; prefer its `images` subfolder."""
    for entry in sorted(os.listdir(src)):
        if entry.lower() == name.lower() and os.path.isdir(os.path.join(src, entry)):
            path = os.path.join(src, entry)
            images = os.path.join(path, "images")
            return entry, images if os.path.isdir(images) else path
    raise SystemExit(f"folder {name!r} not found in {src}; found: {sorted(os.listdir(src))}")


def build(src, dst, abnormal, normal, force=False):
    meta_path = os.path.join(dst, "meta.json")
    if os.path.exists(meta_path) and not force:
        raise SystemExit(f"{meta_path} exists; pass --force to overwrite")
    os.makedirs(dst, exist_ok=True)
    cls_info, counts = [], {}
    for label, wanted in ((1, abnormal), (0, normal)):
        entry, path = find_dir(src, wanted)
        link = os.path.join(dst, entry)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.abspath(path), link)
        names = sorted(n for n in os.listdir(path) if n.lower().endswith((".png", ".jpg", ".jpeg")))
        counts[entry] = len(names)
        for name in names:
            cls_info.append(dict(img_path=f"{entry}/{name}", mask_path="", cls_name="chest",
                                 specie_name=entry, anomaly=label))
    info = dict(train={}, test={"chest": cls_info})
    with open(meta_path, "w") as f:
        f.write(json.dumps(info, indent=4) + "\n")
    n_abn = sum(1 for r in cls_info if r["anomaly"] == 1)
    print("counts per folder:", counts)
    print(f"total {len(cls_info)}, anomalous {n_abn} ({100.0 * n_abn / max(len(cls_info), 1):.1f}%)")
    return counts, n_abn, len(cls_info)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--abnormal", default="COVID")
    ap.add_argument("--normal", default="Normal")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build(a.src, a.dst, a.abnormal, a.normal, a.force)
