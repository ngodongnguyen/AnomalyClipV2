#!/usr/bin/env python3
"""Audit EXP-012 paired outputs and bundle the small files needed for analysis.

On the GPU server: python scripts/collect_exp012_artifacts.py
The default input is results/EXP-012/matched-retry-20261002.
"""

import argparse
import csv
import json
import math
import tarfile
from pathlib import Path


DATASETS = ("ISIC", "ClinicDB", "ColonDB", "Kvasir", "Endo", "TN3K")
ARMS = ("ecp_extent", "zoom_only")
FILES = (
    "pixel_level_metrics.json",
    "pixel_per_image_predictions.csv",
    "evaluation_metadata.json",
    "training_manifest.json",
    "command.txt",
    "stdout_stderr.log",
    "per_image.csv",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_run(folder):
    for name in FILES:
        require((folder / name).is_file(), f"missing {folder / name}")
    metrics = json.loads((folder / "pixel_level_metrics.json").read_text())
    metadata = json.loads((folder / "evaluation_metadata.json").read_text())
    manifest = json.loads((folder / "training_manifest.json").read_text())
    with (folder / "pixel_per_image_predictions.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "mask_area_fraction", "per_image_pixel_auroc"}
        require(required <= set(reader.fieldnames or []), f"missing CSV columns: {folder}")
        rows = {}
        for row in reader:
            sample_id = row["sample_id"]
            require(sample_id and sample_id not in rows, f"empty/duplicate sample ID: {folder}")
            area = float(row["mask_area_fraction"])
            auroc = float(row["per_image_pixel_auroc"])
            require(math.isfinite(area) and 0 < area < 1, f"invalid mask area: {folder}/{sample_id}")
            require(math.isfinite(auroc) and 0 <= auroc <= 1, f"invalid AUROC: {folder}/{sample_id}")
            rows[sample_id] = (area, auroc)
    count = metadata["sample_count"]
    excluded = metadata["images_excluded_from_per_image_auc"]
    require(len(rows) + excluded == count, f"CSV/sample count mismatch: {folder}")
    require(metadata["images_with_binary_masks"] == len(rows), f"mask count mismatch: {folder}")
    require(metadata["positive_count"] + metadata["negative_count"] == count,
            f"image label count mismatch: {folder}")
    for name, values in metrics.items():
        for key in ("pixel_auroc", "pixel_aupro"):
            value = float(values[key])
            require(math.isfinite(value) and 0 <= value <= 1,
                    f"invalid {name}/{key}: {folder}")
    require("mean" in metrics, f"missing mean metrics: {folder}")
    require(manifest["checkpoint_sha256"] == metadata["checkpoint_sha256"],
            f"checkpoint hash mismatch: {folder}")
    return metrics, metadata, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("results/EXP-012/matched-retry-20261002"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/EXP-012/exp012_analysis_bundle.tar.gz"))
    args = parser.parse_args()
    require(args.root.is_dir(), f"missing EXP-012 output directory: {args.root}")
    runs = {}
    common_source = None
    for dataset in DATASETS:
        for arm in ARMS:
            folder = args.root / dataset / arm
            runs[dataset, arm] = read_run(folder)
            source = runs[dataset, arm][1]["evaluation_source_sha256"]
            if common_source is None:
                common_source = source
            require(source == common_source, f"evaluation source changed: {folder}")

    print("dataset     paired  excluded  pixel AUROC ECP/zoom (delta pp)  AUPRO ECP/zoom (delta pp)")
    for dataset in DATASETS:
        a_metrics, a_meta, a_rows = runs[dataset, ARMS[0]]
        b_metrics, b_meta, b_rows = runs[dataset, ARMS[1]]
        for key in ("meta_json_sha256", "sample_count", "dataset_mode", "sigma", "seed"):
            require(a_meta[key] == b_meta[key], f"{dataset}: {key} differs between arms")
        for key in ("features_list", "image_size", "depth", "n_ctx", "t_n_ctx",
                    "distractor_suppress", "distractor_rerank"):
            require(a_meta["arguments"][key] == b_meta["arguments"][key],
                    f"{dataset}: {key} differs between arms")
        require(a_rows.keys() == b_rows.keys(), f"{dataset}: sample IDs differ between arms")
        for sample_id in a_rows:
            require(abs(a_rows[sample_id][0] - b_rows[sample_id][0]) <= 1e-12,
                    f"{dataset}: mask area differs for {sample_id}")
        require(a_meta["images_excluded_from_per_image_auc"] ==
                b_meta["images_excluded_from_per_image_auc"],
                f"{dataset}: excluded mask counts differ")
        a = a_metrics["mean"]
        b = b_metrics["mean"]
        auc_delta = 100 * (a["pixel_auroc"] - b["pixel_auroc"])
        pro_delta = 100 * (a["pixel_aupro"] - b["pixel_aupro"])
        print(f"{dataset:<11} {len(a_rows):>6}  {a_meta['images_excluded_from_per_image_auc']:>8}  "
              f"{100*a['pixel_auroc']:6.2f}/{100*b['pixel_auroc']:6.2f} ({auc_delta:+6.2f})  "
              f"{100*a['pixel_aupro']:6.2f}/{100*b['pixel_aupro']:6.2f} ({pro_delta:+6.2f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    require(not args.out.exists(), f"bundle already exists: {args.out}")
    with tarfile.open(args.out, "w:gz") as bundle:
        for dataset in DATASETS:
            for arm in ARMS:
                folder = args.root / dataset / arm
                for name in FILES:
                    bundle.add(folder / name, arcname=f"runs/{dataset}/{arm}/{name}")
        inventory = args.root.parent / "input_inventory.json"
        if inventory.is_file():
            bundle.add(inventory, arcname="input_inventory.json")
    print(f"EXP-012 artifact audit passed; bundle: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"EXP-012 artifact audit failed: {exc}") from exc
