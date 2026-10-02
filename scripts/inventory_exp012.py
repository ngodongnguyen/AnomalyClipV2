#!/usr/bin/env python3
"""Read-only EXP-012 input inventory; run with the server's project Python.

Example:
  python scripts/inventory_exp012.py \
    --ecp-checkpoint checkpoints/ecp_extent/epoch_15.pth \
    --zoom-checkpoint checkpoints/zoom_mvtec/epoch_15.pth \
    --data-root data --clip-weights "$HOME/.cache/clip/ViT-L-14-336px.pt" \
    --out results/EXP-012/input_inventory.json

This cannot prove training zoom probability, seed, split, or other training
arguments: train.py does not store them in the checkpoint. Verify the original
training commands/logs before treating the two models as augmentation-matched.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys


DATASETS = (
    ("ISIC", "ISIC", "skin"),
    ("ClinicDB", "CVC/CVC-ClinicDB", "colon"),
    ("ColonDB", "CVC/CVC-ColonDB", "colon"),
    ("Kvasir", "CVC/Kvasir", "colon"),
    ("Endo", "EndoTect_2020_Segmentation_Test_Dataset", "colon"),
    ("TN3K", "TN3K/Thyroid Dataset/tn3k", "thyroid"),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_file(path, description):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{description} missing: {path}")
    return path


def checkpoint_info(path, expected_mode):
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # compatibility with the server's older PyTorch
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "prompt_learner" not in checkpoint:
        raise ValueError(f"checkpoint has no prompt_learner state: {path}")
    mode = checkpoint.get("extent_cond", "none")
    if mode != expected_mode:
        raise ValueError(f"{path}: expected extent_cond={expected_mode!r}, got {mode!r}")
    if expected_mode == "extent" and "conditioner" not in checkpoint:
        raise ValueError(f"ECP checkpoint has no conditioner state: {path}")
    if expected_mode == "none" and "conditioner" in checkpoint:
        raise ValueError(f"zoom-only checkpoint unexpectedly has conditioner state: {path}")
    shapes = {name: list(value.shape) for name, value in checkpoint["prompt_learner"].items()}
    return {"path": str(path), "sha256": sha256(path), "extent_cond": mode,
            "prompt_shapes": shapes}


def dataset_info(data_root, name, relative, expected_class):
    root = (data_root / relative).resolve()
    meta_path = required_file(root / "meta.json", f"{name} meta.json")
    meta = json.loads(meta_path.read_text())
    test = meta.get("test")
    if not isinstance(test, dict) or set(test) != {expected_class}:
        raise ValueError(f"{name}: expected test class {expected_class!r}; got {list(test) if isinstance(test, dict) else test!r}")
    samples = test[expected_class]
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{name}: empty or invalid test split")
    seen = set()
    counts = {"normal": 0, "anomalous": 0}
    manifest = hashlib.sha256()
    for index, item in enumerate(samples):
        if not isinstance(item, dict) or item.get("cls_name") != expected_class:
            raise ValueError(f"{name} sample {index}: invalid class metadata")
        label = item.get("anomaly")
        if label not in (0, 1):
            raise ValueError(f"{name} sample {index}: anomaly label must be 0 or 1")
        image = required_file(root / item["img_path"], f"{name} sample {index} image")
        key = item["img_path"]
        if key in seen:
            raise ValueError(f"{name}: duplicate image path in test split: {key}")
        seen.add(key)
        counts["anomalous" if label else "normal"] += 1
        mask_record = None
        if label:
            mask = required_file(root / item["mask_path"], f"{name} sample {index} positive mask")
            mask_stat = mask.stat()
            mask_record = [str(mask), mask_stat.st_size, mask_stat.st_mtime_ns]
        image_stat = image.stat()
        record = [key, label, str(image), image_stat.st_size, image_stat.st_mtime_ns, mask_record]
        manifest.update((json.dumps(record, separators=(",", ":")) + "\n").encode())
    if not counts["anomalous"]:
        raise ValueError(f"{name}: no positive masks to analyze")
    return {"path": str(root), "class": expected_class, "split": "test",
            "meta_json_sha256": sha256(meta_path), "file_manifest_sha256": manifest.hexdigest(),
            "sample_count": len(samples), **counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecp-checkpoint", required=True)
    parser.add_argument("--zoom-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--clip-weights", required=True)
    parser.add_argument("--expected-ecp-sha256", help="optional EXP-011 ECP checkpoint hash")
    parser.add_argument("--out", help="write the inventory JSON to this path (no overwrite)")
    args = parser.parse_args()
    ecp_path = required_file(args.ecp_checkpoint, "ECP checkpoint")
    zoom_path = required_file(args.zoom_checkpoint, "zoom-only checkpoint")
    clip_path = required_file(args.clip_weights, "CLIP weights")
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise ValueError(f"data root missing: {data_root}")
    ecp = checkpoint_info(ecp_path, "extent")
    zoom = checkpoint_info(zoom_path, "none")
    if ecp["prompt_shapes"] != zoom["prompt_shapes"]:
        raise ValueError("ECP and zoom-only prompt tensor names/shapes differ")
    if args.expected_ecp_sha256 and ecp["sha256"] != args.expected_ecp_sha256:
        raise ValueError("ECP checkpoint hash differs from the expected EXP-011 checkpoint")
    report = {
        "experiment": "EXP-012",
        "ecp": ecp,
        "zoom_only": zoom,
        "clip_weights": {"path": str(clip_path), "sha256": sha256(clip_path)},
        "datasets": {name: dataset_info(data_root, name, rel, cls)
                     for name, rel, cls in DATASETS},
        "training_provenance": "UNVERIFIED: checkpoints omit zoom_aug_p, training seed, training split and full arguments; check original command/log records for BOTH arms",
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as handle:
            handle.write(serialized)
        print(f"EXP-012 input inventory written to {destination}")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"EXP-012 inventory error: {exc}", file=sys.stderr)
        raise SystemExit(1)
