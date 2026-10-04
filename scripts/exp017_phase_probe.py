#!/usr/bin/env python3
"""Frozen ECP shift/phase probe with exact EXP-012 identity-map parity."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from exp014_source_readout import DATASETS, ECP_SHA, FrozenBranches, dataset, load_reference, provenance, sha
from exp014_readout_core import auc
from exp017_phase_core import phase_statistics, pilot_gate, stable_key

PILOT_DATASETS = ("ClinicDB", "ColonDB", "Kvasir")
OFFSETS = {f"{group}_{axis}{sign}":
           ((shift if sign == "pos" else -shift), 0) if axis == "x" else
           (0, (shift if sign == "pos" else -shift))
           for group, shift in (("half", 7), ("full", 14))
           for axis in ("x", "y") for sign in ("pos", "neg")}


def translated_reflect(image, dx, dy):
    """Translation of the preprocessed tensor; no resizing or relabeling."""
    import torch.nn.functional as F
    if image.ndim != 3 or image.shape[-2:] != (518, 518):
        raise ValueError("expected preprocessed 3x518x518 image")
    pad = 14
    padded = F.pad(image.unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
    return padded[0, :, pad-dy:pad-dy+518, pad-dx:pad-dx+518]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def manifest(args):
    import torch
    meta = provenance(args)  # reuses EXP-012 checkpoint/CLIP/source validations
    meta.update({"phase_probe_sha256": sha(__file__),
                 "phase_core_sha256": sha(ROOT / "scripts/exp017_phase_core.py"),
                 "exp014_model_helper_sha256": sha(ROOT / "scripts/exp014_source_readout.py"),
                 "exp014_stats_helper_sha256": sha(ROOT / "scripts/exp014_readout_core.py"),
                 "command": shlex.join([sys.executable, *sys.argv]),
                 "torch": str(torch.__version__), "cuda": str(torch.version.cuda)})
    return meta


def checked_previous(path, meta, stage):
    previous = json.loads((path / "metadata.json").read_text())
    if previous["stage"] != stage or previous["provenance"] != meta:
        # Commands/output paths differ by stage, so compare identities only.
        old = previous["provenance"]
        keys = ("checkpoint_sha256", "clip_sha256", "source_meta_sha256",
                "phase_probe_sha256", "phase_core_sha256",
                "exp014_model_helper_sha256", "exp014_stats_helper_sha256")
        if previous["stage"] != stage or any(old[k] != meta[k] for k in keys):
            raise ValueError(f"{stage} provenance differs from current code/weights")
    return json.loads((path / "summary.json").read_text())


def run(args):
    import numpy as np
    import torch
    from scipy.ndimage import gaussian_filter

    if args.out.exists():
        raise FileExistsError(args.out)
    meta = manifest(args)
    if args.stage in ("pilot", "full"):
        smoke = checked_previous(args.out.parent / "smoke", meta, "smoke")
        if smoke["max_reference_abs_error"] > 1e-6 or smoke["counts"] != {"ClinicDB": 2}:
            raise ValueError("EXP-017 smoke did not pass")
    if args.stage == "full":
        pilot = checked_previous(args.out.parent / "pilot", meta, "pilot")
        if not pilot["prospective_gate"]["pass"]:
            raise ValueError("pilot gate failed; full six-set extension is not authorized by protocol")
    names = ("ClinicDB",) if args.stage == "smoke" else PILOT_DATASETS if args.stage == "pilot" else tuple(DATASETS)
    model = FrozenBranches(args)
    args.out.mkdir(parents=True)
    all_rows = {}
    exclusions = {}
    max_error = 0.0
    try:
        for name in names:
            folder, mode = DATASETS[name]
            reference = load_reference(args, name)
            ds = dataset(args, folder, mode)
            indices = sorted(range(len(ds)), key=lambda i: stable_key(ds.data_all[i]["img_path"]))
            if args.stage == "pilot":
                indices = indices[:96]
            rows = []
            excluded = []
            for j, i in enumerate(indices, 1):
                item = ds[i]
                sample_id = str(Path(item["img_path"]).relative_to(args.data_root / folder))
                if sample_id not in reference:
                    raise ValueError(f"{name}/{sample_id}: missing EXP-012 reference")
                mask = item["img_mask"][0].numpy() > 0.5
                image = item["img"]
                baseline = model(image, baseline=True)["baseline"]
                parity = auc(mask.ravel(), baseline.ravel())
                if parity is None:
                    raise ValueError(f"{name}/{sample_id}: binary mask absent")
                error = abs(parity - reference[sample_id])
                max_error = max(max_error, error)
                if error > 1e-6:
                    raise ValueError(f"{name}/{sample_id}: identity AUROC mismatch {error}")
                interior = mask[14:-14, 14:-14]
                if not (interior.any() and (~interior).any()):
                    excluded.append({"sample_id": sample_id, "reason": "one mask class absent from common interior"})
                    continue
                maps = {"identity": baseline}
                for view, (dx, dy) in OFFSETS.items():
                    shifted = translated_reflect(image, dx, dy)
                    score = model(shifted, baseline=True)["baseline"]
                    from exp017_phase_core import inverse_align
                    maps[view] = inverse_align(score, dx, dy)
                # Starting map already has sigma=4; these are strong smoothing-only controls.
                for sigma in (8, 16, 32):
                    maps[f"smooth{sigma}"] = gaussian_filter(
                        baseline, sigma=(sigma*sigma - 4*4)**0.5)
                measures = phase_statistics(mask, maps)
                rows.append({"dataset": name, "sample_id": sample_id,
                             "baseline_full_pixel_auroc": parity,
                             "exp012_baseline_abs_error": error, **measures})
                if j % 20 == 0 or j == len(indices):
                    print(f"EXP-017 {name}: {j}/{len(indices)}", flush=True)
                if args.stage == "smoke" and len(rows) == 2:
                    break
            if args.stage == "full" and {r["sample_id"] for r in rows} | {r["sample_id"] for r in excluded} != set(reference):
                raise ValueError(f"{name}: full split is incomplete")
            if not rows:
                raise ValueError(f"{name}: no eligible images")
            with (args.out / f"{name}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            all_rows[name] = rows
            exclusions[name] = excluded
    finally:
        model.close()
    summary = {"stage": args.stage, "counts": {k: len(v) for k, v in all_rows.items()},
               "exclusions": exclusions,
               "max_reference_abs_error": max_error,
               "mean_metrics": {k: {m: float(np.mean([r[m] for r in rows]))
                                   for m in ("half", "full", "auc_identity", "auc_ensemble_half",
                                             "auc_ensemble_full", "auc_smooth8", "auc_smooth16", "auc_smooth32")}
                                for k, rows in all_rows.items()}}
    if args.stage == "pilot":
        summary["prospective_gate"] = pilot_gate(all_rows)
    write_json(args.out / "summary.json", summary)
    write_json(args.out / "metadata.json", {"stage": args.stage, "provenance": meta,
               "dataset_meta_sha256": {k: sha(args.data_root / DATASETS[k][0] / "meta.json")
                                       for k in names},
               "reference_csv_sha256": {k: sha(args.reference_root / k / "ecp_extent/pixel_per_image_predictions.csv")
                                        for k in names},
               "offsets": OFFSETS, "common_interior_border_px": 14,
               "source_order": "SHA256(EXP-017:111:full_image_path)",
               "note": "Frozen model; GT used only to score candidate maps"})
    print(f"EXP-017 {args.stage.upper()} COMPLETED: {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "pilot", "full"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip-weights", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    random.seed(111)
    np.random.seed(111)
    torch.manual_seed(111)
    torch.cuda.manual_seed_all(111)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    run(args)


if __name__ == "__main__":
    main()
