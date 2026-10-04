#!/usr/bin/env python3
"""EXP-015: read-only preserved-local-content / distant-context diagnostic.

Stages: calibrate (source MVTec only), smoke (two ClinicDB images), pilot
(48 or coverage-triggered 96 images on each of three development datasets).
No training, checkpoint modification, or target-data parameter selection.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp015_probe_core import (GRID, PATCH, PHOTO_GAMMAS, center_label,
                               intervention, paired_auc, select_centers, stable_key)

DATASETS = {
    "ClinicDB": ("CVC/CVC-ClinicDB", "colon"),
    "ColonDB": ("CVC/CVC-ColonDB", "colon"),
    "Kvasir": ("CVC/Kvasir", "colon"),
}
ECP_SHA = "7278af957c4087d7c00e7495e4cba3a65c253b5d0c9646b36434b951e3df0e7c"
CONFIG = dict(seed=111, image_size=518, feature=24, sigma=4, z_pix=1.6,
              depth=9, n_ctx=12, t_n_ctx=4, first_images=48,
              expanded_images=96, required_eligible_images=30,
              high_centers=3, middle_centers=3, protected_patches=7,
              photo_gammas=list(PHOTO_GAMMAS), bootstrap_repetitions=2000)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def csv_rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def save_rows(path, rows, columns):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def seed_all():
    import numpy as np
    import torch
    random.seed(111)
    np.random.seed(111)
    torch.manual_seed(111)
    torch.cuda.manual_seed_all(111)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def provenance(args):
    import torch
    if not torch.cuda.is_available():
        raise ValueError("EXP-015 requires the project CUDA environment")
    for p in (args.checkpoint, args.clip_weights, args.manifest,
              args.data_root / "mvtec/meta.json"):
        if not p.is_file():
            raise FileNotFoundError(p)
    cache = Path.home() / ".cache/clip/ViT-L-14-336px.pt"
    if args.clip_weights.resolve() != cache.resolve():
        raise ValueError("AnomalyCLIP_lib.load reads the home CLIP cache; CLIP_WEIGHTS must name it")
    manifest = read(args.manifest)
    if sha(args.checkpoint) != ECP_SHA or manifest["checkpoint_sha256"] != ECP_SHA:
        raise ValueError("wrong matched ECP checkpoint")
    if manifest["extent_cond"] != "extent" or manifest["seed"] != 111:
        raise ValueError("wrong ECP training manifest")
    if sha(args.clip_weights) != manifest["clip_weights_sha256"]:
        raise ValueError("CLIP weights differ from matched ECP training")
    if sha(args.data_root / "mvtec/meta.json") != manifest["mvtec_meta_sha256"]:
        raise ValueError("source MVTec partition differs")
    source_hashes = {rel: sha(ROOT / rel) for rel in
                     ("scripts/exp015_context_probe.py", "scripts/exp015_probe_core.py",
                      "scripts/run_exp015_probe.sh", "tests/test_exp015_probe_core.py")}
    source_hashes["mvtec/meta.json"] = sha(args.data_root / "mvtec/meta.json")
    refs = {}
    for name, (folder, mode) in DATASETS.items():
        ref = args.reference_root / name / "ecp_extent"
        meta = read(ref / "evaluation_metadata.json")
        expected = dict(metrics="pixel-level", dataset=mode, features_list=[24],
                        image_size=518, depth=9, n_ctx=12, t_n_ctx=4, sigma=4,
                        seed=111, ec_z_img=-0.69, ec_z_pix=1.6, ec_const_z=None,
                        ec_oracle=False, distractor_suppress="none",
                        distractor_rerank="none")
        if any(meta["arguments"].get(k) != v for k, v in expected.items()):
            raise ValueError(f"{name}: EXP-012 settings differ")
        if meta["checkpoint_sha256"] != ECP_SHA or meta["pretrained_clip_weights_sha256"] != sha(args.clip_weights):
            raise ValueError(f"{name}: reference weights differ")
        if meta["meta_json_sha256"] != sha(args.data_root / folder / "meta.json"):
            raise ValueError(f"{name}: reference split differs")
        for rel, digest in meta["evaluation_source_sha256"].items():
            if sha(ROOT / rel) != digest:
                raise ValueError(f"{name}: evaluation dependency changed: {rel}")
        refs[name] = {p: sha(ref / p) for p in
                      ("evaluation_metadata.json", "pixel_per_image_predictions.csv")}
    return dict(config=CONFIG, checkpoint_sha256=ECP_SHA,
                clip_sha256=sha(args.clip_weights), source_sha256=source_hashes,
                references=refs, python=sys.version.split()[0],
                torch=str(torch.__version__), cuda=torch.version.cuda)


def source_calibration(args, prov):
    import numpy as np
    from dataset import Dataset
    from utils import get_transform
    transform, mask_transform = get_transform(argparse.Namespace(image_size=518))
    ds = Dataset(str(args.data_root / "mvtec"), transform, mask_transform,
                 dataset_name="mvtec", mode="test")
    indexed = sorted(((i, ds.data_all[i]["img_path"]) for i in range(len(ds))),
                     key=lambda t: stable_key(t[1]))
    cal = [(i, p) for i, p in indexed if int(stable_key(p)[:8], 16) % 5 != 4][:32]
    held = [(i, p) for i, p in indexed if int(stable_key(p)[:8], 16) % 5 == 4][:32]
    if len(cal) != 32 or len(held) != 32:
        raise ValueError("source diagnostic partition is too small")
    changes = defaultdict(list)
    for i, sample_id in cal:
        image = ds[i]["img"].unsqueeze(0)
        for kind in ("sham", "photo", "rearrange", "near_photo"):
            edited = intervention(image, 18, 18, kind, sample_id)
            changes[kind].append(float((edited - image).abs().mean()))
    if max(changes["sham"]) != 0:
        raise ValueError("source sham is not an exact identity")
    if any(min(changes[k]) <= 0 for k in ("photo", "rearrange", "near_photo")):
        raise ValueError("one source intervention did not change image pixels")
    out = args.out_root / "calibration.json"
    if out.exists():
        raise FileExistsError(out)
    args.out_root.mkdir(parents=True, exist_ok=True)
    write(out, dict(provenance=prov, source_partition_rule="SHA256(EXP-015:111:img_path) first32 per modulo5 bucket",
                    calibration_ids=[p for _, p in cal], held_out_ids=[p for _, p in held],
                    input_mean_abs_change={k: dict(min=float(np.min(v)),
                                                   median=float(np.median(v)),
                                                   max=float(np.max(v)))
                                           for k, v in changes.items()},
                    command=shlex.join([sys.executable, *sys.argv])))
    print("EXP-015 SOURCE CALIBRATION PASSED", flush=True)


def load_model(args):
    import torch
    import AnomalyCLIP_lib
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from extent_prompt import ExtentConditioner, conditioned_text_features
    params = dict(Prompt_length=12, learnabel_text_embedding_depth=9,
                  learnabel_text_embedding_length=4)
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device="cuda",
                                   design_details=params)
    learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
    try:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(args.checkpoint, map_location="cpu")
    if state.get("extent_cond") != "extent":
        raise ValueError("checkpoint is not extent-conditioned")
    learner.load_state_dict(state["prompt_learner"])
    model.to("cuda").eval()
    learner.to("cuda").eval()
    model.visual.DAPM_replace(DPAM_layer=20)
    conditioner = ExtentConditioner(mode="extent").to("cuda").eval()
    conditioner.load_state_dict(state["conditioner"])
    with torch.no_grad():
        _, c_pos, c_neg = conditioner(torch.zeros(1, 784, device="cuda"),
                                      z_override=torch.full((1,), 1.6, device="cuda"))
        text = conditioned_text_features(model, learner, c_pos, c_neg)[0]
    return model, text


def score(model, text, image, full_map=False):
    import torch
    import torch.nn.functional as F
    import AnomalyCLIP_lib
    with torch.no_grad():
        _, features = model.encode_image(image, [24], DPAM_layer=20)
        native = F.normalize(features[-1].float(), dim=-1)
        margins = ((native[:, 1:] @ (text[1] - text[0])) / 0.07).reshape(GRID, GRID)
        result = dict(margins=margins.detach().cpu().numpy(),
                      tokens=native[0, 1:].detach().cpu())
        if full_map:
            from scipy.ndimage import gaussian_filter
            sim, _ = AnomalyCLIP_lib.compute_similarity(native, text)
            map2 = AnomalyCLIP_lib.get_similarity_map(sim[:, 1:], 518)
            raw = (map2[..., 1] + 1 - map2[..., 0]) / 2
            result["map"] = gaussian_filter(raw[0].detach().cpu().numpy(), sigma=4)
    return result


def dataset_and_indices(args, name, cap):
    from dataset import Dataset
    from utils import get_transform
    folder, mode = DATASETS[name]
    transform, target_transform = get_transform(argparse.Namespace(image_size=518))
    ds = Dataset(str(args.data_root / folder), transform, target_transform,
                 dataset_name=mode, mode="test")
    indexed = sorted(range(len(ds)), key=lambda i: stable_key(ds.data_all[i]["img_path"]))
    return ds, indexed[:cap]


def reference_scores(args, name):
    path = args.reference_root / name / "ecp_extent/pixel_per_image_predictions.csv"
    rows = csv_rows(path)
    result = {r["sample_id"]: float(r["per_image_pixel_auroc"]) for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"{name}: duplicate reference IDs")
    return result


def identify(args, ds, index, name, model, text, reference):
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    from sklearn.metrics import roc_auc_score
    item = ds[index]
    sample_id = str(Path(item["img_path"]).relative_to(args.data_root / DATASETS[name][0]))
    if sample_id not in reference:
        raise ValueError(f"{name}: missing reference ID {sample_id}")
    image = item["img"].unsqueeze(0).to("cuda")
    baseline = score(model, text, image, full_map=True)
    centers = select_centers(baseline["margins"])
    # The mask is not read until the model-only centers have been fixed.
    mask = item["img_mask"][0].numpy() > 0.5
    if not (mask.any() and (~mask).any()):
        raise ValueError(f"{name}/{sample_id}: mask is not binary")
    auc = float(roc_auc_score(mask.ravel(), baseline["map"].ravel()))
    if abs(auc - reference[sample_id]) > 1e-6:
        raise ValueError(f"{name}/{sample_id}: baseline does not reproduce EXP-012: {auc} vs {reference[sample_id]}")
    far = distance_transform_edt(~mask)
    selected = []
    for r, c, group in centers:
        label, fraction = center_label(mask, r, c, far)
        if label == "far_FP" and group != "high":
            label = "far_healthy_control"
        selected.append(dict(row=r, col=c, rank_group=group,
                             label=label, lesion_fraction=fraction,
                             baseline_margin=float(baseline["margins"][r, c])))
    return dict(index=index, sample_id=sample_id, centers=selected,
                baseline_per_image_auroc=auc)


def dataset_plan(args, name, ds, indices, model, text, reference, smoke):
    inspected = [identify(args, ds, i, name, model, text, reference) for i in indices]
    if smoke:
        chosen = inspected
    else:
        first = inspected[:48]
        eligible = lambda rec: (len(rec["centers"]) == 6 and
              {"TP", "far_FP"} <= {c["label"] for c in rec["centers"]})
        chosen = first if sum(eligible(r) for r in first) >= 30 else inspected[:96]
    return chosen


def process_dataset(args, name, ds, plan, model, text, output, smoke):
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    from metrics import pixel_level_metrics
    from scipy.ndimage import distance_transform_edt
    output.mkdir(parents=True, exist_ok=False)
    write(output / "selection.json", plan)
    rows, maps = [], {"baseline": [], "photo": [], "rearrange": []}
    map_masks, map_ids = [], []
    refs = reference_scores(args, name)
    for number, record in enumerate(plan, 1):
        item = ds[record["index"]]
        image = item["img"].unsqueeze(0).to("cuda")
        mask = item["img_mask"][0].numpy() > 0.5
        baseline = score(model, text, image, full_map=True)
        auc = float(roc_auc_score(mask.ravel(), baseline["map"].ravel()))
        if abs(auc - refs[record["sample_id"]]) > 1e-6:
            raise ValueError(f"{name}/{record['sample_id']}: repeated baseline parity failed")
        if not record["centers"]:
            continue
        map_masks.append(mask.astype(np.uint8))
        map_ids.append(record["sample_id"])
        maps["baseline"].append(baseline["map"])
        for ci, center in enumerate(record["centers"][:2] if smoke else record["centers"]):
            r, c = center["row"], center["col"]
            index = r * GRID + c
            original_margin = float(baseline["margins"][r, c])
            original_token = baseline["tokens"][index]
            sham = score(model, text, intervention(image, r, c, "sham",
                         record["sample_id"]), full_map=False)
            sham_drift = abs(float(sham["margins"][r, c]) - original_margin)
            if sham_drift > 1e-5:
                raise ValueError(f"{name}/{record['sample_id']}: sham margin drift {sham_drift}")
            kinds = ("photo", "rearrange", "near_photo") if ci == 0 else ("photo", "rearrange")
            for kind in kinds:
                edited = intervention(image, r, c, kind, record["sample_id"])
                changed_pixels = (edited - image).abs().amax(dim=1)[0] > 0
                coords = torch.nonzero(changed_pixels).float()
                if len(coords) == 0:
                    raise ValueError(f"{name}/{record['sample_id']}: {kind} did not change pixels")
                query = coords.new_tensor([r * PATCH + 6.5, c * PATCH + 6.5])
                seam_distance = float(torch.linalg.vector_norm(coords - query, dim=1).min())
                if seam_distance <= 42:
                    raise ValueError(f"{name}/{record['sample_id']}: edit reached queried patch buffer")
                changed = score(model, text, edited,
                                full_map=(ci == 0 and kind in ("photo", "rearrange")))
                drift = float(changed["margins"][r, c]) - original_margin
                displacement = float(torch.linalg.vector_norm(changed["tokens"][index] - original_token))
                if ci == 0 and kind in ("photo", "rearrange"):
                    maps[kind].append(changed["map"])
                rows.append(dict(dataset=name, sample_id=record["sample_id"],
                    center_row=r, center_col=c, center_index=ci, rank_group=center["rank_group"],
                    label=center["label"], lesion_fraction=center["lesion_fraction"],
                    mask_area_fraction=float(mask.mean()), baseline_margin=original_margin,
                    baseline_per_image_auroc=auc, family=kind, signed_margin_drift=drift,
                    sham_abs_margin_drift=sham_drift,
                    sensitivity=abs(drift)-sham_drift,
                    projected_token_l2=displacement,
                    input_mean_abs_change=float((edited-image).abs().mean()),
                    minimum_query_to_edit_distance_px=seam_distance))
        if number % 10 == 0 or number == len(plan):
            print(f"EXP-015 {name}: {number}/{len(plan)} images", flush=True)
    if not rows:
        raise ValueError(f"{name}: no centers were selected")
    save_rows(output / "center_effects.csv", rows, list(rows[0]))
    if not smoke:
        per_image_maps = []
        for family, predictions in maps.items():
            for sample_id, mask, prediction in zip(map_ids, map_masks, predictions):
                per_image_maps.append(dict(dataset=name, sample_id=sample_id,
                    family=family, per_image_pixel_auroc=float(roc_auc_score(
                        mask.ravel(), prediction.ravel()))))
        save_rows(output / "descriptive_map_per_image.csv", per_image_maps,
                  ["dataset", "sample_id", "family", "per_image_pixel_auroc"])
        summary = {}
        for family, predictions in maps.items():
            if len(predictions) != len(map_masks):
                raise ValueError(f"{name}: incomplete descriptive map family {family}")
            metric_input = {"class": {"imgs_masks": np.stack(map_masks),
                                      "anomaly_maps": np.stack(predictions)}}
            try:
                auroc = float(pixel_level_metrics(metric_input, "class", "pixel-auroc"))
                aupro = float(pixel_level_metrics(metric_input, "class", "pixel-aupro"))
                if not (math.isfinite(auroc) and math.isfinite(aupro)):
                    raise ValueError("nonfinite descriptive pooled metric")
                summary[family] = dict(pixel_auroc=auroc, pixel_aupro=aupro)
            except (ValueError, ZeroDivisionError, FloatingPointError) as exc:
                summary[family] = dict(metric_error=str(exc))
        write(output / "descriptive_map_metrics.json",
              dict(sample_count=len(map_masks), only_first_center_per_image=True,
                   metrics=summary))
    return rows


def image_bootstrap(values, repetitions=2000):
    import numpy as np
    rng = np.random.default_rng(111)
    data = np.asarray(values, dtype=float)
    means = data[rng.integers(0, len(data), size=(repetitions, len(data)))].mean(1)
    return float(data.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def analyze_dataset(rows, selected):
    import numpy as np
    eligible = {r["sample_id"] for r in selected if len(r["centers"]) == 6 and
                {"TP", "far_FP"} <= {c["label"] for c in r["centers"]}}
    result = dict(selected_images=len(selected), eligible_images=len(eligible),
                  selected_centers=sum(len(r["centers"]) for r in selected))
    near = [row for row in rows if row["family"] == "near_photo"]
    result["near_ring_control"] = dict(n=len(near),
        median_abs_margin_drift=float(np.median([abs(float(x["signed_margin_drift"]))
                                          for x in near])) if near else None,
        median_input_mean_abs_change=float(np.median([float(x["input_mean_abs_change"])
                                           for x in near])) if near else None)
    for family in ("photo", "rearrange"):
        image_aucs, adjusted_aucs, sham_drifts, active_drifts = [], [], [], []
        grouped = defaultdict(list)
        for row in rows:
            if row["sample_id"] in eligible and row["family"] == family:
                grouped[row["sample_id"]].append(row)
        for image_rows in grouped.values():
            fp = [float(x["sensitivity"]) for x in image_rows if x["label"] == "far_FP"]
            tp = [float(x["sensitivity"]) for x in image_rows if x["label"] == "TP"]
            score = paired_auc(fp, tp)
            if score is None:
                continue
            image_aucs.append(score)
            x = np.asarray([[float(row["baseline_margin"]),
                             float(row["minimum_query_to_edit_distance_px"])]
                            for row in image_rows])
            y = np.asarray([float(row["sensitivity"]) for row in image_rows])
            centered = x - x.mean(axis=0)
            coefficients = np.linalg.lstsq(centered, y - y.mean(), rcond=None)[0]
            residual = y - centered @ coefficients
            adjusted_aucs.append(paired_auc(
                [float(residual[i]) for i, row in enumerate(image_rows) if row["label"] == "far_FP"],
                [float(residual[i]) for i, row in enumerate(image_rows) if row["label"] == "TP"]))
            sham_drifts.extend(float(row["sham_abs_margin_drift"]) for row in image_rows)
            active_drifts.extend(abs(float(row["signed_margin_drift"])) for row in image_rows)
        if image_aucs:
            mean, lower, upper = image_bootstrap(image_aucs)
            adjusted_mean, adjusted_lower, adjusted_upper = image_bootstrap(adjusted_aucs)
            sham_median = float(np.median(sham_drifts))
            active_median = float(np.median(active_drifts))
            result[family] = dict(n_images=len(image_aucs), within_image_auc_mean=mean,
                ci95=[lower, upper], margin_and_seam_adjusted_auc_mean=adjusted_mean,
                margin_and_seam_adjusted_ci95=[adjusted_lower, adjusted_upper],
                median_sham_abs_margin_drift=sham_median,
                median_active_abs_margin_drift=active_median,
                sham_to_active_ratio=sham_median / active_median if active_median > 1e-8 else None)
        else:
            result[family] = dict(n_images=0)
    return result


def run_probe(args, prov, smoke):
    import torch
    from scipy.ndimage import distance_transform_edt  # validates dependency
    from sklearn.metrics import roc_auc_score  # validates dependency
    seed_all()
    out = args.out_root / ("smoke" if smoke else "pilot")
    if out.exists():
        raise FileExistsError(f"preserve old output and set a fresh EXP015_ROOT: {out}")
    out.mkdir(parents=True)
    write(out / "run_metadata.json", dict(provenance=prov,
          command=shlex.join([sys.executable, *sys.argv]),
          checkpoint_path=str(args.checkpoint), data_root=str(args.data_root)))
    model, text = load_model(args)
    names = ("ClinicDB",) if smoke else tuple(DATASETS)
    summaries = {}
    for name in names:
        ds, indices = dataset_and_indices(args, name, 2 if smoke else 96)
        reference = reference_scores(args, name)
        selected = dataset_plan(args, name, ds, indices, model, text, reference, smoke)
        dataset_out = out / name
        rows = process_dataset(args, name, ds, selected, model, text, dataset_out, smoke)
        summaries[name] = analyze_dataset(rows, selected)
        write(dataset_out / "analysis.json", summaries[name])
    write(out / "summary.json", summaries)
    if smoke:
        if not summaries["ClinicDB"]["selected_centers"]:
            raise ValueError("smoke selected no centers")
        write(out / "passed.json", prov)
        print("EXP-015 SMOKE PASSED", flush=True)
    else:
        adequate = all(x["eligible_images"] >= 30 for x in summaries.values())
        qualified = []
        for name, values in summaries.items():
            if values["eligible_images"] < 30:
                continue
            both = True
            for family in ("photo", "rearrange"):
                item = values[family]
                both &= (item["n_images"] >= 30 and
                         item["within_image_auc_mean"] >= .60 and
                         item["ci95"][0] > .50 and
                         item["margin_and_seam_adjusted_auc_mean"] >= .55 and
                         item["margin_and_seam_adjusted_ci95"][0] > .50 and
                         item["sham_to_active_ratio"] is not None and
                         item["sham_to_active_ratio"] < .20)
            if both:
                qualified.append(name)
        verdict = ("INCONCLUSIVE_COVERAGE" if not adequate else
                   "PASS_DIAGNOSTIC_GATE" if len(qualified) >= 2 else
                   "FAIL_DIAGNOSTIC_GATE")
        write(out / "verdict.json", dict(verdict=verdict, qualified_datasets=qualified,
                                         summaries=summaries,
                                         interpretation="Diagnostic association only; no causal method claim"))
        print(f"EXP-015 PILOT COMPLETED: {out}; verdict={verdict}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "smoke", "pilot"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--clip-weights", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    args = parser.parse_args()
    prov = provenance(args)
    if args.stage == "calibrate":
        source_calibration(args, prov)
        return
    cal = read(args.out_root / "calibration.json")
    if cal["provenance"] != prov:
        raise ValueError("source calibration and current inputs/code differ")
    if args.stage == "pilot" and read(args.out_root / "smoke/passed.json") != prov:
        raise ValueError("run passing two-image smoke before pilot with identical inputs/code")
    run_probe(args, prov, smoke=args.stage == "smoke")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"EXP-015 error: {exc}") from exc
