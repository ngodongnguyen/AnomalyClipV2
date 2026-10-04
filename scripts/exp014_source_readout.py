#!/usr/bin/env python3
"""EXP-014 source-matched Q-K/V-V readout; frozen AnomalyCLIP, MVTec-only fit.

Run through run_exp014_source_readout.sh. This does not train model parameters.
The source-fitted scalar rules are diagnostics, not a proposed inference module.
"""

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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp014_readout_core import auc, gate, hotspot, source_partition

ECP_SHA = "7278af957c4087d7c00e7495e4cba3a65c253b5d0c9646b36434b951e3df0e7c"
DATASETS = {
    "ISIC": ("ISIC", "ISBI"),
    "ClinicDB": ("CVC/CVC-ClinicDB", "colon"),
    "ColonDB": ("CVC/CVC-ColonDB", "colon"),
    "Kvasir": ("CVC/Kvasir", "colon"),
    "Endo": ("EndoTect_2020_Segmentation_Test_Dataset", "colon"),
    "TN3K": ("TN3K/Thyroid Dataset/tn3k", "thyroid"),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def dataset(args, folder, mode):
    from dataset import Dataset
    from utils import get_transform
    transform, mask_transform = get_transform(argparse.Namespace(image_size=518))
    return Dataset(str(args.data_root / folder), transform, mask_transform,
                   dataset_name=mode, mode="test")


class FrozenBranches:
    def __init__(self, args):
        import torch
        import AnomalyCLIP_lib
        from extent_prompt import ExtentConditioner
        from prompt_ensemble import AnomalyCLIP_PromptLearner
        self.torch = torch
        self.lib = AnomalyCLIP_lib
        params = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9,
                  "learnabel_text_embedding_length": 4}
        model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device="cuda", design_details=params)
        model.eval()
        learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if checkpoint.get("extent_cond") != "extent" or "conditioner" not in checkpoint:
            raise ValueError("checkpoint does not contain an extent conditioner")
        learner.load_state_dict(checkpoint["prompt_learner"])
        self.learner = learner.to("cuda").eval()
        self.model = model.to("cuda")
        self.model.visual.DAPM_replace(DPAM_layer=20)
        self.conditioner = ExtentConditioner(mode="extent").to("cuda")
        self.conditioner.load_state_dict(checkpoint["conditioner"])
        self.conditioner.eval()
        self.captured = {}

        def capture(_module, _inputs, output):
            if not isinstance(output, list) or len(output) != 2:
                raise RuntimeError("final visual block did not yield two streams")
            self.captured["vv"], self.captured["qk"] = output
        self.hook = self.model.visual.transformer.resblocks[-1].register_forward_hook(capture)

    def __call__(self, image, baseline=False):
        import torch.nn.functional as F
        from extent_prompt import conditioned_text_features, visual_descriptor
        torch = self.torch
        self.captured.clear()
        with torch.no_grad():
            image_features, patches = self.model.encode_image(
                image.unsqueeze(0).to("cuda"), [24], DPAM_layer=20)
            if not self.captured:
                raise RuntimeError("branch hook did not fire")
            vv = self.model.visual.ln_post(self.captured["vv"].permute(1, 0, 2)) @ self.model.visual.proj
            qk = self.model.visual.ln_post(self.captured["qk"].permute(1, 0, 2)) @ self.model.visual.proj
            error = float((vv - patches[-1]).abs().max())
            if error > 1e-5:
                raise RuntimeError(f"V-V hook disagrees with existing path: {error}")
            result = {"vv": F.normalize(vv[0, 1:].float(), dim=-1).cpu().numpy(),
                      "qk": F.normalize(qk[0, 1:].float(), dim=-1).cpu().numpy(),
                      "hook_error": error}
            if baseline:
                desc = visual_descriptor(F.normalize(image_features, dim=-1), patches)
                _, c_pos, c_neg = self.conditioner(
                    desc, z_override=torch.full((1,), 1.6, device="cuda"))
                text = conditioned_text_features(self.model, self.learner, c_pos, c_neg)
                similarity, _ = self.lib.compute_similarity(F.normalize(patches[-1], dim=-1), text[0])
                smap = self.lib.get_similarity_map(similarity[:, 1:, :], 518)
                raw = (smap[..., 1] + 1 - smap[..., 0]) / 2
                from scipy.ndimage import gaussian_filter
                result["baseline"] = gaussian_filter(raw[0].cpu().numpy(), sigma=4)
        return result

    def close(self):
        self.hook.remove()


def source_mask(item):
    import torch.nn.functional as F
    fractions = F.adaptive_avg_pool2d(item["img_mask"].unsqueeze(0).float(),
                                      (37, 37))[0, 0].numpy().ravel()
    # Exclude ambiguous boundary patches from source fitting.
    return fractions >= 0.5, fractions <= 0.1


def native_scores(features, directions):
    return {name: features[name] @ directions[name] for name in ("vv", "qk")}


def maps(scores):
    import torch
    import torch.nn.functional as F
    from scipy.ndimage import gaussian_filter
    result = {}
    for name, values in scores.items():
        tensor = torch.from_numpy(values.reshape(1, 1, 37, 37)).float()
        full = F.interpolate(tensor, size=(518, 518), mode="bilinear")
        result[name] = gaussian_filter(full[0, 0].numpy(), sigma=4)
    return result


def labels_and_scores(item, features, directions):
    import numpy as np
    positive, negative = source_mask(item)
    sample_id = item["img_path"]
    selected = np.concatenate((sample_indices(positive, sample_id + ":positive"),
                               sample_indices(negative, sample_id + ":negative")))
    if len(selected) == 0:
        return np.empty((0, 3)), np.empty((0,), dtype=int)
    raw = native_scores(features, directions)
    # Same local context available to the two-coefficient V-V-only control.
    from scipy.ndimage import gaussian_filter
    vv_local = gaussian_filter(raw["vv"].reshape(37, 37), sigma=1).ravel()
    x = np.stack((raw["vv"][selected], raw["qk"][selected], vv_local[selected]), axis=1)
    return x, positive[selected].astype(int)


def sample_indices(allowed, sample_id):
    """Cap one eligible class per image, excluding boundary patches."""
    import numpy as np
    indices = np.flatnonzero(allowed)
    if len(indices) <= 32:
        return indices
    seed = int(hashlib.sha256(("EXP-014:patch:" + sample_id).encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).choice(indices, 32, replace=False)


def provenance(args):
    import torch
    if not torch.cuda.is_available():
        raise ValueError("EXP-014 source readout requires CUDA")
    for path in (args.checkpoint, args.clip_weights, args.manifest,
                 args.data_root / "mvtec/meta.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.clip_weights.resolve() != (Path.home() / ".cache/clip/ViT-L-14-336px.pt").resolve():
        raise ValueError("CLIP_WEIGHTS must name the cache file read by AnomalyCLIP_lib.load")
    manifest = json.loads(args.manifest.read_text())
    if sha(args.checkpoint) != ECP_SHA or manifest["checkpoint_sha256"] != ECP_SHA:
        raise ValueError("checkpoint differs from matched EXP-012 ECP checkpoint")
    if manifest["extent_cond"] != "extent" or manifest["seed"] != 111:
        raise ValueError("wrong ECP manifest")
    if sha(args.clip_weights) != manifest["clip_weights_sha256"]:
        raise ValueError("CLIP weights differ from ECP manifest")
    if sha(args.data_root / "mvtec/meta.json") != manifest["mvtec_meta_sha256"]:
        raise ValueError("MVTec split differs from ECP training")
    return {"checkpoint_sha256": ECP_SHA, "clip_sha256": sha(args.clip_weights),
            "source_meta_sha256": sha(args.data_root / "mvtec/meta.json"),
            "probe_sha256": sha(__file__),
            "core_sha256": sha(ROOT / "scripts/exp014_readout_core.py"),
            "command": shlex.join([sys.executable, *sys.argv]),
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                  text=True).strip()}


def fit(args, model, prov):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    source = dataset(args, "mvtec", "mvtec")
    sums = {name: {0: np.zeros(768, dtype=np.float64), 1: np.zeros(768, dtype=np.float64)}
            for name in ("vv", "qk")}
    counts = {0: 0, 1: 0}
    ids = {key: [] for key in ("direction", "fit", "validation")}
    for i in range(len(source)):
        item = source[i]
        part = source_partition(item["img_path"])
        ids[part].append(item["img_path"])
        if part != "direction":
            continue
        features = model(item["img"])
        positive, negative = source_mask(item)
        for label, allowed in ((1, positive), (0, negative)):
            indices = sample_indices(allowed, item["img_path"] + str(label))
            if len(indices):
                counts[label] += len(indices)
                for name in sums:
                    sums[name][label] += features[name][indices].sum(axis=0)
        if (i + 1) % 100 == 0:
            print(f"EXP-014 direction scan {i+1}/{len(source)}", flush=True)
    if min(counts.values()) == 0:
        raise ValueError("source partition lacks positive or negative patches")
    directions = {}
    for name in sums:
        difference = sums[name][1] / counts[1] - sums[name][0] / counts[0]
        norm = np.linalg.norm(difference)
        if norm < 1e-10:
            raise ValueError(f"{name}: no source separation")
        directions[name] = difference / norm
    matrices = {"fit": [], "validation": []}
    labels = {"fit": [], "validation": []}
    for i in range(len(source)):
        item = source[i]
        part = source_partition(item["img_path"])
        if part == "direction":
            continue
        x, y = labels_and_scores(item, model(item["img"]), directions)
        if len(y):
            matrices[part].append(x)
            labels[part].append(y)
        if (i + 1) % 100 == 0:
            print(f"EXP-014 scalar-fit scan {i+1}/{len(source)}", flush=True)
    x_fit = np.concatenate(matrices["fit"])
    y_fit = np.concatenate(labels["fit"])
    x_val = np.concatenate(matrices["validation"])
    y_val = np.concatenate(labels["validation"])
    if len(np.unique(y_fit)) != 2 or len(np.unique(y_val)) != 2:
        raise ValueError("source scalar partitions lack both labels")
    # Standardize with fitting partition only. Both rules have two coefficients.
    mean = x_fit.mean(axis=0)
    std = x_fit.std(axis=0)
    if np.any(std < 1e-10):
        raise ValueError("constant source feature")
    z_fit = (x_fit - mean) / std
    z_val = (x_val - mean) / std
    rules = {}
    for name, columns in (("two_branch", (0, 1)), ("vv_only", (0, 2))):
        clf = LogisticRegression(C=1, class_weight="balanced", max_iter=500,
                                 random_state=111, solver="lbfgs")
        clf.fit(z_fit[:, columns], y_fit)
        rules[name] = {"columns": columns, "coef": clf.coef_[0].tolist(),
                       "intercept": float(clf.intercept_[0])}
    # A source-selected fixed convex average of individually standardized branches.
    weights = (0, 0.25, 0.5, 0.75, 1.0)
    scores = [auc(y_val, (1-w)*z_val[:, 0] + w*z_val[:, 1]) for w in weights]
    best = int(np.argmax(scores))
    artifact = {"provenance": prov, "source_partition":
                "SHA256(EXP-014:111:sample_id) modulo100: 0-69 direction, 70-84 fit, 85-99 validation",
                "source_counts": counts, "source_image_counts": {k: len(v) for k, v in ids.items()},
                "source_ids_sha256": {k: hashlib.sha256("\n".join(v).encode()).hexdigest() for k, v in ids.items()},
                "direction": {k: v.tolist() for k, v in directions.items()},
                "standard_mean": mean.tolist(), "standard_std": std.tolist(),
                "rules": rules, "fixed_qk_weight": weights[best],
                "validation_auc_by_qk_weight": dict(zip(map(str, weights), scores)),
                "source_fit_patch_count": len(y_fit), "source_validation_patch_count": len(y_val)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    write_json(args.out, artifact)
    print(f"EXP-014 source readout written: {args.out}")


def load_reference(args, name):
    folder, mode = DATASETS[name]
    base = args.reference_root / name / "ecp_extent"
    meta = json.loads((base / "evaluation_metadata.json").read_text())
    expected = {"dataset": mode, "metrics": "pixel-level", "features_list": [24],
                "image_size": 518, "depth": 9, "n_ctx": 12, "t_n_ctx": 4,
                "sigma": 4, "seed": 111, "ec_z_img": -0.69, "ec_z_pix": 1.6,
                "ec_const_z": None, "ec_oracle": False,
                "distractor_suppress": "none", "distractor_rerank": "none"}
    if any(meta["arguments"].get(k) != v for k, v in expected.items()):
        raise ValueError(f"{name}: EXP-012 settings mismatch")
    if meta["checkpoint_sha256"] != ECP_SHA or meta["pretrained_clip_weights_sha256"] != sha(args.clip_weights):
        raise ValueError(f"{name}: checkpoint/CLIP mismatch")
    if meta["meta_json_sha256"] != sha(args.data_root / folder / "meta.json"):
        raise ValueError(f"{name}: split mismatch")
    for relative, digest in meta["evaluation_source_sha256"].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f"{name}: evaluation dependency changed: {relative}")
    with (base / "pixel_per_image_predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    reference = {r["sample_id"]: float(r["per_image_pixel_auroc"]) for r in rows}
    if len(reference) != len(rows):
        raise ValueError(f"{name}: duplicate reference IDs")
    return reference


def evaluate(args, model, prov):
    import numpy as np
    from scipy.ndimage import gaussian_filter
    fitted = json.loads(args.fitted.read_text())
    if fitted["provenance"]["checkpoint_sha256"] != prov["checkpoint_sha256"] or \
            fitted["provenance"]["clip_sha256"] != prov["clip_sha256"] or \
            fitted["provenance"]["source_meta_sha256"] != prov["source_meta_sha256"] or \
            fitted["provenance"]["probe_sha256"] != prov["probe_sha256"] or \
            fitted["provenance"]["core_sha256"] != prov["core_sha256"]:
        raise ValueError("source readout was fitted with different inputs/code")
    directions = {k: np.array(v) for k, v in fitted["direction"].items()}
    mean = np.array(fitted["standard_mean"])
    std = np.array(fitted["standard_std"])
    names = ("ClinicDB",) if args.stage == "smoke" else tuple(DATASETS)
    all_rows = {}
    out = args.out
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    for name in names:
        folder, mode = DATASETS[name]
        ref = load_reference(args, name)
        ds = dataset(args, folder, mode)
        rows = []
        count = min(len(ds), 2) if args.stage == "smoke" else len(ds)
        for i in range(count):
            item = ds[i]
            sample_id = str(Path(item["img_path"]).relative_to(args.data_root / folder))
            if sample_id not in ref:
                raise ValueError(f"{name}: missing reference ID {sample_id}")
            mask = item["img_mask"][0].numpy() > 0.5
            features = model(item["img"], baseline=True)
            baseline = features["baseline"]
            parity = auc(mask.ravel(), baseline.ravel())
            if parity is None or abs(parity - ref[sample_id]) > 1e-6:
                raise ValueError(f"{name}/{sample_id}: V-V parity failure {parity} vs {ref[sample_id]}")
            native = native_scores(features, directions)
            vv_local = gaussian_filter(native["vv"].reshape(37, 37), sigma=1).ravel()
            x = np.stack((native["vv"], native["qk"], vv_local), axis=1)
            z = (x - mean) / std
            scores = {}
            for rule_name, rule in fitted["rules"].items():
                columns = rule["columns"]
                scores[rule_name] = z[:, columns] @ np.array(rule["coef"]) + rule["intercept"]
            w = fitted["fixed_qk_weight"]
            scores["fixed_mixture"] = (1-w)*z[:, 0] + w*z[:, 1]
            score_maps = maps(scores)
            score_maps["baseline"] = baseline
            score_maps["smoothing_only"] = gaussian_filter(baseline, sigma=4)
            tp, fp, hot = hotspot(mask, baseline, score_maps)
            row = {"dataset": name, "sample_id": sample_id,
                   "mask_area_fraction": float(mask.mean()), "hotspot_tp": tp,
                   "hotspot_far_fp": fp, "baseline_pixel_auroc": parity,
                   "max_vv_hook_error": features["hook_error"]}
            for method, score_map in score_maps.items():
                row[f"pixel_auroc_{method}"] = auc(mask.ravel(), score_map.ravel())
                row[f"hotspot_{method}"] = hot[method]
            rows.append(row)
            if (i + 1) % 50 == 0 or i + 1 == count:
                print(f"EXP-014 {name}: {i+1}/{count}", flush=True)
        if args.stage == "full" and {r["sample_id"] for r in rows} != set(ref):
            raise ValueError(f"{name}: incomplete split")
        with (out / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        all_rows[name] = rows
    summary = {"stage": args.stage, "counts": {k: len(v) for k, v in all_rows.items()},
               "eligible_counts": {k: sum(r["hotspot_vv_only"] is not None for r in v)
                                   for k, v in all_rows.items()},
               "max_vv_reference_abs_error": max(abs(r["baseline_pixel_auroc"] -
                    load_reference(args, k)[r["sample_id"]])
                    for k, v in all_rows.items() for r in v)}
    if args.stage == "full":
        summary["prospective_gate"] = gate(all_rows)
    write_json(out / "summary.json", summary)
    write_json(out / "metadata.json", {"provenance": prov, "fitted_sha256": sha(args.fitted),
                                       "datasets": names, "sampling": "top 5% baseline V-V pixels; healthy >28px from mask"})
    print(f"EXP-014 {args.stage.upper()} COMPLETED: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "smoke", "full"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip-weights", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--fitted", type=Path, required=True)
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
    prov = provenance(args)
    model = FrozenBranches(args)
    try:
        fit(args, model, prov) if args.stage == "fit" else evaluate(args, model, prov)
    finally:
        model.close()


if __name__ == "__main__":
    main()
