#!/usr/bin/env python3
"""EXP-016: frozen ECP + small visual residual, matched exploratory ablations.

Use run_exp016.sh on the GPU server. Existing test.py owns evaluation and metrics.
No target masks/images are used to fit the residual or choose its final epoch.
"""
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import random
import shlex
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARMS = ("plain", "aug_only", "context")
DATASETS = {
    "ISIC": ("ISIC", "ISBI", "skin"),
    "ClinicDB": ("CVC/CVC-ClinicDB", "colon", "colon"),
    "ColonDB": ("CVC/CVC-ColonDB", "colon", "colon"),
    "Kvasir": ("CVC/Kvasir", "colon", "colon"),
    "Endo": ("EndoTect_2020_Segmentation_Test_Dataset", "colon", "colon"),
    "TN3K": ("TN3K/Thyroid Dataset/tn3k", "thyroid", "thyroid"),
}
PROTOCOL = dict(experiment="EXP-016", seed=111, epochs=3, batch_size=2,
                lr=1e-4, rank=32, consistency_weight=1.0, residual_weight=0.01,
                core_focal_weight=1.0, image_size=518, sigma=4, ec_z_pix=1.6,
                ec_z_img=-0.69, depth=9, n_ctx=12, t_n_ctx=4,
                features_list=[24], source_split="test", zoom_aug_p=0.0)
EXPECTED_ECP = "7278af957c4087d7c00e7495e4cba3a65c253b5d0c9646b36434b951e3df0e7c"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def load_state(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def seed_all():
    import numpy as np
    import torch
    random.seed(111)
    np.random.seed(111)
    torch.manual_seed(111)
    torch.cuda.manual_seed_all(111)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preflight(args):
    """Pin all original dependencies to the successfully evaluated EXP-012 run."""
    import torch
    import numpy as np
    import scipy
    import sklearn
    import test as evaluation
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use the project GPU environment")
    if not hasattr(evaluation, "_write_pixel_level_artifacts"):
        raise ValueError("test.py lacks the EXP-012 pixel export instrumentation")
    paths = [args.checkpoint, args.clip_weights, args.manifest,
             args.data_root / "mvtec/meta.json", ROOT / "context_visual_residual.py"]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
    if args.clip_weights.resolve() != (Path.home() / ".cache/clip/ViT-L-14-336px.pt").resolve():
        raise ValueError("CLIP_WEIGHTS must identify the existing home CLIP cache used by test.py")
    ecp_sha, clip_sha = sha(args.checkpoint), sha(args.clip_weights)
    manifest = read(args.manifest)
    if ecp_sha != EXPECTED_ECP or ecp_sha != manifest["checkpoint_sha256"]:
        raise ValueError("expected the verified EXP-012 matched ECP checkpoint")
    if clip_sha != manifest["clip_weights_sha256"]:
        raise ValueError("CLIP hash differs from the ECP training manifest")
    if sha(args.data_root / "mvtec/meta.json") != manifest["mvtec_meta_sha256"]:
        raise ValueError("MVTec auxiliary split changed since matched ECP training")
    state = load_state(args.checkpoint)
    if state.get("extent_cond") != "extent" or "conditioner" not in state:
        raise ValueError("ECP checkpoint must contain extent conditioning")
    del state
    source_hashes = {}
    references = {}
    for name, (folder, mode, _) in DATASETS.items():
        ref = args.reference_root / name / "ecp_extent"
        meta = read(ref / "evaluation_metadata.json")
        if meta["checkpoint_sha256"] != ecp_sha or meta["pretrained_clip_weights_sha256"] != clip_sha:
            raise ValueError(f"{name}: mismatched reference checkpoint/CLIP")
        if sha(args.data_root / folder / "meta.json") != meta["meta_json_sha256"]:
            raise ValueError(f"{name}: reference dataset split differs")
        expected = {k: PROTOCOL[k] for k in ("sigma", "image_size", "depth", "n_ctx",
                    "t_n_ctx", "seed", "features_list", "ec_z_img", "ec_z_pix")}
        expected.update(dataset=mode, metrics="pixel-level", ec_const_z=None,
                        ec_oracle=False, distractor_suppress="none", distractor_rerank="none")
        if any(meta["arguments"].get(k) != v for k, v in expected.items()):
            raise ValueError(f"{name}: reference evaluation settings differ")
        for rel, old_hash in meta["evaluation_source_sha256"].items():
            current = sha(ROOT / rel)
            if current != old_hash:
                raise ValueError(f"{name}: baseline evaluation dependency changed: {rel}")
            source_hashes[rel] = current
        references[name] = {f: sha(ref / f) for f in
                            ("evaluation_metadata.json", "pixel_level_metrics.json", "pixel_per_image_predictions.csv")}
    for folder in ["mvtec", *[v[0] for v in DATASETS.values()]]:
        base = args.data_root / folder
        entries = read(base / "meta.json")["test"]
        for samples in entries.values():
            for sample in samples:
                if not (base / sample["img_path"]).is_file():
                    raise FileNotFoundError(base / sample["img_path"])
                if sample["anomaly"] and not (base / sample["mask_path"]).is_file():
                    raise FileNotFoundError(base / sample["mask_path"])
    for rel in ("loss.py", "context_visual_residual.py", "scripts/exp016_context_residual.py",
                "scripts/run_exp016.sh", "tests/test_context_visual_residual.py"):
        source_hashes[rel] = sha(ROOT / rel)
    provenance = dict(protocol=PROTOCOL, base_checkpoint_sha256=ecp_sha, clip_sha256=clip_sha,
                      source_sha256=source_hashes, references=references,
                      mvtec_meta_sha256=sha(args.data_root / "mvtec/meta.json"),
                      environment=dict(python=sys.version.split()[0], torch=torch.__version__,
                                       cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                                       numpy=np.__version__, scipy=scipy.__version__,
                                       sklearn=sklearn.__version__))
    return provenance


def frozen_ecp(args):
    import torch
    import AnomalyCLIP_lib
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from extent_prompt import ExtentConditioner, conditioned_text_features
    params = dict(Prompt_length=12, learnabel_text_embedding_depth=9,
                  learnabel_text_embedding_length=4)
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device="cuda", design_details=params)
    learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
    state = load_state(args.checkpoint)
    learner.load_state_dict(state["prompt_learner"])
    model.to("cuda").eval()
    learner.to("cuda").eval()
    model.visual.DAPM_replace(DPAM_layer=20)
    model.to("cuda")
    cond = ExtentConditioner(mode="extent").to("cuda").eval()
    cond.load_state_dict(state["conditioner"])
    for module in (model, learner, cond):
        module.requires_grad_(False)
    # Fixed-z conditioning is independent of the visual descriptor in extent mode.
    with torch.no_grad():
        _, pos, neg = cond(torch.zeros(1, 784, device="cuda"),
                           z_override=torch.full((1,), 1.6, device="cuda"))
        text = conditioned_text_features(model, learner, pos, neg)[0]
    return model, text


def probability_map(tokens, text):
    import torch.nn.functional as F
    import AnomalyCLIP_lib
    sim, _ = AnomalyCLIP_lib.compute_similarity(F.normalize(tokens, dim=-1), text)
    return AnomalyCLIP_lib.get_similarity_map(sim[:, 1:], 518).permute(0, 3, 1, 2).contiguous()


def fit(args, arm, output, provenance, smoke=False):
    import torch
    import torch.nn.functional as F
    from dataset import Dataset
    from utils import get_transform
    from loss import FocalLoss, BinaryDiceLoss
    from context_visual_residual import ContextVisualResidual, make_context_pair
    seed_all()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "manifest.json", dict(provenance, arm=arm, smoke=smoke,
          command=shlex.join([sys.executable, *sys.argv]), checkpoint_path=str(args.checkpoint),
          data_root=str(args.data_root)))
    preprocess, target_transform = get_transform(argparse.Namespace(image_size=518))
    source = Dataset(root=str(args.data_root / "mvtec"), transform=preprocess,
                     target_transform=target_transform, dataset_name="mvtec", mode="test", zoom_aug_p=0)
    if smoke:
        # Deterministic source-only samples; no target data are used for the step.
        source.data_all = source.data_all[:2]
        source.length = len(source.data_all)
    order_rng = torch.Generator().manual_seed(111)
    pair_rng = torch.Generator().manual_seed(7111)
    loader = torch.utils.data.DataLoader(source, batch_size=2, shuffle=True,
                                        generator=order_rng, num_workers=0)
    model, text = frozen_ecp(args)
    # Backbone initialization must not determine adapter RNG or differ across arms.
    torch.manual_seed(111)
    adapter = ContextVisualResidual().to("cuda").train()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-4)
    focal, dice = FocalLoss(), BinaryDiceLoss()
    initial = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    records = []
    epochs = 2 if smoke else 3
    for epoch in range(1, epochs + 1):
        sums = dict(loss=0.0, segmentation=0.0, core_focal=0.0, consistency=0.0, residual=0.0)
        batches = 0
        for item in loader:
            image = item["img"].to("cuda")
            mask = (item["img_mask"].to("cuda") > 0.5).float()
            edited, core_pixels, core_tokens = make_context_pair(image, mask, pair_rng)
            if arm == "plain":
                edited = image
            with torch.no_grad():
                _, feature = model.encode_image(image, [24], DPAM_layer=20)
                _, changed_feature = model.encode_image(edited, [24], DPAM_layer=20)
            base, other = feature[-1].float(), changed_feature[-1].float()
            adapted, adapted_other = adapter(base), adapter(other)
            probs, other_probs = probability_map(adapted, text), probability_map(adapted_other, text)
            gt = mask[:, 0].contiguous()
            segmentation = focal(probs, gt) + dice(probs[:, 1].contiguous(), gt) + dice(probs[:, 0].contiguous(), 1 - gt)
            # The edited far field can contain rearranged lesion pixels. Only the
            # unchanged central region retains the original segmentation labels.
            selected = core_pixels[:, 0].bool()
            local_probs = other_probs.permute(0, 2, 3, 1)[selected].contiguous()
            # FocalLoss expects a channel dimension on the target even when
            # the selected pixels have been flattened to [N, 2].
            core_loss = focal(local_probs, gt[selected].unsqueeze(1))
            a = F.normalize(adapted[:, 1:], dim=-1)
            b = F.normalize(adapted_other[:, 1:], dim=-1)
            consistency = (a - b).square().sum(-1)[core_tokens.bool()].mean()
            residual = ((adapted[:, 1:] - base[:, 1:]).square().mean() +
                        (adapted_other[:, 1:] - other[:, 1:]).square().mean()) / 2
            loss = segmentation + core_loss + .01 * residual
            if arm == "context":
                loss = loss + consistency
            if not torch.isfinite(loss):
                raise ValueError(f"{arm}: nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("frozen CLIP unexpectedly received gradients")
            optimizer.step()
            for key, val in zip(sums, (loss, segmentation, core_loss, consistency, residual)):
                sums[key] += float(val.detach())
            batches += 1
        if not batches:
            raise ValueError("empty source dataset")
        record = dict(epoch=epoch, batches=batches, **{k: v / batches for k, v in sums.items()})
        records.append(record)
        print(arm, json.dumps(record), flush=True)
        write(output / "loss_history.json", records)
        saved = dict(adapter={k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                     arm=arm, epoch=epoch, smoke=smoke, provenance=provenance)
        torch.save(saved, output / f"epoch_{epoch}.pth")
    if all(torch.equal(initial[k], v.detach().cpu()) for k, v in adapter.state_dict().items()):
        raise RuntimeError("optimizer did not change any residual parameter")
    write(output / "complete.json", dict(epoch=epochs, arm=arm, smoke=smoke,
                                         checkpoint_sha256=sha(output / f"epoch_{epochs}.pth")))
    del model, adapter, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def install_adapter(model, adapter):
    """Wrap the existing encoder result; image features and backbone are untouched."""
    model.context_residual = adapter
    original = model.encode_image

    def encode_image(self, image, feature_list=None, **kwargs):
        features = [24] if feature_list is None else feature_list
        if features != [24]:
            raise ValueError("EXP-016 requires feature24 only")
        image_features, patches = original(image, features, **kwargs)
        if len(patches) != 1:
            raise ValueError("expected exactly one patch feature layer")
        return image_features, [self.context_residual(patches[0])]

    model.encode_image = types.MethodType(encode_image, model)


def evaluate_one(args):
    import AnomalyCLIP_lib
    import test as evaluation
    from context_visual_residual import ContextVisualResidual
    seed_all()
    out = args.output
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    load_original = AnomalyCLIP_lib.load
    if args.adapter != "none":
        def wrapped_load(*a, **kw):
            model, transform = load_original(*a, **kw)
            adapter = ContextVisualResidual()
            if args.adapter != "identity":
                state = load_state(args.adapter)
                if state["provenance"]["base_checkpoint_sha256"] != sha(args.checkpoint):
                    raise ValueError("adapter belongs to a different ECP checkpoint")
                adapter.load_state_dict(state["adapter"])
            adapter.requires_grad_(False).eval()
            install_adapter(model, adapter)
            return model, transform
        AnomalyCLIP_lib.load = wrapped_load
    if args.limit:
        dataset_original = evaluation.Dataset
        def small_dataset(*a, **kw):
            dataset = dataset_original(*a, **kw)
            dataset.data_all = dataset.data_all[:args.limit]
            dataset.length = len(dataset.data_all)
            return dataset
        evaluation.Dataset = small_dataset
    # Existing visualizer writes one PNG per image. This pilot retains test.py's
    # inference and metric path while omitting visualization-only disk output.
    evaluation.visualizer = lambda *_args, **_kwargs: None
    folder, dataset_mode, _ = DATASETS[args.dataset_name]
    params = argparse.Namespace(data_path=str(args.data_root / folder), save_path=str(out),
        checkpoint_path=str(args.checkpoint), dataset=dataset_mode, features_list=[24],
        image_size=518, depth=9, n_ctx=12, t_n_ctx=4, feature_map_layer=[0, 1, 2, 3],
        metrics="pixel-level", export_image_scores=False, export_pixel_metrics=True,
        seed=111, sigma=4, ec_const_z=None, ec_z_img=-.69, ec_z_pix=1.6,
        distractor_suppress="none", distractor_rerank="none", ec_oracle=False)
    evaluation.test(params)
    artifacts = rows(out / "pixel_per_image_predictions.csv")
    if not artifacts or len({r["sample_id"] for r in artifacts}) != len(artifacts):
        raise ValueError("empty or duplicate evaluation rows")
    metadata = read(out / "evaluation_metadata.json")
    if len(artifacts) != metadata["sample_count"]:
        raise ValueError("some samples are missing binary-mask metrics")
    metadata.update(experiment="EXP-016", adapter_path=args.adapter,
        adapter_sha256=sha(args.adapter) if args.adapter not in ("none", "identity") else None,
        smoke_limit=args.limit, residual_applied_after_encode_image=True,
        visualization_export_disabled=True,
        experiment_source_sha256={f: sha(ROOT / f) for f in
            ("context_visual_residual.py", "scripts/exp016_context_residual.py", "scripts/run_exp016.sh")})
    write(out / "evaluation_metadata.json", metadata)


def child_eval(args, name, adapter, output, limit=0):
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "evaluate-one",
               "--data-root", str(args.data_root), "--checkpoint", str(args.checkpoint),
               "--dataset-name", name, "--adapter", str(adapter), "--output", str(output),
               "--limit", str(limit)]
    log = output.parent / (output.name + ".log")
    (output.parent / (output.name + ".command.txt")).write_text(shlex.join(command) + "\n")
    print(f"Evaluating {name}/{output.name}; log: {log}", flush=True)
    with log.open("w") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"evaluation failed; preserve and inspect {log}")


def check_paired(new, ref, subset=False):
    current = rows(new / "pixel_per_image_predictions.csv")
    previous = {r["sample_id"]: r for r in rows(ref / "pixel_per_image_predictions.csv")}
    if len({r['sample_id'] for r in current}) != len(current):
        raise ValueError("duplicate sample IDs")
    if not subset and {r["sample_id"] for r in current} != set(previous):
        raise ValueError("evaluation sample IDs differ from baseline")
    for row in current:
        old = previous[row["sample_id"]]
        if abs(float(row["mask_area_fraction"]) - float(old["mask_area_fraction"])) > 1e-7:
            raise ValueError("paired mask areas changed")
    return current, previous


def summarize(args):
    table = []
    for name, (_, _, cls) in DATASETS.items():
        ref = args.reference_root / name / "ecp_extent"
        base = read(ref / "pixel_level_metrics.json")[cls]
        for arm in ARMS:
            out = args.run_root / "eval" / name / arm
            current, previous = check_paired(out, ref)
            val = read(out / "pixel_level_metrics.json")[cls]
            record = dict(dataset=name, arm=arm, n=len(current), **val,
                delta_auroc_pp=100 * (val["pixel_auroc"] - base["pixel_auroc"]),
                delta_aupro_pp=100 * (val["pixel_aupro"] - base["pixel_aupro"]),
                mean_paired_per_image_delta=sum(float(r["per_image_pixel_auroc"]) -
                     float(previous[r["sample_id"]]["per_image_pixel_auroc"]) for r in current) / len(current))
            table.append(record)
            print(f"{name:9} {arm:8} AUROC {val['pixel_auroc']*100:.4f} "
                  f"({record['delta_auroc_pp']:+.4f}) AUPRO {val['pixel_aupro']*100:.4f} "
                  f"({record['delta_aupro_pp']:+.4f})", flush=True)
    write(args.run_root / "summary.json", table)
    with (args.run_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "train", "eval", "evaluate-one", "summarize"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-weights", type=Path, default=Path.home()/".cache/clip/ViT-L-14-336px.pt")
    parser.add_argument("--manifest", type=Path, default=ROOT/"checkpoints/EXP-012-matched/ecp_extent/training_manifest.json")
    parser.add_argument("--reference-root", type=Path, default=ROOT/"results/EXP-012/matched-retry-20261002")
    parser.add_argument("--run-root", type=Path, default=ROOT/"results/EXP-016")
    parser.add_argument("--train-root", type=Path, default=ROOT/"checkpoints/EXP-016")
    parser.add_argument("--dataset-name", choices=tuple(DATASETS))
    parser.add_argument("--adapter", default="none")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.stage == "evaluate-one":
        if args.dataset_name is None or args.output is None or args.limit < 0:
            parser.error("evaluate-one needs dataset-name/output and nonnegative limit")
        evaluate_one(args)
        return
    if args.stage == "summarize":
        summarize(args)
        return
    provenance = preflight(args)
    smoke_root = args.run_root / "smoke"
    if args.stage == "smoke":
        if smoke_root.exists():
            raise FileExistsError(f"preserve old output and use a new EXP016_ROOT: {smoke_root}")
        smoke_root.mkdir(parents=True)
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT/"tests"),
                        "-p", "test_context_visual_residual.py"], check=True, cwd=ROOT)
        fit(args, "context", smoke_root / "training", provenance, smoke=True)
        for arm, adapter in (("baseline", "none"), ("identity", "identity"),
                             ("trained", smoke_root/"training/epoch_2.pth")):
            child_eval(args, "ClinicDB", adapter, smoke_root / arm, limit=2)
        base, ref = check_paired(smoke_root/"baseline", args.reference_root/"ClinicDB/ecp_extent", subset=True)
        ident = rows(smoke_root/"identity/pixel_per_image_predictions.csv")
        if len(base) != 2 or len(ident) != 2:
            raise ValueError("smoke row count is not two")
        for b, i in zip(base, ident):
            if b != i or abs(float(b["per_image_pixel_auroc"]) - float(ref[b["sample_id"]]["per_image_pixel_auroc"])) > 1e-6:
                raise ValueError("zero-residual/reference smoke parity failed")
        if read(smoke_root/"baseline/pixel_level_metrics.json") != read(smoke_root/"identity/pixel_level_metrics.json"):
            raise ValueError("zero-residual aggregate metric parity failed")
        check_paired(smoke_root/"trained", smoke_root/"baseline")
        write(smoke_root / "passed.json", provenance)
        print("EXP-016 SMOKE PASSED")
        return
    if read(smoke_root / "passed.json") != provenance:
        raise ValueError("inputs/protocol/source changed after smoke; run a new smoke")
    if args.stage == "train":
        if args.train_root.exists():
            raise FileExistsError(args.train_root)
        args.train_root.mkdir(parents=True)
        for arm in ARMS:
            fit(args, arm, args.train_root / arm, provenance)
        print(f"EXP-016 THREE TRAINING ARMS COMPLETED: {args.train_root}")
    else:
        if (args.run_root / "eval").exists():
            raise FileExistsError(args.run_root / "eval")
        for arm in ARMS:
            file = args.train_root / arm / "epoch_3.pth"
            checkpoint = load_state(file)
            if checkpoint["provenance"] != provenance or checkpoint["smoke"] or checkpoint["arm"] != arm or checkpoint["epoch"] != 3:
                raise ValueError(f"{arm}: training configuration/source mismatch")
            if sha(file) != read(args.train_root/arm/"complete.json")["checkpoint_sha256"]:
                raise ValueError(f"{arm}: adapter checkpoint hash mismatch")
        for name in DATASETS:
            for arm in ARMS:
                child_eval(args, name, args.train_root / arm / "epoch_3.pth",
                           args.run_root / "eval" / name / arm)
        summarize(args)
        print(f"EXP-016 18 EVALUATIONS COMPLETED: {args.run_root}")


if __name__ == "__main__":
    main()
