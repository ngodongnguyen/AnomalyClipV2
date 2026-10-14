"""EXP-046: build the two V-V head masks from the EXP-045 MVTec head scores (CPU, numpy only).
  results/EXP-046/head50_mask.npy   top-50% (152 of 304) heads by results/EXP-045/head_scores_mvtec.npy; ranking and tie rule of
                                    vvattn_stats.ranking_from_scores (descending score, NaN -> 0.5, ties by (layer, head)) and mask_from_ranking(.., 'top', 152)
  results/EXP-046/rand50_s0_mask.npy  the EXP-045 random-50% mask of seed 0: vvattn_stats.random_mask(152, 0)
Layout of both: bool [19, 16], row i = V-V layer 6+i, column = head, True = kept. When results/EXP-045/vvattn_ClinicDB_masks.json exists, both are asserted
equal to the masks recorded there (entries 'head_top50' and 'head_rand50_s0', lists of [row, col]).
  python scripts/make_head_masks_exp046.py [--scores ...] [--out_dir results/EXP-046] [--exp045_masks results/EXP-045/vvattn_ClinicDB_masks.json]"""
import os
import sys
import json
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import vvattn_stats as S


def build_masks(scores):
    scores = np.asarray(scores, np.float64)
    assert scores.shape == (S.N_LAYERS, S.N_HEADS), scores.shape
    return {"head50": S.mask_from_ranking(S.ranking_from_scores(scores), "top", 152), "rand50_s0": S.random_mask(152, 0)}


def mask_to_pairs(m):
    return np.argwhere(m).tolist()


def check_against_exp045(masks, path):
    rec = json.load(open(path))
    for mine, key in (("head50", "head_top50"), ("rand50_s0", "head_rand50_s0")):
        assert mask_to_pairs(masks[mine]) == [list(p) for p in rec[key]], f"{mine} differs from {key} recorded in {path}"
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=os.path.join(ROOT, "results", "EXP-045", "head_scores_mvtec.npy"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "results", "EXP-046"))
    ap.add_argument("--exp045_masks", default=os.path.join(ROOT, "results", "EXP-045", "vvattn_ClinicDB_masks.json"))
    a = ap.parse_args()
    masks = build_masks(np.load(a.scores))
    if os.path.isfile(a.exp045_masks):
        check_against_exp045(masks, a.exp045_masks)
        print(f"asserted equal to the EXP-045 masks recorded in {a.exp045_masks}")
    else:
        print(f"WARNING: {a.exp045_masks} not found, equality with the EXP-045 masks NOT checked")
    os.makedirs(a.out_dir, exist_ok=True)
    for name, m in masks.items():
        path = os.path.join(a.out_dir, f"{name}_mask.npy")
        np.save(path, m)
        keep, sha = S.load_keep(path)                                  # re-read: shape/dtype validation
        assert np.array_equal(keep, m)
        print(f"{path}: shape {keep.shape} dtype {m.dtype} kept {int(keep.sum())}/{keep.size}, per-layer kept {keep.sum(1).tolist()}, file sha256 {sha}")
    print("overlap head50 & rand50_s0:", int((masks['head50'] & masks['rand50_s0']).sum()), "of 152")


if __name__ == "__main__":
    main()
