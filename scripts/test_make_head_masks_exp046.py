"""CPU unit tests for scripts/make_head_masks_exp046.py and the mask I/O helpers of vvattn_stats (numpy only).
Run from the repo root: python scripts/test_make_head_masks_exp046.py"""
import os, sys, json, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vvattn_stats as S
import make_head_masks_exp046 as M

rng = np.random.RandomState(1)
scores = rng.rand(19, 16)
scores[3, 4] = np.nan                                  # NaN counts as 0.5
scores[0, 0] = scores[0, 1] = 2.0                      # tie at the top: (layer 6, head 0) before (layer 6, head 1)
m = M.build_masks(scores)
h, r = m["head50"], m["rand50_s0"]
assert h.shape == r.shape == (19, 16) and h.dtype == r.dtype == bool and h.sum() == r.sum() == 152
assert h[0, 0] and h[0, 1]
# equality with the vvattn_stats functions (the ones EXP-045 used)
assert np.array_equal(h, S.mask_from_ranking(S.ranking_from_scores(scores), "top", 152))
assert np.array_equal(r, S.random_mask(152, 0))
# layout: flat index = row * 16 + head; the kept set is exactly the top-152 of the flat scores (ties by flat index, = (layer, head) order)
flat = np.where(np.isnan(scores), 0.5, scores).reshape(-1)
top = sorted(range(304), key=lambda i: (-flat[i], i))[:152]
assert set(np.flatnonzero(h.reshape(-1)).tolist()) == set(top)
assert np.flatnonzero(r.reshape(-1)).tolist() == sorted(np.random.RandomState(0).permutation(304)[:152].tolist())
# agreement check against a recorded EXP-045 json (format of analyze_vvattn: arm -> argwhere list)
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "masks.json")
    json.dump({"head_top50": M.mask_to_pairs(h), "head_rand50_s0": M.mask_to_pairs(r), "base": None}, open(p, "w"))
    assert M.check_against_exp045(m, p)
    bad = M.mask_to_pairs(h)[:-1] + [[18, 15]] if not h[18, 15] else M.mask_to_pairs(h)[:-1]
    json.dump({"head_top50": bad, "head_rand50_s0": M.mask_to_pairs(r)}, open(p, "w"))
    try:
        M.check_against_exp045(m, p); raise SystemExit("mismatch not detected")
    except AssertionError:
        pass
    # file round trip + validation
    np.save(os.path.join(d, "ok.npy"), h)
    keep, sha = S.load_keep(os.path.join(d, "ok.npy"))
    assert np.array_equal(keep, h) and len(sha) == 64
    np.save(os.path.join(d, "ones_f.npy"), np.ones((19, 16), np.float32))          # 0/1 floats accepted
    assert S.load_keep(os.path.join(d, "ones_f.npy"))[0].all()
    for badm in (np.ones((16, 19), bool), np.ones(304, bool), np.full((19, 16), 2)):
        np.save(os.path.join(d, "bad.npy"), badm)
        try:
            S.load_keep(os.path.join(d, "bad.npy")); raise SystemExit("bad mask accepted")
        except ValueError:
            pass
    # resolve_test_mask
    rec = S.mask_record(os.path.join(d, "ok.npy"), h, sha)
    assert rec["n_keep"] == 152 and np.array_equal(np.array(rec["keep"], bool), h)
    assert S.resolve_test_mask("", None) == (None, None)
    k, rr = S.resolve_test_mask("", rec); assert np.array_equal(k, h)                       # checkpoint-only: its own mask is used
    k, rr = S.resolve_test_mask(os.path.join(d, "ok.npy"), rec); assert np.array_equal(k, h)
    np.save(os.path.join(d, "other.npy"), r)
    for flag, rc in ((os.path.join(d, "other.npy"), rec), (os.path.join(d, "ok.npy"), None)):   # mismatch / flag on an unmasked checkpoint -> refused
        try:
            S.resolve_test_mask(flag, rc); raise SystemExit("mismatch accepted")
        except ValueError:
            pass
    tampered = dict(rec, keep=r.astype(int).tolist())
    try:
        S.resolve_test_mask("", tampered); raise SystemExit("tampered record accepted")
    except ValueError:
        pass
    import pickle; pickle.dumps(rec)                                                         # plain python only
print("test_make_head_masks_exp046 OK")
