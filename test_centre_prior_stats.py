"""Known-answer tests for centre_prior_stats.py (EXP-050). numpy/scipy only; the dataset/extent_prompt cross-checks run only where torch is installed.
  python test_centre_prior_stats.py
"""
import math
import random
import numpy as np
import centre_prior_stats as S


def blob(h, w, cy, cx, r):
    yy, xx = np.mgrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def test_geometry():
    assert S.mask_offset(blob(101, 101, 50, 50, 10)) < 1e-9
    m = np.zeros((101, 101), bool); m[0, 0] = True
    assert abs(S.mask_offset(m) - math.sqrt(2)) < 1e-9
    # non-square frame: each axis by its own half-side -> a blob at 3/4 of the width and centred vertically is 0.5 away
    m = blob(101, 201, 50, 150, 5)
    assert abs(S.mask_offset(m) - 0.5) < 1e-9
    # equals zpos_stats.centroid_offset on a square frame
    import zpos_stats
    m = blob(80, 80, 20, 61, 6)
    assert abs(S.mask_offset(m) - zpos_stats.centroid_offset(m)) < 1e-12
    # area_to_z values from the extent_prompt docstring
    assert abs(S.area_to_z(0.005) + 0.6931) < 1e-3 and abs(S.area_to_z(0.02)) < 1e-12 and abs(S.area_to_z(0.15) - 1.0074) < 1e-3
    assert abs(S.area_to_z(0.0001) - S.area_to_z(0.005)) < 1e-12        # clamp
    box = (10, 20, 30, 50)
    assert S.crop(np.ones((60, 60), bool), box).shape == (30, 20)


def test_extent_prompt_constants():
    try:
        import torch
        import extent_prompt as E
    except Exception:
        print("  (torch/extent_prompt not importable here: constant cross-check skipped)")
        return
    assert (E.AREA_MIN, E.Z_REF, E.Z_SCALE) == (S.AREA_MIN, S.Z_REF, S.Z_SCALE)
    a = np.array([0.001, 0.02, 0.3, 0.9])
    assert np.allclose(E.area_to_z(torch.tensor(a)).numpy(), S.area_to_z(a))


def corner_masks(n=60, side=100, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for k in range(n):
        r = rng.randint(4, 12)
        cy, cx = [(r + 3, r + 3), (r + 3, side - r - 4), (side - r - 4, r + 3), (side - r - 4, side - r - 4)][k % 4]
        out.append(blob(side, side, cy, cx, r))
    return out


def centred_box(m):
    """crop that puts the lesion exactly in the middle (odd side so the centroid is the crop centre), clipped inside the frame when possible"""
    ys, xs = np.nonzero(m)
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) // 2 + 1
    x0, y0 = cx - half, cy - half
    if x0 < 0 or y0 < 0 or cx + half + 1 > m.shape[1] or cy + half + 1 > m.shape[0]:
        return None
    return (x0, y0, cx + half + 1, cy + half + 1)


def test_part_a_centralises():
    random.seed(0)
    v = S.simulate_visits(corner_masks(), centred_box, epochs=6, p=0.5)
    st = S.part_a(v, n_boot=300)
    assert st["A1"] < -0.5, st["A1"]
    assert S.a1_label(st) == "CENTRALISES"
    assert S.verdict_a(st)[0] == "SUPPORTED"


def test_part_a_decentralises():
    # native lesions at the centre, the "crop" pushes them into a corner -> A1 > 0 -> NO_CENTRALISATION; sizes and positions independent -> no coupling
    rng = np.random.RandomState(1)
    masks = [blob(100, 100, 50, 50, rng.randint(4, 12)) for _ in range(60)]

    def corner_box(m):
        ys, xs = np.nonzero(m)
        return (xs.min(), ys.min(), xs.min() + 60, ys.min() + 60)
    random.seed(1)
    st = S.part_a(S.simulate_visits(masks, corner_box, epochs=6, p=0.5), n_boot=300)
    assert st["A1"] > 0.3 and S.a1_label(st) == "NO_CENTRALISATION"


def test_part_a_planted_coupling():
    # native: size independent of position (rho_native ~ 0); the crop centres ONLY the large lesions and enlarges them -> mixture rho strongly negative
    rng = np.random.RandomState(2)
    masks = []
    for _ in range(80):
        r = rng.randint(4, 14)
        masks.append(blob(120, 120, rng.randint(r + 1, 120 - r - 1), rng.randint(r + 1, 120 - r - 1), r))

    def big_centred(m):
        return centred_box(m) if m.sum() > np.pi * 9 ** 2 else None
    random.seed(2)
    v = S.simulate_visits(masks, big_centred, epochs=8, p=0.7)
    st = S.part_a(v, n_boot=300)
    assert abs(st["rho_native"]) < 0.3, st["rho_native"]
    assert st["A2"] < -0.2, st
    assert S.a2_label(st) == "COUPLES" and S.verdict_a(st)[0] == "SUPPORTED"


def test_part_a_null():
    # no crop ever succeeds: A1 undefined -> INCOMPLETE; the mixture is the native set repeated -> A2 == 0 exactly
    random.seed(3)
    v = S.simulate_visits(corner_masks(40), lambda m: None, epochs=4, p=1.0)
    st = S.part_a(v, n_boot=100)
    assert not v["zoomed"].any() and abs(st["A2"]) < 1e-12
    assert S.verdict_a(st)[0] == "INCOMPLETE"
    # identity "crop" (whole frame): zoomed visits equal native -> A1 = 0 exactly -> NO_CENTRALISATION and NO_COUPLING -> NOT SUPPORTED
    random.seed(3)
    st = S.part_a(S.simulate_visits(corner_masks(40), lambda m: (0, 0, m.shape[1], m.shape[0]), epochs=4, p=0.5), n_boot=100)
    assert abs(st["A1"]) < 1e-12 and abs(st["A2"]) < 1e-12 and S.verdict_a(st)[0] == "NOT SUPPORTED"
    # attempt / success counts: p = 1 -> every visit is an attempt; a sampler that always fails -> 0 successes
    v = S.simulate_visits(corner_masks(10), lambda m: None, epochs=3, p=1.0)
    assert v["n_attempts"] == 30 and v["n_success"] == 0 and len(v["img"]) == 30


def test_labels_thresholds():
    mk = lambda a1, hi1, a2, hi2: dict(A1=a1, A1_ci=(a1 - 0.01, hi1), A2=a2, A2_ci=(a2 - 0.01, hi2))
    assert S.a1_label(mk(-0.05, -0.001, 0, 0)) == "CENTRALISES"             # exactly at the threshold counts
    assert S.a1_label(mk(-0.05, 0.001, 0, 0)) == "UNCLEAR"                  # CI crosses 0
    assert S.a1_label(mk(-0.02, 0.0, 0, 0)) == "NO_CENTRALISATION"
    assert S.a2_label(mk(0, 0, -0.10, -0.001)) == "COUPLES"
    assert S.a2_label(mk(0, 0, -0.03, 0.01)) == "NO_COUPLING"
    assert S.verdict_a(mk(-0.03, 0.01, -0.05, 0.01))[0] == "INCONCLUSIVE"


def rows(ids, g, sham=0.8, pe_rand=0.7, group="peripheral", status="kept"):
    return [dict(id=i, group=group, status=status, auroc_sham=sham, auroc_pe_rand=pe_rand, auroc_pe_rec=pe_rand + gi) for i, gi in zip(ids, g)]


def test_part_b():
    ids = [f"im{k}" for k in range(50)]
    rng = np.random.RandomState(0)
    hi = rows(ids, 0.10 + 0.01 * rng.randn(50))
    st = S.paired_g(hi, rows(ids, 0.02 + 0.01 * rng.randn(50)), n_boot=500)
    assert st["n"] == 50 and abs(st["Q"] - st["G_lo"] / st["G_hi"]) < 1e-12 and S.set_label_b(st) == "COUPLED"
    assert S.set_label_b(S.paired_g(hi, rows(ids, 0.095 + 0.01 * rng.randn(50)), n_boot=500)) == "NOT"
    assert S.set_label_b(S.paired_g(hi, rows(ids, 0.065 + 0.01 * rng.randn(50)), n_boot=500)) == "MIXED"
    assert S.set_label_b(S.paired_g(rows(ids, [-0.01] * 50), rows(ids, [0.0] * 50))) == "INVALID"
    # exact thresholds: Q = 0.5 with a positive dG -> COUPLED; Q = 0.8 -> NOT
    assert S.set_label_b(S.paired_g(rows(ids, [0.10] * 50), rows(ids, [0.05] * 50))) == "COUPLED"
    assert S.set_label_b(S.paired_g(rows(ids, [0.10] * 50), rows(ids, [0.08] * 50))) == "NOT"
    # pairing: only ids kept in both runs; central rows and dropped rows are ignored
    lo = rows(ids[:30], [0.02] * 30) + rows(ids[30:], [0.02] * 20, status="dropped_region") + rows(["c1"], [0.5], group="central")
    st = S.paired_g(hi, lo)
    assert st["n"] == 30 and st["n_hi_only"] == 20
    bad = rows(ids, [0.02] * 50); bad[3]["sy_rec"] = 5
    try:
        S.paired_g(hi, bad); raise AssertionError("geometry mismatch not caught")
    except ValueError:
        pass
    assert S.paired_g(hi, rows(ids[:2], [0.0, 0.0]))["n"] == 2 and S.set_label_b(S.paired_g(hi, rows(ids[:2], [0.0, 0.0]))) == "INVALID"


def test_verdicts_b():
    d = lambda a, b, c: dict(zip(S.DECISION, (a, b, c)))
    assert S.seed_verdict_b(d("COUPLED", "COUPLED", "NOT")) == "Z-COUPLED"
    assert S.seed_verdict_b(d("NOT", "NOT", "COUPLED")) == "NOT Z-COUPLED"
    assert S.seed_verdict_b(d("COUPLED", "MIXED", "NOT")) == "MIXED"
    assert S.seed_verdict_b(d("COUPLED", "INVALID", "INVALID")) == "INCONCLUSIVE"
    assert S.seed_verdict_b({"ClinicDB": "COUPLED"}) == "INCOMPLETE"
    assert S.verdict_b({"s111": "Z-COUPLED", "s222": "Z-COUPLED", "s333": "MIXED"}) == "Z-COUPLED"
    assert S.verdict_b({"s111": "NOT Z-COUPLED", "s222": "MIXED", "s333": "NOT Z-COUPLED"}) == "NOT Z-COUPLED"
    assert S.verdict_b({"s111": "Z-COUPLED", "s222": "NOT Z-COUPLED", "s333": "MIXED"}) == "MIXED"
    assert S.verdict_b({"s111": "Z-COUPLED", "s222": "INCOMPLETE", "s333": "Z-COUPLED"}) == "INCOMPLETE"


def test_real_crop_box():
    """Dataset._sample_crop_box (the training code itself, compiled from dataset.py): the box always contains the whole lesion, is square,
    lies inside the frame and makes the lesion fill the target range (up to integer rounding of the side)."""
    f, h = S.load_crop_box("dataset.py")
    assert len(h) == 64
    random.seed(0)
    for (H, W, cy, cx, r) in [(300, 300, 80, 210, 20), (300, 400, 150, 30, 15), (200, 200, 100, 100, 25)]:
        m = blob(H, W, cy, cx, r)
        for _ in range(200):
            b = f(m, (0.10, 0.45))
            assert b is not None
            x0, y0, x1, y1 = b
            assert x1 - x0 == y1 - y0 and x0 >= 0 and y0 >= 0 and x1 <= W and y1 <= H
            c = S.crop(m, b)
            assert c.sum() == m.sum()
            assert 0.08 <= c.mean() <= 0.47, c.mean()
    # a lesion too large to zoom out returns None; an empty mask returns None
    assert f(blob(100, 100, 50, 50, 45), (0.10, 0.45)) is None
    assert f(np.zeros((50, 50), bool), (0.10, 0.45)) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("all centre_prior_stats tests passed")
