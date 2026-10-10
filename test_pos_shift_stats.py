"""Known-answer tests for pos_shift_stats.py (run: python test_pos_shift_stats.py). No torch."""
import numpy as np
import pos_shift_stats as P


def brute_auc(y, s):
    pos, neg = s[y], s[~y]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def test_reflect_matches_np_pad():
    a = np.arange(30.0).reshape(5, 6)
    for dx, dy in [(0, 0), (2, -1), (-3, 3), (1, 1), (-2, -2)]:
        pad = 4
        ref = np.pad(a, pad, mode="reflect")[pad - dy:pad - dy + 5, pad - dx:pad - dx + 6]
        assert np.array_equal(P.shift_image(a, dx, dy), ref), (dx, dy)
    # large magnitude: index stays inside and is periodic
    idx = P.shift_indices(10, 37)
    assert idx.min() >= 0 and idx.max() <= 9
    assert np.array_equal(P.shift_image(a, 0, 0), a)


def test_round_trip_and_valid():
    rng = np.random.RandomState(0)
    a = rng.rand(40, 40)
    for dx, dy in [(5, -7), (-12, 3), (0, 9)]:
        back = P.inverse_align(P.shift_image(a, dx, dy), dx, dy)
        v = P.valid_mask(a.shape, dx, dy)
        assert np.array_equal(np.isfinite(back), v)               # NaN exactly outside the valid region
        assert np.allclose(back[v], a[v])                         # round trip is exact on valid pixels
        assert v.sum() == (40 - abs(dx)) * (40 - abs(dy))
    assert P.valid_mask((8, 8), 20, 0).sum() == 0                 # shift larger than canvas


def test_common_valid_excludes_padded():
    s = (50, 50)
    v = P.common_valid(s, [(10, 0), (-6, 8)])
    # (10,0): x+10 < 50 -> x < 40; (-6,8): x-6 >= 0 -> x >= 6 and y+8 < 50 -> y < 42
    exp = np.zeros(s, bool); exp[0:42, 6:40] = True
    assert np.array_equal(v, exp)
    assert P.common_valid(s, [(0, 0)]).all()


def test_region_auroc_exact():
    rng = np.random.RandomState(1)
    y = np.zeros((12, 12), bool); y[3:9, 3:9] = True              # 36 pos, 108 neg
    s = np.round(rng.rand(12, 12), 1)                             # many ties
    reg = np.ones_like(y)
    assert abs(P.region_auroc(y, s, reg) - brute_auc(y.ravel(), s.ravel())) < 1e-12
    s2 = y.astype(float)
    assert P.region_auroc(y, s2, reg) == 1.0
    assert P.region_auroc(y, -s2, reg) == 0.0
    assert abs(P.region_auroc(y, np.zeros_like(s2), reg) - 0.5) < 1e-12
    reg2 = np.zeros_like(y); reg2[:, :9] = True
    assert abs(P.region_auroc(y, s, reg2) - brute_auc(y[:, :9].ravel(), s[:, :9].ravel())) < 1e-12


def test_region_drop_rule():
    y = np.zeros((10, 10), bool); y[0, :] = True; y[1, :9] = True       # 19 lesion pixels
    s = np.random.RandomState(2).rand(10, 10)
    reg = np.ones_like(y)
    assert P.region_auroc(y, s, reg) is None                             # 19 < 20
    y[1, 9] = True                                                       # 20 lesion pixels -> kept
    assert P.region_auroc(y, s, reg) is not None
    reg[2:, :] = False; reg[2:3, :] = True                               # 20 lesion + 10 bg
    assert P.region_auroc(y, s, reg) is None                             # background < 20
    assert P.region_counts(y, reg) == (20, 10)


def test_displacements():
    m = np.zeros((51, 51), bool); m[5:10, 20:31] = True                  # centroid (7, 25) -> centre (25, 25)
    dx, dy = P.recentre_displacement(m)
    assert (dx, dy) == (0, 18)
    cy, cx = P.centroid(m)
    assert abs(cy + dy - 25) < 0.5 and abs(cx + dx - 25) < 0.5      # integer rounding only
    rx, ry = P.random_displacement("a/b.png", 30, -40)
    assert abs(np.hypot(rx, ry) - 50) <= 1.0
    assert P.random_displacement("a/b.png", 30, -40) == (rx, ry)         # deterministic
    angs = {P.random_displacement(f"id{i}", 30, -40) for i in range(20)}
    assert len(angs) > 10
    assert P.random_displacement("x", 0, 0) == (0, 0)


def test_terciles():
    c = np.array([0.5, 0.9, 0.1, 0.9, 0.3, 0.7, 0.2])
    ids = ["g", "f", "e", "a", "d", "c", "b"]
    p, q = P.terciles(c, ids)
    assert len(p) == 3 and len(q) == 3                                   # ceil(7/3)
    assert list(p) == [3, 1, 5]                                          # ties 0.9: id 'a' (idx 3) before 'f' (idx 1)
    assert list(q) == [2, 6, 4]


def toy_model(img, k):
    """Pointwise 'model': content score minus a penalty growing with distance from the canvas centre (a planted position response)."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2) / (h / 2)
    return img - k * r


def run_toy(k, seed, S=120):
    rng = np.random.RandomState(seed)
    y = np.zeros((S, S), bool); y[8:28, 75 + rng.randint(0, 20):95 + rng.randint(0, 20)] = True   # peripheral lesion
    img = 0.6 * y + 0.25 * rng.randn(S, S)
    d = P.recentre_displacement(y)
    r = P.random_displacement(f"toy{seed}", *d)
    reg = P.common_valid((S, S), [d, r])
    out = {}
    for name, (dx, dy) in (("sham", (0, 0)), ("rec", d), ("rand", r)):
        m = P.inverse_align(toy_model(P.shift_image(img, dx, dy), k), dx, dy)
        out[name] = P.region_auroc(y, m, reg)
    return out


def test_planted_position_gives_positive_delta():
    outs = [run_toy(2.0, s) for s in range(60)]
    d = [o["rec"] - o["rand"] for o in outs if o["rec"] is not None]          # None = image dropped by the >= 20 pixels filter
    assert len(d) >= 20
    assert np.mean(d) > 0.05 and np.mean(np.array(d) > 0) > 0.8


def test_position_invariant_gives_zero_delta():
    for s in range(10):
        o = run_toy(0.0, s)
        if o["rec"] is None:
            continue
        assert abs(o["rec"] - o["rand"]) < 1e-12 and abs(o["sham"] - o["rec"]) < 1e-12   # pointwise map: identical on the same pixels


def test_sham_identity():
    rng = np.random.RandomState(3)
    a = rng.rand(30, 30); y = rng.rand(30, 30) > 0.8
    assert np.array_equal(P.inverse_align(P.shift_image(a, 0, 0), 0, 0), a)
    reg = P.common_valid(a.shape, [(7, -3), (0, 0)])
    assert P.region_auroc(y, a, reg) == P.region_auroc(y, P.inverse_align(P.shift_image(a, 0, 0), 0, 0), reg)


def test_boot_and_set_summary():
    x = np.full(50, 0.02)
    m, lo, hi = P.boot_mean(x)
    assert m == 0.02 and lo == 0.02 and hi == 0.02
    rng = np.random.RandomState(4)
    x = rng.randn(400) * 0.05 + 0.02
    m, lo, hi = P.boot_mean(x)
    assert lo < 0.02 < hi and abs(m - x.mean()) < 1e-15
    a0 = rng.rand(40); mag = rng.rand(40) * 100
    st = P.set_summary(a0, a0 + 0.1, a0, mag)
    assert abs(st["delta"][0] - 0.1) < 1e-12 and abs(st["rec_minus_sham"][0] - 0.1) < 1e-12 and abs(st["rand_minus_sham"][0]) < 1e-12


def test_verdict_boundaries():
    S = lambda m, lo=0.001, hi=None: (m, lo, m + 0.02 if hi is None else hi)
    names = ("ClinicDB", "ColonDB", "ISIC", "Endo")
    mk = lambda ms: dict(zip(names, ms))
    # exactly +0.010 with CI lower > 0 passes
    assert P.verdict(mk([S(0.010)] * 3 + [S(0.0)]))[0] == "POSITION-RESPONSE SUPPORTED"
    assert P.verdict(mk([S(0.0099999)] * 3 + [S(0.0)]))[0] != "POSITION-RESPONSE SUPPORTED"
    # CI lower bound exactly 0 does not pass
    assert P.verdict(mk([S(0.02, 0.0)] * 4))[0] != "POSITION-RESPONSE SUPPORTED"
    # only 2 of 4 support -> not supported; not null either -> inconclusive
    assert P.verdict(mk([S(0.02)] * 2 + [S(0.006, -0.01, 0.02)] * 2))[0] == "INCONCLUSIVE"
    # null: mean <= 0.003 passes (boundary), or CI upper < 0.010
    assert P.verdict(mk([S(0.003, -0.01, 0.05)] * 3 + [S(0.02)]))[0] == "NOT SUPPORTED"
    assert P.verdict(mk([S(0.0030001, -0.01, 0.05)] * 3 + [S(0.02)]))[0] == "INCONCLUSIVE"
    assert P.verdict(mk([S(0.008, -0.01, 0.0099)] * 3 + [S(0.02)]))[0] == "NOT SUPPORTED"
    assert P.verdict(mk([S(0.008, -0.01, 0.0100)] * 3 + [S(0.02)]))[0] == "INCONCLUSIVE"
    # negative delta with CI entirely below zero is a null pass
    assert P.verdict(mk([S(-0.02, -0.03, -0.01)] * 4))[0] == "NOT SUPPORTED"
    # missing / NaN set
    assert P.verdict({"ClinicDB": S(0.02)})[0] == "INCOMPLETE"
    assert P.verdict(mk([S(0.02)] * 3 + [(float("nan"),) * 3]))[0] == "INCOMPLETE"
    # supported and null cannot both hold
    for m in (0.0, 0.003, 0.0031, 0.009, 0.010, 0.05):
        assert not (P.pass_support(m, 0.001) and P.pass_null(m, m + 0.02))


def test_hit_rate():
    y = np.zeros((6, 6), bool); y[:3, :] = True
    s = np.zeros((6, 6)); s[0, :] = 1.0
    assert P.hit_rate(y, s, np.ones_like(y), 0.5) == 1 / 3
    assert np.isnan(P.hit_rate(y, s, np.zeros_like(y), 0.5))


def test_fill_shift_matches_reflect_on_valid_and_is_constant_elsewhere():
    rng = np.random.RandomState(0)
    img = rng.randn(3, 20, 24)
    for dx, dy in ((0, 0), (5, -3), (-7, 4), (9, 9), (-10, -8)):
        f = P.shift_image_fill(img, dx, dy, np.array([0.1, -0.2, 0.3]).reshape(3, 1, 1))
        # content: every valid output pixel equals the translated source, and agrees with the reflect version there
        r = P.shift_image(img, dx, dy)
        h, w = img.shape[-2:]
        vo = np.zeros((h, w), bool)
        y0, y1 = max(0, -dy), min(h, h - dy); x0, x1 = max(0, -dx), min(w, w - dx)
        if y1 > y0 and x1 > x0:
            vo[y0 + dy:y1 + dy, x0 + dx:x1 + dx] = True
        assert np.allclose(f[:, vo], r[:, vo])
        # padding: exactly the fill value, no image content
        for c, val in enumerate((0.1, -0.2, 0.3)):
            assert np.all(f[c][~vo] == val)
        # the valid area is the valid_mask translated
        assert vo.sum() == P.valid_mask((h, w), dx, dy).sum()
    # identity
    assert np.array_equal(P.shift_image_fill(img, 0, 0, 0.0), img)
    # scalar fill, shift larger than the canvas -> everything is fill
    assert np.all(P.shift_image_fill(img, 100, 0, 0.0) == 0.0)


def test_artifact_free_verdict_boundaries():
    def S(d, dlo, dhi, r, rlo):
        return dict(delta=(d, dlo, dhi), rec_minus_sham=(r, rlo, r + 0.05))
    ok = S(0.05, 0.03, 0.07, 0.04, 0.02)
    names = ("ClinicDB", "ColonDB", "ISIC", "Endo")
    mk = lambda lst: dict(zip(names, lst))
    assert P.verdict_artifact_free(mk([ok] * 4))[0] == "ARTIFACT-FREE SUPPORT"
    assert P.verdict_artifact_free(mk([ok] * 3 + [S(0.0, -0.02, 0.01, 0.0, -0.01)]))[0] == "ARTIFACT-FREE SUPPORT"
    # delta passes but the recentre gain alone does not (random shifts hurt, recentre does not help): not artifact-free support
    weak_rec = S(0.05, 0.03, 0.07, 0.004, 0.001)
    assert P.verdict_artifact_free(mk([weak_rec] * 4))[0] == "INCONCLUSIVE"
    # recentre gain exactly at the threshold counts, just below does not
    assert P.rec_ok(0.005, 0.0001) and not P.rec_ok(0.0049, 0.0001) and not P.rec_ok(0.01, 0.0)
    nul = S(0.002, -0.01, 0.009, 0.0, -0.01)
    assert P.verdict_artifact_free(mk([nul] * 3 + [ok]))[0] == "ARTIFACT SUSPECTED"
    assert P.verdict_artifact_free(mk([nul] * 2 + [ok] * 2))[0] == "INCONCLUSIVE"
    assert P.verdict_artifact_free({"ClinicDB": ok})[0] == "INCOMPLETE"
    assert P.verdict_artifact_free(mk([ok] * 3 + [None]))[0] == "INCOMPLETE"


if __name__ == "__main__":
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_") and callable(f):
            f(); n += 1; print("ok", k)
    print(f"{n} tests passed")
