import random
import numpy as np
from PIL import Image
import transaug as T

rng = random.Random(0)
W, H = 120, 90
img = Image.fromarray((np.random.RandomState(0).rand(H, W, 3) * 255).astype(np.uint8), "RGB")
m = np.zeros((H, W), np.uint8); m[30:50, 60:100] = 255                 # lesion bbox rows 30..50, cols 60..100
mask = Image.fromarray(m, "L")

# anomalous: the lesion bounding box always stays inside, the area is unchanged, the content moves with the mask
for _ in range(300):
    i2, m2, (dx, dy) = T.translate_pair(img, mask, rng, True)
    a2 = np.asarray(m2) > 127
    assert a2.sum() == (m > 127).sum(), "lesion area changed"
    ys, xs = np.nonzero(a2)
    assert ys.min() >= 0 and ys.max() < H and xs.min() >= 0 and xs.max() < W
    assert (ys.min(), xs.min()) == (30 + dy, 60 + dx)
    # image content under the lesion is the original lesion content
    assert np.array_equal(np.asarray(i2)[a2], np.asarray(img)[m > 127])
# uncovered area is exactly the fill, no image content
i3 = T.translate_fill(img, 15, -10, T.CLIP_MEAN_RGB)
a = np.asarray(i3)
assert np.all(a[:, :15] == np.array(T.CLIP_MEAN_RGB)) and np.all(a[-10:, :] == np.array(T.CLIP_MEAN_RGB))
assert np.array_equal(a[:H - 10, 15:], np.asarray(img)[10:, :W - 15])
# mask translated with fill 0 and the same displacement
m3 = np.asarray(T.translate_fill(mask, 15, -10, 0))
assert m3.sum() <= m.sum() and np.array_equal(m3[:H - 10, 15:], m[10:, :W - 15])
# uniform coverage of feasible shifts and the bounds of the shift
dxs = set(); dys = set()
r2 = random.Random(1)
for _ in range(5000):
    dx, dy = T.sample_shift_anomalous(m > 127, r2); dxs.add(dx); dys.add(dy)
assert min(dxs) == -60 and max(dxs) == W - 100 and min(dys) == -30 and max(dys) == H - 50
# normal images: bounded by the fraction, deterministic for a seeded generator
for _ in range(500):
    dx, dy = T.sample_shift_normal(W, H, r2)
    assert abs(dx) <= round(0.3 * W) and abs(dy) <= round(0.3 * H)
a1 = T.translate_pair(img, Image.new("L", img.size, 0), random.Random(5), False)[2]
a2 = T.translate_pair(img, Image.new("L", img.size, 0), random.Random(5), False)[2]
assert a1 == a2
# a lesion that fills the canvas has exactly one feasible shift
full = np.full((H, W), 255, np.uint8)
assert T.sample_shift_anomalous(full > 127, random.Random(2)) == (0, 0)
# empty mask on an 'anomalous' image falls back to a normal-style shift (no crash)
_ = T.translate_pair(img, Image.new("L", img.size, 0), random.Random(3), True)
print("test_transaug OK")
