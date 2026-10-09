import numpy as np
from zshuffle import derangement_among

rng = np.random.RandomState(0)
for n in range(0, 12):
    for _ in range(50):
        mask = rng.rand(max(n, 1)) < 0.6
        mask = mask[:n] if n else mask[:0]
        src = derangement_among(mask, rng)
        assert sorted(src.tolist()) == list(range(n)), "not a permutation"
        sel = np.flatnonzero(mask)
        assert np.all(src[~mask] == np.arange(n)[~mask]), "non-selected positions must keep their value"
        if sel.size >= 2:
            assert np.all(src[sel] != sel), "every selected position must get a different one"
            assert set(src[sel].tolist()) == set(sel.tolist()), "values stay among the selected positions"
        else:
            assert np.all(src == np.arange(n)), "fewer than 2 selected: identity"
# marginal multiset of z is unchanged, pairing broken
z = np.array([0.1, 0.5, -0.3, 1.2, 0.9, 0.0])
mask = np.array([1, 1, 0, 1, 1, 0], bool)
zs = z[derangement_among(mask, np.random.RandomState(1))]
assert sorted(zs[mask]) == sorted(z[mask]) and np.all(zs[mask] != z[mask]) and np.all(zs[~mask] == z[~mask])
print("test_zshuffle OK")
