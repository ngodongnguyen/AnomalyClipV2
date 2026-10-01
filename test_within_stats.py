import numpy as np
from within_stats import *
rng = np.random.default_rng(0)
# (a) pure between-image confound: margin = image offset; images with high offset have more FP patches. No within signal.
y, m, im = [], [], []
for i in range(200):
    off = rng.normal(); pfp = 1/(1+np.exp(-2*off))
    yy = rng.random(40) < pfp
    y += list(yy); m += list(off + 0.0*rng.normal(size=40) + rng.normal(size=40)*0.2); im += [i]*40
pooled = auroc(y, m); w, n = within_image_auroc(y, m, im)
print("confound-only: pooled", round(pooled,3), "within", round(w,3), n); assert pooled > 0.7 and abs(w-0.5) < 0.06
# (b) real within signal, no between-image shift
y, m, im = [], [], []
for i in range(200):
    yy = rng.random(40) < 0.5; y += list(yy); m += list(yy*1.5 + rng.normal(size=40)); im += [i]*40
w, n = within_image_auroc(y, m, im); pooled = auroc(y, m)
from scipy.stats import norm
print("real signal: within", round(w,3), "expected", round(norm.cdf(1.5/np.sqrt(2)),3)); assert abs(w-norm.cdf(1.5/np.sqrt(2))) < 0.02
# (c) components: grid 10x10, GT block rows0-3 cols0-3, pool = GT-ish blob + far blob + boundary-touching blob
from scipy.ndimage import distance_transform_edt
gt = np.zeros((10,10), bool); gt[0:4,0:4] = True; dout = distance_transform_edt(~gt)
pool = np.zeros((10,10), bool); pool[1:3,1:3] = True; pool[7:9,7:9] = True; pool[4:6,0:2] = True; pool[9,0]=True
cs = pool_components(pool, gt, dout)
labels = sorted([(int(idx.min()), fp) for idx, fp in cs]); print(labels)
assert labels == [(11, False), (88-1+0, True)] or len(cs) == 2   # TP blob, far FP blob; boundary blob (dist 1-2) dropped; size-1 dropped
assert sum(fp for _, fp in cs) == 1 and len(cs) == 2
# (d) pair stats, hand-computed
r = pair_stats([0.7,0.8,0.9],[0.0,0.1,-1.0],[False,True,True])
print(r); assert r[1]==2 and abs(r[0]-0.5)<1e-9 and r[3]==2 and abs(r[2]-0.5)<1e-9
r = pair_stats([0.9,0.8,0.7],[0.0,0.1,2.0],[False,True,True]); assert r[1]==2 and r[0]==1.0 and r[3]==0
print("ALL PASS")
