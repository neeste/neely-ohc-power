import sys, json, numpy as np
sys.path.insert(0, '.')
from cert2 import certify
import exp_abr2 as A2, exp_abr5 as A5
ck = json.load(open('abr5_ckpt.json')); p = dict(ck['params'])
H = A2.Human(**p); gc = A5.gc_exact_dense(H)
for mu in (0.95, 0.97, 0.99):
    g = mu*gc*H.coch.taper(H.x)
    ok, b = certify(H.coch.sec, H.coch.F, H.a, H.r, g, 1024, lgw=(-3.15, 0.35))
    print('human uniform mu', mu, 'clean', ok, [bb[2] for bb in b], flush=True)
