import sys, json, numpy as np
sys.path.insert(0, '.')
import exp_abr2 as A2, exp_abr5 as A5, exp_sfoae8 as S8
ck = json.load(open('abr5_ckpt.json')); p = dict(ck['params']); S0 = ck['S0']
H = A2.Human(**p); gc = A5.gc_exact_dense(H)
H.gam0 = A5.capped_gamma(H, p['mu0'], p['alpha_mu'], gc); H.stable = True
out = dict(f=[0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8], n20=[], n30=[], n40=[])
for spl, key in ((20, 'n20'), (30, 'n30'), (40, 'n40')):
    for fk in out['f']:
        r = S8.gd8(H, S0, f_khz=fk, spl=spl)
        out[key].append(r if isinstance(r, (int, float)) else r)
        print(spl, fk, r, flush=True)
json.dump(out, open('out/sfoae_human.json', 'w'), indent=1, default=float)
