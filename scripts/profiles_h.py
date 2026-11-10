import sys, json, math
import numpy as np
sys.path.insert(0, '.')
import exp_abr2 as A2
import exp_abr5 as A5
ck = json.load(open('abr5_ckpt.json'))
p = dict(ck['params'])
H = A2.Human(**p)
gc = A5.gc_exact_dense(H)
gam = A5.capped_gamma(H, p['mu0'], p['alpha_mu'], gc)
H.gam0 = gam
N = len(H.x) - 1
x = H.x; a = H.a
nu_eff = 1 - gam / (gc * H.coch.taper(x) + 1e-300)
fr = np.geomspace(0.1/20.7, 16/20.7, 40)
xpk = []
for f in fr:
    R = H.line2(f, np.zeros(N + 1))
    xpk.append(float(x[int(np.argmax(np.abs(R['X_b'])))]))
d = json.load(open('out/profiles.json'))
d['human'] = dict(x=x[::8].tolist(), map_khz=(a * 20.7)[::8].tolist(),
                  rho=(H.r / a)[::8].tolist(), nu=nu_eff[::8].tolist(),
                  taper=H.coch.taper(x)[::8].tolist(), gam0=gam[::8].tolist(),
                  gc=gc[::8].tolist(), pm_f_khz=(fr * 20.7).tolist(), pm_x=xpk,
                  B_khz=20.7)
json.dump(d, open('out/profiles.json', 'w'))
for f, xx in zip(fr[::5] * 20.7, xpk[::5]):
    print(f'  {f:8.3f}  {xx:.4f}')
i = np.searchsorted(x, [0.0, 0.1, 0.2, 0.47, 0.6, 0.7])
print('nu at', x[i], nu_eff[i]); print('rho', (H.r/a)[i])
