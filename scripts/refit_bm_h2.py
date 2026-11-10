"""Human, BM-only force: refit all eight abr5 parameters plus the nu floor
and basal cap (section fixed, as in abr5), certified states only."""
import sys, json, math, time
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '.')
import geom
geom.GEOM = 'bm'; geom.install()
import exp_abr2 as A2, exp_abr4 as A4, exp_abr5 as A5
from cert2 import certify
ck = json.load(open('abr5_ckpt.json')); p0 = dict(ck['params']); sec_over = p0.pop('sec_over')
st = json.load(open('out/refit_bm_h.json'))
OUT = 'out/refit_bm_h2.json'
best = dict(rms=1e9)
def unpack(z):
    return dict(alpha_R=z[0], c_R=math.exp(z[1]), mu0=math.exp(z[2]), alpha_mu=z[3],
                k=z[4], F=math.exp(z[5]), p_sat=math.exp(z[6])), z[7], math.exp(z[8]), 1-math.exp(z[9])
def build(z):
    pd, S0, nfl, cap = unpack(z)
    H = A2.Human(**dict(pd, sec_over=sec_over))
    gc = geom.gc_of_rho(H.coch.sec, H.r/H.a, n=160)
    nu = np.clip(pd['mu0']*H.a**pd['alpha_mu'], nfl, 1.0); mu = 1-nu
    t = np.clip((H.x-A5.X0)/(A5.X1-A5.X0), 0, 1); w = 1-(3*t*t-2*t**3)
    mu = (1-w)*mu + w*np.minimum(mu, cap)
    H.gam0 = mu*gc*H.coch.taper(H.x); H.stable = True
    return H, S0, gc
def obj(z):
    pd, S0, nfl, cap = unpack(z)
    if not (10 < S0 < 90) or not (0 < pd['k'] < 0.5): return 10.0
    try:
        H, S0, gc = build(z)
        rms, info = A4.joint_rms(H, S0, A5.TH_GRID, A5.TAU_GRID, A5.PR_GRID)
    except Exception:
        return 10.0
    if rms < best['rms']:
        ok, _ = certify(H.coch.sec, H.coch.F, H.a, H.r, H.gam0, 1024, lgw=(-3.15, 0.35))
        if not ok: return rms + 0.5
        best.update(rms=rms, z=list(map(float, z)), readout={k: info[k] for k in ('theta','tau_a','pr','a')})
        json.dump(best, open(OUT, 'w'), indent=1)
        print('incumbent', round(rms, 4), [round(v, 3) for v in z], flush=True)
    return rms
z1 = st['z']
z0 = [p0['alpha_R'], math.log(p0['c_R']), z1[0], z1[1], p0['k'], math.log(p0['F']),
      math.log(p0['p_sat']), ck['S0'], z1[2], z1[3]]
obj(z0)
minimize(obj, z0, method='Nelder-Mead', options=dict(maxfev=int(sys.argv[1]), xatol=1e-4, fatol=1e-4))
print('done', best.get('rms'), flush=True)
