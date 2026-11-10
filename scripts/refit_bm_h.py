"""refit_bm_h.py: human, OHC force on BM only. Refit only the activity
profile (mu0, alpha_mu, nu floor, basal cap) for wave-V rms, accepting only
certified states. Writes out/refit_bm_h.json."""
import sys, json, math, time
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '.')
import geom
geom.GEOM = 'bm'; geom.install()
import exp_abr2 as A2, exp_abr4 as A4, exp_abr5 as A5
from cert2 import certify
ck = json.load(open('abr5_ckpt.json')); p = dict(ck['params']); S0 = ck['S0']
H = A2.Human(**p)
gc = geom.gc_of_rho(H.coch.sec, H.r/H.a, n=160)
OUT = 'out/refit_bm_h.json'
best = dict(rms=1e9)

def gam_of(z):
    mu0, amu, nfl, cap = math.exp(z[0]), z[1], math.exp(z[2]), 1 - math.exp(z[3])
    nu = np.clip(mu0*H.a**amu, nfl, 1.0); mu = 1 - nu
    t = np.clip((H.x - A5.X0)/(A5.X1 - A5.X0), 0, 1)
    w = 1 - (3*t*t - 2*t**3)
    mu = (1 - w)*mu + w*np.minimum(mu, cap)
    return mu*gc*H.coch.taper(H.x)

def obj(z):
    g = gam_of(z); H.gam0 = g; H.stable = True
    try:
        rms, info = A4.joint_rms(H, S0, A5.TH_GRID, A5.TAU_GRID, A5.PR_GRID)
    except Exception:
        return 10.0
    if rms < best['rms']:
        ok, bb = certify(H.coch.sec, H.coch.F, H.a, H.r, g, 1024, lgw=(-3.15, 0.35))
        if not ok:
            return rms + 0.5
        best.update(rms=rms, z=list(z), readout={k: info[k] for k in ('theta','tau_a','pr','a')},
                    nu_min=float(np.min(1 - g/(gc*H.coch.taper(H.x)+1e-300))))
        json.dump(best, open(OUT, 'w'), indent=1)
        print('incumbent', round(rms, 4), [round(v, 4) for v in z], flush=True)
    return rms

# start from a stable, backed-off state: floor 0.05, cap 0.90
z0 = [math.log(p['mu0']), p['alpha_mu'], math.log(0.05), math.log(0.10)]
print('start', obj(z0), flush=True)
res = minimize(obj, z0, method='Nelder-Mead',
               options=dict(maxfev=int(sys.argv[1]) if len(sys.argv) > 1 else 250,
                            xatol=1e-3, fatol=1e-4,
                            initial_simplex=[z0] + [list(np.add(z0, d)) for d in
                               np.diag([0.5, 0.1, 0.5, 0.5])]))
print('done', best.get('rms'), flush=True)
