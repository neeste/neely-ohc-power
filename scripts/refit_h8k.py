"""Human wave-V refit on a converged grid (N = 8192), same 8 parameters
and read-out grid as exp_abr5, starting from abr5_ckpt. Accepts only
states that pass the argument-principle test at N = 2048 and 8192."""
import sys, json, math, time
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, '.')
import exp_abr2 as A2, exp_abr4 as A4, exp_abr5 as A5
from cert2 import certify
NN = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
A2.N = NN
ck = json.load(open('abr5_ckpt.json')); p0 = dict(ck['params']); sec_over = p0.pop('sec_over'); S0 = ck['S0']
OUT = 'out/refit_h8k.json'
best = dict(rms=1e9)
def build(pd):
    H = A2.Human(**dict(pd, sec_over=sec_over)); gc = A5.gc_exact_dense(H)
    H.gam0 = A5.capped_gamma(H, pd['mu0'], pd['alpha_mu'], gc); H.stable = True
    return H
def cert(H):
    for Nc in (2048, NN):
        x = np.linspace(0, 1, Nc+1)
        ok, _ = certify(H.coch.sec, H.coch.F, np.interp(x, H.x, H.a), np.interp(x, H.x, H.r),
                        np.interp(x, H.x, H.gam0), Nc, lgw=(-3.15, 0.35))
        if not ok: return False
    return True
def obj(z):
    pd, S0_ = A5.unpack(z)
    if not (10 < S0_ < 90): return 10.0
    try:
        H = build(pd); rms, info = A4.joint_rms(H, S0_, A5.TH_GRID, A5.TAU_GRID, A5.PR_GRID)
    except Exception:
        return 10.0
    if rms < best['rms'] - 1e-4:
        if not cert(H): print('reject', round(rms, 4), flush=True); return rms + 0.5
        best.update(rms=rms, params=dict(pd, sec_over=sec_over), S0=S0_,
                    readout={k: info[k] for k in ('theta', 'tau_a', 'pr', 'a')}, N=NN,
                    lat=[float(v) for v in info['lat']], keys=info['keys'])
        json.dump(best, open(OUT, 'w'), indent=1)
        print(time.strftime('%H:%M:%S'), 'incumbent', round(rms, 4), best['readout'], flush=True)
    return rms
import os
if os.path.exists(OUT):   # resume from the last certified incumbent
    _ck = json.load(open(OUT)); p0 = dict(_ck['params']); p0.pop('sec_over', None); S0 = _ck['S0']
    print('resuming from', OUT, 'rms', _ck['rms'], flush=True)
z0 = A5.pack({k: p0[k] for k in ('alpha_R', 'c_R', 'mu0', 'alpha_mu', 'k', 'F', 'p_sat')}, S0)
obj(z0)
minimize(obj, z0, method='Nelder-Mead', options=dict(maxfev=int(sys.argv[1]), xatol=1e-4, fatol=1e-4))
print('done', best['rms'], flush=True)
