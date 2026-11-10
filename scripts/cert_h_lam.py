"""Argument-principle certification of a joint_h state (e.g. the lambda = 0.708 human refit)
at N = 2048 (depth 0.35, 1.0) and N = 8192 (depth 0.35), quiet and at saturated profiles.
Usage (from model/): python3 ../scripts/cert_h_lam.py joint_hs_hs856.json out/cert_h.json  (lambda from the state file)"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, '.')
_lam = os.environ.get('LAM', json.load(open(sys.argv[1])).get('lam'))   # lambda: LAM, else the state file
if _lam is not None:
    import geom, deltaT_3dof as d3
    geom.LAM = float(_lam); geom.install()
    d3.gamma_critical = lambda sec, hi=200.0, tol=1e-6: geom.gamma_critical(sec, 1.0, hi, tol)
import exp_abr2 as A2, exp_abr5 as A5, exp_accumulation as E
from cert2_h8k import certify
A2.N = 8192
c = json.load(open(sys.argv[1])); pd = dict(c['params']); sec_over = pd.pop('sec_over'); S0 = c['S0']; nfl = c['nu_floor']
H = A2.Human(**dict(pd, sec_over=sec_over)); gc = A5.gc_exact_dense(H)
nu = np.clip(pd['mu0']*H.a**pd['alpha_mu'], nfl, 1.0); mu = 1-nu
t = np.clip((H.x-A5.X0)/(A5.X1-A5.X0), 0, 1); w = 1-(3*t*t-2*t**3)
mu = (1-w)*mu + w*np.minimum(mu, A5.CAP)
gam0 = mu*gc*H.coch.taper(H.x); H.gam0 = gam0; H.stable = True
N0 = len(H.x) - 1
states = {'quiet': np.zeros(N0+1)}
for fk, spl in ((0.5, 100), (2.0, 40), (2.0, 100), (8.0, 70), (8.0, 100)):
    f = fk/20.7; amp = 10.0**((spl - S0)/20.0)
    r0 = H.line2(f, np.zeros(N0+1)); C = float(np.max(np.abs(r0['shear'])))
    def FF(sv):
        rr = H.line2(f, sv); u = (amp*np.abs(rr['shear'])/C)**H.p_sat
        return u/(1.0+u)
    s, _, _ = E.anderson(FF, np.zeros(N0+1)); states['%g kHz %d dB SPL' % (fk, spl)] = s
res = dict(source=sys.argv[1], lam=os.environ.get('LAM'))
for name, s in states.items():
    gam = gam0*(1 - s)
    for N in (2048, 8192):
        x = np.linspace(0, 1, N+1)
        a = np.interp(x, H.x, H.a); r = np.interp(x, H.x, H.r); g = np.interp(x, H.x, gam)
        for depth in ((0.35, 1.0) if N <= 2048 else (0.35,)):
            t0 = time.time()
            ok, bands = certify(H.coch.sec, H.coch.F, a, r, g, N, lgw=(-3.15, 0.35), depth=depth)
            res['%s | N=%d depth=%g' % (name, N, depth)] = dict(clean=ok, bands=bands, smax=float(np.max(s)))
            json.dump(res, open(sys.argv[2], 'w'), indent=1)
            print('%-22s N=%d depth=%g: clean=%s smax=%.3f (%.0fs)' % (name, N, depth, ok, np.max(s), time.time()-t0), flush=True)
print('done', all(v['clean'] for k, v in res.items() if isinstance(v, dict)), flush=True)
