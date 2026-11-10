"""Run 4 item 3: positive control for the human stability test. Same construction and certificate as cert_h_lam.py
(argument-principle winding, cert2_h8k.certify, lgw (-3.15, 0.35), 10 bands; N 2048 at depths 0.35 and 1.0, N 8192 at
depth 0.35), quiet line only, at four profiles built on the hs708 state (joint_hs_s708.json):
  hs708        : the adopted profile (nu = clip(mu0 a^alpha_mu, nu_floor, 1), basal cap mu <= 0.945 for x < 0.08,
                 smoothstep to the intended profile at x = 0.14); reference, should be clean
  no basal cap : the same without the cap, i.e. gamma/gamma_c = 1 - nu_floor = 0.956 everywhere (before the taper)
  uniform 0.99 : gamma0 = 0.99 gamma_c taper
  uniform 1.01 : gamma0 = 1.01 gamma_c taper
The taper is the model's own end window (tl_model.Cochlea.taper), as in every gamma profile of the model.
Winding = number of zeros of the line determinant inside each band contour (growing modes); total = sum over bands.
Usage (from model/): python3 ../scripts/run_human.py ../scripts/cert_h_ctrl.py joint_hs_hs856.json out/cert_h_ctrl.json"""
import sys, os, json, time
sys.path.insert(1, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); import cat_state
import numpy as np
sys.path.insert(0, '.')
import exp_abr2 as A2, exp_abr5 as A5, exp_accumulation as E, geom
from cert2_h8k import certify
c = json.load(open(sys.argv[1])); pd = dict(c['params']); sec_over = pd.pop('sec_over'); nfl = c['nu_floor']
H = A2.Human(**dict(pd, sec_over=sec_over)); gc = A5.gc_exact_dense(H); tap = H.coch.taper(H.x)
nu = np.clip(pd['mu0']*H.a**pd['alpha_mu'], nfl, 1.0); mu_nc = 1 - nu
t = np.clip((H.x-A5.X0)/(A5.X1-A5.X0), 0, 1); w = 1-(3*t*t-2*t**3)
mu_c = (1-w)*mu_nc + w*np.minimum(mu_nc, A5.CAP)
prof = {'hs708': mu_c*gc*tap, 'no basal cap': mu_nc*gc*tap, 'uniform 0.99': 0.99*gc*tap, 'uniform 1.01': 1.01*gc*tap}
OUT = sys.argv[2]
res = json.load(open(OUT)) if os.path.exists(OUT) else dict(source=sys.argv[1], lam=geom.LAM, N_model=len(H.x)-1, rows={})
res['ratio_ranges'] = {k: [float(np.min(g[tap > 0.999]/gc[tap > 0.999])), float(np.max(g[tap > 0.999]/gc[tap > 0.999]))] for k, g in prof.items()}
print('[ctrl] lam %.5f N_model %d; gamma/gamma_c range where taper = 1: %s' % (geom.LAM, len(H.x)-1, res['ratio_ranges']), flush=True)
for name, gam in prof.items():
    for N in (2048, 8192):
        for depth in ((0.35, 1.0) if N <= 2048 else (0.35,)):
            key = '%s | N=%d depth=%g' % (name, N, depth)
            if key in res['rows']: continue
            x = np.linspace(0, 1, N+1)
            a = np.interp(x, H.x, H.a); r = np.interp(x, H.x, H.r); g = np.interp(x, H.x, gam)
            t0 = time.time()
            ok, bands = certify(H.coch.sec, H.coch.F, a, r, g, N, lgw=(-3.15, 0.35), depth=depth)
            wtot = float(sum(b[2] for b in bands))
            res['rows'][key] = dict(clean=ok, winding_total=wtot, growing_modes=int(round(wtot)), bands=bands, secs=time.time()-t0)
            json.dump(res, open(OUT, 'w'), indent=1)
            print('%-14s N=%d depth=%g: clean=%s winding total %.3f, per band %s (%.0fs)' % (name, N, depth, ok, wtot,
                  [b[2] for b in bands], time.time()-t0), flush=True)
print('CTRL DONE', flush=True)
