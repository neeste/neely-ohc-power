"""lam_scope.py: location of the OHC power share on the stage-1 line
(world B, uniform gamma) for the stage-1 state (force pair, lambda = 1) and
the lambda refit state. Share = p_ohc / D; location in octaves re the BM
response peak (+ basal). Also checks the energy identity D = p_f + p_ohc.
Usage: lam_scope.py [lam_refit json]"""
import sys, json
import numpy as np
from dataclasses import replace
sys.path.insert(0, '.')
import geom, tl_model as T, deltaT_3dof as d3
geom.install()
T.section_terms = lambda sec, w, a, r=None, gamma=None: geom.terms(sec, w, a, r, gamma)
import exp_best1 as B
SECK = ('K_t', 'M_t', 'R_t', 'R_c', 'R_b', 'K_c', 'K_p', 'M_r')
def state(params, lam):
    coch = replace(T.WORLDS['B'], sec=replace(T.WORLDS['B'].sec, **{k: params[k] for k in SECK}))
    return coch, params['gamma'], lam
d3p = json.load(open('best_d3.json'))['params']
lr = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'out/lam_refit_l0.7.json'))
cases = [('pair, stage 1', d3p, 1.0), ('lambda refit', dict(zip(B.NAMES, lr['p'])), lr['lam']),
         ('lambda refit section, lambda = 1', dict(zip(B.NAMES, lr['p'])), 1.0)]
out = {}
N = 2048
for lab, pp, lam in cases:
    geom.LAM = lam
    coch, g, _ = state(pp, lam)
    gc = geom.gamma_critical(coch.sec)
    if g/gc >= 0.97: g = 0.95*gc
    rows = []
    for x0 in (0.3, 0.5, 0.7):
        f = float(coch.cf(x0))
        r = T.response(coch, f, N=N, gamma=g)
        x = r['x']; a = coch.cf(x); X_b, X_r = r['X_b'], r['X_r']
        X_t = X_r - r['H_o']*X_b/coch.sec.G
        D, p = geom.power(coch.sec, f, a, a, r['gamma_x'], X_b, X_r, X_t)
        Z = 1/(1j*f*r['Y']); pf = 0.5*Z.real*np.abs(1j*f*X_b)**2
        err = float(np.max(np.abs(D - pf - p))/np.max(D))
        ipk = int(np.argmax(np.abs(X_b)))
        octs = np.log2(a/a[ipk])
        share = p/np.maximum(D, 1e-300)
        win = np.abs(octs) < 2
        js = np.where(win)[0][np.argmax(share[win])]
        jp = np.where(win)[0][np.argmax(p[win])]
        rows.append(dict(x0=x0, share_max=float(share[js]), share_max_oct=float(octs[js]),
                         pohc_max_oct=float(octs[jp]), share_at_peak=float(share[ipk]), identity_err=err))
    out[lab] = dict(lam=lam, gamma_ratio=g/gc, rows=rows)
    print(lab, 'lam %.3f ratio %.3f' % (lam, g/gc))
    for rw in rows: print('   ', {k: round(v, 3) for k, v in rw.items()})
json.dump(out, open('out/lam_scope.json', 'w'), indent=1)
