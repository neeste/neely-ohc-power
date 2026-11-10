import sys, json, math, os
sys.path.insert(1, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); import cat_state
sys.path.insert(0, '.')   # model/ modules
os.makedirs('out/fig9', exist_ok=True)
import numpy as np
from dataclasses import replace
import geom, tl_model as T, exp_best1 as B
geom.install()
T.section_terms = lambda sec, w, a, r=None, gamma=None: geom.terms(sec, w, a, r, gamma)
c = json.load(open(cat_state.COMBO))
SECK = ('K_t','M_t','R_t','R_c','R_b','K_c','K_p','M_r')
sec0 = replace(T.WORLDS['B'].sec, **{k: v for k, v in zip(B.NAMES, c['p']) if k in SECK})
coch0 = replace(T.WORLDS['B'], sec=sec0)
q = float(sys.argv[1]) if len(sys.argv) > 1 else 0.955
res = {}
for lam in [0.0, 0.1, 0.3, 0.5, 0.6, 0.65, 0.70846, 0.8561939116132914, 1.0]:
    geom.LAM = lam
    gc = geom.gamma_critical(replace(sec0, gamma=0.0))
    g = q*gc
    coch = coch0.at(g)
    out = {}
    for xf in (0.35, 0.5):
        f = float(coch.cf(xf))
        r = T.response(coch, f, N=8192, gamma=g)
        x = r['x']; cf = r['cf']
        def pk(y):
            i = int(np.argmax(abs(y)[50:-50]))+50; return i
        ib = pk(r['X_b']); ish = pk(r['shear'])
        Xb, Xr = r['X_b'], r['X_r']; Xt = Xr - r['H_o']*Xb/coch.sec.G
        D, p = geom.power(coch.sec, f, cf, cf, r['gamma_x'], Xb, Xr, Xt)
        sh = p/D; m = slice(50, len(x)-50)
        im = int(np.argmax(sh[m]))+50
        oct_ = lambda i, j: (x[i]-x[j])*coch.decades*math.log2(10)   # positive apical (i apical of j)
        out[xf] = dict(wa_bm=f/cf[ib], wa_sh=f/cf[ish], wa_sharemax=f/cf[im], sharemax=float(sh[im]),
                       sharemax_oct_apical_of_bm=float(oct_(im, ib)), shear_peak_oct_apical_of_bm=float(oct_(ish, ib)))
    res[lam] = out
    print(lam, {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in out.items()}, flush=True)
json.dump(res, open('out/fig9/line_peak_q%.3f.json' % q, 'w'), indent=1)
