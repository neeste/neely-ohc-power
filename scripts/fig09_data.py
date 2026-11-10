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
w = np.geomspace(0.3, 1.5, 4000)
rows = []
for lam in np.round(np.arange(0, 1.0001, 0.05), 3).tolist() + [0.70846, 0.8561939116132914]:
    geom.LAM = lam
    gc = geom.gamma_critical(replace(sec0, gamma=0.0)); g = q*gc
    s = replace(sec0, gamma=g)
    M, C, K = geom.matrices(s); Mi = np.linalg.inv(M); Z = np.zeros((3,3))
    ev = np.linalg.eigvals(np.block([[Z, np.eye(3)], [-Mi@K, -Mi@C]])); ev = ev[ev.imag > 0]
    ev = ev[np.argsort(ev.imag)]
    Y, Yr, Ho = geom.terms(s, w, 1.0, 1.0); Xt = Yr - Ho*Y/s.G
    D, p = geom.power(s, w, 1.0, 1.0, g, Y, Yr, Xt); sh = p/D
    above = w[sh > 1]; im = int(np.argmax(sh))
    coch = coch0.at(g); f = float(coch.cf(0.45))
    r = T.response(coch, f, N=8192, gamma=g)
    ib = int(np.argmax(abs(r['X_b'])[50:-50]))+50; ish = int(np.argmax(abs(r['shear'])[50:-50]))+50
    rows.append(dict(lam=lam, gc=gc, modes=[(float(z.imag), float(z.imag/(-2*z.real))) for z in ev],
                     band=[float(above.min()), float(above.max())] if len(above) else None,
                     wa_sharemax=float(w[im]), sharemax=float(sh[im]),
                     wa_bm=float(f/r['cf'][ib]), wa_shear=float(f/r['cf'][ish])))
    print(lam, round(gc,4), [(round(a,3), round(b,1)) for a,b in rows[-1]['modes']], rows[-1]['band'] and [round(x,3) for x in rows[-1]['band']], round(rows[-1]['wa_bm'],3), round(rows[-1]['wa_shear'],3), flush=True)
json.dump(rows, open('out/fig9/fig09_data_q%.3f.json' % q + '', 'w'), indent=1)
