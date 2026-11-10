"""Section-only OHC share against normalized frequency w/a (rho = 1, a = 1),
for the c708 section at several lambda, gamma = q*gamma_c(lambda).
Uses p_ohc = 1/2 g Ka G a^2 w |Xb|^2 Im{Ho' conj(lam Hr - 1)} (|Xb| cancels)."""
import sys, json, math, os
sys.path.insert(1, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); import cat_state
sys.path.insert(0, '.')   # model/ modules
os.makedirs('out/fig9', exist_ok=True)
import os
import numpy as np
from dataclasses import replace
import geom, tl_model as T, exp_best1 as B
c = json.load(open(cat_state.COMBO))
SECK = ('K_t','M_t','R_t','R_c','R_b','K_c','K_p','M_r')
sec0 = replace(T.WORLDS['B'].sec, **{k: v for k, v in zip(B.NAMES, c['p']) if k in SECK})
q = float(sys.argv[1]) if len(sys.argv) > 1 else 0.955
w = np.geomspace(0.2, 1.5, 3000)
out = {}
for lam in [0.0, 0.1, 0.3, 0.5, 0.6, 0.65, 0.70846, 0.8561939116132914, 1.0]:
    geom.LAM = lam
    gc = geom.gamma_critical(replace(sec0, gamma=0.0))
    s = replace(sec0, gamma=q*gc)
    Y, Yr, Ho = geom.terms(s, w, 1.0, 1.0)
    Xb = Y; Xr = Yr
    # recover X_t from Ho = G (Xr - Xt)/Xb
    Xt = Xr - Ho*Xb/s.G
    D, p = geom.power(s, w, 1.0, 1.0, s.gamma, Xb, Xr, Xt)
    sh = p/D
    # check local formula
    Hr = Xr/Xb; Hs = (Xr-Xt)/Xb
    p2 = 0.5*s.gamma*s.K_a*s.G*w*abs(Xb)**2*np.imag(Hs*np.conj(lam*Hr-1))
    err = float(np.max(abs(p2-p))/np.max(abs(p)))
    # local maxima
    i = [k for k in range(1, len(w)-1) if sh[k] > sh[k-1] and sh[k] >= sh[k+1]]
    above = w[sh > 1]
    out[lam] = dict(gc=gc, err=err, maxima=[(float(w[k]), float(sh[k])) for k in i],
                    band=[float(above.min()), float(above.max())] if len(above) else None,
                    share_at_060=float(np.interp(0.60, w, sh)))
    np.savetxt('out/fig9/share_local_q%.3f_l%.3f.dat' % (q, lam), np.c_[w, sh, p, D])
print(json.dumps(out, indent=1))
