"""x_r and BF of x = 0.55 on the scale-invariant line (r = a, uniform gamma p[8] times the taper), combo.py environment.
Usage (in the state's run dir): python3 si_readplace.py combo.json"""
import sys, json, math, numpy as np
sys.path.insert(0, '.')
from dataclasses import replace
CK = json.load(open(sys.argv[1]))
import geom, deltaT_3dof as d3, tl_model as T
geom.LAM = CK['lam']; geom.install()
T.section_terms = lambda sec, w, a, r=None, gamma=None: geom.terms(sec, w, a, r, gamma)
import exp_best1 as B
SECK = ('K_t', 'M_t', 'R_t', 'R_c', 'R_b', 'K_c', 'K_p', 'M_r')
sec = replace(T.WORLDS['B'].sec, **{k: v for k, v in zip(B.NAMES, CK['p']) if k in SECK})
g = CK['p'][8]; coch = replace(T.WORLDS['B'], sec=sec).at(g); BASE = math.exp(CK['z'][9])
fp = float(coch.cf(0.55))
for N in (1024, 2048, 4096):
    r = T.response(coch, fp, N=N, gamma=g); x = r['x']; xr = float(x[int(np.argmax(np.abs(r['X_b'])))])
    j = int(round(0.55*N)); ff = fp*2.0**np.linspace(-3.6, 0.6, 169); bv = [abs(f*T.response(coch, f, N=N, gamma=g)['X_b'][j]) for f in ff]
    print('N %d: x_r %.4f  BF(x=0.55) %.4f kHz' % (N, xr, ff[int(np.argmax(bv))]*BASE), flush=True)
