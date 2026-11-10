"""Check the obs45 fix: run from a directory where the fixed exp_ftc2.py shadows the original.
Usage (from model/, plain python3): python3 ../scripts/check_fix.py earlier/combo_c708.json   (prints obs45 with the installed, corrected exp_ftc2.py)"""
import sys, json, math, numpy as np
sys.path.insert(0, '.')
from dataclasses import replace
import geom, deltaT_3dof as d3, tl_model as T
st = json.load(open(sys.argv[1]))
geom.LAM = st['lam']; geom.install()
T.section_terms = lambda sec, w, a, r=None, gamma=None: geom.terms(sec, w, a, r, gamma)
d3.gamma_critical = lambda sec, hi=200.0, tol=1e-6: geom.gamma_critical(sec, 1.0, hi, tol)
import exp_best1 as B
B.P_ORD = 4.0; B.WIN['fwd'] = (0.0, 12.5)
import exp_ftc1 as F1, exp_ftc2 as X
def gc_dense(rho):
    rg = np.geomspace(max(rho.min()*0.98, 1e-3), rho.max()*1.02, 24); gg = []
    for r_ in rg:
        S = F1.SEC; s2 = replace(S, gamma=0.0, R_b=S.R_b*r_, R_r=S.R_r*r_, R_t=S.R_t*r_, R_p=S.R_p*r_, R_c=S.R_c*r_)
        gg.append(d3.gamma_critical(s2))
    return np.interp(np.log(rho), np.log(rg), np.asarray(gg))
F1.gc_dense = gc_dense
SECK = ('K_t','M_t','R_t','R_c','R_b','K_c','K_p','M_r')
sec = replace(T.WORLDS['B'].sec, **{k: v for k, v in zip(B.NAMES, st['p']) if k in SECK})
coch = replace(T.WORLDS['B'], sec=sec); F1.SEC = X.SEC = sec; F1.COCH = X.COCH = coch
r, g = X.profiles(st['z']); m = X.obs45(r, g)
print('obs45', [round(float(t), 3) for t in m], ' stored', [round(t, 3) for t in st.get('obs45', [])])
