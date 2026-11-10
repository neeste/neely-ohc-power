"""Stage-1 refit with a fitted force split lambda (fraction of the OHC force
acting on the RL; 1 = BM-RL pair, 0 = BM only) and the BM-gain window
(20-60 dB) added to the eleven stage-1 metrics. Usage: lam_refit.py lam0 maxfev tag"""
import sys, json, math, time
import numpy as np
from dataclasses import replace
from scipy.optimize import minimize
sys.path.insert(0, '.')
import geom, deltaT_3dof as d3, tl_model as T, exp_accumulation as E
geom.LAM = float(sys.argv[1]); geom.install()
T.section_terms = lambda sec, w, a, r=None, gamma=None: geom.terms(sec, w, a, r, gamma)
d3.gamma_critical = lambda sec, hi=200.0, tol=1e-6: geom.gamma_critical(sec, 1.0, hi, tol)
import exp_best1 as B
B.P_ORD = 4.0; B.WIN['fwd'] = (0.0, 12.5)
from bmgain_refit import LO, HI, GWIN, bm_gain
TAG = sys.argv[3]; OUT = 'out/lam_refit_%s.json' % TAG
best = dict(score=-1e9)
def unpack(z):
    p = [math.exp(v) for v in z[:13]]; lam = 1/(1+math.exp(-z[13])); return p, lam
def obj(z):
    p, lam = unpack(z); geom.LAM = lam
    for k, v in zip(B.NAMES, p):
        if not (LO[k] <= v <= HI[k]): return 10.0
    try:
        out = B.evaluate(p)
        if out['primary'] <= -9: return 9.0
        gdb, fpk = bm_gain(p)
    except Exception:
        return 10.0
    mg = min(gdb-GWIN[0], GWIN[1]-gdb)/((GWIN[1]-GWIN[0])/2)
    margins = dict(out['margins']); margins['bmg'] = mg; sc = min(margins.values())
    if sc > best['score']:
        best.update(score=sc, p=p, lam=lam, margins=margins, values=dict(out['values'], bmg=gdb), ratio=out['ratio'])
        json.dump(best, open(OUT, 'w'), indent=1)
        w = sorted(margins.items(), key=lambda kv: kv[1])[:3]
        print('%s score %+.3f lam %.3f bmg %.1f ratio %.3f worst %s' % (time.strftime('%H:%M:%S'), sc, lam, gdb, out['ratio'], [(k, round(v, 2)) for k, v in w]), flush=True)
    return -sc
d3p = json.load(open('best_d3.json'))['params']
lam0 = float(sys.argv[1])
z0 = [math.log(d3p[k]) for k in B.NAMES] + [math.log(lam0/(1-lam0)) if 0 < lam0 < 1 else (6.0 if lam0 >= 1 else -6.0)]
# scale gamma so that gamma/gamma_c matches the stage-1 ratio in the new geometry
geom.LAM = lam0
sec = replace(T.WORLDS['B'].sec, **{k: d3p[k] for k in ('K_t','M_t','R_t','R_c','R_b','K_c','K_p','M_r')}, gamma=0.0)
gc = geom.gamma_critical(sec); z0[8] = math.log(0.903*gc)
import os
if os.path.exists(OUT):     # resume from the saved incumbent
    ck = json.load(open(OUT)); z0 = [math.log(v) for v in ck['p']] + [math.log(ck['lam']/(1-ck['lam']))]
    print('resuming', OUT, ck['score'], flush=True)
print('start lam %.2f gamma_c %.4f' % (lam0, gc), flush=True)
obj(z0)
minimize(obj, z0, method='Nelder-Mead', options=dict(maxfev=int(sys.argv[2]), xatol=1e-3, fatol=1e-4))
print('done', best['score'], best.get('lam'), flush=True)
