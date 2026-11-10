"""Shoulder sensitivity to TM parameters at a stage-2 state: scale K_t, M_t, R_t (one at a time,
x0.5 and x2; also K_t and M_t together, keeping the TM resonance) and report the threshold rise
from the tip to +0.2, +0.5 and +0.8 oct at x = 0.5 (shape only; profiles recomputed so gamma stays
the same fraction of the new critical value).  Usage: shoulder_tm.py ckpt.json out.json"""
import sys, json, math
import numpy as np
from dataclasses import replace
sys.path.insert(0, '.')
import geom, deltaT_3dof as d3, tl_model as T
bd = json.load(open('best_d3.json')); geom.LAM = bd.get('lam', 1.0); geom.install()
T.section_terms = lambda sec, w, a, r=None, gamma=None: geom.terms(sec, w, a, r, gamma)
d3.gamma_critical = lambda sec, hi=200.0, tol=1e-6: geom.gamma_critical(sec, 1.0, hi, tol)
import exp_accumulation as E, exp_ftc1 as F1, exp_ftc2 as X
z = json.load(open(sys.argv[1]))['z']; S0 = F1.SEC; xp = 0.5
j = int(np.argmin(np.abs(F1.X - xp)))
def run(sec):
    F1.SEC = X.SEC = sec; F1.COCH = X.COCH = replace(F1.COCH, sec=sec)
    r, gam = X.profiles(z); A = F1.A
    def thr(f):
        Y, Yr, Ho = E.terms(sec, f, A, r, gam); P = T._fd(F1.COCH, f, Y, F1.N, 1.0)
        return -20*math.log10(abs((Ho*Y*P)[j]))
    fr = A[j]*np.logspace(-0.4, 0.5, 120); th = np.array([thr(f) for f in fr])
    i0 = int(np.argmin(th)); ft = fr[i0]
    return {('+%.1f' % o): float(thr(ft*2**o) - th[i0]) for o in (0.2, 0.5, 0.8)} | dict(tip_rel=float(ft/A[j]))
res = {'reference': run(S0)}
for k in ('K_t', 'M_t', 'R_t'):
    for s in (0.5, 2.0):
        res['%s x%.1f' % (k, s)] = run(replace(S0, **{k: getattr(S0, k)*s}))
for s in (0.5, 2.0):
    res['K_t,M_t x%.1f' % s] = run(replace(S0, K_t=S0.K_t*s, M_t=S0.M_t*s))
json.dump(dict(source=sys.argv[1], lam=geom.LAM, x=xp, rise_db_re_tip=res), open(sys.argv[2], 'w'), indent=1)
for k, v in res.items(): print('%-14s' % k, {a: round(b, 1) for a, b in v.items()})
