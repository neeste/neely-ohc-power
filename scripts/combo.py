"""Cat combined polish: section (8) + stage-1 gamma + force split lambda + 7 place-profile
parameters, fitted together against the stage-1 metrics, the BM-gain window and the
stage-2 observation gates. Objective (minimized): -min(score, 0.10) + 0.002*J_ftc, where
score = min over all normalized margins (stage-1 eleven + bmg, stage-2 six; 1 centred,
0 on edge). Feasibility first, robustness to 0.10, then tuning-curve population fit.
Every incumbent with score >= best is certified by argument-principle winding.
Usage: combo.py lam0 maxfev tag    (checkpoint out/combo_<tag>.json, resumes; run from model/)"""
import sys, os, json, math, time
import numpy as np
from dataclasses import replace
from scipy.optimize import minimize
sys.path.insert(0, '.')
import geom, deltaT_3dof as d3, tl_model as T
geom.LAM = float(sys.argv[1]); geom.install()
T.section_terms = lambda sec, w, a, r=None, gamma=None: geom.terms(sec, w, a, r, gamma)
d3.gamma_critical = lambda sec, hi=200.0, tol=1e-6: geom.gamma_critical(sec, 1.0, hi, tol)
import exp_best1 as B
B.P_ORD = 4.0; B.WIN['fwd'] = (0.0, 12.5)
from bmgain_refit import LO, HI, GWIN, bm_gain
import exp_ftc1 as F1, exp_ftc2 as X
SECK = ('K_t', 'M_t', 'R_t', 'R_c', 'R_b', 'K_c', 'K_p', 'M_r')
PZ = (0, 1, 2, 3, 4, 5, 11)          # profile parameters fitted (others fixed)
TAG = sys.argv[3]; OUT = 'out/combo_%s.json' % TAG
bd = json.load(open('best_d3.json'))['params']
p_fix = [bd[k] for k in B.NAMES]
zprof = list(json.load(open('earlier/ftc2_lam_ckpt.json'))['z'])

def gc_dense(rho):                    # 24-point version of F1.gc_dense for the current SEC
    rg = np.geomspace(max(rho.min()*0.98, 1e-3), rho.max()*1.02, 24); gg = []
    for r_ in rg:
        S = F1.SEC; s2 = replace(S, gamma=0.0, R_b=S.R_b*r_, R_r=S.R_r*r_, R_t=S.R_t*r_, R_p=S.R_p*r_, R_c=S.R_c*r_)
        gg.append(d3.gamma_critical(s2))
    return np.interp(np.log(rho), np.log(rg), np.asarray(gg))
F1.gc_dense = gc_dense

def set_section(p):
    sec = replace(T.WORLDS['B'].sec, **{k: v for k, v in zip(B.NAMES, p) if k in SECK})
    coch = replace(T.WORLDS['B'], sec=sec)
    F1.SEC = X.SEC = sec; F1.COCH = X.COCH = coch

def unpack(v):
    p = list(p_fix)
    for i, k in enumerate(B.NAMES[:9]): p[i] = math.exp(v[i])
    lam = 1/(1+math.exp(-v[9])); z = list(zprof)
    for j, i in enumerate(PZ): z[i] = v[10+j]
    return p, lam, z

best = dict(obj=1e9)
CNT = OUT.replace('.json', '_nfev.json')      # cumulative evaluation counter (added 2026-11-05 for stop rules)
cnt = json.load(open(CNT)) if os.path.exists(CNT) else dict(nfev=0, last_improve=0, best_obj=1e9)
def obj(v):
    J = _obj(v); cnt['nfev'] += 1
    if J < cnt['best_obj'] - 1e-9: cnt.update(best_obj=J, last_improve=cnt['nfev'])
    json.dump(cnt, open(CNT, 'w'))
    return J
def _obj(v):
    p, lam, z = unpack(v); geom.LAM = lam
    for k, val in zip(B.NAMES, p):
        if not (LO[k] <= val <= HI[k]): return 10.0
    if not (-0.6 < z[0] < 0.6 and -1.5 < z[1] < 0.6 and 0.08 < z[4] < 1.2 and 0.02 < math.exp(z[5]) < 0.30
            and -1.2 < z[11] < 0.6 and 0.3 < math.exp(z[3]) < 20.0): return 10.0
    try:
        out = B.evaluate(p)
        if out['primary'] <= -9: return 9.0
        gdb, _ = bm_gain(p)
        set_section(p)
        r, gam = X.profiles(z)
        m45 = X.obs45(r, gam)
        mm = F1.ftc_metrics(r, gam, math.exp(z[9]), math.exp(z[6]), math.exp(z[7]), z[8], z[10])
        Jf = X.objective_value(mm)
    except Exception as e:
        return 9.5
    mg = dict(out['margins']); mg['bmg'] = min(gdb-GWIN[0], GWIN[1]-gdb)/((GWIN[1]-GWIN[0])/2)
    for k, m in zip(('s2_tt', 's2_sl', 's2_mp', 's2_ph', 's2_fg', 's2_exc'), X.margins(m45)): mg[k] = m
    sc = min(mg.values()); J = -min(sc, 0.10) + 0.002*Jf
    if J < best['obj']:
        if not F1.certify(r, gam):
            print(time.strftime('%H:%M:%S'), 'candidate rejected by certificate', flush=True); return J + 1.0
        best.update(obj=J, score=sc, J_ftc=Jf, lam=lam, p=p, z=z, margins=mg, values=dict(out['values'], bmg=gdb),
                    obs45=[float(t) for t in m45], ratio=out['ratio'], v=list(map(float, v)))
        json.dump(best, open(OUT, 'w'), indent=1)
        w = sorted(mg.items(), key=lambda kv: kv[1])[:3]
        print('%s n %d obj %.4f score %+.3f Jftc %.2f lam %.3f bmg %.1f ratio %.3f worst %s' % (time.strftime('%H:%M:%S'), cnt['nfev']+1, J, sc, Jf, lam, gdb,
              out['ratio'], [(k, round(t, 3)) for k, t in w]), flush=True)
    return J

if os.path.exists(OUT):
    v0 = json.load(open(OUT))['v']; print('resuming', OUT, flush=True)
else:
    lam0 = float(sys.argv[1]); lam_ref = json.load(open('best_d3.json')).get('lam', 1.0)
    if abs(lam0 - lam_ref) > 1e-3:    # keep the same distance from critical at the new lambda
        s0 = replace(T.WORLDS['B'].sec, **{k: bd[k] for k in SECK})
        geom.LAM = lam_ref; g_ref = geom.gamma_critical(s0, 1.0)
        geom.LAM = lam0; g_new = geom.gamma_critical(s0, 1.0)
        p_fix[8] = bd['gamma']*g_new/g_ref; print('gamma rescaled to', p_fix[8], flush=True)
    v0 =[math.log(p_fix[i]) for i in range(9)] + [math.log(lam0/(1-lam0))] + [zprof[i] for i in PZ]
sim = [np.array(v0)]
for i in range(len(v0)):
    vi = np.array(v0); vi[i] += 0.08 if i != 9 else 0.3; sim.append(vi)
minimize(obj, v0, method='Nelder-Mead', options=dict(maxfev=max(1, int(sys.argv[2]) - cnt['nfev']), xatol=1e-4, fatol=1e-5, initial_simplex=np.array(sim)))
print('done', best.get('score'), best.get('lam'), flush=True)
