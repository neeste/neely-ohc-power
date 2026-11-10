"""Run 7 item 2: stage2/combo.py with the corrected obs45 (fixed exp_ftc2.py: the point-source
excursion s2_exc is computed on the profiled line). Everything else as combo.py: 17 free parameters
(section 8 + stage-1 gamma + lambda via logit + 7 place-profile parameters PZ), the same box,
objective -min(score, 0.10) + 0.002*J_ftc, certification (exp_ftc1.certify) of every new incumbent,
Nelder-Mead with combo.py's initial simplex (+0.08 in logs, +0.3 in logit lambda).
Fixed parts (p[9:] and the non-PZ profile parameters) are taken from the start state; for c708 and c85
they are identical to combo.py's best_d3.json and ftc2_lam_ckpt.json (checked 2026-11-07).
Stop rules (combo_watch.py, applied inside the objective as in run-4 combo_bm.py): nfev >= 1500;
score >= 0.10 and no improvement for 150; no improvement for 300.
The start state is scored afresh with the corrected metric (the stored c708/c85 objective is not used).
Resumption (container restarts): from the incumbent with nfev, last_improve and best_obj carried;
the simplex is rebuilt around the incumbent.
Usage (from model/, plain python3): python3 ../scripts/combo7.py START_JSON TAG  -> out/combo_<tag>.json"""
import sys, os, json, math, time
import numpy as np
from dataclasses import replace
from scipy.optimize import minimize
sys.path.insert(0, '.')
START = sys.argv[1]; TAG = sys.argv[2]
SJ = json.load(open(START))
import geom, deltaT_3dof as d3, tl_model as T
geom.LAM = float(SJ['lam']); geom.install()
T.section_terms = lambda sec, w, a, r=None, gamma=None: geom.terms(sec, w, a, r, gamma)
d3.gamma_critical = lambda sec, hi=200.0, tol=1e-6: geom.gamma_critical(sec, 1.0, hi, tol)
import exp_best1 as B
B.P_ORD = 4.0; B.WIN['fwd'] = (0.0, 12.5)
from bmgain_refit import LO, HI, GWIN, bm_gain
import exp_ftc1 as F1, exp_ftc2 as X
assert 'score_base' not in sys.modules
assert 'sys_orig' in open(X.__file__).read(), 'fixed exp_ftc2.py not installed'
SECK = ('K_t', 'M_t', 'R_t', 'R_c', 'R_b', 'K_c', 'K_p', 'M_r')
PZ = (0, 1, 2, 3, 4, 5, 11)
OUT = 'out/combo_%s.json' % TAG
CNT = OUT.replace('.json', '_nfev.json'); LOG = OUT.replace('.json', '.log')
p_fix = list(SJ['p']); zprof = list(SJ['z'])

def gc_dense(rho):                    # combo.py's 24-point table, verbatim
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

class Stop(Exception): pass
best = json.load(open(OUT)) if os.path.exists(OUT) else dict(obj=1e9)
cnt = json.load(open(CNT)) if os.path.exists(CNT) else dict(nfev=0, last_improve=0, best_obj=1e9, restarts=0)
def say(s):
    print(s, flush=True)
    with open(LOG, 'a') as f: f.write(s + '\n')

def obj(v):
    J = _obj(v); cnt['nfev'] += 1
    if J < cnt['best_obj'] - 1e-9: cnt.update(best_obj=J, last_improve=cnt['nfev'])
    stall = cnt['nfev'] - cnt['last_improve']; why = None
    if cnt['nfev'] >= 1500: why = 'nfev 1500'
    elif best.get('score', -9) >= 0.10 and stall >= 150: why = 'score>=0.10 and stalled 150'
    elif stall >= 300: why = 'no improvement for 300'
    if why: cnt['stopped'] = why; cnt['stopped_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    json.dump(cnt, open(CNT, 'w'))
    if why: raise Stop(why)
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
            say('%s n %d candidate rejected by certificate (obj %.4f score %+.3f)' % (time.strftime('%H:%M:%S'), cnt['nfev']+1, J, sc))
            return J + 1.0
        best.update(obj=J, score=sc, J_ftc=Jf, lam=lam, p=p, z=z, margins={k: float(t) for k, t in mg.items()},
                    values=dict(out['values'], bmg=gdb), obs45=[float(t) for t in m45], ratio=out['ratio'],
                    v=list(map(float, v)), nfev=cnt['nfev']+1, certified=True, start=START)
        json.dump(best, open(OUT, 'w'), indent=1, default=str)
        w = sorted(mg.items(), key=lambda kv: kv[1])[:3]
        say('%s n %d obj %.4f score %+.3f Jftc %.2f lam %.4f bmg %.1f ratio %.4f exc2 %.2f worst %s' % (time.strftime('%H:%M:%S'),
            cnt['nfev']+1, J, sc, Jf, lam, gdb, out['ratio'], m45[5], [(k, round(t, 3)) for k, t in w]))
    return J

if cnt.get('stopped'):
    say('already stopped: %s' % cnt['stopped']); sys.exit(0)
if 'v' in best:
    v0 = best['v']; cnt['restarts'] = cnt.get('restarts', 0) + 1; say('resuming from incumbent, nfev %d' % cnt['nfev'])
else:
    v0 = list(SJ['v']); say('start %s lam %.5f (stored obj %.4f score %.4f, old metric)' % (START, SJ['lam'], SJ['obj'], SJ['score']))
# Nelder-Mead's own convergence test (xatol 1e-4, fatol 1e-5) is not one of the brief's stop rules:
# on convergence the simplex is rebuilt around the incumbent (as on a container restart) and the
# search continues until a stop rule fires (added 2026-11-07 14:20Z; counted in cnt['nm_converged']).
try:
    while True:
        sim = [np.array(v0)]
        for i in range(len(v0)):
            vi = np.array(v0); vi[i] += 0.08 if i != 9 else 0.3; sim.append(vi)
        minimize(obj, v0, method='Nelder-Mead', options=dict(maxfev=max(1, 1500 - cnt['nfev']), xatol=1e-4, fatol=1e-5, initial_simplex=np.array(sim)))
        cnt['nm_converged'] = cnt.get('nm_converged', 0) + 1; json.dump(cnt, open(CNT, 'w'))
        v0 = best['v']; say('%s nelder-mead converged at n %d; simplex rebuilt around the incumbent' % (time.strftime('%H:%M:%S'), cnt['nfev']))
except Stop as e:
    pass
say('STOPPED (%s) at n %d; best score %s obj %s' % (cnt.get('stopped'), cnt['nfev'], best.get('score'), best.get('obj')))
