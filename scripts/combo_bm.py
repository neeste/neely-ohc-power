"""Run 4 item 1: combo.py (stage2/combo.py) with the force split lambda FIXED (not free).
Same objective (-min(score, 0.10) + 0.002*J_ftc), same free parameters except lambda (section 8 +
stage-1 gamma + 7 profile parameters, 16 in all), same stop rules as combo_watch.py (nfev >= 1500;
score >= 0.10 and no improvement for 150; no improvement for 300), applied inside the objective.
gamma_c(rho) rebuilt for the fixed lambda with combo.py's 24-point table (F1.gc_dense; geom.LAM fixed).
Start: the c708 section, stage-1 parameters and profiles (earlier/combo_c708.json), with the uniform
stage-1 gamma p[8] scaled by gc(lam)/gc(lam_c708) of the c708 section, so gamma/gamma_c at the peak
equals the c708 value (0.9566); the stage-2 profile is gam = (1 - nu)*gc_dense(r/A)*TAPER, so its
gamma/gamma_c is unchanged by construction.
One deviation forced by the construction: combo.py's box for gamma (0.01, 0.30) is in absolute units;
at lambda 0 and 0.10 gamma_c is about 2.03 times the c708 value, so the start (0.320) lies outside it.
The gamma box is scaled by the same factor gc(lam)/gc(lam_c708) (same box in units of gamma/gamma_c).
Every new incumbent is certified (exp_ftc1.certify), as in combo.py.
Usage (from model/, plain python3): python3 ../scripts/combo_bm.py lam tag   -> out/combo_<tag>.json"""
import sys, os, json, math, time
import numpy as np
from dataclasses import replace
from scipy.optimize import minimize
sys.path.insert(0, '.')
import geom, deltaT_3dof as d3, tl_model as T
LAMF = float(sys.argv[1]); geom.LAM = LAMF; geom.install()
T.section_terms = lambda sec, w, a, r=None, gamma=None: geom.terms(sec, w, a, r, gamma)
d3.gamma_critical = lambda sec, hi=200.0, tol=1e-6: geom.gamma_critical(sec, 1.0, hi, tol)
import exp_best1 as B
B.P_ORD = 4.0; B.WIN['fwd'] = (0.0, 12.5)
from bmgain_refit import LO, HI, GWIN, bm_gain
import exp_ftc1 as F1, exp_ftc2 as X
assert 'score_base' not in sys.modules
SECK = ('K_t', 'M_t', 'R_t', 'R_c', 'R_b', 'K_c', 'K_p', 'M_r')
PZ = (0, 1, 2, 3, 4, 5, 11)
TAG = sys.argv[2]; OUT = 'out/combo_%s.json' % TAG
CNT = OUT.replace('.json', '_nfev.json'); LOG = OUT.replace('.json', '.log')
CK = json.load(open('earlier/combo_c708.json'))
p_fix = list(CK['p']); zprof = list(CK['z'])
s708 = replace(T.WORLDS['B'].sec, **{k: v for k, v in zip(B.NAMES, p_fix) if k in SECK})
geom.LAM = float(CK['lam']); gc_old = geom.gamma_critical(s708, 1.0)
geom.LAM = LAMF; gc_new = geom.gamma_critical(s708, 1.0)
FAC = gc_new/gc_old
LO = dict(LO); HI = dict(HI); LO['gamma'] *= FAC; HI['gamma'] *= FAC
p_fix[8] = CK['p'][8]*FAC

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
    for i in range(9): p[i] = math.exp(v[i])
    z = list(zprof)
    for j, i in enumerate(PZ): z[i] = v[9+j]
    return p, z

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
    p, z = unpack(v); geom.LAM = LAMF
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
        best.update(obj=J, score=sc, J_ftc=Jf, lam=LAMF, p=p, z=z, margins=mg, values=dict(out['values'], bmg=gdb),
                    obs45=[float(t) for t in m45], ratio=out['ratio'], v=list(map(float, v)), nfev=cnt['nfev']+1,
                    certified=True, gamma_fac=FAC)
        json.dump(best, open(OUT, 'w'), indent=1, default=str)
        w = sorted(mg.items(), key=lambda kv: kv[1])[:3]
        say('%s n %d obj %.4f score %+.3f Jftc %.2f lam %.3f bmg %.1f ratio %.4f worst %s' % (time.strftime('%H:%M:%S'),
            cnt['nfev']+1, J, sc, Jf, LAMF, gdb, out['ratio'], [(k, round(t, 3)) for k, t in w]))
    return J

if 'v' in best:
    v0 = best['v']; cnt['restarts'] = cnt.get('restarts', 0) + 1; say('resuming from incumbent, nfev %d' % cnt['nfev'])
else:
    v0 = [math.log(p_fix[i]) for i in range(9)] + [zprof[i] for i in PZ]
    say('start lam %.3f gamma %.6f (fac %.5f, gc %.5f -> %.5f), gamma box %.4f..%.4f' % (LAMF, p_fix[8], FAC, gc_old, gc_new, LO['gamma'], HI['gamma']))
if cnt.get('stopped'):
    say('already stopped: %s' % cnt['stopped']); sys.exit(0)
sim = [np.array(v0)]
for i in range(len(v0)):
    vi = np.array(v0); vi[i] += 0.08; sim.append(vi)
try:
    minimize(obj, v0, method='Nelder-Mead', options=dict(maxfev=max(1, 1500 - cnt['nfev']), xatol=1e-4, fatol=1e-5, initial_simplex=np.array(sim)))
    cnt['stopped'] = cnt.get('stopped') or 'nelder-mead converged'; json.dump(cnt, open(CNT, 'w'))
except Stop as e:
    pass
say('STOPPED (%s) at n %d; best score %s obj %s' % (cnt.get('stopped'), cnt['nfev'], best.get('score'), best.get('obj')))
