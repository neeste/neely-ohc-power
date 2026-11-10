"""Run 10 items 1-2: velocity versus displacement criterion for the cat population read-out at c856.
Environment: combo.py's own (as run3/run4 c6_state.py): geom.LAM = lambda of c856, geom.install,
geom.terms / geom.gamma_critical installed, 24-point gamma_c(rho) table in exp_ftc1; score_base not imported.
The mechanics are fixed (section, lambda, gamma, place profiles from z, base B = exp(z[9])), so the hair-bundle
shear |xi_hb| at the nine PLACES on the 60-point exp_ftc1 frequency grid is computed once and cached; the
read-out (criterion, C0, f_hp, f_lp, n_lp) is then applied with exp_ftc1.ftc_metrics' own arithmetic.
Check: the cached arithmetic reproduces F1.ftc_metrics (both criteria) to rounding.
Fits: Nelder-Mead over (C0, log f_hp, log f_lp, n_lp) in the stage-2 bounds, xatol = fatol = 1e-4.
Usage (from model/, plain python3): python3 ../scripts/vel_fit.py out/vel_fit.json  (C6CK = start state, default combo_r7b.json)"""
import sys, os, json, math, time
sys.path.insert(1, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); import cat_state
import numpy as np
from dataclasses import replace
from scipy.optimize import minimize
sys.path.insert(0, '.')
CK = json.load(open(os.environ.get('C6CK', 'combo_r7b.json')))   # start: the published (displacement) read-out
import geom, deltaT_3dof as d3, tl_model as T
geom.LAM = float(CK['lam']); geom.install()
T.section_terms = lambda sec, w, a, r=None, gamma=None: geom.terms(sec, w, a, r, gamma)
d3.gamma_critical = lambda sec, hi=200.0, tol=1e-6: geom.gamma_critical(sec, 1.0, hi, tol)
import exp_best1 as B
B.P_ORD = 4.0; B.WIN['fwd'] = (0.0, 12.5)
import exp_ftc1 as F1, exp_ftc2 as X, exp_accumulation as E
SECK = ('K_t', 'M_t', 'R_t', 'R_c', 'R_b', 'K_c', 'K_p', 'M_r')
def gc_dense(rho):
    rg = np.geomspace(max(rho.min()*0.98, 1e-3), rho.max()*1.02, 24); gg = []
    for r_ in rg:
        S = F1.SEC; s2 = replace(S, gamma=0.0, R_b=S.R_b*r_, R_r=S.R_r*r_, R_t=S.R_t*r_, R_p=S.R_p*r_, R_c=S.R_c*r_)
        gg.append(d3.gamma_critical(s2))
    return np.interp(np.log(rho), np.log(rg), np.asarray(gg))
F1.gc_dense = gc_dense
assert 'score_base' not in sys.modules
p0, z = list(CK['p']), list(CK['z'])
sec = replace(T.WORLDS['B'].sec, **{k: v for k, v in zip(B.NAMES, p0) if k in SECK})
F1.SEC = X.SEC = sec; F1.COCH = X.COCH = replace(T.WORLDS['B'], sec=sec)
SEC, COCH, A, N, XX = F1.SEC, F1.COCH, F1.A, F1.N, F1.X
r, gam = X.profiles(z)
base = math.exp(z[9]); OLD = dict(C0=z[10], f_hp=math.exp(z[6]), f_lp=math.exp(z[7]), n_lp=z[8])
OUT = sys.argv[1]
t0 = time.time()
CACHE = []
for x0 in F1.PLACES:
    j = int(np.argmin(np.abs(XX - x0))); cfp = float(A[j])
    fr = cfp*np.logspace(math.log10(0.10), math.log10(1.1), 60); resp = []
    for f in fr:
        Y, Y_r, H_o = E.terms(SEC, float(f), A, r, gam)
        P = T._fd(COCH, float(f), Y, N, 1.0)
        resp.append(abs((H_o*Y*P)[j]))
    CACHE.append((fr, np.array(resp)))
print('cache built %.0f s' % (time.time() - t0), flush=True)

def metrics(crit, C0, f_hp, f_lp, n_lp):
    nlp = n_lp
    out = []
    for fr, resp in CACHE:
        th = C0 - 20*np.log10(resp*F1.crit_factor(fr*base, crit)) + F1.me_db(fr*base, f_hp, f_lp, nlp)
        i0 = int(np.argmin(th)); ftip = fr[i0]*base; tip = th[i0]; rel = fr/fr[i0]; t10 = tip + 10.0
        lo = hi = None
        for i in range(i0, 0, -1):
            if th[i-1] >= t10 >= th[i]:
                lo = np.interp(t10, [th[i], th[i-1]], [rel[i], rel[i-1]]); break
        for i in range(i0, len(rel)-1):
            if th[i] <= t10 <= th[i+1]:
                hi = np.interp(t10, [th[i], th[i+1]], [rel[i], rel[i+1]]); break
        q10 = 1.0/(hi-lo) if (lo and hi) else np.nan
        tt = np.interp(0.5, rel, th) - tip if rel[0] <= 0.5 else np.nan
        out.append((ftip, tip, q10, tt))
    return np.array(out)

def J(crit, C0, f_hp, f_lp, n_lp): return float(X.objective_value(metrics(crit, C0, f_hp, f_lp, n_lp)))

def C0_closed(crit, f_hp, f_lp, nlp, C0ref=0.0):
    """J depends on C0 only through the tip terms, and the covered bins do not depend on C0: shift = -mean(model - data)."""
    mm = metrics(crit, C0ref, f_hp, f_lp, nlp); idx = np.argsort(mm[:, 0]); mm = mm[idx]; mf = mm[:, 0]
    d = [np.interp(ft, mf, mm[:, 1]) - tip for ft, tip, q, tt in F1.TG if mf.min() <= ft <= mf.max()]
    return C0ref - float(np.mean(d))

res = dict(state='c856 (rev/run7/out/combo_r7b.json)', lam=CK['lam'], base_khz=base, old_readout=OLD,
           velocity_convention='omega |xi_hb| with omega/omega_ref = f/(1 kHz); C0_rad_s = C0 + 20 log10(2 pi 1000) for omega in rad/s')
# --- check the cached arithmetic against F1.ftc_metrics ---
for crit in ('disp', 'vel'):
    m_ref = F1.ftc_metrics(r, gam, base, OLD['f_hp'], OLD['f_lp'], OLD['n_lp'], OLD['C0'], crit=crit)
    m_c = metrics(crit, OLD['C0'], OLD['f_hp'], OLD['f_lp'], OLD['n_lp'])
    dmax = float(np.nanmax(np.abs(m_ref - m_c)))
    res['check_' + crit] = dict(max_abs_diff=dmax, J_ref=float(X.objective_value(m_ref)), J_cache=float(X.objective_value(m_c)))
    print('check %s: max |diff| %.2e  J ftc_metrics %.4f  cached %.4f' % (crit, dmax, res['check_' + crit]['J_ref'], res['check_' + crit]['J_cache']), flush=True)
J_disp_old = J('disp', **OLD); res['J_disp_old'] = J_disp_old
print('J_ftc displacement, old read-out %.4f (combo_r7b stored %.4f)' % (J_disp_old, CK['J_ftc']), flush=True)
# C0 closed form, displacement (check: should be near the old C0 if C0 was converged)
c0d = C0_closed('disp', OLD['f_hp'], OLD['f_lp'], OLD['n_lp'])
res['disp_C0only'] = dict(C0=c0d, J=J('disp', c0d, OLD['f_hp'], OLD['f_lp'], OLD['n_lp']))
# velocity, old filter
J_vel_old_raw = J('vel', **OLD)
c0v = C0_closed('vel', OLD['f_hp'], OLD['f_lp'], OLD['n_lp'])
res['vel_old_filter_oldC0'] = dict(J=J_vel_old_raw)
res['vel_C0only'] = dict(C0=c0v, f_hp=OLD['f_hp'], f_lp=OLD['f_lp'], n_lp=OLD['n_lp'], J=J('vel', c0v, OLD['f_hp'], OLD['f_lp'], OLD['n_lp']))
print('velocity, old filter: J %.4f with old C0; C0 closed form %.3f -> J %.4f' % (J_vel_old_raw, c0v, res['vel_C0only']['J']), flush=True)
json.dump(res, open(OUT, 'w'), indent=1)

def fit(crit, start, tag):
    nev = [0]; hist = []
    def obj(v):
        C0, lhp, llp, nlp = v; fhp, flp = math.exp(lhp), math.exp(llp)
        if not (0.05 < fhp < 1.5 and 3.0 < flp < 40.0 and 0.3 < nlp < 3.0): return 60.0
        nev[0] += 1; return J(crit, C0, fhp, flp, nlp)
    v0 = np.array([start['C0'], math.log(start['f_hp']), math.log(start['f_lp']), start['n_lp']])
    t1 = time.time(); best = None
    # restart from the optimum until no further improvement (Nelder-Mead may stall on the piecewise grid)
    for k in range(6):
        o = minimize(obj, v0, method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-4, maxfev=20000, adaptive=False))
        hist.append(dict(restart=k, J=float(o.fun), nfev=int(o.nfev), x=[float(t) for t in o.x], msg=str(o.message)))
        print('  %s restart %d: J %.5f nfev %d' % (tag, k, o.fun, o.nfev), flush=True)
        if best is not None and o.fun > best.fun - 1e-5: best = o if o.fun < best.fun else best; break
        best = o if (best is None or o.fun < best.fun) else best; v0 = o.x
    C0, lhp, llp, nlp = best.x
    # final closed-form C0 polish (exact for the tip term)
    c0p = C0_closed(crit, math.exp(lhp), math.exp(llp), nlp)
    Jp = J(crit, c0p, math.exp(lhp), math.exp(llp), nlp)
    if Jp < best.fun: C0 = c0p
    fv = dict(C0=float(C0), f_hp=math.exp(lhp), f_lp=math.exp(llp), n_lp=float(nlp))
    rec = dict(start=start, fit=fv, J=J(crit, **fv), nfev=nev[0], secs=time.time()-t1, restarts=hist,
               at_bound=[k for k, (lo, hi) in dict(f_hp=(0.05, 1.5), f_lp=(3.0, 40.0), n_lp=(0.3, 3.0)).items() if min(fv[k]-lo, hi-fv[k]) < 0.01*(hi-lo)])
    print('%s: J %.4f  C0 %.3f f_hp %.4f f_lp %.3f n_lp %.4f (%d evals, %.0f s)' % (tag, rec['J'], fv['C0'], fv['f_hp'], fv['f_lp'], fv['n_lp'], nev[0], rec['secs']), flush=True)
    return rec

res['vel_fit_startA'] = fit('vel', dict(C0=c0v, f_hp=OLD['f_hp'], f_lp=OLD['f_lp'], n_lp=OLD['n_lp']), 'vel start A (current values, C0 closed form)')
json.dump(res, open(OUT, 'w'), indent=1)
res['vel_fit_startA_rawC0'] = fit('vel', dict(OLD), 'vel start A0 (current values incl. old C0)')
json.dump(res, open(OUT, 'w'), indent=1)
c0b = C0_closed('vel', 0.3, 20.0, 1.0)
res['vel_fit_startB'] = fit('vel', dict(C0=c0b, f_hp=0.3, f_lp=20.0, n_lp=1.0), 'vel start B (0.3, 20, 1)')
json.dump(res, open(OUT, 'w'), indent=1)
# displacement refit of the same four parameters (control: is the old read-out at its optimum?)
res['disp_fit_control'] = fit('disp', dict(OLD), 'disp control refit')
best = min((res[k] for k in ('vel_fit_startA', 'vel_fit_startA_rawC0', 'vel_fit_startB')), key=lambda t: t['J'])
res['vel_adopted'] = dict(best['fit'], J=best['J'])
mm = metrics('vel', **best['fit']); res['vel_adopted']['metrics'] = mm.tolist()
res['disp_old_metrics'] = metrics('disp', **OLD).tolist()
res['targets'] = F1.TG.tolist()
json.dump(res, open(OUT, 'w'), indent=1)
print('ADOPTED velocity read-out: J %.4f' % best['J'], best['fit'], flush=True)
print('VEL_FIT DONE %.0f s' % (time.time() - t0), flush=True)
