"""Run 4 copy of run3/code/c6_crit.py: the state is read from $C6CK (default cat_state.COMBO); scored at T = 0.9501 (g = 1) gives the state itself.
Run 3 (C6): the six-observation metrics along the distance-from-criticality scan of run-2 item B,
scored exactly as stage2/combo.py scores a state.
Environment: combo.py's own (geom.LAM = lambda of combo_c708, geom.install, geom.terms and
geom.gamma_critical installed in tl_model and deltaT_3dof, 24-point gamma_c(rho) table in exp_ftc1),
with score_base NOT imported. (Under run_cat.py score_base is imported, and it patches E.line,
T.response and AF2.fields to the graded profile; exp_best1.evaluate then scores the graded state instead
of the uniform stage-1 state and does not reproduce combo_c708.json. Stage-2 gates, BM gain and J_ftc are
unaffected; c6_runstate.py computes them under run_cat.py as a cross-check.)
State: combo_c708 with the activity scaled uniformly by g = T/0.955 (T = gamma/gamma_c at the peak, 0.955
adopted), as in run-2 b_crit.py (r2cat.State(gscale=g): gamma(x) = g*(1 - nu)*gamma_c*taper): stage-2
profile gam -> g*gam; uniform stage-1 gamma p[8] -> g*p[8]; BM gain at the scaled p.
Margins: 1 centred, 0 on edge, negative outside. score = min over all 18 margins.
Usage (from model/, plain python3, not through run_cat.py): python3 ../scripts/c6_state.py [T ...]
State: $C6CK (default cat_state.COMBO: combo_c856v.json, or combo_r7b.json with FTC_CRIT=disp); output $C6OUT (default out/c6_scan.json)."""
import sys, os, json, math, time
sys.path.insert(1, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); import cat_state
import numpy as np
from dataclasses import replace
sys.path.insert(0, '.')
CK = json.load(open(os.environ.get('C6CK', cat_state.COMBO)))
import geom, deltaT_3dof as d3, tl_model as T
geom.LAM = float(CK['lam']); geom.install()
T.section_terms = lambda sec, w, a, r=None, gamma=None: geom.terms(sec, w, a, r, gamma)
d3.gamma_critical = lambda sec, hi=200.0, tol=1e-6: geom.gamma_critical(sec, 1.0, hi, tol)
import exp_best1 as B
B.P_ORD = 4.0; B.WIN['fwd'] = (0.0, 12.5)
from bmgain_refit import GWIN, bm_gain
import exp_ftc1 as F1, exp_ftc2 as X
SECK = ('K_t', 'M_t', 'R_t', 'R_c', 'R_b', 'K_c', 'K_p', 'M_r')
def gc_dense(rho):                    # combo.py's 24-point gamma_c table, verbatim
    rg = np.geomspace(max(rho.min()*0.98, 1e-3), rho.max()*1.02, 24); gg = []
    for r_ in rg:
        S = F1.SEC; s2 = replace(S, gamma=0.0, R_b=S.R_b*r_, R_r=S.R_r*r_, R_t=S.R_t*r_, R_p=S.R_p*r_, R_c=S.R_c*r_)
        gg.append(d3.gamma_critical(s2))
    return np.interp(np.log(rho), np.log(rg), np.asarray(gg))
F1.gc_dense = gc_dense
assert 'score_base' not in sys.modules, 'score_base must not be imported'
p0, z, lam = list(CK['p']), list(CK['z']), float(CK['lam'])
sec = replace(T.WORLDS['B'].sec, **{k: v for k, v in zip(B.NAMES, p0) if k in SECK})
F1.SEC = X.SEC = sec; F1.COCH = X.COCH = replace(T.WORLDS['B'], sec=sec)
OUT = os.environ.get('C6OUT', 'out/c6_scan.json')
res = json.load(open(OUT)) if os.path.exists(OUT) else dict(source=os.environ.get('C6CK', cat_state.COMBO), lam=lam, rows=[])
done = {round(r['target'], 3) for r in res['rows']}
S2K = ('s2_tt', 's2_sl', 's2_mp', 's2_ph', 's2_fg', 's2_exc')
TS = [float(a) for a in sys.argv[1:]] or [0.9501, 0.93, 0.90, 0.88, 0.85]
def clean(v):
    if isinstance(v, float) and not math.isfinite(v): return str(v)
    if isinstance(v, list): return [clean(t) for t in v]
    return v
for Tt in TS:
    if round(Tt, 3) in done: continue
    g = Tt/0.9501; t0 = time.time(); print('== T %.3f gscale %.6f' % (Tt, g), flush=True)
    geom.LAM = lam
    p = list(p0); p[8] = p0[8]*g
    out = B.evaluate(p)
    gdb, fpk = bm_gain(p)
    r, gam0 = X.profiles(z); gam = g*gam0
    m45 = X.obs45(r, gam)
    mm = F1.ftc_metrics(r, gam, math.exp(z[9]), math.exp(z[6]), math.exp(z[7]), z[8], z[10])
    Jf = float(X.objective_value(mm))
    cert = bool(F1.certify(r, gam))
    mg = dict(out['margins']); mg['bmg'] = min(gdb - GWIN[0], GWIN[1] - gdb)/((GWIN[1] - GWIN[0])/2)
    for k, m in zip(S2K, X.margins(m45)): mg[k] = float(m)
    sc = min(mg.values()); worst = sorted(mg.items(), key=lambda kv: kv[1])[:3]
    vals = {k: clean(v) for k, v in out['values'].items()}
    vals['bmg'] = gdb; vals['bmg_f_re_cf'] = fpk
    for k, v in zip(S2K, m45): vals[k] = float(v)
    row = dict(target=Tt, gscale=g, stage1_ratio=float(out['ratio']), cert=cert, score=float(sc), J_ftc=Jf,
               obj=-min(sc, 0.10) + 0.002*Jf, margins={k: float(v) for k, v in mg.items()}, values=vals,
               worst=[(k, float(v)) for k, v in worst], secs=time.time() - t0)
    res['rows'].append(row); json.dump(res, open(OUT, 'w'), indent=1)
    print('   score %+.4f J_ftc %.3f cert %s stage1 ratio %.4f worst %s (%.0f s)' % (sc, Jf, cert, out['ratio'],
          [(k, round(v, 3)) for k, v in worst], time.time() - t0), flush=True)
    print('   margins', {k: round(v, 3) for k, v in mg.items()}, flush=True)
print('C6 DONE', flush=True)
