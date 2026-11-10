"""Run 7 item 1 sensitivity: point-source excursion of c708 with the stapes boundary of the
distributed source (outgoing-wave row P_1 - (1 + j k0 h) P_0 = 0, k0 = sqrt(-q_0), Re k0 >= 0,
P_N = 0; exp_af_source2.forced_rad) instead of P(0) = 0 (exp_obs46.source_solve).
The stapes-driven solve (generator and f_dp peak places) is unchanged (P(0) = 1). Emitted signal
U(0) ~ (P_1 - P_0)/h as coded; with the outgoing row this equals j k0 P_0, so |P(0)| gives the
same excursion (both printed). Computed on both lines: the stage-1 exc (exp_best1.evaluate) and
the stage-2 s2_exc (fixed exp_ftc2.obs45, profiled line).
Environment: combo.py's own (as check_fix.py / c6_crit.py). Usage (from model/, plain python3): ps_bc.py earlier/combo_c708.json out/ps_bc_c708.json"""
import sys, json, math
import numpy as np
sys.path.insert(0, '.')
from dataclasses import replace
from scipy.linalg import solve_banded
import geom, deltaT_3dof as d3, tl_model as T
st = json.load(open(sys.argv[1]))
geom.LAM = st['lam']; geom.install()
T.section_terms = lambda sec, w, a, r=None, gamma=None: geom.terms(sec, w, a, r, gamma)
d3.gamma_critical = lambda sec, hi=200.0, tol=1e-6: geom.gamma_critical(sec, 1.0, hi, tol)
import exp_best1 as B
B.P_ORD = 4.0; B.WIN['fwd'] = (0.0, 12.5)
import exp_ftc1 as F1, exp_ftc2 as X, exp_obs46 as O
def gc_dense(rho):
    rg = np.geomspace(max(rho.min()*0.98, 1e-3), rho.max()*1.02, 24); gg = []
    for r_ in rg:
        S = F1.SEC; s2 = replace(S, gamma=0.0, R_b=S.R_b*r_, R_r=S.R_r*r_, R_t=S.R_t*r_, R_p=S.R_p*r_, R_c=S.R_c*r_)
        gg.append(d3.gamma_critical(s2))
    return np.interp(np.log(rho), np.log(rg), np.asarray(gg))
F1.gc_dense = gc_dense
SECK = ('K_t','M_t','R_t','R_c','R_b','K_c','K_p','M_r')
p = list(st['p'])
src_orig, ust_orig = O.source_solve, O.u_stapes
MODE = dict(m='orig')
def source_out(sy, m, S=1.0):
    N, h = sy['N'], sy['h']
    q = -(sy['freq'] ** 2) * O_F[0] * sy['Y']
    k0 = np.sqrt(-q[0] + 0j)
    if k0.real < 0: k0 = -k0
    ab = np.zeros((3, N), dtype=complex); rhs = np.zeros(N, dtype=complex)
    ab[1, 0] = -(1.0 + 1j * k0 * h); ab[0, 1] = 1.0
    ab[1, 1:] = -(2.0 + (h * h) * q[1:N]); ab[0, 2:] = 1.0; ab[2, :-1] = 1.0
    rhs[m] = -h * S
    P = np.zeros(N + 1, dtype=complex); P[:N] = solve_banded((1, 1), ab, rhs)
    return P
O_F = [None]
def src(sy, m, S=1.0):
    return src_orig(sy, m, S) if MODE['m'] == 'orig' else source_out(sy, m, S)
def ust(P, h):
    return abs(P[0]) if MODE['m'] == 'outP0' else ust_orig(P, h)
O.source_solve = src; O.u_stapes = ust
res = dict(state=sys.argv[1], lam=st['lam'], rows=[])
for mode in ('orig', 'out', 'outP0'):
    MODE['m'] = mode
    geom.LAM = st['lam']
    O_F[0] = T.WORLDS['B'].F
    out = B.evaluate(p)
    sec = replace(T.WORLDS['B'].sec, **{k: v for k, v in zip(B.NAMES, p) if k in SECK})
    coch = replace(T.WORLDS['B'], sec=sec); F1.SEC = X.SEC = sec; F1.COCH = X.COCH = coch
    O_F[0] = coch.F
    r, g = X.profiles(st['z']); m45 = X.obs45(r, g)
    mg = list(X.margins(m45))
    row = dict(mode=mode, exc_stage1=float(out['values']['exc']), exc_margin=float(out['margins']['exc']),
               s2_exc=float(m45[5]), s2_exc_margin=float(mg[5]))
    res['rows'].append(row); print(row, flush=True)
json.dump(res, open(sys.argv[2], 'w'), indent=1)
