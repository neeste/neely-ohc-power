"""Run 4 item 2: one-change ablations at the final cat state c708, scored on all 18 combo metrics plus J_ftc in
combo.py's own environment (as run3 c6_crit.py: lambda of combo_c708, geom.install, 24-point gamma_c table;
score_base not imported) and certified (exp_ftc1.certify on the stage-2 profile).
Variants (one process each):
  ref              c708 itself (check against combo_c708.json)
  Kt4, Kt025       K_t x4 and x1/4 (M_t fixed): TM resonance (K_t+K_c)/M_t ... see the log for the realized shift
  clamp            shear divider removed: TM held fixed (X_t = 0), hair-bundle shear = G X_r. Clean switch: the TM
                   degree of freedom is removed from geom.terms (2x2 BM-RL system; the RL row keeps K_c and R_c to the
                   now-fixed TM) and from geom.matrices/margin (2-dof eigenproblem, also used by d3.stability_margin).
  Mt1e4            cross-check of clamp by parameters only: M_t x 1e4
  pass             OHC force removed: gamma = 0 everywhere (stage-1 p[8] = 0, stage-2 profile 0)
  g098, g102       gamma x0.98 and x1.02 (stage-1 p[8] and stage-2 profile)
For the section changes two conventions: suffix '_g' holds the absolute gamma (stage-1 p[8] and the stage-2 profile
gamma(x) of c708 unchanged); suffix '_r' holds gamma/gamma_c (p[8] scaled by gc_new/gc_old of the section at sb = 1;
stage-2 profile rebuilt as (1 - nu) gc_dense(r/A) taper for the new section).
Usage (from model/): python3 ../scripts/abl.py VARIANT   -> out/abl/VARIANT.json"""
import sys, os, json, math, time
sys.path.insert(1, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); import cat_state
import numpy as np
from dataclasses import replace
sys.path.insert(0, '.')
CK = json.load(open(cat_state.COMBO))
import geom, deltaT_3dof as d3, tl_model as T
VAR = sys.argv[1]
geom.LAM = float(CK['lam']); geom.install()
T.section_terms = lambda sec, w, a, r=None, gamma=None: geom.terms(sec, w, a, r, gamma)
d3.gamma_critical = lambda sec, hi=200.0, tol=1e-6: geom.gamma_critical(sec, 1.0, hi, tol)
import exp_accumulation as E
if VAR.startswith('clamp'):
    def terms_clamp(sec, w, a, r=None, gamma=None, mp=1.0, mc=1.0, mt=1.0, mrp=1.0, mrc=1.0):
        w = np.asarray(w); a = np.asarray(a, float)
        r = a if r is None else np.asarray(r, float)
        g = sec.gamma if gamma is None else np.asarray(gamma, float)
        jw = 1j*w; a2 = a*a
        sb = 1.0 if geom._SB is None else geom._SB(a)
        act = g*sec.K_a*sec.G*a2
        Rp, Rc = sec.R_p*mrp, sec.R_c*mrc
        Kp, Kc = sec.K_p*mp, sec.K_c*mc
        pil = jw*Rp*r + Kp*a2
        rl = geom.LAM
        A00 = -w**2*sec.M_b + jw*(sec.R_b + Rp)*r + (sec.K_b*sb + Kp)*a2
        A01 = -pil + act
        A10 = -pil + 0.0*A00
        A11 = -w**2*sec.M_r + jw*(sec.R_r + Rp + Rc)*r + (Kp + Kc)*a2 - rl*act
        det = A00*A11 - A01*A10
        Y = A11/det; Y_r = -A10/det
        return Y, Y_r, sec.G*Y_r/Y
    def matrices_clamp(sec, sb=1.0):
        act = sec.gamma*sec.K_a*sec.G; rl = geom.LAM
        M = np.diag([sec.M_b, sec.M_r]).astype(float)
        C = np.array([[sec.R_b+sec.R_p, -sec.R_p], [-sec.R_p, sec.R_r+sec.R_p+sec.R_c]])
        K = np.array([[sec.K_b*sb+sec.K_p, -sec.K_p+act], [-sec.K_p, sec.K_p+sec.K_c-rl*act]])
        return M, C, K
    def margin_clamp(sec, sb=1.0):
        M, C, K = matrices_clamp(sec, sb); n = M.shape[0]
        Mi = np.linalg.inv(M); Z = np.zeros((n, n))
        A = np.block([[Z, np.eye(n)], [-Mi@K, -Mi@C]])
        return float(np.max(np.linalg.eigvals(A).real))
    geom.terms = terms_clamp; geom.matrices = matrices_clamp; geom.margin = margin_clamp
    E.terms = terms_clamp; d3.stability_margin = lambda sec: margin_clamp(sec)
import exp_best1 as B
B.P_ORD = 4.0; B.WIN['fwd'] = (0.0, 12.5)
from bmgain_refit import GWIN, bm_gain
import exp_ftc1 as F1, exp_ftc2 as X
assert 'score_base' not in sys.modules
SECK = ('K_t', 'M_t', 'R_t', 'R_c', 'R_b', 'K_c', 'K_p', 'M_r')
def gc_dense(rho):                    # combo.py's 24-point gamma_c table, verbatim
    rg = np.geomspace(max(rho.min()*0.98, 1e-3), rho.max()*1.02, 24); gg = []
    for r_ in rg:
        S = F1.SEC; s2 = replace(S, gamma=0.0, R_b=S.R_b*r_, R_r=S.R_r*r_, R_t=S.R_t*r_, R_p=S.R_p*r_, R_c=S.R_c*r_)
        gg.append(d3.gamma_critical(s2))
    return np.interp(np.log(rho), np.log(rg), np.asarray(gg))
F1.gc_dense = gc_dense
p0, z = list(CK['p']), list(CK['z'])
def set_sec(p):
    sec = replace(T.WORLDS['B'].sec, **{k: v for k, v in zip(B.NAMES, p) if k in SECK})
    F1.SEC = X.SEC = sec; F1.COCH = X.COCH = replace(T.WORLDS['B'], sec=sec); return sec
# c708 stage-2 profile with the c708 section (computed before any change; for clamp the gc used here is the clamped
# one, so clamp_g uses the unclamped profile computed in a separate reference: see below)
base = VAR.split('_')[0]; mode = VAR.split('_')[1] if '_' in VAR else None
sec0 = set_sec(p0)
p = list(p0); idx = {k: i for i, k in enumerate(B.NAMES)}
if base == 'Kt4': p[idx['K_t']] *= 4.0
elif base == 'Kt025': p[idx['K_t']] *= 0.25
elif base == 'Mt1e4': p[idx['M_t']] *= 1e4
sec1 = replace(sec0, **{k: v for k, v in zip(B.NAMES, p) if k in SECK})
# gamma_c of the section at sb = 1 (stage-1 convention), unchanged and changed
gc_new = geom.gamma_critical(sec1, 1.0)
if VAR.startswith('clamp'):
    # reference values with the full 3-dof model, for the '_g' convention and the gc ratio
    import importlib
    gc_old = float(json.load(open('out/abl/ref.json'))['gc_section_sb1'])
    gam_ref = np.array(json.load(open('out/abl/ref_profile.json'))['gam'])
else:
    gc_old = None
    set_sec(p0); _, gam_ref = X.profiles(z)
    geom_gc_old = geom.gamma_critical(sec0, 1.0); gc_old = geom_gc_old
info = dict(variant=VAR, gc_section_sb1=gc_new, gc_section_sb1_c708=gc_old)
w_tm = lambda s: math.sqrt((s.K_t + s.K_c)/s.M_t)
info['tm_res_rel_c708'] = w_tm(sec1)/w_tm(sec0); info['tm_res_shift_oct'] = math.log2(info['tm_res_rel_c708'])   # TM on K_t + K_c (RL fixed)
info['tm_res_Kt_only_shift_oct'] = 0.5*math.log2((sec1.K_t/sec1.M_t)/(sec0.K_t/sec0.M_t))                     # TM on its own suspension K_t
gs = 1.0
if base == 'pass': p[8] = 0.0
elif base == 'g098': gs = 0.98; p[8] *= gs
elif base == 'g102': gs = 1.02; p[8] *= gs
if mode == 'r': p[8] *= gc_new/gc_old
set_sec(p)
r, gam_new = X.profiles(z)
if base == 'pass': gam = 0.0*gam_ref
elif mode == 'g': gam = gam_ref.copy()
elif mode == 'r': gam = gam_new
else: gam = gs*gam_ref
t0 = time.time()
out = B.evaluate(p)
gate = 'pass'
if out['primary'] <= -9:
    # the stage-1 scorer rejects ratio >= 0.97 before scoring (combo.py then returns obj 9.0). Diagnostic: the same
    # evaluate with only that gate lifted (source of exp_best1.evaluate, threshold 0.97 -> 9.97), so the metrics exist.
    import inspect, textwrap
    src = inspect.getsource(B.evaluate).replace('if ratio >= 0.97:', 'if ratio >= 9.97:')
    assert 'if ratio >= 9.97:' in src
    ns = dict(B.__dict__); exec(compile(src, 'evaluate_nogate', 'exec'), ns)
    gate = 'rejected (stage-1 ratio %.4f >= 0.97); stage-1 metrics computed with the gate lifted' % out['ratio']
    out = ns['evaluate'](p)
gdb, fpk = bm_gain(p)
set_sec(p)
m45 = X.obs45(r, gam)
mm = F1.ftc_metrics(r, gam, math.exp(z[9]), math.exp(z[6]), math.exp(z[7]), z[8], z[10])
Jf = float(X.objective_value(mm))
cert = bool(F1.certify(r, gam))
S2K = ('s2_tt', 's2_sl', 's2_mp', 's2_ph', 's2_fg', 's2_exc')
mg = dict(out['margins']); mg['bmg'] = min(gdb - GWIN[0], GWIN[1] - gdb)/((GWIN[1] - GWIN[0])/2)
for k, m in zip(S2K, X.margins(m45)): mg[k] = float(m)
def clean(v):
    if isinstance(v, float) and not math.isfinite(v): return str(v)
    if isinstance(v, (list, tuple)): return [clean(t) for t in v]
    return v
vals = {k: clean(v) for k, v in out['values'].items()}; vals['bmg'] = gdb; vals['bmg_f_re_cf'] = fpk
for k, v in zip(S2K, m45): vals[k] = float(v)
sc = min(mg.values())
gpk = gam_ref.max()
info.update(stage1_gate=gate, p=p, stage1_ratio=float(out['ratio']), cert=cert, score=float(sc), J_ftc=Jf, obj=-min(sc, 0.10) + 0.002*Jf,
            margins={k: float(v) for k, v in mg.items()}, values=vals, worst=sorted(mg.items(), key=lambda kv: kv[1])[:4],
            stage2_gamma_max_rel_c708=float(gam.max()/gpk), stage2_ratio_new_over_ref=float(np.max(gam_new)/gpk), secs=time.time()-t0)
os.makedirs('out/abl', exist_ok=True)
json.dump(info, open('out/abl/%s.json' % VAR, 'w'), indent=1, default=str)
if VAR == 'ref': json.dump(dict(gam=list(map(float, gam_ref))), open('out/abl/ref_profile.json', 'w'))
print('%-10s gate %s | score %+.4f J_ftc %.3f cert %s ratio %.4f TM shift %+.2f oct worst %s (%.0f s)' % (VAR, gate[:8], sc, Jf, cert, out['ratio'],
      info['tm_res_shift_oct'], [(k, round(v, 3)) for k, v in info['worst']], time.time() - t0), flush=True)
print('   margins', {k: round(v, 3) for k, v in mg.items()}, flush=True)
