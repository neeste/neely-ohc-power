"""Run 6 item 6: the tuning-curve shoulder. (Run 10 copy: the threshold honours FTC_CRIT=disp|vel via exp_ftc1.crit_factor;
with COMBO=run10/out/combo_c856v.json FTC_CRIT=vel it gives the velocity read-out.)
Environment: combo.py's own (as run3 c6_crit.py: geom.LAM, geom.install, 24-point gamma_c(rho) table computed from the
CURRENT section, score_base not imported), so a changed section or lambda gets its own gamma_c and the stage-2 profile
gam = (1 - nu) gc_dense(r/A) TAPER keeps gamma/gamma_c; the stage-1 gamma p[8] is scaled by gc_new/gc_old (gamma/gamma_c
held fixed in every variant). Run as plain python3 from model/ (state COMBO, default cat_state.COMBO).
Threshold tuning curve (Eq. thr): L(f) = C0 - 20 log10|xi_hb(x_j, f)| + ME(f), state's C0 and ME; place x_j = place of the
quiet peak shear for a 1, 2 or 4 kHz tone (in each variant); 300 frequencies from 0.25 to 2.2 f0. CF = tip frequency.
Shoulder index: the largest excess of threshold over 1.05 to 1.6 CF above the straight line (dB against log f) joining the
tip and the threshold at 1.6 CF. Also thresholds at 1.2, 1.4, 1.6 CF re tip. BM plateau: active |X_b(x_j, f)|; notch = the
first local minimum above CF; plateau = the first local maximum above the notch; levels re |X_b(x_j, CF)|.
Worst margin: the 18 margins as c6_crit.py (stage 1 at the scaled p, stage-2 obs 4 to 6 on the profile, BM gain), J_ftc.
Usage: COMBO=combo.json shoulder.py variant out.json   variant: ref | Kc0.5 | Kc2 | Kt4 | Kt0.25 | lam1"""
import sys, os, json, math, time
sys.path.insert(1, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); import cat_state
import numpy as np
from dataclasses import replace
sys.path.insert(0, '.')
variant, OUT = sys.argv[1], sys.argv[2]
CK = json.load(open(os.environ.get('COMBO', cat_state.COMBO)))
import geom, deltaT_3dof as d3, tl_model as T
LAM0 = float(CK['lam']); geom.LAM = LAM0; geom.install()
T.section_terms = lambda sec, w, a, r=None, gamma=None: geom.terms(sec, w, a, r, gamma)
d3.gamma_critical = lambda sec, hi=200.0, tol=1e-6: geom.gamma_critical(sec, 1.0, hi, tol)
import exp_best1 as B
B.P_ORD = 4.0; B.WIN['fwd'] = (0.0, 12.5)
from bmgain_refit import GWIN, bm_gain
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
sec0 = replace(T.WORLDS['B'].sec, **{k: v for k, v in zip(B.NAMES, p0) if k in SECK})
gc_old = geom.gamma_critical(sec0, 1.0)
p = list(p0); lam = LAM0
if variant == 'Kc0.5': p[B.NAMES.index('K_c')] *= 0.5
elif variant == 'Kc2': p[B.NAMES.index('K_c')] *= 2.0
elif variant == 'Kt4': p[B.NAMES.index('K_t')] *= 4.0
elif variant == 'Kt0.25': p[B.NAMES.index('K_t')] *= 0.25
elif variant == 'lam1': lam = 1.0
elif variant != 'ref': raise SystemExit('unknown variant')
geom.LAM = lam
sec = replace(T.WORLDS['B'].sec, **{k: v for k, v in zip(B.NAMES, p) if k in SECK})
gc_new = geom.gamma_critical(sec, 1.0); p[8] = p0[8]*gc_new/gc_old
F1.SEC = X.SEC = sec; F1.COCH = X.COCH = replace(T.WORLDS['B'], sec=sec)
r, gam = X.profiles(z); A, N, COCH = F1.A, F1.N, F1.COCH
base, fhp, flp, nlp, C0 = math.exp(z[9]), math.exp(z[6]), math.exp(z[7]), z[8], z[10]
t0 = time.time()
def resp(f):
    Y, Yr, Ho = E.terms(sec, f, A, r, gam); P = T._fd(COCH, f, Y, N, 1.0); return Y*P, Ho*Y*P
res = dict(combo=os.environ.get('COMBO', cat_state.COMBO), criterion=F1.CRIT, variant=variant, lam=lam, gc_old=gc_old, gc_new=gc_new, gamma_stage1=p[8],
           stage1_ratio=p[8]/gc_new, sec={k: getattr(sec, k) for k in SECK},
           zero_khz_rho1=float(math.sqrt(sec.K_t/sec.M_t)*0 + 0), places={})
# divider zero and pole (Eq. 5) as frequencies re a: sqrt(K_t/M_t), sqrt((K_t + K_c)/M_t)
res['divider_zero_re_a'] = math.sqrt(sec.K_t/sec.M_t); res['divider_pole_re_a'] = math.sqrt((sec.K_t + sec.K_c)/sec.M_t)
for fk in (1.0, 2.0, 4.0):
    f0 = fk/base; xb, sh = resp(f0); j = int(np.argmax(np.abs(sh)))
    fr = f0*np.geomspace(0.25, 2.2, 300); XB, SH = [], []
    for f in fr:
        a_, b_ = resp(f); XB.append(abs(a_[j])); SH.append(abs(b_[j]))
    XB, SH = np.array(XB), np.array(SH)
    th = C0 - 20*np.log10(SH*F1.crit_factor(fr*base)) + F1.me_db(fr*base, fhp, flp, nlp)   # run 10: FTC_CRIT disp (default) or vel
    i0 = int(np.argmin(th)); cf = fr[i0]; lf = np.log(fr)
    def thi(ff): return float(np.interp(math.log(ff), lf, th))
    t16 = thi(1.6*cf); sel = (fr >= 1.05*cf) & (fr <= 1.6*cf)
    chord = th[i0] + (t16 - th[i0])*(lf[sel] - math.log(cf))/math.log(1.6)
    ex = th[sel] - chord; k = int(np.argmax(ex))
    # BM transfer function at x_j
    sel2 = (fr >= cf) & (fr <= 2.2*f0); xs = XB[sel2]; fs = fr[sel2]
    xcf = float(np.interp(math.log(cf), lf, XB))
    notch = None
    dx = np.diff(xs); kmin = [i for i in range(1, len(xs) - 1) if dx[i-1] < 0 and dx[i] >= 0]
    if kmin:
        kn = kmin[0]; fn = fs[kn]; kmax = [i for i in range(kn + 1, len(xs) - 1) if dx[i-1] > 0 and dx[i] <= 0]
        kp = kmax[0] if kmax else None
        notch = dict(f_re_cf=float(fn/cf), depth_db_re_cf=float(20*math.log10(xs[kn]/xcf)),
                     plateau_db_re_cf=float(20*math.log10(xs[kp]/xcf)) if kp is not None else None,
                     plateau_f_re_cf=float(fs[kp]/cf) if kp is not None else None)
    ibm = int(np.argmax(np.where(fr >= 0.5*f0, XB, 0)))
    res['places']['%g kHz' % fk] = dict(x=float(F1.X[j]) if hasattr(F1, 'X') else None, a_khz=float(A[j]*base), cf_khz=float(cf*base),
        tip_db=float(th[i0]), shoulder_index_db=float(ex[k]), shoulder_f_re_cf=float(fr[sel][k]/cf),
        thr_re_tip={'1.2': thi(1.2*cf) - float(th[i0]), '1.4': thi(1.4*cf) - float(th[i0]), '1.6': t16 - float(th[i0])},
        bm_peak_f_re_cf=float(fr[ibm]/cf), bm_notch=notch, cf_re_a=float(cf/A[j]),
        curve=dict(f_re_cf=(fr/cf)[::3].tolist(), thr_re_tip=(th - th[i0])[::3].tolist(), bm_db_re_cf=(20*np.log10(XB/xcf))[::3].tolist()))
    q = res['places']['%g kHz' % fk]
    print('%s %g kHz: CF %.3f kHz (a %.3f), shoulder %.2f dB at %.2f CF; thr re tip 1.2/1.4/1.6 CF %.1f %.1f %.1f; BM notch %s' % (
        variant, fk, q['cf_khz'], q['a_khz'], q['shoulder_index_db'], q['shoulder_f_re_cf'], q['thr_re_tip']['1.2'], q['thr_re_tip']['1.4'],
        q['thr_re_tip']['1.6'], None if notch is None else {k_: round(v, 2) for k_, v in notch.items()}), flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
# worst margin (c6_crit row)
try:
    out = B.evaluate(p)
    if not out['margins']:
        res['worst'] = dict(note='stage-1 gate rejected (ratio %.4f)' % out['ratio'])
    else:
        gdb, fpk = bm_gain(p); m45 = X.obs45(r, gam)
        mm = F1.ftc_metrics(r, gam, base, fhp, flp, nlp, C0); Jf = float(X.objective_value(mm))
        mg = dict(out['margins']); mg['bmg'] = min(gdb - GWIN[0], GWIN[1] - gdb)/((GWIN[1] - GWIN[0])/2)
        for k_, m_ in zip(('s2_tt', 's2_sl', 's2_mp', 's2_ph', 's2_fg', 's2_exc'), X.margins(m45)): mg[k_] = float(m_)
        sc = min(mg.values()); worst = sorted(mg.items(), key=lambda kv: kv[1])[:4]
        res['worst'] = dict(score=float(sc), J_ftc=Jf, worst=[(k_, float(v)) for k_, v in worst], margins={k_: float(v) for k_, v in mg.items()},
                            stage1_ratio=float(out['ratio']))
        print('   worst margin %+.3f J_ftc %.2f %s' % (sc, Jf, [(k_, round(v, 3)) for k_, v in worst]), flush=True)
except Exception as e:
    res['worst'] = dict(error=repr(e))
res['secs'] = time.time() - t0
json.dump(res, open(OUT, 'w'), indent=1)
print('SHOULDER DONE', flush=True)
