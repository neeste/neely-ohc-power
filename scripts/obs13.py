"""Run 6 item 5: observations 1 to 3 (Table I rows 1 to 3) on the scale-invariant and the profiled cat line, with and
without the neural gate (G_s = 1, Eq. 11: exponent m = 0 makes G_s identically 1).
Mode 'si': scale-invariant line, combo.py's own environment (as run3 c6_crit.py: geom.LAM, geom.install, 24-point
gamma_c table; score_base NOT imported), exp_best1.evaluate. Run as plain python3 in the run dir.
Mode 'prof': profiled line, under run_cat.py (score_base imported; it patches E.line, T.response and
AF2.fields to the graded profile, so the same exp_best1.evaluate scores the profiled line, as run 3 found).
Metrics: obs 1 locus L_n at suppressor 1, 1.5, 2, 2.5, 3 octaves below the probe (score_base.battery [a] protocol, N 2048),
median (in dB SPL with the state's C_supp = 65 - median of the profiled-line gated locus, i.e. the paper's constant) and
slope (dB/decade); obs 2 growth slope gr33 (and rms); obs 3 Delta-T at probe 40 and 50 dB (mdt = better margined) and
the vertical protocol (vsl, vdt), from exp_best1.evaluate (N 1024).
Usage: COMBO=combo.json obs13.py si|prof out.json"""
import sys, os, json, math, time
sys.path.insert(1, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); import cat_state
import numpy as np
from dataclasses import replace
sys.path.insert(0, '.')
mode, OUT = sys.argv[1], sys.argv[2]
CK = json.load(open(os.environ.get('COMBO', cat_state.COMBO)))
import geom, deltaT_3dof as d3, tl_model as T
if mode == 'si':
    geom.LAM = float(CK['lam']); geom.install()
    T.section_terms = lambda sec, w, a, r=None, gamma=None: geom.terms(sec, w, a, r, gamma)
    d3.gamma_critical = lambda sec, hi=200.0, tol=1e-6: geom.gamma_critical(sec, 1.0, hi, tol)
import exp_best1 as B, exp_twotone as TT, exp_accumulation as E
B.P_ORD = 4.0; B.WIN['fwd'] = (0.0, 12.5)
if mode == 'si':
    import exp_ftc1 as F1
    SECK = ('K_t', 'M_t', 'R_t', 'R_c', 'R_b', 'K_c', 'K_p', 'M_r')
    def gc_dense(rho):
        rg = np.geomspace(max(rho.min()*0.98, 1e-3), rho.max()*1.02, 24); gg = []
        for r_ in rg:
            S = F1.SEC; s2 = replace(S, gamma=0.0, R_b=S.R_b*r_, R_r=S.R_r*r_, R_t=S.R_t*r_, R_p=S.R_p*r_, R_c=S.R_c*r_)
            gg.append(d3.gamma_critical(s2))
        return np.interp(np.log(rho), np.log(rg), np.asarray(gg))
    F1.gc_dense = gc_dense
    assert 'score_base' not in sys.modules
else:
    assert 'score_base' in sys.modules and abs(geom.LAM - float(CK['lam'])) < 1e-12
bat = json.load(open('base_battery.json')); CSUPP = 65.0 - float(np.median(bat['locus']['L']))
p0 = list(CK['p'])
res = json.load(open(OUT)) if os.path.exists(OUT) else dict(mode=mode, combo=os.environ.get('COMBO', cat_state.COMBO), lam=CK['lam'], C_supp=CSUPP, rows={})
def locus(p, N=2048):
    (kt, mt, rt, rc, rb, kc, kp, mr, g, q, m_g, u, s_g) = p
    coch = replace(T.WORLDS["B"], sec=replace(T.WORLDS["B"].sec, K_t=kt, M_t=mt, R_t=rt, R_c=rc, R_b=rb, K_c=kc, K_p=kp, M_r=mr))
    f_p = float(coch.cf(E.X_PROBE)); r = T.response(coch, f_p, N=2048, gamma=g); E.X_READ = T._peak(r["x"], r["X_b"])[0]
    orig = T.gamma_for_gain; T.gamma_for_gain = lambda *a, **k: (g, float("nan")); TT.P_ORD = 4.0
    row = []
    try:
        for oc in (1.0, 1.5, 2.0, 2.5, 3.0):
            TT.OCT_BELOW = oc
            tt = B.Gated3(coch, 30.0, u, q, m=m_g, s_g=s_g, N=N)
            io = TT.probe_io(tt, -10, 14, 2.0); c_thr = float(np.interp(0.0, io[:, 0], io[:, 1]))
            L_n, _ = TT.bisect_supp(tt, tt.P0 + TT.SL_NEURAL, c_thr); row.append((oc, float(L_n)))
    finally:
        T.gamma_for_gain = orig
    xs = np.asarray([math.log10(2.0**(-o)) for o, _ in row]); ys = np.asarray([v for _, v in row])
    ok = np.isfinite(ys)
    sl = float(np.polyfit(xs[ok], ys[ok], 1)[0]) if ok.sum() >= 2 else float('nan')
    return dict(L=[v for _, v in row], x_read=float(E.X_READ), median_model=float(np.median(ys)) if ok.all() else float('nan'),
                median_spl=float(np.median(ys)) + CSUPP if ok.all() else float('nan'), slope=sl)
def m(v, lo, hi):
    if v is None or not math.isfinite(v): return None
    return min(v - lo, hi - v)/((hi - lo)/2)
for gate in ('gate', 'nogate'):
    if gate in res['rows']: continue
    p = list(p0)
    if gate == 'nogate': p[10] = 0.0
    t0 = time.time()
    out = B.evaluate(p); v = out['values']
    lc = locus(p)
    row = dict(stage1_ratio=float(out['ratio']), x_read=float(E.X_READ), locus=lc,
               values={k: (v.get(k) if not (isinstance(v.get(k), float) and not math.isfinite(v.get(k))) else str(v.get(k)))
                       for k in ('gr33', 'rms', 'mdt', 'dts', 'vsl', 'vdt')},
               margins={k: float(out['margins'][k]) for k in ('gr33', 'mdt', 'vsl', 'vdt')})
    row['margins']['L_n_median'] = m(lc['median_spl'], 59.0, 71.0); row['margins']['L_n_slope'] = m(lc['slope'], -10.0, 10.0)
    row['secs'] = time.time() - t0
    res['rows'][gate] = row; json.dump(res, open(OUT, 'w'), indent=1)
    print('%s %s: L_n median %.2f dB SPL (model %.2f) slope %+.2f dB/dec | gr33 %s rms %s | dts %s mdt %s | vsl %s vdt %s | margins %s (%.0f s)' % (
        mode, gate, lc['median_spl'], lc['median_model'], lc['slope'], row['values']['gr33'], row['values']['rms'], row['values']['dts'],
        row['values']['mdt'], row['values']['vsl'], row['values']['vdt'], {k: (None if x is None else round(x, 3)) for k, x in row['margins'].items()}, row['secs']), flush=True)
print('OBS13 DONE', flush=True)
